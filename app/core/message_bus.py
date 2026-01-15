"""
Mandrake-inspired async message bus.

Implements information protocol patterns:
- Topic-based pub/sub for agent communication
- Message correlation via job_id (semantic identifier)
- Idempotent message handling
- Event emission for WebSocket updates
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel

from app.models.schemas import WebSocketEvent

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Async message bus implementing Mandrake patterns.
    
    Key features:
    - Topic-based subscriptions for decoupled agent communication
    - Job-id based correlation (semantic identifiers from BSPL)
    - Idempotent handling via message tracking
    - WebSocket event emission for real-time UI updates
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._processed_messages: Set[str] = set()  # For idempotence
        self._event_queues: Dict[str, asyncio.Queue] = {}  # job_id -> event queue
        self._lock = asyncio.Lock()
    
    async def subscribe(self, topic: str, handler: Callable) -> None:
        """
        Subscribe a handler to a topic.
        
        Args:
            topic: Topic name (e.g., "task", "result", "error")
            handler: Async function to call when message received
        """
        async with self._lock:
            self._subscribers[topic].append(handler)
            logger.debug(f"Subscribed handler to topic: {topic}")
    
    async def unsubscribe(self, topic: str, handler: Callable) -> None:
        """Remove a handler from a topic."""
        async with self._lock:
            if handler in self._subscribers[topic]:
                self._subscribers[topic].remove(handler)
    
    async def publish(self, topic: str, message: BaseModel) -> None:
        """
        Publish a message to a topic.
        
        Implements idempotent handling - duplicate messages are ignored.
        
        Args:
            topic: Topic name
            message: Pydantic model to publish
        """
        # Generate message ID for idempotence
        message_id = self._generate_message_id(topic, message)
        
        async with self._lock:
            if message_id in self._processed_messages:
                logger.debug(f"Ignoring duplicate message: {message_id}")
                return
            self._processed_messages.add(message_id)
        
        # Get handlers outside lock to prevent deadlock
        handlers = self._subscribers.get(topic, []).copy()
        
        # Dispatch to all handlers
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error on topic {topic}: {e}")
    
    def _generate_message_id(self, topic: str, message: BaseModel) -> str:
        """Generate unique ID for idempotence checking."""
        # Use job_id + agent_name + topic for correlation
        job_id = getattr(message, "job_id", "unknown")
        agent_name = getattr(message, "agent_name", "")
        timestamp = getattr(message, "timestamp", datetime.utcnow()).isoformat()
        return f"{topic}:{job_id}:{agent_name}:{timestamp}"
    
    async def register_job_events(self, job_id: str) -> asyncio.Queue:
        """
        Register an event queue for a job (for WebSocket streaming).
        
        Args:
            job_id: Job identifier
            
        Returns:
            Queue that will receive WebSocketEvent objects
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._event_queues[job_id] = queue
        return queue
    
    async def unregister_job_events(self, job_id: str) -> None:
        """Remove event queue for a job."""
        async with self._lock:
            self._event_queues.pop(job_id, None)
    
    async def emit_event(
        self,
        job_id: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit an event for WebSocket streaming.
        
        Args:
            job_id: Job identifier
            event_type: Type of event (agent_started, agent_completed, etc.)
            data: Additional event data
        """
        event = WebSocketEvent(
            job_id=job_id,
            event_type=event_type,
            data=data or {}
        )
        
        queue = self._event_queues.get(job_id)
        if queue:
            await queue.put(event)
            logger.debug(f"Emitted event {event_type} for job {job_id}")
    
    async def clear_processed(self, job_id: str) -> None:
        """Clear processed messages for a completed job to free memory."""
        async with self._lock:
            to_remove = [
                mid for mid in self._processed_messages
                if mid.split(":")[1] == job_id
            ]
            for mid in to_remove:
                self._processed_messages.discard(mid)


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus
