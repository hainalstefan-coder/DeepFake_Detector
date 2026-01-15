"""
WebSocket handler for real-time updates.

Streams events from message bus to connected clients.
"""

import asyncio
import json
import logging
from typing import Dict

from fastapi import WebSocket, WebSocketDisconnect

from app.core.message_bus import get_message_bus

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections per job."""
    
    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """Accept WebSocket connection for a job."""
        await websocket.accept()
        self._connections[job_id] = websocket
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, job_id: str) -> None:
        """Remove WebSocket connection."""
        self._connections.pop(job_id, None)
        logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_event(self, job_id: str, event: dict) -> bool:
        """Send event to job's WebSocket."""
        websocket = self._connections.get(job_id)
        if websocket:
            try:
                await websocket.send_json(event)
                return True
            except Exception as e:
                logger.error(f"WebSocket send error for {job_id}: {e}")
                return False
        return False


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint handler.
    
    Streams events from message bus to connected client.
    """
    await manager.connect(websocket, job_id)
    
    # Get event queue from message bus
    message_bus = get_message_bus()
    event_queue = await message_bus.register_job_events(job_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event_type": "connected",
            "job_id": job_id,
            "message": "WebSocket connected, waiting for events..."
        })
        
        # Stream events from queue
        while True:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=30.0
                )
                
                # Convert event to dict
                event_data = {
                    "job_id": event.job_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data
                }
                
                await websocket.send_json(event_data)
                
                # Check if job completed
                if event.event_type in ("job_completed", "job_failed"):
                    # Send final event and close
                    await asyncio.sleep(0.5)  # Allow client to process
                    break
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({
                    "event_type": "ping",
                    "job_id": job_id
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        manager.disconnect(job_id)
        await message_bus.unregister_job_events(job_id)
