"""
Base Agent class with Mandrake fault-tolerance patterns.

Implements:
- Standardized agent interface
- Timeout handling (Remind pattern preparation)
- Error isolation (agent errors don't crash system)
- Result/Error message generation
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from app.config import get_config
from app.models.schemas import (
    AgentStatus,
    ErrorMessage,
    ResultMessage,
    TaskMessage,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for detection agents.
    
    Mandrake concepts applied:
    - Each agent is an independent task
    - Agents communicate via TaskMessage/ResultMessage/ErrorMessage
    - Errors are isolated per-agent
    - Timeout support for Remind pattern
    """
    
    # Class attributes - override in subclasses
    name: str = "BaseAgent"
    description: str = "Base detection agent"
    
    def __init__(self):
        config = get_config()
        self.timeout = config.agents.timeout_seconds
        self.max_retries = config.agents.max_retries
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize agent resources (models, etc.).
        Called once before first use.
        Override in subclasses for lazy model loading.
        """
        self._initialized = True
    
    @abstractmethod
    async def process(self, task: TaskMessage) -> ResultMessage:
        """
        Process a task and return result.
        
        Must be implemented by each agent.
        Should return ResultMessage with:
        - score: 0.0 (real) to 1.0 (fake)
        - explanation: human-readable reason
        - details: optional debug info
        
        Args:
            task: TaskMessage with image path and metadata
            
        Returns:
            ResultMessage with detection results
        """
        pass
    
    async def run(self, task: TaskMessage) -> ResultMessage | ErrorMessage:
        """
        Run agent with error handling and timing.
        
        Wraps process() with:
        - Lazy initialization
        - Timing measurement
        - Error isolation
        
        Args:
            task: TaskMessage to process
            
        Returns:
            ResultMessage on success, ErrorMessage on failure
        """
        start_time = time.perf_counter()
        
        try:
            # Lazy initialization
            if not self._initialized:
                await self.initialize()
            
            # Run processing with timeout
            result = await asyncio.wait_for(
                self.process(task),
                timeout=self.timeout
            )
            
            # Update latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            result.latency_ms = latency_ms
            
            logger.info(
                f"{self.name} completed job {task.job_id} "
                f"with score {result.score:.3f} in {latency_ms:.1f}ms"
            )
            
            return result
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"{self.name} timed out on job {task.job_id} "
                f"after {latency_ms:.1f}ms"
            )
            return ErrorMessage(
                job_id=task.job_id,
                agent_name=self.name,
                error=f"Timeout after {self.timeout}s"
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{self.name} error on job {task.job_id}: {e}",
                exc_info=True
            )
            return ErrorMessage(
                job_id=task.job_id,
                agent_name=self.name,
                error=str(e)
            )
    
    def create_result(
        self,
        task: TaskMessage,
        score: float,
        explanation: str,
        details: Optional[dict] = None
    ) -> ResultMessage:
        """Helper to create a ResultMessage."""
        return ResultMessage(
            job_id=task.job_id,
            agent_name=self.name,
            score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
            explanation=explanation,
            latency_ms=0.0,  # Will be updated by run()
            details=details or {}
        )
    
    def create_error(
        self,
        task: TaskMessage,
        error: str,
        retry_count: int = 0
    ) -> ErrorMessage:
        """Helper to create an ErrorMessage."""
        return ErrorMessage(
            job_id=task.job_id,
            agent_name=self.name,
            error=error,
            retry_count=retry_count
        )
