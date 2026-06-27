"""
Orchestrator - Mandrake-inspired aggregator agent.

Implements fault tolerance patterns:
- Remind: Track pending agents, trigger retry after timeout
- Checkpoint: Persist agent outputs immediately on receipt
- Continue: Finalize when quorum reached (even if some agents fail)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from app.config import get_config
from app.core.message_bus import MessageBus, get_message_bus
from app.models.schemas import (
    AgentState,
    AgentStatus,
    AggregatedResult,
    ErrorMessage,
    JobState,
    ResultMessage,
    TaskMessage,
    WebSocketEvent,
)
from app.verifiers.evidence import EvidenceVerifier
from app.verifiers.robustness import RobustnessVerifier

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestrator implementing Mandrake fault-tolerance patterns.
    
    Responsibilities:
    - Dispatch tasks to all agents
    - Track agent states per job
    - Implement timeout + retry (Remind)
    - Persist results immediately (Checkpoint)
    - Finalize on quorum (Continue)
    - Aggregate results into final verdict
    """
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        job_store: Optional["JobStore"] = None
    ):
        self.config = get_config()
        self.message_bus = message_bus or get_message_bus()
        self.job_store = job_store
        
        # Active jobs tracking
        self._jobs: Dict[str, JobState] = {}
        self._job_locks: Dict[str, asyncio.Lock] = {}
        self._pending_agents: Dict[str, Set[str]] = {}  # job_id -> agent names
        self._timeout_tasks: Dict[str, asyncio.Task] = {}
        
        # Agent registry
        self._agents: Dict[str, "BaseAgent"] = {}
    
    def register_agent(self, agent: "BaseAgent") -> None:
        """Register an agent for task dispatch."""
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent_names(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self._agents.keys())
    
    async def start_job(
        self,
        job_id: str,
        image_path: str,
        face_crop_path: Optional[str] = None
    ) -> JobState:
        """
        Start a new detection job.
        
        Creates job state and dispatches tasks to all agents.
        
        Args:
            job_id: Unique job identifier
            image_path: Path to original image
            face_crop_path: Path to cropped face (optional)
            
        Returns:
            Initial JobState
        """
        # Create job state
        job = JobState(
            job_id=job_id,
            status="processing",
            original_image_path=image_path,
            face_crop_path=face_crop_path,
            agent_states={
                name: AgentState(agent_name=name, status=AgentStatus.QUEUED)
                for name in self._agents.keys()
            }
        )
        
        self._jobs[job_id] = job
        self._job_locks[job_id] = asyncio.Lock()
        self._pending_agents[job_id] = set(self._agents.keys())
        
        # Emit job started event
        await self.message_bus.emit_event(
            job_id,
            "job_started",
            {"agents": list(self._agents.keys())}
        )
        
        # Create task message
        task = TaskMessage(
            job_id=job_id,
            image_path=image_path,
            face_crop_path=face_crop_path
        )
        
        # Dispatch to all agents concurrently (fire and forget using create_task)
        for agent_name, agent in self._agents.items():
            asyncio.create_task(self._dispatch_to_agent(agent, task))
        
        # Start timeout monitor for this job
        self._timeout_tasks[job_id] = asyncio.create_task(
            self._monitor_timeouts(job_id)
        )
        
        logger.info(f"Started job {job_id} with {len(self._agents)} agents")
        return job
    
    async def _dispatch_to_agent(
        self,
        agent: "BaseAgent",
        task: TaskMessage
    ) -> None:
        """Dispatch task to a single agent and handle result."""
        job_id = task.job_id
        agent_name = agent.name
        
        try:
            # Update status to running
            await self._update_agent_status(
                job_id, agent_name, AgentStatus.RUNNING
            )
            
            # Emit agent started event
            await self.message_bus.emit_event(
                job_id,
                "agent_started",
                {"agent_name": agent_name}
            )
            
            # Run agent
            result = await agent.run(task)
            
            # Handle result
            if isinstance(result, ResultMessage):
                await self.on_agent_result(result)
            else:
                await self.on_agent_error(result)
        except Exception as e:
            logger.error(f"Error dispatching to {agent_name}: {e}", exc_info=True)
            error = ErrorMessage(
                job_id=job_id,
                agent_name=agent_name,
                error=str(e)
            )
            await self.on_agent_error(error)
    
    async def on_agent_result(self, result: ResultMessage) -> None:
        """
        Handle successful agent result.
        
        Implements Checkpoint pattern: immediately persist result.
        """
        job_id = result.job_id
        agent_name = result.agent_name
        
        async with self._job_locks.get(job_id, asyncio.Lock()):
            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Result for unknown job: {job_id}")
                return
            
            # Update agent state (Checkpoint: persist immediately)
            agent_state = job.agent_states.get(agent_name)
            if agent_state:
                agent_state.status = AgentStatus.DONE
                agent_state.result = result
                agent_state.completed_at = datetime.utcnow()
            
            # Remove from pending
            self._pending_agents.get(job_id, set()).discard(agent_name)
            
            # Persist to store if available
            if self.job_store:
                await self.job_store.update_job(job)
            
            job.updated_at = datetime.utcnow()
        
        # Emit event with details including arcface_similarity
        event_data = {
            "agent_name": agent_name,
            "score": result.score,
            "explanation": result.explanation,
            "latency_ms": result.latency_ms,
        }
        # Include key details like ArcFace similarity
        if result.details:
            if "arcface_similarity" in result.details:
                event_data["arcface_similarity"] = result.details["arcface_similarity"]
            if "cnn_score" in result.details:
                event_data["cnn_score"] = result.details["cnn_score"]
        
        await self.message_bus.emit_event(
            job_id,
            "agent_completed",
            event_data
        )
        
        logger.info(
            f"Agent {agent_name} completed job {job_id} "
            f"with score {result.score:.3f}"
        )
        
        # Check if we can finalize
        await self._check_completion(job_id)
    
    async def on_agent_error(self, error: ErrorMessage) -> None:
        """
        Handle agent error.
        
        Implements Remind pattern: retry if under max_retries.
        """
        job_id = error.job_id
        agent_name = error.agent_name
        
        async with self._job_locks.get(job_id, asyncio.Lock()):
            job = self._jobs.get(job_id)
            if not job:
                return
            
            agent_state = job.agent_states.get(agent_name)
            if not agent_state:
                return
            
            # Check if we should retry (Remind pattern)
            if error.retry_count < self.config.agents.max_retries:
                logger.info(
                    f"Retrying {agent_name} for job {job_id} "
                    f"(attempt {error.retry_count + 1})"
                )
                
                agent_state.retry_count = error.retry_count + 1
                agent_state.status = AgentStatus.QUEUED
                
                # Emit retry event
                await self.message_bus.emit_event(
                    job_id,
                    "agent_retry",
                    {
                        "agent_name": agent_name,
                        "retry_count": agent_state.retry_count,
                        "error": error.error
                    }
                )
                
                # Retry dispatch
                agent = self._agents.get(agent_name)
                if agent:
                    task = TaskMessage(
                        job_id=job_id,
                        image_path=job.original_image_path,
                        face_crop_path=job.face_crop_path
                    )
                    asyncio.create_task(self._dispatch_to_agent(agent, task))
            else:
                # Max retries reached
                agent_state.status = AgentStatus.ERROR
                agent_state.error = error
                agent_state.completed_at = datetime.utcnow()
                self._pending_agents.get(job_id, set()).discard(agent_name)
                
                # Emit error event
                await self.message_bus.emit_event(
                    job_id,
                    "agent_error",
                    {
                        "agent_name": agent_name,
                        "error": error.error,
                        "retry_count": error.retry_count
                    }
                )
                
                logger.warning(
                    f"Agent {agent_name} failed for job {job_id}: {error.error}"
                )
            
            job.updated_at = datetime.utcnow()
            if self.job_store:
                await self.job_store.update_job(job)
        
        # Check completion
        await self._check_completion(job_id)
    
    async def _update_agent_status(
        self,
        job_id: str,
        agent_name: str,
        status: AgentStatus
    ) -> None:
        """Update agent status in job state."""
        async with self._job_locks.get(job_id, asyncio.Lock()):
            job = self._jobs.get(job_id)
            if job and agent_name in job.agent_states:
                job.agent_states[agent_name].status = status
                if status == AgentStatus.RUNNING:
                    job.agent_states[agent_name].started_at = datetime.utcnow()
    
    async def _monitor_timeouts(self, job_id: str) -> None:
        """
        Background task to monitor agent timeouts.
        
        Part of Remind pattern implementation.
        """
        timeout = self.config.agents.timeout_seconds + 1  # Buffer
        
        try:
            while job_id in self._pending_agents:
                await asyncio.sleep(1)
                
                job = self._jobs.get(job_id)
                if not job:
                    break
                
                now = datetime.utcnow()
                
                for agent_name in list(self._pending_agents.get(job_id, [])):
                    agent_state = job.agent_states.get(agent_name)
                    if not agent_state or agent_state.status != AgentStatus.RUNNING:
                        continue
                    
                    started_at = agent_state.started_at
                    if started_at and (now - started_at).total_seconds() > timeout:
                        # Create timeout error
                        error = ErrorMessage(
                            job_id=job_id,
                            agent_name=agent_name,
                            error=f"Timeout after {timeout}s",
                            retry_count=agent_state.retry_count
                        )
                        await self.on_agent_error(error)
                        
        except asyncio.CancelledError:
            pass
    
    async def _check_completion(self, job_id: str) -> None:
        """
        Check if job can be finalized.
        
        Implements Continue pattern: finalize when quorum reached.
        """
        job = self._jobs.get(job_id)
        if not job or job.status == "completed":
            return
        
        # Count completed agents
        completed = sum(
            1 for s in job.agent_states.values()
            if s.status == AgentStatus.DONE
        )
        total = len(job.agent_states)
        pending = len(self._pending_agents.get(job_id, set()))
        
        # Check quorum (Continue pattern)
        quorum = self.config.agents.quorum
        
        if completed >= quorum or pending == 0:
            # Can finalize
            result = await self.aggregate_results(job_id)
            
            job.aggregated_result = result
            job.status = "completed"
            job.updated_at = datetime.utcnow()
            
            # Cancel timeout monitor
            timeout_task = self._timeout_tasks.pop(job_id, None)
            if timeout_task:
                timeout_task.cancel()
            
            # Persist final state
            if self.job_store:
                await self.job_store.update_job(job)
            
            # Emit completion event with full agent stats
            await self.message_bus.emit_event(
                job_id,
                "job_completed",
                {
                    "verdict": result.verdict,
                    "confidence": result.confidence,
                    "final_score": result.final_score,
                    "quorum_reached": result.quorum_reached,
                    "agents_completed": result.agents_completed,
                    "agents_total": result.agents_total,
                    "explanation": result.explanation
                }
            )
            
            logger.info(
                f"Job {job_id} completed: {result.verdict} "
                f"({result.confidence:.1%} confidence)"
            )
    
    async def aggregate_results(self, job_id: str) -> AggregatedResult:
        """
        Aggregate agent results into final verdict.
        
        Uses weighted average and uncertainty calculation.
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Unknown job: {job_id}")
        
        # Collect successful results
        results: List[ResultMessage] = []
        errors: List[ErrorMessage] = []
        
        for state in job.agent_states.values():
            if state.result:
                results.append(state.result)
            if state.error:
                errors.append(state.error)
        
        if not results:
            # No results - return uncertain
            return AggregatedResult(
                job_id=job_id,
                verdict="UNKNOWN",
                confidence=0.0,
                final_score=0.5,
                uncertainty=1.0,
                quorum_reached=False,
                agents_completed=0,
                agents_total=len(job.agent_states),
                explanation="No agents completed successfully",
                agent_results=[],
                agent_errors=errors,
                evidence_verifier_report=None,
                robustness_verifier_report=None,
            )

        # Apply verifiers if enabled
        evidence_report: Optional[Dict[str, Any]] = None
        robustness_report: Optional[Dict[str, Any]] = None

        if self.config.verifier.evidence_enabled:
            verifier = EvidenceVerifier(
                outlier_threshold=self.config.verifier.outlier_threshold,
                plausibility_margin=self.config.verifier.plausibility_margin,
                down_weight_factor=self.config.verifier.down_weight_factor,
            )
            results, evidence_report = verifier.verify(results)

        if self.config.verifier.robustness_enabled:
            verifier = RobustnessVerifier(
                overconfidence_margin=self.config.verifier.overconfidence_margin,
                instability_factor=self.config.verifier.instability_factor,
                down_weight_factor=self.config.verifier.down_weight_factor,
            )
            results, robustness_report = verifier.verify(results)

        # Get weights
        weights_config = self.config.agents.weights
        weights_dict = {
            "CNNClassifierAgent": weights_config.CNNClassifierAgent,
            "ViTClassifierAgent": weights_config.ViTClassifierAgent,
            "FrequencyAgent": weights_config.FrequencyAgent,
            "EmbeddingAnomalyAgent": weights_config.EmbeddingAnomalyAgent,
            "FaceXrayLikeAgent": weights_config.FaceXrayLikeAgent,
        }
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        scores = []
        
        for result in results:
            weight = weights_dict.get(result.agent_name, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight
            scores.append(result.score)
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Calculate uncertainty (variance)
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            uncertainty = variance ** 0.5
        else:
            uncertainty = 0.5  # High uncertainty with single agent
        
        # Determine verdict
        threshold = self.config.detection.fake_threshold
        verdict = "FAKE" if final_score >= threshold else "REAL"
        
        # Calculate confidence (inverse of uncertainty, clamped)
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        
        # Adjust confidence based on score distance from threshold
        distance_to_threshold = abs(final_score - threshold)
        confidence = confidence * (0.5 + distance_to_threshold)
        confidence = max(0.0, min(1.0, confidence))
        
        # Generate explanation
        explanation = self._generate_explanation(results, verdict, final_score)
        
        quorum = self.config.agents.quorum
        
        return AggregatedResult(
            job_id=job_id,
            verdict=verdict,
            confidence=confidence,
            final_score=final_score,
            uncertainty=uncertainty,
            quorum_reached=len(results) >= quorum,
            agents_completed=len(results),
            agents_total=len(job.agent_states),
            explanation=explanation,
            agent_results=results,
            agent_errors=errors,
            evidence_verifier_report=evidence_report,
            robustness_verifier_report=robustness_report,
        )
    
    def _generate_explanation(
        self,
        results: List[ResultMessage],
        verdict: str,
        final_score: float
    ) -> str:
        """Generate human-readable explanation summary."""
        if not results:
            return "No analysis results available."
        
        # Sort by score (highest first for fake indicators, lowest for real)
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Get top 2 supporting verdict and top 1 opposing
        threshold = self.config.detection.fake_threshold
        
        if verdict == "FAKE":
            supporting = [r for r in sorted_results if r.score >= threshold][:2]
            opposing = [r for r in sorted_results if r.score < threshold][:1]
        else:
            supporting = [r for r in sorted_results if r.score < threshold][:2]
            opposing = [r for r in sorted_results if r.score >= threshold][:1]
        
        # Build explanation
        parts = [
            f"Analysis indicates {verdict} ({final_score:.0%} fake probability)."
        ]
        
        if supporting:
            indicators = [
                f"{r.agent_name} ({r.score:.0%}): {r.explanation}"
                for r in supporting
            ]
            parts.append("Supporting evidence: " + "; ".join(indicators))
        
        if opposing:
            r = opposing[0]
            parts.append(
                f"Counter-evidence: {r.agent_name} ({r.score:.0%}): {r.explanation}"
            )
        
        return " ".join(parts)
    
    def get_job(self, job_id: str) -> Optional[JobState]:
        """Get job state by ID."""
        return self._jobs.get(job_id)
    
    async def cleanup_job(self, job_id: str) -> None:
        """Clean up job resources."""
        self._jobs.pop(job_id, None)
        self._job_locks.pop(job_id, None)
        self._pending_agents.pop(job_id, None)
        
        timeout_task = self._timeout_tasks.pop(job_id, None)
        if timeout_task:
            timeout_task.cancel()
        
        await self.message_bus.clear_processed(job_id)
        await self.message_bus.unregister_job_events(job_id)


# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: Orchestrator) -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator
