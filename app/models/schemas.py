"""
Pydantic models for Mandrake-inspired message patterns.

Based on Mandrake paper concepts:
- TaskMessage: Job assignment to agents
- ResultMessage: Agent completion with score
- ErrorMessage: Agent failure report
- AgentStatus: Tracking agent state for fault tolerance
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent execution status for UI updates and fault tolerance."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    TIMEOUT = "timeout"


class Modality(str, Enum):
    """Media modality - designed for future extension."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class TaskMessage(BaseModel):
    """
    Mandrake-style task message sent to agents.
    
    Key parameters (job_id) enable correlation and idempotence,
    following BSPL (Blindingly Simple Protocol Language) patterns.
    """
    job_id: str = Field(..., description="Unique job identifier (semantic key)")
    image_path: str = Field(..., description="Path to original image")
    face_crop_path: Optional[str] = Field(None, description="Path to cropped face")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    modality: Modality = Field(default=Modality.IMAGE)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResultMessage(BaseModel):
    """
    Mandrake-style result message from agents.
    
    Enables Checkpoint pattern: immediately persist upon receipt.
    """
    job_id: str = Field(..., description="Correlating job identifier")
    agent_name: str = Field(..., description="Name of the reporting agent")
    score: float = Field(..., ge=0.0, le=1.0, description="Fake probability (0=real, 1=fake)")
    explanation: str = Field(..., description="Human-readable explanation")
    latency_ms: float = Field(..., description="Processing time in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional debug info")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorMessage(BaseModel):
    """
    Mandrake-style error message for fault handling.
    
    Enables Remind pattern: track retry_count for timeout recovery.
    """
    job_id: str = Field(..., description="Correlating job identifier")
    agent_name: str = Field(..., description="Name of the failing agent")
    error: str = Field(..., description="Error description")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """Track individual agent state within a job."""
    agent_name: str
    status: AgentStatus = AgentStatus.QUEUED
    result: Optional[ResultMessage] = None
    error: Optional[ErrorMessage] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0


class FaceDetection(BaseModel):
    """Face detection result."""
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence")
    landmarks: Optional[List[List[float]]] = Field(None, description="Facial landmarks")


class PreprocessResult(BaseModel):
    """Result from face preprocessing pipeline."""
    job_id: str
    original_path: str
    face_crop_path: Optional[str] = None
    faces_detected: int = 0
    selected_face: Optional[FaceDetection] = None
    warning: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class AggregatedResult(BaseModel):
    """Final aggregated result from all agents."""
    job_id: str
    verdict: str = Field(..., description="REAL or FAKE")
    confidence: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0)
    quorum_reached: bool = Field(..., description="Whether minimum agents reported")
    agents_completed: int
    agents_total: int
    explanation: str = Field(..., description="Summary explanation")
    agent_results: List[ResultMessage] = Field(default_factory=list)
    agent_errors: List[ErrorMessage] = Field(default_factory=list)
    evidence_verifier_report: Optional[Dict[str, Any]] = Field(
        default=None, description="Evidence verifier findings"
    )
    robustness_verifier_report: Optional[Dict[str, Any]] = Field(
        default=None, description="Robustness verifier findings"
    )
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class JobState(BaseModel):
    """Complete job state for persistence and tracking."""
    job_id: str
    status: str = "pending"  # pending, processing, completed, failed
    original_image_path: str
    face_crop_path: Optional[str] = None
    preprocess_result: Optional[PreprocessResult] = None
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)
    aggregated_result: Optional[AggregatedResult] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WebSocketEvent(BaseModel):
    """Event sent via WebSocket to frontend."""
    job_id: str
    event_type: str  # agent_started, agent_completed, agent_error, job_completed, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    """Request to analyze an image (for API docs)."""
    pass  # File upload handled separately


class AnalyzeResponse(BaseModel):
    """Response from analyze endpoint."""
    job_id: str
    status: str
    message: str
