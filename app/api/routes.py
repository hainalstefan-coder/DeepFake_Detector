"""
REST API routes.

Endpoints:
- POST /analyze: Upload image and start analysis
- GET /result/{job_id}: Get job result
- GET /jobs: List recent jobs
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_config
from app.core.orchestrator import get_orchestrator
from app.models.schemas import (
    AggregatedResult,
    AnalyzeResponse,
    JobState,
)
from app.preprocessing.face_pipeline import get_preprocessor
from app.storage.job_store import get_job_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Upload an image for deepfake analysis.
    
    Returns immediately with a job_id. Connect to WebSocket
    /ws/{job_id} to receive real-time updates.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Get job directory
    config = get_config()
    job_dir = Path(config.storage.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    file_ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    input_path = job_dir / f"input{file_ext}"
    
    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {e}"
        )
    finally:
        await file.close()
    
    # Preprocess face
    preprocessor = get_preprocessor()
    preprocess_result = preprocessor.process(
        str(input_path),
        job_id,
        config.storage.jobs_dir
    )
    
    if not preprocess_result.success:
        return AnalyzeResponse(
            job_id=job_id,
            status="failed",
            message=preprocess_result.error or "Preprocessing failed"
        )
    
    # Start job with orchestrator
    orchestrator = get_orchestrator()
    
    # Run in background
    import asyncio
    asyncio.create_task(
        orchestrator.start_job(
            job_id=job_id,
            image_path=str(input_path),
            face_crop_path=preprocess_result.face_crop_path
        )
    )
    
    # Build response message
    message = f"Analysis started with {len(orchestrator.get_agent_names())} agents"
    if preprocess_result.warning:
        message += f". Warning: {preprocess_result.warning}"
    
    return AnalyzeResponse(
        job_id=job_id,
        status="processing",
        message=message
    )


@router.get("/result/{job_id}")
async def get_result(job_id: str) -> dict:
    """
    Get the result of a completed analysis job.
    
    Returns full job state including all agent results.
    """
    # Try orchestrator first (in-memory)
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)
    
    if not job:
        # Try persistent store
        job_store = await get_job_store()
        job = await job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    # Convert to dict for response
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "agent_states": {
            name: {
                "status": state.status.value,
                "result": state.result.model_dump() if state.result else None,
                "error": state.error.model_dump() if state.error else None,
            }
            for name, state in job.agent_states.items()
        },
        "aggregated_result": job.aggregated_result.model_dump() if job.aggregated_result else None
    }


@router.get("/jobs")
async def list_jobs(limit: int = 10) -> dict:
    """List recent analysis jobs."""
    job_store = await get_job_store()
    jobs = await job_store.get_recent_jobs(limit)
    
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "verdict": job.aggregated_result.verdict if job.aggregated_result else None,
                "confidence": job.aggregated_result.confidence if job.aggregated_result else None,
            }
            for job in jobs
        ]
    }


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    import torch
    
    orchestrator = get_orchestrator()
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "agents": orchestrator.get_agent_names()
    }
