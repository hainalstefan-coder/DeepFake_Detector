"""
Deepfake Detector - Main FastAPI Application

A multi-agent deepfake detection system inspired by Mandrake
fault-tolerant decentralized design.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agents.registry import initialize_agents_in_orchestrator
from app.api.routes import router
from app.api.websocket import websocket_endpoint
from app.config import get_config
from app.core.orchestrator import Orchestrator, set_orchestrator
from app.storage.job_store import close_job_store, get_job_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Deepfake Detector...")
    
    # Initialize job store
    job_store = await get_job_store()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(job_store=job_store)
    set_orchestrator(orchestrator)
    
    # Discover and register agents
    initialize_agents_in_orchestrator(orchestrator)
    
    logger.info("Deepfake Detector ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Deepfake Detector...")
    await close_job_store()


# Create FastAPI app
app = FastAPI(
    title="Deepfake Detector",
    description="Multi-agent deepfake detection system inspired by Mandrake fault-tolerant design",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix="/api")

# WebSocket endpoint
@app.websocket("/ws/{job_id}")
async def websocket_route(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await websocket_endpoint(websocket, job_id)

# Mount static files (frontend)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


# For running directly with python
if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug
    )
