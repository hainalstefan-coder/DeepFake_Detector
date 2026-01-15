"""Agents package - Detection agents for deepfake analysis."""
from app.agents.cnn_classifier import CNNClassifierAgent
from app.agents.vit_classifier import ViTClassifierAgent
from app.agents.frequency import FrequencyAgent
from app.agents.embedding_anomaly import EmbeddingAnomalyAgent
from app.agents.face_xray import FaceXrayLikeAgent
from app.agents.registry import (
    create_all_agents,
    discover_agents,
    get_agent_class,
    get_all_agent_classes,
    initialize_agents_in_orchestrator,
    register_agent,
)

__all__ = [
    "CNNClassifierAgent",
    "ViTClassifierAgent",
    "FrequencyAgent",
    "EmbeddingAnomalyAgent",
    "FaceXrayLikeAgent",
    "create_all_agents",
    "discover_agents",
    "get_agent_class",
    "get_all_agent_classes",
    "initialize_agents_in_orchestrator",
    "register_agent",
]
