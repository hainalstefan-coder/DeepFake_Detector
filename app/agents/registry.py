"""
Agent Registry - Auto-discovery and registration of agents.

Provides a drop-in system for adding new agents.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

from app.core.agent_base import BaseAgent

logger = logging.getLogger(__name__)


# Registry of available agent classes
_agent_registry: Dict[str, Type[BaseAgent]] = {}


def register_agent(agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
    """
    Decorator to register an agent class.
    
    Usage:
        @register_agent
        class MyAgent(BaseAgent):
            ...
    """
    _agent_registry[agent_class.name] = agent_class
    logger.debug(f"Registered agent: {agent_class.name}")
    return agent_class


def get_agent_class(name: str) -> Type[BaseAgent]:
    """Get an agent class by name."""
    if name not in _agent_registry:
        raise ValueError(f"Unknown agent: {name}")
    return _agent_registry[name]


def get_all_agent_classes() -> Dict[str, Type[BaseAgent]]:
    """Get all registered agent classes."""
    return _agent_registry.copy()


def create_all_agents() -> List[BaseAgent]:
    """Create instances of all registered agents."""
    return [cls() for cls in _agent_registry.values()]


def discover_agents() -> None:
    """
    Auto-discover and register all agents in the agents package.
    
    Imports all modules in app.agents and registers any
    BaseAgent subclasses found.
    """
    # Import all agent modules
    from app.agents import (
        cnn_classifier,
        vit_classifier,
        frequency,
        embedding_anomaly,
        face_xray,
    )
    
    # Register each agent
    from app.agents.cnn_classifier import CNNClassifierAgent
    from app.agents.vit_classifier import ViTClassifierAgent
    from app.agents.frequency import FrequencyAgent
    from app.agents.embedding_anomaly import EmbeddingAnomalyAgent
    from app.agents.face_xray import FaceXrayLikeAgent
    
    _agent_registry[CNNClassifierAgent.name] = CNNClassifierAgent
    _agent_registry[ViTClassifierAgent.name] = ViTClassifierAgent
    _agent_registry[FrequencyAgent.name] = FrequencyAgent
    _agent_registry[EmbeddingAnomalyAgent.name] = EmbeddingAnomalyAgent
    _agent_registry[FaceXrayLikeAgent.name] = FaceXrayLikeAgent
    
    logger.info(f"Discovered {len(_agent_registry)} agents: {list(_agent_registry.keys())}")


def initialize_agents_in_orchestrator(orchestrator: "Orchestrator") -> None:
    """
    Discover agents and register them with the orchestrator.
    
    Args:
        orchestrator: Orchestrator instance to register agents with
    """
    discover_agents()
    
    for agent_class in _agent_registry.values():
        agent = agent_class()
        orchestrator.register_agent(agent)
    
    logger.info(f"Registered {len(_agent_registry)} agents with orchestrator")
