"""
Configuration loader for Deepfake Detector
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AgentWeights(BaseModel):
    CNNClassifierAgent: float = 1.0
    ViTClassifierAgent: float = 1.0
    FrequencyAgent: float = 0.8
    EmbeddingAnomalyAgent: float = 0.9
    FaceXrayLikeAgent: float = 0.85


class AgentConfig(BaseModel):
    timeout_seconds: float = 12.0
    max_retries: int = 1
    quorum: int = 3
    weights: AgentWeights = AgentWeights()


class DetectionConfig(BaseModel):
    fake_threshold: float = 0.6
    high_confidence_threshold: float = 0.8


class PreprocessingConfig(BaseModel):
    detector: str = "mtcnn"
    face_size: int = 224
    margin: int = 20


class StorageConfig(BaseModel):
    type: str = "sqlite"
    path: str = "./data/jobs.db"
    jobs_dir: str = "./data/jobs"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


class DeviceConfig(BaseModel):
    prefer_gpu: bool = True
    fallback_cpu: bool = True


class Config(BaseModel):
    agents: AgentConfig = AgentConfig()
    detection: DetectionConfig = DetectionConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    storage: StorageConfig = StorageConfig()
    server: ServerConfig = ServerConfig()
    device: DeviceConfig = DeviceConfig()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        # Look for config in standard locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            return Config(**data)
    
    # Return default config if no file found
    return Config()


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_device() -> str:
    """Get the device to use for PyTorch models."""
    import torch
    
    config = get_config()
    if config.device.prefer_gpu and torch.cuda.is_available():
        return "cuda"
    elif config.device.fallback_cpu:
        return "cpu"
    else:
        raise RuntimeError("No GPU available and CPU fallback is disabled")
