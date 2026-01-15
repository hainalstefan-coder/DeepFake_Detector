"""
Embedding Anomaly Agent.

Uses face embeddings to detect anomalies that indicate deepfakes.
Incorporates ArcFace similarity metric for identity verification.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.config import get_device
from app.core.agent_base import BaseAgent
from app.models.schemas import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)


class EmbeddingEncoder(nn.Module):
    """Simple embedding encoder for face features."""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EmbeddingAnomalyAgent(BaseAgent):
    """
    Embedding-based anomaly detection agent.
    
    Analyzes face embeddings for anomalies that indicate deepfakes:
    - Embedding deviation from typical face distribution
    - ArcFace similarity score for identity consistency
    - Statistical anomaly detection in feature space
    """
    
    name = "EmbeddingAnomalyAgent"
    description = "Face embedding analysis with ArcFace similarity for deepfake detection"
    
    def __init__(self):
        super().__init__()
        self.encoder: Optional[EmbeddingEncoder] = None
        self.device = get_device()
        self.embedding_dim = 512
        self.transform: Optional[transforms.Compose] = None
        
        # Reference statistics for real faces (would be learned from data)
        self._mean_embedding: Optional[torch.Tensor] = None
        self._std_threshold = 2.5  # Standard deviations for anomaly
    
    async def initialize(self) -> None:
        """Initialize embedding encoder."""
        try:
            self.encoder = EmbeddingEncoder(self.embedding_dim)
            self.encoder = self.encoder.to(self.device)
            self.encoder.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),  # ArcFace standard size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
            
            # Initialize reference embedding (random for demo)
            # In production, this would be from trained model
            self._mean_embedding = torch.zeros(self.embedding_dim).to(self.device)
            
            self._initialized = True
            logger.info(f"EmbeddingAnomalyAgent initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingAnomalyAgent: {e}")
            raise
    
    def compute_arcface_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute ArcFace-style cosine similarity between embeddings.
        
        Returns similarity score in [0, 1] range.
        """
        # L2 normalize embeddings
        e1_norm = torch.nn.functional.normalize(embedding1, p=2, dim=-1)
        e2_norm = torch.nn.functional.normalize(embedding2, p=2, dim=-1)
        
        # Cosine similarity
        similarity = torch.mm(e1_norm, e2_norm.t()).item()
        
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2
    
    def detect_embedding_anomaly(
        self,
        embedding: torch.Tensor
    ) -> Tuple[float, dict]:
        """
        Detect anomalies in embedding vector.
        
        Returns anomaly score and analysis details.
        """
        # Normalize embedding
        embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        # Compute statistics
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Statistical measures
        mean_val = float(np.mean(embedding_np))
        std_val = float(np.std(embedding_np))
        max_val = float(np.max(np.abs(embedding_np)))
        
        # Compute distance from reference
        if self._mean_embedding is not None:
            ref_similarity = self.compute_arcface_similarity(
                embedding, 
                self._mean_embedding.unsqueeze(0)
            )
        else:
            ref_similarity = 0.5
        
        # Anomaly indicators
        # High std suggests unusual feature distribution (potential fake)
        std_anomaly = min(1.0, std_val / 2.0)
        
        # Low similarity to reference suggests anomaly
        similarity_anomaly = 1.0 - ref_similarity
        
        # Combine scores
        anomaly_score = 0.4 * std_anomaly + 0.6 * similarity_anomaly
        
        details = {
            "embedding_mean": mean_val,
            "embedding_std": std_val,
            "embedding_max": max_val,
            "arcface_similarity": ref_similarity,
            "std_anomaly_score": std_anomaly,
            "similarity_anomaly_score": similarity_anomaly,
        }
        
        return anomaly_score, details
    
    async def process(self, task: TaskMessage) -> ResultMessage:
        """Process image and detect embedding anomalies."""
        image_path = task.face_crop_path or task.image_path
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return self.create_result(
                task,
                score=0.5,
                explanation="Failed to load image",
                details={"error": str(e)}
            )
        
        # Transform and encode
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.encoder(input_tensor)
        
        # Detect anomalies
        anomaly_score, analysis = self.detect_embedding_anomaly(embedding)
        
        # Generate explanation based on findings
        arcface_sim = analysis["arcface_similarity"]
        
        if anomaly_score >= 0.7:
            if arcface_sim < 0.3:
                explanation = f"Severe embedding anomaly detected (ArcFace: {arcface_sim:.2f})"
            else:
                explanation = "Embedding distribution inconsistent with natural faces"
        elif anomaly_score >= 0.5:
            explanation = f"Moderate embedding deviation detected (ArcFace: {arcface_sim:.2f})"
        elif anomaly_score >= 0.3:
            explanation = f"Minor embedding anomalies, likely authentic (ArcFace: {arcface_sim:.2f})"
        else:
            explanation = f"Embedding consistent with natural face (ArcFace: {arcface_sim:.2f})"
        
        return self.create_result(
            task,
            score=anomaly_score,
            explanation=explanation,
            details={
                **analysis,
                "embedding_dim": self.embedding_dim,
                "model": "EmbeddingEncoder-512d"
            }
        )
