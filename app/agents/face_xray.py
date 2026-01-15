"""
Face X-Ray Like Agent.

Detects deepfakes by analyzing blending boundaries in face-swapped images.
Inspired by the Face X-Ray paper methodology.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms

from app.config import get_device
from app.core.agent_base import BaseAgent
from app.models.schemas import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)


class BoundaryDetector(nn.Module):
    """CNN for detecting blending boundaries."""
    
    def __init__(self):
        super().__init__()
        # Encoder for boundary detection
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
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
        features = self.encoder(x)
        return self.classifier(features)


class FaceXrayLikeAgent(BaseAgent):
    """
    Face X-Ray inspired blending detection agent.
    
    Detects face-swap deepfakes by analyzing:
    - Blending boundaries around facial regions
    - Edge inconsistencies at face boundaries
    - Color/lighting discontinuities
    """
    
    name = "FaceXrayLikeAgent"
    description = "Blending boundary detection for face-swap deepfakes"
    
    def __init__(self):
        super().__init__()
        self.model: Optional[BoundaryDetector] = None
        self.device = get_device()
        self.transform: Optional[transforms.Compose] = None
    
    async def initialize(self) -> None:
        """Initialize boundary detection model."""
        try:
            self.model = BoundaryDetector()
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self._initialized = True
            logger.info(f"FaceXrayLikeAgent initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FaceXrayLikeAgent: {e}")
            raise
    
    def extract_boundary_features(
        self,
        image: Image.Image
    ) -> Tuple[np.ndarray, dict]:
        """
        Extract features related to blending boundaries.
        
        Returns edge map and analysis metrics.
        """
        # Convert to numpy
        img_array = np.array(image)
        
        # Apply edge detection (Sobel-like via PIL)
        gray = image.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)
        
        # Compute edge statistics
        edge_mean = float(np.mean(edge_array))
        edge_std = float(np.std(edge_array))
        edge_max = float(np.max(edge_array))
        
        # Analyze edge distribution in different regions
        h, w = edge_array.shape
        
        # Center region (face area)
        center_y, center_x = h // 2, w // 2
        margin = min(h, w) // 4
        center_region = edge_array[
            center_y - margin:center_y + margin,
            center_x - margin:center_x + margin
        ]
        center_edge_mean = float(np.mean(center_region)) if center_region.size > 0 else 0
        
        # Border region (potential blending boundary)
        border_mask = np.ones_like(edge_array, dtype=bool)
        border_mask[margin:-margin, margin:-margin] = False
        border_region = edge_array[border_mask]
        border_edge_mean = float(np.mean(border_region)) if border_region.size > 0 else 0
        
        # Blending indicator: high edge activity at border relative to center
        edge_ratio = border_edge_mean / (center_edge_mean + 1e-8)
        
        # Color channel analysis for blending artifacts
        if len(img_array.shape) == 3:
            channel_stds = [float(np.std(img_array[:, :, c])) for c in range(3)]
            channel_imbalance = max(channel_stds) - min(channel_stds)
        else:
            channel_stds = [float(np.std(img_array))]
            channel_imbalance = 0.0
        
        details = {
            "edge_mean": edge_mean,
            "edge_std": edge_std,
            "edge_max": edge_max,
            "center_edge_mean": center_edge_mean,
            "border_edge_mean": border_edge_mean,
            "edge_ratio": edge_ratio,
            "channel_stds": channel_stds,
            "channel_imbalance": channel_imbalance,
        }
        
        return edge_array, details
    
    async def process(self, task: TaskMessage) -> ResultMessage:
        """Analyze image for blending boundaries."""
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
        
        # Extract boundary features
        edge_map, boundary_details = self.extract_boundary_features(image)
        
        # CNN-based classification
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logit = self.model(input_tensor)
            cnn_score = torch.sigmoid(logit).item()
        
        # Combine CNN and heuristic scores
        edge_ratio = boundary_details["edge_ratio"]
        channel_imbalance = boundary_details["channel_imbalance"]
        
        # Heuristic scoring
        # High edge ratio at border suggests blending
        edge_score = min(1.0, edge_ratio / 2.0) if edge_ratio > 1.0 else edge_ratio * 0.3
        
        # Channel imbalance can indicate color correction artifacts
        color_score = min(1.0, channel_imbalance / 50.0)
        
        # Weighted combination
        final_score = 0.5 * cnn_score + 0.3 * edge_score + 0.2 * color_score
        final_score = max(0.0, min(1.0, final_score))
        
        # Generate explanation
        if final_score >= 0.7:
            if edge_ratio > 1.5:
                explanation = "Strong blending boundary detected around face region"
            else:
                explanation = "Facial boundary inconsistencies indicate manipulation"
        elif final_score >= 0.5:
            explanation = "Moderate boundary artifacts suggest possible face swap"
        elif final_score >= 0.3:
            explanation = "Minor boundary variations, likely natural"
        else:
            explanation = "No blending boundaries detected, appears authentic"
        
        return self.create_result(
            task,
            score=final_score,
            explanation=explanation,
            details={
                **boundary_details,
                "cnn_score": cnn_score,
                "edge_score": edge_score,
                "color_score": color_score,
                "model": "BoundaryDetector-128"
            }
        )
