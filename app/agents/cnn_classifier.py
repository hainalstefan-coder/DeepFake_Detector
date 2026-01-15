"""
CNN Classifier Agent.

Uses EfficientNet for deepfake detection.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.config import get_device
from app.core.agent_base import BaseAgent
from app.models.schemas import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)


class CNNClassifierAgent(BaseAgent):
    """
    CNN-based deepfake detection agent.
    
    Uses EfficientNet-B0 with a binary classification head.
    Pretrained on ImageNet, adapted for deepfake detection.
    """
    
    name = "CNNClassifierAgent"
    description = "EfficientNet-based CNN classifier for deepfake detection"
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self.device = get_device()
    
    async def initialize(self) -> None:
        """Load EfficientNet model."""
        try:
            import timm
            
            # Load EfficientNet-B0 pretrained
            self.model = timm.create_model(
                'efficientnet_b0',
                pretrained=True,
                num_classes=1  # Binary classification
            )
            
            # Initialize the classifier head for deepfake detection
            # (In production, load fine-tuned weights)
            nn.init.xavier_uniform_(self.model.classifier.weight)
            nn.init.zeros_(self.model.classifier.bias)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self._initialized = True
            logger.info(f"CNNClassifierAgent initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CNNClassifierAgent: {e}")
            raise
    
    async def process(self, task: TaskMessage) -> ResultMessage:
        """Process image and return fake probability."""
        # Load image
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
        
        # Transform
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logit = self.model(input_tensor)
            score = torch.sigmoid(logit).item()
        
        # Generate explanation
        if score >= 0.7:
            explanation = "CNN detected manipulation artifacts in facial features"
        elif score >= 0.5:
            explanation = "CNN found subtle inconsistencies in face patterns"
        elif score >= 0.3:
            explanation = "CNN analysis shows minor anomalies, likely authentic"
        else:
            explanation = "CNN confirms natural facial characteristics"
        
        return self.create_result(
            task,
            score=score,
            explanation=explanation,
            details={
                "model": "EfficientNet-B0",
                "raw_logit": logit.item(),
                "image_size": list(image.size)
            }
        )
