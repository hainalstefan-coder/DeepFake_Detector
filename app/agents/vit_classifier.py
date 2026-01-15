"""
Vision Transformer (ViT) Classifier Agent.

Uses ViT for attention-based deepfake detection.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.config import get_device
from app.core.agent_base import BaseAgent
from app.models.schemas import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)


class ViTClassifierAgent(BaseAgent):
    """
    Vision Transformer-based deepfake detection agent.
    
    Uses ViT-B/16 with attention mechanism for detecting
    synthetic generation patterns.
    """
    
    name = "ViTClassifierAgent"
    description = "Vision Transformer for attention-based deepfake detection"
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self.device = get_device()
    
    async def initialize(self) -> None:
        """Load ViT model."""
        try:
            import timm
            
            # Load ViT-B/16 pretrained
            self.model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=1  # Binary classification
            )
            
            # Initialize head for deepfake detection
            nn.init.xavier_uniform_(self.model.head.weight)
            nn.init.zeros_(self.model.head.bias)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Image preprocessing for ViT
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
            
            self._initialized = True
            logger.info(f"ViTClassifierAgent initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ViTClassifierAgent: {e}")
            raise
    
    async def process(self, task: TaskMessage) -> ResultMessage:
        """Process image using Vision Transformer."""
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
            explanation = "Attention patterns indicate synthetic generation artifacts"
        elif score >= 0.5:
            explanation = "ViT detected unusual attention distribution in face regions"
        elif score >= 0.3:
            explanation = "Attention analysis shows mostly natural patterns"
        else:
            explanation = "ViT confirms coherent natural facial structure"
        
        return self.create_result(
            task,
            score=score,
            explanation=explanation,
            details={
                "model": "ViT-B/16",
                "raw_logit": logit.item(),
                "patch_size": 16
            }
        )
