"""
Frequency Analysis Agent.

Detects deepfakes using FFT/DCT frequency domain analysis.
GAN-generated images often have distinctive frequency patterns.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import fftpack

from app.config import get_device
from app.core.agent_base import BaseAgent
from app.models.schemas import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)


class FrequencyMLP(nn.Module):
    """Small MLP for classifying frequency features."""
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrequencyAgent(BaseAgent):
    """
    Frequency domain analysis agent.
    
    Uses FFT to extract frequency spectrum features.
    GAN-generated images often have artifacts in high frequencies.
    """
    
    name = "FrequencyAgent"
    description = "FFT/DCT frequency spectrum analysis for GAN artifact detection"
    
    def __init__(self):
        super().__init__()
        self.model: Optional[FrequencyMLP] = None
        self.device = get_device()
        self.feature_dim = 256
    
    async def initialize(self) -> None:
        """Initialize frequency classifier."""
        try:
            self.model = FrequencyMLP(self.feature_dim)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info(f"FrequencyAgent initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FrequencyAgent: {e}")
            raise
    
    def extract_frequency_features(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Extract frequency domain features using FFT.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Feature vector and analysis details
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Apply 2D FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        
        # Compute magnitude spectrum
        magnitude = np.abs(fft_shifted)
        log_magnitude = np.log1p(magnitude)
        
        # Normalize
        log_magnitude = (log_magnitude - log_magnitude.mean()) / (log_magnitude.std() + 1e-8)
        
        # Extract radial frequency profile
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radial bins
        num_bins = self.feature_dim // 2
        max_radius = min(center_y, center_x)
        
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        radial_profile = []
        for i in range(num_bins):
            r_inner = i * max_radius / num_bins
            r_outer = (i + 1) * max_radius / num_bins
            mask = (distances >= r_inner) & (distances < r_outer)
            if mask.any():
                radial_profile.append(log_magnitude[mask].mean())
            else:
                radial_profile.append(0.0)
        
        radial_profile = np.array(radial_profile)
        
        # Extract high-frequency energy ratio
        high_freq_mask = distances > (max_radius * 0.7)
        low_freq_mask = distances < (max_radius * 0.3)
        
        high_energy = magnitude[high_freq_mask].sum() if high_freq_mask.any() else 0
        low_energy = magnitude[low_freq_mask].sum() if low_freq_mask.any() else 1
        
        hf_ratio = high_energy / (low_energy + 1e-8)
        
        # Angular features (for detecting periodic patterns)
        angles = np.arctan2(y_coords - center_y, x_coords - center_x)
        num_angle_bins = self.feature_dim // 2
        
        angular_profile = []
        for i in range(num_angle_bins):
            a_start = -np.pi + i * 2 * np.pi / num_angle_bins
            a_end = -np.pi + (i + 1) * 2 * np.pi / num_angle_bins
            mask = (angles >= a_start) & (angles < a_end)
            if mask.any():
                angular_profile.append(log_magnitude[mask].mean())
            else:
                angular_profile.append(0.0)
        
        angular_profile = np.array(angular_profile)
        
        # Combine features
        features = np.concatenate([radial_profile, angular_profile])
        
        details = {
            "high_frequency_ratio": float(hf_ratio),
            "spectrum_mean": float(log_magnitude.mean()),
            "spectrum_std": float(log_magnitude.std()),
        }
        
        return features, details
    
    async def process(self, task: TaskMessage) -> ResultMessage:
        """Analyze frequency spectrum of image."""
        image_path = task.face_crop_path or task.image_path
        
        try:
            image = np.array(Image.open(image_path).convert('RGB'))
        except Exception as e:
            return self.create_result(
                task,
                score=0.5,
                explanation="Failed to load image",
                details={"error": str(e)}
            )
        
        # Extract features
        features, freq_details = self.extract_frequency_features(image)
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Classify
        with torch.no_grad():
            logit = self.model(feature_tensor)
            base_score = torch.sigmoid(logit).item()
        
        # Adjust score based on high-frequency ratio
        # GAN images often have suppressed high frequencies
        hf_ratio = freq_details["high_frequency_ratio"]
        
        # Heuristic: very low HF ratio suggests GAN
        if hf_ratio < 0.1:
            score = base_score * 0.7 + 0.3  # Bias toward fake
        elif hf_ratio > 0.5:
            score = base_score * 0.7  # Bias toward real
        else:
            score = base_score
        
        # Generate explanation
        if score >= 0.6:
            if hf_ratio < 0.15:
                explanation = "High-frequency spectrum suppression indicates GAN generation"
            else:
                explanation = "Frequency anomalies detected in image spectrum"
        else:
            explanation = "Frequency spectrum consistent with natural photography"
        
        return self.create_result(
            task,
            score=score,
            explanation=explanation,
            details={
                **freq_details,
                "base_score": base_score,
                "feature_dim": self.feature_dim
            }
        )
