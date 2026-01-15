"""
Face preprocessing pipeline.

Simplified version using PIL only (no OpenCV/MTCNN dependency).
Uses basic face detection heuristics.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import get_config
from app.models.schemas import FaceDetection, PreprocessResult

logger = logging.getLogger(__name__)


class FacePreprocessor:
    """
    Simplified face preprocessing pipeline.
    
    For the MVP, uses center-crop approach.
    Can be upgraded to use MTCNN when dependencies are available.
    """
    
    def __init__(self):
        self.config = get_config()
        self.face_size = self.config.preprocessing.face_size
        self.margin = self.config.preprocessing.margin
        self._initialized = False
    
    def _initialize(self) -> None:
        """Initialize preprocessor."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("Initialized simplified face preprocessor")
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Simplified face detection using center crop.
        
        For MVP, assumes face is centered in image.
        """
        self._initialize()
        
        h, w = image.shape[:2]
        
        # Assume face is in center, create bounding box
        # Use 60% of image as face region
        face_ratio = 0.6
        fw, fh = int(w * face_ratio), int(h * face_ratio)
        x1 = (w - fw) // 2
        y1 = (h - fh) // 2
        x2 = x1 + fw
        y2 = y1 + fh
        
        return [FaceDetection(
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            confidence=0.95,
            landmarks=None
        )]
    
    def select_largest(self, faces: List[FaceDetection]) -> Optional[FaceDetection]:
        """Select the largest face by bounding box area."""
        if not faces:
            return None
        
        def area(face: FaceDetection) -> float:
            x1, y1, x2, y2 = face.bbox
            return (x2 - x1) * (y2 - y1)
        
        return max(faces, key=area)
    
    def align_and_crop(
        self,
        image: np.ndarray,
        face: FaceDetection
    ) -> np.ndarray:
        """
        Crop face from image and resize.
        """
        x1, y1, x2, y2 = [int(c) for c in face.bbox]
        h, w = image.shape[:2]
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop
        face_crop = image[y1:y2, x1:x2]
        
        # Resize using PIL
        pil_img = Image.fromarray(face_crop)
        pil_img = pil_img.resize((self.face_size, self.face_size), Image.LANCZOS)
        
        return np.array(pil_img)
    
    def process(
        self,
        image_path: str,
        job_id: str,
        jobs_dir: Optional[str] = None
    ) -> PreprocessResult:
        """
        Full preprocessing pipeline.
        """
        try:
            # Load image using PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image = np.array(pil_image)
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                return PreprocessResult(
                    job_id=job_id,
                    original_path=image_path,
                    faces_detected=0,
                    success=False,
                    error="No faces detected"
                )
            
            # Select largest face
            selected = self.select_largest(faces)
            
            # Crop and align
            face_crop = self.align_and_crop(image, selected)
            
            # Save cropped face
            face_crop_path = None
            if jobs_dir:
                job_dir = Path(jobs_dir) / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                
                face_crop_path = str(job_dir / "face_crop.png")
                Image.fromarray(face_crop).save(face_crop_path)
            
            return PreprocessResult(
                job_id=job_id,
                original_path=image_path,
                face_crop_path=face_crop_path,
                faces_detected=len(faces),
                selected_face=selected,
                warning=None,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}", exc_info=True)
            return PreprocessResult(
                job_id=job_id,
                original_path=image_path,
                success=False,
                error=str(e)
            )


# Global preprocessor instance
_preprocessor: Optional[FacePreprocessor] = None


def get_preprocessor() -> FacePreprocessor:
    """Get the global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = FacePreprocessor()
    return _preprocessor

