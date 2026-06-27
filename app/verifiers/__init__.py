"""
Verifier modules for deepfake detection.

Implements:
- EvidenceVerifier: cross-detector consistency and plausibility checks.
- RobustnessVerifier: signal stability assessment; flags unstable detectors.
"""

from app.models.schemas import ResultMessage

__all__ = ["EvidenceVerifier", "RobustnessVerifier"]
