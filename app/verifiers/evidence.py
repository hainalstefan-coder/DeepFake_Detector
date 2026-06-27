"""
Evidence Verifier.

Performs cross-detector consistency and plausibility checks.
May down-weight or flag evidence from detectors that disagree
strongly with the consensus.
"""

import logging
import statistics
from typing import Dict, List, Optional, Tuple

from app.models.schemas import ResultMessage

logger = logging.getLogger(__name__)


class EvidenceVerifier:
    """
    Cross-detector consistency verifier.

    Metrics:
    - Pairwise score divergence
    - Distance from median (outlier detection)
    - Plausibility band check (scores near 0.5 are less plausible as strong evidence)
    """

    def __init__(
        self,
        outlier_threshold: float = 0.25,
        plausibility_margin: float = 0.15,
        down_weight_factor: float = 0.5,
    ):
        self.outlier_threshold = outlier_threshold
        self.plausibility_margin = plausibility_margin
        self.down_weight_factor = down_weight_factor

    def verify(self, results: List[ResultMessage]) -> Tuple[List[ResultMessage], Dict]:
        """
        Verify cross-detector consistency.

        Args:
            results: list of successful agent results

        Returns:
            (adjusted_results, report)
        """
        if not results:
            return results, {"status": "no_results"}

        scores = [r.score for r in results]
        median_score = statistics.median(scores)
        mean_score = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        adjusted: List[ResultMessage] = []
        flags: List[Dict] = []

        for r in results:
            flag = {
                "agent_name": r.agent_name,
                "original_score": r.score,
                "median_score": median_score,
                "divergence": abs(r.score - median_score),
                "plausibility_issue": abs(r.score - 0.5) < self.plausibility_margin,
                "outlier": False,
                "down_weighted": False,
                "adjustment": 1.0,
            }

            adjusted_score = r.score

            # Outlier check
            if abs(r.score - median_score) > self.outlier_threshold:
                flag["outlier"] = True
                flag["down_weighted"] = True
                flag["adjustment"] = self.down_weight_factor
                # Pull score toward median
                adjusted_score = median_score + (r.score - median_score) * self.down_weight_factor
                adjusted_score = max(0.0, min(1.0, adjusted_score))

            # Plausibility check: near-0.5 scores are weak evidence
            if flag["plausibility_issue"]:
                flag["plausibility_issue"] = True
                # Slightly reduce confidence in this detector's contribution
                if not flag["down_weighted"]:
                    flag["adjustment"] = 0.8
                    adjusted_score = median_score + (r.score - median_score) * 0.8
                    adjusted_score = max(0.0, min(1.0, adjusted_score))

            flag["adjusted_score"] = adjusted_score
            flags.append(flag)

            # Build adjusted result (preserve all original fields)
            adjusted.append(
                ResultMessage(
                    job_id=r.job_id,
                    agent_name=r.agent_name,
                    score=adjusted_score,
                    explanation=f"{r.explanation} [evidence_verified]",
                    latency_ms=r.latency_ms,
                    details={**r.details, "evidence_verifier": flag},
                )
            )

        report = {
            "status": "applied",
            "median_score": median_score,
            "mean_score": mean_score,
            "stdev": stdev,
            "agents_reviewed": len(results),
            "outliers_detected": sum(1 for f in flags if f["outlier"]),
            "plausibility_issues": sum(1 for f in flags if f["plausibility_issue"]),
            "flags": flags,
        }

        logger.info(
            "EvidenceVerifier: median=%.3f stdev=%.3f outliers=%d",
            median_score,
            stdev,
            report["outliers_detected"],
        )

        return adjusted, report
