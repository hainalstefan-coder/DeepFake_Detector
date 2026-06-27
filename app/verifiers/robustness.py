"""
Robustness Verifier.

Assesses signal stability and flags detectors that may be:
- Overconfident (scores too close to 0 or 1)
- Exhibiting extreme variance relative to the ensemble
- Showing suspicious distribution shifts

May down-weight unstable signals before aggregation.
"""

import logging
import statistics
from typing import Dict, List, Tuple

from app.models.schemas import ResultMessage

logger = logging.getLogger(__name__)


class RobustnessVerifier:
    """
    Signal stability verifier.

    Checks:
    - Overconfidence bias (scores near 0 or 1 without strong consensus)
    - Variance-based instability
    - Extreme disagreement with the trimmed mean
    """

    def __init__(
        self,
        overconfidence_margin: float = 0.15,
        instability_factor: float = 2.0,
        down_weight_factor: float = 0.6,
    ):
        self.overconfidence_margin = overconfidence_margin
        self.instability_factor = instability_factor
        self.down_weight_factor = down_weight_factor

    def verify(self, results: List[ResultMessage]) -> Tuple[List[ResultMessage], Dict]:
        """
        Verify robustness of detector signals.

        Args:
            results: list of successful agent results

        Returns:
            (adjusted_results, report)
        """
        if not results:
            return results, {"status": "no_results"}

        scores = [r.score for r in results]
        n = len(scores)

        trimmed_mean = self._trimmed_mean(scores, proportion=0.2)
        stdev = statistics.stdev(scores) if n > 1 else 0.0
        median = statistics.median(scores)

        adjusted: List[ResultMessage] = []
        flags: List[Dict] = []

        for r in results:
            flag = {
                "agent_name": r.agent_name,
                "original_score": r.score,
                "trimmed_mean": trimmed_mean,
                "stdev": stdev,
                "overconfident": False,
                "unstable": False,
                "down_weighted": False,
                "adjustment": 1.0,
            }

            adjusted_score = r.score

            # Overconfidence: score extremely close to 0 or 1 but not strongly supported
            if (
                r.score < self.overconfidence_margin
                or r.score > 1.0 - self.overconfidence_margin
            ):
                # Only flag if this is not in strong agreement with the ensemble
                if abs(r.score - trimmed_mean) > stdev * self.instability_factor:
                    flag["overconfident"] = True
                    flag["down_weighted"] = True
                    flag["adjustment"] = self.down_weight_factor
                    adjusted_score = trimmed_mean + (r.score - trimmed_mean) * self.down_weight_factor
                    adjusted_score = max(0.0, min(1.0, adjusted_score))

            # Instability: score far from median combined with high ensemble variance
            if stdev > 0.2 and abs(r.score - median) > self.instability_factor * stdev:
                flag["unstable"] = True
                if not flag["down_weighted"]:
                    flag["down_weighted"] = True
                    flag["adjustment"] = self.down_weight_factor
                    adjusted_score = median + (r.score - median) * self.down_weight_factor
                    adjusted_score = max(0.0, min(1.0, adjusted_score))

            flag["adjusted_score"] = adjusted_score
            flags.append(flag)

            adjusted.append(
                ResultMessage(
                    job_id=r.job_id,
                    agent_name=r.agent_name,
                    score=adjusted_score,
                    explanation=f"{r.explanation} [robustness_verified]",
                    latency_ms=r.latency_ms,
                    details={**r.details, "robustness_verifier": flag},
                )
            )

        report = {
            "status": "applied",
            "trimmed_mean": trimmed_mean,
            "stdev": stdev,
            "agents_reviewed": n,
            "overconfident_count": sum(1 for f in flags if f["overconfident"]),
            "unstable_count": sum(1 for f in flags if f["unstable"]),
            "down_weighted_count": sum(1 for f in flags if f["down_weighted"]),
            "flags": flags,
        }

        logger.info(
            "RobustnessVerifier: trimmed_mean=%.3f stdev=%.3f overconfident=%d unstable=%d",
            trimmed_mean,
            stdev,
            report["overconfident_count"],
            report["unstable_count"],
        )

        return adjusted, report

    @staticmethod
    def _trimmed_mean(values: List[float], proportion: float = 0.2) -> float:
        """Compute trimmed mean by discarding extreme tails."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        k = int(n * proportion)
        if k == 0:
            return statistics.mean(sorted_vals)
        trimmed = sorted_vals[k:-k] if k > 0 else sorted_vals
        if not trimmed:
            return statistics.mean(sorted_vals)
        return statistics.mean(trimmed)
