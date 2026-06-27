"""
Tests for Evidence and Robustness Verifiers.
"""

import pytest

from app.models.schemas import ResultMessage
from app.verifiers.evidence import EvidenceVerifier
from app.verifiers.robustness import RobustnessVerifier


@pytest.fixture
def sample_results():
    return [
        ResultMessage(
            job_id="test-job",
            agent_name="CNNClassifierAgent",
            score=0.75,
            explanation="CNN detected manipulation",
            latency_ms=150.0,
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="ViTClassifierAgent",
            score=0.82,
            explanation="ViT found synthetic patterns",
            latency_ms=200.0,
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="FrequencyAgent",
            score=0.65,
            explanation="High frequency anomalies",
            latency_ms=50.0,
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="EmbeddingAnomalyAgent",
            score=0.45,
            explanation="Embedding within normal range",
            latency_ms=100.0,
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="FaceXrayLikeAgent",
            score=0.70,
            explanation="Boundary inconsistencies detected",
            latency_ms=80.0,
        ),
    ]


@pytest.fixture
def outlier_results():
    return [
        ResultMessage(
            job_id="test-job",
            agent_name="GoodAgent1",
            score=0.65,
            explanation="Consistent",
            latency_ms=100.0,
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="GoodAgent2",
            score=0.70,
            explanation="Consistent",
            latency_ms=100.0,
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="OutlierAgent",
            score=0.95,
            explanation="Extreme score",
            latency_ms=100.0,
        ),
    ]


class TestEvidenceVerifier:
    def test_outlier_detection(self, outlier_results):
        verifier = EvidenceVerifier(outlier_threshold=0.2)
        adjusted, report = verifier.verify(outlier_results)

        assert report["status"] == "applied"
        assert report["outliers_detected"] == 1
        assert any(f["agent_name"] == "OutlierAgent" for f in report["flags"])

    def test_no_outlier_on_consensus(self, sample_results):
        verifier = EvidenceVerifier(outlier_threshold=0.25)
        adjusted, report = verifier.verify(sample_results)

        # With scores ranging 0.45-0.82, median ~0.7, no outlier beyond 0.25
        assert report["outliers_detected"] == 0

    def test_down_weight_moves_score_toward_median(self, outlier_results):
        verifier = EvidenceVerifier(outlier_threshold=0.2, down_weight_factor=0.5)
        adjusted, report = verifier.verify(outlier_results)

        outlier_flag = next(f for f in report["flags"] if f["agent_name"] == "OutlierAgent")
        assert outlier_flag["down_weighted"] is True
        assert outlier_flag["adjusted_score"] < outlier_flag["original_score"]

    def test_plausibility_issue_flags_near_middle(self):
        verifier = EvidenceVerifier(plausibility_margin=0.15)
        results = [
            ResultMessage(
                job_id="j1", agent_name="WeakAgent", score=0.50, explanation="Weak", latency_ms=10.0
            ),
            ResultMessage(
                job_id="j1", agent_name="StrongAgent", score=0.90, explanation="Strong", latency_ms=10.0
            ),
        ]
        adjusted, report = verifier.verify(results)
        weak_flag = next(f for f in report["flags"] if f["agent_name"] == "WeakAgent")
        assert weak_flag["plausibility_issue"] is True

    def test_empty_results(self):
        verifier = EvidenceVerifier()
        adjusted, report = verifier.verify([])
        assert report["status"] == "no_results"
        assert adjusted == []


class TestRobustnessVerifier:
    def test_down_weights_overconfident_outlier(self):
        verifier = RobustnessVerifier(
            overconfidence_margin=0.05,
            instability_factor=1.0,
            down_weight_factor=0.6,
        )
        results = [
            ResultMessage(
                job_id="j1", agent_name="Stable1", score=0.50, explanation="Stable", latency_ms=10.0
            ),
            ResultMessage(
                job_id="j1", agent_name="Stable2", score=0.52, explanation="Stable", latency_ms=10.0
            ),
            ResultMessage(
                job_id="j1",
                agent_name="Overconfident",
                score=0.97,
                explanation="Overconfident",
                latency_ms=10.0,
            ),
        ]
        adjusted, report = verifier.verify(results)
        flag = next(f for f in report["flags"] if f["agent_name"] == "Overconfident")
        assert flag["overconfident"] is True
        assert flag["down_weighted"] is True

    def test_reports_metrics(self):
        verifier = RobustnessVerifier()
        results = [
            ResultMessage(
                job_id="j1", agent_name="A", score=0.60, explanation="A", latency_ms=10.0
            ),
            ResultMessage(
                job_id="j1", agent_name="B", score=0.80, explanation="B", latency_ms=10.0
            ),
        ]
        _, report = verifier.verify(results)
        assert "trimmed_mean" in report
        assert "stdev" in report
        assert report["agents_reviewed"] == 2

    def test_empty_results(self):
        verifier = RobustnessVerifier()
        adjusted, report = verifier.verify([])
        assert report["status"] == "no_results"
        assert adjusted == []
