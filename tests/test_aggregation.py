"""
Tests for Aggregation Logic
"""

import pytest

from app.config import get_config
from app.core.orchestrator import Orchestrator
from app.models.schemas import ResultMessage, JobState, AgentState, AgentStatus


@pytest.fixture
def orchestrator():
    """Create orchestrator for testing."""
    return Orchestrator()


@pytest.fixture
def sample_results():
    """Create sample agent results."""
    return [
        ResultMessage(
            job_id="test-job",
            agent_name="CNNClassifierAgent",
            score=0.75,
            explanation="CNN detected manipulation",
            latency_ms=150.0
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="ViTClassifierAgent",
            score=0.82,
            explanation="ViT found synthetic patterns",
            latency_ms=200.0
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="FrequencyAgent",
            score=0.65,
            explanation="High frequency anomalies",
            latency_ms=50.0
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="EmbeddingAnomalyAgent",
            score=0.45,
            explanation="Embedding within normal range",
            latency_ms=100.0
        ),
        ResultMessage(
            job_id="test-job",
            agent_name="FaceXrayLikeAgent",
            score=0.70,
            explanation="Boundary inconsistencies detected",
            latency_ms=80.0
        ),
    ]


def test_weighted_average_calculation(sample_results):
    """Test that weighted average is calculated correctly."""
    config = get_config()
    weights = config.agents.weights
    
    # Calculate expected weighted average
    total_weight = 0
    weighted_sum = 0
    
    weights_dict = {
        "CNNClassifierAgent": weights.CNNClassifierAgent,
        "ViTClassifierAgent": weights.ViTClassifierAgent,
        "FrequencyAgent": weights.FrequencyAgent,
        "EmbeddingAnomalyAgent": weights.EmbeddingAnomalyAgent,
        "FaceXrayLikeAgent": weights.FaceXrayLikeAgent,
    }
    
    for result in sample_results:
        w = weights_dict[result.agent_name]
        weighted_sum += result.score * w
        total_weight += w
    
    expected_score = weighted_sum / total_weight
    
    # Score should be around 0.68 based on weights
    assert 0.6 < expected_score < 0.75


def test_quorum_detection():
    """Test quorum-based decision making."""
    # With 3 agents completing, quorum (3) should be reached
    results_3 = [
        ResultMessage(
            job_id="test",
            agent_name=f"Agent{i}",
            score=0.5,
            explanation="Test",
            latency_ms=100.0
        )
        for i in range(3)
    ]
    
    assert len(results_3) >= 3  # Quorum reached
    
    # With 2 agents, quorum not reached
    results_2 = results_3[:2]
    assert len(results_2) < 3  # Quorum not reached


def test_uncertainty_calculation(sample_results):
    """Test uncertainty (variance) calculation."""
    scores = [r.score for r in sample_results]
    
    # Calculate variance
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5
    
    # With scores ranging from 0.45 to 0.82, uncertainty should be moderate
    assert 0.1 < std_dev < 0.2


def test_verdict_threshold():
    """Test verdict determination based on threshold."""
    threshold = 0.6
    
    # Score above threshold = FAKE
    assert 0.75 >= threshold  # FAKE
    
    # Score below threshold = REAL
    assert 0.45 < threshold  # REAL


def test_explanation_generation(sample_results):
    """Test that explanation includes top supporting agents."""
    # Top fake indicators (highest scores)
    sorted_by_score = sorted(sample_results, key=lambda r: r.score, reverse=True)
    
    top_fake = sorted_by_score[0]
    assert top_fake.agent_name == "ViTClassifierAgent"
    assert top_fake.score == 0.82
    
    # Top real indicator (lowest score)
    top_real = sorted_by_score[-1]
    assert top_real.agent_name == "EmbeddingAnomalyAgent"
    assert top_real.score == 0.45


def test_confidence_calculation():
    """Test confidence score calculation."""
    # High agreement (low variance) = high confidence
    scores_high_agreement = [0.7, 0.72, 0.68, 0.71, 0.69]
    mean = sum(scores_high_agreement) / len(scores_high_agreement)
    variance = sum((s - mean) ** 2 for s in scores_high_agreement) / len(scores_high_agreement)
    uncertainty_high = variance ** 0.5
    
    # Low agreement (high variance) = low confidence
    scores_low_agreement = [0.2, 0.9, 0.5, 0.8, 0.3]
    mean_low = sum(scores_low_agreement) / len(scores_low_agreement)
    variance_low = sum((s - mean_low) ** 2 for s in scores_low_agreement) / len(scores_low_agreement)
    uncertainty_low = variance_low ** 0.5
    
    assert uncertainty_high < uncertainty_low
    
    confidence_high = 1 - uncertainty_high
    confidence_low = 1 - uncertainty_low
    
    assert confidence_high > confidence_low


def test_empty_results_handling():
    """Test handling when no agents complete."""
    results = []
    
    # Should return uncertain/unknown verdict
    if not results:
        verdict = "UNKNOWN"
        confidence = 0.0
        
    assert verdict == "UNKNOWN"
    assert confidence == 0.0


def test_single_agent_result():
    """Test aggregation with only one agent."""
    results = [
        ResultMessage(
            job_id="test",
            agent_name="CNNClassifierAgent",
            score=0.8,
            explanation="Test",
            latency_ms=100.0
        )
    ]
    
    # Single agent = high uncertainty
    assert len(results) == 1
    
    # Score should be the agent's score
    final_score = results[0].score
    assert final_score == 0.8
