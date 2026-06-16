"""Unit tests for LLM Evaluation Framework evaluators."""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluators.instruction_evaluator import InstructionEvaluator
from app.evaluators.accuracy_evaluator import AccuracyEvaluator
from app.evaluators.completeness_evaluator import CompletenessEvaluator
from app.evaluators.hallucination_evaluator import HallucinationEvaluator
from app.evaluators.guideline_engine import GuidelineEngine


def test_instruction_evaluator_format():
    """Test instruction evaluator detects required format."""
    evaluator = InstructionEvaluator()

    result = evaluator.evaluate(
        prompt="Answer only in English and include Verdict and Reason.",
        response="Verdict: Correct\nReason: This follows instructions."
    )

    assert result["instruction_following_score"] >= 4
    assert result["passed"] is True


def test_instruction_evaluator_fail():
    """Test instruction evaluator catches format failures."""
    evaluator = InstructionEvaluator()

    result = evaluator.evaluate(
        prompt="Answer only in English and include Verdict and Reason.",
        response="Yes, the sky is blue."
    )

    assert result["instruction_following_score"] < 4
    assert result["passed"] is False


def test_instruction_evaluator_language_constraint():
    """Test instruction evaluator checks language constraint."""
    evaluator = InstructionEvaluator()

    result = evaluator.evaluate(
        prompt="Answer only in English.",
        response="This is an English response."
    )

    assert result["instruction_following_score"] >= 4


def test_accuracy_evaluator_exact_match():
    """Test accuracy evaluator with exact match."""
    evaluator = AccuracyEvaluator()

    result = evaluator.evaluate(
        response="Paris is the capital of France.",
        reference="Paris is the capital of France."
    )

    assert result["accuracy_score"] >= 4


def test_accuracy_evaluator_partial_match():
    """Test accuracy evaluator with partial match."""
    evaluator = AccuracyEvaluator()

    result = evaluator.evaluate(
        response="Paris is the capital.",
        reference="Paris is the capital of France."
    )

    assert 2 <= result["accuracy_score"] <= 4


def test_accuracy_evaluator_no_match():
    """Test accuracy evaluator with no match."""
    evaluator = AccuracyEvaluator()

    result = evaluator.evaluate(
        response="London is the capital of England.",
        reference="Paris is the capital of France."
    )

    assert result["accuracy_score"] <= 2


def test_accuracy_evaluator_no_reference():
    """Test accuracy evaluator without reference."""
    evaluator = AccuracyEvaluator()

    result = evaluator.evaluate(
        response="Any response here.",
        reference=""
    )

    assert result["accuracy_score"] == 0


def test_completeness_evaluator_full():
    """Test completeness evaluator with full coverage."""
    evaluator = CompletenessEvaluator()

    result = evaluator.evaluate(
        prompt="List 3 items: A, B and C.",
        response="A, B, and C are three items."
    )

    assert result["completeness_score"] >= 3


def test_completeness_evaluator_partial():
    """Test completeness evaluator with partial coverage."""
    evaluator = CompletenessEvaluator()

    result = evaluator.evaluate(
        prompt="List 5 items: A, B, C, D and E.",
        response="A and B are items."
    )

    assert result["completeness_score"] <= 3


def test_completeness_evaluator_empty():
    """Test completeness evaluator with empty response."""
    evaluator = CompletenessEvaluator()

    result = evaluator.evaluate(
        prompt="Answer the question.",
        response=""
    )

    assert result["completeness_score"] == 1


def test_hallucination_evaluator_low_risk():
    """Test hallucination evaluator with matching response."""
    evaluator = HallucinationEvaluator()

    result = evaluator.evaluate(
        response="The sky is blue due to Rayleigh scattering.",
        reference="The sky appears blue because of Rayleigh scattering."
    )

    assert result["hallucination_risk"] == "low"
    assert result["confidence"] >= 0.7


def test_hallucination_evaluator_high_risk():
    """Test hallucination evaluator with contradictory response."""
    evaluator = HallucinationEvaluator()

    result = evaluator.evaluate(
        response="The sky is green due to atmospheric absorption.",
        reference="The sky appears blue because of Rayleigh scattering."
    )

    assert result["hallucination_risk"] == "high"
    assert result["confidence"] >= 0.7


def test_hallucination_evaluator_no_reference():
    """Test hallucination evaluator without reference."""
    evaluator = HallucinationEvaluator()

    result = evaluator.evaluate(
        response="Some response without reference.",
        reference=""
    )

    assert result["hallucination_risk"] == "medium"
    assert result["confidence"] < 0.5


def test_guideline_engine_with_yaml():
    """Test guideline engine loads rules from YAML."""
    config_path = Path(__file__).parent.parent / "app/configs/guidelines.yaml"
    engine = GuidelineEngine(str(config_path))

    rules = engine.get_rules()
    assert len(rules) >= 2


def test_guideline_engine_check_compliance():
    """Test guideline engine compliance checking."""
    config_path = Path(__file__).parent.parent / "app/configs/guidelines.yaml"
    engine = GuidelineEngine(str(config_path))

    compliance = engine.check_compliance(
        prompt="Answer only in English and include Verdict and Reason.",
        response="Verdict: Correct\nReason: The answer follows instructions."
    )

    assert "include_verdict" in compliance
    assert "include_reason" in compliance
    assert "answer_in_english" in compliance
    assert compliance["include_verdict"] is True
    assert compliance["include_reason"] is True
    assert compliance["answer_in_english"] is True


def test_guideline_engine_pass_rate():
    """Test guideline engine pass rate calculation."""
    config_path = Path(__file__).parent.parent / "app/configs/guidelines.yaml"
    engine = GuidelineEngine(str(config_path))

    compliance = {
        "include_verdict": True,
        "include_reason": True,
        "answer_in_english": True
    }

    pass_rate = engine.get_pass_rate(compliance)
    assert pass_rate == 100.0


def test_guideline_engine_pass_rate_partial():
    """Test guideline engine with partial compliance."""
    config_path = Path(__file__).parent.parent / "app/configs/guidelines.yaml"
    engine = GuidelineEngine(str(config_path))

    compliance = {
        "include_verdict": True,
        "include_reason": False,
        "answer_in_english": True
    }

    pass_rate = engine.get_pass_rate(compliance)
    assert pass_rate == 66.67


def test_integration_end_to_end():
    """End-to-end integration test across all evaluators."""
    from app.services.evaluation_service import EvaluationService

    config_path = str(Path(__file__).parent.parent / "app/configs/guidelines.yaml")
    service = EvaluationService(guideline_config_path=config_path)

    result = service.evaluate(
        prompt="Answer only in English and include Verdict and Reason. Is the sky blue?",
        response="Verdict: True\nReason: The sky is blue due to Rayleigh scattering.",
        reference="The sky appears blue because of Rayleigh scattering."
    )

    assert "instruction_following" in result
    assert "accuracy" in result
    assert "completeness" in result
    assert "hallucination" in result
    assert "overall_score" in result
    assert "passed" in result
    assert result["instruction_following"] >= 4
    assert result["hallucination"] == "low"
    assert result["passed"] is True


def test_multi_model_comparison():
    """Test model comparison aggregation."""
    from app.services.evaluation_service import EvaluationService

    config_path = str(Path(__file__).parent.parent / "app/configs/guidelines.yaml")
    service = EvaluationService(guideline_config_path=config_path)

    models = {
        "gpt-4": "Verdict: Correct\nReason: Excellent response following all instructions.",
        "claude-3": "Verdict: Correct\nReason: Good response following instructions.",
        "gemini": "Verdict: Correct\nReason: Response follows instructions adequately.",
    }

    results = []
    for model, response in models.items():
        result = service.evaluate(
            prompt="Answer only in English and include Verdict and Reason. Explain AI.",
            response=response,
            reference="AI is artificial intelligence enabling machines to learn."
        )
        result["model"] = model
        results.append(result)

    assert len(results) == 3

    # All should have scores
    for r in results:
        assert r["instruction_following"] >= 4
        assert r["overall_score"] > 0


def test_empty_response_handling():
    """Test evaluators handle empty responses gracefully."""
    from app.services.evaluation_service import EvaluationService

    config_path = str(Path(__file__).parent.parent / "app/configs/guidelines.yaml")
    service = EvaluationService(guideline_config_path=config_path)

    result = service.evaluate(
        prompt="Answer this question.",
        response="",
        reference=""
    )

    assert result["instruction_following"] == 1
    assert result["accuracy"] == 0
    assert result["completeness"] == 1
    assert result["hallucination"] == "high"
    assert result["overall_score"] == 0.7
    assert result["passed"] is False
