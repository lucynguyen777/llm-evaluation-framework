from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluators import (
    InstructionEvaluator,
    AccuracyEvaluator,
    CompletenessEvaluator,
    HallucinationEvaluator,
    GuidelineEngine,
)


class EvaluationService:
    """Main service for evaluating LLM responses."""

    def __init__(self, guideline_config_path: Optional[str] = None):
        """Initialize evaluators."""
        self.instruction_evaluator = InstructionEvaluator()
        self.accuracy_evaluator = AccuracyEvaluator()
        self.completeness_evaluator = CompletenessEvaluator()
        self.hallucination_evaluator = HallucinationEvaluator()
        self.guideline_engine = GuidelineEngine(guideline_config_path)

    def evaluate(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of LLM response.
        """
        # Run all evaluations
        instruction_result = self.instruction_evaluator.evaluate(prompt, response)
        accuracy_result = self.accuracy_evaluator.evaluate(response, reference)
        completeness_result = self.completeness_evaluator.evaluate(prompt, response)
        hallucination_result = self.hallucination_evaluator.evaluate(
            response, reference, context
        )
        guideline_result = self.guideline_engine.check_compliance(prompt, response)

        # Calculate overall score (skip accuracy/hallucination weights if no reference)
        ref_available = bool(reference and reference.strip())
        overall_score = self._calculate_overall_score(
            instruction_result.get("instruction_following_score", 3),
            accuracy_result.get("accuracy_score", 3),
            completeness_result.get("completeness_score", 3),
            hallucination_result.get("hallucination_risk", "medium"),
            ref_available,
        )

        # Determine if passed (only check applicable metrics)
        passed_checks = []
        passed_checks.append(instruction_result.get("passed", False))
        passed_checks.append(completeness_result.get("passed", False))
        # Only check accuracy/hallucination when reference is available
        if reference:
            passed_checks.append(accuracy_result.get("passed", False))
            passed_checks.append(hallucination_result.get("passed", False))
        passed = all(passed_checks)

        return {
            "instruction_following": instruction_result.get(
                "instruction_following_score"
            ),
            "accuracy": accuracy_result.get("accuracy_score"),
            "completeness": completeness_result.get("completeness_score"),
            "hallucination": hallucination_result.get("hallucination_risk"),
            "hallucination_confidence": hallucination_result.get("confidence"),
            "overall_score": overall_score,
            "guideline_compliance": guideline_result,
            "passed": passed,
            "details": {
                "instruction": instruction_result,
                "accuracy": accuracy_result,
                "completeness": completeness_result,
                "hallucination": hallucination_result,
            },
        }

    def _calculate_overall_score(
        self,
        instruction_score: int,
        accuracy_score: int,
        completeness_score: int,
        hallucination_risk: str,
        ref_available: bool = False,
    ) -> float:
        """Calculate weighted overall score."""
        # Convert hallucination risk to score
        hallucination_score_map = {"low": 5, "medium": 3, "high": 1}
        hallucination_score = hallucination_score_map.get(hallucination_risk, 3)

        # Weighted average - adjust when no reference
        if ref_available:
            weights = {
                "instruction": 0.3,
                "accuracy": 0.3,
                "completeness": 0.2,
                "hallucination": 0.2,
            }
            overall = (
                instruction_score * weights["instruction"]
                + accuracy_score * weights["accuracy"]
                + completeness_score * weights["completeness"]
                + hallucination_score * weights["hallucination"]
            )
        else:
            # No reference: accuracy info unreliable → skip those metrics
            weights = {
                "instruction": 0.5,
                "completeness": 0.5,
            }
            overall = (
                instruction_score * weights["instruction"]
                + completeness_score * weights["completeness"]
            )

        return round(overall, 2)
