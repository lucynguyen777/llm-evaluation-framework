from typing import Optional


class HallucinationEvaluator:
    """Detects hallucinations in LLM responses."""

    def evaluate(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict:
        """Evaluate hallucination risk in response."""
        hallucination_risk = "low"
        confidence = 0.5

        # Empty response → high risk (nothing to verify)
        if not response.strip():
            return {"hallucination_risk": "high", "confidence": 0.85, "passed": False}

        if reference and reference.strip():
            hallucination_risk, confidence = self._detect_against_reference(
                response, reference
            )
        elif context and context.strip():
            hallucination_risk, confidence = self._detect_against_context(
                response, context
            )
        else:
            # No reference or context → cannot verify, flag as medium
            hallucination_risk, confidence = "medium", 0.49

        return {
            "hallucination_risk": hallucination_risk,
            "confidence": round(confidence, 2),
            "passed": hallucination_risk == "low",
        }

    def _detect_against_reference(self, response: str, reference: str) -> tuple:
        """Detect hallucinations by comparing against reference (word-set difference)."""
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())

        if not reference_words:
            return "medium", 0.5

        novel_words = response_words - reference_words
        novel_ratio = len(novel_words) / len(response_words) if response_words else 0

        if novel_ratio > 0.7:
            return "high", 0.9
        elif novel_ratio > 0.6:
            return "medium", 0.7
        else:
            return "low", 0.9

    def _detect_against_context(self, response: str, context: str) -> tuple:
        """Detect hallucinations by comparing against context."""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        overlap = (
            len(response_words & context_words) / len(response_words)
            if response_words
            else 0
        )

        if overlap < 0.3:
            return "high", 0.85
        elif overlap < 0.5:
            return "medium", 0.75
        else:
            return "low", 0.85

    def _detect_internal_consistency(self, response: str) -> tuple:
        """Detect internal contradictions."""
        contradictions = [
            ("yes", "no"),
            ("true", "false"),
            ("always", "never"),
            ("all", "none"),
        ]

        contradiction_count = 0
        for word1, word2 in contradictions:
            if word1 in response.lower() and word2 in response.lower():
                contradiction_count += 1

        if contradiction_count > 2:
            return "high", 0.7
        elif contradiction_count > 0:
            return "medium", 0.6
        else:
            return "medium", 0.5
