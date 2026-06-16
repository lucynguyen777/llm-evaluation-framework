import re
from typing import Optional


class AccuracyEvaluator:
    """Evaluates factual correctness of responses."""

    def evaluate(self, response: str, reference: Optional[str] = None) -> dict:
        """
        Evaluate accuracy of response against reference.
        Uses simple heuristics when reference is not provided.
        """
        # Empty response → lowest score
        if not response.strip():
            return {"accuracy_score": 0, "passed": False}

        # No reference provided → score 0 (cannot verify accuracy)
        if not reference:
            return {"accuracy_score": 0, "passed": False}

        score = self._compare_with_reference(response, reference)

        return {
            "accuracy_score": score,
            "passed": score >= 3
        }

    def _compare_with_reference(self, response: str, reference: str) -> int:
        """Compare response against reference answer via word overlap."""
        response_words = set(self._tokenize(response))
        reference_words = set(self._tokenize(reference))

        if not reference_words or not response_words:
            return 1

        # Jaccard similarity (intersection over union) - more discriminative
        intersection = response_words & reference_words
        union = response_words | reference_words
        similarity = len(intersection) / len(union) if union else 0

        if similarity >= 0.75:
            return 5
        elif similarity >= 0.5:
            return 4
        elif similarity >= 0.3:
            return 3
        elif similarity >= 0.1:
            return 2
        else:
            return 1

    def _tokenize(self, text: str) -> set:
        """Tokenize text into clean word set."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter out common stop words that inflate similarity
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "in", "on", "at", "to", "for", "of", "and", "or", "but",
            "it", "its", "this", "that", "these", "those",
            "with", "without", "by", "from", "as", "has", "had", "have",
        }
        return {w for w in words if w not in stop_words}