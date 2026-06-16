import re
from typing import List, Optional


class CompletenessEvaluator:
    """Evaluates whether all requested information is included in the response."""

    def evaluate(self, prompt: str, response: str, required_fields: Optional[List[str]] = None) -> dict:
        """
        Evaluate completeness of response.
        """
        # Empty/very short response → score 1
        if not response.strip():
            return {
                "completeness_score": 1,
                "passed": False,
                "coverage_percentage": 0.0,
                "missing_fields": 1
            }

        # Extract required fields from prompt if not provided
        if not required_fields:
            required_fields = self._extract_required_fields(prompt)

        # If no specific fields found, use word count ratio as proxy
        if not required_fields:
            prompt_word_count = len(prompt.split())
            response_word_count = len(response.split())
            if response_word_count == 0:
                return {"completeness_score": 1, "passed": False, "coverage_percentage": 0.0, "missing_fields": 0}

            ratio = response_word_count / max(prompt_word_count, 1)
            if ratio >= 0.8:
                score = 5
            elif ratio >= 0.5:
                score = 4
            elif ratio >= 0.3:
                score = 3
            elif ratio >= 0.1:
                score = 2
            else:
                score = 1

            return {
                "completeness_score": score,
                "passed": score >= 3,
                "coverage_percentage": ratio * 100,
                "missing_fields": 0
            }

        # Check if all required fields are present
        missing_count = 0
        for field in required_fields:
            if field.lower() not in response.lower():
                missing_count += 1

        coverage = (len(required_fields) - missing_count) / len(required_fields)

        if coverage >= 0.95:
            score = 5
        elif coverage >= 0.8:
            score = 4
        elif coverage >= 0.6:
            score = 3
        elif coverage >= 0.4:
            score = 2
        else:
            score = 1

        return {
            "completeness_score": score,
            "passed": score >= 3,
            "coverage_percentage": coverage * 100,
            "missing_fields": missing_count
        }

    def _extract_required_fields(self, prompt: str) -> List[str]:
        """Extract required fields from prompt."""
        required = []

        # Look for comma-separated items (like "A, B, C, D and E")
        # Match patterns like "items: A, B, C" or "A, B and C"
        items_pattern = re.findall(r'\b([A-Z])\b[,|\s|and]*', prompt)
        if items_pattern and len(items_pattern) >= 2:
            required.extend(items_pattern)

        keywords = [
            "include", "provide", "must have", "required",
            "need", "should contain", "list", "mention"
        ]
        for keyword in keywords:
            if keyword in prompt.lower():
                idx = prompt.lower().find(keyword)
                segment = prompt[idx:idx + 100]
                quoted = re.findall(r'"([^"]+)"', segment)
                caps = re.findall(r'\b([A-Z][a-z]+)\b', segment)
                required.extend(quoted + caps)

        return list(set(required))