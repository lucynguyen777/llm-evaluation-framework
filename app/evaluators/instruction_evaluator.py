class InstructionEvaluator:
    """Evaluates how well a response follows instructions."""

    def evaluate(self, prompt: str, response: str) -> dict:
        score = 5
        penalties = 0

        # Empty response penalty
        if not response.strip():
            return {"instruction_following_score": 1, "passed": False}

        # Heuristic: Check if basic instructions are followed
        # For a real implementation, we would use an LLM-as-a-judge here.
        # This is a simplified deterministic proxy.

        # Look for specific requirements in prompt
        if "verdict" in prompt.lower() and "verdict" not in response.lower():
            penalties += 1

        if "reason" in prompt.lower() and "reason" not in response.lower():
            penalties += 1

        if "english" in prompt.lower():
            # Very basic check, in reality use a language detection library
            # Assuming ASCII characters dominate English text
            ascii_ratio = sum(c.isascii() for c in response) / max(len(response), 1)
            if ascii_ratio < 0.8:
                penalties += 2

        score = max(1, score - penalties)

        return {"instruction_following_score": score, "passed": score >= 4}
