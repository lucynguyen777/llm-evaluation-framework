import re
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path


class GuidelineEngine:
    """Checks response compliance against custom guidelines."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the guideline engine with config."""
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        self.rules = self.config.get("rules", {})

    def get_rules(self) -> List[str]:
        """Return list of rule names."""
        return list(self.rules.keys())

    def check_compliance(self, prompt: str, response: str) -> Dict[str, bool]:
        """Check response against all guidelines."""
        results = {}

        for rule_name, rule_config in self.rules.items():
            if isinstance(rule_config, dict):
                results[rule_name] = self._check_rule(
                    rule_name, rule_config, prompt, response
                )
            else:
                results[rule_name] = False

        return results

    def _check_rule(
        self, rule_name: str, rule_config: Dict[str, Any], prompt: str, response: str
    ) -> bool:
        """Check a single rule."""
        check_type = rule_config.get("check_type", "pattern")

        if check_type == "pattern":
            return self._check_pattern(response, rule_config)
        elif check_type == "language":
            return self._check_language(response, rule_config)
        elif check_type == "blacklist":
            return self._check_blacklist(response, rule_config)
        elif check_type == "structure":
            return self._check_structure(response, rule_config)
        else:
            return False

    def _check_pattern(self, response: str, rule: Dict[str, Any]) -> bool:
        """Check if response matches a pattern."""
        pattern = rule.get("pattern", "")
        case_insensitive = rule.get("case_insensitive", False)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            return bool(re.search(pattern, response, flags))
        except re.error:
            return False

    def _check_language(self, response: str, rule: Dict[str, Any]) -> bool:
        """Check if response is in expected language."""
        expected_lang = rule.get("expected_language", "en")

        if expected_lang == "en":
            ascii_ratio = sum(c.isascii() for c in response) / max(len(response), 1)
            return ascii_ratio > 0.7

        return True

    def _check_blacklist(self, response: str, rule: Dict[str, Any]) -> bool:
        """Check if response contains blacklisted words."""
        blacklist = rule.get("blacklist_words", [])

        for word in blacklist:
            if word.lower() in response.lower():
                return False

        return True

    def _check_structure(self, response: str, rule: Dict[str, Any]) -> bool:
        """Check if response has required structure."""
        pattern = rule.get("required_pattern", "")

        try:
            return bool(re.search(pattern, response))
        except re.error:
            return False

    def get_passed_rules(self, results: Dict[str, bool]) -> int:
        """Count how many rules passed."""
        return sum(1 for v in results.values() if v)

    def get_total_rules(self) -> int:
        """Get total number of rules."""
        return len(self.rules)

    def get_pass_rate(self, results: Dict[str, bool]) -> float:
        """Calculate pass rate as percentage.
        Uses the provided results keys as the denominator so that
        partial compliance subsets are scored correctly.
        """
        if not results:
            return 0.0
        passed = self.get_passed_rules(results)
        return round((passed / len(results)) * 100, 2)
