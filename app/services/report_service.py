import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class ReportService:
    """Generates evaluation reports in various formats."""

    def __init__(self, output_dir: str = "reports"):
        """Initialize report service."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_evaluation_csv(
        self,
        evaluations: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Save evaluations to CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Convert to DataFrame
        df = pd.DataFrame(evaluations)

        # Reorder columns
        preferred_order = [
            "prompt_id",
            "model",
            "instruction_following",
            "accuracy",
            "completeness",
            "hallucination",
            "overall_score",
            "passed",
        ]

        # Use preferred order if columns exist
        columns = [col for col in preferred_order if col in df.columns]
        remaining = [col for col in df.columns if col not in columns]
        df = df[columns + remaining]

        df.to_csv(filepath, index=False)
        return str(filepath)

    def save_evaluation_json(
        self,
        evaluations: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Save evaluations to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(evaluations, f, indent=2)

        return str(filepath)

    def generate_summary_report(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary statistics from evaluations."""
        if not evaluations:
            return {}

        df = pd.DataFrame(evaluations)

        summary = {
            "total_evaluations": len(evaluations),
            "pass_rate": (df.get("passed", pd.Series([False])).sum() / len(evaluations) * 100)
            if "passed" in df.columns else 0,
            "average_scores": {},
            "hallucination_breakdown": {},
        }

        # Calculate average scores
        score_columns = ["instruction_following", "accuracy", "completeness", "overall_score"]
        for col in score_columns:
            if col in df.columns:
                summary["average_scores"][col] = round(df[col].mean(), 2)

        # Hallucination breakdown
        if "hallucination" in df.columns:
            hallucination_counts = df["hallucination"].value_counts().to_dict()
            summary["hallucination_breakdown"] = {
                "low": hallucination_counts.get("low", 0),
                "medium": hallucination_counts.get("medium", 0),
                "high": hallucination_counts.get("high", 0),
            }

        return summary

    def generate_model_comparison(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate comparison between models."""
        if not evaluations:
            return []

        df = pd.DataFrame(evaluations)

        if "model" not in df.columns:
            return []

        # Group by model
        comparison = []
        for model in df["model"].unique():
            model_df = df[df["model"] == model]

            comparison.append({
                "model": model,
                "count": len(model_df),
                "overall_score": round(model_df.get("overall_score", pd.Series([0])).mean(), 2),
                "instruction_following_score": round(model_df.get("instruction_following", pd.Series([0])).mean(), 2),
                "accuracy_score": round(model_df.get("accuracy", pd.Series([0])).mean(), 2),
                "completeness_score": round(model_df.get("completeness", pd.Series([0])).mean(), 2),
                "hallucination_rate": round(
                    (model_df.get("hallucination", pd.Series(["low"])) == "high").sum() / len(model_df) * 100, 2
                ),
                "pass_rate": round(model_df.get("passed", pd.Series([False])).sum() / len(model_df) * 100, 2),
            })

        # Sort by overall score
        comparison.sort(key=lambda x: x["overall_score"], reverse=True)

        return comparison