from fastapi import FastAPI, HTTPException
import json
from pathlib import Path

from schemas.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    DatasetEvaluationRequest,
)
from services import EvaluationService, ReportService

app = FastAPI(
    title="LLM Evaluation Framework",
    description="Production-quality LLM evaluation system",
    version="1.0.0",
)

# Initialize services
evaluation_service = EvaluationService(
    guideline_config_path="app/configs/guidelines.yaml"
)
report_service = ReportService()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Evaluation Framework",
        "version": "1.0.0",
        "endpoints": ["/docs", "/evaluate", "/batch-evaluate", "/leaderboard"],
    }


@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate(request: EvaluationRequest):
    """Evaluate a single LLM response."""
    try:
        result = evaluation_service.evaluate(
            prompt=request.prompt,
            response=request.response,
            reference=request.reference,
        )

        # Add model info if provided
        if request.model:
            result["model"] = request.model

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-evaluate")
async def batch_evaluate(request: DatasetEvaluationRequest):
    """Run batch evaluation on a dataset file."""
    try:
        filepath = Path(request.dataset_path)
        if not filepath.exists():
            raise HTTPException(
                status_code=404, detail=f"File not found: {request.dataset_path}"
            )

        # Load dataset
        with open(filepath, "r") as f:
            dataset = json.load(f)

        evaluations = []
        for i, item in enumerate(dataset):
            try:
                result = evaluation_service.evaluate(
                    prompt=item.get("prompt", ""),
                    response=item.get("response", ""),
                    reference=item.get("reference", None),
                )

                result["prompt_id"] = item.get("id", i)
                result["model"] = item.get("model", "unknown")

                evaluations.append(result)

            except Exception as e:
                print(f"Error evaluating item {i}: {e}")

        # Generate reports
        csv_path = report_service.save_evaluation_csv(evaluations)
        json_path = report_service.save_evaluation_json(evaluations)
        summary = report_service.generate_summary_report(evaluations)
        model_comparison = report_service.generate_model_comparison(evaluations)

        return {
            "total_evaluated": len(evaluations),
            "reports": {"csv": csv_path, "json": json_path},
            "summary": summary,
            "model_comparison": model_comparison,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/leaderboard")
async def leaderboard():
    """Get model leaderboard."""
    try:
        # Look for existing reports
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return {"message": "No reports found. Run batch evaluation first."}

        # Find latest JSON report
        json_reports = list(reports_dir.glob("*.json"))
        if not json_reports:
            return {"message": "No evaluation reports found."}

        latest_report = max(json_reports, key=lambda p: p.stat().st_mtime)

        with open(latest_report, "r") as f:
            evaluations = json.load(f)

        model_comparison = report_service.generate_model_comparison(evaluations)

        return {"total_models": len(model_comparison), "leaderboard": model_comparison}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
