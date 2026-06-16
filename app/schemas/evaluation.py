from pydantic import BaseModel, Field
from typing import Dict, Optional


class EvaluationRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt or instruction")
    response: str = Field(..., description="The LLM's response")
    reference: Optional[str] = Field(
        None, description="Ground truth or reference answer"
    )
    model: Optional[str] = Field(None, description="The name of the evaluated model")


class DatasetEvaluationRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the JSON dataset file")


class EvaluationResult(BaseModel):
    instruction_following: int = Field(
        ..., description="Instruction following score (1-5)"
    )
    accuracy: int = Field(..., description="Accuracy score (1-5)")
    completeness: int = Field(..., description="Completeness score (1-5)")
    hallucination: str = Field(
        ..., description="Hallucination risk (low, medium, high)"
    )
    overall_score: float = Field(..., description="Calculated overall score")
    guideline_compliance: Dict[str, bool] = Field(
        default_factory=dict, description="Compliance with specific guidelines"
    )
    passed: bool = Field(
        ..., description="Whether the response passed the evaluation criteria"
    )


class ModelComparison(BaseModel):
    model: str
    overall_score: float
    instruction_following_score: float
    accuracy_score: float
    completeness_score: float
    hallucination_rate: float
    pass_rate: float
