# 🤖 LLM Evaluation Framework

A production-quality framework for evaluating Large Language Model outputs using automated metrics and custom guideline-based review systems. Designed for AI Evaluators, RLHF workflows, Data Quality assessment, and LLM Assessment pipelines.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Instruction Following** | Evaluate format compliance, field presence, language constraints, structure adherence |
| **Accuracy Scoring** | Score factual correctness against reference answers (1-5 scale) |
| **Completeness Assessment** | Detect missing requested information and coverage gaps (1-5 scale) |
| **Hallucination Detection** | Identify hallucinations via response-reference comparison with confidence scores |
| **Guideline Compliance** | Custom YAML-defined rules auto-validated against every response |
| **Multi-Model Comparison** | Compare GPT, Claude, Gemini, and local models with automated rankings |
| **Batch Dataset Evaluation** | Process JSON datasets with CSV/JSON report generation |
| **Interactive Dashboard** | Streamlit UI for overview, model comparison, dataset explorer, guideline checker |
| **REST API** | FastAPI endpoints for `/evaluate`, `/batch-evaluate`, `/leaderboard` |

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                      │
│  ┌─────────────┐  ┌──────────────────────────────────┐ │
│  │   FastAPI    │  │         Streamlit Dashboard       │ │
│  │   :8000      │  │         :8501                     │ │
│  └──────┬───────┘  └──────────────────────────────────┘ │
└─────────┼───────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│                    Services Layer                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Evaluation Service                   │  │
│  │  Orchestrates all evaluators → unified results   │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │               Report Service                     │  │
│  │     CSV/JSON export, summaries, comparisons      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│                   Evaluators Layer                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Instruction│ │Accuracy  │ │Completen.│ │Hallucinat│  │
│  │Evaluator  │ │Evaluator │ │Evaluator │ │Evaluator │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Guideline Engine                     │  │
│  │       YAML-defined rules → compliance check       │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│                    Data Layer                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐           │
│  │ datasets │ │ reports  │ │ configs/*.yml│           │
│  └──────────┘ └──────────┘ └──────────────┘           │
└────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Using pip

```bash
# Clone the repo
cd llm-evaluation-framework

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# In another terminal, run the dashboard
streamlit run dashboard/app.py
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individually:
docker build -t llm-eval .
docker run -p 8000:8000 llm-eval
```

## 💻 Usage Examples

### Python API

```python
from app.services import EvaluationService

service = EvaluationService()
result = service.evaluate(
    prompt="Answer only in English and include Verdict and Reason. Is the sky blue?",
    response="Verdict: True\nReason: The sky appears blue due to Rayleigh scattering.",
    reference="The sky is blue because of Rayleigh scattering."
)

print(result)
# {
#   "instruction_following": 5,
#   "accuracy": 5,
#   "completeness": 5,
#   "hallucination": "low",
#   "overall_score": 5.0,
#   "passed": true
# }
```

### REST API

```bash
# Evaluate a single response
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Answer only in English.",
    "response": "This is an English response.",
    "reference": "",
    "model": "gpt-4"
  }'

# Batch evaluate from dataset
curl -X POST http://localhost:8000/batch-evaluate \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "datasets/evaluation_dataset.json"}'

# Get leaderboard
curl http://localhost:8000/leaderboard

# API docs
open http://localhost:8000/docs
```

### Batch Evaluation with Reports

```python
from app.services import EvaluationService, ReportService

eval_service = EvaluationService()
report_service = ReportService()

# Load dataset
import json
with open("datasets/evaluation_dataset.json") as f:
    dataset = json.load(f)

# Evaluate all items
evaluations = []
for item in dataset:
    result = eval_service.evaluate(
        prompt=item["prompt"],
        response=item["response"],
        reference=item.get("reference")
    )
    result["model"] = item["model"]
    evaluations.append(result)

# Generate CSV and JSON reports
csv_path = report_service.save_evaluation_csv(evaluations)
json_path = report_service.save_evaluation_json(evaluations)

# Print summary
summary = report_service.generate_summary_report(evaluations)
print(summary)
```

## 📡 API Documentation

### `POST /evaluate`

Evaluate a single LLM response.

**Request:**
```json
{
  "prompt": "string (required) - The original prompt/instruction",
  "response": "string (required) - The LLM response to evaluate",
  "reference": "string (optional) - Ground truth reference answer",
  "model": "string (optional) - Model identifier for comparison"
}
```

**Response:**
```json
{
  "instruction_following": 5,
  "accuracy": 4,
  "completeness": 5,
  "hallucination": "low",
  "overall_score": 4.6,
  "passed": true
}
```

### `POST /batch-evaluate`

Run evaluation on a dataset file.

**Request:**
```json
{
  "dataset_path": "string (required) - Path to JSON dataset file"
}
```

**Response:**
```json
{
  "total_evaluated": 15,
  "reports": {
    "csv": "reports/evaluation_results_20240101_120000.csv",
    "json": "reports/evaluation_results_20240101_120000.json"
  },
  "summary": {
    "total_evaluations": 15,
    "pass_rate": 93.33,
    "average_scores": {
      "instruction_following": 4.6,
      "accuracy": 4.4,
      "completeness": 4.7,
      "overall_score": 4.5
    },
    "hallucination_breakdown": {
      "low": 12,
      "medium": 2,
      "high": 1
    }
  },
  "model_comparison": [
    {
      "model": "gpt-4",
      "overall_score": 4.8,
      "pass_rate": 100.0
    },
    {
      "model": "claude-3",
      "overall_score": 4.6,
      "pass_rate": 100.0
    },
    {
      "model": "gemini",
      "overall_score": 4.3,
      "pass_rate": 80.0
    }
  ]
}
```

### `GET /leaderboard`

Returns model rankings from the latest evaluation report.

## 📊 Dashboard

The Streamlit dashboard provides 4 pages:

| Page | Description |
|------|-------------|
| **Overview** | Average score, accuracy, hallucination rate, pass rate with charts |
| **Model Comparison** | Ranking table, radar chart, score distribution violin plots |
| **Dataset Explorer** | Upload JSON datasets, filter by score, run batch evaluations |
| **Guideline Checker** | Paste prompt + response → instant compliance check |

Run: `streamlit run dashboard/app.py`

## 🧪 Testing

```bash
# Run all tests with coverage
cd llm-evaluation-framework
pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_evaluators.py -v
```

## ⚙️ Custom Guidelines

Define rules in `app/configs/guidelines.yaml`:

```yaml
rules:
  - include_verdict          # Response must contain "Verdict:"
  - include_reason           # Response must contain "Reason:"
  - answer_in_english        # Response must be in English
  - include_structured_output  # Response must follow structured format
```

The engine auto-validates these rules against every response.

## 📁 Project Structure

```
llm-evaluation-framework/
├── app/
│   ├── api/                  # FastAPI endpoints
│   ├── evaluators/           # Core evaluation modules
│   ├── schemas/              # Pydantic models
│   ├── services/             # Orchestration & reporting
│   ├── configs/              # YAML guidelines
│   └── utils/                # Helpers
├── datasets/                 # Example evaluation datasets
├── reports/                  # Generated evaluation reports
├── dashboard/                # Streamlit application
├── tests/                    # Unit tests
├── examples/                 # Usage examples
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🛤️ Future Roadmap

### Phase 2
- RAG evaluation — assess retrieval-augmented generation pipelines
- Agent evaluation — evaluate multi-step LLM agent trajectories
- Benchmark suite — standard benchmarks (MMLU, HellaSwag, etc.)
- Human review workflow — integration with annotation platforms
- Annotation quality scoring — inter-annotator agreement metrics

### Phase 3
- RLHF dataset auditing — preference data quality checks
- Reward model scoring — evaluate reward model alignment
- Automatic error categorization — classify failure modes
- Enterprise dashboard — team workspaces, audit logs, SSO

## 📝 License

MIT

---

Built for AI Evaluators, RLHF Engineers, and LLM Quality Assurance workflows.