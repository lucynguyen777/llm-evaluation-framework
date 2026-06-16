import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))
from app.services import EvaluationService, ReportService

st.set_page_config(
    page_title="LLM Evaluation Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize services
@st.cache_resource
def init_services():
    eval_service = EvaluationService(
        guideline_config_path=str(
            Path(__file__).parent.parent / "app/configs/guidelines.yaml"
        )
    )
    report_service = ReportService()
    return eval_service, report_service


eval_service, report_service = init_services()

# Sidebar
st.sidebar.title("🤖 LLM Evaluation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Comparison", "Dataset Explorer", "Guideline Checker"],
)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def load_latest_report():
    """Load the latest evaluation report."""
    if not REPORTS_DIR.exists():
        return None
    json_reports = list(REPORTS_DIR.glob("*.json"))
    if not json_reports:
        return None
    latest = max(json_reports, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


if page == "Overview":
    st.title("📊 Evaluation Overview")

    # Try to load latest report
    evaluations = load_latest_report()

    col1, col2, col3, col4 = st.columns(4)

    if evaluations and len(evaluations) > 0:
        df = pd.DataFrame(evaluations)

        avg_score = df["overall_score"].mean() if "overall_score" in df else 0
        avg_accuracy = df["accuracy"].mean() if "accuracy" in df else 0
        hallucination_rate = (
            (df["hallucination"] == "high").sum() / len(df) * 100
            if "hallucination" in df
            else 0
        )
        pass_rate = df["passed"].sum() / len(df) * 100 if "passed" in df else 0

        col1.metric("Average Score", round(avg_score, 2))
        col2.metric("Average Accuracy", round(avg_accuracy, 2))
        col3.metric("Hallucination Rate", f"{hallucination_rate:.1f}%")
        col4.metric("Pass Rate", f"{pass_rate:.1f}%")

        st.markdown("---")

        # Score distributions
        st.subheader("Score Distribution")
        score_cols = [
            "instruction_following",
            "accuracy",
            "completeness",
            "overall_score",
        ]
        score_data = {}
        for col in score_cols:
            if col in df.columns:
                score_data[col] = df[col]

        if score_data:
            score_df = pd.DataFrame(score_data)
            fig = px.box(score_df, title="Score Distribution by Metric")
            st.plotly_chart(fig, use_container_width=True)

        # Hallucination breakdown
        st.subheader("Hallucination Risk Breakdown")
        if "hallucination" in df.columns:
            hall_counts = df["hallucination"].value_counts()
            fig = px.pie(
                values=hall_counts.values,
                names=hall_counts.index,
                title="Hallucination Risk",
                color_discrete_map={
                    "low": "#00cc96",
                    "medium": "#ffa15a",
                    "high": "#ef553b",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "No evaluation reports found. Run a batch evaluation or use the Dataset Explorer to upload data."
        )

elif page == "Model Comparison":
    st.title("🏆 Model Comparison")

    evaluations = load_latest_report()

    if evaluations and len(evaluations) > 0:
        df = pd.DataFrame(evaluations)
        comparison = report_service.generate_model_comparison(evaluations)

        if comparison:
            comp_df = pd.DataFrame(comparison)

            # Ranking table
            st.subheader("Ranking Table")
            display_cols = [
                "model",
                "overall_score",
                "instruction_following_score",
                "accuracy_score",
                "completeness_score",
                "hallucination_rate",
                "pass_rate",
            ]
            display_df = comp_df[[c for c in display_cols if c in comp_df.columns]]
            st.dataframe(display_df, use_container_width=True)

            # Radar chart
            st.subheader("Radar Chart")
            metrics = [
                "overall_score",
                "instruction_following_score",
                "accuracy_score",
                "completeness_score",
            ]
            fig = go.Figure()

            for _, row in comp_df.iterrows():
                values = [row.get(m, 0) for m in metrics] + [row.get(metrics[0], 0)]
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=metrics + [metrics[0]],
                        fill="toself",
                        name=row["model"],
                    )
                )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Score distribution by model
            st.subheader("Score Distribution by Model")
            if "model" in df.columns and "overall_score" in df.columns:
                fig = px.violin(
                    df,
                    x="model",
                    y="overall_score",
                    box=True,
                    points="all",
                    title="Overall Score Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model data found in reports.")
    else:
        st.info("No evaluation reports found. Run a batch evaluation first.")

elif page == "Dataset Explorer":
    st.title("📂 Dataset Explorer")

    uploaded_file = st.file_uploader("Upload Evaluation Dataset (JSON)", type=["json"])

    if uploaded_file is not None:
        try:
            dataset = json.load(uploaded_file)
            df = pd.DataFrame(dataset)
            st.success(f"Loaded {len(dataset)} examples")

            # Show examples
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Filter options
            st.subheader("Filter Examples")

            # Score range filter if scores exist
            if "overall_score" in df.columns:
                score_range = st.slider(
                    "Overall Score Range",
                    float(df["overall_score"].min()),
                    float(df["overall_score"].max()),
                    (
                        float(df["overall_score"].min()),
                        float(df["overall_score"].max()),
                    ),
                )
                filtered = df[
                    (df["overall_score"] >= score_range[0])
                    & (df["overall_score"] <= score_range[1])
                ]
                st.write(f"Showing {len(filtered)} of {len(df)} examples")
                st.dataframe(filtered, use_container_width=True)

            # Run batch evaluation button
            if st.button("Run Batch Evaluation"):
                with st.spinner("Evaluating..."):
                    results = []
                    for _, item in df.iterrows():
                        result = eval_service.evaluate(
                            prompt=item.get("prompt", ""),
                            response=item.get("response", ""),
                            reference=item.get("reference", None),
                        )
                        result["prompt_id"] = item.get("id", str(_))
                        result["model"] = item.get("model", "unknown")
                        results.append(result)

                    # Save results
                    csv_path = report_service.save_evaluation_csv(results)
                    json_path = report_service.save_evaluation_json(results)
                    summary = report_service.generate_summary_report(results)

                    st.success("Evaluation complete!")
                    st.json(summary)
                    st.info(
                        f"Reports saved to:\n- CSV: {csv_path}\n- JSON: {json_path}"
                    )

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

elif page == "Guideline Checker":
    st.title("✅ Guideline Compliance Checker")

    st.markdown("""
    Paste a prompt and its response below to check compliance with defined guidelines.
    """)

    col1, col2 = st.columns(2)

    with col1:
        prompt = st.text_area(
            "Prompt / Instruction",
            height=200,
            placeholder="Enter the original prompt/instruction...",
        )

    with col2:
        response = st.text_area(
            "LLM Response", height=200, placeholder="Enter the LLM response..."
        )

    if st.button("Check Compliance", type="primary"):
        if prompt and response:
            with st.spinner("Running compliance checks..."):
                compliance = eval_service.guideline_engine.check_compliance(
                    prompt, response
                )
                pass_rate = eval_service.guideline_engine.get_pass_rate(compliance)
                passed = eval_service.guideline_engine.get_passed_rules(compliance)
                total = eval_service.guideline_engine.get_total_rules()

                st.subheader("Compliance Results")

                col1, col2, col3 = st.columns(3)
                col1.metric("Rules Passed", f"{passed}/{total}")
                col2.metric("Pass Rate", f"{pass_rate:.1f}%")

                # Show detailed results
                st.subheader("Detailed Check Results")
                for rule_name, passed_flag in compliance.items():
                    icon = "✅" if passed_flag else "❌"
                    st.write(
                        f"{icon} **{rule_name.replace('_', ' ').title()}**: {'Passed' if passed_flag else 'Failed'}"
                    )

                # Run full evaluation
                st.subheader("Full Evaluation")
                full_results = eval_service.evaluate(prompt, response)
                st.json(full_results)
        else:
            st.warning("Please enter both a prompt and a response.")
