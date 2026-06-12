import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Drift Detective", layout="wide")
st.title("Drift Detective")
st.caption("ML Model Observability & Drift Detection Platform")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Upload Data")

baseline_file = st.sidebar.file_uploader("Upload Baseline CSV", type=["csv"])
current_file = st.sidebar.file_uploader("Upload Current CSV", type=["csv"])

if baseline_file:
    res = requests.post(
        f"{API_URL}/upload/baseline",
        files={"file": ("baseline.csv", baseline_file.getvalue(), "text/csv")}
    )
    if res.status_code == 200:
        data = res.json()
        st.sidebar.success(f"Baseline loaded: {data['rows']} rows, {len(data['features'])} features")
    else:
        st.sidebar.error("Failed to upload baseline")

# ── Main Panel ────────────────────────────────────────────────────────────────
if current_file and baseline_file:
    res = requests.post(
        f"{API_URL}/upload/current",
        files={"file": ("current.csv", current_file.getvalue(), "text/csv")}
    )

    if res.status_code == 200:
        results = res.json()

        # ── Model Health Banner
        health = results["model_health"]
        color = {"healthy": "green", "degraded": "orange", "critical": "red"}.get(health, "gray")
        st.markdown(
            f"### Model Health: :{color}[{health.upper()}]"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Features", results["total_features"])
        col2.metric("Drifted Features", results["drifted_features"])
        col3.metric("Drift Rate", f"{round(results['drifted_features'] / results['total_features'] * 100, 1)}%")

        st.divider()

        # ── Feature Drift Table
        st.subheader("Feature Drift Summary")

        rows = []
        for f in results["ranked_features"]:
            rows.append({
                "Feature": f["feature"],
                "KS Statistic": f["ks_test"]["statistic"],
                "PSI Score": f["psi"]["psi_score"],
                "KL Score": f["kl_divergence"]["kl_score"],
                "Drift Votes": f["drift_votes"],
                "Severity": f["overall_severity"],
                "Drift Detected": "✅" if f["drift_detected"] else "❌"
            })

        df_display = pd.DataFrame(rows)
        st.dataframe(df_display, use_container_width=True)

        st.divider()

        # ── Bar Chart: PSI scores per feature
        st.subheader("PSI Score by Feature")
        fig = px.bar(
            df_display,
            x="Feature",
            y="PSI Score",
            color="Severity",
            color_discrete_map={
                "none": "green",
                "low": "yellow",
                "moderate": "orange",
                "high": "red"
            }
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Moderate threshold")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="High threshold")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Retraining Recommendation
        st.subheader("Recommendation")
        if health == "critical":
            st.error("🔴 Significant drift detected across majority of features. Full retraining recommended.")
        elif health == "degraded":
            st.warning("🟠 Moderate drift detected. Monitor closely and consider fine-tuning on recent data.")
        else:
            st.success("🟢 Model distributions are stable. No action required.")

    else:
        st.error(f"Error: {res.json().get('detail', 'Something went wrong')}")

else:
    st.info("Upload both baseline and current CSV files from the sidebar to run drift detection.")