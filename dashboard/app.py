
"""
AI-Powered-NIDS
Professional Security Operations Center Dashboard

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / "models"

sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="AI-Powered-NIDS | Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080d18;
    color: #e5eaf3;
}

[data-testid="stSidebar"] {
    background: #0d1422;
    border-right: 1px solid #1c2a40;
}

[data-testid="stSidebar"] * {
    color: #dce6f5;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1600px;
}

h1, h2, h3 {
    color: #f3f7ff !important;
    font-weight: 700 !important;
}

p, label, .stCaption {
    color: #9caec5;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111d30, #0d1728);
    border: 1px solid #263750;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #91a8c5 !important;
}

[data-testid="stMetricValue"] {
    color: #f3f7ff !important;
}

[data-testid="stMetricDelta"] {
    font-size: 12px;
}

div[data-testid="stFileUploader"] {
    background: #101b2d;
    border: 1px dashed #3a5b80;
    border-radius: 12px;
    padding: 12px;
}

.stButton > button {
    background: linear-gradient(90deg, #1677ff, #00b8d9);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    border: 1px solid #64caff;
    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263750;
    border-radius: 10px;
}

.soc-header {
    background: linear-gradient(115deg, #102541, #102b3e, #102039);
    border: 1px solid #294766;
    border-radius: 16px;
    padding: 25px 28px;
    margin-bottom: 25px;
}

.soc-title {
    font-size: 27px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.soc-subtitle {
    color: #a9bfd8;
    font-size: 13px;
}

.section-label {
    color: #58c7ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 7px;
}

.status-pill {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    background: #153b36;
    color: #5de2ba;
    border: 1px solid #276a5b;
}

.info-panel {
    background: #101b2d;
    border: 1px solid #263750;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
}

hr {
    border-color: #263750;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

@st.cache_resource
def load_models():
    rf_path = MODEL_DIR / "rf_classifier.pkl"
    iso_path = MODEL_DIR / "isolation_forest.pkl"

    clf = joblib.load(rf_path) if rf_path.exists() else None
    iso = joblib.load(iso_path) if iso_path.exists() else None

    return clf, iso


def get_model_features(clf, df):
    """
    Align uploaded data with the features used during training.
    """

    if hasattr(clf, "feature_names_in_"):
        expected = list(clf.feature_names_in_)
    else:
        expected = list(df.select_dtypes(include=[np.number]).columns)

    missing = [c for c in expected if c not in df.columns]

    if missing:
        raise ValueError(
            "Uploaded CSV is missing model features: "
            + ", ".join(missing[:15])
        )

    X = df[expected].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError(
            "Some model features contain missing or invalid values. "
            "Clean the CSV and try again."
        )

    return X, expected


def analyze_traffic(uploaded_file, clf, iso):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("The uploaded CSV contains no records.")

    X, feature_cols = get_model_features(clf, df)

    predictions = clf.predict(X)

    df["Prediction"] = np.where(
        predictions == 1, "Attack", "Benign"
    )

    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(X)

        if 1 in clf.classes_:
            attack_index = list(clf.classes_).index(1)
            df["Attack Probability"] = probabilities[:, attack_index]
        else:
            df["Attack Probability"] = 0.0

    if iso is not None:
        anomaly_predictions = iso.predict(X)
        df["Anomaly"] = np.where(
            anomaly_predictions == -1, "Anomaly", "Normal"
        )
        df["Anomaly Score"] = iso.decision_function(X)

    return df, feature_cols


def get_severity(probability):
    if probability >= 0.90:
        return "Critical"
    elif probability >= 0.75:
        return "High"
    elif probability >= 0.50:
        return "Medium"
    return "Low"


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd8e8"),
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    return fig


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="padding:10px 0 22px 0;">
        <div style="font-size:34px;">🛡️</div>
        <div style="font-size:20px;font-weight:800;color:#ffffff;">
            AI-Powered-NIDS
        </div>
        <div style="font-size:11px;color:#70baff;letter-spacing:1px;">
            SECURITY OPERATIONS CENTER
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Network Analysis",
            "Threat Detection",
            "AI Threat Intelligence",
            "Incident Management",
            "Model Information",
        ],
        label_visibility="visible",
    )

    st.markdown("---")

    st.markdown("### Dataset")

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        help="Upload a CSV containing the features expected by your trained model.",
    )

    st.markdown("---")

    st.markdown("### System Components")

    st.markdown(
        '<span class="status-pill">SOC DASHBOARD ACTIVE</span>',
        unsafe_allow_html=True
    )

    st.caption("ML: Random Forest")
    st.caption("Anomaly Detection: Isolation Forest")
    st.caption("Intelligence: RAG + Gemini")
    st.caption("Incident Storage: SQLite")

    st.markdown("---")
    st.caption("AI-Powered Network Intrusion Detection")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="soc-header">
    <div class="section-label">CYBERSECURITY / SECURITY OPERATIONS</div>
    <div class="soc-title">AI Security Operations Center</div>
    <div class="soc-subtitle">
        Intelligent network threat detection, AI-powered threat intelligence
        and incident investigation.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

try:
    clf, iso = load_models()
    model_error = None
except Exception as e:
    clf, iso = None, None
    model_error = str(e)


if model_error:
    st.error(f"Could not load trained models: {model_error}")


# --------------------------------------------------
# RUN DETECTION
# --------------------------------------------------

analysis_df = None
feature_cols = []

if uploaded_file is not None and clf is not None:

    try:
        with st.spinner("Analyzing network traffic..."):
            analysis_df, feature_cols = analyze_traffic(
                uploaded_file, clf, iso
            )

        st.session_state["analysis_df"] = analysis_df

    except Exception as e:
        st.error(f"Traffic analysis failed: {e}")

elif "analysis_df" in st.session_state:
    analysis_df = st.session_state["analysis_df"]

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown("## Security Overview")
    st.caption("Summary of the currently analyzed traffic dataset.")

    if analysis_df is None:

        st.info(
            "Upload a network traffic CSV from the sidebar to begin "
            "your security analysis."
        )

        if clf is None:
            st.warning(
                "Trained models are not available. "
                "Train the models before running detection."
            )

    else:

        total = len(analysis_df)
        attacks = int((analysis_df["Prediction"] == "Attack").sum())
        benign = total - attacks
        attack_rate = (attacks / total * 100) if total else 0

        anomalies = (
            int((analysis_df["Anomaly"] == "Anomaly").sum())
            if "Anomaly" in analysis_df.columns else 0
        )

        critical = (
            int((analysis_df["Attack Probability"] >= 0.90).sum())
            if "Attack Probability" in analysis_df.columns else 0
        )

        st.markdown(
            '<span class="status-pill">DATASET ANALYSIS COMPLETED</span>',
            unsafe_allow_html=True
        )

        st.markdown("### Key Security Metrics")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Total Flows", f"{total:,}")
        c2.metric("Threats Detected", f"{attacks:,}")
        c3.metric("Benign Flows", f"{benign:,}")
        c4.metric("Anomalies", f"{anomalies:,}")
        c5.metric("Threat Rate", f"{attack_rate:.2f}%")

        st.markdown("---")
        st.markdown("### Traffic Intelligence")

        left, right = st.columns(2)

        with left:

            counts = analysis_df["Prediction"].value_counts()

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.65,
                title="Benign vs Malicious Traffic",
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

        with right:

            if "Attack Probability" in analysis_df.columns:

                fig = px.histogram(
                    analysis_df,
                    x="Attack Probability",
                    color="Prediction",
                    nbins=30,
                    title="Attack Probability Distribution",
                )

                st.plotly_chart(
                    style_chart(fig), use_container_width=True
                )

            else:
                st.info("Model probability output is unavailable.")

        st.markdown("### Recent Detection Results")

        display_cols = [
            c for c in [
                "Prediction",
                "Anomaly",
                "Attack Probability",
                "Anomaly Score",
            ]
            if c in analysis_df.columns
        ]

        st.dataframe(
            analysis_df[display_cols].head(10),
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# NETWORK ANALYSIS
# --------------------------------------------------

elif page == "Network Analysis":

    st.markdown("## Network Traffic Analysis")

    st.caption(
        "Analyze uploaded network flow datasets using trained ML models."
    )

    if analysis_df is None:

        st.info("Upload a CSV dataset using the sidebar.")

    else:

        st.markdown("### Dataset Summary")

        st.write(
            f"**Records:** {len(analysis_df):,}  |  "
            f"**Features used:** {len(feature_cols) if feature_cols else 'Model features'}"
        )

        search = st.text_input(
            "Search detection results",
            placeholder="Enter Attack, Benign or another value..."
        )

        filtered = analysis_df.copy()

        if search:
            mask = filtered.astype(str).apply(
                lambda col: col.str.contains(
                    search, case=False, na=False
                )
            ).any(axis=1)

            filtered = filtered[mask]

        st.dataframe(
            filtered,
            use_container_width=True,
            height=450,
            hide_index=True
        )

        st.download_button(
            "Download Analysis Results",
            data=analysis_df.to_csv(index=False).encode("utf-8"),
            file_name="ai_nids_detection_results.csv",
            mime="text/csv",
        )

        st.markdown("### Feature Statistics")

        numeric = analysis_df.select_dtypes(
            include=[np.number]
        )

        if not numeric.empty:
            st.dataframe(
                numeric.describe().T,
                use_container_width=True
            )


# --------------------------------------------------
# THREAT DETECTION
# --------------------------------------------------

elif page == "Threat Detection":

    st.markdown("## Threat Detection Center")

    st.caption(
        "Machine learning based network intrusion classification "
        "and anomaly identification."
    )

    if analysis_df is None:

        st.info("Upload a network traffic CSV to run threat detection.")

    else:

        attacks_df = analysis_df[
            analysis_df["Prediction"] == "Attack"
        ].copy()

        if "Attack Probability" in attacks_df.columns:

            attacks_df["Severity"] = attacks_df[
                "Attack Probability"
            ].apply(get_severity)

        st.markdown("### Detection Summary")

        a, b, c = st.columns(3)

        a.metric("Detected Attacks", len(attacks_df))

        b.metric(
            "High/Critical Threats",
            int(
                attacks_df["Severity"].isin(
                    ["High", "Critical"]
                ).sum()
            ) if "Severity" in attacks_df.columns else "N/A"
        )

        c.metric(
            "Anomalies",
            int(
                (analysis_df["Anomaly"] == "Anomaly").sum()
            ) if "Anomaly" in analysis_df.columns else "N/A"
        )

        if not attacks_df.empty:

            if "Severity" in attacks_df.columns:

                severity_counts = attacks_df[
                    "Severity"
                ].value_counts().reset_index()

                severity_counts.columns = [
                    "Severity", "Count"
                ]

                fig = px.bar(
                    severity_counts,
                    x="Severity",
                    y="Count",
                    color="Severity",
                    title="Threat Severity Distribution",
                )

                st.plotly_chart(
                    style_chart(fig), use_container_width=True
                )

            st.markdown("### Detected Threats")

            st.dataframe(
                attacks_df.head(500),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.success("No attacks were classified in this dataset.")


# --------------------------------------------------
# AI THREAT INTELLIGENCE
# --------------------------------------------------

elif page == "AI Threat Intelligence":

    st.markdown("## AI Threat Intelligence Assistant")

    st.caption(
        "Investigate network threats using the existing "
        "RAG-based intelligence engine."
    )

    st.markdown(
        '<div class="info-panel">'
        '<b>Intelligence Engine</b><br>'
        '<span style="color:#9caec5;">'
        'Retrieval-Augmented Generation · ChromaDB · Gemini'
        '</span></div>',
        unsafe_allow_html=True
    )

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Alternatively configure GEMINI_API_KEY in your environment."
    )

    query = st.text_area(
        "Threat Investigation Query",
        placeholder=(
            "Example: Explain a DDoS attack, its indicators "
            "and recommended mitigation steps."
        ),
        height=110
    )

    if st.button("Investigate Threat", use_container_width=False):

        if not query.strip():
            st.warning("Enter a threat investigation query.")

        else:

            try:
                from rag_engine import ask_threat_intel

                key = api_key or os.environ.get(
                    "GEMINI_API_KEY", ""
                )

                with st.spinner("Investigating threat intelligence..."):

                    answer, context = ask_threat_intel(
                        query, key
                    )

                st.markdown("### Investigation Results")
                st.markdown(answer)

                with st.expander("Retrieved Intelligence Context"):

                    if context:
                        for item in context:
                            st.markdown(f"- {item}")
                    else:
                        st.info("No retrieved context was returned.")

            except Exception as e:
                st.error(f"Threat intelligence error: {e}")


# --------------------------------------------------
# INCIDENT MANAGEMENT
# --------------------------------------------------

elif page == "Incident Management":

    st.markdown("## Incident Management")

    st.caption(
        "Review incidents recorded by the existing SOC automation module."
    )

    try:

        from automation import get_recent_incidents

        rows = get_recent_incidents()

        if rows:

            columns = [
                "ID",
                "Timestamp",
                "Source IP",
                "Attack Type",
                "Severity",
                "Action Taken",
                "AI Summary",
            ]

            log_df = pd.DataFrame(
                rows, columns=columns
            )

            st.markdown("### Incident Statistics")

            total = len(log_df)

            severity = (
                log_df["Severity"]
                .astype(str)
                .str.lower()
            )

            critical_count = int(
                severity.isin(["critical", "high"]).sum()
            )

            x, y = st.columns(2)

            x.metric("Total Recorded Incidents", total)
            y.metric("High/Critical Incidents", critical_count)

            st.markdown("### Incident Records")

            st.dataframe(
                log_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Export Incident Log",
                data=log_df.to_csv(index=False).encode("utf-8"),
                file_name="nids_incident_log.csv",
                mime="text/csv",
            )

        else:

            st.info(
                "No incidents recorded yet. "
                "Run the SOC agent pipeline to generate incidents."
            )

    except Exception as e:

        st.error(f"Could not load incident log: {e}")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.markdown("## Machine Learning Model Information")

    st.caption(
        "Overview of the models used by the AI-Powered-NIDS platform."
    )

    st.markdown("### Random Forest Classifier")

    if clf is not None:

        st.success("Random Forest model loaded.")

        if hasattr(clf, "n_features_in_"):
            st.metric(
                "Expected Input Features",
                clf.n_features_in_
            )

        if hasattr(clf, "feature_names_in_"):

            st.markdown("#### Training Feature Names")

            st.dataframe(
                pd.DataFrame({
                    "Feature": clf.feature_names_in_
                }),
                use_container_width=True,
                hide_index=True
            )

        if hasattr(clf, "feature_importances_"):

            importance = pd.DataFrame({
                "Feature": (
                    clf.feature_names_in_
                    if hasattr(clf, "feature_names_in_")
                    else [
                        f"Feature {i}"
                        for i in range(len(clf.feature_importances_))
                    ]
                ),
                "Importance": clf.feature_importances_,
            }).sort_values(
                "Importance", ascending=False
            ).head(15)

            fig = px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Random Forest Feature Importances",
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

    else:

        st.warning("Random Forest model file was not found.")

    st.markdown("---")
    st.markdown("### Isolation Forest")

    if iso is not None:
        st.success("Isolation Forest model loaded.")
    else:
        st.warning("Isolation Forest model file was not found.")

    st.markdown("---")

    st.markdown("### Model Artifacts")

    for filename in [
        "rf_classifier.pkl",
        "isolation_forest.pkl",
    ]:

        path = MODEL_DIR / filename

        if path.exists():
            st.success(f"{filename} — Available")
        else:
            st.warning(f"{filename} — Not available")

    st.markdown("---")

    st.markdown("### Explainability")

    st.info(
        "SHAP explainability is not yet connected to this dashboard. "
        "Feature importance shown above comes from the Random Forest "
        "model and is not a SHAP explanation."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#647b96;font-size:11px;">
        AI-POWERED-NIDS · INTELLIGENT NETWORK SECURITY PLATFORM<br>
        Random Forest · Isolation Forest · RAG · Multi-Agent SOC
    </div>
    """,
    unsafe_allow_html=True
)
