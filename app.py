"""
================================================================================
BITS Pilani - Work Integrated Learning Programmes (WILP)
M.Tech in Artificial Intelligence & Machine Learning (AIML)
Course: Machine Learning | Assignment - 2 (Marks: 15)
Student ID: 2026Ak05001
Interactive Streamlit Web Application for Model Demonstration & Evaluation
================================================================================
Dataset: Bank Marketing Dataset (UCI Machine Learning Repository / Kaggle)
Criteria Satisfied:
  - Public Repository: UCI ML Repository / Kaggle
  - Number of Features: 16 (Minimum requirement: >= 12)
  - Number of Instances: 4,521 (Minimum requirement: >= 500)
  - Task Type: Binary Classification (deposit: 0 = No, 1 = Yes)
Models Implemented:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier (KNN)
  4. Naive Bayes Classifier (GaussianNB)
  5. Support Vector Machine (SVM with RBF kernel)
Evaluation Metrics:
  1. Accuracy
  2. AUC Score (ROC-AUC)
  3. Precision
  4. Recall
  5. F1 Score
  6. Matthews Correlation Coefficient (MCC Score)
================================================================================
"""

import os
import io
import json
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 | BITS Pilani (2026Ak05001)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished BITS Pilani AIML academic branding & enlarged typography
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0b2545 0%, #134074 50%, #001220 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .main-header h1 {
        color: #eef4f8;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        padding-bottom: 6px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 15px;
        margin: 0;
    }
    .dataset-badge-default {
        background: linear-gradient(90deg, #e0f2fe 0%, #f0f9ff 100%);
        border-left: 7px solid #0284c7;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 22px;
        box-shadow: 0 3px 8px rgba(2, 132, 199, 0.12);
    }
    .dataset-badge-default h2 {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0369a1 !important;
        margin: 10px 0 6px 0 !important;
    }
    .dataset-badge-default p {
        font-size: 17px !important;
        line-height: 1.6 !important;
        color: #0c4a6e !important;
        margin: 0 !important;
    }
    .dataset-badge-upload {
        background: linear-gradient(90deg, #fef3c7 0%, #fffbeb 100%);
        border-left: 7px solid #d97706;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 22px;
        box-shadow: 0 3px 8px rgba(217, 119, 6, 0.12);
    }
    .dataset-badge-upload h2 {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #b45309 !important;
        margin: 10px 0 6px 0 !important;
    }
    .dataset-badge-upload p {
        font-size: 17px !important;
        line-height: 1.6 !important;
        color: #78350f !important;
        margin: 0 !important;
    }
    .enlarged-clear-btn button {
        font-size: 18px !important;
        font-weight: 800 !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        background-color: #dc2626 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(220, 38, 38, 0.25) !important;
        cursor: pointer !important;
    }
    .enlarged-clear-btn button:hover {
        background-color: #b91c1c !important;
        box-shadow: 0 6px 14px rgba(220, 38, 38, 0.35) !important;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & FEATURE DEFINITIONS
# -----------------------------------------------------------------------------
STUDENT_ID = "2026Ak05001"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_TEST_PATH = os.path.join(DATA_DIR, "test_data.csv")

NUMERIC_FEATURES = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'deposit'

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "K-Nearest Neighbor Classifier",
    "Naive Bayes Classifier",
    "Support Vector Machine"
]

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree Classifier": "decision_tree_classifier.joblib",
    "K-Nearest Neighbor Classifier": "k-nearest_neighbor_classifier.joblib",
    "Naive Bayes Classifier": "naive_bayes_classifier.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS: CACHE MANAGEMENT & RETRIEVAL
# -----------------------------------------------------------------------------
def clear_application_cache():
    """Purges Streamlit data/resource caches, closes open plots, and triggers garbage collection."""
    st.cache_data.clear()
    st.cache_resource.clear()
    plt.close('all')
    gc.collect()


def ensure_models_exist():
    """Checks if model files exist; if not, runs training script on startup."""
    missing = [
        f for f in MODEL_FILE_MAP.values()
        if not os.path.exists(os.path.join(MODELS_DIR, f))
    ]
    if missing or not os.path.exists(DEFAULT_TEST_PATH):
        with st.spinner("Initializing models and dataset on first launch... Please wait a few seconds."):
            import train
            train.train_and_evaluate_all()


@st.cache_data(ttl=3600, max_entries=5)
def load_default_test_data():
    """Loads the fixed test dataset selected initially."""
    if os.path.exists(DEFAULT_TEST_PATH):
        return pd.read_csv(DEFAULT_TEST_PATH)
    else:
        ensure_models_exist()
        return pd.read_csv(DEFAULT_TEST_PATH)


@st.cache_resource(ttl=3600, max_entries=10)
def load_saved_model(model_name: str):
    """Loads a pre-trained scikit-learn pipeline from disk without retraining."""
    ensure_models_exist()
    file_name = MODEL_FILE_MAP[model_name]
    file_path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(file_path):
        ensure_models_exist()
    return joblib.load(file_path)


def compute_all_metrics(y_true, y_pred, y_proba):
    """Computes all 6 required evaluation metrics."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC Score": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC Score": float(matthews_corrcoef(y_true, y_pred))
    }


def validate_test_dataframe(df: pd.DataFrame):
    """
    Validates uploaded test CSV:
    1. Must contain target column ('deposit' or 'y')
    2. Must contain at least 12 input features (excluding target)
    3. Must contain the required feature columns expected by the pre-trained pipelines
    """
    # 1. Target Column Check
    has_target = TARGET_COLUMN in df.columns or 'y' in df.columns
    if not has_target:
        return False, f"Target column '{TARGET_COLUMN}' (or 'y') is missing from the uploaded CSV. Ground-truth target is required to compute evaluation metrics."

    # Normalize target column name if 'y' was passed
    if 'y' in df.columns and TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = df['y']

    # 2. Minimum 12 Input Features Check
    input_features_present = [c for c in df.columns if c not in [TARGET_COLUMN, 'y']]
    if len(input_features_present) < 12:
        return False, f"Uploaded dataset has only {len(input_features_present)} input features. Assignment Step 1 mandates a Minimum Feature Size of 12."

    # 3. Model Expected Feature Columns Check
    missing_required = [col for col in ALL_FEATURES if col not in df.columns]
    if missing_required:
        return False, f"Uploaded CSV is missing required feature columns expected by the trained models: {missing_required}. (Found: {len(input_features_present)} features)."

    # Map yes/no strings to 1/0 if target is formatted as text
    if df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].str.lower().map({'yes': 1, 'no': 0, '1': 1, '0': 0})

    return True, "Validation Successful"


# -----------------------------------------------------------------------------
# 4. SESSION STATE FOR DATASET CLEAR / RESET
# -----------------------------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# -----------------------------------------------------------------------------
# 5. SIDEBAR CONTROLS (WITH BITS ID: 2026Ak05001)
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/BITS_Pilani-Logo.svg/1200px-BITS_Pilani-Logo.svg.png",
    width=130
)

# Prominent BITS ID in sidebar
st.sidebar.markdown(f"""
<div style="background-color: #0b2545; color: white; padding: 12px 14px; border-radius: 8px; text-align: center; margin: 10px 0 16px 0; border: 1.5px solid #1e40af;">
    <span style="font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: #94a3b8; font-weight: 700;">Student ID / BITS ID</span><br>
    <span style="font-size: 21px; font-weight: 800; letter-spacing: 1.5px; color: #38bdf8;">{STUDENT_ID}</span><br>
    <span style="font-size: 12px; color: #cbd5e1;">M.Tech AIML | Machine Learning</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### **M.Tech AIML - ML Assignment 2**")
st.sidebar.markdown("**Course:** Machine Learning | **Marks:** 15")
st.sidebar.markdown("---")

# Dataset Information Card
st.sidebar.markdown("#### 📂 **Dataset Choice & Specifications**")
st.sidebar.info("""
**Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **Repository:** UCI ML / Kaggle
* **Instances:** 4,521 *(Requirement: ≥ 500)*
* **Features:** 16 *(Requirement: ≥ 12)*
* **Task:** Binary Classification
* **Target:** `deposit` (0 = No, 1 = Yes)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧪 **Step 3.1: Test Dataset Selection**")

data_source = st.sidebar.radio(
    "Choose Test Data Mode:",
    options=["Default Test Dataset (Default)", "Upload Custom Test CSV"],
    help="Step 3: Default fixed test dataset is loaded on launch. Upload option allows testing custom CSVs without retraining."
)

default_df = load_default_test_data()
active_df = None
is_custom_dataset = False
custom_filename = ""
upload_error_msg = None

if data_source == "Default Test Dataset (Default)":
    active_df = default_df
    is_custom_dataset = False
    st.sidebar.success(f"Loaded Default Test Dataset (**{len(active_df)} instances**)")
else:
    # Requirement: Upload option with strict validation
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test CSV (≥ 12 features + target)",
        type=["csv"],
        key=f"uploader_{st.session_state['uploader_key']}",
        help="Upload only test data. Models evaluate without retraining."
    )

    if uploaded_file is not None:
        custom_filename = uploaded_file.name
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            is_valid, msg = validate_test_dataframe(uploaded_df)
            if is_valid:
                active_df = uploaded_df
                is_custom_dataset = True
                upload_error_msg = None
                st.sidebar.success(f"✅ Valid test file: {len(active_df)} instances, {len([c for c in uploaded_df.columns if c != TARGET_COLUMN])} features.")
            else:
                # Requirement 1: DO NOT LOAD DEFAULT DATASET ON UNEXPECTED FILES
                upload_error_msg = msg
                active_df = None
                is_custom_dataset = True
                st.sidebar.error(f"❌ {upload_error_msg}")
        except Exception as e:
            upload_error_msg = f"Error parsing CSV file: {e}"
            active_df = None
            is_custom_dataset = True
            st.sidebar.error(f"❌ {upload_error_msg}")
    else:
        # Show default dataset while waiting for upload
        active_df = default_df
        is_custom_dataset = False
        upload_error_msg = None
        st.sidebar.info("ℹ️ No file uploaded yet. Showing Default Test Dataset. Upload a CSV to evaluate custom test data.")

    # Clear button in sidebar
    if uploaded_file is not None:
        if st.sidebar.button("🔄 Clear Uploaded Dataset & Load Default", type="secondary", use_container_width=True):
            st.session_state["uploader_key"] += 1
            clear_application_cache()
            st.rerun()

# Download button for sample test data
st.sidebar.markdown("---")
sample_csv = default_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Sample Test CSV",
    data=sample_csv,
    file_name="sample_test_data.csv",
    mime="text/csv",
    help="Download this test CSV to verify the upload feature."
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🤖 **Step 3.2: Model Selection Dropdown**")
model_options = ["All Models (Comparative Benchmark)"] + MODEL_NAMES
selected_model_option = st.sidebar.selectbox(
    "Select Model to Inspect:",
    options=model_options,
    index=0,
    help="Select 'All Models' to view comparative metrics, confusion matrices, and reports, or pick an individual model."
)

# Memory & Cache Management for Cloud Optimization
st.sidebar.markdown("---")
with st.sidebar.expander("⚡ Cloud Memory & Cache Manager", expanded=False):
    st.caption("Streamlit Community Cloud has a ~1 GB RAM ceiling. Clear cached objects to optimize cloud performance and prevent memory spikes.")
    if st.button("🧹 Clear All Cache & Free RAM", use_container_width=True, help="Purges st.cache_data, st.cache_resource, closes Matplotlib figure canvases, and runs garbage collection"):
        clear_application_cache()
        st.success("Cache cleared & memory reclaimed!")
        st.rerun()

st.sidebar.caption(f"BITS Pilani WILP Division | M.Tech (AIML) | ID: {STUDENT_ID}")

# -----------------------------------------------------------------------------
# 6. MAIN CONTENT HEADER (WITH BITS ID: 2026Ak05001)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div>
            <h1>BITS Pilani — Machine Learning Classification Dashboard</h1>
            <p>M.Tech in Artificial Intelligence & Machine Learning | Assignment - 2 (15 Marks)</p>
        </div>
        <div style="background: rgba(255, 255, 255, 0.12); border: 1.5px solid rgba(255, 255, 255, 0.35); padding: 8px 18px; border-radius: 10px; text-align: center;">
            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #93c5fd; font-weight: bold;">BITS Student ID</span><br>
            <span style="font-size: 22px; font-weight: 800; color: #ffffff; letter-spacing: 1.5px;">{STUDENT_ID}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# REQUIREMENT 1 & 2: ERROR SCREEN WHEN UNEXPECTED FILE UPLOADED (NO DEFAULT LOAD)
# -----------------------------------------------------------------------------
if upload_error_msg is not None:
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #fee2e2 0%, #fff1f2 100%); border-left: 7px solid #dc2626; padding: 22px 26px; border-radius: 10px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.12);">
        <span style="background-color: #dc2626; color: white; padding: 5px 14px; border-radius: 4px; font-weight: 800; font-size: 13px; letter-spacing: 0.5px;">UPLOAD VALIDATION ERROR</span>
        <h2 style="color: #991b1b; margin: 12px 0 8px 0; font-size: 26px; font-weight: 800;">❌ Unexpected File Uploaded: {custom_filename}</h2>
        <p style="margin: 0 0 12px 0; color: #7f1d1d; font-size: 18px; font-weight: 600;">
            {upload_error_msg}
        </p>
        <p style="margin: 0; color: #991b1b; font-size: 15px;">
            <b>Evaluation Blocked:</b> To preserve evaluation integrity, default data is <b>not loaded</b> for invalid or mismatched files. Please upload a valid CSV containing at least 12 input features and the target column (<code>deposit</code> or <code>y</code>), or click the clear button below to restore the default dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enlarged Clear Button
    col_err_btn, _ = st.columns([2, 3])
    with col_err_btn:
        st.markdown('<div class="enlarged-clear-btn">', unsafe_allow_html=True)
        if st.button("🔄 Clear Unexpected File & Load Default", key="err_screen_reset_btn"):
            st.session_state["uploader_key"] += 1
            clear_application_cache()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Stop execution cleanly so default dataset metrics are NOT loaded
    st.stop()

# -----------------------------------------------------------------------------
# REQUIREMENT 2: ENLARGED FONT SIZE FOR ACTIVE DATASET TITLE & CLEAR BUTTON
# -----------------------------------------------------------------------------
if not is_custom_dataset:
    st.markdown(f"""
    <div class="dataset-badge-default">
        <span style="background-color: #0284c7; color: white; padding: 5px 14px; border-radius: 4px; font-weight: 800; font-size: 13px; letter-spacing: 0.5px;">DEFAULT DATASET ACTIVE</span>
        <h2>📂 ACTIVE DATASET: Default Test Dataset</h2>
        <p>
            <b>Dataset Source:</b> UCI Bank Marketing (Fixed 20% Stratified Holdout Test Split)<br>
            <b>Dataset Specifications:</b> <span style="font-weight: 700; color: #0369a1;">905 rows × 17 columns</span> &nbsp;|&nbsp; 
            <span style="font-weight: 700; color: #0369a1;">16 Input Features</span> (Meets criteria ≥ 12) &nbsp;|&nbsp; 
            <b>Target Column:</b> <code>{TARGET_COLUMN}</code> (Binary: 0 = No, 1 = Yes)
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    num_input_features = len([c for c in active_df.columns if c != TARGET_COLUMN])
    st.markdown(f"""
    <div class="dataset-badge-upload">
        <span style="background-color: #d97706; color: white; padding: 5px 14px; border-radius: 4px; font-weight: 800; font-size: 13px; letter-spacing: 0.5px;">CUSTOM UPLOADED DATASET ACTIVE</span>
        <h2>📤 ACTIVE DATASET: Uploaded File — {custom_filename}</h2>
        <p>
            <b>Dataset Status:</b> Custom Test Cohort Loaded Successfully<br>
            <b>Dataset Specifications:</b> <span style="font-weight: 700; color: #b45309;">{len(active_df)} rows × {active_df.shape[1]} columns</span> &nbsp;|&nbsp; 
            <span style="font-weight: 700; color: #b45309;">{num_input_features} Input Features Validated</span> (Meets criteria ≥ 12) &nbsp;|&nbsp; 
            <b>Model Evaluation:</b> Evaluated directly on pre-saved pipelines without retraining
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enlarged Clear Button on main screen
    col_reset1, col_reset2 = st.columns([2, 3])
    with col_reset1:
        st.markdown('<div class="enlarged-clear-btn">', unsafe_allow_html=True)
        if st.button("🔄 Clear Uploaded Dataset & Load Default", key="main_reset_btn"):
            st.session_state["uploader_key"] += 1
            clear_application_cache()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Extract evaluation features and target
X_eval = active_df[ALL_FEATURES]
y_eval = active_df[TARGET_COLUMN]

# -----------------------------------------------------------------------------
# 7. LOGIC: ALL MODELS COMPARATIVE BENCHMARK VIEW
# -----------------------------------------------------------------------------
if selected_model_option == "All Models (Comparative Benchmark)":
    dataset_title_tag = f"Uploaded File: {custom_filename}" if is_custom_dataset else "Default Test Dataset"
    st.subheader(f"📊 Comparative Benchmark: All 5 Classification Models [{dataset_title_tag}]")
    st.markdown("""
    All models are evaluated on the **exact same test dataset** without retraining.
    Explore the tabs below for the **Performance Leaderboard**, **Confusion Matrices**, and **Classification Reports**:
    """)

    # Precompute predictions and metrics for all models
    results_list = []
    roc_data = {}
    cm_dict = {}
    cr_dict = {}

    for name in MODEL_NAMES:
        model = load_saved_model(name)
        preds = model.predict(X_eval)
        proba = model.predict_proba(X_eval)[:, 1]

        metrics = compute_all_metrics(y_eval, preds, proba)
        metrics["Model"] = name
        results_list.append(metrics)

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_eval, proba)
        roc_data[name] = (fpr, tpr, metrics["AUC Score"])

        # Confusion Matrix & Classification Report
        cm_dict[name] = confusion_matrix(y_eval, preds)
        cr_dict[name] = classification_report(y_eval, preds, output_dict=True, zero_division=0)

    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df[["Model", "Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]]

    tab_overview, tab_cm, tab_cr = st.tabs([
        "🏆 Evaluation Metrics Leaderboard",
        "🟦 Confusion Matrices (All 5 Models)",
        "📄 Classification Reports (All 5 Models)"
    ])

    # --- TAB 1: OVERVIEW & COMPARISON ---
    with tab_overview:
        st.markdown("#### 📋 6 Mandatory Evaluation Metrics Table")

        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #d1fae5; font-weight: bold' if v else '' for v in is_max]

        styled_df = comparison_df.style.apply(
            highlight_max,
            subset=["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
        ).format({
            "Accuracy": "{:.4f}",
            "AUC Score": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC Score": "{:.4f}"
        })

        st.dataframe(styled_df, use_container_width=True)

        # Champion KPI Cards
        best_acc = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        best_auc = comparison_df.loc[comparison_df['AUC Score'].idxmax()]
        best_prec = comparison_df.loc[comparison_df['Precision'].idxmax()]
        best_rec = comparison_df.loc[comparison_df['Recall'].idxmax()]
        best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
        best_mcc = comparison_df.loc[comparison_df['MCC Score'].idxmax()]

        col_b1, col_b2, col_b3, col_b4, col_b5, col_b6 = st.columns(6)
        col_b1.metric("Highest Accuracy", f"{best_acc['Accuracy']:.4f}", best_acc['Model'])
        col_b2.metric("Highest AUC Score", f"{best_auc['AUC Score']:.4f}", best_auc['Model'])
        col_b3.metric("Highest Precision", f"{best_prec['Precision']:.4f}", best_prec['Model'])
        col_b4.metric("Highest Recall", f"{best_rec['Recall']:.4f}", best_rec['Model'])
        col_b5.metric("Highest F1 Score", f"{best_f1['F1 Score']:.4f}", best_f1['Model'])
        col_b6.metric("Highest MCC Score", f"{best_mcc['MCC Score']:.4f}", best_mcc['Model'])

        st.markdown("---")
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.markdown("##### 📈 Performance Metrics Comparison Bar Chart")
            fig, ax = plt.subplots(figsize=(8, 5))
            melted_df = pd.melt(
                comparison_df,
                id_vars=["Model"],
                value_vars=["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"],
                var_name="Metric",
                value_name="Score"
            )
            sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette="crest", ax=ax)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score (0 - 1)")
            ax.set_title("Metric Comparison Across Models", fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_plot2:
            st.markdown("##### 🎯 Combined ROC Curves (All 5 Models)")
            fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
            for name, (fpr, tpr, auc_val) in roc_data.items():
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)
            ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label="Random Guess")
            ax_roc.set_xlim([-0.02, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
            ax_roc.set_ylabel("True Positive Rate (Recall)")
            ax_roc.set_title("ROC Curves Comparison", fontsize=12, fontweight='bold')
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_roc.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)

    # --- TAB 2: CONFUSION MATRICES FOR ALL 5 MODELS ---
    with tab_cm:
        st.markdown("#### 🟦 Confusion Matrices Across All 5 Classification Models")
        st.markdown("Side-by-side heatmaps illustrating True Negatives, False Positives, False Negatives, and True Positives:")

        col_cm1, col_cm2, col_cm3 = st.columns(3)
        cols_row1 = [col_cm1, col_cm2, col_cm3]
        for idx in range(3):
            name = MODEL_NAMES[idx]
            cm = cm_dict[name]
            with cols_row1[idx]:
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    annot_kws={"size": 11, "weight": "bold"},
                    ax=ax_cm
                )
                ax_cm.set_title(name, fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
                tn, fp, fn, tp = cm.ravel()
                st.caption(f"**TN:** {tn} | **FP:** {fp} | **FN:** {fn} | **TP:** {tp}")

        st.markdown("---")
        col_cm4, col_cm5, col_cm_summary = st.columns(3)
        for idx, col in [(3, col_cm4), (4, col_cm5)]:
            name = MODEL_NAMES[idx]
            cm = cm_dict[name]
            with col:
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    annot_kws={"size": 11, "weight": "bold"},
                    ax=ax_cm
                )
                ax_cm.set_title(name, fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
                tn, fp, fn, tp = cm.ravel()
                st.caption(f"**TN:** {tn} | **FP:** {fp} | **FN:** {fn} | **TP:** {tp}")

        with col_cm_summary:
            st.info("""
            **Confusion Matrix Interpretation:**
            * **TN (True Negative):** Correctly predicted non-subscribers.
            * **FP (False Positive):** Type I error (incorrectly predicted deposit).
            * **FN (False Negative):** Type II error (missed actual deposit subscribers).
            * **TP (True Positive):** Correctly predicted deposit subscribers.
            """)

    # --- TAB 3: CLASSIFICATION REPORTS FOR ALL 5 MODELS ---
    with tab_cr:
        st.markdown("#### 📄 Classification Reports (All 5 Models)")
        st.markdown("Detailed class-wise breakdown (Class 0: No Deposit, Class 1: Subscribed Deposit):")

        cr_tabs = st.tabs(MODEL_NAMES)
        for idx, name in enumerate(MODEL_NAMES):
            with cr_tabs[idx]:
                st.markdown(f"##### Classification Report: **{name}**")
                report_df = pd.DataFrame(cr_dict[name]).transpose()
                styled_report = report_df.style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                })
                st.dataframe(styled_report, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🎯 Positive Class (Class 1: Deposit = Yes) Performance Comparison")
        class1_data = []
        for name in MODEL_NAMES:
            c1_stats = cr_dict[name].get('1', {})
            class1_data.append({
                "Model": name,
                "Precision (Class 1)": c1_stats.get('precision', 0),
                "Recall (Class 1)": c1_stats.get('recall', 0),
                "F1-Score (Class 1)": c1_stats.get('f1-score', 0),
                "Support": c1_stats.get('support', 0)
            })
        c1_df = pd.DataFrame(class1_data)
        st.dataframe(c1_df.style.format({
            "Precision (Class 1)": "{:.4f}",
            "Recall (Class 1)": "{:.4f}",
            "F1-Score (Class 1)": "{:.4f}",
            "Support": "{:.0f}"
        }), use_container_width=True)

# -----------------------------------------------------------------------------
# 8. LOGIC: INDIVIDUAL MODEL INSPECTION VIEW
# -----------------------------------------------------------------------------
else:
    model_name = selected_model_option
    dataset_title_tag = f"Uploaded File: {custom_filename}" if is_custom_dataset else "Default Test Dataset"
    st.subheader(f"🔍 Detailed Evaluation: {model_name} [{dataset_title_tag}]")

    model = load_saved_model(model_name)
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = compute_all_metrics(y_eval, y_pred, y_proba)

    st.markdown("#### 📌 Mandatory Evaluation Metrics Scorecard")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    m_col2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
    m_col3.metric("Precision", f"{metrics['Precision']:.4f}")
    m_col4.metric("Recall", f"{metrics['Recall']:.4f}")
    m_col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    m_col6.metric("MCC Score", f"{metrics['MCC Score']:.4f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Confusion Matrix & Classification Report",
        "📉 ROC & Precision-Recall Curves",
        "🔎 Sample Predictions Inspection",
        "📑 Test Dataset Preview"
    ])

    # Tab 1: Confusion Matrix & Classification Report
    with tab1:
        c_col1, c_col2 = st.columns(2)

        with c_col1:
            st.markdown("##### 🟦 Confusion Matrix")
            cm = confusion_matrix(y_eval, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Predicted No (0)", "Predicted Yes (1)"],
                yticklabels=["Actual No (0)", "Actual Yes (1)"],
                ax=ax_cm,
                annot_kws={"size": 13, "weight": "bold"}
            )
            ax_cm.set_ylabel("True Label", fontweight='bold')
            ax_cm.set_xlabel("Predicted Label", fontweight='bold')
            ax_cm.set_title(f"Confusion Matrix: {model_name}", fontweight='bold', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            tn, fp, fn, tp = cm.ravel()
            st.caption(f"**TN:** {tn} | **FP:** {fp} | **FN:** {fn} | **TP:** {tp}")

        with c_col2:
            st.markdown("##### 📄 Classification Report")
            report_dict = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()

            styled_report = report_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}"
            })
            st.dataframe(styled_report, use_container_width=True)

            st.info("""
            * **Precision:** Accuracy of positive predictions ($TP / (TP + FP)$).
            * **Recall:** Coverage of actual positives ($TP / (TP + FN)$).
            * **F1-Score:** Harmonic mean of precision and recall.
            * **MCC:** Matthews Correlation Coefficient balanced quality metric ($[-1, +1]$).
            """)

    # Tab 2: ROC & PR Curves
    with tab2:
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown("##### 🎯 Receiver Operating Characteristic (ROC) Curve")
            fpr, tpr, _ = roc_curve(y_eval, y_proba)
            fig_r, ax_r = plt.subplots(figsize=(6, 4.5))
            ax_r.plot(fpr, tpr, color='#1e3a8a', lw=2.5, label=f"ROC (AUC = {metrics['AUC Score']:.4f})")
            ax_r.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label="Random Chance")
            ax_r.set_xlim([-0.02, 1.0])
            ax_r.set_ylim([0.0, 1.05])
            ax_r.set_xlabel("False Positive Rate", fontweight='bold')
            ax_r.set_ylabel("True Positive Rate", fontweight='bold')
            ax_r.set_title(f"ROC Curve - {model_name}", fontweight='bold', fontsize=12)
            ax_r.legend(loc="lower right")
            ax_r.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)

        with r_col2:
            st.markdown("##### ⚖️ Precision-Recall Curve")
            prec, rec, _ = precision_recall_curve(y_eval, y_proba)
            fig_pr, ax_pr = plt.subplots(figsize=(6, 4.5))
            pr_auc = auc(rec, prec)
            ax_pr.plot(rec, prec, color='#059669', lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
            ax_pr.set_xlim([0.0, 1.05])
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.set_xlabel("Recall", fontweight='bold')
            ax_pr.set_ylabel("Precision", fontweight='bold')
            ax_pr.set_title(f"Precision-Recall Curve - {model_name}", fontweight='bold', fontsize=12)
            ax_pr.legend(loc="lower left")
            ax_pr.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_pr)
            plt.close(fig_pr)

    # Tab 3: Predictions Inspection
    with tab3:
        st.markdown("##### 🔎 Sample Test Instances with Predictions and Probabilities")
        pred_df = active_df.copy()
        pred_df["Predicted_Deposit"] = y_pred
        pred_df["Prediction_Probability"] = np.round(y_proba, 4)
        pred_df["Prediction_Match"] = np.where(pred_df[TARGET_COLUMN] == pred_df["Predicted_Deposit"], "✅ Correct", "❌ Misclassified")

        cols_to_show = ["age", "job", "balance", "duration", "campaign", TARGET_COLUMN, "Predicted_Deposit", "Prediction_Probability", "Prediction_Match"]
        st.dataframe(pred_df[cols_to_show].head(50), use_container_width=True)

    # Tab 4: Raw Test Data Preview
    with tab4:
        st.markdown(f"##### 📑 Active Test Dataset Preview ({len(active_df)} rows × {active_df.shape[1]} columns)")
        st.dataframe(active_df, use_container_width=True)

# -----------------------------------------------------------------------------
# 9. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; font-size: 13px;">
    <b>BITS Pilani - Work Integrated Learning Programmes Division</b> | M.Tech (AIML) Machine Learning Assignment - 2<br>
    <b>Student ID:</b> {STUDENT_ID} &nbsp;|&nbsp; Models: Logistic Regression | Decision Tree | KNN | Naive Bayes | SVM
</div>
""", unsafe_allow_html=True)
