"""
Page C — Model Performance

Compares Logistic Regression vs ANN (1 hidden layer) on the 2025 test set.
Shows metrics table, ROC curves, confusion matrices, threshold sweep, and
LR feature-importance ranking derived from standardized coefficients.
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helper_functions import get_predictor, header

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

header("📈 Page C — Model Performance")

st.markdown(
    """
This page evaluates both trained models — **Logistic Regression** (with L2 regularization)
and **ANN (1 hidden layer, 16→128→1)** — on the held-out **2025 test set**
(27,040 laps). All metrics are computed live from the saved artifacts.
"""
)

# ── Load artifacts and test set ──
predictor = get_predictor()
DATASET_DIR = os.path.join(_ROOT, "Dataset")


@st.cache_data
def load_test_set():
    df = pd.read_csv(os.path.join(DATASET_DIR, "f1_test_processed.csv"))
    y = df["PitNextLap"].values.astype(int)
    feature_cols = [c for c in df.columns if c not in ["PitNextLap", "Year"]]
    X = df[feature_cols].values.astype(np.float64)
    return X, y, feature_cols


@st.cache_data
def compute_probs():
    """Compute LR and ANN probabilities on the 2025 test set."""
    X, y, feat_cols = load_test_set()

    # Load raw weights
    data = np.load(os.path.join(DATASET_DIR, "model_weights.npz"))

    # LR forward
    lr_W = data["lr_W"]
    lr_b = data["lr_b"][0]
    z = X @ lr_W + lr_b
    lr_proba = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    lr_proba = lr_proba.flatten()

    # ANN forward (1 hidden layer, ReLU + Sigmoid)
    W0, b0 = data["ann1_W0"], data["ann1_b0"]
    W1, b1 = data["ann1_W1"], data["ann1_b1"]
    H = np.maximum(0, X @ W0 + b0)
    z2 = H @ W1 + b1
    ann_proba = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))
    ann_proba = ann_proba.flatten()

    return y, lr_proba, ann_proba, lr_W.flatten(), feat_cols


def metrics(y_true, y_pred, y_proba):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    # ROC-AUC via rank-sum (Mann-Whitney U, no sklearn)
    pos = y_proba[y_true == 1]
    neg = y_proba[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        # Efficient AUC
        order = np.argsort(y_proba)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y_proba) + 1)
        sum_ranks_pos = ranks[y_true == 1].sum()
        auc = (sum_ranks_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "auc": auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


y_true, lr_proba, ann_proba, lr_W, feat_cols = compute_probs()

# Load tuned thresholds
with open(os.path.join(DATASET_DIR, "model_config.json")) as f:
    config = json.load(f)
lr_threshold = config["all_thresholds"]["lr"]
ann_threshold = config["all_thresholds"]["ann_1layer"]

lr_pred = (lr_proba >= lr_threshold).astype(int)
ann_pred = (ann_proba >= ann_threshold).astype(int)

lr_m = metrics(y_true, lr_pred, lr_proba)
ann_m = metrics(y_true, ann_pred, ann_proba)

# ── Metrics table ──
st.markdown("## 🏆 Metrics Comparison (2025 Test Set)")

table = pd.DataFrame({
    "Metric": ["F1-score", "Precision", "Recall", "Accuracy", "ROC-AUC", "Threshold"],
    "Logistic Regression": [
        f"{lr_m['f1']:.4f}", f"{lr_m['precision']:.4f}", f"{lr_m['recall']:.4f}",
        f"{lr_m['accuracy']:.4f}", f"{lr_m['auc']:.4f}", f"{lr_threshold:.4f}",
    ],
    "ANN 1-Layer (16→128→1)": [
        f"{ann_m['f1']:.4f}", f"{ann_m['precision']:.4f}", f"{ann_m['recall']:.4f}",
        f"{ann_m['accuracy']:.4f}", f"{ann_m['auc']:.4f}", f"{ann_threshold:.4f}",
    ],
})

WIN  = "background-color:#d5f5e3; font-weight:bold;"
NONE = ""

def highlight_winner(row):
    """Return per-cell style strings for one table row."""
    styles = [NONE, NONE, NONE]          # [Metric, LR, ANN]
    if row["Metric"] == "Threshold":     # lower ≠ better; skip
        return styles
    try:
        lr_val  = float(row["Logistic Regression"])
        ann_val = float(row["ANN 1-Layer (16→128→1)"])
        if ann_val > lr_val:
            styles[2] = WIN
        elif lr_val > ann_val:
            styles[1] = WIN
    except ValueError:
        pass
    return styles


st.dataframe(
    table.style.apply(highlight_winner, axis=1),
    hide_index=True, use_container_width=True,
)

st.caption(
    f"🟢 Highlighted cell = winning model on that metric. "
    f"**ANN is the deployed model** because it wins F1 by "
    f"{(ann_m['f1'] - lr_m['f1']):+.4f}."
)

st.markdown("---")

# ── ROC Curves ──
st.markdown("## 📉 ROC Curves")


def roc_curve_points(y, proba, n_points=200):
    thresholds = np.linspace(0, 1, n_points)
    tprs, fprs = [], []
    P = (y == 1).sum()
    N = (y == 0).sum()
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tprs.append(tp / P if P > 0 else 0)
        fprs.append(fp / N if N > 0 else 0)
    return fprs, tprs


col_roc, col_conf = st.columns([1, 1])

with col_roc:
    lr_fpr, lr_tpr = roc_curve_points(y_true, lr_proba)
    ann_fpr, ann_tpr = roc_curve_points(y_true, ann_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lr_fpr, y=lr_tpr, mode="lines",
        name=f"LR (AUC = {lr_m['auc']:.4f})",
        line=dict(color="#3498db", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=ann_fpr, y=ann_tpr, mode="lines",
        name=f"ANN (AUC = {ann_m['auc']:.4f})",
        line=dict(color="#e67e22", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random baseline",
        line=dict(color="gray", dash="dash", width=1.5),
    ))
    fig.update_layout(
        title="ROC Curves — LR vs ANN",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        legend=dict(x=0.55, y=0.1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Confusion matrices ──
with col_conf:
    st.markdown("#### Confusion Matrices (at tuned thresholds)")

    def conf_heatmap(m, title, colorscale):
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        text = [
            [f"TN<br>{m['tn']:,}", f"FP<br>{m['fp']:,}"],
            [f"FN<br>{m['fn']:,}", f"TP<br>{m['tp']:,}"],
        ]
        fig = go.Figure(go.Heatmap(
            z=z, text=text, texttemplate="%{text}",
            x=["Pred NO PIT", "Pred PIT"],
            y=["Actual NO PIT", "Actual PIT"],
            colorscale=colorscale, showscale=False,
        ))
        fig.update_layout(title=title, height=220, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    st.plotly_chart(
        conf_heatmap(lr_m, "Logistic Regression", "Blues"),
        use_container_width=True,
    )
    st.plotly_chart(
        conf_heatmap(ann_m, "ANN 1-Layer", "Oranges"),
        use_container_width=True,
    )

st.markdown("---")

# ── Threshold sweep (F1 vs threshold) ──
st.markdown("## 🎚️ Threshold Sweep — F1 vs Decision Threshold")

st.markdown(
    "Shows how **F1**, **Precision**, and **Recall** change as we slide the decision "
    "threshold. The tuned threshold is where F1 is maximized on the validation set."
)

model_choice = st.radio("Which model?", ["ANN (deployed)", "Logistic Regression"], horizontal=True)
proba_for_sweep = ann_proba if model_choice.startswith("ANN") else lr_proba
tuned_t = ann_threshold if model_choice.startswith("ANN") else lr_threshold

thresholds = np.linspace(0.05, 0.95, 91)
prec_list, rec_list, f1_list = [], [], []
P = (y_true == 1).sum()
for t in thresholds:
    pred = (proba_for_sweep >= t).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = P - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=f1_list,   mode="lines", name="F1",        line=dict(color="#e74c3c", width=3)))
fig.add_trace(go.Scatter(x=thresholds, y=prec_list, mode="lines", name="Precision", line=dict(color="#3498db", width=2)))
fig.add_trace(go.Scatter(x=thresholds, y=rec_list,  mode="lines", name="Recall",    line=dict(color="#27ae60", width=2)))
fig.add_vline(
    x=tuned_t, line_dash="dash", line_color="black",
    annotation_text=f"Tuned t = {tuned_t:.4f}",
    annotation_position="top right",
)
fig.update_layout(
    title=f"{model_choice}: Precision / Recall / F1 vs Threshold",
    xaxis_title="Decision Threshold",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1]),
    height=450,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── LR Feature importance ──
st.markdown("## 🧮 Feature Importance (LR Standardized Coefficients)")

st.markdown(
    "Since all numeric features are z-scored, the **magnitude of the LR coefficient** "
    "directly reflects how strongly that feature influences the pit decision. "
    "Positive = pushes toward PIT, negative = pushes toward STAY OUT."
)

imp_df = pd.DataFrame({
    "Feature": feat_cols,
    "Coefficient": lr_W,
    "AbsCoef": np.abs(lr_W),
}).sort_values("AbsCoef", ascending=True)

colors = ["#e74c3c" if c > 0 else "#3498db" for c in imp_df["Coefficient"]]

fig = go.Figure(go.Bar(
    x=imp_df["Coefficient"], y=imp_df["Feature"],
    orientation="h", marker_color=colors,
    text=[f"{c:+.3f}" for c in imp_df["Coefficient"]],
    textposition="outside",
))
fig.update_layout(
    title="LR Coefficients (sorted by magnitude)",
    xaxis_title="Standardized Coefficient",
    yaxis_title="Feature",
    height=550,
    margin=dict(l=10, r=60, t=50, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "🔴 Red = pushes toward PIT (positive coefficient). "
    "🔵 Blue = pushes toward STAY OUT (negative coefficient)."
)

st.markdown("---")

# ── Final summary ──
with st.expander("📄 Summary & takeaways"):
    st.markdown(
        f"""
### Why ANN was chosen as the deployed model
- ANN Test F1 = **{ann_m['f1']:.4f}** vs LR Test F1 = **{lr_m['f1']:.4f}** → a gain of
  **{(ann_m['f1'] - lr_m['f1']):+.4f}**.
- The non-linear hidden layer lets ANN capture interactions (e.g. *high tyre life
  AND late race progress AND slowing lap times*) that a linear model can't.
- Both models use the same preprocessing and the same class-rate-aware threshold
  tuning protocol (selected on 2024 validation, evaluated on 2025 hold-out).

### About the threshold
- Default `0.5` would over-predict "NO PIT" because pit stops are the minority class (~33%).
- The tuned threshold **{ann_threshold:.4f}** was chosen to maximize F1 on the
  **2024 validation set** — not on the test set — which keeps the evaluation honest
  (no threshold leakage).

### LR feature importance
- Top drivers are typically `TyreLife`, `Cumulative_Degradation`, and `RaceProgress` —
  all intuitive: older tyres on a later lap → more likely to pit.
- `Position` and `Position_Change` are smaller — tactical signals matter less than raw
  tyre condition.
        """
    )
