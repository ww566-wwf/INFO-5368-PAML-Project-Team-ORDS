"""
Page A — Predict Pit Stop

Core interactive page: user enters race conditions through sliders and dropdowns,
and the app returns a real-time pit-stop prediction using the trained ANN model.
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import plotly.graph_objects as go

from helper_functions import (
    get_predictor, header,
    FEATURE_RANGES, COMPOUND_OPTIONS, DRIVER_OPTIONS,
)

st.set_page_config(page_title="Predict Pit Stop", page_icon="🎯", layout="wide")

header("🎯 Page A — Predict Pit Stop")

st.markdown(
    """
Enter race conditions on the **left sidebar** to get a real-time prediction for whether
the driver will pit on the next lap. The prediction uses the **ANN model** trained on
2022+2024 data with a tuned decision threshold (t = 0.2418).
"""
)

# ── Load the predictor once (cached) ──
predictor = get_predictor()

# ── Sidebar: user inputs ──
st.sidebar.markdown("## 🎛️ Race Conditions")

st.sidebar.markdown("### Driver & Tyre")
driver = st.sidebar.selectbox("Driver", options=DRIVER_OPTIONS, index=0, help="Three-letter driver code")
compound = st.sidebar.selectbox("Tyre Compound", options=COMPOUND_OPTIONS, index=0, help="Current tyre compound")

def _slider(label, feat_key):
    """Streamlit slider driven by FEATURE_RANGES metadata."""
    r = FEATURE_RANGES[feat_key]
    return st.sidebar.slider(
        label,
        min_value=r["min"], max_value=r["max"],
        value=r["default"], step=r["step"],
        help=r.get("help"),
    )


st.sidebar.markdown("### Race State")
lap_number    = _slider("Lap Number",    "LapNumber")
stint         = _slider("Stint",         "Stint")
tyre_life     = _slider("Tyre Life (laps)", "TyreLife")
position      = _slider("Position",      "Position")
race_progress = _slider("Race Progress", "RaceProgress")

st.sidebar.markdown("### Pace & Degradation")
lap_time   = _slider("Lap Time (s)",           "LapTime (s)")
lap_delta  = _slider("Lap Time Delta (s)",     "LapTime_Delta")
cum_deg    = _slider("Cumulative Degradation", "Cumulative_Degradation")
pos_change = _slider("Position Change",        "Position_Change")
deg_rate   = _slider("Degradation Rate",       "Degradation_Rate")

# ── Assemble raw input ──
raw_input = {
    "Driver": driver,
    "Compound": compound,
    "LapNumber": lap_number,
    "Stint": stint,
    "TyreLife": tyre_life,
    "Position": position,
    "LapTime (s)": lap_time,
    "LapTime_Delta": lap_delta,
    "Cumulative_Degradation": cum_deg,
    "RaceProgress": race_progress,
    "Position_Change": pos_change,
    "Degradation_Rate": deg_rate,
}

# ── Run prediction ──
result = predictor.predict_from_raw(raw_input)
proba = result["probability"]
label = result["label"]
threshold = result["threshold"]

# ── Layout: Prediction output (top) + Feature summary (bottom) ──
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🎯 Prediction Result")

    # Color-coded prediction card
    if label == "PIT":
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6; padding:22px; border-radius:10px; border-left:6px solid #e74c3c;">
                <h2 style="color:#c0392b; margin:0;">🛑 PIT</h2>
                <p style="color:#333; font-size:16px; margin-top:8px;">
                    The driver is predicted to <b>pit on the next lap</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#e6f7e6; padding:22px; border-radius:10px; border-left:6px solid #27ae60;">
                <h2 style="color:#1e8449; margin:0;">✅ NO PIT</h2>
                <p style="color:#333; font-size:16px; margin-top:8px;">
                    The driver is predicted to <b>stay out</b> next lap.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        delta={"reference": threshold * 100, "relative": False, "suffix": "%"},
        title={"text": "Pit Probability"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#e74c3c" if proba >= threshold else "#27ae60"},
            "steps": [
                {"range": [0, threshold * 100], "color": "#eafaf1"},
                {"range": [threshold * 100, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Decision threshold = **{threshold:.4f}** (tuned on 2024 validation set)")

with col_right:
    st.markdown("### 📋 Input Summary")

    summary_data = [
        ("Driver", driver),
        ("Compound", compound),
        ("Lap Number", lap_number),
        ("Stint", stint),
        ("Tyre Life", f"{tyre_life} laps"),
        ("Position", f"P{position}"),
        ("Race Progress", f"{race_progress:.0%}"),
        ("Lap Time", f"{lap_time:.2f} s"),
        ("Lap Time Δ", f"{lap_delta:+.2f} s"),
        ("Cum. Degradation", f"{cum_deg:.1f}"),
        ("Position Change", f"{pos_change:+d}"),
        ("Degradation Rate", f"{deg_rate:+.3f}"),
    ]

    import pandas as pd
    summary_df = pd.DataFrame(summary_data, columns=["Feature", "Value"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True, height=460)

st.markdown("---")

# ── Model info ──
with st.expander("ℹ️ About this prediction"):
    st.markdown(
        f"""
**Model architecture:** ANN (1 hidden layer, 128 neurons, ReLU + Sigmoid)
**Training data:** 2022 + 2024 seasons (49,351 laps)
**Tuned decision threshold:** {threshold:.4f} (maximizes F1 on the 2024 validation set)
**Test-set F1 (2025):** 0.6263

The model outputs a **probability** between 0 and 1. When the probability is **above the
threshold** ({threshold:.4f}), the model predicts a pit stop on the next lap.
A lower threshold (vs the default 0.5) was chosen because it maximizes F1 — balancing
precision and recall — and helps identify more real pit stops at the cost of some
false alarms.
        """
    )
