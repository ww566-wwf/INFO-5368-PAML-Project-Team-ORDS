"""
Page B — Sensitivity Analysis

Varies one feature across its range while holding all other features fixed,
showing how the pit-stop probability changes. Helps interpret which features
drive model decisions.
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from helper_functions import (
    get_predictor, header,
    FEATURE_RANGES, COMPOUND_OPTIONS, DRIVER_OPTIONS,
)

st.set_page_config(page_title="Sensitivity Analysis", page_icon="🔍", layout="wide")

header("🔍 Page B — Sensitivity Analysis")

st.markdown(
    """
Select a **feature to sweep**. The app will vary that feature across its realistic
range while holding all other features fixed at your chosen baseline values, and
plot how the **pit probability** changes. This reveals which features the model
considers most influential.
"""
)

predictor = get_predictor()
threshold = predictor.threshold

# ── Sidebar: baseline feature values ──
st.sidebar.markdown("## 🎛️ Baseline Race Conditions")
st.sidebar.caption("These values are held fixed while the chosen feature is swept.")

driver = st.sidebar.selectbox("Driver", options=DRIVER_OPTIONS, index=0)
compound = st.sidebar.selectbox("Tyre Compound", options=COMPOUND_OPTIONS, index=0)

baseline = {
    "Driver": driver,
    "Compound": compound,
    "LapNumber":              st.sidebar.slider("Lap Number",      FEATURE_RANGES["LapNumber"]["min"],              FEATURE_RANGES["LapNumber"]["max"],              FEATURE_RANGES["LapNumber"]["default"],              FEATURE_RANGES["LapNumber"]["step"]),
    "Stint":                  st.sidebar.slider("Stint",           FEATURE_RANGES["Stint"]["min"],                  FEATURE_RANGES["Stint"]["max"],                  FEATURE_RANGES["Stint"]["default"],                  FEATURE_RANGES["Stint"]["step"]),
    "TyreLife":               st.sidebar.slider("Tyre Life",       FEATURE_RANGES["TyreLife"]["min"],               FEATURE_RANGES["TyreLife"]["max"],               FEATURE_RANGES["TyreLife"]["default"],               FEATURE_RANGES["TyreLife"]["step"]),
    "Position":               st.sidebar.slider("Position",        FEATURE_RANGES["Position"]["min"],               FEATURE_RANGES["Position"]["max"],               FEATURE_RANGES["Position"]["default"],               FEATURE_RANGES["Position"]["step"]),
    "RaceProgress":           st.sidebar.slider("Race Progress",   FEATURE_RANGES["RaceProgress"]["min"],           FEATURE_RANGES["RaceProgress"]["max"],           FEATURE_RANGES["RaceProgress"]["default"],           FEATURE_RANGES["RaceProgress"]["step"]),
    "LapTime (s)":            st.sidebar.slider("Lap Time (s)",    FEATURE_RANGES["LapTime (s)"]["min"],            FEATURE_RANGES["LapTime (s)"]["max"],            FEATURE_RANGES["LapTime (s)"]["default"],            FEATURE_RANGES["LapTime (s)"]["step"]),
    "LapTime_Delta":          st.sidebar.slider("Lap Time Delta",  FEATURE_RANGES["LapTime_Delta"]["min"],          FEATURE_RANGES["LapTime_Delta"]["max"],          FEATURE_RANGES["LapTime_Delta"]["default"],          FEATURE_RANGES["LapTime_Delta"]["step"]),
    "Cumulative_Degradation": st.sidebar.slider("Cum Degradation", FEATURE_RANGES["Cumulative_Degradation"]["min"], FEATURE_RANGES["Cumulative_Degradation"]["max"], FEATURE_RANGES["Cumulative_Degradation"]["default"], FEATURE_RANGES["Cumulative_Degradation"]["step"]),
    "Position_Change":        st.sidebar.slider("Position Change", FEATURE_RANGES["Position_Change"]["min"],        FEATURE_RANGES["Position_Change"]["max"],        FEATURE_RANGES["Position_Change"]["default"],        FEATURE_RANGES["Position_Change"]["step"]),
    "Degradation_Rate":       st.sidebar.slider("Degradation Rate",FEATURE_RANGES["Degradation_Rate"]["min"],       FEATURE_RANGES["Degradation_Rate"]["max"],       FEATURE_RANGES["Degradation_Rate"]["default"],       FEATURE_RANGES["Degradation_Rate"]["step"]),
}

# ── Feature to sweep ──
st.markdown("### Pick a feature to sweep")
sweep_options = list(FEATURE_RANGES.keys()) + ["Compound"]
sweep_feature = st.selectbox("Feature", options=sweep_options, index=sweep_options.index("TyreLife"))

n_points = st.slider("Resolution (points across range)", 10, 200, 80, 10)

# ── Run sweep ──
probs = []
sweep_values = []

if sweep_feature == "Compound":
    sweep_values = COMPOUND_OPTIONS
    for c in sweep_values:
        inp = dict(baseline); inp["Compound"] = c
        result = predictor.predict_from_raw(inp)
        probs.append(result["probability"])
else:
    lo = FEATURE_RANGES[sweep_feature]["min"]
    hi = FEATURE_RANGES[sweep_feature]["max"]
    sweep_values = np.linspace(lo, hi, n_points).tolist()
    for v in sweep_values:
        inp = dict(baseline)
        # Cast integer-stepped features back to int for display consistency
        if FEATURE_RANGES[sweep_feature]["step"] >= 1 and isinstance(FEATURE_RANGES[sweep_feature]["default"], int):
            v_cast = int(round(v))
        else:
            v_cast = float(v)
        inp[sweep_feature] = v_cast
        result = predictor.predict_from_raw(inp)
        probs.append(result["probability"])

# ── Plot sensitivity curve ──
col_plot, col_stats = st.columns([2, 1])

with col_plot:
    fig = go.Figure()

    if sweep_feature == "Compound":
        colors = ["#e74c3c" if p >= threshold else "#27ae60" for p in probs]
        fig.add_trace(go.Bar(
            x=sweep_values, y=probs, marker_color=colors,
            text=[f"{p:.3f}" for p in probs], textposition="outside",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=sweep_values, y=probs, mode="lines+markers",
            line=dict(color="#3498db", width=3),
            marker=dict(size=5),
            name="Pit Probability",
        ))
        # Fill above threshold in red
        fig.add_hrect(
            y0=threshold, y1=1.0,
            fillcolor="#fadbd8", opacity=0.3, line_width=0,
            annotation_text=" PIT region", annotation_position="top left",
        )

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="black",
        annotation_text=f"Threshold = {threshold:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Pit Probability vs {sweep_feature}",
        xaxis_title=sweep_feature,
        yaxis_title="P(Pit on next lap)",
        yaxis=dict(range=[0, 1]),
        height=480,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.markdown("### 📊 Sweep Stats")
    st.metric("Min probability", f"{min(probs):.4f}")
    st.metric("Max probability", f"{max(probs):.4f}")
    st.metric("Range (Δ)",       f"{max(probs) - min(probs):.4f}")
    st.metric("Mean",            f"{sum(probs)/len(probs):.4f}")

    n_pit_region = sum(1 for p in probs if p >= threshold)
    pit_pct = n_pit_region / len(probs) * 100
    st.metric("% of sweep predicting PIT", f"{pit_pct:.1f}%")

    st.caption(
        f"A larger **Range (Δ)** means the model is more sensitive to this feature."
    )

st.markdown("---")

with st.expander("ℹ️ How to interpret this chart"):
    st.markdown(
        f"""
- The **blue curve** shows how the predicted pit probability changes as `{sweep_feature}`
  is varied, while all other features stay at their baseline values.
- The **dashed black line** marks the decision threshold ({threshold:.3f}). When the
  probability is above it, the model predicts a pit stop.
- The **shaded red region** highlights where the model predicts PIT.
- Try sweeping `TyreLife` or `Cumulative_Degradation` — you should see the probability
  rise as tires wear out. Compare that with `Position` (usually much flatter).
        """
    )
