"""
Shared helper functions for F1 Pit Stop Streamlit app.

Loads the trained F1PitPredictor into Streamlit session_state once,
so all pages reuse the same artifacts without re-loading.
"""

import streamlit as st
import os
import sys

# Ensure the project root is on the import path so `inference` resolves
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inference import F1PitPredictor  # noqa: E402


@st.cache_resource
def load_predictor(artifact_dir=None):
    """
    Cached loader — builds F1PitPredictor exactly once per Streamlit session.
    The @st.cache_resource decorator keeps the instance in memory across reruns.
    """
    if artifact_dir is None:
        artifact_dir = os.path.join(_ROOT, "Dataset")
    return F1PitPredictor(artifact_dir=artifact_dir)


def get_predictor():
    """Convenience wrapper — call this from any page."""
    predictor = load_predictor()
    # Also expose in session_state for consistency with HW pattern
    st.session_state["predictor"] = predictor
    return predictor


# ── Feature metadata for slider UI ──
# Realistic F1 ranges derived from the training set statistics.
FEATURE_RANGES = {
    "LapNumber":              {"min": 1,     "max": 80,   "default": 30,    "step": 1,    "help": "Current lap number"},
    "Stint":                  {"min": 1,     "max": 6,    "default": 2,     "step": 1,    "help": "Stint number (how many times pitted so far + 1)"},
    "TyreLife":               {"min": 0,     "max": 50,   "default": 15,    "step": 1,    "help": "Number of laps on current tyres"},
    "Position":               {"min": 1,     "max": 20,   "default": 10,    "step": 1,    "help": "Current race position"},
    "LapTime (s)":            {"min": 60.0,  "max": 130.0,"default": 93.0,  "step": 0.1,  "help": "Current lap time in seconds"},
    "LapTime_Delta":          {"min": -5.0,  "max": 5.0,  "default": 0.0,   "step": 0.1,  "help": "Change in lap time vs previous lap (s)"},
    "Cumulative_Degradation": {"min": -200.0,"max": 200.0,"default": -30.0, "step": 1.0,  "help": "Cumulative tyre degradation proxy (lower = more degraded)"},
    "RaceProgress":           {"min": 0.0,   "max": 1.0,  "default": 0.5,   "step": 0.01, "help": "Fraction of race completed (0 = start, 1 = finish)"},
    "Position_Change":        {"min": -15,   "max": 15,   "default": 0,     "step": 1,    "help": "Position change vs previous lap (positive = gained)"},
    "Degradation_Rate":       {"min": -2.0,  "max": 2.0,  "default": 0.0,   "step": 0.01, "help": "Tyre degradation rate per lap"},
}

COMPOUND_OPTIONS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

DRIVER_OPTIONS = [
    "VER", "LEC", "HAM", "NOR", "RUS", "PIA", "SAI", "ALO", "PER", "GAS",
    "OCO", "STR", "TSU", "ALB", "HUL", "MAG", "BOT", "ZHO", "RIC", "LAW",
    "COL", "BEA", "DOO", "ANT", "BOR", "HAD", "LAT", "MSC", "SAR", "DEV", "VET",
]


def header(subtitle):
    """Render consistent header across pages."""
    st.markdown("# 🏎️ Practical Applications of Machine Learning (PAML)")
    st.markdown("### Final Project — Modeling Pit Stop Strategy in Formula 1")
    st.markdown(f"## {subtitle}")
    st.markdown("---")
