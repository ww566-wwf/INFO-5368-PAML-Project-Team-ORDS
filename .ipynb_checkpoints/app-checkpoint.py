import streamlit as st

st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="wide",
)

st.markdown("# 🏎️ Practical Applications of Machine Learning (PAML)")

st.markdown("### Final Project — Modeling Pit Stop Strategy in Formula 1")

st.markdown("**Team ORDS:** He Sun · Yilin Wang · Weifan Wu · Peter Ye")
st.markdown("**Course:** INFO 5368-030 — Cornell Tech")

st.markdown("---")

st.markdown("## 📖 Project Overview")

st.markdown(
    """
This project develops a machine learning system that predicts whether a Formula 1 driver
will **pit on the next lap** using structured lap-level race data. Pit stop timing is
one of the most important decisions in F1 race strategy, balancing tire degradation,
track position, and under/overcut opportunities.

### Objectives
- **Build 2 offline ML models** from scratch in NumPy — Logistic Regression & ANN (1 hidden layer)
- **Evaluate** using Precision, Recall, F1-score, and ROC-AUC (F1 is primary metric)
- **Deploy** an interactive Streamlit app for real-time pit-stop prediction
    """
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📊 Dataset")
    st.markdown(
        """
- **Source:** Kaggle — F1 lap-level race data (built from FastF1 telemetry)
- **Size:** 101,371 lap observations
- **Seasons:** 2022, 2024, 2025 (2023 excluded — anomalous labeling)
- **Target:** `PitNextLap` (binary, class rate ≈ 33%)
- **Features:** 16 after preprocessing (10 numeric + compound one-hot + driver encoded)
        """
    )

with col2:
    st.markdown("## 🧠 ML Pipeline")
    st.markdown(
        """
- **Train:** 2022 + 2024 (49,351 rows)
- **Validation:** 2024 fold (for threshold + λ tuning)
- **Test:** 2025 (27,040 rows — temporal hold-out)
- **Best model:** ANN 1-Layer (16→128→1, ReLU+Sigmoid)
- **Tuned threshold:** 0.2418 → **Test F1 = 0.6263**
        """
    )

st.markdown("---")

st.markdown("## 🧭 Navigation")

st.markdown(
    """
Use the sidebar to navigate between the application pages:

- **A — Predict Pit Stop** 🎯
  *Input race conditions through sliders and dropdowns; get an instant pit-stop prediction.*

- **B — Sensitivity Analysis** 🔍
  *Vary one feature while holding others fixed and see how the pit probability changes.*

- **C — Model Performance** 📈
  *View the full comparison of LR vs ANN, confusion matrices, and ROC curves on the 2025 test set.*
    """
)

st.markdown("---")

st.markdown("## 👥 Target Users")
st.markdown(
    """
This tool is designed for **Formula 1 fans, students, and race analysts** who want to
explore how pit stop decisions change under different race conditions. No technical
background is required — all inputs are controlled via intuitive sliders and dropdowns.
    """
)

st.markdown("---")

st.info("👉 Click **A Predict** in the sidebar to get started.")
