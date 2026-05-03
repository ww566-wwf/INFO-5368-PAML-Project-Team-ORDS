# F1 Pit Stop Strategy Predictor

**Course:** INFO 5368-030 — Practical Applications of Machine Learning, Cornell Tech  
**Team ORDS:** He Sun · Yilin Wang · Weifan Wu · Peter Ye

---

## 🌐 Live Application

**[https://ww566-wwf-info-5368-paml-project-team-ords-app-wpjq3c.streamlit.app](https://ww566-wwf-info-5368-paml-project-team-ords-app-wpjq3c.streamlit.app)**

---

## 📖 Project Overview

This project builds a machine learning system that predicts whether a Formula 1 driver will **pit on the next lap**, using structured lap-level race data. Pit stop timing is one of the most strategically critical decisions in F1 — balancing tyre degradation, track position, and under/overcut opportunities.

Two models are implemented **from scratch in NumPy** (no ML libraries):
- **Logistic Regression** with L2 regularization (λ tuning via time-series cross-validation)
- **ANN** — 1 hidden layer (16 → 128 → 1, ReLU + Sigmoid, He initialization)

**Best model:** ANN with tuned threshold (t = 0.2418) → **Test F1 = 0.6263** on 2025 season hold-out.

---

## 🗂 Repository Structure

```
├── app.py                        # Streamlit homepage
├── pages/
│   ├── A_Predict.py              # Page A — real-time pit stop prediction
│   ├── B_Sensitivity.py          # Page B — sensitivity analysis (feature sweep)
│   └── C_Model_Performance.py   # Page C — model comparison, ROC curves, feature importance
├── helper_functions.py           # Shared utilities (cached predictor, feature ranges)
├── inference.py                  # F1PitPredictor — end-to-end inference class
├── data_preprocessing.ipynb      # Data cleaning, EDA, feature engineering, train/test split
├── model_training.ipynb          # Model training, threshold tuning, L2 regularization, evaluation
├── requirements.txt              # Python dependencies
└── Dataset/
    ├── f1_strategy_dataset_v4.csv    # Raw dataset (101,371 laps, 2022–2025)
    ├── f1_train_processed.csv        # Processed train set (2022 + 2024, 49,351 rows)
    ├── f1_test_processed.csv         # Processed test set (2025, 27,040 rows)
    ├── model_weights.npz             # Saved LR + ANN weights
    ├── model_config.json             # Selected model, tuned threshold, layer sizes
    ├── scaler_params.csv             # Standardization mean/std (train set only)
    ├── feature_order.json            # Locked feature column order for inference
    └── driver_mapping.csv            # Driver label encoding map
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/ww566-wwf/INFO-5368-PAML-Project-Team-ORDS.git
cd INFO-5368-PAML-Project-Team-ORDS
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Launch the Streamlit app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | [Kaggle — F1 Lap-Level Race Data](https://www.kaggle.com/) (built from FastF1 telemetry) |
| Raw size | 101,371 lap observations × 16 columns |
| Seasons | 2022, 2024, 2025 (2023 excluded — anomalous pit labeling) |
| Target | `PitNextLap` (binary; class rate ≈ 33%) |
| Train set | 2022 + 2024 → 49,351 rows |
| Test set | 2025 → 27,040 rows (temporal hold-out) |
| Features | 16 after preprocessing (10 numeric + 5 compound one-hot + 1 driver encoded) |

---

## 🧠 Model Results (2025 Test Set)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|-----------|--------|----|---------|
| LR (tuned, L2 λ=0.01) | 0.2568 | 0.427 | 0.713 | 0.535 | 0.664 |
| LR (tuned, no reg) | 0.2424 | 0.430 | 0.722 | 0.539 | 0.666 |
| ANN 1-Layer (tuned) ⭐ | 0.2418 | 0.485 | 0.883 | **0.626** | 0.794 |

---

## 📋 Requirements

```
streamlit>=1.30.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.18.0
matplotlib>=3.7.0
```
