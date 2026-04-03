"""
config.py — Customer Churn Prediction
All parameters in one place.
"""

# ── Data ───────────────────────────────────────────────────────────────────────
N_SAMPLES    = 5_000
RANDOM_SEED  = 42
TEST_SIZE    = 0.20

# ── Features ───────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "contract_type", "payment_method", "internet_service",
    "tech_support", "partner", "dependents",
]
NUMERICAL_COLS = [
    "tenure", "monthly_charges", "total_charges",
    "num_products", "senior_citizen",
]
TARGET = "churn"

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression":  {"max_iter": 500, "random_state": RANDOM_SEED},
    "Random Forest":        {"n_estimators": 200, "random_state": RANDOM_SEED, "n_jobs": -1},
    "Gradient Boosting":    {"n_estimators": 200, "random_state": RANDOM_SEED, "learning_rate": 0.05},
}
CV_FOLDS    = 5
CV_SCORING  = "roc_auc"

# ── Output ──────────────────────────────────────────────────────────────────────
CHART_EDA         = "churn_eda.png"
CHART_IMPORTANCE  = "churn_feature_importance.png"
CHART_ROC         = "churn_roc_curves.png"
CHART_CONFUSION   = "churn_confusion_matrix.png"
CHART_DPI         = 150
