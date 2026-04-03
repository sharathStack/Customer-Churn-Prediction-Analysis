"""
dashboard.py — EDA charts, feature importance, ROC curves, confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import config

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Customer Churn — Exploratory Data Analysis",
                 fontsize=14, fontweight="bold")

    # Churn rate by contract type
    churn_by = df.groupby("contract_type")[config.TARGET].mean().reset_index()
    axes[0].bar(churn_by["contract_type"], churn_by[config.TARGET] * 100,
                color=COLORS[:3])
    axes[0].set_title("Churn Rate by Contract Type (%)")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].tick_params(axis="x", rotation=12)
    for i, v in enumerate(churn_by[config.TARGET] * 100):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    # Monthly charges distribution
    axes[1].hist(df[df[config.TARGET] == 1]["monthly_charges"],
                 bins=30, alpha=0.70, color=COLORS[0], label="Churned")
    axes[1].hist(df[df[config.TARGET] == 0]["monthly_charges"],
                 bins=30, alpha=0.70, color=COLORS[1], label="Retained")
    axes[1].set_title("Monthly Charges Distribution")
    axes[1].set_xlabel("Monthly Charges ($)")
    axes[1].legend()

    # Tenure boxplot
    df.boxplot(column="tenure", by=config.TARGET, ax=axes[2],
               patch_artist=True,
               boxprops=dict(facecolor="#3498db", alpha=0.7))
    axes[2].set_title("Tenure by Churn Status")
    axes[2].set_xlabel("Churn (0 = Retained, 1 = Churned)")
    axes[2].set_ylabel("Tenure (months)")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(config.CHART_EDA, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"EDA saved → {config.CHART_EDA}")


def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names)\
            .sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(imp)), imp.values, color=COLORS[1], alpha=0.85)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels(imp.index, rotation=35, ha="right", fontsize=9)
    ax.set_title(f"Top-10 Feature Importances — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score")
    for bar, val in zip(bars, imp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(config.CHART_IMPORTANCE, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"Feature importance saved → {config.CHART_IMPORTANCE}")


def plot_roc(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, res) in enumerate(results.items()):
        ax.plot(res["fpr"], res["tpr"],
                label=f"{name}  (AUC={res['test_auc']:.3f})",
                color=COLORS[i], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Churn Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.CHART_ROC, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved → {config.CHART_ROC}")


def plot_confusion(cm: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(config.CHART_CONFUSION, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {config.CHART_CONFUSION}")
