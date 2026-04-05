"""
models.py — Train and cross-validate Logistic Regression, Random Forest, Gradient Boosting
"""

import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics        import (classification_report, roc_auc_score,
                                    roc_curve, confusion_matrix)
import config


def _build_model(name: str):
    params = config.MODELS[name]
    if name == "Logistic Regression":
        return LogisticRegression(**params)
    elif name == "Random Forest":
        return RandomForestClassifier(**params)
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def train_all(X_train, X_test, y_train, y_test) -> dict:
    """Train all models, return results dict keyed by model name."""
    results = {}

    for name in config.MODELS:
        print(f"\n{'='*50}")
        print(f"  {name}")
        model = _build_model(name)

        # Cross-validation
        cv_auc = cross_val_score(
            model, X_train, y_train,
            cv=config.CV_FOLDS, scoring=config.CV_SCORING
        ).mean()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"  CV AUC  : {cv_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Retained", "Churned"]))

        results[name] = {
            "model":    model,
            "cv_auc":   cv_auc,
            "test_auc": test_auc,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "fpr":      fpr,
            "tpr":      tpr,
            "cm":       cm,
        }

    best = max(results, key=lambda k: results[k]["test_auc"])
    print(f"\n  Best model: {best}  (AUC = {results[best]['test_auc']:.4f})")
    return results, best
