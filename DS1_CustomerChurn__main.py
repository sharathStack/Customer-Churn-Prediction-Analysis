"""
main.py — Customer Churn Prediction entry point

Run order:
  1. Generate synthetic telecom dataset
  2. EDA visualisation
  3. Preprocess (encode + scale + split)
  4. Train Logistic Regression, Random Forest, Gradient Boosting
  5. Plot ROC, feature importance, confusion matrix
"""

import config
from data_gen   import generate
from features   import preprocess
from models     import train_all
from dashboard  import (plot_eda, plot_feature_importance,
                         plot_roc, plot_confusion)


def main():
    print("=" * 55)
    print("  CUSTOMER CHURN PREDICTION & ANALYSIS")
    print("=" * 55)

    # 1. Data
    print("\n[1] Generating dataset...")
    df = generate()

    # 2. EDA
    print("\n[2] Plotting EDA...")
    plot_eda(df)

    # 3. Preprocess
    print("\n[3] Preprocessing...")
    X_train, X_test, y_train, y_test, feat_names = preprocess(df)

    # 4. Train
    print("\n[4] Training models...")
    results, best_name = train_all(X_train, X_test, y_train, y_test)

    # 5. Plots
    print("\n[5] Generating charts...")
    plot_roc(results, y_test)
    plot_feature_importance(results[best_name]["model"], feat_names, best_name)
    plot_confusion(results[best_name]["cm"], best_name)

    # Summary
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    for name, res in results.items():
        print(f"  {name:<25}  AUC = {res['test_auc']:.4f}")
    print(f"\n  Best: {best_name}  →  AUC = {results[best_name]['test_auc']:.4f}")
    print("\n  Done ✓")


if __name__ == "__main__":
    main()
