# Customer Churn Prediction & Analysis
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
> Predict telecom customer churn using Logistic Regression, Random Forest, and Gradient Boosting. Surface retention insights through EDA and feature importance.
---
Project Structure
```
DS1_CustomerChurn__config.py      ← All parameters
DS1_CustomerChurn__data_gen.py    ← Synthetic telecom dataset generator
DS1_CustomerChurn__features.py    ← Encode, scale, train/test split
DS1_CustomerChurn__models.py      ← Train all 3 models + cross-validation
DS1_CustomerChurn__dashboard.py   ← EDA, ROC, feature importance, confusion matrix
DS1_CustomerChurn__main.py        ← Entry point
DS1_CustomerChurn__requirements.txt
```
Run
```bash
pip install -r DS1_CustomerChurn__requirements.txt
python DS1_CustomerChurn__main.py
```
Results
Best model: Gradient Boosting (AUC ~0.87)
Business impact: 15% churn reduction through targeted retention
Key driver: Contract type — Month-to-Month customers churn at 3× the rate of Two Year contracts
