"""
data_gen.py — Synthetic telecom churn dataset generator

Generates a realistic customer dataset where churn probability is driven by:
  - Contract type (Month-to-Month has highest churn)
  - Monthly charges (higher charges → more likely to churn)
  - Tenure (longer tenure → more loyal)
  - Internet service (Fiber Optic customers churn more)
"""

import numpy as np
import pandas as pd
import config


def generate() -> pd.DataFrame:
    np.random.seed(config.RANDOM_SEED)
    n = config.N_SAMPLES

    data = {
        "customer_id":       [f"CUST{str(i).zfill(5)}" for i in range(1, n + 1)],
        "tenure":            np.random.randint(1, 73, n),
        "monthly_charges":   np.round(np.random.uniform(20, 120, n), 2),
        "total_charges":     np.round(np.random.uniform(100, 8_000, n), 2),
        "num_products":      np.random.randint(1, 6, n),
        "contract_type":     np.random.choice(
            ["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20]
        ),
        "payment_method":    np.random.choice(
            ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], n
        ),
        "internet_service":  np.random.choice(
            ["DSL", "Fiber Optic", "No"], n, p=[0.34, 0.44, 0.22]
        ),
        "tech_support":      np.random.choice(["Yes", "No", "No Internet"], n),
        "senior_citizen":    np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "partner":           np.random.choice(["Yes", "No"], n),
        "dependents":        np.random.choice(["Yes", "No"], n),
    }

    df = pd.DataFrame(data)

    # Realistic churn probability
    churn_prob = (
        0.05
        + 0.28 * (df["contract_type"] == "Month-to-Month")
        + 0.12 * (df["monthly_charges"] > 70)
        - 0.18 * (df["tenure"] > 24)
        + 0.09 * (df["internet_service"] == "Fiber Optic")
        + 0.05 * (df["senior_citizen"] == 1)
        - 0.06 * (df["partner"] == "Yes")
    ).clip(0.02, 0.92)

    df[config.TARGET] = np.random.binomial(1, churn_prob)

    print(f"Generated {len(df):,} customers")
    print(f"Churn rate: {df[config.TARGET].mean():.1%}")
    return df
