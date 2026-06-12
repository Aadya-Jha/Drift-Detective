import numpy as np
import pandas as pd

np.random.seed(42)

baseline = pd.DataFrame({
    "age": np.random.normal(35, 10, 1000),
    "income": np.random.normal(60000, 15000, 1000),
    "credit_score": np.random.normal(700, 50, 1000),
    "loan_amount": np.random.normal(20000, 5000, 1000)
})

current = pd.DataFrame({
    "age": np.random.normal(50, 12, 500),           # drifted
    "income": np.random.normal(60000, 15000, 500),  # stable
    "credit_score": np.random.normal(650, 80, 500), # drifted
    "loan_amount": np.random.normal(20000, 5000, 500) # stable
})

baseline.to_csv("data/baseline/baseline.csv", index=False)
current.to_csv("data/logs/current.csv", index=False)
print("Sample data generated.")