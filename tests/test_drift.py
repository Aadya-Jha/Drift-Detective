import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from app.core.ks_test import ks_test
from app.core.psi import calculate_psi
from app.core.kl_divergence import kl_divergence
from app.core.scorer import score_feature, score_all_features

# ── Sample Data ───────────────────────────────────────────────────────────────
np.random.seed(42)

baseline = np.random.normal(50, 10, 1000).tolist()
no_drift = np.random.normal(50, 10, 500).tolist()       # same distribution
mild_drift = np.random.normal(55, 10, 500).tolist()     # slight shift
high_drift = np.random.normal(80, 15, 500).tolist()     # heavy shift

def test_ks_no_drift():
    result = ks_test(baseline, no_drift)
    print(f"KS No Drift: {result}")
    assert result["drift_detected"] == False

def test_ks_high_drift():
    result = ks_test(baseline, high_drift)
    print(f"KS High Drift: {result}")
    assert result["drift_detected"] == True

def test_psi_no_drift():
    result = calculate_psi(baseline, no_drift)
    print(f"PSI No Drift: {result}")
    assert result["severity"] == "none"

def test_psi_high_drift():
    result = calculate_psi(baseline, high_drift)
    print(f"PSI High Drift: {result}")
    assert result["drift_detected"] == True

def test_kl_no_drift():
    result = kl_divergence(baseline, no_drift)
    print(f"KL No Drift: {result}")
    assert result["drift_detected"] == False

def test_kl_high_drift():
    result = kl_divergence(baseline, high_drift)
    print(f"KL High Drift: {result}")
    assert result["drift_detected"] == True

def test_score_feature():
    result = score_feature("age", baseline, high_drift)
    print(f"Score Feature: {result}")
    assert result["drift_detected"] == True
    assert result["drift_votes"] >= 2

def test_score_all_features():
    baseline_df = pd.DataFrame({
        "age": np.random.normal(50, 10, 1000),
        "income": np.random.normal(60000, 15000, 1000),
        "score": np.random.normal(0.7, 0.1, 1000)
    })
    current_df = pd.DataFrame({
        "age": np.random.normal(65, 10, 500),       # drifted
        "income": np.random.normal(60000, 15000, 500),  # stable
        "score": np.random.normal(0.4, 0.15, 500)   # drifted
    })
    result = score_all_features(baseline_df, current_df)
    print(f"Score All Features: {result['model_health']}")
    print(f"Drifted: {result['drifted_features']}/{result['total_features']}")
    assert result["drifted_features"] >= 1

if __name__ == "__main__":
    print("Running tests...\n")
    test_ks_no_drift()
    test_ks_high_drift()
    test_psi_no_drift()
    test_psi_high_drift()
    test_kl_no_drift()
    test_kl_high_drift()
    test_score_feature()
    test_score_all_features()
    print("\nAll tests passed.")