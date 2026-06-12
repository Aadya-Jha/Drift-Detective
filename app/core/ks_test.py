import numpy as np
from scipy import stats

def ks_test(baseline: list, current: list) -> dict:
    """
    Kolmogorov-Smirnov test between baseline and current feature distributions.
    Returns statistic, p-value, and a drift flag.
    """
    baseline = np.array(baseline)
    current = np.array(current)

    statistic, p_value = stats.ks_2samp(baseline, current)

    return {
    "statistic": round(float(statistic), 4),
    "p_value": round(float(p_value), 4),
    "drift_detected": bool(p_value < 0.05)
    }