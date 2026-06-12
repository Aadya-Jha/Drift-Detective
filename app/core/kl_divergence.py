import numpy as np

def kl_divergence(baseline: list, current: list, bins: int = 10) -> dict:
    """
    KL Divergence between baseline and current distributions.
    Measures information loss when using current distribution
    to approximate baseline distribution.
    Higher KL = more drift.
    """
    baseline = np.array(baseline)
    current = np.array(current)

    # create bins from combined range
    min_val = min(baseline.min(), current.min())
    max_val = max(baseline.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    # calculate proportions
    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts = np.histogram(current, bins=breakpoints)[0]

    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)

    # avoid log(0)
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)

    kl = np.sum(baseline_pct * np.log(baseline_pct / current_pct))

    if kl < 0.1:
        severity = "none"
    elif kl < 0.5:
        severity = "moderate"
    else:
        severity = "high"

    return {
        "kl_score": round(float(kl), 4),
        "severity": severity,
        "drift_detected": kl > 0.1
    }