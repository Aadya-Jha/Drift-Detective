import numpy as np

def calculate_psi(baseline: list, current: list, bins: int = 10) -> dict:
    """
    Population Stability Index between baseline and current distributions.
    PSI < 0.1: no drift
    PSI 0.1-0.2: moderate drift
    PSI > 0.2: significant drift
    """
    baseline = np.array(baseline)
    current = np.array(current)

    # create bins from baseline
    breakpoints = np.linspace(baseline.min(), baseline.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # calculate proportions
    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts = np.histogram(current, bins=breakpoints)[0]

    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)

    # avoid division by zero
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)

    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    psi_score = np.sum(psi_values)

    if psi_score < 0.1:
        severity = "none"
    elif psi_score < 0.2:
        severity = "moderate"
    else:
        severity = "high"

    return {
    "psi_score": round(float(psi_score), 4),
    "severity": severity,
    "drift_detected": bool(psi_score > 0.1)
    }