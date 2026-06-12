from app.core.ks_test import ks_test
from app.core.psi import calculate_psi
from app.core.kl_divergence import kl_divergence

def score_feature(feature_name: str, baseline: list, current: list) -> dict:
    """
    Runs all three drift detection methods on a single feature
    and returns a combined drift score and verdict.
    """
    ks = ks_test(baseline, current)
    psi = calculate_psi(baseline, current)
    kl = kl_divergence(baseline, current)

    # count how many methods flagged drift
    drift_votes = sum([
        ks["drift_detected"],
        psi["drift_detected"],
        kl["drift_detected"]
    ])

    if drift_votes == 0:
        overall_severity = "none"
    elif drift_votes == 1:
        overall_severity = "low"
    elif drift_votes == 2:
        overall_severity = "moderate"
    else:
        overall_severity = "high"

    return {
        "feature": feature_name,
        "ks_test": ks,
        "psi": psi,
        "kl_divergence": kl,
        "drift_votes": drift_votes,
        "overall_severity": overall_severity,
        "drift_detected": drift_votes >= 2
    }


def score_all_features(baseline_df, current_df) -> dict:
    """
    Scores drift for all numerical features across baseline and current dataframes.
    Returns per-feature results and a ranked list by severity.
    """
    import pandas as pd

    numerical_cols = baseline_df.select_dtypes(include=["float64", "int64"]).columns
    results = {}

    for col in numerical_cols:
        if col in current_df.columns:
            results[col] = score_feature(
                feature_name=col,
                baseline=baseline_df[col].dropna().tolist(),
                current=current_df[col].dropna().tolist()
            )

    # rank features by drift votes descending
    ranked = sorted(results.values(), key=lambda x: x["drift_votes"], reverse=True)

    total_features = len(results)
    drifted_features = sum(1 for r in results.values() if r["drift_detected"])

    return {
        "total_features": total_features,
        "drifted_features": drifted_features,
        "model_health": "critical" if drifted_features / total_features > 0.5 else
                        "degraded" if drifted_features / total_features > 0.2 else "healthy",
        "feature_results": results,
        "ranked_features": ranked
    }