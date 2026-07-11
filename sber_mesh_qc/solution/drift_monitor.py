"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Drift Monitor [v7.3]
===============================================================================
Phase 10 Production Monitoring & Feature Drift. Evaluates Population Stability Index
(PSI) and Wasserstein Distance between train and validation/production features.
===============================================================================
"""
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None
    print("[WARNING] scipy not installed. Drift monitoring features will be limited.")

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    """
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    percentiles = np.percentile(expected, np.linspace(0, 100, num_bins + 1))
    # Adjust endpoints slightly to ensure all values are included
    percentiles[0] -= 1e-5
    percentiles[-1] += 1e-5
    
    expected_counts = np.histogram(expected, bins=percentiles)[0]
    actual_counts = np.histogram(actual, bins=percentiles)[0]
    
    expected_pcts = expected_counts / len(expected)
    actual_pcts = actual_counts / len(actual)
    
    # Handle zero counts using standard Laplacian smoothing
    expected_pcts = np.where(expected_pcts == 0, 1e-4, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, 1e-4, actual_pcts)
    
    # Recalculate pcts to sum to 1
    expected_pcts /= expected_pcts.sum()
    actual_pcts /= actual_pcts.sum()
    
    psi_value = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi_value)

def monitor_feature_drift(train_features: np.ndarray, prod_features: np.ndarray, feature_names: list) -> dict:
    """
    Evaluates drift across all features using PSI and Wasserstein distance.
    """
    num_features = train_features.shape[1]
    drift_report = {}
    
    for idx in range(num_features):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        train_col = train_features[:, idx]
        prod_col = prod_features[:, idx]
        
        psi = calculate_psi(train_col, prod_col)
        # Calculate Wasserstein distance
        try:
            if stats is not None:
                w_dist = float(stats.wasserstein_distance(train_col, prod_col))
            else:
                w_dist = 0.0
        except Exception:
            w_dist = 0.0
            
        status = "STABLE"
        if psi >= 0.25:
            status = "ACTION_REQUIRED"
        elif psi >= 0.10:
            status = "WARNING"
            
        drift_report[name] = {
            "psi": psi,
            "wasserstein_distance": w_dist,
            "status": status
        }
        
    return drift_report

if __name__ == "__main__":
    # Self-test
    np.random.seed(42)
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0.2, 1, 1000)
    
    psi = calculate_psi(expected, actual)
    w_dist = stats.wasserstein_distance(expected, actual)
    print(f"[Self-Test] Calculated PSI: {psi:.4f} (Status: {'STABLE' if psi < 0.1 else 'DRIFTED'})")
    print(f"[Self-Test] Wasserstein distance: {w_dist:.4f}")
