import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance

# Set random seed
SEED = 42
np.random.seed(SEED)

DEFECT_COLS = [
    "abstract", "artifacts", "intersection", "lowpoly", "noisy",
    "open", "partial", "scale", "set", "simple"
]

def derive_quality(pred_matrix: np.ndarray) -> np.ndarray:
    """Quality is 1 iff all 10 defect columns are 0."""
    return (pred_matrix.sum(axis=1) == 0).astype(int)

def compute_f1_final(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Calculate the competition's final metric."""
    f1_defects = []
    for col in DEFECT_COLS:
        f1_defects.append(f1_score(true_df[col], pred_df[col], zero_division=0))
    avg_f1_defects = np.mean(f1_defects)
    f1_quality = f1_score(true_df["quality"], pred_df["quality"], zero_division=0)
    final_score = 10 * f1_quality + 10 * avg_f1_defects
    return {
        "f1_quality": f1_quality,
        "f1_defects": avg_f1_defects,
        "f1_final": final_score
    }

def main():
    print("Loading data...")
    train_csv_path = "data/train.csv"
    test_csv_path = "data/test.csv"
    train_features_path = "data/mesh_features_train_extended.npz"
    test_features_path = "data/mesh_features_test_extended.npz"

    if not os.path.exists(train_features_path):
        print(f"✗ Train features NPZ not found at {train_features_path}. Run extract_features.py first.")
        return

    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    
    # Load feature arrays
    train_data = np.load(train_features_path)
    X = train_data["features"]
    y = train_df[DEFECT_COLS].values
    
    test_data = np.load(test_features_path)
    X_test = test_data["features"]

    print(f"Train features shape: {X.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Set up 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Matrices to hold OOF and test predictions
    oof_preds = np.zeros((len(train_df), len(DEFECT_COLS)))
    test_preds = np.zeros((len(X_test), len(DEFECT_COLS)))

    # Feature names (mapping from solution/mesh_features.py)
    # We will use general indices if names aren't fully resolved
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    
    # Try to load exact feature names from mesh_features config to make importances interpretable
    try:
        from solution.mesh_features import FEATURE_ORDER
        # First 68 are base order, next are SHTD (25), topological (3), QEM (1), physics (3)
        feature_names = list(FEATURE_ORDER)
        feature_names += [f"SHTD_L4_{i}" for i in range(25)]
        feature_names += ["Betti_0", "Betti_1", "Euler_Characteristic"]
        feature_names += ["QEM_Decimation_Stability"]
        feature_names += ["COM_Height", "Support_Radius", "Tipping_Angle"]
    except Exception:
        pass

    # Train a model for each class
    print("\nTraining classical Gradient Boosting ensembles per defect class...")
    for class_idx, class_name in enumerate(DEFECT_COLS):
        print(f"  Class: {class_name}")
        y_class = y[:, class_idx]
        
        class_oof = np.zeros(len(train_df))
        class_test = np.zeros(len(X_test))
        
        # HistGradientBoosting is optimized for CPU tabular learning
        model_params = {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "l2_regularization": 1.5,
            "random_state": SEED,
            "class_weight": "balanced"  # Handle extreme class imbalance
        }

        # Calculate mathematical Point-Biserial correlations to check connections
        from scipy.stats import pointbiserialr
        correlations = []
        for i in range(X.shape[1]):
            feat = X[:, i]
            if np.std(feat) == 0:
                corr = 0.0
            else:
                try:
                    corr, _ = pointbiserialr(feat, y_class)
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0
            correlations.append(abs(corr))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class)):
            X_train, y_train = X[train_idx], y_class[train_idx]
            X_val, y_val = X[val_idx], y_class[val_idx]
            
            clf = HistGradientBoostingClassifier(**model_params)
            clf.fit(X_train, y_train)
            
            # Predict validation
            val_proba = clf.predict_proba(X_val)[:, 1]
            class_oof[val_idx] = val_proba
            
            # Predict test
            test_proba = clf.predict_proba(X_test)[:, 1]
            class_test += test_proba / 5.0

        oof_preds[:, class_idx] = class_oof
        test_preds[:, class_idx] = class_test
        
        # Report the top 3 most important geometric features driving this defect
        top_indices = np.argsort(correlations)[::-1][:3]
        print(f"    Connections found (Top 3 geometric drivers):")
        for rank, idx in enumerate(top_indices):
            val = correlations[idx]
            if val > 0.01:
                name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                print(f"      {rank+1}. {name} (correlation: {val:.4f})")

    # Evaluate validation metrics
    oof_pred_labels = (oof_preds >= 0.5).astype(int)
    oof_pred_df = pd.DataFrame(oof_pred_labels, columns=DEFECT_COLS)
    oof_pred_df["quality"] = derive_quality(oof_pred_labels)
    
    true_df = train_df[DEFECT_COLS].copy()
    true_df["quality"] = derive_quality(y)
    
    metrics = compute_f1_final(true_df, oof_pred_df)
    print("\n" + "=" * 60)
    print("  LOCAL CPU CROSS-VALIDATION SUMMARY (Threshold = 0.5)")
    print("=" * 60)
    print(f"  F1 Quality Score:   {metrics['f1_quality']:.4f}")
    print(f"  F1 Defects Score:   {metrics['f1_defects']:.4f}")
    print(f"  Final Blend Metric: {metrics['f1_final']:.2f} / 20.0")
    print("=" * 60)

    # Save OOF and Test predictions for blending step
    np.savez_compressed("data/tabular_oof_preds.npz", oof=oof_preds)
    np.savez_compressed("data/tabular_test_preds.npz", test=test_preds)
    print("\n[SUCCESS] Saved tabular probabilities to data/")

if __name__ == "__main__":
    main()
