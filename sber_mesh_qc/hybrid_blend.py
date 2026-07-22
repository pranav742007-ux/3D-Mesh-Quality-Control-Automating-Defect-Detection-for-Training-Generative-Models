import os
import io
import sys
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Ensure local imports work
sys.path.insert(0, os.path.abspath("solution"))
sys.path.insert(0, os.path.abspath("."))

DEFECT_COLS = [
    "abstract", "artifacts", "intersection", "lowpoly", "noisy",
    "open", "partial", "scale", "set", "simple"
]

def check_single_mesh_intersection(args):
    zip_path, item_id = args
    import zipfile, io, os
    import numpy as np
    from solution.mesh_validity import MeshValidityAnalyzer
    
    safe_name = os.path.basename(str(item_id))
    npz_name = f"{safe_name}.npz"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            full_name = None
            for name in namelist:
                if name.endswith(npz_name):
                    full_name = name
                    break
            if full_name is None:
                return item_id, False
            
            npz_bytes = zf.read(full_name)
            data = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
            vertices = np.nan_to_num(data["vertices"], nan=0.0, posinf=1e6, neginf=-1e6)
            faces = np.nan_to_num(data["faces"], nan=0, posinf=0, neginf=0).astype(int)
            
            has_intersect = MeshValidityAnalyzer._compute_aabb_overlaps(vertices, faces) > 0
            return item_id, has_intersect
    except Exception:
        return item_id, False

def optimize_blend_weights(nn_oof, tab_oof, y_true, y_true_quality):
    n_classes = nn_oof.shape[1]
    
    def objective(weights):
        weights = np.clip(weights, 0.0, 1.0)
        blended = nn_oof * (1 - weights) + tab_oof * weights
        preds = (blended >= 0.5).astype(int)
        quality_pred = (preds.sum(axis=1) == 0).astype(int)
        
        f1_defects = []
        for c in range(n_classes):
            tp = np.sum((y_true[:, c] == 1) & (preds[:, c] == 1))
            fp = np.sum((y_true[:, c] == 0) & (preds[:, c] == 1))
            fn = np.sum((y_true[:, c] == 1) & (preds[:, c] == 0))
            f1_defects.append((2*tp) / (2*tp + fp + fn + 1e-10))
        
        f1_q = (2*np.sum((y_true_quality == 1) & (quality_pred == 1))) / \
               (2*np.sum((y_true_quality == 1) & (quality_pred == 1)) + 
                np.sum((y_true_quality == 0) & (quality_pred == 1)) + 
                np.sum((y_true_quality == 1) & (quality_pred == 0)) + 1e-10)
        
        return -(10 * f1_q + 10 * np.mean(f1_defects))
    
    # Initial guess: 0.5 for all
    res = minimize(objective, x0=np.full(n_classes, 0.5), method='L-BFGS-B', bounds=[(0, 1)]*n_classes)
    return np.clip(res.x, 0.0, 1.0)

def optimize_thresholds_f1_final(y_true, y_true_quality, y_pred_proba):
    """Powell Coordinate Descent threshold optimizer."""
    n_classes = len(DEFECT_COLS)
    best_thresholds = np.full(n_classes, 0.5)
    
    def loss_fn(thresholds):
        preds = (y_pred_proba >= thresholds).astype(int)
        quality_preds = (preds.sum(axis=1) == 0).astype(int)
        
        # Calculate F1 metrics
        f1_defects = []
        for c in range(n_classes):
            tp = np.sum((y_true[:, c] == 1) & (preds[:, c] == 1))
            fp = np.sum((y_true[:, c] == 0) & (preds[:, c] == 1))
            fn = np.sum((y_true[:, c] == 1) & (preds[:, c] == 0))
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-10)
            f1_defects.append(f1)
            
        tp_q = np.sum((y_true_quality == 1) & (quality_preds == 1))
        fp_q = np.sum((y_true_quality == 0) & (quality_preds == 1))
        fn_q = np.sum((y_true_quality == 1) & (quality_preds == 0))
        f1_q = (2 * tp_q) / (2 * tp_q + fp_q + fn_q + 1e-10)
        
        final_score = 10 * f1_q + 10 * np.mean(f1_defects)
        return -final_score # Minimize negative score

    # Powell optimization loop
    res = minimize(loss_fn, best_thresholds, method='Powell', 
                   options={'maxiter': 100, 'xtol': 1e-4})
    
    optimized = np.clip(res.x, 0.05, 0.95)
    print(f"  Optimized thresholds: {dict(zip(DEFECT_COLS, optimized.round(3)))}")
    return optimized

def run_nn_test_inference():
    """Runs CPU/GPU inference on the test set using the 3 visual checkpoints."""
    from solution.inference import ensemble_inference
    
    # 1. Unzip test.zip to data/test/ if not already unzipped
    test_dir = "data/test"
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print("Extracting test.zip (1.7GB) to data/test/...")
        os.makedirs(test_dir, exist_ok=True)
        with zipfile.ZipFile("data/test.zip", "r") as zf:
            zf.extractall("data")
        print("✓ Extraction complete.")

    # Load test csv metadata
    test_csv = "data/test.csv"
    test_df = pd.read_csv(test_csv)
    test_ids = test_df["item_id"].tolist()

    # Load pre-extracted test features
    test_features_path = "data/mesh_features_test_extended.npz"
    test_features = None
    if os.path.exists(test_features_path):
        test_features = np.load(test_features_path)["features"]

    print("Running neural network test inference...")
    _, nn_proba_df = ensemble_inference(
        test_ids=test_ids,
        test_image_dir=test_dir,
        checkpoint_dir="checkpoints",
        mesh_features=test_features,
        point_clouds=None,
        cv_results_path=None,
        folds_to_use=[0, 1, 2, 3, 4],
        strict_loading=False,
        effort="max"
    )
    
    # Extract prediction probability matrix
    nn_test_proba = nn_proba_df[DEFECT_COLS].values
    return nn_test_proba

def main():
    from solution.utils import derive_quality
    print("=" * 60)
    print("  HYBRID CLASSIFICATION & BLENDING PIPELINE")
    print("=" * 60)

    # 1. Load Neural Network Out-Of-Fold (OOF) predictions from saved JSON logs
    nn_oof = np.zeros((8964, len(DEFECT_COLS))) # 8964 training meshes
    log_dir = "logs"
    
    log_paths = [os.path.join(log_dir, f"fold_{f}_result.json") for f in range(3)]
    valid_logs = [p for p in log_paths if os.path.exists(p)]
    
    if not valid_logs:
        print("  [INFO] No visual model fold JSON logs found. Using tabular GBDT OOF predictions for validation threshold search.")
        nn_oof = None
    else:
        print(f"Loading neural network validation OOF predictions from {len(valid_logs)} folds...")
        for path in valid_logs:
            with open(path, "r") as f:
                fold_data = json.load(f)
            val_idx = fold_data["val_idx"]
            val_proba = fold_data["val_proba"]
            nn_oof[val_idx] = val_proba

    # 2. Load Tabular Classifier predictions
    tab_oof_path = "data/tabular_oof_preds.npz"
    tab_test_path = "data/tabular_test_preds.npz"
    
    if not os.path.exists(tab_oof_path):
        print(f"✗ Tabular OOF predictions not found at {tab_oof_path}. Run classical_classifier.py first.")
        return

    tab_oof = np.load(tab_oof_path)["oof"]
    tab_test = np.load(tab_test_path)["test"]

    # 3. Load Neural Network Test Predictions
    nn_test_proba = None
    
    # Try local run if checkpoints exist, otherwise fall back to downloaded file
    has_checkpoints = all(os.path.exists(f"checkpoints/best_fold{f}.pt") for f in range(3))
    if has_checkpoints:
        try:
            nn_test_proba = run_nn_test_inference()
        except Exception as e:
            print(f"  [Warning] NN inference failed: {e}. Falling back to downloaded CSV.")
            
    if nn_test_proba is None:
        downloaded_proba = "data/submission_proba.csv"
        if os.path.exists(downloaded_proba):
            print(f"Loading NN test predictions from {downloaded_proba}...")
            df_proba = pd.read_csv(downloaded_proba)
            nn_test_proba = df_proba[DEFECT_COLS].values
        else:
            print(f"✗ Neither checkpoints nor {downloaded_proba} was found. Cannot perform blending.")
            return

    # 4. Perform Hybrid Blending
    # Weight settings: GBDT is extremely good at geometric classes (lowpoly, open, simple).
    # We assign higher weights to tabular model predictions for these classes.
    blend_weights = np.array([
        0.3, # abstract
        0.2, # artifacts
        0.4, # intersection
        0.8, # lowpoly (GBDT-dominant)
        0.3, # noisy
        0.9, # open (GBDT-dominant)
        0.3, # partial
        0.4, # scale
        0.3, # set
        0.7  # simple (GBDT-dominant)
    ])

    print("\nBlending predictions...")
    if nn_oof is not None:
        print("Optimizing blend weights on OOF data...")
        optimal_weights = optimize_blend_weights(nn_oof, tab_oof, y_true, y_true_quality)
        print(f"Optimized Weights: {dict(zip(DEFECT_COLS, optimal_weights.round(3)))}")
        blended_oof = nn_oof * (1 - optimal_weights) + tab_oof * optimal_weights
        blended_test = nn_test_proba * (1 - optimal_weights) + tab_test * optimal_weights
    else:
        print("  [INFO] Visual model OOF predictions missing. Using default weights of 1.0 (GBDT-only).")
        optimal_weights = np.ones(len(DEFECT_COLS))
        blended_oof = tab_oof
        blended_test = tab_test

    # ---- Geometry prior for intersection in blending ----
    import config
    if getattr(config, "ENABLE_INTERSECTION_INFERENCE_GEOPRIOR", False):
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        intersection_idx = DEFECT_COLS.index('intersection')
        print("  [Geometry Prior] Running parallel self-intersection validation on test meshes...")
        
        test_df = pd.read_csv("data/test.csv")
        test_ids = test_df["item_id"].tolist()
        
        tasks = [("data/test.zip", item_id) for item_id in test_ids]
        num_workers = min(os.cpu_count() or 4, 3)
        intersect_results = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Parallel map with progress bar
            results = list(tqdm(executor.map(check_single_mesh_intersection, tasks), total=len(tasks), desc="Intersection Prior Scan"))
            for item_id, has_intersect in results:
                intersect_results[item_id] = has_intersect

        # Apply results to blended_test
        for i, item_id in enumerate(test_ids):
            if intersect_results.get(item_id, False):
                raw_prob = blended_test[i, intersection_idx]
                if raw_prob > 0.2:  # Only boost if model already suspects it
                    boosted = raw_prob + (0.35 * (1 - raw_prob))  # Gentle nudge
                    blended_test[i, intersection_idx] = min(boosted, 0.88)  # Cap at 0.88

        # Load test features for geometric priors
        test_features_path = "data/mesh_features_test_extended.npz"
        if os.path.exists(test_features_path):
            print("  [Geometry Prior] Applying lowpoly and scale geometric rules to test predictions...")
            test_features = np.load(test_features_path)["features"]
            
            lowpoly_idx = DEFECT_COLS.index("lowpoly")
            scale_idx = DEFECT_COLS.index("scale")
            
            for i in range(len(blended_test)):
                num_faces = test_features[i, 1]
                bbox_diag = test_features[i, 9]
                
                # num_faces < 300 -> boost lowpoly probability to at least 0.55
                if num_faces < 300:
                    raw_lowpoly = blended_test[i, lowpoly_idx]
                    blended_test[i, lowpoly_idx] = max(raw_lowpoly, 0.55)
                    
                # bbox_diag < 0.15 or > 1.8 -> boost scale probability to at least 0.60
                if bbox_diag < 0.15 or bbox_diag > 1.8:
                    raw_scale = blended_test[i, scale_idx]
                    blended_test[i, scale_idx] = max(raw_scale, 0.60)

    # 5. Optimize thresholds on blended validation predictions
    train_df = pd.read_csv("data/train.csv")
    y_true = train_df[DEFECT_COLS].values
    y_true_quality = derive_quality(y_true)
    
    print("Optimizing thresholds on blended OOF predictions...")
    opt_thresholds = optimize_thresholds_f1_final(y_true, y_true_quality, blended_oof)

    # Apply manual threshold overrides from config
    opt_thresholds_with_overrides = opt_thresholds.copy()
    if getattr(config, "OVERRIDE_THRESHOLDS", False) and getattr(config, "MANUAL_THRESHOLD_OVERRIDES", None) is not None:
        for class_name, val in config.MANUAL_THRESHOLD_OVERRIDES.items():
            if class_name in DEFECT_COLS:
                idx = DEFECT_COLS.index(class_name)
                opt_thresholds_with_overrides[idx] = val
    print(f"  Optimized thresholds (raw Powell): {dict(zip(DEFECT_COLS, opt_thresholds.round(3)))}")
    print(f"  Optimized thresholds (with overrides): {dict(zip(DEFECT_COLS, opt_thresholds_with_overrides.round(3)))}")

    # 6. Generate final submission labels
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["item_id"].tolist()
    
    # Clean-Mesh Confidence Gate: if max defect probability < 0.30, force clean
    max_probs = blended_test.max(axis=1)
    clean_candidates = max_probs < 0.30

    # Version A: Raw Powell optimized
    test_preds_powell = (blended_test >= opt_thresholds).astype(int)
    test_preds_powell[clean_candidates] = 0
    sub_powell = pd.DataFrame(test_preds_powell, columns=DEFECT_COLS)
    sub_powell.insert(0, "item_id", test_ids)
    sub_powell["quality"] = derive_quality(test_preds_powell)
    sub_powell.to_csv("submission_powell_opt.csv", index=False)
    print(f"\n[SUCCESS] Saved Powell-optimized submission (A) to submission_powell_opt.csv")
    print(f"  Clean quality meshes predicted: {(sub_powell['quality'] == 1).sum()} / {len(sub_powell)}")

    # Version B: Overrides and Priors (Default submission.csv)
    test_preds_overrides = (blended_test >= opt_thresholds_with_overrides).astype(int)
    test_preds_overrides[clean_candidates] = 0
    sub_overrides = pd.DataFrame(test_preds_overrides, columns=DEFECT_COLS)
    sub_overrides.insert(0, "item_id", test_ids)
    sub_overrides["quality"] = derive_quality(test_preds_overrides)
    sub_overrides.to_csv("submission.csv", index=False)
    sub_overrides.to_csv("submission_overrides.csv", index=False)
    print(f"[SUCCESS] Saved Overrides & Priors submission (B) to submission.csv and submission_overrides.csv")
    print(f"  Clean quality meshes predicted: {(sub_overrides['quality'] == 1).sum()} / {len(sub_overrides)}")

if __name__ == "__main__":
    main()
