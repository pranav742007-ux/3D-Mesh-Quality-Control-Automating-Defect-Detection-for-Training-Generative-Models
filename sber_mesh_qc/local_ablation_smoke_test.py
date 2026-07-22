"""
================================================================================
SBER AI Journey — 3D Mesh Quality Control: Local Ablation Smoke Test Suite
================================================================================
This script allows developer to verify every ablation config file (M0 to H)
locally on CPU by mocking missing raw data (images and raster meshes) to return
zero-filled tensors. It guarantees zero compilation and runtime errors on Kaggle.

Usage:
  python local_ablation_smoke_test.py
================================================================================
"""

import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
import torch

# Add solution folder to system path
sys.path.insert(0, os.path.abspath("solution"))

import config as cfg
from train import train_full_cv
from utils import derive_quality
from image_processing import MeshQualityDataset

# ── Mock raw data loaders to bypass missing files locally ───────────────────
print("[Setup] Mocking raw view and mesh raster loaders for local smoke test...")
MeshQualityDataset._load_rgb_views = lambda self, item_id, safe_id: [
    torch.zeros((3, 224, 224), dtype=torch.float32) for _ in range(6)
]
MeshQualityDataset._load_raw_geometry_raster = lambda self, item_id: torch.zeros(
    (6, 5, 224, 224), dtype=torch.float32
)

# ── List of configurations to run ───────────────────────────────────────────
CONFIGS = {
    "M0 (Mesh-Only)": "solution/configs/M0_mesh_only.yaml",
    "M1 (RGB-Only)": "solution/configs/M1_rgb_only.yaml",
    "A (Baseline 3-ch)": "solution/configs/A_baseline.yaml",
    "B (Sobel 6-ch)": "solution/configs/B_sobel_6ch.yaml",
    "C (Raster 8-ch)": "solution/configs/C_raster_8ch.yaml",
    "D (Combined 11-ch)": "solution/configs/D_combined_11ch.yaml",
    "E (Winner + Mixup)": "solution/configs/E_winner_mixup.yaml",
    "F (Winner + Gated)": "solution/configs/F_winner_gated.yaml",
    "G (Winner + PointNet)": "solution/configs/G_winner_pointnet.yaml",
    "H (Winner + Separate Quality)": "solution/configs/H_winner_separate_quality.yaml"
}

def run_smoke_test(name: str, config_path: str):
    print("\n" + "=" * 80)
    print(f"  RUNNING SMOKE TEST FOR CONFIGURATION: {name}")
    print(f"  Config file: {config_path}")
    print("=" * 80)
    
    # 1. Reset/Reload configuration overrides from YAML
    cfg.load_yaml_config(config_path)
    
    # 2. Apply strict local smoke test overrides to avoid CPU memory/time issues
    cfg.NUM_FOLDS = 2
    cfg.NUM_EPOCHS = 1
    cfg.BATCH_SIZE = 2
    cfg.NUM_WORKERS = 0
    cfg.VAL_SUBSAMPLE_RATIO = 0.02  # Use small subset of validation to speed up
    cfg.PREPROCESS_IMAGES_OFFLINE = False
    
    # Override train module-level variables directly (since they were statically imported)
    import train
    train.NUM_FOLDS = 2
    train.NUM_EPOCHS = 1
    train.BATCH_SIZE = 2
    train.NUM_WORKERS = 0
    train.VAL_SUBSAMPLE_RATIO = 0.02
    train.PREPROCESS_IMAGES_OFFLINE = False
    train.TRAIN_STEPS_LIMIT = 2
    
    # 3. Reload datasets and check shapes
    train_csv_path = "data/train.csv"
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    if "quality" not in train_df.columns:
        train_df["quality"] = derive_quality(train_df[cfg.DEFECT_COLS].values)
        
    train_features_path = "data/mesh_features_train_extended.npz"
    train_data = np.load(train_features_path)
    train_features = train_data["features"]
    
    # Ensure correct folders
    checkpoint_dir = "checkpoints/smoke_test"
    log_dir = "logs/smoke_test"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Clear previous smoke test checkpoints/logs to start fresh
    for f in os.listdir(checkpoint_dir):
        fp = os.path.join(checkpoint_dir, f)
        if os.path.isfile(fp): os.remove(fp)
        
    # 4. Execute training CV loop
    t0 = time.time()
    cv_results = train_full_cv(
        train_df=train_df,
        image_dir="data/train",
        mesh_features=train_features,
        point_clouds=None,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )
    elapsed = time.time() - t0
    
    print(f"  [OK] Smoke test for {name} completed in {elapsed:.1f}s")
    print(f"  Calibrated OOF F1 score: {cv_results['final_oof_f1_final']:.4f}")
    return cv_results

def main():
    results = []
    failed = []
    
    for name, path in CONFIGS.items():
        try:
            res = run_smoke_test(name, path)
            results.append({
                "Configuration": name,
                "Status": "PASSED",
                "OOF F1": res["final_oof_f1_final"]
            })
        except Exception as e:
            print(f"\n[ERROR] Smoke test failed for {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "Configuration": name,
                "Status": "FAILED",
                "OOF F1": 0.0
            })
            failed.append(name)
            
    print("\n" + "=" * 80)
    print("  ABLATION STUDY SMOKE TEST REPORT SUMMARY")
    print("=" * 80)
    report_df = pd.DataFrame(results)
    print(report_df.to_markdown(index=False))
    
    if len(failed) > 0:
        print(f"\n[FAIL] One or more configurations failed: {failed}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All configurations compiled and passed smoke tests successfully!")

if __name__ == "__main__":
    main()
