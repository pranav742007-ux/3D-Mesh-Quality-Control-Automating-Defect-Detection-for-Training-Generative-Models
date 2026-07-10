"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Main Orchestrator  [v7.2 Master Engine]
===============================================================================
End-to-end pipeline:
  1. Download & extract competition data (via 16-worker parallel downloader)
  2. Pre-extract 100D geometric & topological mesh features -> .npz cache
  3. (Optional) Pre-extract PointNet 3D point clouds -> .npz cache
  4. Train CV ensemble (Octopus MoE + GLM-5.2 IndexShare + Kimi DPO + OmniRoute)
  5. Run inference with test-time augmentation (TTA) and threshold optimization
  6. Generate competition submission.csv
==============================================================================
"""

from typing import Optional
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd


def resolve_paths(args):
    """Resolve all directory paths based on mode."""
    base_dir = args.base_dir

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(base_dir, "data")

    checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "checkpoints")
    log_dir = args.log_dir or os.path.join(base_dir, "logs")
    submission_path = args.output or os.path.join(base_dir, "submission.csv")

    return base_dir, data_dir, checkpoint_dir, log_dir, submission_path


def step_download(data_dir, base_dir):
    """Step 1: Download and extract competition data."""
    print("\n" + "=" * 60)
    print("  STEP 1: DOWNLOADING DATA")
    print("=" * 60)

    from data_utils import download_data, prepare_data_dirs, validate_data_integrity

    download_data(data_dir)

    # Validate layout
    try:
        paths = prepare_data_dirs(data_dir)
    except FileNotFoundError as e:
        print(f"  [WARNING] Layout validation failed: {e}")
        print("  Attempting integrity check with default paths...")
        paths = None

    # Validate data integrity (v2.0 FIX: correct 4-arg signature)
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if os.path.isfile(train_csv) and os.path.isfile(test_csv):
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            validate_data_integrity(train_csv, train_dir, test_csv, test_dir)
        else:
            print("  [WARNING] train/ or test/ directory not found — skipping integrity check")
    else:
        print("  [WARNING] CSV files not found — skipping integrity check")

    print("  Data download and extraction complete.")
    print(f"  Data directory contents ({data_dir}):")
    try:
        for item in sorted(os.listdir(data_dir))[:20]:
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                n_files = len(os.listdir(item_path))
                print(f"    📁 {item}/ ({n_files} items)")
            else:
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"    📄 {item} ({size_mb:.1f} MB)")
    except Exception:
        pass
    return data_dir


def _find_cache_file(filename: str, base_dir: str, data_dir: str):
    """Search for a cached feature .npz file across multiple candidate paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "solution", filename),
        os.path.join(script_dir, filename),
        os.path.join(data_dir, filename),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def step_extract_features(data_dir, base_dir, extended=True):
    """
    Step 2: Pre-extract mesh geometric features into .npz cache.

    If cache files already exist in base_dir, solution/, script_dir, or data_dir,
    they are loaded directly (0 seconds). Otherwise, features are computed on CPU.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: EXTRACTING MESH FEATURES")
    print("=" * 60)

    from mesh_features import batch_extract_mesh_features

    import config as cfg
    cache_format = getattr(cfg, "FEATURE_CACHE_FORMAT", "mmap")
    suffix = "extended" if extended else "basic"
    
    if cache_format == "mmap":
        train_cache_name = f"mesh_features_train_{suffix}.npy"
        test_cache_name = f"mesh_features_test_{suffix}.npy"
    else:
        train_cache_name = f"mesh_features_train_{suffix}.npz"
        test_cache_name = f"mesh_features_test_{suffix}.npz"

    cache_path_out = os.path.join(base_dir, train_cache_name)
    cache_path_test_out = os.path.join(base_dir, test_cache_name)

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_csv)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    train_ids = [str(x) for x in train_df["item_id"].tolist()]
    expected_train_dim = 100 if extended else 58

    def _validate_and_load_cache(cache_path: str, expected_ids: list, expected_dim: int) -> Optional[np.ndarray]:
        if not os.path.isfile(cache_path):
            return None
        try:
            if cache_path.endswith(".npz"):
                data = np.load(cache_path, allow_pickle=False)
                if "features" not in data or "item_ids" not in data:
                    return None
                feats = data["features"]
                cached_ids = [str(x) for x in data["item_ids"]]
            else:
                ids_path = cache_path.replace(".npy", "_ids.npy")
                if not os.path.isfile(ids_path):
                    return None
                mmap_mode = "r" if cache_format == "mmap" else None
                feats = np.load(cache_path, mmap_mode=mmap_mode)
                cached_ids = [str(x) for x in np.load(ids_path, allow_pickle=True)]
                
            if feats.shape[0] != len(expected_ids) or feats.shape[1] != expected_dim:
                print(f"  [CACHE INVALID] Shape mismatch in {os.path.basename(cache_path)}: {feats.shape} vs expected ({len(expected_ids)}, {expected_dim})")
                return None
            if cached_ids != expected_ids:
                print(f"  [CACHE INVALID] Item ID mismatch/reordering in {os.path.basename(cache_path)}")
                return None
            return feats
        except Exception as err:
            print(f"  [CACHE ERROR] Failed to read cache {cache_path}: {err}")
            return None

    # --- Train features ---
    if cache_format == "mmap":
        found_train = _find_cache_file(train_cache_name, base_dir, data_dir) or \
                      _find_cache_file("mesh_features_train_extended.npy", base_dir, data_dir) or \
                      _find_cache_file("mesh_features_train.npy", base_dir, data_dir)
    else:
        found_train = _find_cache_file(train_cache_name, base_dir, data_dir) or \
                      _find_cache_file("mesh_features_train_extended.npz", base_dir, data_dir) or \
                      _find_cache_file("mesh_features_train.npz", base_dir, data_dir)

    train_features = _validate_and_load_cache(found_train, train_ids, expected_train_dim) if found_train else None

    if train_features is not None:
        print(f"  Validated pre-computed train features loaded from {found_train} — shape: {train_features.shape}")
    else:
        npz_dir = _find_npz_dir(data_dir, "train")
        print(f"  Extracting {len(train_ids)} train features ({'68-dim extended' if extended else '58-dim basic'}) from {npz_dir}")
        print(f"  This may take 10-30 minutes depending on mesh complexity...")

        t0 = time.time()
        train_features = batch_extract_mesh_features(train_ids, npz_dir, extended=extended)
        print(f"  Features extracted in {time.time() - t0:.1f}s — shape: {train_features.shape}")

        if cache_format == "mmap":
            ids_path = cache_path_out.replace(".npy", "_ids.npy")
            np.save(cache_path_out, train_features)
            np.save(ids_path, np.array(train_ids, dtype=object))
            print(f"  Cached to {cache_path_out} and {ids_path} (mmap format)")
        else:
            np.savez_compressed(cache_path_out, features=train_features, item_ids=np.array(train_ids))
            print(f"  Cached to {cache_path_out} (npz compressed)")

    # --- Test features ---
    if not os.path.exists(test_csv):
        from data_utils import _auto_generate_test_csv_if_missing
        _auto_generate_test_csv_if_missing(data_dir)
    test_df = pd.read_csv(test_csv)
    test_df = test_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    test_ids = [str(x) for x in test_df["item_id"].tolist()]
    expected_test_dim = expected_train_dim

    if cache_format == "mmap":
        found_test = _find_cache_file(test_cache_name, base_dir, data_dir) or \
                     _find_cache_file("mesh_features_test_extended.npy", base_dir, data_dir) or \
                     _find_cache_file("mesh_features_test.npy", base_dir, data_dir)
    else:
        found_test = _find_cache_file(test_cache_name, base_dir, data_dir) or \
                     _find_cache_file("mesh_features_test_extended.npz", base_dir, data_dir) or \
                     _find_cache_file("mesh_features_test.npz", base_dir, data_dir)

    test_features = _validate_and_load_cache(found_test, test_ids, expected_test_dim) if found_test else None

    if test_features is not None:
        print(f"  Validated pre-computed test features loaded from {found_test} — shape: {test_features.shape}")
    else:
        npz_dir = _find_npz_dir(data_dir, "test")
        print(f"  Extracting {len(test_ids)} test features from {npz_dir}")

        t0 = time.time()
        test_features = batch_extract_mesh_features(test_ids, npz_dir, extended=extended)
        print(f"  Features extracted in {time.time() - t0:.1f}s — shape: {test_features.shape}")

        if cache_format == "mmap":
            ids_path = cache_path_test_out.replace(".npy", "_ids.npy")
            np.save(cache_path_test_out, test_features)
            np.save(ids_path, np.array(test_ids, dtype=object))
            print(f"  Cached to {cache_path_test_out} and {ids_path} (mmap format)")
        else:
            np.savez_compressed(cache_path_test_out, features=test_features, item_ids=np.array(test_ids))
            print(f"  Cached to {cache_path_test_out} (npz compressed)")

    print("  Mesh feature extraction complete.")
    return train_features, test_features


def step_extract_point_clouds(data_dir, base_dir, num_points=1024):
    """
    Step 2b: Extract point clouds for PointNet branch (Limitation #1).

    Optional step — only called when USE_POINTNET_BRANCH=True.
    Gracefully degrades if extraction fails.
    """
    print("\n" + "=" * 60)
    print("  STEP 2b: EXTRACTING POINT CLOUDS (for PointNet branch)")
    print("=" * 60)

    try:
        from pointnet_lite import batch_extract_point_clouds
    except ImportError as e:
        print(f"  [WARNING] Cannot import pointnet_lite: {e}")
        print("  Skipping point cloud extraction.")
        return None, None

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    cache_train = os.path.join(base_dir, "point_clouds_train.npz")
    cache_test = os.path.join(base_dir, "point_clouds_test.npz")

    train_df = pd.read_csv(train_csv)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    test_df = pd.read_csv(test_csv)
    test_df = test_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    train_ids = train_df["item_id"].tolist()
    test_ids = test_df["item_id"].tolist()

    # Train point clouds
    try:
        train_npz_dir = _find_npz_dir(data_dir, "train")
        train_pcs = batch_extract_point_clouds(
            train_ids, train_npz_dir,
            num_points=num_points,
            cache_path=cache_train,
        )
        print(f"  Train point clouds: {train_pcs.shape}")
    except Exception as e:
        print(f"  [WARNING] Train point cloud extraction failed: {e}")
        train_pcs = None

    # Test point clouds
    try:
        test_npz_dir = _find_npz_dir(data_dir, "test")
        test_pcs = batch_extract_point_clouds(
            test_ids, test_npz_dir,
            num_points=num_points,
            cache_path=cache_test,
        )
        print(f"  Test point clouds: {test_pcs.shape}")
    except Exception as e:
        print(f"  [WARNING] Test point cloud extraction failed: {e}")
        test_pcs = None

    if train_pcs is not None and test_pcs is not None:
        print("  Point cloud extraction complete.")
    else:
        print("  Point cloud extraction partially/fully failed — PointNet branch will be disabled.")

    return train_pcs, test_pcs


def step_train(train_features, data_dir, checkpoint_dir, log_dir, point_clouds=None):
    """Step 3: Train CV ensemble.

    v2.0: Passes point_clouds for optional PointNet branch.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: TRAINING CV ENSEMBLE")
    print("=" * 60)

    from train import train_full_cv

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    if "quality" not in train_df.columns:
        from utils import derive_quality
        import config
        train_df["quality"] = derive_quality(train_df[config.DEFECT_COLS].values)

    # Find train image directory
    train_image_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_image_dir):
        for candidate in [data_dir, os.path.join(data_dir, "train_images")]:
            if os.path.isdir(candidate) and any(f.endswith(".png") for f in os.listdir(candidate)[:5]):
                train_image_dir = candidate
                break

    print(f"  Training on {len(train_df)} samples")
    print(f"  Image directory: {train_image_dir}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Mesh features shape: {train_features.shape}")
    if point_clouds is not None:
        print(f"  Point clouds shape: {point_clouds.shape}")

    t0 = time.time()
    cv_results = train_full_cv(
        train_df=train_df,
        image_dir=train_image_dir,
        mesh_features=train_features,
        point_clouds=point_clouds,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )
    elapsed = time.time() - t0

    print(f"\n  Training complete in {elapsed / 60:.1f} minutes.")
    return cv_results


def step_infer(test_features, data_dir, checkpoint_dir, log_dir, submission_path,
               point_clouds=None):
    """Step 4: Inference with TTA + submission generation.

    v2.0: Passes point_clouds for optional PointNet branch.
    """
    print("\n" + "=" * 60)
    print("  STEP 4: INFERENCE + SUBMISSION GENERATION")
    print("=" * 60)

    from inference import generate_submission

    test_csv = os.path.join(data_dir, "test.csv")
    test_image_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_image_dir):
        for candidate in [data_dir, os.path.join(data_dir, "test_images")]:
            if os.path.isdir(candidate) and any(f.endswith(".png") for f in os.listdir(candidate)[:5]):
                test_image_dir = candidate
                break

    cv_results_path = os.path.join(log_dir, "cv_results.json")

    print(f"  Test image directory: {test_image_dir}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  CV results: {cv_results_path}")

    generate_submission(
        test_csv_path=test_csv,
        test_image_dir=test_image_dir,
        checkpoint_dir=checkpoint_dir,
        output_path=submission_path,
        cv_results_path=cv_results_path,
        mesh_features=test_features,
        point_clouds=point_clouds,
    )

    print(f"\n  Submission saved to: {submission_path}")
    return submission_path


def step_pseudo_label(test_features, data_dir, checkpoint_dir, log_dir, point_clouds=None):
    """
    Self-Training: Predict on test set, keep predictions with >98% confidence,
    and save augmented dataset to train_pseudo.csv.
    """
    print("\n" + "=" * 60)
    print("  STEP: PSEUDO-LABELING TEST SET (SELF-TRAINING)")
    print("=" * 60)

    test_csv = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_csv)
    test_df = test_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    if "Unnamed: 0" in test_df.columns:
        test_df = test_df.drop(columns=["Unnamed: 0"])
    test_ids = test_df["item_id"].tolist()

    test_image_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_image_dir):
        for candidate in [data_dir, os.path.join(data_dir, "test_images")]:
            if os.path.isdir(candidate) and any(f.endswith(".png") for f in os.listdir(candidate)[:5]):
                test_image_dir = candidate
                break

    cv_results_path = os.path.join(log_dir, "cv_results.json")

    # 1. Run ensembled inference
    from inference import ensemble_inference
    import config as cfg

    print(f"  Test image directory: {test_image_dir}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  CV results: {cv_results_path}")

    # Generate probabilities (returned as second return value of ensemble_inference)
    _, proba_df = ensemble_inference(
        test_ids=test_ids,
        test_image_dir=test_image_dir,
        checkpoint_dir=checkpoint_dir,
        mesh_features=test_features,
        point_clouds=point_clouds,
        cv_results_path=cv_results_path,
        effort="max",
    )

    test_proba = proba_df[cfg.DEFECT_COLS].values

    # 2. Filter for high-confidence predictions (probability > 98% or < 2%)
    max_probs = test_proba.max(axis=1)
    min_probs = test_proba.min(axis=1)
    confidence_mask = (max_probs > 0.98) | (min_probs < 0.02)

    pseudo_indices = np.where(confidence_mask)[0]
    print(f"\n  Found {len(pseudo_indices)} high-confidence pseudo-labels out of {len(test_ids)} total test samples.")

    if len(pseudo_indices) == 0:
        print("  [WARNING] No high-confidence pseudo-labels found. Skipping augmentation.")
        return

    # 3. Generate pseudo-labeled DataFrame
    from utils import derive_quality
    pseudo_preds = (test_proba[pseudo_indices] >= 0.5).astype(int)
    pseudo_df = pd.DataFrame(pseudo_preds, columns=cfg.DEFECT_COLS)
    pseudo_df.insert(0, "item_id", [test_ids[i] for i in pseudo_indices])
    pseudo_df["quality"] = derive_quality(pseudo_preds)

    # 4. Append to train.csv and save train_pseudo.csv
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    # Clean up train columns
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    if "Unnamed: 0" in train_df.columns:
        train_df = train_df.drop(columns=["Unnamed: 0"])

    augmented_train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
    augmented_path = os.path.join(data_dir, "train_pseudo.csv")
    augmented_train_df.to_csv(augmented_path, index=False)
    print(f"  [OK] Saved augmented training dataset with pseudo-labels to: {augmented_path}")
    print(f"       Original train samples: {len(train_df)}")
    print(f"       Added pseudo-labels:    {len(pseudo_df)}")
    print(f"       Total train_pseudo:     {len(augmented_train_df)}")


def _find_npz_dir(data_dir, split="train"):
    """Find the directory containing .npz files."""
    candidates = [
        os.path.join(data_dir, split),
        os.path.join(data_dir, split, "npz"),
        os.path.join(data_dir, "npz"),
        data_dir,
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d)[:10]):
            return d

    # Fallback: search recursively
    import glob
    npz_files = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
    if npz_files:
        return os.path.dirname(npz_files[0])

    raise FileNotFoundError(
        f"No .npz files found under {data_dir}. "
        f"Please verify the data extraction layout."
    )


def main():
    parser = argparse.ArgumentParser(
        description="SBER AI Journey — 3D Mesh Quality Control Pipeline"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "download", "features", "train", "infer", "all_no_download", "preprocess_images", "preprocess-images", "pseudo_label", "pseudo-label", "distill"],
        help="Pipeline mode: full (everything), download, features, train, infer, "
             "all_no_download (features+train+infer), preprocess_images (offline cropping), "
             "pseudo_label (self-training test pseudo-label generation), "
             "distill (train fast student model from soft teacher targets)",
    )
    parser.add_argument("--base-dir", "--base_dir", type=str, default=None,
                        help="Project base directory (auto-detected if not set)")
    parser.add_argument("--data-dir", "--data_dir", type=str, default=None,
                        help="Data directory (contains train.csv, test.csv, train/, test/)")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", type=str, default=None,
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--log-dir", "--log_dir", type=str, default=None,
                        help="Directory for training logs and CV results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for submission.csv")
    parser.add_argument("--folds", type=int, default=None,
                        help="Number of CV folds (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--pointnet", action="store_true", default=None,
                        help="Enable PointNet 3D branch (overrides config)")
    parser.add_argument("--extended-features", "--extended_features", action="store_true", default=None,
                        help="Enable extended 68-dim mesh features")
    parser.add_argument("--no-extended", action="store_true",
                        help="Use 58-dim features instead of 68-dim extended")
    parser.add_argument("--use-moe", "--use_moe", action="store_true", default=None,
                        help="Enable 3-expert Octopus MoE architecture")
    parser.add_argument("--no-moe", action="store_true",
                        help="Disable Octopus MoE architecture")
    parser.add_argument("--use-transformer", "--use_transformer", action="store_true", default=None,
                        help="Enable Cross-View Transformer Fusion (d=256)")
    parser.add_argument("--use-query-decoder", "--use_query_decoder", action="store_true", default=None,
                        help="Enable Defect Query Decoder")
    parser.add_argument("--use-normals", "--use_normals", action="store_true", default=None,
                        help="Enable 6-channel Sobel pseudo-normals")
    parser.add_argument("--use-co-attention", "--use_co_attention", action="store_true", default=None,
                        help="Enable Cross-Modal Co-Attention")
    parser.add_argument("--smoke-test", "--smoke_test", action="store_true", default=False,
                        help="Run fast 1-epoch 2-fold sanity check for Kaggle environment validation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--use-flash-attention", action="store_true", default=None,
                        help="Enable FlashAttention-2 Cross-Modal SDPA")
    parser.add_argument("--use-deepseek-mla", action="store_true", default=None,
                        help="Enable DeepSeek-V3 Multi-Head Latent Attention")
    parser.add_argument("--use-kimi-latent-memory", action="store_true", default=None,
                        help="Enable Kimi K1.5 Latent Memory Compressor")
    parser.add_argument("--use-glm-spatial-aligner", action="store_true", default=None,
                        help="Enable GLM-5.2 Image Spatial Aligner")
    parser.add_argument("--use-xai-router", action="store_true", default=None,
                        help="Enable xAI Grok-3 MoE Dynamic Gated Router")
    parser.add_argument("--use-flexible-effort", action="store_true", default=None,
                        help="Enable FlexibleThinkingEffortController")
    parser.add_argument("--use-kimi-dpo-loss", action="store_true", default=None,
                        help="Enable Kimi Quality Preference DPO Loss")
    parser.add_argument("--use-omni-route", action="store_true", default=None,
                        help="Enable OmniRoute Dynamic Path Dispatcher")
    parser.add_argument("--preprocess-images", "--preprocess_images", action="store_true", default=False,
                        help="Pre-crop and pre-resize all images offline (saves time)")

    args, unknown = parser.parse_known_args()

    # ── Resolve base directory ──────────────────────────────────────────────
    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # BUGFIX: Write back to args so resolve_paths() can use it
    # (previously args.base_dir stayed None, causing TypeError in os.path.join)
    args.base_dir = base_dir

    print("=" * 60)
    print("  SBER AI JOURNEY — 3D MESH QUALITY CONTROL v7.2 Master Engine")
    print("  Multi-label Defect Classification Pipeline")
    print("=" * 60)
    # ── Import config (must be before any config references below) ──────────
    import config

    print(f"  Base directory: {base_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Architecture: {'Octopus MoE (' + str(len(config.MOE_EXPERT_CONFIGS)) + ' experts)' if config.USE_MOE else 'Single backbone (' + config.IMAGE_BACKBONE + ')' }")

    # ── Override config if specified ────────────────────────────────────────
    if args.folds:
        config.NUM_FOLDS = args.folds
        print(f"  Overriding NUM_FOLDS = {args.folds}")
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
        print(f"  Overriding NUM_EPOCHS = {args.epochs}")
    if getattr(args, "batch_size", None):
        config.BATCH_SIZE = args.batch_size
        print(f"  Overriding BATCH_SIZE = {args.batch_size}")
    if getattr(args, "pointnet", None):
        config.USE_POINTNET_BRANCH = True
        print(f"  Enabling PointNet branch")
    if getattr(args, "use_moe", None):
        config.USE_MOE = True
        print(f"  Enabling Octopus MoE architecture")
    if getattr(args, "no_moe", None):
        config.USE_MOE = False
        print(f"  Disabling Octopus MoE architecture")
    if getattr(args, "use_transformer", None):
        config.USE_CROSS_VIEW_TRANSFORMER = True
        print(f"  Enabling Cross-View Transformer Fusion (d=256)")
    if getattr(args, "use_query_decoder", None):
        config.USE_DEFECT_QUERY_DECODER = True
        print(f"  Enabling Defect Query Decoder")
    if getattr(args, "use_normals", None):
        config.USE_GRADIENT_NORMALS = True
        print(f"  Enabling 6-Channel Sobel Pseudo-Normals")
    if getattr(args, "use_co_attention", None):
        config.USE_CROSS_MODAL_ATTENTION = True
        print(f"  Enabling Cross-Modal Co-Attention")
    if getattr(args, "use_flash_attention", None):
        config.USE_FLASH_ATTENTION = True
        print(f"  Enabling FlashAttention-2")
    if getattr(args, "use_deepseek_mla", None):
        config.USE_DEEPSEEK_MLA = True
        print(f"  Enabling DeepSeek-V3 MLA")
    if getattr(args, "use_kimi_latent_memory", None):
        config.USE_KIMI_LATENT_MEMORY = True
        print(f"  Enabling Kimi Latent Memory")
    if getattr(args, "use_glm_spatial_aligner", None):
        config.USE_GLM_SPATIAL_ALIGNER = True
        print(f"  Enabling GLM Spatial Aligner")
    if getattr(args, "use_xai_router", None):
        config.USE_XAI_ROUTER = True
        print(f"  Enabling xAI Grok-3 Router")
    if getattr(args, "use_flexible_effort", None):
        config.USE_FLEXIBLE_EFFORT = True
        print(f"  Enabling FlexibleThinking Effort Controller")
    if getattr(args, "use_kimi_dpo_loss", None):
        config.USE_KIMI_DPO_LOSS = True
        print(f"  Enabling Kimi DPO Loss")
    if getattr(args, "use_omni_route", None):
        config.USE_OMNI_ROUTE = True
        print(f"  Enabling OmniRoute")
    if getattr(args, "loss", None):
        config.LOSS_FUNCTION = args.loss
        print(f"  Overriding LOSS_FUNCTION = {args.loss}")
    if getattr(args, "extended_features", None):
        config.USE_EXTENDED_FEATURES = True
        print(f"  Enabling 100-dim extended features")
    if getattr(args, "no_extended", None):
        config.USE_EXTENDED_FEATURES = False
        print(f"  Using 58-dim basic features (extended disabled)")
    if getattr(args, "smoke_test", False):
        config.NUM_FOLDS = args.folds or 2
        config.NUM_EPOCHS = args.epochs or 1
        print(f"  [SMOKE TEST MODE ACTIVE] FOLDS={config.NUM_FOLDS}, EPOCHS={config.NUM_EPOCHS}")
    if getattr(args, "seed", None) is not None:
        config.SEED = args.seed
        print(f"  Overriding SEED = {args.seed}")

    # ── Resolve paths ──────────────────────────────────────────────────────
    base_dir, data_dir, checkpoint_dir, log_dir, submission_path = resolve_paths(args)

    # ── Ensure solution/ is on sys.path ─────────────────────────────────────
    solution_dir = os.path.join(base_dir, "solution")
    if solution_dir not in sys.path:
        sys.path.insert(0, solution_dir)

    # ── Short-circuit for offline image preprocessing if requested ──────────
    if getattr(args, "preprocess_images", False):
        from image_processing import preprocess_images_offline
        train_csv = os.path.join(data_dir, "train.csv")
        test_csv = os.path.join(data_dir, "test.csv")
        
        if os.path.isfile(train_csv):
            train_df = pd.read_csv(train_csv)
            train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
            train_ids = train_df["item_id"].tolist()
            train_img_dir = os.path.join(data_dir, "train")
            train_out_dir = os.path.join(data_dir, "train_tensors")
            print(f"Preprocessing train images ({len(train_ids)} items)...")
            preprocess_images_offline(train_img_dir, train_out_dir, train_ids, image_size=config.IMAGE_SIZE)
            
        if os.path.isfile(test_csv):
            test_df = pd.read_csv(test_csv)
            test_df = test_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
            test_ids = test_df["item_id"].tolist()
            test_img_dir = os.path.join(data_dir, "test")
            test_out_dir = os.path.join(data_dir, "test_tensors")
            print(f"Preprocessing test images ({len(test_ids)} items)...")
            preprocess_images_offline(test_img_dir, test_out_dir, test_ids, image_size=config.IMAGE_SIZE)
            
        print("[OK] Images preprocessed offline.")
        sys.exit(0)

    # ── Execute pipeline ───────────────────────────────────────────────────
    train_features = None
    test_features = None
    train_pcs = None
    test_pcs = None

    if args.mode == "full":
        step_download(data_dir, base_dir)
        train_features, test_features = step_extract_features(
            data_dir, base_dir, extended=config.USE_EXTENDED_FEATURES
        )
        if config.USE_POINTNET_BRANCH:
            train_pcs, test_pcs = step_extract_point_clouds(
                data_dir, base_dir, num_points=config.POINTNET_NUM_POINTS
            )
        step_train(train_features, data_dir, checkpoint_dir, log_dir, train_pcs)
        step_infer(test_features, data_dir, checkpoint_dir, log_dir, submission_path, test_pcs)

    elif args.mode == "download":
        step_download(data_dir, base_dir)

    elif args.mode == "features":
        train_features, test_features = step_extract_features(
            data_dir, base_dir, extended=config.USE_EXTENDED_FEATURES
        )
        if config.USE_POINTNET_BRANCH:
            train_pcs, test_pcs = step_extract_point_clouds(
                data_dir, base_dir, num_points=config.POINTNET_NUM_POINTS
            )

    elif args.mode == "train":
        if train_features is None:
            train_features, _ = step_extract_features(
                data_dir, base_dir, extended=config.USE_EXTENDED_FEATURES
            )
        if config.USE_POINTNET_BRANCH and train_pcs is None:
            train_pcs, _ = step_extract_point_clouds(
                data_dir, base_dir, num_points=config.POINTNET_NUM_POINTS
            )
        step_train(train_features, data_dir, checkpoint_dir, log_dir, train_pcs)

    elif args.mode == "infer":
        if test_features is None:
            _, test_features = step_extract_features(
                data_dir, base_dir, extended=config.USE_EXTENDED_FEATURES
            )
        if config.USE_POINTNET_BRANCH and test_pcs is None:
            _, test_pcs = step_extract_point_clouds(
                data_dir, base_dir, num_points=config.POINTNET_NUM_POINTS
            )
        step_infer(test_features, data_dir, checkpoint_dir, log_dir, submission_path, test_pcs)

    elif args.mode == "all_no_download":
        train_features, test_features = step_extract_features(
            data_dir, base_dir, extended=config.USE_EXTENDED_FEATURES
        )
        if config.USE_POINTNET_BRANCH:
            train_pcs, test_pcs = step_extract_point_clouds(
                data_dir, base_dir, num_points=config.POINTNET_NUM_POINTS
            )
        step_train(train_features, data_dir, checkpoint_dir, log_dir, train_pcs)
        step_infer(test_features, data_dir, checkpoint_dir, log_dir, submission_path, test_pcs)

    elif args.mode in ["preprocess_images", "preprocess-images"]:
        from image_processing import preprocess_images_offline
        
        train_csv = os.path.join(data_dir, "train.csv")
        test_csv = os.path.join(data_dir, "test.csv")
        
        if os.path.isfile(train_csv):
            train_df = pd.read_csv(train_csv)
            train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
            train_ids = train_df["item_id"].tolist()
            train_img_dir = os.path.join(data_dir, "train")
            train_out_dir = os.path.join(data_dir, "train_tensors")
            print(f"Preprocessing train images ({len(train_ids)} items)...")
            preprocess_images_offline(train_img_dir, train_out_dir, train_ids, image_size=config.IMAGE_SIZE)
            
        if os.path.isfile(test_csv):
            test_df = pd.read_csv(test_csv)
            test_df = test_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
            test_ids = test_df["item_id"].tolist()
            test_img_dir = os.path.join(data_dir, "test")
            test_out_dir = os.path.join(data_dir, "test_tensors")
            print(f"Preprocessing test images ({len(test_ids)} items)...")
            preprocess_images_offline(test_img_dir, test_out_dir, test_ids, image_size=config.IMAGE_SIZE)
            
        print("[OK] Offline image preprocessing complete.")

    elif args.mode in ["pseudo_label", "pseudo-label"]:
        if test_features is None:
            _, test_features = step_extract_features(
                data_dir, base_dir, extended=config.USE_EXTENDED_FEATURES
            )
        if config.USE_POINTNET_BRANCH and test_pcs is None:
            _, test_pcs = step_extract_point_clouds(
                data_dir, base_dir, num_points=config.POINTNET_NUM_POINTS
            )
        step_pseudo_label(test_features, data_dir, checkpoint_dir, log_dir, test_pcs)

    elif args.mode == "distill":
        train_csv = os.path.join(data_dir, "train.csv")
        train_df = pd.read_csv(train_csv)
        train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
        if "Unnamed: 0" in train_df.columns:
            train_df = train_df.drop(columns=["Unnamed: 0"])
            
        train_img_dir = os.path.join(data_dir, "train")
        if not os.path.isdir(train_img_dir):
            for candidate in [data_dir, os.path.join(data_dir, "train_images")]:
                if os.path.isdir(candidate) and any(f.endswith(".png") for f in os.listdir(candidate)[:5]):
                    train_img_dir = candidate
                    break
                    
        if train_features is None:
            train_features, _ = step_extract_features(
                data_dir, base_dir, extended=config.USE_EXTENDED_FEATURES
            )
            
        from distill import distill_student
        distill_student(
            train_df=train_df,
            image_dir=train_img_dir,
            mesh_features=train_features,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
