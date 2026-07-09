"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Inference Pipeline  [v7.2 Master Engine]
===============================================================================
Loads trained fold models, applies multi-modal Test-Time Augmentation (TTA),
averages fold probabilities, applies calibrated per-class thresholds,
derives quality labels, and generates competition submission.csv.
===============================================================================
"""
import config
import os
import json
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast as _autocast_new
    def safe_autocast(device_type="cuda", enabled=True):
        return _autocast_new(device_type=device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast_old
    def safe_autocast(device_type="cuda", enabled=True):
        return _autocast_old(enabled=enabled)

from config import (
    SEED, DEFECT_COLS, BATCH_SIZE, IMAGE_SIZE, VIEW_GRID,
    IMAGE_BACKBONE, IMAGE_EMBED_DIM, IMAGE_HIDDEN_DIM, IMAGE_DROPOUT,
    MESH_HIDDEN_DIMS, MESH_DROPOUT, MESH_FEATURE_DIM, MESH_FEATURE_DIM_EXTENDED,
    FUSION_METHOD, FUSION_IMAGE_WEIGHT, FUSION_MESH_WEIGHT,
    DEVICE, NUM_WORKERS, PIN_MEMORY, MIXED_PRECISION,
    USE_TTA, TTA_FLIPS, TTA_ROTATIONS, NUM_FOLDS,
    # v2.0 flags
    USE_POINTNET_BRANCH, POINTNET_NUM_POINTS, POINTNET_DROPOUT, POINTNET_WEIGHT,
    USE_EXTENDED_FEATURES, AUTO_DETECT_GRID, SEQUENTIAL_VIEW_PROCESSING,
    # v3.0 Octopus MoE flags
    USE_MOE, MOE_EXPERT_CONFIGS, MOE_TOP_K, MOE_ROUTER_HIDDEN_DIM,
    MOE_ROUTER_NOISE_STD, MOE_PROJECTION_DIM, SEQUENTIAL_VIEWS_IN_MOE,
)
from utils import set_seed, derive_quality, safe_collate
from image_processing import MeshQualityDataset, TTATransform
from models import (
    MultiViewImageModel, MeshFeatureMLP, FusedEnsembleModel,
    OctopusMoEModel,  # v3.0 Octopus MoE
)


# Cache whether we're in MoE mode (set once at import time)
_IS_MOE = USE_MOE


def build_model_for_inference(fold: int, input_mesh_dim: Optional[int] = None):
    """
    Build model architecture matching training configuration.
    v7.2: Calls unified build_model_from_config for 100% state-dict key matching.
    """
    if input_mesh_dim is not None:
        effective_mesh_dim = input_mesh_dim
    else:
        effective_mesh_dim = MESH_FEATURE_DIM_EXTENDED if USE_EXTENDED_FEATURES else MESH_FEATURE_DIM

    import config
    from models import build_model_from_config
    return build_model_from_config(cfg=config, effective_mesh_dim=effective_mesh_dim)


def inference_with_tta(
    model: nn.Module,  # FusedEnsembleModel or OctopusMoEModel
    data_loader: DataLoader,
    device: str = DEVICE,
    use_tta: bool = USE_TTA,
    temperature: float = 1.0,
    effort: str = "max",
) -> np.ndarray:
    """
    Run inference with optional TTA and temperature scaling.

    v2.1: Added temperature parameter for calibrated probabilities.

    Args:
        temperature: divide logits by this value before sigmoid.
                      T > 1 = softer probs, T < 1 = sharper probs.

    For each batch, if TTA is enabled:
      - Apply all TTA transforms to get multiple predictions
      - Average the predictions across TTA variants

    v2.0: Handles PointNet point clouds in batch data.
    """
    device_type = "cuda" if "cuda" in device else "cpu"
    model.eval()
    tta = TTATransform(flips=TTA_FLIPS, rotations=TTA_ROTATIONS) if use_tta else TTATransform(flips=[False], rotations=[0])
    n_tta = len(tta)

    all_proba = []

    with torch.no_grad():
        for batch in data_loader:
            views = batch["views"].to(device)  # (B, V, 3, H, W)
            mesh_feat = batch["mesh_features"]
            if mesh_feat is not None:
                mesh_feat = mesh_feat.to(device)
            pc = batch.get("point_cloud")
            if pc is not None:
                pc = pc.to(device)

            # Apply TTA transforms
            tta_views_list = tta.apply(views)

            is_moe_active = getattr(config, "USE_MOE", False) or hasattr(model, "forward_simple")
            batch_logits = []
            for tta_views in tta_views_list:
                with safe_autocast(device_type=device_type, enabled=MIXED_PRECISION):
                    if is_moe_active:
                        try:
                            logits = model.forward_simple(tta_views, mesh_feat, pc, effort=effort)
                        except TypeError:
                            logits = model.forward_simple(tta_views, mesh_feat, pc)
                    else:
                        try:
                            logits = model(tta_views, mesh_feat, pc, effort=effort)
                        except TypeError:
                            logits = model(tta_views, mesh_feat, pc)
                    if temperature != 1.0:
                        logits = logits / temperature
                batch_logits.append(logits)

            # P1-17 FIX: Average logits in logit space before sigmoid
            avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
            avg_proba = torch.sigmoid(avg_logits)
            all_proba.append(avg_proba.cpu().numpy())

    return np.concatenate(all_proba, axis=0)


def ensemble_inference(
    test_ids: list,
    test_image_dir: str,
    checkpoint_dir: str,
    thresholds: np.ndarray = None,
    mesh_features: np.ndarray = None,
    point_clouds: np.ndarray = None,
    cv_results_path: str = None,
    folds_to_use: list = None,
    strict_loading: bool = True,
    effort: str = "max",
) -> pd.DataFrame:
    """
    Run ensemble inference across all trained folds.

    v2.0: Supports PointNet point clouds, extended features, auto-detect grid.
    v2.1: Loads temperature from cv_results for calibrated inference.
    v3.1: Enforces strict model state dict verification for security & integrity.

    Args:
        test_ids: list of test item_ids
        test_image_dir: directory with test PNG files
        checkpoint_dir: directory with fold checkpoint .pt files
        thresholds: (10,) per-class thresholds (None = use 0.5)
        mesh_features: (N_test, D) pre-extracted mesh features
        point_clouds: (N_test, P, 3) pre-extracted point clouds (Limitation #1)
        cv_results_path: path to cv_results.json (for loading thresholds)
        folds_to_use: which folds to use (default: all NUM_FOLDS)
        strict_loading: whether to enforce strict checkpoint state dict key matching

    Returns:
        Tuple of (submission_df, proba_df)
    """
    set_seed(SEED)

    # ── Load thresholds and temperature ───────────────────────────────────
    temperature = 1.0
    if cv_results_path is not None and os.path.isfile(cv_results_path):
        try:
            with open(cv_results_path, "r") as f:
                cv_results = json.load(f)
            
            # Defensive validation of loaded JSON content
            if not isinstance(cv_results, dict):
                raise ValueError("CV results must be a JSON object")
            
            if thresholds is None:
                raw_thresh = cv_results.get("avg_thresholds")
                if not isinstance(raw_thresh, list) or len(raw_thresh) != len(DEFECT_COLS):
                    raise ValueError(f"avg_thresholds must be a list of length {len(DEFECT_COLS)}")
                for val in raw_thresh:
                    if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val) or val < 0.0 or val > 1.0:
                        raise ValueError(f"Invalid threshold value: {val}")
                thresholds = np.array(raw_thresh)
                print(f"Loaded thresholds from CV results: {dict(zip(DEFECT_COLS, thresholds.round(3)))}")
            
            # Validate temperature
            raw_temp = cv_results.get("avg_temperature", 1.0)
            if not isinstance(raw_temp, (int, float)) or np.isnan(raw_temp) or np.isinf(raw_temp) or raw_temp < 0.01 or raw_temp > 10.0:
                raise ValueError(f"Invalid temperature value: {raw_temp}")
            temperature = float(raw_temp)
            if temperature != 1.0:
                print(f"Using temperature scaling: T={temperature:.3f}")
        except Exception as e:
            print(f"  [WARNING] Failed to load/validate CV results: {e} — falling back to defaults")
            if thresholds is None:
                thresholds = np.full(len(DEFECT_COLS), 0.5)
            temperature = 1.0
    elif thresholds is None:
        thresholds = np.full(len(DEFECT_COLS), 0.5)
        print("Using default threshold 0.5 for all classes")

    # ── Grid for view splitting (Limitation #3) ───────────────────────────
    view_grid = None if AUTO_DETECT_GRID else VIEW_GRID

    # ── Build test dataset & loader ────────────────────────────────────────
    # NOTE: No view subsampling at inference — always all 6 views
    test_dataset = MeshQualityDataset(
        item_ids=test_ids,
        labels_df=None,
        image_dir=test_image_dir,
        mesh_features=mesh_features,
        point_clouds=point_clouds,
        image_size=IMAGE_SIZE,
        view_grid=view_grid,
        augment=False,
        views_subsample=None,  # Always all views at inference
    )

    import config as cfg
    num_workers = getattr(cfg, "NUM_WORKERS", NUM_WORKERS)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=safe_collate,
    )

    # ── Ensemble across folds ──────────────────────────────────────────────
    if folds_to_use is None:
        folds_to_use = list(range(NUM_FOLDS))

    # Pre-check: Ensure at least one checkpoint file exists in checkpoint_dir
    import glob
    existing_ckpts = glob.glob(os.path.join(checkpoint_dir, "best_fold*.pt"))
    if not existing_ckpts:
        raise RuntimeError(
            f"No fold checkpoints matching 'best_fold*.pt' found in directory '{checkpoint_dir}'. "
            f"At least one fold checkpoint must exist for ensemble inference."
        )

    fold_proba_list = []

    for fold in folds_to_use:
        checkpoint_path = os.path.join(checkpoint_dir, f"best_fold{fold}.pt")

        if not os.path.exists(checkpoint_path):
            if strict_loading:
                raise FileNotFoundError(
                    f"[SECURITY & INTEGRITY FATAL] Checkpoint not found for fold {fold}: {checkpoint_path}"
                )
            print(f"  Warning: Checkpoint not found for fold {fold}, skipping")
            continue

        print(f"  Loading fold {fold} model from {checkpoint_path}")

        input_mesh_dim = mesh_features.shape[1] if mesh_features is not None and hasattr(mesh_features, "shape") and len(mesh_features.shape) > 1 else None
        model = build_model_for_inference(fold, input_mesh_dim=input_mesh_dim).to(DEVICE)

        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        try:
            model.load_state_dict(state_dict, strict=strict_loading)
        except RuntimeError as err:
            if strict_loading:
                raise RuntimeError(
                    f"[SECURITY & INTEGRITY FATAL] Key mismatch loading checkpoint fold {fold} ({checkpoint_path}): {err}. "
                    f"Set strict_loading=False only if explicitly loading legacy/modified checkpoints."
                ) from err
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"    [WARNING] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"    [WARNING] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        fold_proba = inference_with_tta(model, test_loader, DEVICE, USE_TTA, temperature=temperature, effort=effort)
        fold_proba_list.append(fold_proba)

        # Free memory
        del model
        if "cuda" in DEVICE:
            torch.cuda.empty_cache()

    if not fold_proba_list:
        raise RuntimeError("No valid fold checkpoints found!")

    # ── Average predictions across folds ────────────────────────────────────
    ensemble_proba = np.mean(fold_proba_list, axis=0)
    print(f"\n  Ensemble of {len(fold_proba_list)} folds")
    print(f"  Prediction shape: {ensemble_proba.shape}")

    # ── Apply thresholds ───────────────────────────────────────────────────
    predictions = (ensemble_proba >= thresholds).astype(int)

    # ── Derive quality ─────────────────────────────────────────────────────
    quality = derive_quality(predictions)

    # ── Build submission DataFrame ─────────────────────────────────────────
    submission = pd.DataFrame(predictions, columns=DEFECT_COLS)
    submission.insert(0, "item_id", test_ids)
    submission["quality"] = quality

    # Also save probabilities for analysis
    proba_df = pd.DataFrame(ensemble_proba, columns=DEFECT_COLS)
    proba_df.insert(0, "item_id", test_ids)

    print(f"\n  Prediction statistics:")
    print(f"    Total samples: {len(submission)}")
    print(f"    Good (quality=1): {quality.sum()} ({quality.mean()*100:.1f}%)")
    print(f"    Bad (quality=0): {len(quality) - quality.sum()} ({(1-quality.mean())*100:.1f}%)")
    for col in DEFECT_COLS:
        pos = predictions[:, DEFECT_COLS.index(col)].sum()
        print(f"    {col:15s}: {pos} ({pos/len(predictions)*100:.1f}%)")

    return submission, proba_df


def generate_submission(
    test_csv_path: str,
    test_image_dir: str,
    checkpoint_dir: str,
    output_path: str,
    cv_results_path: str = None,
    mesh_features: np.ndarray = None,
    point_clouds: np.ndarray = None,
    effort: str = "max",
):
    """
    Complete inference pipeline: load test IDs, run ensemble, save submission.

    v2.0: Accepts point_clouds for optional PointNet branch.
    """
    test_df = pd.read_csv(test_csv_path)
    # Strip "OUTPUT:" prefix from columns if present
    test_df = test_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    if "Unnamed: 0" in test_df.columns:
        test_df = test_df.drop(columns=["Unnamed: 0"])
    test_ids = test_df["item_id"].tolist()

    submission, proba_df = ensemble_inference(
        test_ids=test_ids,
        test_image_dir=test_image_dir,
        checkpoint_dir=checkpoint_dir,
        mesh_features=mesh_features,
        point_clouds=point_clouds,
        cv_results_path=cv_results_path,
        effort=effort,
    )

    # Ensure column order matches expected format
    col_order = ["item_id"] + DEFECT_COLS + ["quality"]
    submission = submission[col_order]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\n  Submission saved to: {output_path}")

    # Save probabilities for analysis
    proba_path = output_path.replace(".csv", "_proba.csv")
    proba_df.to_csv(proba_path, index=False)
    print(f"  Probabilities saved to: {proba_path}")

    return submission
