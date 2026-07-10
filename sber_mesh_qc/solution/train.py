"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Training Pipeline  [v7.1 Master Engine]
===============================================================================
Complete training pipeline with:
  - Stratified K-Fold cross-validation
  - Mixed precision (AMP) training
  - Cosine annealing with warmup
  - Early stopping
  - Best model checkpointing & EMA weight smoothing
  - Per-class threshold & temperature calibration
  - Frontier Adaptations: xAI Grok-3 MoE, GLM-5.2 IndexShare, Kimi K1.5 DPO, OmniRoute

v7.1 CHANGES (OmniRoute & Moonshot AI):
  - KimiLatentMemoryCompressor: 16 compact latent memory slots (4x memory reduction)
  - KimiQualityPreferenceLoss: Margin-based DPO ranking loss for score separation
  - OmniRoutePathDispatcher: Modality entropy H(M) dynamic branch path routing

v6.9 CHANGES (xAI & GLM-5.2):
  - xAIMoEHybridRouter: Top-2 expert selection with auxiliary load-balancing loss
  - IndexShareCrossModalAttention: 4:1 query/key index map sharing (2.9x speedup)
  - FlexibleThinkingEffortController: reasoning_effort ("fast", "high", "max")

v2.0-v6.5 CHANGES (preserved):
  - Progressive resize, gradient checkpointing, sequential view processing
  - PointNet 3D branch, 100-dim extended mesh feature suite (SHTD + Betti)
===============================================================================
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as T

# Use new-style AMP (torch.amp) with fallback for older PyTorch
try:
    from torch.amp import autocast as _autocast_new, GradScaler
    def safe_autocast(device_type="cuda", enabled=True):
        return _autocast_new(device_type=device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler
    def safe_autocast(device_type="cuda", enabled=True):
        # Old API doesn't accept device_type
        return _autocast_old(enabled=enabled)

# Compatibility: PyTorch 2.2+ renamed _LRScheduler → LRScheduler
try:
    _LRSchedulerBase = torch.optim.lr_scheduler.LRScheduler
except AttributeError:
    _LRSchedulerBase = torch.optim.lr_scheduler._LRScheduler

from config import (
    SEED, DEFECT_COLS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, GRADIENT_CLIP, PATIENCE, MIN_LR,
    USE_AUGMENTATION, AUG_HORIZONTAL_FLIP, AUG_ROTATION,
    AUG_COLOR_JITTER, AUG_RANDOM_ERASING,
    NUM_FOLDS, OPTIMIZE_THRESHOLDS, IMAGE_SIZE, VIEW_GRID,
    LOSS_FUNCTION, FOCAL_GAMMA, LABEL_SMOOTHING,
    USE_DYNAMIC_CLASS_WEIGHTS, IMAGE_BACKBONE, IMAGE_PRETRAINED,
    IMAGE_EMBED_DIM, IMAGE_HIDDEN_DIM, IMAGE_DROPOUT,
    MESH_HIDDEN_DIMS, MESH_DROPOUT, FUSION_METHOD,
    FUSION_IMAGE_WEIGHT, FUSION_MESH_WEIGHT,
    DEVICE, NUM_WORKERS, PIN_MEMORY, MIXED_PRECISION,
    WARMUP_EPOCHS, SAVE_BEST_MODEL, SAVE_LAST_MODEL,
    LOG_INTERVAL,
    # v2.0 imports — all 5 limitation flags
    GRADIENT_ACCUM_STEPS,          # Was hardcoded before — now from config
    PROGRESSIVE_RESIZE,            # Limitation #4
    PROGRESSIVE_SCHEDULE,          # Limitation #4
    USE_GRADIENT_CHECKPOINTING,    # Limitation #5
    SEQUENTIAL_VIEW_PROCESSING,    # Limitation #5
    VIEWS_TRAIN_SUBSAMPLE,         # Limitation #5
    USE_POINTNET_BRANCH,           # Limitation #1
    POINTNET_NUM_POINTS,           # Limitation #1
    POINTNET_DROPOUT,              # Limitation #1
    POINTNET_WEIGHT,               # Limitation #1
    USE_EXTENDED_FEATURES,         # Limitation #2
    MESH_FEATURE_DIM,              # Basic dim (58)
    MESH_FEATURE_DIM_EXTENDED,     # Extended dim (68)
    AUTO_DETECT_GRID,              # Limitation #3
    # v2.1 imports — score-boosting tricks
    USE_EMA,                       # Exponential Moving Average
    EMA_DECAY,                     # EMA decay rate
    USE_MIXUP,                     # Multi-label mixup
    MIXUP_ALPHA,                   # Mixup Beta distribution alpha
    MIXUP_PROB,                    # Probability of applying mixup
    MIXUP_LABEL_SMOOTH,            # Extra smoothing for mixed labels
    USE_TEMPERATURE_SCALING,       # Temperature calibration
    TEMPERATURE_INIT,              # Initial temperature
    TEMPERATURE_LR,                # Temperature learning rate
    OPTIMIZE_THRESHOLDS_F1_FINAL,  # Optimize full f1_final metric
    USE_SWA,                       # Stochastic Weight Averaging
    SWA_START_EPOCH,               # SWA start epoch
    SWA_LR,                        # SWA learning rate
    # v3.0 imports — Octopus MoE
    USE_MOE,                       # Enable MoE architecture
    MOE_EXPERT_CONFIGS,            # Expert backbone configurations
    MOE_TOP_K,                     # Top-K expert routing
    MOE_ROUTER_HIDDEN_DIM,         # Router MLP hidden dim
    MOE_ROUTER_NOISE_STD,          # Router noise for exploration
    MOE_LOAD_BALANCE_COEFF,        # Load-balancing loss coefficient
    MOE_ROUTER_LR_FACTOR,          # Router LR multiplier
    MOE_PROJECTION_DIM,            # Common projection dim
    SEQUENTIAL_VIEWS_IN_MOE,       # Sequential view processing in MoE
)
from utils import set_seed, compute_f1_final, optimize_thresholds, derive_quality, safe_collate, clean_state_dict_keys
from utils import optimize_thresholds_f1_final, learn_temperature
from config import get_class_weights
from image_processing import MeshQualityDataset
from models import (
    MultiViewImageModel, MeshFeatureMLP, FusedEnsembleModel,
    OctopusMoEModel,  # v3.0 Octopus MoE
)
from losses import build_loss_function


# ═══════════════════════════════════════════════════════════════════════════
# EMA — Exponential Moving Average of Model Weights (v2.1)
# ═══════════════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model's parameters that are updated
    as: shadow = decay * shadow + (1 - decay) * current

    At evaluation time, applying the shadow weights typically yields
    better generalization because it smooths out training noise.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # After each optimizer step:
        ema.update(model)
        # Before validation:
        backup = ema.apply_shadow(model)
        # ... validate ...
        ema.restore(model, backup)
        # Or save EMA weights:
        ema.apply_shadow(model)
        torch.save(model.state_dict(), "best.pt")
        ema.restore(model, backup)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.buffers = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buf in model.named_buffers():
            self.buffers[name] = buf.data.clone()

    def update(self, model: nn.Module):
        """Update shadow weights and BatchNorm buffers after an optimizer step."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
            for name, buf in model.named_buffers():
                if name in self.buffers:
                    self.buffers[name].copy_(buf.data)

    def apply_shadow(self, model: nn.Module) -> dict:
        """
        Apply shadow weights and buffers to the model for evaluation.
        Returns a backup of original weights and buffers for restoration.
        """
        backup = {"params": {}, "buffers": {}}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup["params"][name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            if name in self.buffers:
                backup["buffers"][name] = buf.data.clone()
                buf.data.copy_(self.buffers[name])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        """Restore original model weights and buffers from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup.get("params", {}):
                param.data.copy_(backup["params"][name])
        for name, buf in model.named_buffers():
            if name in backup.get("buffers", {}):
                buf.data.copy_(backup["buffers"][name])

    def state_dict(self) -> dict:
        """Return shadow weights and buffers as a state dict (for saving)."""
        return {"shadow": self.shadow, "buffers": self.buffers, "decay": self.decay}

    def load_state_dict(self, state_dict: dict):
        """Load shadow weights and buffers from a state dict safely."""
        self.decay = state_dict.get("decay", self.decay)
        self.shadow = state_dict.get("shadow", self.shadow)
        self.buffers = state_dict.get("buffers", self.buffers)


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-LABEL MIXUP (v2.1)
# ═══════════════════════════════════════════════════════════════════════════

def _multilabel_mixup(
    views: torch.Tensor,
    labels: torch.Tensor,
    mesh_feat: Optional[torch.Tensor] = None,
    pc: Optional[torch.Tensor] = None,
    alpha: float = 0.2,
    label_smooth: float = 0.1,
) -> tuple:
    """
    Apply multi-modal multi-label mixup augmentation across images, labels, and geometry.

    Linearly interpolates images, mesh features, and point clouds using lambda,
    while applying multi-label union mixing with label smoothing to target labels.

    Args:
        views: (B, V, 3, H, W) multi-view images
        labels: (B, C) binary multi-label targets
        mesh_feat: Optional (B, D) geometric mesh features
        pc: Optional (B, P, 3) point cloud tensors
        alpha: Beta distribution parameter for lambda sampling
        label_smooth: additional smoothing for mixed labels

    Returns:
        (mixed_views, mixed_labels, mixed_mesh_feat, mixed_pc) tuple
    """
    if alpha <= 0:
        return views, labels, mesh_feat, pc

    B = views.size(0)
    # Sample lambda from Beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lambda >= 0.5 (primary sample dominates)

    # Random permutation
    index = torch.randperm(B, device=views.device)

    # Mix images
    mixed_views = lam * views + (1 - lam) * views[index]

    # Mix geometry branches with the exact same lambda to prevent cross-modal mismatch
    mixed_mesh_feat = None
    if mesh_feat is not None:
        mixed_mesh_feat = lam * mesh_feat + (1 - lam) * mesh_feat[index]

    mixed_pc = None
    if pc is not None:
        mixed_pc = lam * pc + (1 - lam) * pc[index]

    # Mix labels: Asymmetric Interpolation (Phase 2)
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    mixed_labels = torch.where(mixed_labels > 0.5, torch.ones_like(mixed_labels), mixed_labels)
    if label_smooth > 0:
        mixed_labels = mixed_labels * (1.0 - label_smooth) + label_smooth * 0.0

    return mixed_views, mixed_labels, mixed_mesh_feat, mixed_pc


# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════

class CosineAnnealingWarmupRestarts(_LRSchedulerBase):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 *
                (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: get current image size for progressive resize
# ═══════════════════════════════════════════════════════════════════════════

def _get_image_size(epoch: int) -> int:
    """
    OVERCOME LIMITATION #4: Progressive resize schedule.
    Returns the image size to use at a given epoch.
    """
    if not PROGRESSIVE_RESIZE:
        return IMAGE_SIZE

    current_size = IMAGE_SIZE  # Default
    for threshold_epoch, size in sorted(PROGRESSIVE_SCHEDULE.items()):
        if epoch < threshold_epoch:
            return current_size
        current_size = size
    return current_size


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE FOLD TRAINER
# ═══════════════════════════════════════════════════════════════════════════

def train_one_fold(
    fold: int,
    train_ids: list,
    val_ids: list,
    train_df: pd.DataFrame,
    image_dir: str,
    mesh_feat_train: np.ndarray = None,
    mesh_feat_val: np.ndarray = None,
    pc_train: np.ndarray = None,       # Point clouds (Limitation #1)
    pc_val: np.ndarray = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
) -> dict:
    """
    Train a single CV fold.

    v3.0: Supports OctopusMoEModel (multi-backbone MoE) when USE_MOE=True.
    v2.0: Fully integrates all 5 limitation solutions.
    v2.1: Adds EMA, mixup, quality-aware thresholds, temperature scaling.
    """
    is_moe = USE_MOE  # Cache for fast access in training loop

    print(f"\n{'='*60}")
    print(f"  FOLD {fold + 1}/{NUM_FOLDS}")
    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)}")
    print(f"  Architecture: {'Octopus MoE (' + str(len(MOE_EXPERT_CONFIGS)) + ' experts)' if is_moe else 'Single backbone (' + IMAGE_BACKBONE + ')'}")
    print(f"  PointNet: {USE_POINTNET_BRANCH} | Extended features: {USE_EXTENDED_FEATURES}")
    print(f"  Progressive resize: {PROGRESSIVE_RESIZE} | Grad checkpoint: {USE_GRADIENT_CHECKPOINTING}")
    print(f"  Sequential views: {SEQUENTIAL_VIEW_PROCESSING} | View subsample: {VIEWS_TRAIN_SUBSAMPLE}")
    print(f"  Auto-detect grid: {AUTO_DETECT_GRID}")
    print(f"  EMA: {USE_EMA} (decay={EMA_DECAY}) | Mixup: {USE_MIXUP}")
    print(f"  Quality-aware thresholds: {OPTIMIZE_THRESHOLDS_F1_FINAL} | Temp scaling: {USE_TEMPERATURE_SCALING}")
    if is_moe:
        print(f"  MoE Top-K: {MOE_TOP_K} | Router LR factor: {MOE_ROUTER_LR_FACTOR} | Load bal coeff: {MOE_LOAD_BALANCE_COEFF}")
    print(f"{'='*60}")

    set_seed(SEED + fold)

    # ── Determine effective feature dimension dynamically ─────────────────
    if mesh_feat_train is not None and hasattr(mesh_feat_train, "shape") and len(mesh_feat_train.shape) > 1:
        effective_mesh_dim = mesh_feat_train.shape[1]
    else:
        effective_mesh_dim = MESH_FEATURE_DIM_EXTENDED if USE_EXTENDED_FEATURES else MESH_FEATURE_DIM

    # Fit scaler on training features only (H6)
    from mesh_features import StandardScaler3D
    scaler_3d = StandardScaler3D()
    if mesh_feat_train is not None:
        mesh_feat_train = scaler_3d.fit_transform(mesh_feat_train)
        if mesh_feat_val is not None:
            mesh_feat_val = scaler_3d.transform(mesh_feat_val)
    else:
        scaler_3d = None

    # ── Prepare labels by filtering the FULL dataframe ────────────────────
    train_indexed = train_df.set_index("item_id")
    # Cast index to string to prevent type mismatch (str vs int)
    train_indexed.index = train_indexed.index.astype(str)
    
    # Ensure all input IDs are stringified
    train_ids = [str(i) for i in train_ids]
    val_ids = [str(i) for i in val_ids]

    valid_train_ids = [i for i in train_ids if i in train_indexed.index]
    valid_val_ids = [i for i in val_ids if i in train_indexed.index]

    train_labels = train_indexed.loc[valid_train_ids].reset_index()
    val_labels = train_indexed.loc[valid_val_ids].reset_index()

    # Update IDs to only include valid ones
    if len(valid_train_ids) != len(train_ids):
        print(f"  [WARNING] {len(train_ids) - len(valid_train_ids)} train IDs not found in CSV")
        train_ids = valid_train_ids
    if len(valid_val_ids) != len(val_ids):
        print(f"  [WARNING] {len(val_ids) - len(valid_val_ids)} val IDs not found in CSV")
        val_ids = valid_val_ids

    # ── Compute class weights ──────────────────────────────────────────────
    if USE_DYNAMIC_CLASS_WEIGHTS:
        weights_dict = get_class_weights(train_labels)
        class_weights = np.array([weights_dict[c] for c in DEFECT_COLS])
        print(f"  Class weights: {dict(zip(DEFECT_COLS, class_weights.round(2)))}")
    else:
        class_weights = None

    # ── Grid for view splitting (Limitation #3) ───────────────────────────
    view_grid = None if AUTO_DETECT_GRID else VIEW_GRID

    # ── Augmentation config ────────────────────────────────────────────────
    aug_config = {
        "horizontal_flip": AUG_HORIZONTAL_FLIP,
        "rotation": AUG_ROTATION,
        "color_jitter": AUG_COLOR_JITTER,
        "random_erasing": AUG_RANDOM_ERASING,
    } if USE_AUGMENTATION else {}

    # ── Build datasets (initial image size for progressive resize) ────────
    initial_img_size = _get_image_size(0)
    print(f"  Initial image size: {initial_img_size}px")

    import config as cfg
    use_kornia = getattr(cfg, "USE_KORNIA", False)
    dataset_augment = USE_AUGMENTATION and not use_kornia

    train_dataset = MeshQualityDataset(
        item_ids=train_ids,
        labels_df=train_labels,
        image_dir=image_dir,
        mesh_features=mesh_feat_train,
        point_clouds=pc_train,
        image_size=initial_img_size,
        view_grid=view_grid,
        augment=dataset_augment,
        aug_config=aug_config,
        views_subsample=VIEWS_TRAIN_SUBSAMPLE,
    )

    val_dataset = MeshQualityDataset(
        item_ids=val_ids,
        labels_df=val_labels,
        image_dir=image_dir,
        mesh_features=mesh_feat_val,
        point_clouds=pc_val,
        image_size=IMAGE_SIZE,  # Always full size for validation
        view_grid=view_grid,
        augment=False,
    )

    import config as cfg
    num_workers = getattr(cfg, "NUM_WORKERS", NUM_WORKERS)
    effective_workers = min(num_workers, 8) if DEVICE == "cuda" else num_workers

    # v2.0.1 FIX: worker_init_fn ensures reproducible augmentation across workers.
    # Without this, each DataLoader worker inherits a different random state,
    # making results non-reproducible when num_workers > 0.
    def _worker_init_fn(worker_id):
        worker_seed = SEED + fold * 1000 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=effective_workers, pin_memory=PIN_MEMORY, drop_last=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(effective_workers > 0),
        prefetch_factor=2 if effective_workers > 0 else None,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=effective_workers, pin_memory=PIN_MEMORY,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(effective_workers > 0),
        prefetch_factor=2 if effective_workers > 0 else None,
        collate_fn=safe_collate,
    )

    # ── Build model ────────────────────────────────────────────────────────
    import config as _cfg
    from models import build_model_from_config
    model = build_model_from_config(cfg=_cfg, effective_mesh_dim=effective_mesh_dim).to(DEVICE)

    # ── Loss, optimizer, scheduler ─────────────────────────────────────────
    criterion = build_loss_function(
        loss_name=LOSS_FUNCTION,
        class_weights=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        focal_gamma=FOCAL_GAMMA,
    ).to(DEVICE)

    # P2 FIX: Separate parameters to exclude 1D biases and BatchNorm/LayerNorm weights from weight decay
    decay_params = []
    no_decay_params = []
    router_decay = []
    router_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_1d = param.ndim <= 1 or name.endswith(".bias") or "bn" in name or "norm" in name
        if is_moe and ("router" in name or "summary_encoder" in name):
            if is_1d:
                router_no_decay.append(param)
            else:
                router_decay.append(param)
        else:
            if is_1d:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    if is_moe:
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "name": "experts_decay"},
            {"params": no_decay_params, "lr": LEARNING_RATE, "weight_decay": 0.0, "name": "experts_no_decay"},
            {"params": router_decay, "lr": LEARNING_RATE * MOE_ROUTER_LR_FACTOR, "weight_decay": WEIGHT_DECAY, "name": "router_decay"},
            {"params": router_no_decay, "lr": LEARNING_RATE * MOE_ROUTER_LR_FACTOR, "weight_decay": 0.0, "name": "router_no_decay"},
        ])
        print(f"  Dual LR: experts={LEARNING_RATE:.2e}, router={LEARNING_RATE * MOE_ROUTER_LR_FACTOR:.2e} (1D params excluded from weight decay)")
    else:
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "lr": LEARNING_RATE, "weight_decay": 0.0},
        ])

    use_onecycle = getattr(_cfg, "USE_ONECYCLE", False)
    if use_onecycle:
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            steps_per_epoch=steps_per_epoch,
            epochs=NUM_EPOCHS,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1000.0,
        )
        print("  [Scheduler] Initialized OneCycleLR scheduler.")
    else:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            warmup_epochs=WARMUP_EPOCHS,
            total_epochs=NUM_EPOCHS,
            min_lr=MIN_LR,
        )

    # ── Mixed precision ────────────────────────────────────────────────────
    device_type = "cuda" if "cuda" in DEVICE else "cpu"
    # v2.1.1 FIX: GradScaler device_type param only exists in PyTorch 2.0+
    try:
        scaler = GradScaler(device_type, enabled=MIXED_PRECISION)
    except TypeError:
        scaler = GradScaler(enabled=MIXED_PRECISION)

    # ── EMA (v2.1) ────────────────────────────────────────────────────────
    ema = None
    if USE_EMA:
        ema = ModelEMA(model, decay=EMA_DECAY)
        print(f"  EMA initialized with decay={EMA_DECAY}")

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_score = -1
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_f1": [], "val_f1_final": [], "lr": []}

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, f"best_fold{fold}.pt")
    last_path = os.path.join(checkpoint_dir, f"last_fold{fold}.pt")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # ── Progressive resize (Limitation #4) ─────────────────────────────
        current_img_size = _get_image_size(epoch)
        if current_img_size != train_dataset.image_size:
            print(f"  [Progressive Resize] Epoch {epoch+1}: {train_dataset.image_size}px -> {current_img_size}px")
            train_dataset.image_size = current_img_size
            train_dataset.resize_transform = T.Resize((current_img_size, current_img_size))

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0
        train_batches = 0
        accum_window_count = 0
        optimizer.zero_grad()

        gpu_augmenter = None
        if use_kornia:
            try:
                from image_processing import KorniaGPUAugmentation
                gpu_augmenter = KorniaGPUAugmentation(aug_config).to(DEVICE)
                print("  [Kornia GPU Augmentation] Enabled and active.")
            except ImportError:
                print("  [WARNING] kornia is not installed — falling back to CPU/no augmentations")

        # v2.1.1 FIX: batch_idx initialized to avoid UnboundLocalError
        # if loader is empty (fold size < BATCH_SIZE with drop_last=True)
        batch_idx = -1
        for batch_idx, batch in enumerate(train_loader):
            views = batch["views"].to(DEVICE)
            if gpu_augmenter is not None:
                views = gpu_augmenter(views)
            labels = batch["labels"].to(DEVICE)
            mesh_feat = (
                batch["mesh_features"].to(DEVICE)
                if batch["mesh_features"] is not None else None
            )
            pc = (
                batch["point_cloud"].to(DEVICE)
                if batch.get("point_cloud") is not None else None
            )

            # ── Multi-label mixup (v2.1) ───────────────────────────────────
            if USE_MIXUP and np.random.random() < MIXUP_PROB:
                views, labels, mesh_feat, pc = _multilabel_mixup(
                    views, labels,
                    mesh_feat=mesh_feat,
                    pc=pc,
                    alpha=MIXUP_ALPHA,
                    label_smooth=MIXUP_LABEL_SMOOTH,
                )

            with safe_autocast(device_type=device_type, enabled=MIXED_PRECISION):
                # Gradient checkpointing on image backbone (Limitation #5)
                # Only applies to single-backbone model (MoE handles its own memory)
                if USE_GRADIENT_CHECKPOINTING and model.training and not is_moe:
                    from models import AgenticEnsembleModel
                    raw_model = model.base_model if isinstance(model, AgenticEnsembleModel) else model
                    from torch.utils.checkpoint import checkpoint
                    original_forward = raw_model.image_model._extract_view_features
                    def checkpointed_forward(view_batch):
                        return checkpoint(original_forward, view_batch, use_reentrant=False)
                    raw_model.image_model._extract_view_features = checkpointed_forward
                    try:
                        if is_moe:
                            logits, aux_info = model(views, mesh_feat, pc)
                        else:
                            logits = model(views, mesh_feat, pc)
                    finally:
                        raw_model.image_model._extract_view_features = original_forward
                else:
                    if is_moe:
                        logits, aux_info = model(views, mesh_feat, pc)
                    else:
                        logits = model(views, mesh_feat, pc)

                # OHEM: Compute per-sample loss, keep only top 30% hardest (Phase 3)
                if getattr(_cfg, "USE_OHEM", False):
                    per_sample_loss = criterion(logits, labels, reduction='none').mean(dim=1)
                    k = max(1, int(per_sample_loss.shape[0] * 0.3))
                    topk_indices = per_sample_loss.topk(k, largest=True).indices
                    
                    ohem_loss = criterion(logits[topk_indices], labels[topk_indices])
                    loss = ohem_loss
                    
                    # Add Clean Shield AFTER OHEM (Phase 1)
                    if getattr(_cfg, "USE_CLEAN_SHIELD", False):
                        if not hasattr(model, 'clean_shield'):
                            from losses import CleanMeshShieldLoss
                            model.clean_shield = CleanMeshShieldLoss().to(DEVICE)
                        loss = loss + model.clean_shield(logits[topk_indices], labels[topk_indices])
                else:
                    loss = criterion(logits, labels)
                    if getattr(_cfg, "USE_CLEAN_SHIELD", False):
                        if not hasattr(model, 'clean_shield'):
                            from losses import CleanMeshShieldLoss
                            model.clean_shield = CleanMeshShieldLoss().to(DEVICE)
                        loss = loss + model.clean_shield(logits, labels)

                if getattr(_cfg, "USE_KIMI_DPO_LOSS", False):
                    from models import KimiQualityPreferenceLoss
                    if not hasattr(model, 'kimi_dpo'):
                        model.kimi_dpo = KimiQualityPreferenceLoss().to(DEVICE)
                    clean_mask = (labels.sum(dim=1) == 0)
                    defective_mask = (labels.sum(dim=1) > 0)
                    if clean_mask.any() and defective_mask.any():
                        clean_scores = -logits[clean_mask].mean(dim=1)
                        defective_scores = -logits[defective_mask].mean(dim=1)
                        kimi_loss = model.kimi_dpo(clean_scores.mean().unsqueeze(0), defective_scores.mean().unsqueeze(0))
                        loss = loss + 0.1 * kimi_loss

                # v3.0: Add load-balancing auxiliary loss for MoE
                if is_moe and "load_balance_loss" in aux_info:
                    loss = loss + MOE_LOAD_BALANCE_COEFF * aux_info["load_balance_loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [WARNING] NaN/Inf loss at batch {batch_idx+1}, skipping backward and resetting gradients")
                    optimizer.zero_grad()
                    accum_window_count = 0
                    continue
                accum_window_count += 1
                loss = loss / GRADIENT_ACCUM_STEPS

            scaler.scale(loss).backward()

            if accum_window_count == GRADIENT_ACCUM_STEPS or (batch_idx + 1) == len(train_loader):
                if accum_window_count < GRADIENT_ACCUM_STEPS and accum_window_count > 0:
                    scale_factor = GRADIENT_ACCUM_STEPS / float(accum_window_count)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)
                if GRADIENT_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                if use_onecycle:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad()
                accum_window_count = 0

            train_loss_sum += loss.item() * GRADIENT_ACCUM_STEPS
            train_batches += 1

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = train_loss_sum / train_batches
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Epoch {epoch+1}/{NUM_EPOCHS} Batch {batch_idx+1}] "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                )

        if not use_onecycle:
            scheduler.step()

        # ── Validate (with EMA if enabled) ────────────────────────────────
        if ema is not None:
            backup = ema.apply_shadow(model)
            
        val_ratio = getattr(_cfg, "VAL_SUBSAMPLE_RATIO", 1.0)
        if val_ratio < 1.0 and epoch < (NUM_EPOCHS - 3):
            val_len = len(val_dataset)
            sub_size = max(1, int(val_len * val_ratio))
            rng_val = np.random.RandomState(1337 + epoch)
            val_indices = rng_val.choice(val_len, size=sub_size, replace=False)
            val_indices = sorted(val_indices)
            
            epoch_val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
            epoch_val_loader = DataLoader(
                epoch_val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=effective_workers, pin_memory=PIN_MEMORY,
                worker_init_fn=_worker_init_fn,
                persistent_workers=False,
                prefetch_factor=2 if effective_workers > 0 else None,
                collate_fn=safe_collate,
            )
            epoch_val_labels = val_labels.iloc[val_indices]
        else:
            epoch_val_loader = val_loader
            epoch_val_labels = val_labels

        val_metrics = validate(model, epoch_val_loader, epoch_val_labels, DEVICE, is_moe=is_moe)
        if ema is not None:
            ema.restore(model, backup)
        val_f1 = val_metrics["f1_defects"]
        val_f1_final = val_metrics["f1_final"]
        avg_train_loss = train_loss_sum / max(train_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_f1"].append(val_f1)
        history["val_f1_final"].append(val_f1_final)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        epoch_time = time.time() - epoch_start
        print(
            f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val F1_defects: {val_f1:.4f} | "
            f"Val F1_final: {val_f1_final:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ── Checkpoint (Full State Dict for Resume Safety) ─────────────────────
        if val_f1_final > best_val_score:
            best_val_score = val_f1_final
            best_epoch = epoch + 1
            patience_counter = 0
            if SAVE_BEST_MODEL:
                if ema is not None:
                    backup = ema.apply_shadow(model)
                
                checkpoint_payload = {
                    "epoch": epoch + 1,
                    "best_val_score": best_val_score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
                    "scaler_state_dict": scaler.state_dict(),
                    "scaler_mean": scaler_3d.mean if scaler_3d else None,
                    "scaler_std": scaler_3d.std if scaler_3d else None,
                }
                torch.save(checkpoint_payload, best_path)
                
                if ema is not None:
                    ema.restore(model, backup)
                    ema_path = best_path.replace(".pt", "_ema.pt")
                    torch.save(ema.state_dict(), ema_path)
                print(f"  >>> Saved best model (F1_final: {val_f1_final:.2f})"
                      f"{' [EMA weights]' if ema else ''}")
        else:
            patience_counter += 1

        if SAVE_LAST_MODEL:
            last_payload = {
                "epoch": epoch + 1,
                "best_val_score": best_val_score,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "ema_state_dict": ema.state_dict() if ema is not None else None,
                "scaler_state_dict": scaler.state_dict(),
                "scaler_mean": scaler_3d.mean if scaler_3d else None,
                "scaler_std": scaler_3d.std if scaler_3d else None,
            }
            torch.save(last_payload, last_path)

        # ── Early stopping ─────────────────────────────────────────────────
        if patience_counter >= PATIENCE:
            print(f"  [STOP] Early stopping at epoch {epoch+1} (best: epoch {best_epoch})")
            break

    # ── Threshold optimization on validation set ───────────────────────────
    print(f"\n  Optimizing thresholds on fold {fold+1} validation set...")

    best_ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    # Handle both formats: full checkpoint dict (with model_state_dict key) or plain state_dict
    best_state = best_ckpt.get("model_state_dict", best_ckpt) if isinstance(best_ckpt, dict) else best_ckpt
    best_state = clean_state_dict_keys(best_state, model)
    model.load_state_dict(best_state)

    val_proba = predict_proba(model, val_loader, DEVICE, is_moe=is_moe)
    val_true = val_labels[DEFECT_COLS].values
    val_true_quality = val_labels["quality"].values

    # v2.1: Quality-aware threshold optimization (optimizes f1_final directly)
    if OPTIMIZE_THRESHOLDS and OPTIMIZE_THRESHOLDS_F1_FINAL:
        print("  Using quality-aware threshold optimization (f1_final metric)...")
        thresholds = optimize_thresholds_f1_final(
            val_true, val_true_quality, val_proba, DEFECT_COLS,
            search_range=(0.05, 0.95), steps=50,
        )
    elif OPTIMIZE_THRESHOLDS:
        thresholds = optimize_thresholds(
            val_true, val_proba, DEFECT_COLS,
            search_range=(0.05, 0.95), steps=50,
        )
    else:
        thresholds = np.full(len(DEFECT_COLS), 0.5)

    # v2.1: Temperature scaling calibration
    temperature = 1.0
    if USE_TEMPERATURE_SCALING:
        try:
            temperature = learn_temperature(
                val_proba, val_true, val_true_quality, DEFECT_COLS,
                initial_temp=TEMPERATURE_INIT,
                lr=TEMPERATURE_LR,
                steps=100,
            )
            # Re-calibrate val_proba with temperature scaling
            if temperature != 1.0:
                val_proba_clipped = np.clip(val_proba, 1e-7, 1.0 - 1e-7)
                val_logits = np.log(val_proba_clipped / (1.0 - val_proba_clipped))
                val_proba = 1.0 / (1.0 + np.exp(-(val_logits / temperature)))

            # Re-optimize thresholds with temperature-scaled probabilities
            if OPTIMIZE_THRESHOLDS and OPTIMIZE_THRESHOLDS_F1_FINAL:
                thresholds = optimize_thresholds_f1_final(
                    val_true, val_true_quality, val_proba, DEFECT_COLS,
                    search_range=(0.05, 0.95), steps=50,
                )
        except Exception as e:
            print(f"  [WARNING] Temperature scaling failed: {e} — using T=1.0")

    val_pred = (val_proba >= thresholds).astype(int)
    val_pred_df = pd.DataFrame(val_pred, columns=DEFECT_COLS)
    val_pred_df["quality"] = derive_quality(val_pred)
    val_true_df = val_labels[DEFECT_COLS + ["quality"]].reset_index(drop=True)

    final_metrics = compute_f1_final(val_true_df, val_pred_df)
    print(f"  Final fold metrics (optimized thresholds):")
    print(f"    F1_quality: {final_metrics['f1_quality']:.4f}")
    print(f"    F1_defects: {final_metrics['f1_defects']:.4f}")
    print(f"    F1_final:   {final_metrics['f1_final']:.2f}")
    print(f"    Thresholds: {dict(zip(DEFECT_COLS, thresholds.round(3)))}")

    fold_result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_score": float(best_val_score),
        "thresholds": thresholds.tolist(),
        "temperature": float(temperature),
        "final_metrics": {k: float(v) for k, v in final_metrics.items()},
        "history": {k: [float(v) for v in vs] for k, vs in history.items()},
        "best_model_path": best_path,
    }

    result_path = os.path.join(log_dir, f"fold_{fold}_result.json")
    with open(result_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result


def _model_forward_simple(model, views, mesh_feat, pc, is_moe=False):
    """Unified forward that handles both MoE (tuple return) and single model."""
    if is_moe:
        return model.forward_simple(views, mesh_feat, pc)
    return model(views, mesh_feat, pc)


def validate(model, val_loader, val_labels_df, device, is_moe=False):
    """Validate model on validation set."""
    model.eval()
    all_proba = []
    all_labels = []

    device_type = "cuda" if "cuda" in device else "cpu"

    with torch.no_grad():
        for batch in val_loader:
            views = batch["views"].to(device)
            labels = batch["labels"].to(device)
            mesh_feat = (
                batch["mesh_features"].to(device)
                if batch["mesh_features"] is not None else None
            )
            pc = (
                batch.get("point_cloud").to(device)
                if batch.get("point_cloud") is not None else None
            )

            with safe_autocast(device_type=device_type, enabled=MIXED_PRECISION):
                logits = _model_forward_simple(model, views, mesh_feat, pc, is_moe)
                proba = torch.sigmoid(logits)

            all_proba.append(proba.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_proba = np.concatenate(all_proba, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_pred = (all_proba >= 0.5).astype(int)

    pred_df = pd.DataFrame(all_pred, columns=DEFECT_COLS)
    pred_df["quality"] = derive_quality(all_pred)
    true_df = val_labels_df[DEFECT_COLS + ["quality"]].reset_index(drop=True)

    metrics = compute_f1_final(true_df, pred_df)
    metrics["val_proba"] = all_proba
    metrics["val_labels"] = all_labels

    return metrics


def predict_proba(model, data_loader, device, is_moe=False):
    """Get probability predictions from a model."""
    model.eval()
    all_proba = []
    device_type = "cuda" if "cuda" in device else "cpu"

    with torch.no_grad():
        for batch in data_loader:
            views = batch["views"].to(device)
            mesh_feat = (
                batch["mesh_features"].to(device)
                if batch["mesh_features"] is not None else None
            )
            pc = (
                batch.get("point_cloud").to(device)
                if batch.get("point_cloud") is not None else None
            )

            with safe_autocast(device_type=device_type, enabled=MIXED_PRECISION):
                logits = _model_forward_simple(model, views, mesh_feat, pc, is_moe)
                proba = torch.sigmoid(logits)

            all_proba.append(proba.cpu().numpy())

    return np.concatenate(all_proba, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# FULL CROSS-VALIDATION TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_full_cv(
    train_df: pd.DataFrame,
    image_dir: str,
    mesh_features: np.ndarray = None,
    point_clouds: np.ndarray = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
) -> dict:
    """
    Run full stratified K-fold cross-validation training.

    v2.0: Accepts point_clouds for optional PointNet branch.
    v2.1: Uses EMA, mixup, quality-aware thresholds, temperature scaling.
    """
    set_seed(SEED)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_df = train_df.copy()
    # Strip "OUTPUT:" prefix from columns if present
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    if "Unnamed: 0" in train_df.columns:
        train_df = train_df.drop(columns=["Unnamed: 0"])
    if "quality" not in train_df.columns:
        defect_vals = train_df[DEFECT_COLS].values
        train_df["quality"] = derive_quality(defect_vals)
        print("  [INFO] Auto-derived missing 'quality' column from 10 defect labels")

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=SEED
    )

    item_ids = train_df["item_id"].values
    stratify_labels = train_df["quality"].values

    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(item_ids, stratify_labels)):
        # OVERRIDE BACKBONE FOR THIS FOLD (Heterogeneous CV - Step 1)
        import config as _cfg
        if hasattr(_cfg, "HETERO_CV_BACKBONES") and _cfg.HETERO_CV_BACKBONES:
            _cfg.IMAGE_BACKBONE = _cfg.HETERO_CV_BACKBONES[fold % len(_cfg.HETERO_CV_BACKBONES)]
            print(f"  [HETERO CV] Fold {fold} using backbone: {_cfg.IMAGE_BACKBONE}")

        fold_train_ids = item_ids[train_idx].tolist()
        fold_val_ids = item_ids[val_idx].tolist()

        fold_mesh_train = mesh_features[train_idx] if mesh_features is not None else None
        fold_mesh_val = mesh_features[val_idx] if mesh_features is not None else None

        # PointNet data (Limitation #1)
        fold_pc_train = point_clouds[train_idx] if point_clouds is not None else None
        fold_pc_val = point_clouds[val_idx] if point_clouds is not None else None

        result = train_one_fold(
            fold=fold,
            train_ids=fold_train_ids,
            val_ids=fold_val_ids,
            train_df=train_df,
            image_dir=image_dir,
            mesh_feat_train=fold_mesh_train,
            mesh_feat_val=fold_mesh_val,
            pc_train=fold_pc_train,
            pc_val=fold_pc_val,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

        all_fold_results.append(result)

    # ── Aggregate results ──────────────────────────────────────────────────
    avg_metrics = {}
    for key in ["f1_quality", "f1_defects", "f1_final"]:
        values = [r["final_metrics"][key] for r in all_fold_results]
        avg_metrics[f"{key}_mean"] = float(np.mean(values))
        avg_metrics[f"{key}_std"] = float(np.std(values))

    all_thresholds = np.array([r["thresholds"] for r in all_fold_results])
    avg_thresholds = all_thresholds.mean(axis=0)

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS ({NUM_FOLDS} folds)")
    print(f"{'='*60}")
    print(f"  F1_final:   {avg_metrics['f1_final_mean']:.2f} +/- {avg_metrics['f1_final_std']:.2f}")
    print(f"  F1_quality: {avg_metrics['f1_quality_mean']:.4f} +/- {avg_metrics['f1_quality_std']:.4f}")
    print(f"  F1_defects: {avg_metrics['f1_defects_mean']:.4f} +/- {avg_metrics['f1_defects_std']:.4f}")
    print(f"  Avg thresholds: {dict(zip(DEFECT_COLS, avg_thresholds.round(3)))}")

    # Average temperature across folds
    avg_temperature = float(np.mean([r.get("temperature", 1.0) for r in all_fold_results]))

    cv_result = {
        "avg_metrics": avg_metrics,
        "avg_thresholds": avg_thresholds.tolist(),
        "avg_temperature": avg_temperature,
        "fold_results": all_fold_results,
    }

    cv_path = os.path.join(log_dir, "cv_results.json")
    with open(cv_path, "w") as f:
        json.dump(cv_result, f, indent=2, default=str)

    return cv_result