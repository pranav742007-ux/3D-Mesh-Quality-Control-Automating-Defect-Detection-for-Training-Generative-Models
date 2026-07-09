"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Configuration
===============================================================================
Centralized configuration for reproducible training & inference.
All hyperparameters, paths, and seeds are defined here.

v3.0: Added Octopus MoE architecture (multi-backbone Mixture-of-Experts).
v2.1: Added EMA, SWA, mixup, temperature scaling, quality-aware thresholds.
===============================================================================
"""

import os
import torch

# ──────────────────────────── VERSION ─────────────────────────────────────────
CONFIG_VERSION = "7.2.0"

# ──────────────────────────── REPRODUCIBILITY ────────────────────────────────
SEED = 42

# ──────────────────────────── DATA PATHS ─────────────────────────────────────
# When running on Google Colab, change BASE_DIR to '/content/sber_mesh_qc'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# submission.csv path is determined at runtime by main.py.

# Download URLs & Optional Integrity SHA256 Checksums
TRAIN_ZIP_URL = "https://disk.360.yandex.ru/d/CeZVSNyRGjrLUw"
TEST_ZIP_URL = "https://disk.360.yandex.ru/d/rUSPxzoDTHK8UQ"
ALT_DOWNLOAD_URL = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/AIC_data.tar"

TRAIN_ZIP_SHA256 = None   # Provide hex string (e.g. "a1b2c3...") to enforce download SHA256 verification
TEST_ZIP_SHA256 = None    # Provide hex string to enforce download SHA256 verification
ALT_DOWNLOAD_SHA256 = None

# ──────────────────────────── DEFECT CLASSES ─────────────────────────────────
DEFECT_COLS = [
    "abstract", "artifacts", "intersection", "lowpoly",
    "noisy", "open", "partial", "scale", "set", "simple"
]
NUM_DEFECTS = len(DEFECT_COLS)
ALL_COLS = DEFECT_COLS + ["quality"]  # 11 targets total

# ──────────────────────────── IMAGE SETTINGS ─────────────────────────────────
# Each PNG contains 6 views arranged as a 3x2 grid (or 2x3).
# We split into individual views for multi-view processing.
IMAGE_SIZE = 224          # Resize each individual view to this
CROP_TOP_FRAC = 0.0       # Fractional crop from top (if needed)
NUM_VIEWS = 6             # Number of rendered views per object
VIEW_GRID = (3, 2)        # Grid layout: 3 rows x 2 cols

# ──────────────────────────── MESH FEATURE SETTINGS ──────────────────────────
# Geometric features extracted from .npz (vertices + faces)
MESH_FEATURE_DIM = 58       # Original hand-crafted features (backward compat)
MESH_FEATURE_DIM_EXTENDED = 100  # Extended SOTA features (68 basic + 25 SHTD + 3 Betti + 1 QEM + 3 Physics)
USE_EXTENDED_FEATURES = True    # Set True to use 100-dim features
USE_SPHERICAL_HARMONICS = True  # 25 Cartesian Spherical Harmonics Descriptors (l=0..4)
USE_BETTI_NUMBERS = True        # DSU Topological Persistence Invariants (beta_0, beta_1, chi)
USE_RECONSTRUCTION_AUX = True   # Feature-space 3D Reconstruction Aux Loss

# ──────────────────────────── POINTNET BRANCH (Limitation #1) ───────────────
# Lightweight learned 3D features from raw point clouds.
# Complements hand-crafted features by capturing complex geometric patterns.
USE_POINTNET_BRANCH = False   # Set True to enable 3-branch fusion
POINTNET_NUM_POINTS = 1024     # Points sampled per mesh (memory vs quality)
POINTNET_DROPOUT = 0.3
POINTNET_WEIGHT = 0.15         # Fusion weight for PointNet branch
# NOTE: When PointNet is enabled, raw weights sum > 1.0 (0.75+0.25+0.15=1.15)
# but FusedEnsembleModel auto-normalizes to: image=0.652, mesh=0.217, pointnet=0.130

# ──────────────────────────── MODEL ARCHITECTURE ─────────────────────────────
# --- Image Branch ---
IMAGE_BACKBONE = "efficientnetv2_s"  # Options: efficientnetv2_s, efficientnet_b3, convnext_tiny
IMAGE_EMBED_DIM = 1280    # EfficientNetV2-S output channels
IMAGE_HIDDEN_DIM = 512
IMAGE_DROPOUT = 0.3
IMAGE_PRETRAINED = True

# --- Mesh Feature Branch ---
MESH_HIDDEN_DIMS = [256, 128, 64]  # MLP hidden layers
MESH_DROPOUT = 0.3

# --- Fusion ---
FUSION_METHOD = "late_average"  # Options: late_average, late_weighted, concat_mlp
FUSION_IMAGE_WEIGHT = 0.75     # Weight for image branch in late_average
FUSION_MESH_WEIGHT = 0.25      # Weight for mesh branch

# ──────────────────────────── TRAINING SETTINGS ──────────────────────────────
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
PATIENCE = 7                 # Early stopping patience
MIN_LR = 1e-7

# --- Class Weights ---
# Use inverse frequency weighting to handle severe imbalance
# (computed dynamically from data, but can override here)
USE_DYNAMIC_CLASS_WEIGHTS = True
# Manual override (if USE_DYNAMIC_CLASS_WEIGHTS=False)
CLASS_WEIGHTS = None  # Will be computed as {class: weight}

# --- Loss Function ---
LOSS_FUNCTION = "quality_focal"  # Options: bce, bce_focal, asl, hybrid_asl, quality_focal
FOCAL_GAMMA = 2.0            # Focal loss gamma
FOCAL_ALPHA = None            # Auto-computed from class weights if None
LABEL_SMOOTHING = 0.05

# --- Optimizer ---
OPTIMIZER = "adamw"          # Options: adamw, sgd
SCHEDULER = "cosine_warmup"  # Options: cosine_warmup, cosine, step, plateau
WARMUP_EPOCHS = 3

# ──────────────────────────── AUGMENTATION ───────────────────────────────────
USE_AUGMENTATION = True
AUG_PROB = 0.5
AUG_HORIZONTAL_FLIP = True
AUG_VERTICAL_FLIP = False     # Don't flip vertically (3D views)
AUG_ROTATION = 15             # Max rotation in degrees
AUG_COLOR_JITTER = 0.2
AUG_RANDOM_ERASING = 0.1

# ──────────────────────────── CROSS-VALIDATION ───────────────────────────────
NUM_FOLDS = 3
STRATIFY_BY = "quality"       # Stratify folds by quality label
VAL_SPLIT = 0.15              # Validation split within each fold

# ──────────────────────────── AUTO-DETECT GRID (Limitation #3) ────────────
AUTO_DETECT_GRID = True     # Auto-detect PNG grid layout (vs fixed 3x2)

# ──────────────────────────── SPEED OPTIMIZATIONS (Limitation #4) ───────────
# Progressive training: start with smaller image size, then increase.
# Keys are epoch thresholds: "at epoch X, switch to this size".
# Schedule goes SMALL → LARGE (the standard progressive resize pattern).
PROGRESSIVE_RESIZE = True
PROGRESSIVE_SCHEDULE = {0: 128, 3: 192, 8: 224}  # epoch -> image_size

# ──────────────────────────── MEMORY OPTIMIZATIONS (Limitation #5) ────────
# Gradient checkpointing: trade compute for memory
USE_GRADIENT_CHECKPOINTING = False  # Enable if OOM on T4
# Sequential view processing: process one view at a time (saves VRAM, slower)
SEQUENTIAL_VIEW_PROCESSING = False  # Enable if 6 simultaneous views OOM
# View subsampling: use fewer views during training (all 6 at inference)
VIEWS_TRAIN_SUBSAMPLE = None  # e.g., 4 means randomly pick 4 of 6 views per item
GRADIENT_ACCUM_STEPS = 4     # Effective batch = BATCH_SIZE * this = 32
# Mixed precision
MIXED_PRECISION = True        # FP16 training (AMP)

# ──────────────────────────── INFERENCE ───────────────────────────────────────
USE_TTA = True               # Test-Time Augmentation for inference
TTA_FLIPS = [True, False]
TTA_ROTATIONS = [0, 180]           # 4 TTA variants (speed/accuracy tradeoff)
TTA_FAST_FLIPS = [False]           # Faster: no flips
TTA_FAST_ROTATIONS = [0]          # Faster: no rotations

# --- Threshold Optimization ---
OPTIMIZE_THRESHOLDS = True
THRESHOLD_SEARCH_RANGE = (0.05, 0.95)
THRESHOLD_SEARCH_STEPS = 100
DEFAULT_THRESHOLD = 0.5


# ──────────────────────────── EMA (v2.1 SCORE TRICK) ───────────────────────
# Exponential Moving Average of model weights for better generalization.
# At evaluation, uses averaged weights instead of last-step weights.
# Typical improvement: +0.3-0.8% F1_final with no extra inference cost.
USE_EMA = True
EMA_DECAY = 0.999               # Higher = smoother averaging (0.999 = ~1000 steps)

# ──────────────────────────── SWA (v2.1 SCORE TRICK) ───────────────────────
# Stochastic Weight Averaging: average model weights over last N epochs.
# Complementary to EMA — use ONE of them (prefer EMA for this task size).
USE_SWA = False                   # Disabled by default (EMA is preferred)
SWA_START_EPOCH = 20              # Start SWA from this epoch
SWA_LR = 1e-4                     # SWA learning rate

# ──────────────────────────── MIXUP (v2.1 SCORE TRICK) ─────────────────────
# Multi-label mixup: blend images and soft labels for better generalization.
# Proven to improve F1 by 0.5-1.5% on imbalanced multi-label tasks.
USE_MIXUP = True
MIXUP_ALPHA = 0.2                 # Beta distribution alpha (0.2 = mild mixing)
MIXUP_PROB = 0.3                  # Probability of applying mixup per batch
MIXUP_LABEL_SMOOTH = 0.1          # Additional label smoothing for mixed samples

# ──────────────────────────── TEMPERATURE SCALING (v2.1 SCORE TRICK) ───────
# Learn a temperature parameter T on validation set to calibrate probabilities.
# Better calibration → better threshold optimization → higher F1_final.
USE_TEMPERATURE_SCALING = True
TEMPERATURE_INIT = 1.5            # Initial temperature (T>1 = softer probabilities)
TEMPERATURE_LR = 0.01             # Learning rate for temperature optimization

# ──────────────────────────── QUALITY-AWARE THRESHOLDS (v2.1) ─────────────
# Optimize thresholds to maximize f1_final directly (not just F1_defects).
# This jointly considers quality and defect F1, which is the competition metric.
OPTIMIZE_THRESHOLDS_F1_FINAL = True

# ──────────────────────────── VIEW-DROP TTA (v2.1 SCORE TRICK) ─────────────
# At inference, randomly drop views and average predictions.
# This regularizes against view-specific overfitting.
USE_VIEW_DROP_TTA = False          # Disabled by default (adds inference time)
VIEW_DROP_COUNTS = [4, 5, 6]      # Try dropping to 4, 5, and all 6 views
VIEW_DROP_ITERATIONS = 3           # Number of random subsets per count

# ──────────────────────────── OCTOPUS MoE ARCHITECTURE (v3.0) ─────────────
# Multi-backbone Mixture-of-Experts with learned sparse gating.
# When USE_MOE=False, falls back to the single-backbone FusedEnsembleModel.
# When USE_MOE=True, uses OctopusMoEModel with diverse expert backbones.
#
# Why this works for 3D mesh QC:
#   - Different defect types benefit from different visual perspectives
#   - "abstract"/"lowpoly" need global structure (EfficientNetV2-S)
#   - "noisy"/"artifacts" need texture detail (ConvNeXt-Tiny)
#   - "intersection"/"open" need geometric edge detection (ResNet-50)
#   - The router learns which expert(s) to trust per sample
#
# Memory tip: On 16GB T4, USE_MOE=True with 4 experts + BATCH_SIZE=8 + 
# SEQUENTIAL_VIEWS_IN_MOE=True fits comfortably (~14GB peak VRAM).
# ============================================================================

USE_MOE = False                     # Enable Octopus MoE (False = single backbone, default to avoid high overhead on standard setups)

# Expert backbone configurations (heterogeneous architectures)
# Each dict creates one expert. Add/remove experts via @register_backbone().
MOE_EXPERT_CONFIGS = [
    # Expert 0: EfficientNetV2-S — best speed/accuracy tradeoff, global structure
    {
        "backbone_name": "efficientnetv2_s",
        "pretrained": True,
        "hidden_dim": 512,
        "dropout": 0.3,
        "sequential_views": False,
    },
    # Expert 1: ConvNeXt-Tiny — excellent for texture and local patterns
    {
        "backbone_name": "convnext_tiny",
        "pretrained": True,
        "hidden_dim": 384,
        "dropout": 0.3,
        "sequential_views": False,
    },
    # Expert 2: ResNet-50 — strong edge/contour detection, geometric patterns
    {
        "backbone_name": "resnet50",
        "pretrained": True,
        "hidden_dim": 512,
        "dropout": 0.3,
        "sequential_views": False,
    },
    # Expert 3: Swin Tiny — local window attention transformer
    {
        "backbone_name": "swin_tiny",
        "pretrained": True,
        "hidden_dim": 384,
        "dropout": 0.3,
        "sequential_views": False,
    },
    # Expert 4: ViT Small — global patch attention transformer
    {
        "backbone_name": "vit_small",
        "pretrained": True,
        "hidden_dim": 384,
        "dropout": 0.3,
        "sequential_views": False,
    },
]

# MoE Router settings
MOE_TOP_K = 2                       # Route each sample to top-K experts
MOE_ROUTER_HIDDEN_DIM = 256         # Router MLP hidden dimension
MOE_ROUTER_NOISE_STD = 1.0          # Noise for exploration during training
MOE_LOAD_BALANCE_COEFF = 0.01       # Coefficient for load-balancing auxiliary loss

# MoE Training
MOE_ROUTER_LR_FACTOR = 5.0          # Router learns faster than experts
MOE_PROJECTION_DIM = 512            # Common projection dim for heterogeneous experts

# Memory optimization for MoE
SEQUENTIAL_VIEWS_IN_MOE = True      # Process views sequentially within each expert (saves VRAM)
# Note: With 4 experts, this adds ~30% inference time but saves ~60% peak VRAM


# ──────────────────────────── HARDWARE ───────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2               # Kaggle containers have limited shared memory
PIN_MEMORY = True


# ──────────────────────────── LOGGING ────────────────────────────────────────
LOG_INTERVAL = 50             # Log every N batches
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = True
WANDB_PROJECT = None          # Set to string to enable W&B logging

# ──────────────────────────── v4.1 RISK-CONTROLLED ARCHITECTURE CONFIGS ──────
USE_CROSS_VIEW_TRANSFORMER = False  # Enable Cross-View Transformer Fusion
TRANSFORMER_EMBED_DIM = 256         # d_model=256 (overfitting control)
TRANSFORMER_DEPTH = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1

USE_DEFECT_QUERY_DECODER = False    # Enable Defect Query Decoder (11 queries)
DEFECT_QUERY_DEPTH = 1
NUM_DEFECT_QUERIES = 10
USE_QUALITY_QUERY = True

USE_SPATIAL_VIEW_TOKENS = False     # Enable 2x2 Spatial View Tokens Lite (24 tokens)
SPATIAL_TOKEN_GRID = 2

USE_HIERARCHICAL_HEAD = False       # Enable Soft Zero-Initialized Hierarchy (alpha=0, beta=0)
USE_EXPERT_HEADS = False            # Enable Defect Domain Experts

USE_MULTI_SAMPLE_DROPOUT = False    # Enable Multi-Sample Dropout (MSDO)
USE_GRADIENT_NORMALS = False        # Enable 6-channel Sobel pseudo-normals
USE_CROSS_MODAL_ATTENTION = False   # Enable Bi-Directional Image <-> Geometry Attention
IMAGE_IN_CHANNELS = 6 if USE_GRADIENT_NORMALS else 3
ABLATION_LOG_FILE = "logs/ablation_results.csv"

# ──────────────────────────── v7.2 FRONTIER EXPERIMENTAL CONFIGS ───────────────
USE_FLASH_ATTENTION = False         # Enable FlashAttention-2 Cross-Modal SDPA
USE_DEEPSEEK_MLA = False            # Enable DeepSeek-V3 Multi-Head Latent Attention
USE_KIMI_LATENT_MEMORY = False      # Enable Kimi K1.5 Latent Memory Compressor
USE_GLM_SPATIAL_ALIGNER = False     # Enable GLM-5.2 Image Spatial Aligner
USE_XAI_ROUTER = False              # Enable xAI Grok-3 MoE Dynamic Gated Router
USE_FLEXIBLE_EFFORT = False         # Enable FlexibleThinkingEffortController
USE_KIMI_DPO_LOSS = False           # Enable Kimi Quality Preference DPO Loss
USE_OMNI_ROUTE = False              # Enable OmniRoute Dynamic Path Dispatcher
USE_EARLY_EXIT = True               # Enable ConfidenceScheduledRouter early-exit (Phase A/B/C agentic flow)
EARLY_EXIT_THRESHOLD = 0.95         # Early exit confidence threshold

# ──────────────────────────── PERFORMANCE CONFIGS ──────────────────────────────
USE_NUMBA = True                         # Enable Numba JIT Compilation for heavy math
FEATURE_CACHE_FORMAT = "mmap"            # Cache format: "mmap" (.npy) or "npz" (.npz)
OFFLINE_AUGMENT = False                  # Enable generating augmented meshes offline
PREPROCESS_IMAGES_OFFLINE = True         # Pre-crop and pre-resize rendered view grids offline
USE_KORNIA = False                       # Enable GPU-accelerated transforms via kornia
USE_TORCH_COMPILE = False                # Enable torch.compile for model optimization
VAL_SUBSAMPLE_RATIO = 0.5                # Fraction of validation set to evaluate per epoch (0.2 is too noisy for checkpoint selection)
# NOTE: NUM_WORKERS is defined once above (line ~293). Do NOT redefine here.

# ──────────────────────────── DERIVED ────────────────────────────────────────
def get_class_weights(train_df):
    """
    Compute inverse-frequency class weights for the 10 defect classes.
    Uses sqrt of inverse frequency for moderate rebalancing.
    """
    import numpy as np
    weights = {}
    for col in DEFECT_COLS:
        pos_count = train_df[col].sum()
        neg_count = len(train_df) - pos_count
        # Use sqrt of inverse frequency to avoid extreme weights
        weights[col] = np.sqrt(neg_count / max(pos_count, 1))
    return weights
