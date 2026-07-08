"""
================================================================================
SBER AI Journey — 3D Mesh Quality Control: KAGGLE NOTEBOOK SCRIPT
================================================================================
This script is designed to be run cell-by-cell in a Kaggle Notebook.

HOW TO USE:
  1. Upload the sber_mesh_qc code as a Kaggle Dataset (see instructions below)
  2. Create a new Kaggle Notebook with GPU T4 enabled
  3. Add your code dataset + competition data dataset
  4. Copy each "# ── CELL N" section into a separate Kaggle cell
  5. Run cells in order

KAGGLE-SPECIFIC ADAPTATIONS:
  - Paths mapped to /kaggle/input/ (read-only) and /kaggle/working/ (writable)
  - Only missing dependencies installed (torch/torchvision pre-installed by Kaggle)
  - USE_MOE=False by default to fit in T4 VRAM (15.4GB)
  - Smoke test option for quick validation before full training
  - Automatic VRAM monitoring
  - Disk space checks (Kaggle limit: ~20GB working, ~70GB total)
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 1: Environment Setup & Dependency Installation
# ══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os

print("=" * 60)
print("  CELL 1: ENVIRONMENT SETUP")
print("=" * 60)

# ── Check GPU availability ────────────────────────────────────────────────
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
else:
    print("[WARNING] No GPU detected! Training will be VERY slow on CPU.")

# ── Install ONLY missing dependencies ──────────────────────────────────────
# IMPORTANT: Do NOT install torch/torchvision — Kaggle's pre-installed versions
# are GPU-enabled. Installing from requirements.txt would replace them with
# CPU-only versions and break CUDA support.
MISSING_PACKAGES = [
    "gdown",           # For downloading from Google Drive/Yandex
    "py7zr",           # For extracting .7z archives
    "trimesh",         # For 3D mesh processing
    "tqdm",            # Progress bars (usually pre-installed but just in case)
]

for pkg in MISSING_PACKAGES:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg} already installed")
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        print(f"  ✓ {pkg} installed")

# ── Verify critical imports ────────────────────────────────────────────────
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from PIL import Image

print(f"\ntorchvision: {torchvision.__version__}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"PIL: {Image.__version__}")

# ── Check disk space ──────────────────────────────────────────────────────
import shutil
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"\nDisk space (working): {free / 1024**3:.1f} GB free / {total / 1024**3:.1f} GB total")
if free / 1024**3 < 5:
    print("[WARNING] Less than 5GB free disk space!")

print("\n✓ Environment setup complete!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 2: Path Configuration & Code Import
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  CELL 2: PATH CONFIGURATION")
print("=" * 60)

# ── IMPORTANT: Configure these paths to match YOUR Kaggle setup ───────────
#
# OPTION A: Code uploaded as a Kaggle Dataset named "sber-mesh-qc-code"
#           Data uploaded as a separate dataset OR downloaded at runtime
#
# OPTION B: Code uploaded directly into the notebook working directory
#
# Kaggle path structure:
#   /kaggle/input/          ← READ-ONLY (your uploaded datasets appear here)
#   /kaggle/working/        ← READ-WRITE (output files, checkpoints, etc.)
#   /kaggle/tmp/            ← TEMPORARY (cleared between sessions)

# ── Auto-detect code location ──────────────────────────────────────────────
CODE_DATASET_CANDIDATES = [
    "/kaggle/input/sber-mesh-qc-solution1/solution",
    "/kaggle/input/sber-mesh-qc-code/sber_mesh_qc/solution",
    "/kaggle/input/sber-mesh-qc-code/solution",
    "/kaggle/input/sber-mesh-qc/sber_mesh_qc/solution",
    "/kaggle/input/sber-mesh-qc/solution",
    "/kaggle/working/sber_mesh_qc/solution",
    "/kaggle/working/solution",
]

# Dynamic fallback search in case the dataset has a custom name
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "config.py" in files and "models.py" in files and root.endswith("solution"):
            if root not in CODE_DATASET_CANDIDATES:
                CODE_DATASET_CANDIDATES.insert(0, root)

SOLUTION_DIR = None
for candidate in CODE_DATASET_CANDIDATES:
    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.py")):
        SOLUTION_DIR = candidate
        break

if SOLUTION_DIR is None:
    print("[ERROR] Could not find the solution code!")
    print("Expected to find config.py in one of these locations:")
    for c in CODE_DATASET_CANDIDATES:
        print(f"  {c} — {'EXISTS' if os.path.isdir(c) else 'NOT FOUND'}")
    print("\nPlease upload the sber_mesh_qc folder as a Kaggle Dataset.")
    raise FileNotFoundError("Solution code not found. See instructions above.")

print(f"  Solution code found at: {SOLUTION_DIR}")

# ── Add solution directory to Python path ──────────────────────────────────
if SOLUTION_DIR not in sys.path:
    sys.path.insert(0, SOLUTION_DIR)

# Also add parent directory (for relative imports)
PARENT_DIR = os.path.dirname(SOLUTION_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ── Auto-detect data location ─────────────────────────────────────────────
DATA_CANDIDATES = [
    "/kaggle/input/sber-mesh-qc-data/data",
    "/kaggle/input/sber-mesh-qc-data",
    "/kaggle/input/aic-data/data",
    "/kaggle/input/aic-data",
    "/kaggle/working/data",
]

DATA_DIR = None
for candidate in DATA_CANDIDATES:
    if os.path.isdir(candidate):
        # Check if it has train.csv or a train/ directory
        has_csv = os.path.isfile(os.path.join(candidate, "train.csv"))
        has_dir = os.path.isdir(os.path.join(candidate, "train"))
        if has_csv or has_dir:
            DATA_DIR = candidate
            break

if DATA_DIR is None:
    print("[INFO] Competition data not found in /kaggle/input/")
    print("       Will attempt to download it in the next cell.")
    DATA_DIR = "/kaggle/working/data"
else:
    print(f"  Competition data found at: {DATA_DIR}")

# ── Set up output directories (must be in /kaggle/working/) ────────────────
WORKING_DIR = "/kaggle/working"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
LOG_DIR = os.path.join(WORKING_DIR, "logs")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"  Data directory: {DATA_DIR}")
print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
print(f"  Log dir:        {LOG_DIR}")
print(f"  Submission:     {SUBMISSION_PATH}")

# ── Override config module paths BEFORE importing it ───────────────────────
import config

# Smart search across all attached /kaggle/input datasets
train_csv_path = None
test_csv_path = None
train_dir_path = None
test_dir_path = None

if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files and train_csv_path is None:
            train_csv_path = os.path.join(root, "train.csv")
        if "test.csv" in files and test_csv_path is None:
            test_csv_path = os.path.join(root, "test.csv")
        if "train" in dirs and train_dir_path is None:
            train_dir_path = os.path.join(root, "train")
        if "test" in dirs and test_dir_path is None:
            test_dir_path = os.path.join(root, "test")

if train_csv_path is None: train_csv_path = os.path.join(DATA_DIR, "train.csv")
if test_csv_path is None: test_csv_path = os.path.join(DATA_DIR, "test.csv")
if train_dir_path is None: train_dir_path = os.path.join(DATA_DIR, "train")
if test_dir_path is None: test_dir_path = os.path.join(DATA_DIR, "test")

config.BASE_DIR = WORKING_DIR
config.DATA_DIR = DATA_DIR
config.TRAIN_DIR = train_dir_path
config.TEST_DIR = test_dir_path
config.CHECKPOINT_DIR = CHECKPOINT_DIR
config.LOG_DIR = LOG_DIR
config.TRAIN_CSV = train_csv_path
config.TEST_CSV = test_csv_path

# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL KAGGLE CONFIGURATION OVERRIDES
# ═══════════════════════════════════════════════════════════════════════════
#
# These override defaults in config.py to avoid Kaggle-specific issues:
#
# 1. USE_MOE=False: MoE with 3 experts uses ~14GB VRAM on T4 (15.4GB).
#    Leaves almost no headroom for data/activations. Single backbone is safer.
#
# 2. BATCH_SIZE=8 with GRADIENT_ACCUM_STEPS=4: Effective batch=32 while
#    keeping per-step memory usage low.
#
# 3. NUM_WORKERS=2: Kaggle containers have limited shared memory (/dev/shm).
#    More workers cause "bus error" / "out of shared memory" crashes.
#
# 4. SEQUENTIAL_VIEW_PROCESSING=False: Only enable if you get OOM errors.
#    Trades ~20% speed for ~40% VRAM savings.

config.USE_MOE = False          # Single backbone = safer on T4
config.BATCH_SIZE = 8           # Safe for T4
config.NUM_WORKERS = 2          # Kaggle shm limit
config.GRADIENT_ACCUM_STEPS = 4 # Effective batch = 32
config.NUM_FOLDS = 3            # 3-fold CV (faster than 5)
config.NUM_EPOCHS = 20          # Full training
config.USE_POINTNET_BRANCH = False  # Disable PointNet (saves VRAM)
config.USE_EXTENDED_FEATURES = True # Use 68-dim features
config.PROGRESSIVE_RESIZE = True    # Speed optimization
config.MIXED_PRECISION = True       # FP16 for speed + memory

print("\n✓ Configuration overrides applied for Kaggle T4")
print(f"  Architecture: Single backbone ({config.IMAGE_BACKBONE})")
print(f"  Batch size: {config.BATCH_SIZE} × {config.GRADIENT_ACCUM_STEPS} accumulation = {config.BATCH_SIZE * config.GRADIENT_ACCUM_STEPS} effective")
print(f"  Folds: {config.NUM_FOLDS}, Epochs: {config.NUM_EPOCHS}")
print(f"  Mixed precision: {config.MIXED_PRECISION}")
print(f"  Extended features (68-dim): {config.USE_EXTENDED_FEATURES}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 3: Download & Prepare Data (skip if data already uploaded)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  CELL 3: DATA PREPARATION")
print("=" * 60)

train_csv = os.path.join(DATA_DIR, "train.csv")
test_csv = os.path.join(DATA_DIR, "test.csv")
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

if os.path.isfile(train_csv) and os.path.isfile(test_csv) and \
   os.path.isdir(train_dir) and os.path.isdir(test_dir):
    train_count = len(os.listdir(train_dir))
    test_count = len(os.listdir(test_dir))
    print(f"  Data already present!")
    print(f"  train.csv: ✓")
    print(f"  test.csv:  ✓")
    print(f"  train/ directory: {train_count} files")
    print(f"  test/ directory:  {test_count} files")
else:
    print("  Data not found. Attempting download...")
    print("  NOTE: If download fails, upload the dataset manually to Kaggle.")
    print("        See the guide in the plan artifact for instructions.")
    try:
        from data_utils import download_data
        download_data(DATA_DIR)
        print("  ✓ Download complete!")
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"\n  ┌────────────────────────────────────────────────────┐")
        print(f"  │  MANUAL DATA UPLOAD REQUIRED                       │")
        print(f"  │                                                    │")
        print(f"  │  1. Download the dataset from the competition page │")
        print(f"  │  2. Go to Kaggle → Datasets → New Dataset         │")
        print(f"  │  3. Upload train.csv, test.csv, train/, test/      │")
        print(f"  │  4. Name it 'sber-mesh-qc-data'                   │")
        print(f"  │  5. Add it to this notebook via Add Data           │")
        print(f"  │  6. Re-run Cell 2 to detect the new path          │")
        print(f"  └────────────────────────────────────────────────────┘")

# ── Show data summary ─────────────────────────────────────────────────────
if os.path.isfile(train_csv):
    train_df = pd.read_csv(train_csv)
    print(f"\n  Train dataset: {len(train_df)} samples")
    print(f"  Columns: {list(train_df.columns)}")
    if 'quality' in train_df.columns:
        print(f"  Quality distribution: good={train_df['quality'].sum()}, "
              f"bad={len(train_df) - train_df['quality'].sum()}")
    # Show per-class positive rates
    defect_cols = [c for c in train_df.columns if c in config.DEFECT_COLS]
    if defect_cols:
        print(f"\n  Per-class positive rates:")
        for col in defect_cols:
            rate = train_df[col].mean() * 100
            print(f"    {col:15s}: {rate:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 4: Extract Mesh Features
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  CELL 4: MESH FEATURE EXTRACTION")
print("=" * 60)

import time
from mesh_features import batch_extract_mesh_features

train_csv_path = os.path.join(DATA_DIR, "train.csv")
test_csv_path = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(train_csv_path)
train_ids = [str(x) for x in train_df["item_id"].tolist()]

# ── Check for cached features first ──────────────────────────────────────
cache_train = os.path.join(WORKING_DIR, "mesh_features_train_extended.npz")
cache_test = os.path.join(WORKING_DIR, "mesh_features_test_extended.npz")

# Also check /kaggle/input/ for pre-cached features
for input_cache in [
    "/kaggle/input/sber-mesh-qc-code/sber_mesh_qc/mesh_features_train_extended.npz",
    "/kaggle/input/sber-mesh-qc-data/mesh_features_train_extended.npz",
]:
    if os.path.isfile(input_cache) and not os.path.isfile(cache_train):
        import shutil
        shutil.copy2(input_cache, cache_train)
        print(f"  Copied cached features from {input_cache}")
        break

if os.path.isfile(cache_train):
    data = np.load(cache_train, allow_pickle=False)
    train_features = data["features"]
    print(f"  Loaded cached train features: {train_features.shape}")
else:
    npz_dir = os.path.join(DATA_DIR, "train")
    print(f"  Extracting {len(train_ids)} train features (68-dim extended)...")
    print(f"  This may take 10-30 minutes on first run...")
    t0 = time.time()
    train_features = batch_extract_mesh_features(train_ids, npz_dir, extended=True)
    elapsed = time.time() - t0
    print(f"  Extracted in {elapsed:.1f}s — shape: {train_features.shape}")
    np.savez_compressed(cache_train, features=train_features, item_ids=np.array(train_ids))
    print(f"  Cached to {cache_train}")

# Test features
if os.path.isfile(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    test_ids = [str(x) for x in test_df["item_id"].tolist()]

    if os.path.isfile(cache_test):
        data = np.load(cache_test, allow_pickle=False)
        test_features = data["features"]
        print(f"  Loaded cached test features: {test_features.shape}")
    else:
        npz_dir = os.path.join(DATA_DIR, "test")
        print(f"  Extracting {len(test_ids)} test features...")
        t0 = time.time()
        test_features = batch_extract_mesh_features(test_ids, npz_dir, extended=True)
        elapsed = time.time() - t0
        print(f"  Extracted in {elapsed:.1f}s — shape: {test_features.shape}")
        np.savez_compressed(cache_test, features=test_features, item_ids=np.array(test_ids))
        print(f"  Cached to {cache_test}")
else:
    test_features = None
    print("  [INFO] No test.csv found — skipping test feature extraction")

print("\n✓ Feature extraction complete!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 5: SMOKE TEST (Optional but HIGHLY Recommended)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  CELL 5: SMOKE TEST (1 epoch, 2 folds)")
print("=" * 60)

# ── Override to minimal settings for smoke test ───────────────────────────
SMOKE_TEST = True  # ← Set to False to skip smoke test and go straight to full training

if SMOKE_TEST:
    # Save original settings
    _orig_folds = config.NUM_FOLDS
    _orig_epochs = config.NUM_EPOCHS

    config.NUM_FOLDS = 2
    config.NUM_EPOCHS = 1

    print(f"  Running smoke test: {config.NUM_FOLDS} folds × {config.NUM_EPOCHS} epoch")
    print(f"  This should take ~1-2 minutes on T4...")

    from train import train_full_cv
    from utils import set_seed, derive_quality

    train_df = pd.read_csv(train_csv_path)
    if "quality" not in train_df.columns:
        train_df["quality"] = derive_quality(train_df[config.DEFECT_COLS].values)

    train_image_dir = os.path.join(DATA_DIR, "train")

    try:
        t0 = time.time()
        smoke_results = train_full_cv(
            train_df=train_df,
            image_dir=train_image_dir,
            mesh_features=train_features,
            point_clouds=None,
            checkpoint_dir=CHECKPOINT_DIR,
            log_dir=LOG_DIR,
        )
        elapsed = time.time() - t0
        print(f"\n  ✓ Smoke test PASSED in {elapsed:.1f}s")
        print(f"  F1_final: {smoke_results['avg_metrics']['f1_final_mean']:.2f}")

        # Check GPU memory
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  Peak VRAM: {peak_mem:.1f} GB / {total_mem:.1f} GB ({peak_mem/total_mem*100:.0f}%)")
            if peak_mem > total_mem * 0.9:
                print("  [WARNING] VRAM usage > 90%! Consider enabling SEQUENTIAL_VIEW_PROCESSING.")
    except Exception as e:
        print(f"\n  ✗ Smoke test FAILED: {e}")
        print("  Fix the error before running full training.")
        raise

    # Restore original settings
    config.NUM_FOLDS = _orig_folds
    config.NUM_EPOCHS = _orig_epochs
    print(f"\n  Settings restored: {config.NUM_FOLDS} folds × {config.NUM_EPOCHS} epochs")
else:
    print("  Smoke test skipped (SMOKE_TEST = False)")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 6: FULL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  CELL 6: FULL TRAINING")
print("=" * 60)

from train import train_full_cv
from utils import set_seed, derive_quality

train_df = pd.read_csv(train_csv_path)
if "quality" not in train_df.columns:
    train_df["quality"] = derive_quality(train_df[config.DEFECT_COLS].values)

train_image_dir = os.path.join(DATA_DIR, "train")

print(f"  Training configuration:")
print(f"    Folds: {config.NUM_FOLDS}")
print(f"    Epochs: {config.NUM_EPOCHS}")
print(f"    Batch size: {config.BATCH_SIZE} × {config.GRADIENT_ACCUM_STEPS} = {config.BATCH_SIZE * config.GRADIENT_ACCUM_STEPS}")
print(f"    Learning rate: {config.LEARNING_RATE}")
print(f"    EMA: {config.USE_EMA} (decay={config.EMA_DECAY})")
print(f"    Mixup: {config.USE_MIXUP} (alpha={config.MIXUP_ALPHA})")
print(f"    Patience: {config.PATIENCE} epochs")
print(f"    Estimated time: ~2-3 hours on T4")

t0 = time.time()
cv_results = train_full_cv(
    train_df=train_df,
    image_dir=train_image_dir,
    mesh_features=train_features,
    point_clouds=None,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
)
training_time = time.time() - t0

print(f"\n  ✓ Training complete in {training_time / 60:.1f} minutes")
print(f"  F1_final: {cv_results['avg_metrics']['f1_final_mean']:.2f} ± {cv_results['avg_metrics']['f1_final_std']:.2f}")

# Check disk usage after training
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"  Disk space remaining: {free / 1024**3:.1f} GB")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 7: INFERENCE & SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  CELL 7: INFERENCE & SUBMISSION GENERATION")
print("=" * 60)

from inference import generate_submission

test_csv_path = os.path.join(DATA_DIR, "test.csv")
test_image_dir = os.path.join(DATA_DIR, "test")
cv_results_path = os.path.join(LOG_DIR, "cv_results.json")

print(f"  Test CSV: {test_csv_path}")
print(f"  Test images: {test_image_dir}")
print(f"  Checkpoints: {CHECKPOINT_DIR}")
print(f"  Output: {SUBMISSION_PATH}")

generate_submission(
    test_csv_path=test_csv_path,
    test_image_dir=test_image_dir,
    checkpoint_dir=CHECKPOINT_DIR,
    output_path=SUBMISSION_PATH,
    cv_results_path=cv_results_path,
    mesh_features=test_features,
    point_clouds=None,
)

# ── Verify submission ──────────────────────────────────────────────────────
if os.path.isfile(SUBMISSION_PATH):
    sub_df = pd.read_csv(SUBMISSION_PATH)
    print(f"\n  ✓ Submission generated: {SUBMISSION_PATH}")
    print(f"  Shape: {sub_df.shape}")
    print(f"  Columns: {list(sub_df.columns)}")
    print(f"  First 5 rows:")
    print(sub_df.head())
    print(f"\n  Quality distribution: good={sub_df['quality'].sum()}, bad={len(sub_df) - sub_df['quality'].sum()}")
else:
    print("  [ERROR] Submission file not generated!")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"  Download submission.csv from: {SUBMISSION_PATH}")
print(f"  Or find it in the 'Output' tab of your Kaggle notebook")
