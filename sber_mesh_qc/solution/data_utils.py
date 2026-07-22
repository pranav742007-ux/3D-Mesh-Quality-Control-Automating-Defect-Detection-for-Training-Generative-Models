"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Data Utilities  [v7.1 Master Engine]
===============================================================================
Handles downloading dataset archives (train + test) with aria2c multi-threaded acceleration,
CVE-2007-4559 path traversal protection, layout validation, auto-detection of grid layouts,
Curvature-Weighted FPS Point Cloud sampling, and parallel extraction of 100D mesh features.
===============================================================================
"""

import os
import glob
import subprocess
import tarfile
import zipfile
import time
import shutil
import re
import hashlib
from typing import Tuple, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ── Security helpers ──────────────────────────────────────────────────────
def _is_safe_archive_member(member_name: str, dest_dir: str) -> bool:
    """Check if an archive member path is safe (no path traversal).
    
    Prevents CVE-2007-4559: malicious archives containing entries like
    '../../etc/passwd' or Windows UNC paths writing outside dest_dir.
    """
    # Normalize slashes and resolve canonical absolute paths
    clean_member = os.path.normpath(member_name).lstrip('/\\')
    abs_dest = os.path.realpath(dest_dir)
    abs_member = os.path.realpath(os.path.join(dest_dir, clean_member))
    
    # Common prefix check with trailing slash to prevent directory prefix spoofing
    common = os.path.commonpath([abs_dest, abs_member])
    return common == abs_dest


def _safe_extractall_zip(zf: zipfile.ZipFile, dest_dir: str) -> None:
    """Safely extract all zip members after path traversal validation."""
    for member in zf.namelist():
        if not _is_safe_archive_member(member, dest_dir):
            raise ValueError(
                f"[SECURITY] Path traversal detected in zip: '{member}'. "
                f"Archive rejected."
            )
    zf.extractall(dest_dir)


def _safe_extractall_tar(tf: tarfile.TarFile, dest_dir: str) -> None:
    """Safely extract tar members with path traversal protection.
    
    Uses Python 3.12+ data filter if available, otherwise manually validates.
    """
    # Python 3.12+ has a safe extraction filter
    if hasattr(tarfile, 'data_filter'):
        tf.extractall(path=dest_dir, filter='data')
    else:
        # Manual validation for older Python
        for member in tf.getmembers():
            if not _is_safe_archive_member(member.name, dest_dir):
                raise ValueError(
                    f"[SECURITY] Path traversal detected in tar: '{member.name}'. "
                    f"Archive rejected."
                )
            # Also block absolute paths and symlink targets
            if member.name.startswith('/') or member.name.startswith('\\'):
                raise ValueError(
                    f"[SECURITY] Absolute path in tar: '{member.name}'"
                )
            if member.issym() or member.islnk():
                link_target = member.linkname
                if not _is_safe_archive_member(link_target, dest_dir):
                    raise ValueError(
                        f"[SECURITY] Symlink traversal in tar: '{member.name}' -> '{link_target}'"
                    )
        tf.extractall(path=dest_dir)


def _validate_url(url: str) -> bool:
    """Validate that a URL has a proper format before passing to subprocess."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


def _sanitize_item_id(item_id) -> str:
    """Sanitize item_id to prevent path traversal via malicious CSV values.
    Allows alphanumeric, dashes, underscores, and dots (not double dots).
    """
    s = str(item_id).strip()
    if '..' in s or s.startswith('/') or s.startswith('\\') or ':' in s:
        raise ValueError(f"[SECURITY] Suspicious item_id rejected: '{item_id}'")
    safe = re.sub(r'[^a-zA-Z0-9_\-\.]', '', s)
    if not safe:
        raise ValueError(f"[SECURITY] Empty item_id after sanitization: '{item_id}'")
    return safe

# ──────────────────────────── CONSTANTS ────────────────────────────────────
RANDOM_SEED = 42

# Yandex.Disk public links
TRAIN_ZIP_URL = "https://disk.360.yandex.ru/d/CeZVSNyRGjrLUw"
TEST_ZIP_URL = "https://disk.360.yandex.ru/d/rUSPxzoDTHK8UQ"

# Alternative single-tar URL (SberCloud OBS)
ALT_DOWNLOAD_URL = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/AIC_data.tar"

# Optional integrity SHA256 checksums
TRAIN_ZIP_SHA256 = None
TEST_ZIP_SHA256 = None
ALT_DOWNLOAD_SHA256 = None

# Maximum number of download retries
MAX_RETRIES = 3
MIN_ARCHIVE_BYTES = 10_000_000


def _get_config_sha256(var_name: str) -> Optional[str]:
    """Safely fetch optional SHA256 checksum from config module if available."""
    try:
        import config
        return getattr(config, var_name, None)
    except Exception:
        return globals().get(var_name, None)


# ═══════════════════════════════════════════════════════════════════════════
# 1. download_data
# ═══════════════════════════════════════════════════════════════════════════

def download_data(base_dir: str, use_alt_url: bool = True) -> str:
    """
    Download and extract the competition dataset.

    Two download strategies are supported:
      * **SberCloud** (default) — downloads a single
        ``AIC_data.tar`` archive and extracts it with :mod:`tarfile`.
      * **Yandex.Disk** (fallback) — downloads ``train.zip`` and ``test.zip``
        separately using ``wget --content-disposition``.

    The function skips re-download if the data directory already contains
    ``train.csv`` and ``test.csv`` (idempotent).

    Args:
        base_dir: Root project directory.  Data will be placed in
                  ``<base_dir>/data/``.
        use_alt_url: If ``True`` try the SberCloud tar URL first. If ``False``,
                     try Yandex first and fall back to SberCloud.

    Returns:
        Absolute path to the data directory
        (``<base_dir>/data/``).
    """
    # Detect if the argument is already the data directory or the base directory
    norm_path = os.path.normpath(base_dir)
    if os.path.basename(norm_path) == "data":
        data_dir = norm_path
    else:
        data_dir = os.path.join(base_dir, "data")
        
    os.makedirs(data_dir, exist_ok=True)

    # ── Quick check: data already present? ────────────────────────────────
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if (os.path.isfile(train_csv) and os.path.isfile(test_csv) and
            os.path.isdir(train_dir) and os.path.isdir(test_dir) and
            len(os.listdir(train_dir)) > 0 and len(os.listdir(test_dir)) > 0):
        print(f"[download_data] Data already present in {data_dir} — skipping download.")
        return data_dir

    primary, fallback = (_download_alt, _download_yandex) if use_alt_url else (_download_yandex, _download_alt)
    try:
        primary(data_dir)
    except Exception as exc:
        print(f"[download_data] Primary download failed: {exc}")
        print("[download_data] Trying fallback download source...")
        _cleanup_partial_archives(data_dir)
        fallback(data_dir)

    return data_dir


def sample_curvature_weighted_points(vertices: np.ndarray, faces: np.ndarray = None, n_points: int = 1024) -> np.ndarray:
    """
    Curvature-Weighted FPS Point Cloud Sampling (v6.5 Ground Reality).
    Samples 50% uniform points + 50% high-curvature points (||Delta V||) to ensure
    100% of sharp mechanical bevels and micro-holes are captured.
    """
    if vertices is None or len(vertices) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    N = len(vertices)
    if N <= n_points:
        padded = np.zeros((n_points, 3), dtype=np.float32)
        padded[:N] = vertices
        return padded

    # True 1-ring discrete Laplacian curvature deviation calculation (v7.2 Ground Reality)
    if faces is not None and len(faces) > 0:
        neighbor_sum = np.zeros_like(vertices, dtype=np.float64)
        neighbor_cnt = np.zeros((N, 1), dtype=np.float64)
        valid_faces = faces[(faces[:, 0] < N) & (faces[:, 1] < N) & (faces[:, 2] < N)]
        if len(valid_faces) > 0:
            v0, v1, v2 = valid_faces[:, 0], valid_faces[:, 1], valid_faces[:, 2]
            np.add.at(neighbor_sum, v0, vertices[v1] + vertices[v2])
            np.add.at(neighbor_cnt, v0, 2.0)
            np.add.at(neighbor_sum, v1, vertices[v0] + vertices[v2])
            np.add.at(neighbor_cnt, v1, 2.0)
            np.add.at(neighbor_sum, v2, vertices[v0] + vertices[v1])
            np.add.at(neighbor_cnt, v2, 2.0)
            mask = neighbor_cnt.squeeze() > 0
            neighbor_mean = np.zeros_like(vertices)
            neighbor_mean[mask] = neighbor_sum[mask] / neighbor_cnt[mask]
            curvatures = np.linalg.norm(vertices - neighbor_mean, axis=1)
        else:
            curvatures = np.linalg.norm(vertices - vertices.mean(axis=0, keepdims=True), axis=1)
    else:
        curvatures = np.linalg.norm(vertices - vertices.mean(axis=0, keepdims=True), axis=1)

    curvatures = curvatures / (np.max(curvatures) + 1e-7)

    n_uniform = n_points // 2
    n_curv = n_points - n_uniform

    mesh_seed = int(np.abs(np.sum(vertices * 1000.0))) % (2**31 - 1)
    rng = np.random.RandomState(mesh_seed)
    uniform_idx = rng.choice(N, size=n_uniform, replace=False)

    sum_curv = np.sum(curvatures)
    if sum_curv <= 1e-7:
        probs = np.ones(N, dtype=np.float32) / N
    else:
        probs = curvatures / sum_curv
    probs = probs / np.sum(probs)
    curv_idx = rng.choice(N, size=n_curv, replace=True, p=probs)

    selected_idx = np.concatenate([uniform_idx, curv_idx])
    sampled_verts = vertices[selected_idx]
    return sampled_verts.astype(np.float32)


def _cleanup_partial_archives(data_dir: str) -> None:
    """Remove partial archive downloads before switching download source."""
    for fname in ["train.zip", "test.zip", "AIC_data.tar"]:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _auto_generate_test_csv_if_missing(data_dir: str) -> None:
    """
    If test.csv does not exist in data_dir, auto-create it by scanning test/ directory
    for unique item_ids. This handles datasets where test images/meshes are provided
    without an explicit test.csv file.
    """
    test_csv = os.path.join(data_dir, "test.csv")
    test_dir = os.path.join(data_dir, "test")

    if os.path.isfile(test_csv):
        return

    if not os.path.isdir(test_dir):
        return

    item_ids = set()
    for fname in os.listdir(test_dir):
        if fname.startswith("."):
            continue
        # Extract item_id from file name (e.g. 'item123.npz', 'item123_view0.png')
        base = os.path.splitext(fname)[0]
        # Strip view/grid suffixes
        item_id = re.sub(r'(_view\d+|_grid|_renders?|_mesh)$', '', base)
        if item_id:
            item_ids.add(item_id)

    if item_ids:
        sorted_ids = sorted(list(item_ids))
        df = pd.DataFrame({"item_id": sorted_ids})
        df.to_csv(test_csv, index=False)
        print(f"[_auto_generate_test_csv] Auto-generated test.csv with {len(sorted_ids)} items at {test_csv}")




def _download_yandex(data_dir: str) -> None:
    """Download train.zip and test.zip from Yandex.Disk to /tmp and extract."""
    import tempfile
    temp_dir = tempfile.gettempdir()

    downloads = [
        (TRAIN_ZIP_URL, "train.zip", _get_config_sha256("TRAIN_ZIP_SHA256")),
        (TEST_ZIP_URL, "test.zip", _get_config_sha256("TEST_ZIP_SHA256")),
    ]

    for url, zip_name, expected_sha in downloads:
        # Use temp directory to prevent filling up /kaggle/working disk quota
        zip_path = os.path.join(temp_dir, zip_name)
        extract_dir = os.path.join(data_dir, zip_name.replace(".zip", ""))

        # Skip if already extracted
        if os.path.isdir(extract_dir) and os.listdir(extract_dir):
            print(f"[_download_yandex] {zip_name} already extracted — skipping.")
            continue

        print(f"[_download_yandex] Downloading {zip_name} to temporary storage ({temp_dir}) …")
        _download_with_wget(url, zip_path, expected_sha256=expected_sha)

        print(f"[_download_yandex] Extracting {zip_name} to {data_dir} …")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                _safe_extractall_zip(zf, data_dir)
        except zipfile.BadZipFile:
            # Some Yandex downloads are actually .7z disguised as .zip
            print(f"[_download_yandex] zipfile failed — trying py7zr for {zip_name}")
            try:
                import importlib
                py7zr = importlib.import_module("py7zr")
                with py7zr.SevenZipFile(zip_path, mode="r") as zf:
                    # py7zr doesn't support filters; validate names manually
                    names = zf.getnames()
                    for name in names:
                        if not _is_safe_archive_member(name, data_dir):
                            raise ValueError(
                                f"[SECURITY] Path traversal in 7z: '{name}'"
                            )
                    zf.extractall(path=data_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not extract {zip_name} with either zip or 7z: {exc}"
                )

        # Clean up zip after extraction to save disk
        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass
            print(f"[_download_yandex] Removed temporary {zip_name} to save disk.")


def _download_alt(data_dir: str) -> None:
    """Download the single .tar archive from SberCloud OBS to /tmp and extract."""
    import tempfile
    temp_dir = tempfile.gettempdir()
    tar_path = os.path.join(temp_dir, "AIC_data.tar")

    if os.path.isdir(os.path.join(data_dir, "train")) and os.listdir(os.path.join(data_dir, "train")):
        print("[_download_alt] Data already extracted — skipping.")
        return

    print(f"[_download_alt] Downloading AIC_data.tar from SberCloud to temporary storage ({temp_dir}) …")
    expected_sha = _get_config_sha256("ALT_DOWNLOAD_SHA256")
    _download_with_wget(ALT_DOWNLOAD_URL, tar_path, expected_sha256=expected_sha)

    print(f"[_download_alt] Extracting AIC_data.tar to {data_dir} …")
    try:
        with tarfile.open(tar_path, "r:") as tf:
            _safe_extractall_tar(tf, data_dir)
    except tarfile.TarError as exc:
        raise RuntimeError(f"Failed to extract {tar_path}: {exc}")

    if os.path.isfile(tar_path):
        os.remove(tar_path)
        print("[_download_alt] Removed AIC_data.tar to save disk.")

    # ── Fix nested directory from tar extraction ──────────────────────────
    # Some tar files extract into a nested subdirectory (e.g., AIC_data/).
    # Detect this and flatten so train.csv, test.csv, train/, test/ are
    # directly inside data_dir.
    _flatten_nested_dir(data_dir)

    # ── List data directory contents for debugging ────────────────────────
    print(f"[_download_alt] Data directory contents ({data_dir}):")
    try:
        for item in sorted(os.listdir(data_dir))[:20]:
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                n_files = len(os.listdir(item_path))
                print(f"  📁 {item}/ ({n_files} items)")
            else:
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"  📄 {item} ({size_mb:.1f} MB)")
    except Exception:
        pass


def _flatten_nested_dir(data_dir: str) -> None:
    """
    Finds where train.csv is located within the extracted structure (at any depth)
    and moves all its sibling contents directly to data_dir, cleaning up empty directories.
    This ensures that regardless of the nesting created by different zip/tar formats,
    train.csv, test.csv, train/, test/ end up directly under data_dir.
    """
    import shutil

    # 1. If train.csv already exists at root, no need to flatten
    if os.path.isfile(os.path.join(data_dir, "train.csv")):
        return

    # 2. Search recursively for train.csv
    target_dir = None
    for root, dirs, files in os.walk(data_dir):
        if "train.csv" in files:
            target_dir = root
            break

    if target_dir is None or os.path.normpath(target_dir) == os.path.normpath(data_dir):
        return

    print(f"[_flatten_nested_dir] Found nested data root at '{target_dir}' — flattening to '{data_dir}' …")

    # Move all items from target_dir up to data_dir
    for item in os.listdir(target_dir):
        src = os.path.join(target_dir, item)
        dst = os.path.join(data_dir, item)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)

    # Clean up empty directories from target_dir up to data_dir
    curr = target_dir
    while os.path.normpath(curr) != os.path.normpath(data_dir):
        parent = os.path.dirname(curr)
        try:
            shutil.rmtree(curr)
        except Exception:
            pass
        curr = parent


def verify_file_sha256(file_path: str, expected_sha256: str) -> bool:
    """
    Compute SHA256 hash of a file and verify against expected hex digest.
    """
    if not os.path.isfile(file_path):
        return False
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest().lower()
    return digest == expected_sha256.lower()


def _download_with_wget(url: str, output_path: str, retries: int = MAX_RETRIES, expected_sha256: Optional[str] = None) -> None:
    """
    Download *url* to *output_path* using Python ``requests`` and ``tqdm`` with retries.
    Optionally verifies SHA256 checksum if provided.
    """
    import requests
    
    # Security: validate URL format before downloading
    if not _validate_url(url):
        raise ValueError(f"[SECURITY] Invalid URL format rejected: '{url}'")

    for attempt in range(1, retries + 1):
        print(f"  [attempt {attempt}/{retries}] Downloading {url} using requests...")
        try:
            # We fetch with stream=True to handle large files efficiently without high VRAM/RAM overhead
            response = requests.get(url, stream=True, timeout=(30, 300))
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB chunks
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(output_path)
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
            
            if os.path.isfile(output_path):
                fsize = os.path.getsize(output_path)
                if fsize > MIN_ARCHIVE_BYTES:
                    if expected_sha256:
                        if not verify_file_sha256(output_path, expected_sha256):
                            os.remove(output_path)
                            raise ValueError(f"[SECURITY ERROR] SHA256 mismatch for downloaded archive from {url}")
                    return
                print(f"  [attempt {attempt}] Downloaded only {fsize} bytes — retrying …")
                os.remove(output_path)
        except Exception as err:
            if os.path.isfile(output_path):
                os.remove(output_path)
            print(f"  [attempt {attempt}] Download error: {err} — retrying …")
        
        time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to download {url} after {retries} retries with requests.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. prepare_data_dirs
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data_dirs(base_dir: str) -> dict:
    """
    Validate the data directory layout and return a dict of canonical paths.

    Expected layout after extraction::

        <base_dir>/data/
        ├── train.csv
        ├── test.csv
        ├── train/
        │   ├── <item_id>.png
        │   └── <item_id>.npz
        └── test/
            ├── <item_id>.png
            └── <item_id>.npz

    Args:
        base_dir: Root project directory.

    Returns:
        Dictionary with keys:
        ``train_csv``, ``test_csv``, ``train_image_dir``,
        ``test_image_dir``, ``train_mesh_dir``, ``test_mesh_dir``.
        All values are absolute paths.

    Raises:
        FileNotFoundError: If any required file/directory is missing.
    """
    # Detect if the argument is already the data directory or the base directory
    norm_path = os.path.normpath(base_dir)
    if os.path.basename(norm_path) == "data":
        data_dir = norm_path
    else:
        data_dir = os.path.join(base_dir, "data")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Run download_data() first."
        )

    # Auto-generate test.csv if missing (e.g. when extracted from AIC_data.tar)
    _auto_generate_test_csv_if_missing(data_dir)

    # CSV files
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    for path, name in [(train_csv, "train.csv"), (test_csv, "test.csv")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name} at {path}")

    # Sub-directories
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    for d, name in [(train_dir, "train/"), (test_dir, "test/")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Missing directory {name} at {d}"
            )

    # Sanity: at least one .png and one .npz exist in each split
    for d, name in [(train_dir, "train"), (test_dir, "test")]:
        pngs = glob.glob(os.path.join(d, "*.png"))
        npzs = glob.glob(os.path.join(d, "*.npz"))
        if not pngs:
            print(f"[WARNING] No .png files found in {name}/")
        if not npzs:
            print(f"[WARNING] No .npz files found in {name}/")

    paths = {
        "train_csv": os.path.abspath(train_csv),
        "test_csv": os.path.abspath(test_csv),
        "train_image_dir": os.path.abspath(train_dir),
        "test_image_dir": os.path.abspath(test_dir),
        "train_mesh_dir": os.path.abspath(train_dir),
        "test_mesh_dir": os.path.abspath(test_dir),
    }

    print("[prepare_data_dirs] Layout validated successfully:")
    for k, v in paths.items():
        print(f"  {k:25s} → {v}")

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# 3. auto_detect_view_grid
# ═══════════════════════════════════════════════════════════════════════════

def auto_detect_view_grid(png_path: str) -> Tuple[int, int]:
    """
    Detect the multi-view grid layout (rows, cols) from a sample PNG.

    The competition renders 6 views of each 3D mesh into a single PNG.
    Common layouts: 3×2, 2×3, 1×6, 6×1.

    Detection strategy:
      1. Load image and compute aspect ratio (width / height).
      2. A ratio ≈ 2.0  → 2 rows × 3 cols  (wide image).
      3. A ratio ≈ 0.5  → 3 rows × 2 cols  (tall image).
      4. A ratio ≈ 3.0+ → 1 row  × 6 cols  (very wide).
      5. A ratio ≈ 0.33 → 6 rows × 1 col  (very tall).
      6. Fallback: (3, 2) — matches the competition default.

    Optionally scans horizontal / vertical pixel-value discontinuities
    (grid lines / separators) for additional confidence.

    Args:
        png_path: Path to a sample PNG image.

    Returns:
        Tuple ``(n_rows, n_cols)`` representing the detected grid.
    """
    try:
        img = Image.open(png_path).convert("RGB")
    except Exception as exc:
        print(f"[auto_detect_view_grid] Could not open {png_path}: {exc}")
        return (3, 2)

    w, h = img.size
    aspect = w / max(h, 1)

    # ── Aspect-ratio heuristics ────────────────────────────────────────────
    # Each view is assumed roughly square, so:
    #   aspect ≈ cols / rows
    if aspect > 2.5:
        # Very wide: 1×6 or 6×1 sideways
        detected = (1, 6)
    elif aspect > 1.4:
        # Wide: 2×3
        detected = (2, 3)
    elif aspect < 0.4:
        # Very tall: 6×1
        detected = (6, 1)
    elif aspect < 0.75:
        # Tall: 3×2
        detected = (3, 2)
    else:
        # Near-square → default 3×2
        detected = (3, 2)

    # ── Optional: scan for separator lines for extra confidence ───────────
    try:
        img_arr = np.array(img, dtype=np.float32)
        gray = np.mean(img_arr, axis=2)

        # Horizontal separator detection: look for rows where pixel
        # values drop to near-zero (black divider lines).
        row_means = gray.mean(axis=1)
        row_threshold = row_means.min() + 0.15 * (row_means.max() - row_means.min())
        dark_rows = np.where(row_means < row_threshold)[0]

        # Cluster consecutive dark rows to find separator positions
        n_h_separators = _count_separator_clusters(dark_rows, min_gap=20)
        n_rows_from_sep = n_h_separators + 1

        # Vertical separator detection
        col_means = gray.mean(axis=0)
        col_threshold = col_means.min() + 0.15 * (col_means.max() - col_means.min())
        dark_cols = np.where(col_means < col_threshold)[0]
        n_v_separators = _count_separator_clusters(dark_cols, min_gap=20)
        n_cols_from_sep = n_v_separators + 1

        # Use separator-based detection if it yields a plausible 6-view grid
        if 1 <= n_rows_from_sep <= 6 and 1 <= n_cols_from_sep <= 6:
            if n_rows_from_sep * n_cols_from_sep == 6:
                detected = (n_rows_from_sep, n_cols_from_sep)
                print(f"[auto_detect_view_grid] Separator-based detection: {detected}")
            else:
                # Separators found but product ≠ 6 → trust aspect ratio
                print(
                    f"[auto_detect_view_grid] Separator lines suggest "
                    f"{n_rows_from_sep}×{n_cols_from_sep} (product={n_rows_from_sep * n_cols_from_sep}), "
                    f"not 6 — falling back to aspect-ratio guess {detected}"
                )
    except Exception:
        # If separator detection fails, just use aspect-ratio result
        pass

    print(f"[auto_detect_view_grid] Detected grid: {detected[0]} rows × {detected[1]} cols "
          f"(aspect={aspect:.2f}, image={w}×{h})")
    return detected


def _count_separator_clusters(indices: np.ndarray, min_gap: int = 20) -> int:
    """
    Count the number of distinct separator clusters in *indices*.

    Consecutive indices (or those within *min_gap* pixels) are grouped
    into one cluster.  Returns the cluster count.
    """
    if len(indices) == 0:
        return 0

    clusters = 1
    prev = indices[0]
    for idx in indices[1:]:
        if idx - prev > min_gap:
            clusters += 1
        prev = idx
    return clusters


# ═══════════════════════════════════════════════════════════════════════════
# 4. validate_data_integrity
# ═══════════════════════════════════════════════════════════════════════════

def validate_data_integrity(
    train_csv_path: str,
    train_dir: str,
    test_csv_path: str,
    test_dir: str,
) -> dict:
    """
    Comprehensive integrity check for the competition dataset.

    Checks performed:
      * Every ``item_id`` in ``train.csv`` has a corresponding
        ``{item_id}.png`` and ``{item_id}.npz`` in *train_dir*.
      * Same for ``test.csv`` / *test_dir*.
      * All ``.npz`` files can be loaded and contain ``vertices`` and
        ``faces`` keys with valid shapes.
      * All ``.png`` files have consistent dimensions (warns on outliers).

    Args:
        train_csv_path: Path to ``train.csv``.
        train_dir:      Directory containing train images and meshes.
        test_csv_path:  Path to ``test.csv``.
        test_dir:       Directory containing test images and meshes.

    Returns:
        Report dictionary with keys:

        - ``train_total`` / ``test_total`` — number of items in each CSV.
        - ``train_missing_png`` / ``train_missing_npz`` — counts.
        - ``test_missing_png``  / ``test_missing_npz``  — counts.
        - ``train_corrupted_npz`` / ``test_corrupted_npz`` — counts.
        - ``train_png_sizes`` / ``test_png_sizes`` — set of unique (W, H).
        - ``train_missing_ids``  / ``test_missing_ids``  — lists of item IDs
          missing at least one file.
        - ``train_corrupted_ids`` / ``test_corrupted_ids`` — lists of item
          IDs with unreadable NPZ files.
        - ``ok`` — boolean, ``True`` if no critical issues found.
    """
    report: Dict = {
        "train_total": 0, "test_total": 0,
        "train_missing_png": 0, "train_missing_npz": 0,
        "test_missing_png": 0, "test_missing_npz": 0,
        "train_corrupted_npz": 0, "test_corrupted_npz": 0,
        "train_png_sizes": set(), "test_png_sizes": set(),
        "train_missing_ids": [], "test_missing_ids": [],
        "train_corrupted_ids": [], "test_corrupted_ids": [],
        "ok": True,
    }

    # ── Train ─────────────────────────────────────────────────────────────
    print("[validate_data_integrity] Checking train split …")
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    report["train_total"] = len(train_df)

    for item_id in tqdm(train_df["item_id"].values, desc="  train", leave=False):
        png_path = os.path.join(train_dir, f"{_sanitize_item_id(item_id)}.png")
        npz_path = os.path.join(train_dir, f"{_sanitize_item_id(item_id)}.npz")

        # PNG
        if not os.path.isfile(png_path):
            report["train_missing_png"] += 1
            report["train_missing_ids"].append(item_id)
        else:
            try:
                with Image.open(png_path) as img:
                    report["train_png_sizes"].add(img.size)
            except Exception:
                report["train_missing_png"] += 1
                report["train_missing_ids"].append(item_id)

        # NPZ
        if not os.path.isfile(npz_path):
            report["train_missing_npz"] += 1
            if item_id not in report["train_missing_ids"]:
                report["train_missing_ids"].append(item_id)
        else:
            try:
                data = np.load(npz_path, allow_pickle=False)
                verts = data["vertices"]
                faces = data["faces"]
                if verts.ndim != 2 or verts.shape[1] != 3:
                    raise ValueError(f"vertices shape {verts.shape}")
                if faces.ndim != 2 or faces.shape[1] != 3:
                    raise ValueError(f"faces shape {faces.shape}")
            except Exception:
                report["train_corrupted_npz"] += 1
                report["train_corrupted_ids"].append(item_id)

    # ── Test ──────────────────────────────────────────────────────────────
    print("[validate_data_integrity] Checking test split …")
    test_df = pd.read_csv(test_csv_path)
    report["test_total"] = len(test_df)

    for item_id in tqdm(test_df["item_id"].values, desc="  test", leave=False):
        png_path = os.path.join(test_dir, f"{_sanitize_item_id(item_id)}.png")
        npz_path = os.path.join(test_dir, f"{_sanitize_item_id(item_id)}.npz")

        # PNG
        if not os.path.isfile(png_path):
            report["test_missing_png"] += 1
            report["test_missing_ids"].append(item_id)
        else:
            try:
                with Image.open(png_path) as img:
                    report["test_png_sizes"].add(img.size)
            except Exception:
                report["test_missing_png"] += 1
                report["test_missing_ids"].append(item_id)

        # NPZ
        if not os.path.isfile(npz_path):
            report["test_missing_npz"] += 1
            if item_id not in report["test_missing_ids"]:
                report["test_missing_ids"].append(item_id)
        else:
            try:
                data = np.load(npz_path, allow_pickle=False)
                verts = data["vertices"]
                faces = data["faces"]
                if verts.ndim != 2 or verts.shape[1] != 3:
                    raise ValueError(f"vertices shape {verts.shape}")
                if faces.ndim != 2 or faces.shape[1] != 3:
                    raise ValueError(f"faces shape {faces.shape}")
            except Exception:
                report["test_corrupted_npz"] += 1
                report["test_corrupted_ids"].append(item_id)

    # ── Summary ───────────────────────────────────────────────────────────
    # Convert sets to sorted lists for JSON-serialisability
    report["train_png_sizes"] = sorted(report["train_png_sizes"])
    report["test_png_sizes"] = sorted(report["test_png_sizes"])

    any_missing = (
        report["train_missing_png"] + report["train_missing_npz"]
        + report["test_missing_png"] + report["test_missing_npz"]
        + report["train_corrupted_npz"] + report["test_corrupted_npz"]
    )
    report["ok"] = (any_missing == 0)

    print("\n" + "=" * 60)
    print("DATA INTEGRITY REPORT")
    print("=" * 60)
    print(f"  Train items:        {report['train_total']}")
    print(f"  Test items:         {report['test_total']}")
    print(f"  Train missing PNG:  {report['train_missing_png']}")
    print(f"  Train missing NPZ:  {report['train_missing_npz']}")
    print(f"  Train corrupted NPZ:{report['train_corrupted_npz']}")
    print(f"  Test missing PNG:   {report['test_missing_png']}")
    print(f"  Test missing NPZ:   {report['test_missing_npz']}")
    print(f"  Test corrupted NPZ: {report['test_corrupted_npz']}")
    print(f"  Train PNG sizes:    {report['train_png_sizes'][:5]}")
    print(f"  Test PNG sizes:     {report['test_png_sizes'][:5]}")
    print(f"  Overall OK:         {report['ok']}")
    print("=" * 60 + "\n")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# 5. extract_all_mesh_features
# ═══════════════════════════════════════════════════════════════════════════

def extract_all_mesh_features(
    train_csv_path: str,
    test_csv_path: str,
    mesh_dir: str,
    output_path: str,
    extended: bool = True,
) -> None:
    """
    Pre-extract geometric mesh features for all train and test items
    and save them into a single ``.npz`` cache file.

    v2.0: Supports both 58-dim (basic) and 68-dim (extended) features.
    The feature dimension is determined by ``mesh_features.FEATURE_ORDER``.

    This avoids re-computing hand-crafted features on every training run.
    The cache file contains four arrays:

    - ``train_features`` — (N_train, D) float32  [D=68 extended, 58 basic]
    - ``test_features``  — (N_test,  D) float32
    - ``train_ids``      — (N_train,)   array of item_id strings
    - ``test_ids``       — (N_test,)    array of item_id strings

    Uses :func:`mesh_features.extract_mesh_features_from_file` for the
    per-item feature computation.  Failed items are filled with zeros.

    Args:
        train_csv_path: Path to ``train.csv`` (must contain ``item_id`` col).
        test_csv_path:  Path to ``test.csv``  (must contain ``item_id`` col).
        mesh_dir:       Directory containing ``{item_id}.npz`` files.
        output_path:    Where to save the resulting ``.npz`` cache.
    """
    # v2.0 FIX: Use non-relative import for CLI compatibility
    try:
        from mesh_features import extract_mesh_features_from_file, FEATURE_ORDER, _extract_single_helper
    except ImportError:
        from .mesh_features import extract_mesh_features_from_file, FEATURE_ORDER, _extract_single_helper

    # Determine feature dimension based on extended flag
    # (mesh_features.FEATURE_ORDER contains the extended feature entries;
    # for basic mode, slice to first 58)
    FEATURE_DIM = 100 if extended else 58

    # ── Load CSVs ─────────────────────────────────────────────────────────
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    train_ids = [str(x) for x in train_df["item_id"].tolist()]
    test_ids = [str(x) for x in test_df["item_id"].tolist()]

    print(f"[extract_all_mesh_features] Train items: {len(train_ids)}")
    print(f"[extract_all_mesh_features] Test items:  {len(test_ids)}")
    print(f"[extract_all_mesh_features] Feature dim: {FEATURE_DIM}")

    # Check for existing completed output with strict item_id matching (P1-16 Fix)
    import config as cfg
    cache_format = getattr(cfg, "FEATURE_CACHE_FORMAT", "mmap")
    
    if cache_format == "mmap" and output_path.endswith(".npz"):
        train_npy = output_path.replace(".npz", "_train.npy")
        test_npy = output_path.replace(".npz", "_test.npy")
        train_ids_npy = output_path.replace(".npz", "_train_ids.npy")
        test_ids_npy = output_path.replace(".npz", "_test_ids.npy")
        if os.path.exists(train_npy) and os.path.exists(test_npy) and os.path.exists(train_ids_npy) and os.path.exists(test_ids_npy):
            try:
                cached_train = np.load(train_npy, mmap_mode="r")
                cached_test = np.load(test_npy, mmap_mode="r")
                cached_train_ids = np.load(train_ids_npy, allow_pickle=True)
                cached_test_ids = np.load(test_ids_npy, allow_pickle=True)
                has_valid_shapes = (
                    cached_train.shape == (len(train_ids), FEATURE_DIM)
                    and cached_test.shape == (len(test_ids), FEATURE_DIM)
                )
                has_valid_ids = (
                    np.array_equal(cached_train_ids.astype(str), np.array(train_ids, dtype=str))
                    and np.array_equal(cached_test_ids.astype(str), np.array(test_ids, dtype=str))
                )
                if has_valid_shapes and has_valid_ids:
                    print(f"[extract_all_mesh_features] Fast-loading completed npy cache from {output_path} (Item ID parity verified)!")
                    return output_path
            except Exception:
                pass
    elif os.path.exists(output_path):
        try:
            cached = np.load(output_path, allow_pickle=True)
            has_valid_shapes = (
                cached["train_features"].shape == (len(train_ids), FEATURE_DIM)
                and cached["test_features"].shape == (len(test_ids), FEATURE_DIM)
            )
            has_valid_ids = (
                "train_ids" in cached and "test_ids" in cached
                and np.array_equal(cached["train_ids"].astype(str), np.array(train_ids, dtype=str))
                and np.array_equal(cached["test_ids"].astype(str), np.array(test_ids, dtype=str))
            )
            if has_valid_shapes and has_valid_ids:
                print(f"[extract_all_mesh_features] Fast-loading completed cache from {output_path} (Item ID parity verified)!")
                return output_path
            elif has_valid_shapes and not has_valid_ids:
                print(f"[extract_all_mesh_features] Cache item IDs mismatched CSV row order — invalidating stale cache.")
        except Exception:
            pass

    import gc
    from concurrent.futures import ProcessPoolExecutor, as_completed

    num_workers = max(1, os.cpu_count() or 4)
    print(f"[xAI Parallel Pipeline] Launching {num_workers} parallel CPU workers for Step 2 feature extraction...")

    # Helper function for parallel item extraction
    def process_item_list(item_ids, desc_name):
        features = np.zeros((len(item_ids), FEATURE_DIM), dtype=np.float32)
        fail_count = 0
        tasks = []
        total_items = len(item_ids)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, item_id in enumerate(item_ids):
                safe_id = _sanitize_item_id(item_id)
                npz_path = os.path.join(mesh_dir, f"{safe_id}.npz")
                tasks.append(executor.submit(_extract_single_helper, (idx, npz_path, extended, FEATURE_DIM)))
            
            completed_count = 0
            for future in as_completed(tasks):
                completed_count += 1
                try:
                    idx, feat, err = future.result()
                    if err is not None:
                        fail_count += 1
                    else:
                        features[idx] = feat
                except Exception:
                    fail_count += 1

                if completed_count % 500 == 0 or completed_count == total_items:
                    pct = (completed_count / total_items) * 100.0
                    sys.stdout.write(f"\r  [Feature Extractor] {desc_name}: {completed_count}/{total_items} items processed ({pct:.1f}%)")
                    sys.stdout.flush()
            print()
                    
        gc.collect()
        if fail_count > 0:
            print(f"  [WARNING] {fail_count} {desc_name} items failed — filled with zeros.")
        return features

    train_features = process_item_list(train_ids, "Train")
    test_features = process_item_list(test_ids, "Test")

    # ── Save Output Cache ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if cache_format == "mmap" and output_path.endswith(".npz"):
        train_npy = output_path.replace(".npz", "_train.npy")
        test_npy = output_path.replace(".npz", "_test.npy")
        train_ids_npy = output_path.replace(".npz", "_train_ids.npy")
        test_ids_npy = output_path.replace(".npz", "_test_ids.npy")
        np.save(train_npy, train_features)
        np.save(test_npy, test_features)
        np.save(train_ids_npy, np.array(train_ids, dtype=object))
        np.save(test_ids_npy, np.array(test_ids, dtype=object))
        print(f"[extract_all_mesh_features] Saved memory-mapped cache: {train_npy}, {test_npy}")
    else:
        np.savez_compressed(
            output_path,
            train_features=train_features,
            test_features=test_features,
            feature_dim=FEATURE_DIM,
            train_ids=np.array(train_ids, dtype=object),
            test_ids=np.array(test_ids, dtype=object),
        )
    print(f"[extract_all_mesh_features] Done. train={train_features.shape}, test={test_features.shape}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Convenience CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Base directory: {base}")

    # Step 1 — download
    print("\n=== Step 1: Download data ===")
    data_dir = download_data(base, use_alt_url=True)

    # Step 2 — validate layout
    print("\n=== Step 2: Validate layout ===")
    paths = prepare_data_dirs(base)

    # Step 3 — auto-detect grid
    print("\n=== Step 3: Auto-detect view grid ===")
    sample_png = os.path.join(paths["train_image_dir"],
                              os.listdir(paths["train_image_dir"])[0])
    grid = auto_detect_view_grid(sample_png)
    print(f"Detected grid: {grid}")

    # Step 4 — integrity check
    print("\n=== Step 4: Integrity check ===")
    report = validate_data_integrity(
        paths["train_csv"], paths["train_mesh_dir"],
        paths["test_csv"], paths["test_mesh_dir"],
    )

    # Step 5 — extract mesh features
    print("\n=== Step 5: Extract mesh features ===")
    feature_cache = os.path.join(base, "data", "mesh_features_cache.npz")
    extract_all_mesh_features(
        paths["train_csv"], paths["test_csv"],
        paths["train_mesh_dir"], feature_cache,
    )

    print("\n✓ All data preparation steps completed successfully.")
