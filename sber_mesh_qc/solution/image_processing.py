"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Image Processing  [v7.2 Master Engine]
===============================================================================
Handles loading, splitting 6-view PNG renders, Sobel pseudo-normal calculation,
Direct CPU orthographic fast rasterization, and multi-view data augmentation.
===============================================================================
"""

import os
import re
import hashlib
import numpy as np
from PIL import Image, ImageFile
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Prevent decompression bomb DOS attacks by setting a strict but reasonable pixel limit
Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True


from data_utils import _sanitize_item_id


class DirectMeshRasterizer:
    """
    Direct CPU Orthographic Fast Mesh Rasterizer (v7.3).
    Generates 6-view 5-channel Depth + true normals + mask (6, 5, 224, 224) directly from raw vertices/faces
    in <10ms without requiring external Blender or OpenGL offscreen display windows.
    """

    @staticmethod
    def rasterize_views(
        vertices: np.ndarray, faces: np.ndarray = None, img_size: int = 224
    ) -> torch.Tensor:
        if vertices is None or len(vertices) == 0:
            return torch.zeros((6, 5, img_size, img_size), dtype=torch.float32)

        centered = vertices - vertices.mean(axis=0)
        max_bound = np.max(np.abs(centered)) + 1e-7
        norm_verts = centered / max_bound

        # Compute vertex normals
        if faces is not None and len(faces) > 0 and len(vertices) > 0:
            N = len(norm_verts)
            valid_f = faces[(faces[:, 0] < N) & (faces[:, 1] < N) & (faces[:, 2] < N)]
            if len(valid_f) > 0:
                v0 = norm_verts[valid_f[:, 0]]
                v1 = norm_verts[valid_f[:, 1]]
                v2 = norm_verts[valid_f[:, 2]]
                cross = np.cross(v1 - v0, v2 - v0)
                face_normals = cross / (
                    np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10
                )

                # Accumulate vertex normals
                vert_normals = np.zeros_like(norm_verts)
                for i in range(3):
                    np.add.at(vert_normals, valid_f[:, i], face_normals)
                vert_norms = np.linalg.norm(vert_normals, axis=1, keepdims=True) + 1e-10
                vert_normals = vert_normals / vert_norms

                # Interpolate to face centers and edge midpoints
                n0 = vert_normals[valid_f[:, 0]]
                n1 = vert_normals[valid_f[:, 1]]
                n2 = vert_normals[valid_f[:, 2]]
                center_n = (n0 + n1 + n2) / 3.0
                m01_n = (n0 + n1) / 2.0
                m12_n = (n1 + n2) / 2.0
                m20_n = (n2 + n0) / 2.0

                center_n = center_n / (
                    np.linalg.norm(center_n, axis=1, keepdims=True) + 1e-10
                )
                m01_n = m01_n / (np.linalg.norm(m01_n, axis=1, keepdims=True) + 1e-10)
                m12_n = m12_n / (np.linalg.norm(m12_n, axis=1, keepdims=True) + 1e-10)
                m20_n = m20_n / (np.linalg.norm(m20_n, axis=1, keepdims=True) + 1e-10)

                dense_verts = np.concatenate(
                    [
                        norm_verts,
                        (v0 + v1 + v2) / 3.0,
                        (v0 + v1) / 2.0,
                        (v1 + v2) / 2.0,
                        (v2 + v0) / 2.0,
                    ],
                    axis=0,
                )
                dense_normals = np.concatenate(
                    [vert_normals, center_n, m01_n, m12_n, m20_n], axis=0
                )
            else:
                dense_verts = norm_verts
                dense_normals = np.zeros_like(norm_verts)
        else:
            dense_verts = norm_verts
            dense_normals = np.zeros_like(norm_verts)

        directions = [
            (0, 1, 2),  # +Z (Front): u=X, v=Y, depth=+Z
            (0, 1, 2),  # -Z (Back): u=-X, v=Y, depth=-Z
            (2, 1, 0),  # +X (Right): u=-Z, v=Y, depth=+X
            (2, 1, 0),  # -X (Left): u=+Z, v=Y, depth=-X
            (0, 2, 1),  # +Y (Top): u=X, v=+Z, depth=+Y
            (0, 2, 1),  # -Y (Bottom): u=X, v=-Z, depth=-Y
        ]
        flip_u = [False, True, True, False, False, False]
        flip_v = [False, False, False, False, False, True]
        flip_z = [False, True, False, True, False, True]

        views_tensor = []
        for i, (u_axis, v_axis, depth_axis) in enumerate(directions):
            depth_map = np.zeros((img_size, img_size), dtype=np.float32)
            normal_map = np.zeros((img_size, img_size, 3), dtype=np.float32)
            mask = np.zeros((img_size, img_size), dtype=np.float32)

            u_vals = dense_verts[:, u_axis]
            v_vals = dense_verts[:, v_axis]
            if flip_u[i]:
                u_vals = -u_vals
            if flip_v[i]:
                v_vals = -v_vals

            u_coords = ((u_vals + 1.0) * 0.5 * (img_size - 1)).astype(int)
            v_coords = ((v_vals + 1.0) * 0.5 * (img_size - 1)).astype(int)
            u_coords = np.clip(u_coords, 0, img_size - 1)
            v_coords = np.clip(v_coords, 0, img_size - 1)

            z_vals = dense_verts[:, depth_axis]
            if flip_z[i]:
                z_vals = -z_vals
            z_vals = (z_vals + 1.0) * 0.5

            # Project normals to camera frame
            nu = dense_normals[:, u_axis]
            if flip_u[i]:
                nu = -nu
            nv = dense_normals[:, v_axis]
            if flip_v[i]:
                nv = -nv
            nz = dense_normals[:, depth_axis]
            if flip_z[i]:
                nz = -nz

            camera_normals = np.stack([nu, nv, nz], axis=1)

            # Sort by depth to implement Z-buffer writing canonically
            order = np.argsort(z_vals)
            u_sorted = u_coords[order]
            v_sorted = v_coords[order]
            z_sorted = z_vals[order]
            normals_sorted = camera_normals[order]

            depth_map[v_sorted, u_sorted] = z_sorted
            normal_map[v_sorted, u_sorted] = normals_sorted
            mask[v_sorted, u_sorted] = 1.0

            five_chan = np.empty((5, img_size, img_size), dtype=np.float32)
            five_chan[0] = depth_map
            five_chan[1:4] = normal_map.transpose(2, 0, 1)
            five_chan[4] = mask
            views_tensor.append(torch.from_numpy(five_chan))

        return torch.stack(views_tensor, dim=0)


def _smooth_1d(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    1D uniform smoothing with fallback from scipy to numpy.
    scipy.ndimage.uniform_filter1d is faster but not always available.
    """
    try:
        from scipy.ndimage import uniform_filter1d

        return uniform_filter1d(arr, kernel_size)
    except ImportError:
        # Pure numpy fallback: running mean
        kernel = np.ones(kernel_size) / kernel_size
        # Pad to avoid edge effects
        pad = kernel_size // 2
        padded = np.pad(arr, pad, mode="reflect")
        return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def auto_detect_grid(image: Image.Image, n_views: int = 6) -> tuple:
    """
    OVERCOME LIMITATION #3: Auto-detect the grid layout of multi-view renders.

    Multi-strategy approach:
    1. Try intensity-based gap detection (primary, most accurate)
    2. Fall back to aspect-ratio heuristic
    3. Final fallback: (3, 2) — competition default

    Args:
        image: PIL Image of the full multi-view rendering
        n_views: expected number of views (default 6)

    Returns:
        (rows, cols) tuple for the grid layout
    """
    w, h = image.size

    # ── Strategy 1: Aspect ratio heuristic (always works) ─────────────────
    aspect = w / max(h, 1)
    if aspect > 2.5:
        aspect_guess = (1, n_views)
    elif aspect > 1.4:
        aspect_guess = (2, n_views // 2)
    elif aspect < 0.4:
        aspect_guess = (n_views, 1)
    elif aspect < 0.75:
        aspect_guess = (n_views // 2, 2)
    else:
        aspect_guess = (3, 2)

    # ── Strategy 2: Intensity-based gap detection ─────────────────────────
    try:
        gray = np.array(image.convert("L"), dtype=np.float32)

        # Compute row and column mean intensities
        row_means = gray.mean(axis=1)  # (H,)
        col_means = gray.mean(axis=0)  # (W,)

        # Smooth to reduce noise
        kernel_size = max(3, min(w, h) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        row_smooth = _smooth_1d(row_means, kernel_size)
        col_smooth = _smooth_1d(col_means, kernel_size)

        # Find gaps: regions where intensity drops below threshold
        row_thresh = row_smooth.mean() * 0.85
        col_thresh = col_smooth.mean() * 0.85

        def count_gaps(profile, thresh, length):
            below = profile < thresh
            gaps = 0
            in_gap = below[0] if length > 0 else False
            for i in range(1, length):
                currently_below = below[i]
                if currently_below and not in_gap:
                    gaps += 1
                in_gap = currently_below
            return gaps

        n_row_gaps = count_gaps(row_smooth, row_thresh, h)
        n_col_gaps = count_gaps(col_smooth, col_thresh, w)

        n_row_separators = max(n_row_gaps - 1, 0)
        n_col_separators = max(n_col_gaps - 1, 0)
        rows = n_row_separators + 1
        cols = n_col_separators + 1

        # Validate
        if rows * cols == n_views and 1 <= rows <= n_views and 1 <= cols <= n_views:
            return (rows, cols)

        # Try all valid factorizations that match aspect ratio
        for r in range(1, n_views + 1):
            if n_views % r == 0:
                c = n_views // r
                grid_aspect = (w / c) / (h / r) if h > 0 and c > 0 else 1.0
                if 0.5 < grid_aspect < 2.0:
                    return (r, c)

    except Exception:
        pass  # Fall through to aspect ratio

    return aspect_guess


def split_six_views(image: Image.Image, grid: tuple = None) -> list:
    """
    Split a multi-view PNG into individual view images.

    OVERCOME LIMITATION #3: If grid is None, auto-detect the layout.

    Args:
        image: PIL Image of the full multi-view rendering
        grid: (rows, cols) layout of views. If None, auto-detect.

    Returns:
        list of PIL Images, one per view
    """
    if grid is None:
        try:
            grid = auto_detect_grid(image, n_views=6)
        except Exception:
            grid = (3, 2)

    w, h = image.size
    rows, cols = grid
    view_w = w // cols
    view_h = h // rows

    views = []
    for r in range(rows):
        for c in range(cols):
            left = c * view_w
            upper = r * view_h
            right = left + view_w
            lower = upper + view_h
            view = image.crop((left, upper, right, lower))
            views.append(view)

    return views


class MeshQualityDataset(Dataset):
    """
    Dataset for 3D Mesh Quality Control.

    Loads 6-view images and optionally mesh features for each item.

    OVERCOME LIMITATION #5: Supports VIEWS_TRAIN_SUBSAMPLE to randomly
    select fewer than 6 views during training, reducing memory usage.
    """

    def __init__(
        self,
        item_ids: list,
        labels_df=None,
        image_dir: str = "",
        mesh_features: np.ndarray = None,
        point_clouds: np.ndarray = None,
        image_size: int = 224,
        view_grid: tuple = None,  # None = auto-detect per image (Limitation #3)
        augment: bool = False,
        aug_config: dict = None,
        views_subsample: int = None,  # Limitation #5: e.g., 4 means random 4 of 6
        max_corrupt_ratio: float = 0.01,
        use_image: bool = True,
        use_mesh_features: bool = True,
        geometry_mean=None,
        geometry_std=None,
        geometry_cache_dir: str = None,
    ):
        """
        Args:
            item_ids: list of item_id strings
            labels_df: DataFrame with labels (None for test set).
                       v2.0.1: If it has 'item_id' as a column, we build an
                       O(1) lookup dict for fast __getitem__ access.
            image_dir: directory containing {item_id}.png files
            mesh_features: (N, D) pre-extracted mesh features
            point_clouds: (N, P, 3) pre-extracted point clouds (Limitation #1)
            image_size: resize each view to this size
            view_grid: grid layout for splitting. None = auto-detect.
            augment: whether to apply data augmentation
            aug_config: augmentation configuration dict
            views_subsample: if set, randomly select this many views per item
        """
        self.item_ids = item_ids
        self.labels_df = labels_df
        self.image_dir = image_dir
        self.mesh_features = mesh_features
        self.point_clouds = point_clouds
        self.image_size = image_size
        self.view_grid = view_grid
        self.augment = augment
        self.aug_config = aug_config or {}
        self.views_subsample = views_subsample
        self.max_corrupt_ratio = max_corrupt_ratio
        self.corrupt_count = 0
        self.use_image = bool(use_image)
        self.use_mesh_features = bool(use_mesh_features)
        self.geometry_mean = None if geometry_mean is None else torch.as_tensor(
            geometry_mean, dtype=torch.float32
        ).reshape(5)
        self.geometry_std = None if geometry_std is None else torch.as_tensor(
            geometry_std, dtype=torch.float32
        ).reshape(5)
        if (self.geometry_mean is None) != (self.geometry_std is None):
            raise ValueError("geometry_mean and geometry_std must be supplied together.")
        if self.geometry_std is not None and torch.any(self.geometry_std <= 0):
            raise ValueError("geometry_std must contain only positive values.")
        self.geometry_cache_dir = geometry_cache_dir or os.path.join(
            os.path.dirname(self.image_dir), "geometry_cache_v1"
        )
        self._mesh_hash_cache = {}

        self.defect_cols = [
            "abstract",
            "artifacts",
            "intersection",
            "lowpoly",
            "noisy",
            "open",
            "partial",
            "scale",
            "set",
            "simple",
        ]

        # v2.0.1 FIX: Build O(1) label lookup dict instead of O(N) DataFrame scan.
        # The old code did `labels_df[labels_df['item_id'] == item_id]` per item,
        # which is O(N) per __getitem__ call — ~16M comparisons per epoch.
        self._label_lookup = {}
        self._abstract_indices = []
        self._intersection_indices = []
        self._artifacts_indices = []
        if labels_df is not None:
            if "item_id" in labels_df.columns:
                for _, row in labels_df.iterrows():
                    label_values = row[self.defect_cols].values.astype(np.float32)
                    self._label_lookup[str(row["item_id"])] = label_values
            else:
                # Fallback: labels_df is already indexed by item_id
                for idx_val in labels_df.index:
                    label_values = labels_df.loc[
                        idx_val, self.defect_cols
                    ].values.astype(np.float32)
                    self._label_lookup[str(idx_val)] = label_values

            # defect_cols order is defined as:
            # ["abstract", "artifacts", "intersection", "lowpoly", ...]
            # so intersection is index 2.
            for pos, item_id in enumerate(self.item_ids):
                label_values = self._label_lookup.get(str(item_id))
                if label_values is not None and label_values[0] == 1:
                    self._abstract_indices.append(pos)
                if (
                    label_values is not None
                    and len(label_values) > 2
                    and label_values[2] == 1
                ):
                    self._intersection_indices.append(pos)
                if (
                    label_values is not None
                    and len(label_values) > 1
                    and label_values[1] == 1
                ):
                    self._artifacts_indices.append(pos)

        try:
            import config as cfg

            self.abstract_oversample = (
                augment
                and bool(getattr(cfg, "USE_ABSTRACT_OVERSAMPLING", False))
                and len(self._abstract_indices) > 0
            )
            self.abstract_oversample_prob = float(
                getattr(cfg, "ABSTRACT_OVERSAMPLE_PROB", 0.0)
            )

            # Intersection oversampling (default OFF for production safety)
            self.intersection_oversample = (
                augment
                and bool(getattr(cfg, "INTERSECTION_OVERSAMPLING", False))
                and len(self._intersection_indices) > 0
            )
            self.intersection_oversample_prob = float(
                getattr(cfg, "INTERSECTION_OVERSAMPLE_PROB", 0.0)
            )

            # Artifacts oversampling
            self.artifacts_oversample = (
                augment
                and bool(getattr(cfg, "ARTIFACTS_OVERSAMPLING", False))
                and len(self._artifacts_indices) > 0
            )
            self.artifacts_oversample_prob = float(
                getattr(cfg, "ARTIFACTS_OVERSAMPLE_PROB", 0.0)
            )
        except Exception:
            self.abstract_oversample = False
            self.abstract_oversample_prob = 0.0
            self.intersection_oversample = False
            self.intersection_oversample_prob = 0.0
            self.artifacts_oversample = False
            self.artifacts_oversample_prob = 0.0

        # Base transforms (applied to each individual view)
        import config as cfg
        # Pseudo-normal channels are an input contract, not an augmentation.
        self.use_gradient_normals = bool(getattr(cfg, "USE_GRADIENT_NORMALS", False))
        self.resize_transform = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.norm_3ch = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.norm_6ch = T.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
            std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25],
        )
        self.base_transform = self._apply_view_transforms

        # Random erasing (applied after ToTensor)
        erase_prob = float(self.aug_config.get("random_erasing", 0.0))
        self.random_erase = T.RandomErasing(p=erase_prob) if augment and erase_prob > 0 else None

        # Pre-create ColorJitter transform (avoid re-instantiation per __getitem__)
        cj = self.aug_config.get("color_jitter", 0)
        self.color_jitter = (
            T.ColorJitter(
                brightness=cj,
                contrast=cj,
                saturation=cj * 0.5,
                hue=cj * 0.1,
            )
            if augment and cj > 0
            else None
        )

    def __len__(self):
        return len(self.item_ids)

    def _legacy_getitem(self, idx):
        # Abstract oversampling
        if (
            self.abstract_oversample
            and np.random.random() < self.abstract_oversample_prob
        ):
            idx = int(np.random.choice(self._abstract_indices))

        # Intersection oversampling (kept independent from abstract to avoid coupling)
        if (
            self.intersection_oversample
            and np.random.random() < self.intersection_oversample_prob
        ):
            idx = int(np.random.choice(self._intersection_indices))

        # Artifacts oversampling
        if (
            self.artifacts_oversample
            and np.random.random() < self.artifacts_oversample_prob
        ):
            idx = int(np.random.choice(self._artifacts_indices))

        item_id = self.item_ids[idx]
        safe_item_id = _sanitize_item_id(item_id)

        # ── Load and process image ─────────────────────────────────────────
        loaded = False
        import config as cfg

        preprocess_offline = getattr(cfg, "PREPROCESS_IMAGES_OFFLINE", False)

        if preprocess_offline:
            # Determine directory based on the active image directory
            tensor_dir_name = (
                "train_tensors" if "train" in self.image_dir else "test_tensors"
            )
            parent_dir = os.path.dirname(self.image_dir)
            tensor_dir = os.path.join(parent_dir, tensor_dir_name)
            pt_path = os.path.join(tensor_dir, f"{safe_item_id}_views.pt")
            if os.path.isfile(pt_path):
                try:
                    stacked_tensor = torch.load(
                        pt_path, map_location="cpu", weights_only=False
                    )
                    views = [stacked_tensor[i] for i in range(stacked_tensor.shape[0])]
                    loaded = True
                except Exception:
                    pass

        if not loaded:
            img_path = os.path.join(self.image_dir, f"{safe_item_id}.png")
            if not os.path.isfile(img_path):
                alt_path = os.path.join(self.image_dir, safe_item_id)
                alt_jpg = os.path.join(self.image_dir, f"{safe_item_id}.jpg")
                if os.path.isfile(alt_path):
                    img_path = alt_path
                elif os.path.isfile(alt_jpg):
                    img_path = alt_jpg

            try:
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Image file missing: {img_path}")
                image = Image.open(img_path).convert("RGB")
            except Exception as err:
                self.corrupt_count += 1
                ratio = self.corrupt_count / float(len(self.item_ids))
                print(
                    f"[DATA CORRUPTION WARNING] Item '{item_id}' image load failed ({err}). Corrupt ratio: {ratio:.2%}"
                )
                if ratio > self.max_corrupt_ratio:
                    raise RuntimeError(
                        f"[DATA CORRUPTION FATAL] Corrupt image ratio ({ratio:.2%}) exceeded maximum "
                        f"threshold ({self.max_corrupt_ratio:.2%}). Halting execution to prevent data poisoning."
                    ) from err
                if self.labels_df is not None:
                    raise RuntimeError(f"[DATA CORRUPTION FATAL] Corrupt image '{item_id}' loaded during training: {err}") from err
                # Fallback canvas if image is missing or corrupted within allowable limit
                image = Image.new("RGB", (672, 448), (128, 128, 128))

            # Split into 6 views (auto-detect if view_grid is None)
            views = split_six_views(image, self.view_grid)

        # View subsampling (Limitation #5)
        if self.views_subsample is not None and len(views) > self.views_subsample:
            import hashlib

            seed = int(hashlib.md5(str(item_id).encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(views), self.views_subsample, replace=False)
            indices = sorted(indices)
            views = [views[i] for i in indices]

        # Track indices for geometry rendering alignment
        indices_list = (
            list(indices) if "indices" in locals() else list(range(len(views)))
        )

        # Apply augmentation to each view (works on PIL images and tensors)
        if self.augment:
            views = self._augment_views(views)

        # Transform each view to tensor
        view_tensors = []
        for v in views:
            if isinstance(v, torch.Tensor):
                vt = self._apply_tensor_view_transforms(v)
            else:
                vt = self.base_transform(v)
            if self.random_erase is not None:
                vt = self.random_erase(vt)
            view_tensors.append(vt)

        # Stack: (num_views, 3, H, W)
        views_tensor = torch.stack(view_tensors, dim=0)

        # ── Upgraded true geometry channels concatenation (Phase 2) ───────────
        npz_path = os.path.join(self.image_dir, f"{safe_item_id}.npz")
        if os.path.isfile(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=False)
                v_data = data["vertices"]
                f_data = data["faces"]
                geom_tensor = DirectMeshRasterizer.rasterize_views(
                    v_data, f_data, img_size=self.image_size
                )
                # Slice view indices to match
                geom_tensor = geom_tensor[indices_list]
            except Exception:
                geom_tensor = torch.zeros(
                    (len(views), 5, self.image_size, self.image_size),
                    dtype=torch.float32,
                )
        else:
            geom_tensor = torch.zeros(
                (len(views), 5, self.image_size, self.image_size), dtype=torch.float32
            )

        # Stack 3ch RGB + 5ch Geometry = 8-channel visual representation
        # Only concatenate if IMAGE_IN_CHANNELS specifies 8 or 11 channels (meaning geometry channels are enabled)
        import config as cfg
        in_channels = getattr(cfg, "IMAGE_IN_CHANNELS", 8)
        if in_channels in [8, 11]:
            views_tensor = torch.cat([views_tensor, geom_tensor], dim=1)

        # ── Mesh features ──────────────────────────────────────────────────
        mesh_feat = None
        if self.mesh_features is not None:
            mesh_feat = torch.tensor(self.mesh_features[idx], dtype=torch.float32)

        # ── Point clouds (Limitation #1) ───────────────────────────────────
        pc = None
        if self.point_clouds is not None:
            pc = torch.tensor(self.point_clouds[idx], dtype=torch.float32)

        # ── Labels (v2.0.1: O(1) dict lookup instead of O(N) DataFrame scan) ──
        labels = None
        if self.labels_df is not None:
            lbl = self._label_lookup.get(str(item_id))
            if lbl is not None:
                labels = torch.tensor(lbl, dtype=torch.float32)
            else:
                # Item not in this fold's labels — return zeros
                labels = torch.zeros(len(self.defect_cols), dtype=torch.float32)

        return {
            "item_id": item_id,
            "views": views_tensor,  # (V, 3, H, W)
            "mesh_features": mesh_feat,  # (D,) or None
            "point_cloud": pc,  # (P, 3) or None
            "labels": labels,  # (10,) or None
        }

    def _source_mesh_hash(self, npz_path: str) -> str:
        """Return a stable content hash for cache invalidation within this worker."""
        cached = self._mesh_hash_cache.get(npz_path)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with open(npz_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        result = digest.hexdigest()
        self._mesh_hash_cache[npz_path] = result
        return result

    def _geometry_cache_path(self, item_id: str, npz_path: str) -> str:
        source_hash = self._source_mesh_hash(npz_path)
        safe_id = _sanitize_item_id(item_id)
        filename = f"{safe_id}_{self.image_size}_v1_{source_hash[:16]}.pt"
        return os.path.join(self.geometry_cache_dir, filename)

    @staticmethod
    def _validate_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> tuple:
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices contain NaN or infinity")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")
        if not np.isfinite(faces).all():
            raise ValueError("faces contain NaN or infinity")
        if not np.equal(faces, np.floor(faces)).all():
            raise ValueError("faces must contain integer indices")
        faces = faces.astype(np.int64, copy=False)
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("faces contain indices outside [0, len(vertices))")
        return vertices.astype(np.float32, copy=False), faces

    def _load_raw_geometry_raster(self, item_id: str) -> torch.Tensor:
        """Load/cache all six raw raster views as a (6, 5, H, W) tensor."""
        safe_id = _sanitize_item_id(item_id)
        npz_path = os.path.join(self.image_dir, f"{safe_id}.npz")
        if not os.path.isfile(npz_path):
            if self.labels_df is not None:
                raise FileNotFoundError(f"Missing mesh file for training/validation item: {npz_path}")
            return torch.zeros((6, 5, self.image_size, self.image_size), dtype=torch.float32)

        cache_path = self._geometry_cache_path(item_id, npz_path)
        if os.path.isfile(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                if isinstance(cached, torch.Tensor) and tuple(cached.shape) == (6, 5, self.image_size, self.image_size):
                    return cached.to(dtype=torch.float32)
            except Exception:
                # Treat an interrupted or stale cache entry as a cache miss.
                pass

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                vertices, faces = self._validate_mesh_arrays(data["vertices"], data["faces"])
            raster = DirectMeshRasterizer.rasterize_views(
                vertices, faces, img_size=self.image_size
            ).to(dtype=torch.float32)
            if tuple(raster.shape) != (6, 5, self.image_size, self.image_size):
                raise RuntimeError(f"Unexpected raster shape for {item_id}: {tuple(raster.shape)}")
        except Exception:
            if self.labels_df is not None:
                raise
            return torch.zeros((6, 5, self.image_size, self.image_size), dtype=torch.float32)

        try:
            os.makedirs(self.geometry_cache_dir, exist_ok=True)
            temporary_path = f"{cache_path}.{os.getpid()}.tmp"
            torch.save(raster, temporary_path)
            os.replace(temporary_path, cache_path)
        except Exception:
            # Cache failure must not change the tensor returned to the model.
            try:
                if 'temporary_path' in locals() and os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass
        return raster

    def _load_geometry_raster(self, item_id: str, view_indices: list) -> torch.Tensor:
        raster = self._load_raw_geometry_raster(item_id)[view_indices]
        if self.geometry_mean is not None:
            mean = self.geometry_mean.view(1, 5, 1, 1)
            std = self.geometry_std.view(1, 5, 1, 1).clamp_min(1e-6)
            raster = (raster - mean) / std
        return raster

    def compute_geometry_stats(self, item_ids: list, eps: float = 1e-6):
        """Compute fold-local per-channel statistics from raw, cached rasters."""
        import config as cfg
        if not getattr(cfg, "USE_GEOMETRY_RASTER", False):
            return None, None
        channel_sum = torch.zeros(5, dtype=torch.float64)
        channel_sq_sum = torch.zeros(5, dtype=torch.float64)
        pixel_count = 0
        for item_id in item_ids:
            raster = self._load_raw_geometry_raster(item_id).to(dtype=torch.float64)
            channel_sum += raster.sum(dim=(0, 2, 3))
            channel_sq_sum += raster.square().sum(dim=(0, 2, 3))
            pixel_count += raster.shape[0] * raster.shape[2] * raster.shape[3]
        if pixel_count == 0:
            raise ValueError("Cannot compute geometry statistics from an empty training fold.")
        mean = channel_sum / pixel_count
        variance = (channel_sq_sum / pixel_count - mean.square()).clamp_min(0.0)
        std = variance.sqrt().clamp_min(eps)
        return mean.to(dtype=torch.float32).tolist(), std.to(dtype=torch.float32).tolist()

    def _load_rgb_views(self, item_id: str, safe_item_id: str) -> list:
        """Load unnormalised RGB views; training/validation never silently falls back."""
        import config as cfg
        if getattr(cfg, "PREPROCESS_IMAGES_OFFLINE", False):
            tensor_dir_name = "train_tensors" if "train" in self.image_dir else "test_tensors"
            tensor_path = os.path.join(
                os.path.dirname(self.image_dir), tensor_dir_name, f"{safe_item_id}_views.pt"
            )
            if os.path.isfile(tensor_path):
                tensor = torch.load(tensor_path, map_location="cpu", weights_only=False)
                if tensor.ndim == 4 and tensor.shape[1] == 3:
                    return [tensor[i].to(dtype=torch.float32) for i in range(tensor.shape[0])]

        image_path = os.path.join(self.image_dir, f"{safe_item_id}.png")
        if not os.path.isfile(image_path):
            alternate = os.path.join(self.image_dir, safe_item_id)
            jpg = os.path.join(self.image_dir, f"{safe_item_id}.jpg")
            image_path = alternate if os.path.isfile(alternate) else jpg if os.path.isfile(jpg) else image_path
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file missing: {image_path}")
            image = Image.open(image_path).convert("RGB")
        except Exception:
            if self.labels_df is not None:
                raise
            image = Image.new("RGB", (672, 448), (128, 128, 128))
        return split_six_views(image, self.view_grid)

    def _spatial_augmentation_params(self) -> tuple:
        if not self.augment:
            return False, 0.0
        flip = bool(self.aug_config.get("horizontal_flip", False) and np.random.random() < 0.5)
        max_rotation = float(self.aug_config.get("rotation", 0.0))
        angle = float(np.random.uniform(-max_rotation, max_rotation)) if max_rotation > 0 and np.random.random() < 0.3 else 0.0
        return flip, angle

    @staticmethod
    def _apply_spatial_transform(tensor: torch.Tensor, flip: bool, angle: float) -> torch.Tensor:
        if flip:
            tensor = torch.flip(tensor, dims=[-1])
        if angle != 0.0:
            tensor = TF.rotate(
                tensor, angle, interpolation=T.InterpolationMode.BILINEAR
            )
        return tensor

    def __getitem__(self, idx):
        # Keep the legacy implementation above only as a reference while the
        # production path below enforces the explicit modality contract.
        if self.abstract_oversample and np.random.random() < self.abstract_oversample_prob:
            idx = int(np.random.choice(self._abstract_indices))
        if self.intersection_oversample and np.random.random() < self.intersection_oversample_prob:
            idx = int(np.random.choice(self._intersection_indices))
        if self.artifacts_oversample and np.random.random() < self.artifacts_oversample_prob:
            idx = int(np.random.choice(self._artifacts_indices))

        item_id = self.item_ids[idx]
        safe_item_id = _sanitize_item_id(item_id)
        views_tensor = None

        if self.use_image:
            views = self._load_rgb_views(item_id, safe_item_id)
            view_indices = list(range(len(views)))
            if self.views_subsample is not None and len(views) > self.views_subsample:
                seed = int(hashlib.md5(str(item_id).encode()).hexdigest(), 16) % (2**32)
                selected = np.random.RandomState(seed).choice(
                    len(views), self.views_subsample, replace=False
                )
                view_indices = sorted(selected.tolist())
                views = [views[i] for i in view_indices]

            raw_rgb = []
            for view in views:
                if isinstance(view, torch.Tensor):
                    tensor = view.to(dtype=torch.float32)
                    if tensor.ndim != 3 or tensor.shape[0] != 3:
                        raise ValueError(f"Offline RGB tensor for {item_id} must have shape (3,H,W).")
                    if tuple(tensor.shape[-2:]) != (self.image_size, self.image_size):
                        tensor = TF.resize(tensor, [self.image_size, self.image_size])
                else:
                    tensor = self.to_tensor(self.resize_transform(view))
                raw_rgb.append(tensor)
            raw_rgb = torch.stack(raw_rgb, dim=0)

            geometry = None
            import config as cfg
            if getattr(cfg, "USE_GEOMETRY_RASTER", False):
                geometry = self._load_geometry_raster(item_id, view_indices)

            flip, angle = self._spatial_augmentation_params()
            raw_rgb = self._apply_spatial_transform(raw_rgb, flip, angle)
            if geometry is not None:
                geometry = self._apply_spatial_transform(geometry, flip, angle)
                # The mask remains a binary channel after interpolation.
                geometry[:, 4:5] = (geometry[:, 4:5] >= 0.5).to(geometry.dtype)

            pseudo_normals = None
            if self.use_gradient_normals:
                pseudo_normals = torch.stack(
                    [compute_sobel_pseudo_normals(view) for view in raw_rgb], dim=0
                )

            if self.color_jitter is not None:
                raw_rgb = torch.stack([self.color_jitter(view) for view in raw_rgb], dim=0)
            if self.random_erase is not None:
                raw_rgb = torch.stack([self.random_erase(view) for view in raw_rgb], dim=0)

            if pseudo_normals is not None:
                image_tensor = torch.cat([raw_rgb, pseudo_normals], dim=1)
                image_tensor = torch.stack([self.norm_6ch(view) for view in image_tensor], dim=0)
            else:
                image_tensor = torch.stack([self.norm_3ch(view) for view in raw_rgb], dim=0)
            views_tensor = torch.cat([image_tensor, geometry], dim=1) if geometry is not None else image_tensor

        mesh_feat = None
        if self.use_mesh_features and self.mesh_features is not None:
            mesh_feat = torch.tensor(self.mesh_features[idx], dtype=torch.float32)
        pc = torch.tensor(self.point_clouds[idx], dtype=torch.float32) if self.point_clouds is not None else None

        labels = None
        if self.labels_df is not None:
            label_values = self._label_lookup.get(str(item_id))
            if label_values is None:
                raise KeyError(f"No labels found for item_id={item_id}")
            labels = torch.tensor(label_values, dtype=torch.float32)

        return {
            "item_id": item_id,
            "views": views_tensor,
            "mesh_features": mesh_feat,
            "point_cloud": pc,
            "labels": labels,
        }

    def _augment_views(self, views: list) -> list:
        """
        Apply consistent augmentation across all views.
        The same random parameters are used for all views to maintain
        spatial consistency.
        """
        # Color jitter (same for all views, pre-created in __init__)
        if self.color_jitter is not None:
            views = [self.color_jitter(v) for v in views]

        # Random horizontal flip (same for all views)
        if self.aug_config.get("horizontal_flip", False) and np.random.random() < 0.5:
            views = [TF.hflip(v) for v in views]

        # Random rotation (same for all views, small angle)
        if self.aug_config.get("rotation", 0) > 0 and np.random.random() < 0.3:
            angle = np.random.uniform(
                -self.aug_config["rotation"], self.aug_config["rotation"]
            )
            views = [TF.rotate(v, angle) for v in views]

        return views

    def _apply_view_transforms(self, view_img):
        """
        Applies pipeline:
        1. Resize
        2. ToTensor [0, 1]
        3. (Optional) Sobel Pseudo-Normals -> Concat (6, H, W)
        4. Normalize
        """
        v = self.resize_transform(view_img)
        t = self.to_tensor(v)
        if self.use_gradient_normals:
            pn = compute_sobel_pseudo_normals(t)
            concat = torch.cat([t, pn], dim=0)
            return self.norm_6ch(concat)
        return self.norm_3ch(t)

    def _apply_tensor_view_transforms(self, view_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies normalisation and pseudo-normal computation directly on a pre-resized tensor.
        """
        if self.use_gradient_normals:
            pn = compute_sobel_pseudo_normals(view_tensor)
            concat = torch.cat([view_tensor, pn], dim=0)
            return self.norm_6ch(concat)
        return self.norm_3ch(view_tensor)


def preprocess_images_offline(
    image_dir: str,
    output_tensor_dir: str,
    item_ids: list,
    view_grid=(3, 2),
    image_size: int = 224,
) -> None:
    """
    Pre-crops and pre-resizes views for each item, saving them as a
    torch tensor file of shape (6, 3, 224, 224).
    """
    from PIL import Image
    from tqdm import tqdm

    os.makedirs(output_tensor_dir, exist_ok=True)

    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )

    for item_id in tqdm(
        item_ids, desc=f"Preprocessing views to {os.path.basename(output_tensor_dir)}"
    ):
        safe_id = _sanitize_item_id(item_id)
        pt_path = os.path.join(output_tensor_dir, f"{safe_id}_views.pt")
        if os.path.exists(pt_path):
            continue

        img_path = os.path.join(image_dir, f"{safe_id}.png")
        if not os.path.isfile(img_path):
            alt_path = os.path.join(image_dir, safe_id)
            alt_jpg = os.path.join(image_dir, f"{safe_id}.jpg")
            if os.path.isfile(alt_path):
                img_path = alt_path
            elif os.path.isfile(alt_jpg):
                img_path = alt_jpg

        try:
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file missing: {img_path}")
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (672, 448), (128, 128, 128))

        views = split_six_views(image, view_grid)
        view_tensors = [transform(v) for v in views]
        stacked = torch.stack(view_tensors, dim=0)  # (6, 3, 224, 224)
        torch.save(stacked, pt_path)


_SOBEL_KERNELS_CACHE = {}


def compute_sobel_pseudo_normals(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes 3-channel pseudo-normal map from a (3, H, W) RGB tensor in [0, 1].
    Uses Sobel gradient operators to derive surface orientation vectors (Nx, Ny, Nz).
    """
    gray = (
        0.2989 * rgb_tensor[0:1] + 0.5870 * rgb_tensor[1:2] + 0.1140 * rgb_tensor[2:3]
    ).unsqueeze(0)

    key = (rgb_tensor.device, rgb_tensor.dtype)
    if key not in _SOBEL_KERNELS_CACHE:
        sx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=rgb_tensor.dtype,
            device=rgb_tensor.device,
        ).view(1, 1, 3, 3)
        sy = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=rgb_tensor.dtype,
            device=rgb_tensor.device,
        ).view(1, 1, 3, 3)
        _SOBEL_KERNELS_CACHE[key] = (sx, sy)

    sobel_x, sobel_y = _SOBEL_KERNELS_CACHE[key]
    dx = F.conv2d(gray, sobel_x, padding=1).squeeze(0)
    dy = F.conv2d(gray, sobel_y, padding=1).squeeze(0)

    nx = -dx
    ny = -dy
    nz = torch.ones_like(dx)

    norm = torch.sqrt(nx**2 + ny**2 + nz**2 + 1e-8)
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    pseudo_normal = torch.cat(
        [(nx + 1.0) / 2.0, (ny + 1.0) / 2.0, (nz + 1.0) / 2.0], dim=0
    )
    return pseudo_normal


class TTATransform:
    """
    Test-Time Augmentation transforms for inference.
    Generates multiple augmented versions of each input.
    """

    def __init__(self, flips: list = None, rotations: list = None):
        self.transforms = []

        for flip in flips or [False]:
            for rot in rotations or [0]:
                self.transforms.append({"flip": flip, "rotation": rot})

    def __len__(self):
        return len(self.transforms)

    def apply(self, views_tensor: torch.Tensor) -> list:
        """
        Apply all TTA variants to a views tensor.
        Returns list of views tensors.
        """
        results = []
        for t in self.transforms:
            v = views_tensor.clone()
            if t["flip"]:
                # Input is (B, V, C, H, W): horizontal flip is W, not H.
                v = torch.flip(v, dims=[4])
            if t["rotation"] != 0:
                if t["rotation"] % 90 != 0:
                    raise ValueError(
                        "TTATransform rotations must be multiples of 90 degrees; "
                        f"got {t['rotation']}."
                    )
                # Rotate spatial dimensions only; rotating C/H corrupts channels.
                v = torch.rot90(v, k=t["rotation"] // 90, dims=[3, 4])
            results.append(v)
        return results


class KorniaGPUAugmentation(torch.nn.Module):
    def __init__(self, aug_config: dict):
        super().__init__()
        import kornia.augmentation as K

        self.augs = torch.nn.Sequential()
        if aug_config.get("color_jitter", 0) > 0:
            cj = aug_config["color_jitter"]
            self.augs.add_module(
                "jitter",
                K.ColorJitter(
                    brightness=cj, contrast=cj, saturation=cj * 0.5, hue=cj * 0.1, p=0.5
                ),
            )
        if aug_config.get("horizontal_flip", False):
            self.augs.add_module("flip", K.RandomHorizontalFlip(p=0.5))
        if aug_config.get("rotation", 0) > 0:
            rot = float(aug_config["rotation"])
            self.augs.add_module("rotate", K.RandomRotation(degrees=(-rot, rot), p=0.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, V, C, H, W)
        B, V, C, H, W = x.shape
        flat_x = x.view(B * V, C, H, W)
        flat_x = self.augs(flat_x)
        return flat_x.view(B, V, C, H, W)
