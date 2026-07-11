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
    def rasterize_views(vertices: np.ndarray, faces: np.ndarray = None, img_size: int = 224) -> torch.Tensor:
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
                face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10)
                
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
                
                center_n = center_n / (np.linalg.norm(center_n, axis=1, keepdims=True) + 1e-10)
                m01_n = m01_n / (np.linalg.norm(m01_n, axis=1, keepdims=True) + 1e-10)
                m12_n = m12_n / (np.linalg.norm(m12_n, axis=1, keepdims=True) + 1e-10)
                m20_n = m20_n / (np.linalg.norm(m20_n, axis=1, keepdims=True) + 1e-10)
                
                dense_verts = np.concatenate([norm_verts, (v0 + v1 + v2) / 3.0, (v0 + v1) / 2.0, (v1 + v2) / 2.0, (v2 + v0) / 2.0], axis=0)
                dense_normals = np.concatenate([vert_normals, center_n, m01_n, m12_n, m20_n], axis=0)
            else:
                dense_verts = norm_verts
                dense_normals = np.zeros_like(norm_verts)
        else:
            dense_verts = norm_verts
            dense_normals = np.zeros_like(norm_verts)

        directions = [
            (0, 1, 2),   # +Z (Front): u=X, v=Y, depth=+Z
            (0, 1, 2),   # -Z (Back): u=-X, v=Y, depth=-Z
            (2, 1, 0),   # +X (Right): u=-Z, v=Y, depth=+X
            (2, 1, 0),   # -X (Left): u=+Z, v=Y, depth=-X
            (0, 2, 1),   # +Y (Top): u=X, v=+Z, depth=+Y
            (0, 2, 1),   # -Y (Bottom): u=X, v=-Z, depth=-Y
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
        padded = np.pad(arr, pad, mode='reflect')
        return np.convolve(padded, kernel, mode='valid')[:len(arr)]


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
        row_means = gray.mean(axis=1)   # (H,)
        col_means = gray.mean(axis=0)   # (W,)

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
        view_grid: tuple = None,     # None = auto-detect per image (Limitation #3)
        augment: bool = False,
        aug_config: dict = None,
        views_subsample: int = None,  # Limitation #5: e.g., 4 means random 4 of 6
        max_corrupt_ratio: float = 0.01,
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

        self.defect_cols = [
            "abstract", "artifacts", "intersection", "lowpoly",
            "noisy", "open", "partial", "scale", "set", "simple"
        ]

        # v2.0.1 FIX: Build O(1) label lookup dict instead of O(N) DataFrame scan.
        # The old code did `labels_df[labels_df['item_id'] == item_id]` per item,
        # which is O(N) per __getitem__ call — ~16M comparisons per epoch.
        self._label_lookup = {}
        if labels_df is not None:
            if 'item_id' in labels_df.columns:
                for _, row in labels_df.iterrows():
                    self._label_lookup[str(row['item_id'])] = row[self.defect_cols].values.astype(np.float32)
            else:
                # Fallback: labels_df is already indexed by item_id
                for idx_val in labels_df.index:
                    self._label_lookup[str(idx_val)] = labels_df.loc[idx_val, self.defect_cols].values.astype(np.float32)

        # Base transforms (applied to each individual view)
        self.use_gradient_normals = aug_config.get("use_gradient_normals", False) if aug_config else False
        self.resize_transform = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.norm_3ch = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm_6ch = T.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
            std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25]
        )
        self.base_transform = self._apply_view_transforms

        # Random erasing (applied after ToTensor)
        self.random_erase = T.RandomErasing(p=0.1) if augment else None

        # Pre-create ColorJitter transform (avoid re-instantiation per __getitem__)
        cj = self.aug_config.get("color_jitter", 0)
        self.color_jitter = T.ColorJitter(
            brightness=cj, contrast=cj, saturation=cj * 0.5, hue=cj * 0.1,
        ) if augment and cj > 0 else None

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        safe_item_id = _sanitize_item_id(item_id)

        # ── Load and process image ─────────────────────────────────────────
        loaded = False
        import config as cfg
        preprocess_offline = getattr(cfg, "PREPROCESS_IMAGES_OFFLINE", False)

        if preprocess_offline:
            # Determine directory based on the active image directory
            tensor_dir_name = "train_tensors" if "train" in self.image_dir else "test_tensors"
            parent_dir = os.path.dirname(self.image_dir)
            tensor_dir = os.path.join(parent_dir, tensor_dir_name)
            pt_path = os.path.join(tensor_dir, f"{safe_item_id}_views.pt")
            if os.path.isfile(pt_path):
                try:
                    stacked_tensor = torch.load(pt_path, map_location="cpu", weights_only=False)
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
                print(f"[DATA CORRUPTION WARNING] Item '{item_id}' image load failed ({err}). Corrupt ratio: {ratio:.2%}")
                if ratio > self.max_corrupt_ratio:
                    raise RuntimeError(
                        f"[DATA CORRUPTION FATAL] Corrupt image ratio ({ratio:.2%}) exceeded maximum "
                        f"threshold ({self.max_corrupt_ratio:.2%}). Halting execution to prevent data poisoning."
                    ) from err
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
        indices_list = list(indices) if 'indices' in locals() else list(range(len(views)))

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
                geom_tensor = DirectMeshRasterizer.rasterize_views(v_data, f_data, img_size=self.image_size)
                # Slice view indices to match
                geom_tensor = geom_tensor[indices_list]
            except Exception:
                geom_tensor = torch.zeros((len(views), 5, self.image_size, self.image_size), dtype=torch.float32)
        else:
            geom_tensor = torch.zeros((len(views), 5, self.image_size, self.image_size), dtype=torch.float32)

        # Stack 3ch RGB + 5ch Geometry = 8-channel visual representation
        views_tensor = torch.cat([views_tensor, geom_tensor], dim=1)

        # ── Mesh features ──────────────────────────────────────────────────
        mesh_feat = None
        if self.mesh_features is not None:
            mesh_feat = torch.tensor(
                self.mesh_features[idx], dtype=torch.float32
            )

        # ── Point clouds (Limitation #1) ───────────────────────────────────
        pc = None
        if self.point_clouds is not None:
            pc = torch.tensor(
                self.point_clouds[idx], dtype=torch.float32
            )

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
            "views": views_tensor,        # (V, 3, H, W)
            "mesh_features": mesh_feat,   # (D,) or None
            "point_cloud": pc,            # (P, 3) or None
            "labels": labels,             # (10,) or None
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
    view_grid = (3, 2),
    image_size: int = 224
) -> None:
    """
    Pre-crops and pre-resizes views for each item, saving them as a 
    torch tensor file of shape (6, 3, 224, 224).
    """
    from PIL import Image
    from tqdm import tqdm
    
    os.makedirs(output_tensor_dir, exist_ok=True)
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    for item_id in tqdm(item_ids, desc=f"Preprocessing views to {os.path.basename(output_tensor_dir)}"):
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
        stacked = torch.stack(view_tensors, dim=0) # (6, 3, 224, 224)
        torch.save(stacked, pt_path)


_SOBEL_KERNELS_CACHE = {}


def compute_sobel_pseudo_normals(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes 3-channel pseudo-normal map from a (3, H, W) RGB tensor in [0, 1].
    Uses Sobel gradient operators to derive surface orientation vectors (Nx, Ny, Nz).
    """
    gray = (0.2989 * rgb_tensor[0:1] + 0.5870 * rgb_tensor[1:2] + 0.1140 * rgb_tensor[2:3]).unsqueeze(0)
    
    key = (rgb_tensor.device, rgb_tensor.dtype)
    if key not in _SOBEL_KERNELS_CACHE:
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=rgb_tensor.dtype, device=rgb_tensor.device).view(1, 1, 3, 3)
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=rgb_tensor.dtype, device=rgb_tensor.device).view(1, 1, 3, 3)
        _SOBEL_KERNELS_CACHE[key] = (sx, sy)
    
    sobel_x, sobel_y = _SOBEL_KERNELS_CACHE[key]
    dx = F.conv2d(gray, sobel_x, padding=1).squeeze(0)
    dy = F.conv2d(gray, sobel_y, padding=1).squeeze(0)
    
    nx = -dx
    ny = -dy
    nz = torch.ones_like(dx)
    
    norm = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2 + 1e-8)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    
    pseudo_normal = torch.cat([(nx + 1.0) / 2.0, (ny + 1.0) / 2.0, (nz + 1.0) / 2.0], dim=0)
    return pseudo_normal


class TTATransform:
    """
    Test-Time Augmentation transforms for inference.
    Generates multiple augmented versions of each input.
    """

    def __init__(self, flips: list = None, rotations: list = None):
        self.transforms = []

        for flip in (flips or [False]):
            for rot in (rotations or [0]):
                self.transforms.append({"flip": flip, "rotation": rot})

    def __len__(self):
        return len(self.transforms) + 2

    def apply(self, views_tensor: torch.Tensor) -> list:
        """
        Apply all TTA variants to a views tensor.
        Returns list of views tensors.
        """
        results = []
        for t in self.transforms:
            v = views_tensor.clone()
            if t["flip"]:
                v = torch.flip(v, dims=[3])  # Flip height/width
            if t["rotation"] != 0:
                v = torch.rot90(v, k=t["rotation"] // 90, dims=[2, 3])
            results.append(v)
            
        # Photometric TTA: Test rendering robustness (Step 5)
        results.append((views_tensor * 1.05).clamp(0, 1))         # Brightness +
        results.append(((views_tensor - 0.5) * 1.1) + 0.5).clamp(0, 1) # Contrast +

        return results


class KorniaGPUAugmentation(torch.nn.Module):
    def __init__(self, aug_config: dict):
        super().__init__()
        import kornia.augmentation as K
        self.augs = torch.nn.Sequential()
        if aug_config.get("color_jitter", 0) > 0:
            cj = aug_config["color_jitter"]
            self.augs.add_module("jitter", K.ColorJitter(brightness=cj, contrast=cj, saturation=cj*0.5, hue=cj*0.1, p=0.5))
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