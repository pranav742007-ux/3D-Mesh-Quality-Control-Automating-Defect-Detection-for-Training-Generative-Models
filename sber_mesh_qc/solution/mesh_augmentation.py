"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Offline Augmentation  [v7.2]
===============================================================================
Generates augmented copies of 3D meshes (noise, holes, scale) and pre-extracts
their features to expand training data without on-the-fly execution overhead.
===============================================================================
"""

import os
import sys
import numpy as np
from typing import Tuple

sol_dir = os.path.dirname(os.path.abspath(__file__))
if sol_dir not in sys.path:
    sys.path.insert(0, sol_dir)


def augment_mesh_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    noise_std: float = 0.01,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    hole_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies geometric augmentations:
    - Random scale change
    - Vertex jittering (additive Gaussian noise)
    - Face dropout (creates boundary edge loops/holes)
    """
    rng = np.random.RandomState(seed)
    
    # 1. Scale
    scale = rng.uniform(scale_range[0], scale_range[1])
    aug_vertices = vertices * scale
    
    # 2. Jitter
    if noise_std > 0 and len(aug_vertices) > 0:
        noise = rng.normal(0, noise_std, size=aug_vertices.shape)
        aug_vertices = aug_vertices + noise
        
    # 3. Holes
    aug_faces = faces.copy()
    if hole_ratio > 0 and len(faces) > 0:
        n_drop = int(len(faces) * hole_ratio)
        if n_drop > 0:
            keep_indices = rng.choice(len(faces), size=len(faces) - n_drop, replace=False)
            aug_faces = faces[keep_indices]
            
            # Remap vertices to exclude unreferenced ones and prevent feature corruption (CR1)
            unique_v, remapped = np.unique(aug_faces, return_inverse=True)
            aug_vertices = aug_vertices[unique_v]
            aug_faces = remapped.reshape(-1, 3)
            
    return aug_vertices, aug_faces


def _offline_aug_worker(task):
    src, dst, idx, n_std, h_ratio = task
    try:
        data = np.load(src, allow_pickle=False)
        v = data["vertices"]
        f = data["faces"]
        av, af = augment_mesh_geometry(v, f, noise_std=n_std, hole_ratio=h_ratio, seed=42 + idx)
        np.savez(dst, vertices=av, faces=af)
        return True
    except Exception as e:
        print(f"Failed to augment {src} -> {dst}: {e}")
        return False


def generate_offline_augmentations(
    train_ids: list,
    train_dir: str,
    output_aug_dir: str,
    num_augmentations: int = 3,
    noise_std: float = 0.01,
    hole_ratio: float = 0.05
) -> None:
    """
    Iterates over train_ids, generates augmented copies, and saves them
    to output_aug_dir.
    """
    os.makedirs(output_aug_dir, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    import math
    
    # Pre-check: only process if files don't exist yet
    tasks = []
    for item_id in train_ids:
        safe_id = os.path.basename(str(item_id))
        src_path = os.path.join(train_dir, f"{safe_id}.npz")
        if not os.path.isfile(src_path):
            continue
        for i in range(num_augmentations):
            dst_path = os.path.join(output_aug_dir, f"{safe_id}_aug_{i}.npz")
            if not os.path.isfile(dst_path):
                tasks.append((src_path, dst_path, len(tasks), noise_std, hole_ratio))
                
    if not tasks:
        print("[Offline Augment] All augmented meshes already exist.")
        return

    print(f"[Offline Augment] Generating {len(tasks)} augmented meshes in parallel...")
    num_workers = min(os.cpu_count() or 4, 16)
            
    chunk_size = max(1, math.ceil(len(tasks) / (num_workers * 4)))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_offline_aug_worker, tasks, chunksize=chunk_size))
        
    print(f"[Offline Augment] Successfully generated augmented meshes in '{output_aug_dir}'.")
