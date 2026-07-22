import os
import io
import sys
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure solution is in import path
sys.path.insert(0, os.path.abspath("solution"))
sys.path.insert(0, os.path.abspath("."))

from solution.mesh_features import (
    compute_mesh_features,
    compute_spherical_harmonics_descriptors,
    compute_topological_invariants,
    compute_qem_decimation_stability,
    compute_physics_stability_metric,
    FEATURE_ORDER,
    MESH_FEATURE_DIM_EXTENDED
)

def extract_features_from_bytes(npz_bytes, extended=True):
    """Computes geometric features directly from raw zip file bytes in memory."""
    try:
        data = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
        vertices = np.nan_to_num(data["vertices"], nan=0.0, posinf=1e6, neginf=-1e6)
        faces = np.nan_to_num(data["faces"], nan=0, posinf=0, neginf=0).astype(int)

        # Degenerate mesh safety check
        if len(vertices) < 4 or len(faces) < 1:
            feat_dim = MESH_FEATURE_DIM_EXTENDED if extended else 58
            return np.full(feat_dim, -5.0, dtype=np.float32)

        features = compute_mesh_features(vertices, faces)

        if extended:
            base_vector = np.array([features.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32)
            shtd_vector = compute_spherical_harmonics_descriptors(vertices)
            topo_vector = compute_topological_invariants(vertices, faces)
            qem_score = np.array([compute_qem_decimation_stability(vertices, faces)], dtype=np.float32)
            phys_dict = compute_physics_stability_metric(vertices, faces)
            phys_vector = np.array([phys_dict["com_height"], phys_dict["support_radius"], phys_dict["tipping_angle_deg"]], dtype=np.float32)
            return np.concatenate([base_vector, shtd_vector, topo_vector, qem_score, phys_vector], axis=0)
        else:
            order = FEATURE_ORDER[:58]
            return np.array([features.get(k, 0.0) for k in order], dtype=np.float32)
    except Exception as e:
        feat_dim = MESH_FEATURE_DIM_EXTENDED if extended else 58
        return np.zeros(feat_dim, dtype=np.float32)

def process_zip_chunk(zip_path, item_ids):
    """Processes a chunk of item_ids from a specific zip file in a single process."""
    features = []
    # Open zip file within the worker process to avoid shared file pointer issues
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Create mapping of base filename to actual archive path for fast lookup
        namelist = zf.namelist()
        base_to_full = {}
        for name in namelist:
            if name.endswith(".npz"):
                base_to_full[os.path.basename(name)] = name

        for item_id in item_ids:
            safe_name = os.path.basename(str(item_id))
            npz_name = f"{safe_name}.npz"
            if npz_name in base_to_full:
                full_name = base_to_full[npz_name]
                try:
                    npz_bytes = zf.read(full_name)
                    feat = extract_features_from_bytes(npz_bytes, extended=True)
                    features.append(feat)
                except Exception:
                    features.append(np.zeros(MESH_FEATURE_DIM_EXTENDED, dtype=np.float32))
            else:
                features.append(np.zeros(MESH_FEATURE_DIM_EXTENDED, dtype=np.float32))
    return features

def extract_all(zip_path, csv_path, out_npz_path):
    """Main extraction coordinator using ProcessPoolExecutor for parallel streaming."""
    print(f"Extracting features from {zip_path} using {csv_path} metadata...")
    df = pd.read_csv(csv_path)
    item_ids = df["item_id"].tolist()

    num_workers = min(os.cpu_count() or 4, 8)
    chunk_size = 15
    chunks = [item_ids[i:i + chunk_size] for i in range(0, len(item_ids), chunk_size)]

    all_features = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_zip_chunk, zip_path, chunk): idx for idx, chunk in enumerate(chunks)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Feature Extraction"):
            idx = futures[future]
            try:
                all_features[idx] = future.result()
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")
                # Fallback to zero features for this chunk
                all_features[idx] = [np.zeros(MESH_FEATURE_DIM_EXTENDED, dtype=np.float32)] * len(chunks[idx])

    # Flatten and verify
    flattened_features = []
    for chunk in all_features:
        flattened_features.extend(chunk)

    features_arr = np.array(flattened_features, dtype=np.float32)
    np.savez_compressed(out_npz_path, features=features_arr)
    print(f"[SUCCESS] Saved features array of shape {features_arr.shape} to {out_npz_path}\n")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # Extract test (Train already successfully completed and saved)
    extract_all("data/test.zip", "data/test.csv", "data/mesh_features_test_extended.npz")
