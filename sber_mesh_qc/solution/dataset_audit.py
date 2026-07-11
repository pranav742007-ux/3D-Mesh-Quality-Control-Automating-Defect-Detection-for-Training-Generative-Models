"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Dataset Audit & Integrity [v7.3]
===============================================================================
Phase 0 Dataset Audit and Integrity Scan. Computes duplicate rates, label
distributions, complexity statistics, and scans for file corruptions,
NaNs, inf values, and topological degeneracies.
===============================================================================
"""
import os
import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from mesh_validity import MeshValidityAnalyzer

def compute_mesh_checksum(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Compute deterministic checksum of mesh geometry to identify duplicates."""
    # Round vertices to avoid floating-point representation differences
    v_rounded = np.round(vertices, 6)
    # Sort vertices and faces canonically to ensure duplicate matching regardless of order
    v_flat = v_rounded[np.lexsort(v_rounded.T.copy())].tobytes()
    f_sorted = np.sort(faces, axis=1)
    f_flat = f_sorted[np.lexsort(f_sorted.T.copy())].tobytes()
    
    hasher = hashlib.md5()
    hasher.update(v_flat)
    hasher.update(f_flat)
    return hasher.hexdigest()

def compute_image_checksum(img_path: str) -> str:
    """Compute MD5 checksum of raw image pixels to identify render duplicates."""
    try:
        with Image.open(img_path) as img:
            pixels = np.array(img.convert("RGB")).tobytes()
        return hashlib.md5(pixels).hexdigest()
    except Exception:
        return ""

def run_dataset_audit(data_dir: str, train_csv_path: str, test_csv_path: str) -> dict:
    print(f"[Audit] Starting dataset integrity scan on: {data_dir}")
    
    report = {
        "train": {
            "total": 0, "corrupted": 0, "missing_npz": 0, "missing_png": 0,
            "nan_verts": 0, "inf_verts": 0, "zero_area_faces": 0, "duplicates": 0,
            "non_manifold_edges": 0, "non_watertight": 0, "self_intersections": 0
        },
        "test": {
            "total": 0, "corrupted": 0, "missing_npz": 0, "missing_png": 0,
            "nan_verts": 0, "inf_verts": 0, "zero_area_faces": 0, "duplicates": 0,
            "non_manifold_edges": 0, "non_watertight": 0, "self_intersections": 0
        },
        "statistics": {},
        "label_distribution": {},
        "inconsistencies": 0
    }
    
    # ── Audit Train Split ──────────────────────────────────────────────────
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.rename(columns=lambda x: x.replace("OUTPUT:", ""))
    report["train"]["total"] = len(train_df)
    
    defect_cols = ["abstract", "artifacts", "intersection", "lowpoly", "noisy", "open", "partial", "scale", "set", "simple"]
    
    # Track label stats
    for col in defect_cols:
        if col in train_df.columns:
            report["label_distribution"][col] = int(train_df[col].sum())
    if "quality" in train_df.columns:
        report["label_distribution"]["quality"] = int(train_df["quality"].sum())
        
        # Check label inconsistencies
        defects_sum = train_df[defect_cols].sum(axis=1).values
        quality = train_df["quality"].values
        inconsistent = ((quality == 1) & (defects_sum > 0)) | ((quality == 0) & (defects_sum == 0))
        report["inconsistencies"] = int(inconsistent.sum())
        if report["inconsistencies"] > 0:
            print(f"[Audit WARNING] Found {report['inconsistencies']} label inconsistencies in train.csv!")
            
    mesh_checksums = {}
    image_checksums = {}
    vertex_counts = []
    face_counts = []
    
    train_img_dir = os.path.join(data_dir, "train")
    for item_id in tqdm(train_df["item_id"].values, desc="Auditing Train files"):
        safe_id = os.path.basename(str(item_id))
        png_path = os.path.join(train_img_dir, f"{safe_id}.png")
        npz_path = os.path.join(train_img_dir, f"{safe_id}.npz")
        
        if not os.path.exists(png_path):
            report["train"]["missing_png"] += 1
        else:
            img_hash = compute_image_checksum(png_path)
            if img_hash:
                if img_hash in image_checksums:
                    report["train"]["duplicates"] += 1
                else:
                    image_checksums[img_hash] = item_id
                    
        if not os.path.exists(npz_path):
            report["train"]["missing_npz"] += 1
        else:
            try:
                data = np.load(npz_path, allow_pickle=False)
                v = data["vertices"]
                f = data["faces"]
                
                # Check NaNs and Inf
                if np.isnan(v).any():
                    report["train"]["nan_verts"] += 1
                if np.isinf(v).any():
                    report["train"]["inf_verts"] += 1
                    
                # Check degenerate faces
                if len(f) > 0 and len(v) > 0:
                    v0 = v[f[:, 0]]
                    v1 = v[f[:, 1]]
                    v2 = v[f[:, 2]]
                    cross = np.cross(v1 - v0, v2 - v0)
                    areas = 0.5 * np.linalg.norm(cross, axis=1)
                    zero_area = (areas < 1e-12).sum()
                    if zero_area > 0:
                        report["train"]["zero_area_faces"] += int(zero_area)
                
                # Check topological degeneracies using MeshValidityAnalyzer
                validity = MeshValidityAnalyzer.analyze_mesh(v, f)
                if validity["non_manifold_edge_count"] > 0:
                    report["train"]["non_manifold_edges"] += 1
                if validity["watertight"] < 0.5:
                    report["train"]["non_watertight"] += 1
                if validity["self_intersections"] > 0:
                    report["train"]["self_intersections"] += 1
                        
                # Check mesh duplicate
                m_hash = compute_mesh_checksum(v, f)
                if m_hash in mesh_checksums:
                    pass
                else:
                    mesh_checksums[m_hash] = item_id
                    
                vertex_counts.append(len(v))
                face_counts.append(len(f))
            except Exception:
                report["train"]["corrupted"] += 1
                
    # ── Audit Test Split ───────────────────────────────────────────────────
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        report["test"]["total"] = len(test_df)
        test_img_dir = os.path.join(data_dir, "test")
        
        for item_id in tqdm(test_df["item_id"].values, desc="Auditing Test files"):
            safe_id = os.path.basename(str(item_id))
            png_path = os.path.join(test_img_dir, f"{safe_id}.png")
            npz_path = os.path.join(test_img_dir, f"{safe_id}.npz")
            
            if not os.path.exists(png_path):
                report["test"]["missing_png"] += 1
            else:
                img_hash = compute_image_checksum(png_path)
                if img_hash:
                    if img_hash in image_checksums:
                        report["test"]["duplicates"] += 1
                        
            if not os.path.exists(npz_path):
                report["test"]["missing_npz"] += 1
            else:
                try:
                    data = np.load(npz_path, allow_pickle=False)
                    v = data["vertices"]
                    f = data["faces"]
                    if np.isnan(v).any():
                        report["test"]["nan_verts"] += 1
                    if np.isinf(v).any():
                        report["test"]["inf_verts"] += 1
                    if len(f) > 0 and len(v) > 0:
                        v0 = v[f[:, 0]]
                        v1 = v[f[:, 1]]
                        v2 = v[f[:, 2]]
                        cross = np.cross(v1 - v0, v2 - v0)
                        areas = 0.5 * np.linalg.norm(cross, axis=1)
                        zero_area = (areas < 1e-12).sum()
                        if zero_area > 0:
                            report["test"]["zero_area_faces"] += int(zero_area)
                    
                    # Check topological degeneracies using MeshValidityAnalyzer
                    validity = MeshValidityAnalyzer.analyze_mesh(v, f)
                    if validity["non_manifold_edge_count"] > 0:
                        report["test"]["non_manifold_edges"] += 1
                    if validity["watertight"] < 0.5:
                        report["test"]["non_watertight"] += 1
                    if validity["self_intersections"] > 0:
                        report["test"]["self_intersections"] += 1
                except Exception:
                    report["test"]["corrupted"] += 1
                    
    # Aggregated Stats
    if vertex_counts:
        report["statistics"] = {
            "vertex_mean": float(np.mean(vertex_counts)),
            "vertex_std": float(np.std(vertex_counts)),
            "vertex_min": int(np.min(vertex_counts)),
            "vertex_max": int(np.max(vertex_counts)),
            "face_mean": float(np.mean(face_counts)),
            "face_std": float(np.std(face_counts)),
            "face_min": int(np.min(face_counts)),
            "face_max": int(np.max(face_counts)),
        }
        
    print("[Audit Complete]")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    import sys
    # Add parent directory to sys.path to find config
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    log_dir = getattr(cfg, "LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    rep = run_dataset_audit(
        data_dir=getattr(cfg, "DATA_DIR", "data"),
        train_csv_path=getattr(cfg, "TRAIN_CSV", "data/train.csv"),
        test_csv_path=getattr(cfg, "TEST_CSV", "data/test.csv")
    )
    
    out_path = os.path.join(log_dir, "dataset_report.json")
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Audit report saved to: {out_path}")
