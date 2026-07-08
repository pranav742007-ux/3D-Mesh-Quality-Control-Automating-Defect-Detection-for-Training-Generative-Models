"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Industrial FastAPI Microservice  [v7.1]
===============================================================================
Production HTTP REST API microservice endpoints:
  - POST /api/v1/inspect: Accepts 100D mesh features & renders -> returns JSON QC report
  - POST /api/v1/repair: Uploads 3D mesh -> closes holes & purges degenerate faces
  - GET /health: Health check probe for AWS ALB / GCP Kubernetes
===============================================================================
"""

import os
import sys
import time
import tempfile
import numpy as np
from typing import Optional, List, Dict, Tuple

sol_dir = os.path.dirname(os.path.abspath(__file__))
if sol_dir not in sys.path:
    sys.path.insert(0, sol_dir)

try:
    from fastapi import FastAPI, HTTPException, File, UploadFile
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    class FastAPI:
        def __init__(self, **kwargs): pass
        def get(self, path): return lambda f: f
        def post(self, path): return lambda f: f
    class BaseModel: pass
    class Field:
        def __init__(self, **kwargs): pass
    class File: pass
    class UploadFile: pass

import torch
import config as cfg
from models import build_model_from_config
from utils import derive_quality, sigmoid, compute_uncertainty_scores

DEFECT_COLS = [
    "abstract", "artifacts", "intersection", "lowpoly",
    "noisy", "open", "partial", "scale", "set", "simple"
]

import io
import base64
import traceback
from PIL import Image
import torchvision.transforms as T

app = FastAPI(
    title="3D Mesh Quality Control Industrial API",
    description="Sub-50ms Multi-Modal 3D Mesh Quality Inspection REST Service",
    version="7.2",
)

GLOBAL_MODEL = None


def load_pytorch_model():
    global GLOBAL_MODEL
    try:
        mesh_dim = getattr(cfg, "MESH_FEATURE_DIM_EXTENDED", 100) if getattr(cfg, "USE_EXTENDED_FEATURES", True) else getattr(cfg, "MESH_FEATURE_DIM", 58)
        model = build_model_from_config(cfg, effective_mesh_dim=mesh_dim)
        ckpt_path = os.path.join(sol_dir, "checkpoints", "best_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            res = model.load_state_dict(state, strict=False)
            print(f"  [API Startup] Loaded PyTorch checkpoint from {ckpt_path}. Missing keys: {res.missing_keys}, Unexpected: {res.unexpected_keys}")
        else:
            print("  [API Startup] No checkpoint file found — initialized SOTA v7.2 architecture ready.")
        model.eval()
        GLOBAL_MODEL = model
    except Exception as e:
        print(f"  [API Startup Error] Could not initialize PyTorch model: {e}")


if HAS_FASTAPI:
    @app.on_event("startup")
    def startup_event():
        load_pytorch_model()


def parse_obj_bytes(obj_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Parse Wavefront .obj file bytes into (vertices, faces) numpy arrays."""
    text = obj_bytes.decode("utf-8", errors="ignore")
    verts = []
    faces = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            face_indices = []
            for p in parts[1:4]:
                idx_str = p.split("/")[0]
                idx = int(idx_str)
                # Handle negative 1-indexed relative offsets
                idx = idx - 1 if idx > 0 else len(verts) + idx
                face_indices.append(idx)
            faces.append(face_indices)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("No valid 3D vertices or faces found in .obj file")
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


class MeshInspectionRequest(BaseModel):
    item_id: str = "item_001"
    mesh_features: Optional[List[float]] = None  # 100-dim geometric vector
    views_base64: Optional[List[str]] = None     # List of 6 Base64-encoded PNG image strings
    views_flat: Optional[List[float]] = None     # Flat float array of shape (6, 3, 224, 224)
    effort: Optional[str] = "max"                # reasoning effort settings: fast, high, max


class MeshInspectionResponse(BaseModel):
    item_id: str
    quality: str  # "GOOD" or "BAD"
    quality_score: float
    defect_probabilities: Dict[str, float]
    defects_detected: List[str]
    confidence_metrics: Dict[str, float]
    requires_human_review: bool
    latency_ms: float


@app.get("/health")
def health_check() -> Dict:
    if GLOBAL_MODEL is None:
        load_pytorch_model()
    is_ready = GLOBAL_MODEL is not None
    return {
        "status": "HEALTHY" if is_ready else "DEGRADED",
        "model_loaded": is_ready,
        "device": "cpu",
        "version": "7.2.0",
    }


@app.post("/api/v1/inspect")
def inspect_mesh(request: MeshInspectionRequest) -> Dict:
    t0 = time.time()
    
    if GLOBAL_MODEL is None:
        load_pytorch_model()
        if GLOBAL_MODEL is None:
            raise HTTPException(status_code=503, detail="PyTorch model runtime not initialized")

    if request.mesh_features is None and request.views_base64 is None and request.views_flat is None:
        raise HTTPException(status_code=422, detail="Request payload must contain at least mesh_features or views_base64 or views_flat")

    try:
        views_t = torch.zeros(1, 6, 3, 224, 224)
        if request.views_base64 is not None and len(request.views_base64) == 6:
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            view_tensors = []
            for b64_str in request.views_base64:
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                view_tensors.append(transform(img))
            views_t = torch.stack(view_tensors, dim=0).unsqueeze(0)  # (1, 6, 3, 224, 224)
        elif request.views_flat is not None and len(request.views_flat) == 6 * 3 * 224 * 224:
            views_t = torch.tensor(request.views_flat, dtype=torch.float32).reshape(1, 6, 3, 224, 224)

        mesh_t = None
        if request.mesh_features is not None:
            mesh_t = torch.tensor([request.mesh_features], dtype=torch.float32)
        
        with torch.no_grad():
            logits = GLOBAL_MODEL(views_t, mesh_t, effort=request.effort)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()[0]
    except Exception as e:
        print(f"  [API Error Traceback] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Model inference execution failed: {str(e)}")
    
    # P1-18 FIX: Load calibrated per-class thresholds from CV results if available
    thresholds = 0.5
    thresh_path = os.path.join(sol_dir, "checkpoints", "cv_results.json")
    if os.path.exists(thresh_path):
        try:
            import json
            with open(thresh_path, "r") as f:
                res_data = json.load(f)
                if "best_thresholds" in res_data:
                    thresholds = np.array(res_data["best_thresholds"])
        except Exception:
            pass

    preds = (probs >= thresholds).astype(int)
    quality_val = int(np.sum(preds) == 0)
    quality_str = "GOOD" if quality_val == 1 else "BAD"

    detected = [DEFECT_COLS[i] for i in range(10) if preds[i] == 1]
    prob_dict = {DEFECT_COLS[i]: round(float(probs[i]), 4) for i in range(10)}
    
    uncertainty = compute_uncertainty_scores(probs)
    latency = round((time.time() - t0) * 1000.0, 2)

    return {
        "item_id": request.item_id,
        "quality": quality_str,
        "quality_score": round(float(1.0 - np.mean(probs)), 4),
        "defect_probabilities": prob_dict,
        "defects_detected": detected,
        "confidence_metrics": uncertainty,
        "requires_human_review": uncertainty["requires_human_review"],
        "latency_ms": latency,
    }


@app.post("/api/v1/repair")
async def repair_mesh_endpoint(mesh_file: UploadFile = File(...)):
    """
    Automated 3D Geometric Mesh Repair Endpoint (v7.2 Ground Reality).
    Accepts .npz or .obj 3D mesh files, executes ear-clipping hole filling and
    degenerate face purging, returning clean mesh status and repaired .obj geometry.
    """
    if not HAS_FASTAPI:
        raise HTTPException(status_code=500, detail="FastAPI server runtime not available")

    filename = os.path.basename(mesh_file.filename or "mesh.npz")
    if not (filename.endswith(".npz") or filename.endswith(".obj")):
        raise HTTPException(status_code=400, detail="Only .npz or .obj files are supported")

    content = await mesh_file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # P1-19 FIX: Enforce 50MB payload size limit to prevent DoS memory exhaustion
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Uploaded mesh file exceeds 50MB maximum payload limit")

    # P1-20 FIX: Secure randomized UUID temporary filename to prevent TOCTOU symlink attacks
    import uuid
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}_{filename}")
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        from mesh_repair import auto_repair_mesh
        if filename.endswith(".npz"):
            data = np.load(tmp_path, allow_pickle=False)
            vertices = data["vertices"]
            faces = data["faces"]
        else:
            vertices, faces = parse_obj_bytes(content)

        repaired_verts, repaired_faces, report = auto_repair_mesh(vertices, faces)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Stream the repaired OBJ file (v7.2 Ground Reality) to prevent high memory usage
        from fastapi.responses import StreamingResponse

        def obj_generator():
            yield f"# Repaired Wavefront .obj mesh (v7.2 Ground Reality)\n"
            yield f"# Vertices: {len(repaired_verts)}, Faces: {len(repaired_faces)}\n"
            for v in repaired_verts:
                yield f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
            for f in repaired_faces:
                yield f"f {f[0]+1} {f[1]+1} {f[2]+1}\n"

        headers = {
            "Content-Disposition": f"attachment; filename=repaired_{filename.replace('.npz', '.obj')}",
            "X-Repaired": str(report["repaired"]),
            "X-Degenerate-Faces-Purged": str(report["degenerate_faces_purged"]),
            "X-Boundary-Holes-Filled": str(report["boundary_holes_filled"]),
            "X-Final-Vertex-Count": str(report["final_vertex_count"]),
            "X-Final-Face-Count": str(report["final_face_count"])
        }
        return StreamingResponse(obj_generator(), media_type="model/obj", headers=headers)

    except ValueError as ve:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=422, detail=f"Invalid 3D mesh file content: {str(ve)}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Mesh repair execution failed: {str(e)}")


if __name__ == "__main__":
    load_pytorch_model()
    if HAS_FASTAPI:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not installed locally. Install via `pip install fastapi uvicorn` to run web server.")
