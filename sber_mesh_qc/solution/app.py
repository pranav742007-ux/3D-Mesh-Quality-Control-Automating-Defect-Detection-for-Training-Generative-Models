"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Universal Deployment Engine [v7.2]
===============================================================================
Single-file deployment that runs on ANY target:

  MODE 1 — REST API Server (Cloud / Docker / Kubernetes):
      python app.py --mode server
      Endpoints: POST /api/v1/inspect, POST /api/v1/repair, GET /health

  MODE 2 — Desktop CLI (PC / Mac / Linux):
      python app.py --mode cli --input model.obj
      python app.py --mode cli --input model.npz --effort max

  MODE 3 — Live Camera Feed (Smart Camera / Webcam):
      python app.py --mode camera --device 0
      python app.py --mode camera --device /dev/video0 --gpio

  MODE 4 — Batch Directory Processing (Offline QA Pipeline):
      python app.py --mode batch --input-dir ./meshes/ --output report.csv

  MODE 5 — Edge ONNX Inference (Jetson / Coral / VPU):
      python app.py --mode cli --input model.obj --backend onnx

Backends:
  - pytorch : Full PyTorch model (default, requires torch)
  - onnx    : ONNX Runtime (lightweight, no torch dependency for inference)
===============================================================================
"""

import os
import sys
import time
import json
import argparse
import tempfile
import numpy as np
from typing import Optional, List, Dict, Tuple

sol_dir = os.path.dirname(os.path.abspath(__file__))
if sol_dir not in sys.path:
    sys.path.insert(0, sol_dir)

# ─── Optional Dependency Detection ──────────────────────────────────────────
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
    def File(*args, **kwargs): pass
    class UploadFile: pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

import io
import base64
import traceback

DEFECT_COLS = [
    "abstract", "artifacts", "intersection", "lowpoly",
    "noisy", "open", "partial", "scale", "set", "simple"
]


# =============================================================================
# SECTION 1: Universal Inference Engine (Abstracts PyTorch vs ONNX backends)
# =============================================================================

class UniversalInferenceEngine:
    """
    Unified inference engine that transparently supports both PyTorch and
    ONNX Runtime backends. Provides a single `.predict()` interface regardless
    of the underlying execution provider (GPU, CPU, TensorRT, OpenVINO, etc).
    """

    def __init__(self, backend: str = "pytorch", checkpoint_path: str = None,
                 onnx_path: str = None, device: str = "auto"):
        self.backend = backend
        self.model = None
        self.session = None
        self.device_str = device
        self.thresholds = 0.5
        self._load_calibrated_thresholds()

        if backend == "onnx":
            self._init_onnx(onnx_path)
        else:
            self._init_pytorch(checkpoint_path)

    def _load_calibrated_thresholds(self):
        """Load per-class calibrated thresholds from cross-validation results."""
        thresh_path = os.path.join(sol_dir, "checkpoints", "cv_results.json")
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path, "r") as f:
                    res_data = json.load(f)
                    if "best_thresholds" in res_data:
                        self.thresholds = np.array(res_data["best_thresholds"])
            except Exception:
                pass

    def _init_pytorch(self, checkpoint_path: str = None):
        """Initialize the full PyTorch model with trained weights."""
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed. Install via `pip install torch torchvision` "
                "or use --backend onnx for lightweight edge inference."
            )
        import config as cfg
        from models import build_model_from_config
        from utils import clean_state_dict_keys

        mesh_dim = (getattr(cfg, "MESH_FEATURE_DIM_EXTENDED", 100)
                     if getattr(cfg, "USE_EXTENDED_FEATURES", True)
                     else getattr(cfg, "MESH_FEATURE_DIM", 58))
        model = build_model_from_config(cfg, effective_mesh_dim=mesh_dim)

        self.scaler_mean = None
        self.scaler_std = None

        ckpt_path = checkpoint_path or os.path.join(sol_dir, "checkpoints", "best_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            state = clean_state_dict_keys(state, model)
            res = model.load_state_dict(state, strict=False)
            print(f"  [Engine] Loaded PyTorch checkpoint: {ckpt_path}")
            if isinstance(ckpt, dict) and "scaler_mean" in ckpt:
                self.scaler_mean = ckpt["scaler_mean"]
                self.scaler_std = ckpt["scaler_std"]
                print("           Loaded StandardScaler3D statistics.")
            if res.missing_keys:
                print(f"           Missing keys: {res.missing_keys[:5]}...")
        else:
            print(f"  [Engine] No checkpoint found at {ckpt_path} — using random weights.")

        # Auto-detect best available device
        if self.device_str == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        model = model.to(self.device)
        model.eval()
        self.model = model
        print(f"  [Engine] PyTorch backend ready on {self.device_str.upper()}")

    def _init_onnx(self, onnx_path: str = None):
        """Initialize the ONNX Runtime inference session."""
        if not HAS_ONNX:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install via `pip install onnxruntime` "
                "(CPU) or `pip install onnxruntime-gpu` (CUDA/TensorRT)."
            )
        model_path = onnx_path or os.path.join(sol_dir, "checkpoints", "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Export it first: python export_onnx.py --output {model_path}"
            )

        # Auto-select the fastest available execution provider
        available = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "OpenVINOExecutionProvider" in available:
            providers.append("OpenVINOExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        active_provider = self.session.get_providers()[0]
        print(f"  [Engine] ONNX Runtime backend ready — Provider: {active_provider}")

    def predict(self, views: np.ndarray, mesh_features: np.ndarray = None,
                effort: str = "max") -> Dict:
        """
        Run inference on a single sample and return a structured QC report.

        Args:
            views: (1, 6, 3, 224, 224) or (1, 6, 6, 224, 224) float32 array
            mesh_features: (1, 100) float32 array or None
            effort: 'fast', 'high', or 'max'

        Returns:
            dict with quality, defect_probabilities, confidence, latency, etc.
        """
        t0 = time.time()

        if mesh_features is not None and self.scaler_mean is not None:
            # Scale input features using serialized scaler parameters (H6)
            mesh_features = np.where(np.isnan(mesh_features) | np.isinf(mesh_features), self.scaler_mean, mesh_features)
            mesh_features = (mesh_features - self.scaler_mean) / (self.scaler_std + 1e-7)

        if self.backend == "onnx":
            probs = self._predict_onnx(views, mesh_features)
        else:
            probs = self._predict_pytorch(views, mesh_features, effort)

        # Apply calibrated thresholds
        preds = (probs >= self.thresholds).astype(int)
        quality_val = int(np.sum(preds) == 0)

        detected = [DEFECT_COLS[i] for i in range(10) if preds[i] == 1]
        prob_dict = {DEFECT_COLS[i]: round(float(probs[i]), 4) for i in range(10)}

        # Compute epistemic uncertainty scores
        from utils import compute_uncertainty_scores
        uncertainty = compute_uncertainty_scores(probs)
        latency = round((time.time() - t0) * 1000.0, 2)

        return {
            "quality": "GOOD" if quality_val == 1 else "BAD",
            "quality_score": round(float(1.0 - np.mean(probs)), 4),
            "defect_probabilities": prob_dict,
            "defects_detected": detected,
            "confidence_metrics": uncertainty,
            "requires_human_review": uncertainty["requires_human_review"],
            "latency_ms": latency,
        }

    def _predict_pytorch(self, views: np.ndarray, mesh_features: np.ndarray,
                         effort: str) -> np.ndarray:
        """Run inference through the full PyTorch model."""
        views_t = torch.tensor(views, dtype=torch.float32).to(self.device)
        mesh_t = None
        if mesh_features is not None:
            mesh_t = torch.tensor(mesh_features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(views_t, mesh_t, effort=effort)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs

    def _predict_onnx(self, views: np.ndarray, mesh_features: np.ndarray) -> np.ndarray:
        """Run inference through the ONNX Runtime session."""
        input_names = [inp.name for inp in self.session.get_inputs()]
        feed = {input_names[0]: views.astype(np.float32)}
        if len(input_names) > 1 and mesh_features is not None:
            feed[input_names[1]] = mesh_features.astype(np.float32)
        elif len(input_names) > 1:
            feed[input_names[1]] = np.zeros((1, 100), dtype=np.float32)

        outputs = self.session.run(None, feed)
        logits = outputs[0][0]
        # Numerically stable sigmoid
        probs = np.where(logits >= 0,
                         1.0 / (1.0 + np.exp(-logits)),
                         np.exp(logits) / (1.0 + np.exp(logits)))
        return probs


# =============================================================================
# SECTION 2: Image Preprocessing Utilities
# =============================================================================

def preprocess_image_pil(pil_image, target_size: int = 224) -> np.ndarray:
    """Convert a PIL Image to a (3, H, W) float32 tensor normalized to [0, 1]."""
    from PIL import Image
    img = pil_image.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


def preprocess_image_cv2(bgr_frame: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Convert an OpenCV BGR frame to a (3, H, W) float32 tensor normalized to [0, 1]."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


def load_mesh_features_from_file(file_path: str) -> Optional[np.ndarray]:
    """Extract the 100D geometric feature vector from a .npz or .obj mesh file."""
    try:
        from mesh_features import extract_mesh_features_from_file
        if file_path.endswith(".npz"):
            feat = extract_mesh_features_from_file(file_path, extended=True)
            return feat.reshape(1, -1)
        elif file_path.endswith(".obj"):
            # Parse OBJ into vertices/faces, save as temp npz, extract
            with open(file_path, "rb") as f:
                verts, faces = parse_obj_bytes(f.read())
            tmp_path = os.path.join(tempfile.gettempdir(), "temp_mesh.npz")
            np.savez(tmp_path, vertices=verts, faces=faces)
            feat = extract_mesh_features_from_file(tmp_path, extended=True)
            os.remove(tmp_path)
            return feat.reshape(1, -1)
    except Exception as e:
        print(f"  [Warning] Could not extract mesh features: {e}")
    return None


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
                idx = idx - 1 if idx > 0 else len(verts) + idx
                face_indices.append(idx)
            faces.append(face_indices)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("No valid 3D vertices or faces found in .obj file")
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


# =============================================================================
# SECTION 3: MODE 1 — REST API Server (FastAPI + Uvicorn)
# =============================================================================

app = FastAPI(
    title="3D Mesh Quality Control — Universal Deployment Engine",
    description="Sub-50ms Multi-Modal 3D Mesh Quality Inspection (v7.3)",
    version="7.3.0",
)

GLOBAL_ENGINE: Optional[UniversalInferenceEngine] = None


def _ensure_engine():
    """Lazy-initialize the global inference engine on first request."""
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is None:
        GLOBAL_ENGINE = UniversalInferenceEngine(backend="pytorch")


class MeshInspectionRequest(BaseModel):
    item_id: str = "item_001"
    mesh_features: Optional[List[float]] = None
    views_base64: Optional[List[str]] = None
    views_flat: Optional[List[float]] = None
    effort: Optional[str] = "max"


class MeshInspectionResponse(BaseModel):
    item_id: str
    quality: str
    quality_score: float
    defect_probabilities: Dict[str, float]
    defects_detected: List[str]
    confidence_metrics: Dict[str, float]
    requires_human_review: bool
    latency_ms: float


if HAS_FASTAPI:
    @app.on_event("startup")
    def startup_event():
        _ensure_engine()


@app.get("/health")
def health_check() -> Dict:
    _ensure_engine()
    return {
        "status": "HEALTHY" if GLOBAL_ENGINE is not None else "DEGRADED",
        "model_loaded": GLOBAL_ENGINE is not None,
        "backend": GLOBAL_ENGINE.backend if GLOBAL_ENGINE else "none",
        "version": "7.3.0",
    }


@app.post("/api/v1/inspect")
def inspect_mesh(request: MeshInspectionRequest) -> Dict:
    _ensure_engine()
    if GLOBAL_ENGINE is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    if request.mesh_features is None and request.views_base64 is None and request.views_flat is None:
        raise HTTPException(status_code=422, detail="Provide mesh_features, views_base64, or views_flat")

    try:
        from PIL import Image
        views = np.zeros((1, 6, 3, 224, 224), dtype=np.float32)
        if request.views_base64 is not None and len(request.views_base64) == 6:
            tensors = []
            for b64_str in request.views_base64:
                img = Image.open(io.BytesIO(base64.b64decode(b64_str)))
                tensors.append(preprocess_image_pil(img))
            views = np.stack(tensors, axis=0)[np.newaxis, ...]
        elif request.views_flat is not None and len(request.views_flat) == 6 * 3 * 224 * 224:
            views = np.array(request.views_flat, dtype=np.float32).reshape(1, 6, 3, 224, 224)

        mesh_t = None
        if request.mesh_features is not None:
            mesh_t = np.array([request.mesh_features], dtype=np.float32)

        result = GLOBAL_ENGINE.predict(views, mesh_t, effort=request.effort or "max")
        result["item_id"] = request.item_id
        return result

    except Exception as e:
        print(f"  [API Error] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/v1/repair")
async def repair_mesh_endpoint(mesh_file: UploadFile = File(...)):
    """
    Automated 3D Mesh Repair: ear-clipping hole filling + degenerate face purging.
    Accepts .npz or .obj files, streams back a cleaned Wavefront .obj geometry.
    """
    if not HAS_FASTAPI:
        raise HTTPException(status_code=500, detail="FastAPI not available")

    filename = os.path.basename(mesh_file.filename or "mesh.npz")
    if not (filename.endswith(".npz") or filename.endswith(".obj")):
        raise HTTPException(status_code=400, detail="Only .npz or .obj files supported")

    content = await mesh_file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50MB limit")

    import uuid
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}_{filename}")
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        from mesh_repair import auto_repair_mesh
        if filename.endswith(".npz"):
            data = np.load(tmp_path, allow_pickle=False)
            vertices, faces = data["vertices"], data["faces"]
        else:
            vertices, faces = parse_obj_bytes(content)

        repaired_verts, repaired_faces, report = auto_repair_mesh(vertices, faces)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        from fastapi.responses import StreamingResponse

        def obj_generator():
            yield f"# Repaired mesh (v7.2) | V:{len(repaired_verts)} F:{len(repaired_faces)}\n"
            for v in repaired_verts:
                yield f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
            for face in repaired_faces:
                yield f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"

        headers = {
            "Content-Disposition": f"attachment; filename=repaired_{filename.replace('.npz', '.obj')}",
            "X-Repaired": str(report["repaired"]),
            "X-Degenerate-Faces-Purged": str(report["degenerate_faces_purged"]),
            "X-Boundary-Holes-Filled": str(report["boundary_holes_filled"]),
            "X-Final-Vertex-Count": str(report["final_vertex_count"]),
            "X-Final-Face-Count": str(report["final_face_count"]),
        }
        return StreamingResponse(obj_generator(), media_type="model/obj", headers=headers)

    except ValueError as ve:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SECTION 4: MODE 2 — Desktop CLI (Single File Inspection)
# =============================================================================

def run_cli_mode(args):
    """
    Inspect a single 3D mesh file from the command line.
    Outputs a JSON QC report to stdout (or to --output file).
    """
    print("=" * 60)
    print("  3D MESH QUALITY CONTROL — CLI INSPECTION MODE")
    print("=" * 60)

    engine = UniversalInferenceEngine(
        backend=args.backend,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx_model,
    )

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"  [Error] File not found: {input_path}")
        sys.exit(1)

    # Extract mesh geometry features if available
    mesh_feat = load_mesh_features_from_file(input_path)

    # For CLI without images, use zero-filled views (geometry-only mode)
    views = np.zeros((1, 6, 3, 224, 224), dtype=np.float32)

    # If image directory is provided, load the 6 multi-view renders
    if args.views_dir and os.path.isdir(args.views_dir):
        from PIL import Image
        img_files = sorted([f for f in os.listdir(args.views_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:6]
        if img_files:
            tensors = []
            for fname in img_files:
                img = Image.open(os.path.join(args.views_dir, fname))
                tensors.append(preprocess_image_pil(img))
            # Pad to 6 views if fewer provided
            while len(tensors) < 6:
                tensors.append(np.zeros((3, 224, 224), dtype=np.float32))
            views = np.stack(tensors[:6], axis=0)[np.newaxis, ...]
            print(f"  Loaded {len(img_files)} view images from {args.views_dir}")

    result = engine.predict(views, mesh_feat, effort=args.effort)
    result["input_file"] = os.path.basename(input_path)
    result["backend"] = args.backend

    # Pretty-print the result
    print("\n" + "─" * 60)
    quality_icon = "✅" if result["quality"] == "GOOD" else "❌"
    print(f"  {quality_icon}  Quality: {result['quality']} (score: {result['quality_score']:.4f})")
    print(f"  ⏱  Latency: {result['latency_ms']:.1f} ms")
    if result["defects_detected"]:
        print(f"  🔍 Defects: {', '.join(result['defects_detected'])}")
    else:
        print("  🔍 Defects: None detected")
    print(f"  📊 Confidence: {result['confidence_metrics']['confidence_percent']:.1f}%")
    if result["requires_human_review"]:
        print("  ⚠️  FLAGGED FOR HUMAN REVIEW (low confidence)")
    print("─" * 60)

    # Save JSON report if --output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Report saved to {args.output}")

    return result


# =============================================================================
# SECTION 5: MODE 3 — Live Camera Feed (Smart Camera / Webcam / Edge Device)
# =============================================================================

def run_camera_mode(args):
    """
    Continuously capture frames from a camera sensor, run real-time QC inference,
    and optionally trigger GPIO pins for hardware sorting (pneumatic arms, gates).
    """
    if not HAS_OPENCV:
        print("[Error] OpenCV not installed. Install via: pip install opencv-python-headless")
        sys.exit(1)

    print("=" * 60)
    print("  3D MESH QUALITY CONTROL — LIVE CAMERA MODE")
    print("=" * 60)

    engine = UniversalInferenceEngine(
        backend=args.backend,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx_model,
    )

    # Initialize GPIO for edge hardware control (Jetson / Raspberry Pi)
    gpio_enabled = args.gpio
    if gpio_enabled:
        try:
            import Jetson.GPIO as GPIO
            REJECT_PIN = args.gpio_reject_pin
            ACCEPT_PIN = args.gpio_accept_pin
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(REJECT_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(ACCEPT_PIN, GPIO.OUT, initial=GPIO.LOW)
            print(f"  GPIO initialized: REJECT={REJECT_PIN}, ACCEPT={ACCEPT_PIN}")
        except ImportError:
            try:
                import RPi.GPIO as GPIO
                REJECT_PIN = args.gpio_reject_pin
                ACCEPT_PIN = args.gpio_accept_pin
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(REJECT_PIN, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(ACCEPT_PIN, GPIO.OUT, initial=GPIO.LOW)
                print(f"  RPi.GPIO initialized: REJECT={REJECT_PIN}, ACCEPT={ACCEPT_PIN}")
            except ImportError:
                print("  [Warning] No GPIO library found. Running in display-only mode.")
                gpio_enabled = False

    # Open camera
    device = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"  [Error] Cannot open camera device: {args.device}")
        sys.exit(1)
    print(f"  Camera device {args.device} opened. Press 'q' to quit.\n")

    frame_count = 0
    inspect_interval = max(1, args.inspect_every)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  [Warning] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % inspect_interval != 0:
                # Show live feed without inference on intermediate frames
                if not args.headless:
                    cv2.imshow("3D Mesh QC — Live Feed", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Pre-process the captured frame
            img_tensor = preprocess_image_cv2(frame)
            # Replicate as 6 views (single-camera setup)
            views = np.stack([img_tensor] * 6, axis=0)[np.newaxis, ...]

            # Run QC inference
            result = engine.predict(views, mesh_features=None, effort=args.effort)

            # Print result
            icon = "✅" if result["quality"] == "GOOD" else "❌"
            print(f"  Frame {frame_count:>6d} | {icon} {result['quality']:4s} | "
                  f"Conf: {result['confidence_metrics']['confidence_percent']:5.1f}% | "
                  f"Defects: {result['defects_detected'] or 'None':30s} | "
                  f"{result['latency_ms']:.1f}ms")

            # Trigger GPIO hardware sorting
            if gpio_enabled:
                if result["quality"] == "GOOD":
                    GPIO.output(ACCEPT_PIN, GPIO.HIGH)
                    GPIO.output(REJECT_PIN, GPIO.LOW)
                else:
                    GPIO.output(REJECT_PIN, GPIO.HIGH)
                    GPIO.output(ACCEPT_PIN, GPIO.LOW)
                # Brief pulse then reset
                time.sleep(0.05)
                GPIO.output(ACCEPT_PIN, GPIO.LOW)
                GPIO.output(REJECT_PIN, GPIO.LOW)

            # Display annotated frame
            if not args.headless:
                color = (0, 255, 0) if result["quality"] == "GOOD" else (0, 0, 255)
                label = f"{result['quality']} ({result['confidence_metrics']['confidence_percent']:.0f}%)"
                cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, color, 3, cv2.LINE_AA)
                if result["defects_detected"]:
                    defect_text = ", ".join(result["defects_detected"][:3])
                    cv2.putText(frame, defect_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("3D Mesh QC — Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n  [Camera] Stopped by user.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if gpio_enabled:
            GPIO.cleanup()
        print("  [Camera] Resources released.")


# =============================================================================
# SECTION 6: MODE 4 — Batch Directory Processing (Offline QA Pipeline)
# =============================================================================

def run_batch_mode(args):
    """
    Process all .npz / .obj mesh files in a directory and generate a CSV QA report.
    """
    print("=" * 60)
    print("  3D MESH QUALITY CONTROL — BATCH PROCESSING MODE")
    print("=" * 60)

    engine = UniversalInferenceEngine(
        backend=args.backend,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx_model,
    )

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"  [Error] Directory not found: {input_dir}")
        sys.exit(1)

    mesh_files = sorted([f for f in os.listdir(input_dir)
                         if f.endswith('.npz') or f.endswith('.obj')])

    if not mesh_files:
        print(f"  [Error] No .npz or .obj files found in {input_dir}")
        sys.exit(1)

    print(f"  Found {len(mesh_files)} mesh files. Processing...\n")

    results = []
    for i, fname in enumerate(mesh_files):
        fpath = os.path.join(input_dir, fname)
        mesh_feat = load_mesh_features_from_file(fpath)
        views = np.zeros((1, 6, 3, 224, 224), dtype=np.float32)

        result = engine.predict(views, mesh_feat, effort=args.effort)
        result["filename"] = fname

        icon = "✅" if result["quality"] == "GOOD" else "❌"
        print(f"  [{i+1:>4d}/{len(mesh_files)}] {icon} {fname:40s} "
              f"| {result['quality']:4s} | {result['latency_ms']:.1f}ms "
              f"| Defects: {result['defects_detected'] or 'None'}")
        results.append(result)

    # Generate CSV report
    output_path = args.output or "batch_qc_report.csv"
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "quality", "quality_score", "confidence_pct",
                   "requires_review", "latency_ms"] + DEFECT_COLS
        writer.writerow(header)
        for r in results:
            row = [
                r["filename"], r["quality"], r["quality_score"],
                r["confidence_metrics"]["confidence_percent"],
                r["requires_human_review"], r["latency_ms"]
            ] + [r["defect_probabilities"].get(c, 0.0) for c in DEFECT_COLS]
            writer.writerow(row)

    # Summary statistics
    total = len(results)
    good = sum(1 for r in results if r["quality"] == "GOOD")
    bad = total - good
    flagged = sum(1 for r in results if r["requires_human_review"])
    avg_latency = np.mean([r["latency_ms"] for r in results])

    print(f"\n{'=' * 60}")
    print(f"  BATCH REPORT SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Total meshes:       {total}")
    print(f"  ✅ GOOD:            {good} ({good/total*100:.1f}%)")
    print(f"  ❌ BAD:             {bad} ({bad/total*100:.1f}%)")
    print(f"  ⚠️  Human review:   {flagged} ({flagged/total*100:.1f}%)")
    print(f"  ⏱  Avg latency:    {avg_latency:.1f} ms/mesh")
    print(f"  📄 Report saved:    {output_path}")
    print(f"{'=' * 60}")


# =============================================================================
# SECTION 7: CLI Argument Parser & Entry Point
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D Mesh Quality Control — Universal Deployment Engine (v7.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Server mode:    python app.py --mode server
  CLI inspection: python app.py --mode cli --input model.obj
  Camera feed:    python app.py --mode camera --device 0
  Batch QA:       python app.py --mode batch --input-dir ./meshes/ --output report.csv
  Edge ONNX:      python app.py --mode cli --input model.npz --backend onnx
        """
    )

    parser.add_argument("--mode", type=str, default="server",
                        choices=["server", "cli", "camera", "batch", "benchmark"],
                        help="Deployment mode (default: server)")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "onnx"],
                        help="Inference backend (default: pytorch)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="Path to ONNX model file (.onnx)")
    parser.add_argument("--effort", type=str, default="max",
                        choices=["fast", "high", "max"],
                        help="Inference effort level (default: max)")

    # CLI mode
    parser.add_argument("--input", type=str, default=None,
                        help="Input mesh file path (.npz or .obj)")
    parser.add_argument("--views-dir", type=str, default=None,
                        help="Directory containing 6 multi-view render images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (JSON for cli, CSV for batch)")

    # Camera mode
    parser.add_argument("--device", type=str, default="0",
                        help="Camera device index or path (default: 0)")
    parser.add_argument("--inspect-every", type=int, default=30,
                        help="Run inference every N frames (default: 30)")
    parser.add_argument("--headless", action="store_true",
                        help="Run camera mode without display window")
    parser.add_argument("--gpio", action="store_true",
                        help="Enable GPIO pins for hardware sorting control")
    parser.add_argument("--gpio-reject-pin", type=int, default=18,
                        help="GPIO pin for reject signal (default: 18)")
    parser.add_argument("--gpio-accept-pin", type=int, default=23,
                        help="GPIO pin for accept signal (default: 23)")

    # Batch mode
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing .npz/.obj mesh files")

    # Server mode
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server bind port (default: 8000)")

    return parser


def run_benchmark_mode(args):
    print(f"\n{'=' * 60}")
    print(f"  DEPLOYMENT BENCHMARK SUITE")
    print(f"{'=' * 60}")
    
    # 1. Create a synthetic mesh
    v = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5]
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ], dtype=np.int32)
    
    # Render synthetic views using DirectMeshRasterizer
    from image_processing import DirectMeshRasterizer
    views_tensor = DirectMeshRasterizer.rasterize_views(v, f, img_size=224).unsqueeze(0)
    views_tensor = torch.cat([views_tensor, torch.zeros((1, 6, 3, 224, 224))], dim=2) # 8 channels
    mesh_feat = torch.zeros((1, 103), dtype=torch.float32)
    
    print("  [1/2] Loading PyTorch inference engine...")
    try:
        from models import MultiViewImageModel, MeshFeatureMLP, FusedEnsembleModel
        img_model = MultiViewImageModel(in_channels=8, embed_dim=1280)
        mesh_model = MeshFeatureMLP(input_dim=103)
        model = FusedEnsembleModel(image_model=img_model, mesh_model=mesh_model, fusion_method="gated")
        model.eval()
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(views_tensor, mesh_feat)
                
        # Benchmark runs
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(views_tensor, mesh_feat)
            latencies.append((time.perf_counter() - start) * 1000.0)
            
        avg_lat = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        fps = 1000.0 / avg_lat
        
        print("  [RESULT] PyTorch Backend:")
        print(f"    Avg Latency:    {avg_lat:.2f} ms")
        print(f"    Median (p50):   {p50:.2f} ms")
        print(f"    p95 Latency:    {p95:.2f} ms")
        print(f"    Throughput:     {fps:.1f} FPS")
    except Exception as e:
        print(f"  [ERROR] PyTorch benchmark failed: {e}")
        
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "server":
        # ── REST API Server Mode ──
        _ensure_engine()
        if HAS_FASTAPI:
            import uvicorn
            uvicorn.run(app, host=args.host, port=args.port)
        else:
            print("[Error] FastAPI not installed. Run: pip install fastapi uvicorn")
            sys.exit(1)

    elif args.mode == "cli":
        # ── Desktop CLI Mode ──
        if not args.input:
            print("[Error] --input is required for CLI mode.")
            parser.print_help()
            sys.exit(1)
        run_cli_mode(args)

    elif args.mode == "camera":
        # ── Live Camera Mode ──
        run_camera_mode(args)

    elif args.mode == "batch":
        # ── Batch Directory Mode ──
        if not args.input_dir:
            print("[Error] --input-dir is required for batch mode.")
            parser.print_help()
            sys.exit(1)
        run_batch_mode(args)

    elif args.mode == "benchmark":
        # ── Benchmarking Mode ──
        run_benchmark_mode(args)
