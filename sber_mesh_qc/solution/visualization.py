"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Visualization  [v7.1 Master Engine]
===============================================================================
Methods for explaining model predictions and localizing defects:
  1. MultiViewGradCAM (6-view class-discriminative saliency heatmaps)
  2. View attention weights analysis across orthographic renders
  3. Perturbation-based 100D mesh feature sensitivity analysis
  4. Combined publication-quality defect localization figures
===============================================================================
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from typing import Optional


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



# ═══════════════════════════════════════════════════════════════════════════
# 1. GRAD-CAM FOR MULTI-VIEW MODEL
# ═══════════════════════════════════════════════════════════════════════════

class MultiViewGradCAM:
    """
    Grad-CAM for the multi-view image model.
    
    Generates class-discriminative saliency maps for each of the 6 views,
    showing which image regions the model uses for each defect prediction.
    
    v2.1 FIX: Now compatible with both parallel and sequential view processing.
    When sequential_views=True, processes each view individually to avoid
    hook overwrite issues (previous version would only capture the last view).
    
    Usage:
        cam = MultiViewGradCAM(model)
        # For a single item:
        cams = cam.generate(views_tensor, target_class=3)  # 6 heatmaps
    """
    
    def __init__(self, model, target_layer_name: str = None):
        """
        Args:
            model: FusedEnsembleModel or AgenticEnsembleModel wrapper
            target_layer_name: name of the last conv layer (auto-detected if None)
        """
        self.model = model
        self.model.eval()
        
        # Resolve base model if wrapped in AgenticEnsembleModel wrapper
        from models import AgenticEnsembleModel
        self.raw_model = model.base_model if isinstance(model, AgenticEnsembleModel) else model
        
        self.sequential_views = getattr(self.raw_model.image_model, "sequential_views", False)
        
        # Auto-detect the last convolutional layer in the backbone
        if target_layer_name is None:
            self.target_layer = self._find_last_conv(self.raw_model.image_model.feature_extractor)
        else:
            self.target_layer = dict(self.raw_model.image_model.named_modules())[target_layer_name]
        
        self.handles = []
    
    def _find_last_conv(self, module):
        """Find the last Conv2d layer in a module."""
        last_conv = None
        for name, child in module.named_modules():
            if isinstance(child, torch.nn.Conv2d):
                last_conv = child
        return last_conv
    
    def _generate_per_view(self, view_tensor, target_class, device):
        """
        Generate Grad-CAM for a single view batch: (B, 3, H, W).
        Returns (B, h, w) CAM numpy array.
        """
        gradients = None
        activations = None
        
        def forward_hook(module, inp, out):
            nonlocal activations
            activations = out.detach()
        
        def backward_hook(module, grad_inp, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()
        
        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Forward through the feature extractor only
            feat = self.raw_model.image_model._extract_view_features(view_tensor)
            
            # We need to propagate gradients from the final logits back to the features.
            # To do this properly, we run the full image_model forward on a single view
            # by reshaping to (B, 1, 3, H, W).
            single_view = view_tensor.unsqueeze(1)  # (B, 1, 3, H, W)
            logits = self.raw_model.image_model(single_view)  # (B, 10)
            
            self.model.zero_grad()
            logits[0, target_class].backward(retain_graph=False)
            
            # Compute Grad-CAM
            if gradients is None or activations is None:
                return None
            
            # Global average pool gradients: (C_feat,)
            weights = gradients[0].mean(dim=(1, 2))
            cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations[0]).sum(dim=0)
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.cpu().numpy()
        finally:
            h1.remove()
            h2.remove()
    
    def generate(
        self,
        views: torch.Tensor,
        target_class: int = None,
    ) -> list:
        """
        Generate Grad-CAM heatmaps for each of the 6 views.
        
        Args:
            views: (1, 6, 3, H, W) tensor for a single item
            target_class: which defect class to visualize (None = most confident)
        
        Returns:
            list of 6 numpy arrays (H_cam, W_cam), values in [0, 1]
        """
        views = views.to(next(self.model.parameters()).device)
        B, V, C, H, W = views.shape
        
        # First, get predictions to determine target class
        with torch.no_grad():
            logits = self.raw_model.image_model(views)
            probs = torch.sigmoid(logits)
        
        if target_class is None:
            target_class = probs[0].argmax().item()
        
        if self.sequential_views:
            # ── Sequential mode: process each view individually ─────────────
            # This avoids the hook-overwrite problem where sequential processing
            # only captures the last view's activations/gradients.
            cam_list = []
            for v in range(V):
                view_batch = views[:, v]  # (B, 3, H, W)
                cam = self._generate_per_view(view_batch, target_class, views.device)
                if cam is not None:
                    cam_list.append(cam)
                else:
                    # Fallback: blank CAM
                    cam_list.append(np.zeros((H // 8, W // 8), dtype=np.float32))
            return cam_list
        
        else:
            # ── Parallel mode: all views at once (original fast path) ───────
            gradients = [None]
            activations = [None]
            
            def forward_hook(module, inp, out):
                activations[0] = out.detach()
            
            def backward_hook(module, grad_inp, grad_out):
                gradients[0] = grad_out[0].detach()
            
            h1 = self.target_layer.register_forward_hook(forward_hook)
            h2 = self.target_layer.register_full_backward_hook(backward_hook)
            self.handles = [h1, h2]
            
            try:
                self.model.zero_grad()
                logits = self.raw_model.image_model(views)
                logits[0, target_class].backward(retain_graph=False)
                
                if gradients[0] is None or activations[0] is None:
                    return [np.zeros((H // 8, W // 8), dtype=np.float32)] * V
                
                grad_shape = gradients[0].shape
                act_shape = activations[0].shape
                
                # (B*V, C_feat, h, w) -> (B, V, C_feat, h, w)
                grads = gradients[0].reshape(B, V, grad_shape[1], grad_shape[2], grad_shape[3])
                acts = activations[0].reshape(B, V, act_shape[1], act_shape[2], act_shape[3])
                
                cam_list = []
                for v in range(V):
                    weights = grads[0, v].mean(dim=(1, 2))
                    cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts[0, v]).sum(dim=0)
                    cam = F.relu(cam)
                    cam = cam - cam.min()
                    cam = cam / (cam.max() + 1e-8)
                    cam_list.append(cam.cpu().numpy())
                
                return cam_list
            finally:
                self.remove_hooks()
    
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIEW ATTENTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def get_view_attention(model, views: torch.Tensor) -> np.ndarray:
    """
    Get the learned attention weights for each of the 6 views.
    
    Returns:
        (6,) array of attention weights (sum to 1)
    """
    model.eval()
    from models import AgenticEnsembleModel
    raw_model = model.base_model if isinstance(model, AgenticEnsembleModel) else model
    with torch.no_grad():
        attn = raw_model.image_model.get_attention_weights(views)
    return attn.cpu().numpy().flatten()


# ═══════════════════════════════════════════════════════════════════════════
# 3. MESH FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_mesh_feature_importance(model, mesh_features: np.ndarray) -> dict:
    """
    Analyze which mesh features are most important for each defect prediction
    using a simple perturbation-based sensitivity analysis.
    
    For each feature, perturb it by ±1 std and measure change in output.
    """
    # v2.0 FIX: Try both import styles for package/CLI compatibility
    try:
        from mesh_features import FEATURE_ORDER
    except ImportError:
        from .mesh_features import FEATURE_ORDER
    
    model.eval()
    device = next(model.parameters()).device
    
    # Resolve mesh_model from wrapper if wrapped
    mesh_model = getattr(model, "mesh_model", None)
    if mesh_model is None and hasattr(model, "base_model"):
        mesh_model = getattr(model.base_model, "mesh_model", None)
        
    if mesh_model is None:
        raise AttributeError("No mesh_model found in architecture.")
        
    feat_tensor = torch.tensor(mesh_features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        base_logits = mesh_model(feat_tensor)
        base_probs = torch.sigmoid(base_logits)
    
    # Compute per-feature std for perturbation
    feat_std = np.std(mesh_features, axis=0, keepdims=True) + 1e-8
    
    # Build full 100D feature list (P1-26 Fix: SHTD, Betti, QEM, Physics features)
    full_feature_names = list(FEATURE_ORDER)
    if mesh_features.shape[1] >= 100:
        for l_idx in range(5):
            for m_idx in range(-l_idx, l_idx + 1):
                full_feature_names.append(f"shtd_L{l_idx}_m{m_idx}")
        full_feature_names.extend(["betti_0", "betti_1", "betti_2"])
        full_feature_names.append("qem_decimation_stability")
        full_feature_names.extend(["phys_center_mass_z", "phys_inertia_ratio", "phys_tipping_stability"])

    num_feats = min(mesh_features.shape[1], len(full_feature_names))
    importance = {}
    for i in range(num_feats):
        fname = full_feature_names[i]
        # Perturb feature i upward
        feat_up = feat_tensor.clone()
        feat_up[:, i] += 2 * float(feat_std[0, i])
        
        with torch.no_grad():
            probs_up = torch.sigmoid(mesh_model(feat_up))
        
        # Average absolute change across all samples and classes
        delta = (probs_up - base_probs).abs().mean().item()
        importance[fname] = delta
    
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    return importance


# ═══════════════════════════════════════════════════════════════════════════
# 4. COMBINED VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def create_defect_visualization(
    item_id: str,
    image_path: str,
    model,
    device: str,
    defect_cols: list,
    mesh_features: np.ndarray = None,
    predictions: np.ndarray = None,
    true_labels: np.ndarray = None,
    save_path: str = None,
    image_size: int = 224,
):
    """
    Create a comprehensive visualization for a single item showing:
    - 6 views with Grad-CAM overlays
    - View attention bar chart
    - Prediction vs ground truth comparison
    - Top contributing mesh features (if available)
    
    Produces a publication-quality figure.
    """
    if not HAS_MATPLOTLIB:
        print("  [WARNING] matplotlib is not installed — skipping combined visualization figure creation")
        return None

    from image_processing import split_six_views
    from torchvision.transforms.functional import resize, to_tensor, normalize
    
    # ── Load image ─────────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    views = split_six_views(image)
    
    # Prepare tensor: (1, 6, 3, H, W)
    view_tensors = []
    for v in views:
        v = resize(v, [image_size, image_size])
        v_t = to_tensor(v)
        v_t = normalize(v_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        view_tensors.append(v_t)
    views_tensor = torch.stack(view_tensors, dim=0).unsqueeze(0).to(device)
    
    # ── Get predictions ────────────────────────────────────────────────────
    model.eval()
    from models import AgenticEnsembleModel
    raw_model = model.base_model if isinstance(model, AgenticEnsembleModel) else model
    with torch.no_grad():
        logits = raw_model.image_model(views_tensor)
        probs = torch.sigmoid(logits)
    if predictions is None:
        predictions = probs.cpu().numpy().flatten()
    
    # ── Grad-CAM ───────────────────────────────────────────────────────────
    grad_cam = MultiViewGradCAM(model)
    
    # Determine most confident defect
    pred_class = int(np.argmax(predictions))
    cam_maps = grad_cam.generate(views_tensor, target_class=pred_class)
    grad_cam.remove_hooks()
    
    # ── View attention ─────────────────────────────────────────────────────
    attn_weights = get_view_attention(model, views_tensor)
    
    # ── Create figure ──────────────────────────────────────────────────────
    view_names = ["View 1\n(0°)", "View 2\n(90°)", "View 3\n(180°)",
                  "View 4\n(270°)", "Top\nView", "Bottom\nView"]
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Defect Analysis: {item_id[:12]}...\n"
        f"Top Prediction: {defect_cols[pred_class]} (p={predictions[pred_class]:.3f})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    
    # ── Row 1: views with Grad-CAM ──────────────────────────────────────
    # v2.1.1 FIX: Use actual number of views/cams instead of hardcoded 6
    n_views_show = min(len(views), len(cam_maps))
    for i in range(n_views_show):
        ax = fig.add_subplot(3, max(n_views_show, 6), i + 1)
        
        # Show original view
        ax.imshow(views[i].resize((image_size, image_size)))
        
        # Overlay Grad-CAM
        _resample = getattr(Image, "Resampling", None)
        if _resample is not None:
            _resample = _resample.BILINEAR
        else:
            _resample = Image.BILINEAR
        cam_resized = np.array(Image.fromarray((cam_maps[i] * 255).astype(np.uint8)).resize(
            (image_size, image_size), _resample
        )) / 255.0
        ax.imshow(cam_resized, cmap="jet", alpha=0.4)
        
        ax.set_title(f"{view_names[i]}\nattn: {attn_weights[i]:.3f}", fontsize=9)
        ax.axis("off")
    
    # ── Row 2: Attention bar + Per-class predictions ───────────────────────
    # Attention weights bar chart
    ax_attn = fig.add_subplot(3, 6, 7)
    colors = plt.cm.RdYlGn(attn_weights)
    ax_attn.barh(range(6), attn_weights, color=colors)
    ax_attn.set_yticks(range(6))
    ax_attn.set_yticklabels([f"V{i+1}" for i in range(6)], fontsize=8)
    ax_attn.set_title("View Attention", fontsize=10, fontweight="bold")
    ax_attn.set_xlim(0, max(attn_weights) * 1.3 + 0.01)
    
    # Per-class prediction bars
    ax_pred = fig.add_subplot(3, 6, (8, 12))
    y_pos = np.arange(len(defect_cols))
    bars = ax_pred.barh(y_pos, predictions, color=["#e74c3c" if p > 0.5 else "#3498db" for p in predictions])
    
    if true_labels is not None:
        for i, (p, t) in enumerate(zip(predictions, true_labels)):
            marker = "●" if (p > 0.5) == (t > 0.5) else "✗"
            color = "green" if (p > 0.5) == (t > 0.5) else "red"
            ax_pred.text(p + 0.01, i, marker, color=color, fontsize=10, va="center")
    
    ax_pred.set_yticks(y_pos)
    ax_pred.set_yticklabels(defect_cols, fontsize=9)
    ax_pred.set_xlim(0, 1.1)
    ax_pred.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax_pred.set_title("Class Probabilities", fontsize=10, fontweight="bold")
    
    # ── Row 3: Detailed info ───────────────────────────────────────────────
    # Predictions table
    ax_table = fig.add_subplot(3, 6, (13, 18))
    ax_table.axis("off")
    
    table_data = []
    for i, col in enumerate(defect_cols):
        pred_label = "YES" if predictions[i] > 0.5 else "no"
        true_str = "YES" if true_labels is not None and true_labels[i] > 0.5 else "no"
        row = [col, f"{predictions[i]:.3f}", pred_label, true_str if true_labels is not None else "?"]
        table_data.append(row)
    
    col_labels = ["Defect", "Probability", "Predicted", "Ground Truth"]
    table = ax_table.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.25, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color cells
    for i, row_data in enumerate(table_data):
        prob = float(row_data[1])
        for j in range(4):
            cell = table[i + 1, j]
            if j == 2:  # Predicted column
                if row_data[2] == "YES":
                    cell.set_facecolor("#ffcccc")
                else:
                    cell.set_facecolor("#ccffcc")
            elif j == 3 and true_labels is not None:  # Ground truth column
                if row_data[3] == "YES":
                    cell.set_facecolor("#ffcccc")
                else:
                    cell.set_facecolor("#ccffcc")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved visualization to {save_path}")
    
    plt.close()
    return fig


def create_batch_visualizations(
    item_ids: list,
    image_dir: str,
    model,
    device: str,
    defect_cols: list,
    predictions: np.ndarray,
    true_labels: np.ndarray = None,
    output_dir: str = "visualizations",
    n_samples: int = 20,
    mesh_features: np.ndarray = None,
):
    """
    Create visualizations for a batch of items, selecting:
    - Samples with the most confident predictions
    - Failure cases (wrong predictions)
    - Good predictions
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Select diverse samples
    n = min(n_samples, len(item_ids))
    
    # Sort by max probability (most confident)
    max_probs = predictions.max(axis=1)
    confident_idx = np.argsort(-max_probs)[:n]
    
    for rank, idx in enumerate(confident_idx):
        item_id = item_ids[idx]
        safe_item = _sanitize_item_id(item_id)
        img_path = os.path.join(image_dir, f"{safe_item}.png")
        
        if not os.path.exists(img_path):
            continue
        
        save_path = os.path.join(output_dir, f"{rank:02d}_{safe_item[:8]}.png")
        
        mf = mesh_features[idx:idx+1] if mesh_features is not None else None
        tl = true_labels[idx] if true_labels is not None else None
        
        try:
            create_defect_visualization(
                item_id=item_id,
                image_path=img_path,
                model=model,
                device=device,
                defect_cols=defect_cols,
                mesh_features=mf,
                predictions=predictions[idx],
                true_labels=tl,
                save_path=save_path,
            )
        except Exception as e:
            print(f"  Failed to visualize {item_id}: {e}")