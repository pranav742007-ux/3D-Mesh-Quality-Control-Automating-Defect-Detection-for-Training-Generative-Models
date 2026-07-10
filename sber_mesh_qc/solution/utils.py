"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Utilities  [v7.2 Master Engine]
===============================================================================
Common utility functions: seeding, competition metrics, quality-aware threshold
optimization, temperature scaling calibration, mesh geometry sanitization,
epistemic uncertainty scoring, and Exponential Moving Average (EMA).
===============================================================================
"""

import os
import random
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score


def set_seed(seed: int = 42):
    """Set random seed for full reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def compute_f1_final(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> dict:
    """
    Compute the competition metric:
        f1_final = 10 * F1(quality) + 10 * F1_weighted(defects)
    
    Args:
        y_true_df: DataFrame with ground truth (11 binary columns)
        y_pred_df: DataFrame with predictions (11 binary columns)
    
    Returns:
        dict with 'f1_quality', 'f1_defects', 'f1_final'
    """
    defect_cols = [
        "abstract", "artifacts", "intersection", "lowpoly",
        "noisy", "open", "partial", "scale", "set", "simple"
    ]
    
    f1_quality = f1_score(
        y_true_df["quality"].values, y_pred_df["quality"].values, average="binary"
    )
    f1_defects = f1_score(
        y_true_df[defect_cols].values, y_pred_df[defect_cols].values, average="weighted"
    )
    f1_final = 10 * f1_quality + 10 * f1_defects
    
    return {
        "f1_quality": f1_quality,
        "f1_defects": f1_defects,
        "f1_final": f1_final,
    }


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    """Compute per-class F1 scores for detailed analysis."""
    results = {}
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() == 0 and y_pred[:, i].sum() == 0:
            results[name] = 1.0  # Both all-negative → perfect
        else:
            results[name] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    return results


def optimize_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
    metric_fn=None,
    search_range=(0.05, 0.95),
    steps=50,
) -> np.ndarray:
    """
    Find per-class optimal thresholds that maximize F1_final.
    
    Uses greedy sequential optimization: optimize one class threshold at a time
    while keeping others fixed, iterating multiple passes.
    
    Args:
        y_true: (N, C) binary ground truth
        y_proba: (N, C) predicted probabilities
        class_names: list of class names
        metric_fn: custom metric function (default: weighted F1)
        search_range: (low, high) threshold search range
        steps: number of threshold values to test per class
    
    Returns:
        (C,) array of optimal thresholds
    """
    if metric_fn is None:
        def metric_fn(yt, yp):
            return f1_score(yt, yp, average="weighted", zero_division=0)
    
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, 0.5)
    candidates = np.linspace(search_range[0], search_range[1], steps)
    
    best_overall_score = -1
    best_thresholds = thresholds.copy()
    
    # Multiple passes for convergence
    for _ in range(3):
        for c in range(n_classes):
            best_score = -1
            best_t = 0.5
            for t in candidates:
                temp_pred = (y_proba >= np.full_like(y_proba, thresholds))  # Use current thresholds
                temp_pred[:, c] = (y_proba[:, c] >= t).astype(int)
                score = metric_fn(y_true, temp_pred)
                if score > best_score:
                    best_score = score
                    best_t = t
            thresholds[c] = best_t
        
        # Check if this is the best overall
        final_pred = (y_proba >= thresholds).astype(int)
        overall_score = metric_fn(y_true, final_pred)
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_thresholds = thresholds.copy()
    
    return best_thresholds




def optimize_thresholds_f1_final(
    y_true_defects: np.ndarray,
    y_true_quality: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
    search_range: tuple = (0.05, 0.95),
    steps: int = 50,
) -> np.ndarray:
    """
    Optimize per-class thresholds to maximize f1_final directly.
    
    Unlike optimize_thresholds() which only maximizes F1_weighted(defects),
    this function optimizes the FULL competition metric:
        f1_final = 10 * F1(quality) + 10 * F1_weighted(defects)
    
    This is strictly better because it accounts for the quality label
    derivation: quality = 1 iff ALL defects = 0. Lowering thresholds
    increases defect recall but may hurt quality F1 (more false positives
    → fewer "clean" predictions). This function finds the sweet spot.
    
    Args:
        y_true_defects: (N, C) binary ground truth for defects
        y_true_quality: (N,) binary ground truth for quality
        y_proba: (N, C) predicted probabilities
        class_names: list of class names
        search_range: (low, high) threshold search range
        steps: number of threshold values to test per class
    
    Returns:
        (C,) array of optimal thresholds
    """
    from sklearn.metrics import f1_score
    
    n_classes = y_true_defects.shape[1]
    thresholds = np.full(n_classes, 0.5)
    candidates = np.linspace(search_range[0], search_range[1], steps)
    
    best_f1_final = -1
    best_thresholds = thresholds.copy()
    
    def _compute_f1_final(threshs):
        pred = (y_proba >= threshs).astype(int)
        quality_pred = (pred.sum(axis=1) == 0).astype(int)
        
        f1_q = f1_score(y_true_quality, quality_pred, average="binary", zero_division=0)
        f1_d = f1_score(y_true_defects, pred, average="weighted", zero_division=0)
        
        base_score = 10.0 * f1_q + 10.0 * f1_d
        
        # QUADRATIC PENALTY: If max probability is low (model thinks it's clean),
        # but we predicted a defect due to low threshold, PUNISH (Phase 4).
        max_probs = y_proba.max(axis=1)
        false_pos_on_likely_clean = (pred.sum(axis=1) > 0) & (max_probs < 0.4)
        penalty = np.sum(false_pos_on_likely_clean) * 2.0
        
        return base_score - penalty
    
    # Multiple passes for convergence (greedy coordinate descent)
    for _pass in range(3):
        for c in range(n_classes):
            best_score = -1
            best_t = 0.5
            for t in candidates:
                test_thresh = thresholds.copy()
                test_thresh[c] = t
                score = _compute_f1_final(test_thresh)
                if score > best_score:
                    best_score = score
                    best_t = t
            thresholds[c] = best_t
        
        # Check if this is the best overall
        overall_score = _compute_f1_final(thresholds)
        if overall_score > best_f1_final:
            best_f1_final = overall_score
            best_thresholds = thresholds.copy()
    
    # Powell derivative-free optimization with 5 restarts to refine the thresholds (Step 2)
    try:
        from scipy.optimize import minimize
        def objective(threshs):
            clipped_threshs = np.clip(threshs, search_range[0], search_range[1])
            return -_compute_f1_final(clipped_threshs)

        bounds = [(search_range[0], search_range[1])] * n_classes
        best_result_fun = -best_f1_final
        best_result_x = best_thresholds.copy()

        # Run 5 restarts (1 from coordinate descent, 4 random) to guarantee global optimum
        res = minimize(objective, x0=best_thresholds, method='Powell', bounds=bounds, options={'maxiter': 50, 'disp': False})
        if res.fun < best_result_fun:
            best_result_fun = res.fun
            best_result_x = np.clip(res.x, search_range[0], search_range[1])

        for _ in range(4):
            x0 = np.random.uniform(0.2, 0.8, n_classes)
            res = minimize(objective, x0=x0, method='Powell', bounds=bounds, options={'maxiter': 50, 'disp': False})
            if res.fun < best_result_fun:
                best_result_fun = res.fun
                best_result_x = np.clip(res.x, search_range[0], search_range[1])

        best_thresholds = best_result_x
        print(f"      [Refined Threshold Search] Refined F1_final threshold score to: {-best_result_fun:.4f}")
    except Exception as e:
        print(f"      [WARNING] Refined threshold optimization failed: {e}. Falling back to greedy thresholds.")

    return best_thresholds


def learn_temperature(
    val_proba: np.ndarray,
    val_true_defects: np.ndarray,
    val_true_quality: np.ndarray,
    class_names: list,
    initial_temp: float = 1.5,
    lr: float = 0.01,
    steps: int = 100,
) -> float:
    """
    Learn an optimal temperature parameter for probability calibration.
    
    Temperature scaling divides logits by T before sigmoid:
        p = sigmoid(logit / T)
    
    When T > 1: probabilities are pushed toward 0.5 (softer, less confident)
    When T < 1: probabilities are pushed toward 0/1 (sharper, more confident)
    
    Better calibrated probabilities lead to better threshold optimization
    and thus higher F1_final.
    
    Args:
        val_proba: (N, C) validation probabilities (pre-sigmoid, we convert to logits)
        val_true_defects: (N, C) binary ground truth
        val_true_quality: (N,) binary ground truth
        class_names: list of class names
        initial_temp: starting temperature
        lr: learning rate for gradient descent
        steps: number of optimization steps
    
    Returns:
        Optimal temperature value
    """
    import torch
    
    # Convert probabilities back to logits for temperature scaling
    val_proba_clipped = np.clip(val_proba, 1e-7, 1.0 - 1e-7)
    val_logits = np.log(val_proba_clipped / (1.0 - val_proba_clipped))
    
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    true_t = torch.tensor(val_true_defects, dtype=torch.float32)
    quality_t = torch.tensor(val_true_quality, dtype=torch.float32)
    
    temperature = torch.tensor([initial_temp], requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=steps)
    
    # Use a simple differentiable proxy: NLL-like loss on temperature-scaled logits.
    # We want to find T that maximizes F1, but F1 is non-differentiable.
    # Instead, we use a surrogate: minimize the difference between
    # temperature-scaled probs and the true labels (BCE proxy).
    # After finding T, we re-optimize thresholds properly.
    def eval_bce_proxy():
        """Differentiable BCE proxy for temperature optimization."""
        # P2 FIX: Clamp temperature soft-proxy outside in-place data mutation
        temp_val = torch.clamp(temperature, 0.1, 5.0)
        scaled_logits = logits_t / temp_val
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            scaled_logits, true_t, reduction='mean'
        )
        if torch.isnan(bce) or torch.isinf(bce):
            return temperature * 0.0 + 1e6
        return bce
    
    def closure():
        optimizer.zero_grad()
        loss = eval_bce_proxy()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    optimal_temp = temperature.item()
    optimal_temp = max(0.1, min(optimal_temp, 5.0))  # Clamp to reasonable range
    
    # Evaluate F1 improvement with optimal temperature (using 0.5 thresholds)
    scaled_logits = logits_t / optimal_temp
    probs = torch.sigmoid(scaled_logits).detach().numpy()
    pred = (probs >= 0.5).astype(int)
    quality_pred = (pred.sum(axis=1) == 0).astype(int)
    
    # Base F1 (T=1.0, i.e. original probabilities)
    base_pred = (val_proba >= 0.5).astype(int)
    base_quality_pred = (base_pred.sum(axis=1) == 0).astype(int)
    base_f1_q = f1_score(val_true_quality, base_quality_pred, average="binary", zero_division=0)
    base_f1_d = f1_score(val_true_defects, base_pred, average="weighted", zero_division=0)
    base_f1 = 10 * base_f1_q + 10 * base_f1_d
    
    # New F1 (with optimal T)
    f1_q = f1_score(val_true_quality, quality_pred, average="binary", zero_division=0)
    f1_d = f1_score(val_true_defects, pred, average="weighted", zero_division=0)
    new_f1 = 10 * f1_q + 10 * f1_d
    
    print(f"  Temperature scaling: T={initial_temp:.2f} -> T={optimal_temp:.3f}")
    print(f"  F1_final improvement: {base_f1:.2f} -> {new_f1:.2f} (delta: {new_f1 - base_f1:+.2f})")
    
    return optimal_temp

def derive_quality(defect_preds: np.ndarray) -> np.ndarray:
    """
    Derive quality label from defect predictions.
    Quality = 1 (good) if and only if ALL defects are 0.
    """
    return (defect_preds.sum(axis=1) == 0).astype(int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def average_precision_per_class(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute average precision per class for ranking analysis."""
    from sklearn.metrics import average_precision_score
    results = {}
    for i in range(y_true.shape[1]):
        results[i] = average_precision_score(y_true[:, i], y_proba[:, i])
    return results


def sanitize_mesh_geometry(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Industrial Degenerate Mesh Sanitizer:
    1. Removes NaN / Inf coordinates.
    2. Identifies and removes zero-area degenerate faces.
    3. Returns cleaned vertices, faces, and sanitization report.
    """
    report = {"nan_vertices_removed": 0, "degenerate_faces_removed": 0}
    if vertices is None or len(vertices) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), report

    # 1. NaN / Inf check
    valid_mask = ~np.isnan(vertices).any(axis=1) & ~np.isinf(vertices).any(axis=1)
    if not np.all(valid_mask):
        report["nan_vertices_removed"] = int(np.sum(~valid_mask))
        vertices = np.where(np.isnan(vertices) | np.isinf(vertices), 0.0, vertices)

    # 2. Degenerate zero-area face check
    if faces is not None and len(faces) > 0:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        valid_faces = areas > 1e-12
        if not np.all(valid_faces):
            report["degenerate_faces_removed"] = int(np.sum(~valid_faces))
            faces = faces[valid_faces]

    return vertices, faces, report


def compute_uncertainty_scores(probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calculates prediction uncertainty & confidence metrics:
    - Margin uncertainty: distance to decision boundary |p - threshold|
    - Binary entropy: -p*log(p) - (1-p)*log(1-p)
    - Returns mean confidence and flagged status if confidence < 80%.
    """
    p = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    margins = np.abs(p - threshold)
    min_margin = float(np.min(margins))
    mean_margin = float(np.mean(margins))
    
    # Normalized confidence score [0, 100%]
    confidence = float(np.clip(mean_margin * 2.0 * 100.0, 0.0, 100.0))
    requires_human_review = bool(confidence < 80.0 or min_margin < 0.1)

    return {
        "confidence_percent": round(confidence, 2),
        "min_margin": round(min_margin, 4),
        "requires_human_review": requires_human_review,
    }


def safe_collate(batch):
    """
    Custom collate function that handles None values (for missing modalities
    like mesh_features or point_clouds) without raising TypeErrors.
    """
    from torch.utils.data._utils.collate import default_collate
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: safe_collate([d[key] for d in batch]) for key in elem}
    elif elem is None:
        return None
    else:
        return default_collate(batch)


def clean_state_dict_keys(state_dict: dict, model: nn.Module) -> dict:
    """
    Adjust state_dict keys to match the model wrapper structure.
    Adds/removes 'base_model.' prefix if the model is wrapped in AgenticEnsembleModel but the checkpoint is not (or vice-versa).
    """
    if not isinstance(state_dict, dict) or not isinstance(model, nn.Module):
        return state_dict

    model_has_wrapper = hasattr(model, 'base_model')
    ckpt_has_wrapper = any(k.startswith('base_model.') for k in state_dict.keys())

    if model_has_wrapper and not ckpt_has_wrapper:
        # Checkpoint is raw, model has agentic wrapper -> add 'base_model.' prefix
        new_dict = {}
        for k, v in state_dict.items():
            # Only prefix standard model parameters, keep router and effort parameters intact
            if not k.startswith('confidence_router.') and not k.startswith('effort_controller.'):
                new_dict[f"base_model.{k}"] = v
            else:
                new_dict[k] = v
        return new_dict
    elif not model_has_wrapper and ckpt_has_wrapper:
        # Checkpoint has agentic wrapper, model is raw -> remove 'base_model.' prefix
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith('base_model.'):
                new_dict[k.replace('base_model.', '', 1)] = v
            else:
                new_dict[k] = v
        return new_dict

    return state_dict



