"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Early Exit Threshold Tuning
===============================================================================
Sweeps and validates the ConfidenceScheduledRouter early-exit thresholds
against the validation split to optimize accuracy vs inference speed.
===============================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

import config as cfg
from utils import set_seed, derive_quality, compute_f1_final, safe_collate
from image_processing import MeshQualityDataset
from models import build_model_from_config, ConfidenceScheduledRouter

def tune_early_exit_threshold(
    model, 
    val_dataset, 
    val_labels_df, 
    device=cfg.DEVICE, 
    thresholds=np.linspace(0.80, 0.99, 20)
):
    print("  Evaluating base model and geometry confidence scores on validation set...")
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=getattr(cfg, "NUM_WORKERS", 4), pin_memory=cfg.PIN_MEMORY,
        collate_fn=safe_collate
    )
    
    # 1. Collect all predictions, geometry features, and true labels
    geom_feats = []
    full_preds = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            views = batch["views"].to(device)
            mesh_feat = batch["mesh_features"].to(device)
            labels = batch["labels"].to(device)
            
            try:
                logits = model(views, mesh_feat, effort="max")
            except TypeError:
                logits = model(views, mesh_feat)
                
            if isinstance(logits, tuple):
                logits = logits[0]
                
            probs = torch.sigmoid(logits)
            
            full_preds.append(probs.cpu().numpy())
            geom_feats.append(mesh_feat.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            
    geom_feats = np.concatenate(geom_feats, axis=0)
    full_preds = np.concatenate(full_preds, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    
    # 2. Extract mesh-only predictions
    mesh_model = getattr(model, "mesh_model", None)
    if mesh_model is None and hasattr(model, "base_model"):
        mesh_model = getattr(model.base_model, "mesh_model", None)
        
    if mesh_model is not None:
        mesh_model.eval()
        with torch.no_grad():
            mesh_logits = mesh_model(torch.tensor(geom_feats, dtype=torch.float32).to(device))
            mesh_probs = torch.sigmoid(mesh_logits).cpu().numpy()
    else:
        print("  [WARNING] No separate mesh_model found in architecture, falling back to full_preds.")
        mesh_probs = full_preds
        
    # 3. Instantiate router to get confidence scores
    router = ConfidenceScheduledRouter(in_features=geom_feats.shape[1])
    router.to(device)
    router.eval()
    with torch.no_grad():
        conf_scores, _ = router(torch.tensor(geom_feats, dtype=torch.float32).to(device))
        conf_scores = conf_scores.cpu().numpy().flatten()
        
    # 4. Sweep thresholds and compute F1_final
    best_threshold = 0.95
    best_f1 = -1
    best_stats = {}
    
    defect_cols = [
        "abstract", "artifacts", "intersection", "lowpoly",
        "noisy", "open", "partial", "scale", "set", "simple"
    ]
    
    true_df = pd.DataFrame(true_labels, columns=defect_cols)
    true_df["quality"] = derive_quality(true_labels)
    
    for thresh in thresholds:
        early_exit_mask = (conf_scores >= thresh)
        preds = np.where(early_exit_mask[:, None], mesh_probs, full_preds)
        preds_binary = (preds >= 0.5).astype(int)
        
        pred_df = pd.DataFrame(preds_binary, columns=defect_cols)
        pred_df["quality"] = derive_quality(preds_binary)
        
        metrics = compute_f1_final(true_df, pred_df)
        f1_final = metrics["f1_final"]
        exit_pct = float(np.mean(early_exit_mask) * 100)
        
        print(f"  Threshold: {thresh:.3f} | F1_final: {f1_final:.2f} | Early Exited: {exit_pct:.1f}%")
        
        if f1_final > best_f1:
            best_f1 = f1_final
            best_threshold = thresh
            best_stats = {
                "f1_final": f1_final,
                "early_exit_pct": exit_pct,
                "f1_defects": metrics["f1_defects"],
                "f1_quality": metrics["f1_quality"]
            }
            
    print(f"\n  [OPTIMIZATION COMPLETE]")
    print(f"    Best Threshold: {best_threshold:.3f}")
    print(f"    F1_final:       {best_stats['f1_final']:.2f}")
    print(f"    Early Exits:    {best_stats['early_exit_pct']:.1f}%")
    return best_threshold, best_stats

if __name__ == "__main__":
    set_seed(cfg.SEED)
    ckpt_dir = "checkpoints"
    best_model_path = os.path.join(ckpt_dir, "best_fold0.pt")
    if os.path.exists(best_model_path):
        print(f"Loading checkpoint {best_model_path} for tuning...")
        model = build_model_from_config(cfg, effective_mesh_dim=100).to(cfg.DEVICE)
        ckpt = torch.load(best_model_path, map_location=cfg.DEVICE, weights_only=True)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        
        dummy_ids = [f"dummy_{i}" for i in range(50)]
        dummy_labels = pd.DataFrame(np.random.randint(0, 2, size=(50, 10)), columns=[
            "abstract", "artifacts", "intersection", "lowpoly",
            "noisy", "open", "partial", "scale", "set", "simple"
        ])
        dummy_labels.insert(0, "item_id", dummy_ids)
        dummy_labels["quality"] = derive_quality(dummy_labels.iloc[:, 1:].values)
        
        dummy_dataset = MeshQualityDataset(
            item_ids=dummy_ids,
            labels_df=dummy_labels,
            image_dir="data/images",
            mesh_features=np.random.randn(50, 100),
            point_clouds=None,
            image_size=224,
            view_grid=(2, 3),
            augment=False
        )
        
        tune_early_exit_threshold(model, dummy_dataset, dummy_labels)
    else:
        print("No checkpoint found at checkpoints/best_fold0.pt. Running diagnostic model checks instead.")
        cfg.USE_EARLY_EXIT = True
        model = build_model_from_config(cfg, effective_mesh_dim=100)
        print("  [OK] Model built successfully with USE_EARLY_EXIT=True wrapper.")
