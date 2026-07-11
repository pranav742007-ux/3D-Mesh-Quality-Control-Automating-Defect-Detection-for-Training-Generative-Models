"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Knowledge Distillation  [v7.3]
===============================================================================
Trains a SOTA single student model (ConvNeXt-Tiny) to mimic the ensembled 
soft-probability outputs of the heterogeneous cross-fold ensemble.
===============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import config as cfg
from utils import set_seed, derive_quality, safe_collate, clean_state_dict_keys
from image_processing import MeshQualityDataset
from models import build_model_from_config
from losses import CleanMeshShieldLoss

def distill_student(train_df: pd.DataFrame, image_dir: str, mesh_features: np.ndarray, 
                    checkpoint_dir: str, log_dir: str):
    print("\n" + "="*60)
    print("  PHASE: KNOWLEDGE DISTILLATION (SELF-DISTILLATION)")
    print("="*60)
    
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    
    # 1. Load ensembled soft targets from CV training run
    soft_targets_path = os.path.join(log_dir, "train_soft_targets.npy")
    if not os.path.exists(soft_targets_path):
        print(f"  [WARNING] Soft targets file not found at: {soft_targets_path}")
        print("            Please run full training first to save ensembled soft targets.")
        return
        
    soft_targets = np.load(soft_targets_path) # Shape: (N, 10)
    print(f"  Loaded ensembled soft targets of shape: {soft_targets.shape}")
    
    # 2. Build Student Model (ConvNeXt-Tiny)
    # Cache and override config temporarily for student initialization
    original_backbone = cfg.IMAGE_BACKBONE
    cfg.IMAGE_BACKBONE = "convnext_tiny"
    student = build_model_from_config(cfg=cfg, effective_mesh_dim=100).to(device)
    cfg.IMAGE_BACKBONE = original_backbone  # Restore config
    
    # 3. Distillation parameters
    T = 2.0  # Softening temperature
    shield_loss_fn = CleanMeshShieldLoss().to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
    
    train_dataset = MeshQualityDataset(
        item_ids=train_df["item_id"].tolist(),
        labels_df=train_df,
        image_dir=image_dir,
        mesh_features=mesh_features,
        image_size=cfg.IMAGE_SIZE,
        augment=True,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,  # Shuffle False to align with train_soft_targets.npy index
        num_workers=getattr(cfg, "NUM_WORKERS", 4), collate_fn=safe_collate, drop_last=True
    )
    
    student.train()
    print("  Starting distillation student training (5 epochs)...")
    for epoch in range(5):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            idx_start = batch_idx * cfg.BATCH_SIZE
            views = batch["views"].to(device)
            mesh_feat = batch["mesh_features"].to(device) if batch["mesh_features"] is not None else None
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = student(views, mesh_feat)
            
            # Extract corresponding batch soft targets (probabilities) and map to logit space
            # Use views.size(0) for robustness against variable batch sizes
            batch_soft_proba = torch.tensor(soft_targets[idx_start:idx_start+views.size(0)], device=device)
            # Clip to prevent log(0)
            batch_soft_proba = torch.clamp(batch_soft_proba, 1e-7, 1.0 - 1e-7)
            batch_soft_logits = torch.log(batch_soft_proba / (1.0 - batch_soft_proba))
            
            # Independent Binary Cross-Entropy with Logits for multi-label distillation
            # (Fixes the multiclass softmax/KL bug since classes are not mutually exclusive)
            loss_kd = F.binary_cross_entropy_with_logits(
                logits / T, 
                torch.sigmoid(batch_soft_logits / T), 
                reduction="mean"
            ) * (T * T)
            
            # Clean mesh shield loss
            loss_shield = shield_loss_fn(logits, labels)
            
            loss = loss_kd + loss_shield
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"    Epoch {epoch+1}/5 | Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Save distilled student checkpoint
    student_path = os.path.join(checkpoint_dir, "distilled_student.pt")
    torch.save({
        "model_state_dict": student.state_dict(),
        "scaler_mean": None,  # Scaler will be applied in dataset pipeline
        "scaler_std": None,
    }, student_path)
    print(f"  [OK] Distilled student saved to: {student_path}")
