"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Loss Functions  [v7.1 Master Engine]
===============================================================================
Loss Functions for Extreme Multi-Label Class Imbalance & Epistemic Uncertainty:
  - AsymmetricLoss (ASL) with positive/negative focusing gammas
  - FocalLoss with dynamic class weighting & label smoothing
  - HybridFocalASLLoss (0.6 ASL + 0.4 Focal)
  - AuxiliaryGeometryLoss (2D-to-3D cosine reconstruction loss)
  - BinaryEvidentialLoss (Beta Dirichlet NLL + KL divergence regularization)
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = None) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = torch.where(
                targets > 0.5,
                torch.ones_like(targets),
                torch.full_like(targets, self.label_smoothing)
            )

        probs = torch.sigmoid(logits)
        
        # Asymmetric Probability Shift for Negative Targets (Ridnik et al., 2021)
        if self.clip > 0:
            p_sub = (probs - self.clip).clamp(min=0.0)
        else:
            p_sub = probs
        
        # Loss for positive examples using numerically stable F.logsigmoid
        pos_loss = targets * F.logsigmoid(logits)
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1.0 - probs) ** self.gamma_pos)
        
        # Loss for negative examples with asymmetric probability shift
        neg_loss = (1.0 - targets) * torch.log((1.0 - p_sub).clamp(min=1e-7))
        if self.gamma_neg > 0:
            neg_loss = neg_loss * (p_sub ** self.gamma_neg)
        
        loss = -(pos_loss + neg_loss)
        
        red = reduction if reduction is not None else self.reduction
        if red == "mean":
            return loss.mean()
        elif red == "sum":
            return loss.sum()
        return loss



class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = None) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = torch.where(
                targets > 0.5,
                torch.ones_like(targets),
                torch.full_like(targets, self.label_smoothing)
            )
        
        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce
        
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)
        
        red = reduction if reduction is not None else self.reduction
        if red == "mean":
            return loss.mean()
        elif red == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = None) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = torch.where(
                targets > 0.5,
                torch.ones_like(targets),
                torch.full_like(targets, self.label_smoothing)
            )
        
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        
        red = reduction if reduction is not None else self.reduction
        if red == "mean":
            return loss.mean()
        elif red == "sum":
            return loss.sum()
        return loss


class HybridFocalASLLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, label_smoothing=label_smoothing, reduction="none")
        self.focal = FocalLoss(gamma=2.0, label_smoothing=label_smoothing, reduction="none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = None) -> torch.Tensor:
        loss_asl = self.asl(logits, targets, reduction="none")
        loss_focal = self.focal(logits, targets, reduction="none")
        loss = 0.6 * loss_asl + 0.4 * loss_focal
        
        red = reduction if reduction is not None else self.reduction
        if red == "mean":
            return loss.mean()
        elif red == "sum":
            return loss.sum()
        return loss


def build_loss_function(
    loss_name: str,
    class_weights: np.ndarray = None,
    label_smoothing: float = 0.05,
    focal_gamma: float = 2.0,
    reduction: str = "mean",
) -> nn.Module:
    alpha = None
    if class_weights is not None:
        alpha = torch.tensor(class_weights, dtype=torch.float32)
        alpha = alpha / alpha.mean()
    
    if loss_name == "bce":
        return WeightedBCELoss(alpha, label_smoothing, reduction)
    
    elif loss_name == "bce_focal":
        return FocalLoss(gamma=focal_gamma, alpha=alpha, label_smoothing=label_smoothing, reduction=reduction)
    
    elif loss_name == "asl":
        return AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05, label_smoothing=label_smoothing, reduction=reduction)
    
    elif loss_name == "hybrid_asl":
        return HybridFocalASLLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05, label_smoothing=label_smoothing, reduction=reduction)
    
    elif loss_name == "quality_focal":
        try:
            import config as cfg
            focal_weight_max = getattr(cfg, "FOCAL_WEIGHT_MAX", 5.0)
        except Exception:
            focal_weight_max = 5.0
        return QualityAwareHardDefectFocalLoss(
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            reduction=reduction,
            focal_weight_max=focal_weight_max,
        )
    
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


class AuxiliaryGeometryLoss(nn.Module):
    """
    Auxiliary Geometry Loss for Multi-Modal Learning.
    Penalizes MSE between predicted 3D mesh geometry proxies (e.g. area, volume)
    and ground-truth mesh feature targets.
    Forces the 2D visual encoder to learn 3D topological structure.
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()

    def forward(self, pred_geom: torch.Tensor, target_geom: torch.Tensor) -> torch.Tensor:
        if pred_geom is None or target_geom is None:
            return torch.tensor(0.0)
        return self.weight * self.mse(pred_geom, target_geom)


class BinaryEvidentialLoss(nn.Module):
    """
    10 Parallel Binary Beta-Evidential Loss (v6.0).
    Computes Negative Log-Marginal Likelihood + Beta KL Divergence Regularization.
    """
    def __init__(self, kl_weight: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.label_smoothing = label_smoothing

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / 2.0
            
        S = alpha + beta
        loss_nll = targets * (torch.digamma(S) - torch.digamma(alpha)) + \
                   (1.0 - targets) * (torch.digamma(S) - torch.digamma(beta))
        
        alpha_tilde = targets + (1.0 - targets) * alpha
        beta_tilde = (1.0 - targets) + targets * beta
        S_tilde = alpha_tilde + beta_tilde
        
        kl = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde) - \
             (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)) - \
             (beta_tilde - 1.0) * (torch.digamma(beta_tilde) - torch.digamma(S_tilde))
             
        total_loss = loss_nll + self.kl_weight * kl
        return total_loss.mean()


class QualityAwareHardDefectFocalLoss(nn.Module):
    """
    Quality-Aware Hard Defect Focal Loss (v7.2 Frontier).
    Dynamically reweights loss gradients by focusing extra penalty on false negative predictions
    that directly ruin the competition quality label (quality = 1 iff ALL 10 defects = 0).
    """
    def __init__(
        self,
        gamma: float = 2.5,
        quality_boost: float = 2.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
        focal_weight_max: float = 5.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.quality_boost = quality_boost
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.focal_weight_max = focal_weight_max

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = None) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = torch.where(
                targets > 0.5,
                torch.ones_like(targets),
                torch.full_like(targets, self.label_smoothing)
            )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_weight = torch.clamp(focal_weight, max=self.focal_weight_max)

        # Hard Defect Mining Weight: Boost weight if sample has defect (target > 0) but model missed it (p < 0.5)
        fn_mask = (targets > 0.5) & (probs < 0.5)
        quality_multiplier = torch.where(fn_mask, self.quality_boost, 1.0)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * quality_multiplier * bce
        
        red = reduction if reduction is not None else self.reduction
        if red == "mean":
            return loss.mean()
        elif red == "sum":
            return loss.sum()
        return loss


class CleanMeshShieldLoss(nn.Module):
    """
    Exponentially penalizes any predicted defect probability > 0.0 
    when the ground truth is a perfectly clean mesh (all defects = 0).
    Prevents the catastrophic quality F1 drop from single False Positives.
    """
    def __init__(self, shield_weight: float = 8.0, sharpness: float = 12.0):
        super().__init__()
        self.shield_weight = shield_weight
        self.sharpness = sharpness

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Identify perfectly clean samples in the batch
        clean_mask = (targets.sum(dim=1) == 0)
        if not clean_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        # For clean samples, push ALL probabilities toward 0
        clean_logits = logits[clean_mask]
        prob = torch.sigmoid(clean_logits)
        # Steep penalty above 0.0 using exp or power function
        clean_loss = torch.mean(self.shield_weight * (prob ** self.sharpness))
        return clean_loss
