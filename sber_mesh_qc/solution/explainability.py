"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: Explainability & OOD [v7.3]
===============================================================================
Phase 8 Explainability & Failure Analysis. Implements OOD Mahalanobis detection,
automated rule explanation generator, and worst-case failure gallery exporter.
===============================================================================
"""
import os
import json
import numpy as np

class OODDetector:
    """
    Out-Of-Distribution (OOD) Detection using Mahalanobis distance on geometry features.
    Computes distance to training feature distribution centroid.
    """
    def __init__(self, train_features: np.ndarray = None):
        self.mean = None
        self.inv_cov = None
        if train_features is not None:
            self.fit(train_features)
            
    def fit(self, X: np.ndarray):
        # Clip inf/nan values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False) + 1e-5 * np.eye(X.shape[1])
        self.inv_cov = np.linalg.inv(cov)
        
    def compute_ood_score(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance as OOD score."""
        if self.mean is None or self.inv_cov is None:
            return 0.0
        x = np.nan_to_num(x, nan=0.0)
        diff = x - self.mean
        dist = np.sqrt(np.dot(np.dot(diff, self.inv_cov), diff.T))
        return float(dist)

    def save_to_dict(self) -> dict:
        """Serialize fitted parameters for json compatibility."""
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "inv_cov": self.inv_cov.tolist() if self.inv_cov is not None else None
        }

    def load_from_dict(self, d: dict):
        """Restore fitted parameters from a dictionary."""
        if d.get("mean") is not None:
            self.mean = np.array(d["mean"])
        if d.get("inv_cov") is not None:
            self.inv_cov = np.array(d["inv_cov"])

class PredictionExplanation:
    """
    prediction explanation generator for 3D Mesh Defect detection.
    Connects model probabilities, geometry features, and rule-based validity flags.
    """
    @staticmethod
    def explain_prediction(
        item_id: str,
        defect_probs: np.ndarray,
        thresholds: np.ndarray,
        validity_report: dict,
        ood_score: float,
        class_names: list
    ) -> dict:
        predicted_defects = []
        contributing_reasons = []
        
        for idx, prob in enumerate(defect_probs):
            if prob >= thresholds[idx]:
                predicted_defects.append(class_names[idx])
                
        # Link validity violations to reasoning
        if validity_report:
            if validity_report.get("boundary_edge_count", 0) > 0:
                contributing_reasons.append(f"Boundary edges found ({validity_report['boundary_edge_count']}): indicates open holes or non-watertight shell.")
            if validity_report.get("non_manifold_edge_count", 0) > 0:
                contributing_reasons.append(f"Non-manifold edges found ({validity_report['non_manifold_edge_count']}): self-intersections or duplicate face connections exist.")
            if validity_report.get("degenerate_face_ratio", 0.0) > 0.01:
                contributing_reasons.append(f"Degenerate face ratio is high ({validity_report['degenerate_face_ratio']:.3f}): contains tiny zero-area triangles.")
            if validity_report.get("connected_components", 1) > 1:
                contributing_reasons.append(f"Mesh has multiple disconnected components ({validity_report['connected_components']}): indicates detached parts or set artifact.")
            if validity_report.get("self_intersections", 0) > 0:
                contributing_reasons.append(f"AABB self-intersection count is {validity_report['self_intersections']}: triangles intersecting inside the model.")
                
        if ood_score > 12.0:
            contributing_reasons.append(f"High Out-Of-Distribution (OOD) score ({ood_score:.2f}): geometry configuration lies far from standard training set distribution.")

        is_clean = len(predicted_defects) == 0
        explanation_summary = (
            "Mesh is clean. No structural defects detected, and all geometry checks passed."
            if is_clean
            else f"Defects detected: {', '.join(predicted_defects)}."
        )

        return {
            "item_id": item_id,
            "prediction": "clean" if is_clean else "defective",
            "defect_probabilities": {name: float(prob) for name, prob in zip(class_names, defect_probs)},
            "predicted_defects": predicted_defects,
            "reasons": contributing_reasons,
            "explanation": explanation_summary,
            "ood_score": ood_score
        }

def save_failure_analysis_gallery(
    val_true: np.ndarray,
    val_proba: np.ndarray,
    item_ids: list,
    thresholds: np.ndarray,
    class_names: list,
    log_dir: str = "logs"
):
    """
    Saves worst false positives and worst false negatives (highest confidence wrong predictions)
    to a Failure Analysis Gallery json file for developers to review.
    """
    os.makedirs(log_dir, exist_ok=True)
    val_pred = (val_proba >= thresholds).astype(int)
    
    worst_false_positives = []
    worst_false_negatives = []
    
    for idx, item_id in enumerate(item_ids):
        true_arr = val_true[idx]
        pred_arr = val_pred[idx]
        prob_arr = val_proba[idx]
        
        for c in range(len(class_names)):
            name = class_names[c]
            prob = prob_arr[c]
            true_val = true_arr[c]
            pred_val = pred_arr[c]
            
            # False Positive
            if pred_val == 1 and true_val == 0:
                worst_false_positives.append({
                    "item_id": item_id,
                    "defect": name,
                    "prob": float(prob),
                    "threshold": float(thresholds[c])
                })
            # False Negative
            elif pred_val == 0 and true_val == 1:
                worst_false_negatives.append({
                    "item_id": item_id,
                    "defect": name,
                    "prob": float(prob),
                    "threshold": float(thresholds[c])
                })
                
    # Sort worst cases by prediction error confidence
    worst_false_positives.sort(key=lambda x: x["prob"], reverse=True)
    worst_false_negatives.sort(key=lambda x: x["prob"], reverse=False) # low probability but true
    
    gallery = {
        "worst_false_positives": worst_false_positives[:50],
        "worst_false_negatives": worst_false_negatives[:50]
    }
    
    out_path = os.path.join(log_dir, "failure_analysis_gallery.json")
    with open(out_path, "w") as f:
        json.dump(gallery, f, indent=2)
    print(f"[Failure Analysis] Gallery with worst {len(gallery['worst_false_positives'])} FP and {len(gallery['worst_false_negatives'])} FN saved to: {out_path}")
