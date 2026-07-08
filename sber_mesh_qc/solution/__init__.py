"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control  [v7.2 Master Engine]
===============================================================================
Multi-label classification of 10 defect types on 3D meshes using 6-view PNG
renders + 100D topology vectors + raw point clouds with Mixture-of-Experts.

v7.2: OmniRoute Modality Entropy Dispatcher, Moonshot Kimi K1.5 DPO Loss,
      xAI Grok-3 MoE Gated Router, GLM-5.2 IndexShare 4:1 Attention.

Modules:
    config           — Centralized configuration & hyperparameters
    utils            — Seeding, metrics, threshold optimization, temperature scaling
    mesh_features    — 100D geometric & topological feature vector from .npz
    image_processing — 6-view splitting, Direct CPU rasterizer, Sobel pseudo-normals
    models           — Octopus MoE, MultiModalMeshQCModelV7, CrossModalCoAttention
    losses           — HybridFocalASLLoss, Beta Dirichlet Evidential Loss, Cosine Aux Loss
    train            — Stratified K-Fold CV training pipeline with EMA weight smoothing
    inference        — Ensemble inference, TTA, calibrated submission generation
    data_utils       — Data download, extraction, validation, aria2c parallel downloader
    visualization    — Grad-CAM, attention weights, 100D feature importance
    pointnet_lite    — 3D Point cloud branch, barycentric surface sampling
    mesh_repair      — Ear-clipping hole repair & degenerate face purging engine
    export_onnx      — ONNX FP16 opset 17 exporter & ONNX Runtime verifier
    app              — Industrial FastAPI HTTP REST microservice
===============================================================================
"""

__version__ = "7.2.0"
__competition__ = "SBER AI Journey — 3D Mesh Quality Control"
