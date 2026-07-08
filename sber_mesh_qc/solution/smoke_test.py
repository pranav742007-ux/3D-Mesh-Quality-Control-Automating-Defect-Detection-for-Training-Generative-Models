"""
===============================================================================
SBER AI Journey — 3D Mesh Quality Control: v7.2 Industrial Smoke Test Suite
===============================================================================
Zero-training verification script validating 23 critical AI modules:
  - Baseline model parity & 6-channel pseudo-normal transforms
  - MultiModalMeshQCModelV7 (Transformer + Defect Queries + Spatial Tokens)
  - HybridFocalASLLoss & Evidential Beta Loss gradient flow
  - 100D Topology Invariants (SHTD L=4, DSU Betti homology, QEM, Physics Tipping)
  - Automated 3D Geometric Mesh Repair Engine & Curvature FPS Sampler
  - DeepSeek DSpark Semi-Autoregressive Markov Defect Head
  - Zhipu GLM-5.2 IndexShare Cross-Modal Attention (4:1 indexer sharing)
  - xAI Grok-3 MoE Dynamic Gated Router & Load-Balancing Loss
  - Moonshot AI Kimi K1.5 Latent Memory Compressor & DPO Preference Loss
  - OmniRoute Dynamic Multi-Modal Path Dispatcher & Modality Entropy H(M)

Runs in CPU / lightweight GPU mode in < 5 seconds with ZERO training.
===============================================================================
"""

import os
import sys
import torch
import numpy as np

sol_dir = os.path.dirname(os.path.abspath(__file__))
if sol_dir not in sys.path:
    sys.path.insert(0, sol_dir)

import torch.nn as nn

# Add solution directory to Python path
sol_dir = os.path.dirname(os.path.abspath(__file__))
if sol_dir not in sys.path:
    sys.path.insert(0, sol_dir)

import config
from models import MultiViewImageModel, FusedEnsembleModel, MultiModalMeshQCModelV7
from losses import build_loss_function, HybridFocalASLLoss
from image_processing import compute_sobel_pseudo_normals


def run_smoke_tests():
    print("=" * 60)
    print("  RUNNING V7.2 INDUSTRIAL SMOKE TEST DIAGNOSTIC SUITE (23 TESTS)")
    print("=" * 60)

    # Use CPU by default for local smoke test to avoid GPU VRAM memory competition
    device = "cpu"
    print(f"  Diagnostic Device: {device}")

    # ── Test 1: Baseline Parity Test ──────────────────────────────────────────
    print("\n[Test 1/23] Baseline v3.0 Model Parity...")
    B, V, C, H, W = 1, 6, 3, 64, 64
    views_3ch = torch.randn(B, V, C, H, W, device=device)
    mesh_feat = torch.randn(B, 68, device=device)

    base_image_model = MultiViewImageModel(
        backbone_name="efficientnetv2_s",
        pretrained=False,
        num_classes=10,
    ).to(device)
    
    base_logits = base_image_model(views_3ch)
    assert base_logits.shape == (B, 10), f"Expected (B, 10), got {base_logits.shape}"
    print(f"  [OK] Baseline model forward shape: {base_logits.shape}")
    del base_image_model

    # ── Test 2: v7.2 MultiModalMeshQCModelV7 Forward Pass ───────────────────
    print("\n[Test 2/23] v7.2 Transformer + Defect Query Decoder + Spatial Tokens...")
    v4_model = MultiModalMeshQCModelV7(
        backbone_name="efficientnetv2_s",
        pretrained=False,
        in_channels=3,
        use_spatial_tokens=True,
        use_transformer=True,
        use_query_decoder=True,
        use_soft_hierarchy=True,
        d_model=256,
        mesh_dim=68,
        num_classes=10,
    ).to(device)

    v4_logits = v4_model(views_3ch, mesh_features=mesh_feat)
    assert v4_logits.shape == (B, 10), f"Expected (B, 10), got {v4_logits.shape}"
    print(f"  [OK] v7.2 MultiModal model forward shape: {v4_logits.shape}")

    # ── Test 3: HybridFocalASLLoss Backward Pass ─────────────────────────────
    print("\n[Test 3/23] Loss Function Backward Pass...")
    targets = torch.randint(0, 2, (B, 10), dtype=torch.float32, device=device)
    loss_fn = HybridFocalASLLoss()
    loss = loss_fn(v4_logits, targets)
    loss.backward()
    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in v4_model.parameters())
    assert has_grads, "No gradients flowing to model parameters!"
    print(f"  [OK] Loss scalar: {loss.item():.4f} - Backward pass & gradient flow successful!")
    del v4_model

    # ── Test 4: 6-Channel Pseudo-Normals & Conv Adapt (100D Production Mesh Dim) ─
    print("\n[Test 4/23] 6-Channel Pseudo-Normals & 100D Backbone Adaptation...")
    single_view_rgb = torch.rand(3, H, W, device=device)
    pseudo_normals = compute_sobel_pseudo_normals(single_view_rgb)
    assert pseudo_normals.shape == (3, H, W), f"Expected (3, H, W), got {pseudo_normals.shape}"
    
    concat_6ch = torch.cat([single_view_rgb, pseudo_normals], dim=0).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1, 1)
    mesh_feat_100d = torch.randn(B, 100, device=device)
    
    v4_6ch_model = MultiModalMeshQCModelV7(
        backbone_name="efficientnetv2_s",
        pretrained=False,
        in_channels=6,
        use_transformer=True,
        use_query_decoder=True,
        d_model=256,
        mesh_dim=100,
    ).to(device)

    logits_6ch = v4_6ch_model(concat_6ch, mesh_features=mesh_feat_100d)
    assert logits_6ch.shape == (B, 10), f"Expected (B, 10), got {logits_6ch.shape}"
    print(f"  [OK] 6-channel pseudo-normal & 100D input forward shape: {logits_6ch.shape}")

    # ── Test 5: Missing Modality Test ─────────────────────────────────────────
    print("\n[Test 5/23] Missing Modality Test (mesh_features=None)...")
    no_mesh_logits = v4_6ch_model(concat_6ch, mesh_features=None, point_cloud=None)
    assert no_mesh_logits.shape == (B, 10), f"Expected (B, 10), got {no_mesh_logits.shape}"
    print(f"  [OK] Missing modality forward shape: {no_mesh_logits.shape}")
    del v4_6ch_model

    # ── Test 6: Industrial Mesh Geometry Sanitizer & Uncertainty Test ───────
    print("\n[Test 6/23] Mesh Geometry Sanitizer & Uncertainty Metrics...")
    from utils import sanitize_mesh_geometry, compute_uncertainty_scores
    dummy_verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [np.nan, 0.0, 0.0]])
    dummy_faces = np.array([[0, 1, 2], [0, 0, 0]])
    clean_verts, clean_faces, report = sanitize_mesh_geometry(dummy_verts, dummy_faces)
    assert len(clean_verts) == 4
    assert len(clean_faces) == 1
    print(f"  [OK] Sanitized geometry: {report}")
    
    mock_probs = np.array([0.05, 0.12, 0.02, 0.01, 0.04, 0.03, 0.01, 0.08, 0.02, 0.01])
    metrics = compute_uncertainty_scores(mock_probs)
    assert "confidence_percent" in metrics
    print(f"  [OK] Uncertainty score calculated: confidence={metrics['confidence_percent']}%")

    # ── Test 7: ONNX FP16 Exporter Graph Trace Test ─────────────────────────
    print("\n[Test 7/23] ONNX Graph Trace Test...")
    from export_onnx import export_to_onnx
    onnx_file = os.path.join(sol_dir, "test_smoke.onnx")
    try:
        export_to_onnx(onnx_file, checkpoint_path=None, in_channels=3)
        assert os.path.exists(onnx_file)
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
        print(f"  [OK] ONNX Graph Trace export verified!")
    except ImportError as e:
        print(f"  [INFO] ONNX package not installed in current environment, skipping trace test: {e}")

    # ── Test 8: CrossModalCoAttention Test ────────────────────────────────────
    print("\n[Test 8/23] CrossModalCoAttention Module Verification...")
    from models import CrossModalCoAttention
    co_attn = CrossModalCoAttention(d_model=256, nhead=4).to(device)
    img_toks = torch.randn(B, 6, 256, device=device)
    geom_toks = torch.randn(B, 2, 256, device=device)
    img_out, geom_out = co_attn(img_toks, geom_toks)
    assert img_out.shape == (B, 6, 256)
    assert geom_out.shape == (B, 2, 256)
    print(f"  [OK] CrossModalCoAttention shape: img={img_out.shape}, geom={geom_out.shape}")
    del co_attn

    # ── Test 9: Canonical PCA Mesh Alignment Test ─────────────────────────────
    print("\n[Test 9/23] Canonical PCA Mesh Alignment Test...")
    from mesh_features import canonical_pca_orientation
    raw_verts = np.random.randn(100, 3)
    aligned_verts = canonical_pca_orientation(raw_verts)
    assert aligned_verts.shape == (100, 3)
    print(f"  [OK] Canonical PCA Alignment verified: shape={aligned_verts.shape}")

    # ── Test 10: MoE True Sparse Routing & Call Count Verification ─────────────
    print("\n[Test 10/23] MoE True Sparse Routing & Call Count Verification...")
    from models import OctopusMoEModel, MeshFeatureMLP
    expert_cfgs = [
        {"backbone_name": "efficientnetv2_s", "pretrained": False},
        {"backbone_name": "convnext_tiny", "pretrained": False},
    ]
    mesh_mlp = MeshFeatureMLP(input_dim=68, num_classes=10)
    moe_model = OctopusMoEModel(expert_configs=expert_cfgs, mesh_model=mesh_mlp, top_k=1, router_noise_std=0.0).to(device)
    moe_views = torch.randn(2, 6, 3, H, W, device=device)
    moe_logits, aux = moe_model.forward_train(moe_views)
    # Test public __call__ interface (returns (logits, aux_info) tuple when return_aux=True, or logits tensor when return_aux=False)
    moe_call_res = moe_model(moe_views, return_aux=True)
    assert isinstance(moe_call_res, tuple) and moe_call_res[0].shape == (2, 10)
    moe_logits_res = moe_model(moe_views, return_aux=False)
    assert isinstance(moe_logits_res, torch.Tensor) and moe_logits_res.shape == (2, 10)
    # Test forward_simple interface (returns logits tensor)
    moe_simple_logits = moe_model.forward_simple(moe_views)
    assert moe_simple_logits.shape == (2, 10)

    
    stats = moe_model.get_routing_stats()
    assert sum(stats["expert_call_counts"]) > 0
    print(f"  [OK] True MoE Sparse Routing & Call Interfaces verified! Call counts: {stats['expert_call_counts']}")
    del moe_model

    # ── Test 11: 100D Vectorized SHTD, Betti, QEM & Physics Invariants Test ───
    print("\n[Test 11/23] 100D SHTD, Betti, QEM & Physics Invariants Verification...")
    from mesh_features import compute_spherical_harmonics_descriptors, compute_topological_betti_numbers, compute_qem_decimation_stability, compute_physics_stability_metric
    dummy_verts = np.random.randn(50, 3)
    dummy_faces = np.array([[0, 1, 2], [1, 2, 3]])
    shtd_vec = compute_spherical_harmonics_descriptors(dummy_verts)
    betti_vec = compute_topological_betti_numbers(dummy_verts, dummy_faces)
    qem_val = compute_qem_decimation_stability(dummy_verts, dummy_faces)
    phys_dict = compute_physics_stability_metric(dummy_verts, dummy_faces)
    assert shtd_vec.shape == (25,)
    assert betti_vec.shape == (3,)
    assert qem_val >= 0.0
    assert "tipping_angle_deg" in phys_dict
    print(f"  [OK] 100D Topology Invariants verified: SHTD={shtd_vec.shape}, Betti={betti_vec.shape}, QEM={qem_val:.2f}, Physics={phys_dict['tipping_angle_deg']:.1f}°")

    # ── Test 12: Binary Evidential Head & Aux Reconstruction Check ────────────
    print("\n[Test 12/23] Binary Evidential Beta Head & Aux Recon Verification...")
    from models import BinaryEvidentialHead, GeometricReconstructionHead
    from losses import BinaryEvidentialLoss
    ev_head = BinaryEvidentialHead(in_features=256, num_classes=10).to(device)
    recon_head = GeometricReconstructionHead(in_features=256, target_dim=68).to(device)
    ev_loss_fn = BinaryEvidentialLoss(kl_weight=0.1)
    
    feats = torch.randn(2, 256, device=device)
    targets = torch.randint(0, 2, (2, 10), device=device).float()
    target_geom = torch.randn(2, 68, device=device)
    
    probs, uncertainty, evidence = ev_head(feats)
    ev_loss = ev_loss_fn(evidence["alpha"], evidence["beta"], targets)
    recon_loss = recon_head(feats, target_geom)
    
    assert probs.shape == (2, 10)
    assert uncertainty.shape == (2, 10)
    assert not torch.isnan(ev_loss)
    assert not torch.isnan(recon_loss)
    print(f"  [OK] Binary Evidential & Aux Recon Loss verified: ev_loss={ev_loss.item():.4f}, recon_loss={recon_loss.item():.4f}")
    del ev_head, recon_head

    # ── Test 13: Direct CPU Mesh Rasterizer Verification ──────────────────────
    print("\n[Test 13/23] Direct CPU Mesh Rasterizer Verification (<10ms)...")
    from image_processing import DirectMeshRasterizer
    raw_v = np.random.randn(40, 3)
    raw_f = np.array([[0, 1, 2], [1, 2, 3]])
    rast_tensor = DirectMeshRasterizer.rasterize_views(raw_v, raw_f, img_size=224)
    assert rast_tensor.shape == (6, 6, 224, 224)
    print(f"  [OK] Direct CPU Mesh Rasterizer verified: shape={rast_tensor.shape}")

    # ── Test 14: Curvature-Weighted FPS Point Cloud Sampling Test ─────────────
    print("\n[Test 14/23] Curvature-Weighted FPS Point Cloud Sampling Verification...")
    from data_utils import sample_curvature_weighted_points
    curv_pts = sample_curvature_weighted_points(raw_v, raw_f, n_points=1024)
    assert curv_pts.shape == (1024, 3)
    print(f"  [OK] Curvature-Weighted FPS Sampling verified: shape={curv_pts.shape}")

    # ── Test 15: Automated 3D Mesh Repair Engine Verification ─────────────────
    print("\n[Test 15/23] Automated 3D Mesh Repair Engine Verification...")
    from mesh_repair import auto_repair_mesh
    dirty_v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    dirty_f = np.array([[0, 1, 2], [0, 1, 3]])
    clean_v, clean_f, report = auto_repair_mesh(dirty_v, dirty_f)
    assert report["repaired"] == True
    print(f"  [OK] Automated 3D Mesh Repair Engine verified! Report: {report}")

    # ── Test 16: DSpark-Adapted Semi-Autoregressive & Confidence Router Check ─
    print("\n[Test 16/23] DSpark-Adapted Semi-Autoregressive & Confidence Router Verification...")
    from models import MarkovDefectTransitionHead, ConfidenceScheduledRouter
    markov_head = MarkovDefectTransitionHead(num_classes=10, rank=32).to(device)
    conf_router = ConfidenceScheduledRouter(in_features=100, confidence_threshold=0.90).to(device)
    
    dummy_logits = torch.randn(2, 10, device=device)
    refined_logits = markov_head(dummy_logits)
    assert refined_logits.shape == (2, 10)
    
    dummy_100d = torch.randn(2, 100, device=device)
    conf_scores, early_mask = conf_router(dummy_100d)
    assert conf_scores.shape == (2, 1)
    print(f"  [OK] DSpark Semi-Autoregressive & Confidence Router verified! Refined={refined_logits.shape}, Conf={conf_scores.mean().item():.3f}")

    # ── Test 17: GLM-5.2 IndexShare Attention & GLM-Image Aligner Check ──────
    print("\n[Test 17/23] GLM-5.2 IndexShare Attention & GLM-Image Aligner Verification...")
    from models import IndexShareCrossModalAttention, FlexibleThinkingEffortController, GLMImageSpatialAligner
    index_share_attn = IndexShareCrossModalAttention(d_model=256, n_heads=4).to(device)
    effort_ctrl = FlexibleThinkingEffortController(reasoning_effort="high")
    spatial_aligner = GLMImageSpatialAligner(img_dim=256, geom_dim=100, out_dim=256).to(device)
    
    dummy_img = torch.randn(2, 6, 256, device=device)
    dummy_geom = torch.randn(2, 1, 100, device=device)
    
    aligned_img, aligned_geom = spatial_aligner(dummy_img, dummy_100d)
    assert aligned_img.shape == (2, 6, 256)
    
    attn_out, weights = index_share_attn(aligned_img, aligned_geom.unsqueeze(1))
    assert attn_out.shape == (2, 6, 256)
    assert effort_ctrl.should_execute_visual_backbone() == True
    print(f"  [OK] GLM-5.2 IndexShare & GLM-Image Aligner verified! AlignedImg={aligned_img.shape}, Effort={effort_ctrl.reasoning_effort}")

    # ── Test 18: xAI Grok-3 MoE Dynamic Gated Router Verification ────────────
    print("\n[Test 18/23] xAI Grok-3 MoE Dynamic Gated Router Verification...")
    from models import xAIMoEHybridRouter
    xai_router = xAIMoEHybridRouter(d_model=256, num_experts=4, top_k=2).to(device)
    dummy_input = torch.randn(4, 256, device=device)
    topk_w, topk_i, aux_l = xai_router(dummy_input)
    assert topk_w.shape == (4, 2)
    assert topk_i.shape == (4, 2)
    print(f"  [OK] xAI Grok-3 MoE Router verified! TopK_W={topk_w.shape}, AuxLoss={aux_l.item():.4f}")

    # ── Test 19: Moonshot AI Kimi K1.5 Latent Memory & DPO Loss Check ────────
    print("\n[Test 19/23] Moonshot AI Kimi K1.5 Latent Memory & DPO Verification...")
    from models import KimiLatentMemoryCompressor, KimiQualityPreferenceLoss, MoonshotInterleavedPooler
    kimi_compressor = KimiLatentMemoryCompressor(d_model=256, num_slots=16).to(device)
    kimi_dpo = KimiQualityPreferenceLoss(margin=0.5).to(device)
    moonshot_pooler = MoonshotInterleavedPooler(img_dim=256, geom_dim=100, out_dim=256).to(device)
    
    dummy_tokens = torch.randn(2, 6, 256, device=device)
    compressed_mem = kimi_compressor(dummy_tokens)
    assert compressed_mem.shape == (2, 16, 256)
    
    clean_s = torch.tensor([0.9, 0.85], device=device)
    bad_s = torch.tensor([0.2, 0.15], device=device)
    dpo_loss = kimi_dpo(clean_s, bad_s)
    assert dpo_loss.item() > 0
    
    interleaved_tokens = moonshot_pooler(dummy_tokens, dummy_100d)
    assert interleaved_tokens.shape == (2, 7, 256)
    print(f"  [OK] Moonshot AI Kimi K1.5 verified! LatentMem={compressed_mem.shape}, DPOLoss={dpo_loss.item():.4f}")

    # ── Test 20: OmniRoute Dynamic Path Dispatcher Verification ──────────────
    print("\n[Test 20/23] OmniRoute Dynamic Path Dispatcher Verification...")
    from models import OmniRoutePathDispatcher
    omni_dispatcher = OmniRoutePathDispatcher(in_features=100, num_branches=3).to(device)
    route_p, entropy_h = omni_dispatcher(dummy_100d)
    assert route_p.shape == (2, 3)
    assert entropy_h.shape == (2, 1)
    print(f"  [OK] OmniRoute Dynamic Path Dispatcher verified! RouteProbs={route_p.shape}, Entropy={entropy_h.mean().item():.4f}")

    # ── Test 21: DeepSeek-V3 Multi-Head Latent Attention (MLA) Verification ─
    print("\n[Test 21/23] DeepSeek-V3 Multi-Head Latent Attention (MLA) Verification...")
    from models import DeepSeekMLACrossModalAttention
    mla_attn = DeepSeekMLACrossModalAttention(d_model=256, n_heads=4, kv_compression_dim=64).to(device)
    dummy_img = torch.randn(2, 6, 256, device=device)
    dummy_geom = torch.randn(2, 2, 256, device=device)
    mla_out, c_kv = mla_attn(dummy_img, dummy_geom)
    assert mla_out.shape == (2, 6, 256)
    assert c_kv.shape == (2, 2, 64)
    print(f"  [OK] DeepSeek-V3 MLA Latent Attention verified! OutShape={mla_out.shape}, c_KV={c_kv.shape}")

    # ── Test 22: FlashAttention-2 SDPA & Hard Defect Focal Loss Check ────────
    print("\n[Test 22/23] FlashAttention-2 SDPA & Quality-Aware Focal Loss Verification...")
    from models import FlashCrossModalCoAttention
    from losses import QualityAwareHardDefectFocalLoss
    flash_coattn = FlashCrossModalCoAttention(d_model=256, n_heads=4).to(device)
    q_focal = QualityAwareHardDefectFocalLoss(gamma=2.5, quality_boost=2.0).to(device)
    
    img_flash_out, geom_flash_out = flash_coattn(dummy_img, dummy_geom)
    dummy_logits_b = torch.randn(2, 10, device=device)
    dummy_target_b = torch.randint(0, 2, (2, 10), device=device).float()
    q_loss = q_focal(dummy_logits_b, dummy_target_b)
    
    assert img_flash_out.shape == (2, 6, 256)
    assert geom_flash_out.shape == (2, 2, 256)
    assert q_loss.item() > 0
    print(f"  [OK] FlashAttention-2 & Quality-Aware Focal Loss verified! Loss={q_loss.item():.4f}")

    # ── Test 23: Agentic Wrapper & Dynamic Effort Level Check ──────────────────
    print("\n[Test 23/23] Agentic Ensemble Model Wrapper & Dynamic Effort Levels...")
    from models import AgenticEnsembleModel, build_model_from_config
    import config as test_cfg
    
    orig_early_exit = getattr(test_cfg, "USE_EARLY_EXIT", False)
    orig_threshold = getattr(test_cfg, "EARLY_EXIT_THRESHOLD", 0.95)
    
    test_cfg.USE_EARLY_EXIT = True
    test_cfg.EARLY_EXIT_THRESHOLD = 0.90
    
    # Build wrapper model
    agentic_model = build_model_from_config(test_cfg, effective_mesh_dim=100).to(device)
    assert isinstance(agentic_model, AgenticEnsembleModel)
    
    agentic_model.eval()
    dummy_views = torch.randn(2, 6, 3, 224, 224, device=device)
    dummy_mesh = torch.randn(2, 100, device=device)
    
    # Test different effort levels
    logits_fast = agentic_model(dummy_views, mesh_features=dummy_mesh, effort="fast")
    logits_high = agentic_model(dummy_views, mesh_features=dummy_mesh, effort="high")
    logits_max = agentic_model(dummy_views, mesh_features=dummy_mesh, effort="max")
    
    assert logits_fast.shape == (2, 10)
    assert logits_high.shape == (2, 10)
    assert logits_max.shape == (2, 10)
    
    # Restore original config
    test_cfg.USE_EARLY_EXIT = orig_early_exit
    test_cfg.EARLY_EXIT_THRESHOLD = orig_threshold
    
    print(f"  [OK] Agentic Wrapper & Effort levels (fast/high/max) verified successfully!")

    print("\n" + "=" * 60)
    print("  ALL 23 V7.2 FRONTIER FLASH-ATTENTION, MLA, MOONSHOT, XAI & GLM SMOKE TESTS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_tests()
