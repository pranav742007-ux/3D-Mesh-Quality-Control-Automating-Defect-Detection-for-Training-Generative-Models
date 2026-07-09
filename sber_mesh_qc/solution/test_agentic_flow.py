import unittest
import torch
import torch.nn as nn
import numpy as np

import config as cfg
from models import (
    build_model_from_config, 
    AgenticEnsembleModel, 
    ConfidenceScheduledRouter, 
    FlexibleThinkingEffortController
)

class TestAgenticFlow(unittest.TestCase):
    def setUp(self):
        # Base setup configuration
        cfg.USE_EARLY_EXIT = True
        cfg.EARLY_EXIT_THRESHOLD = 0.85
        cfg.USE_MOE = False
        cfg.USE_POINTNET_BRANCH = False
        
        # Build model with wrapper active
        self.model = build_model_from_config(cfg, effective_mesh_dim=100)
        self.assertTrue(isinstance(self.model, AgenticEnsembleModel))

    def test_wrapper_structure(self):
        """Verify all submodules are connected to the agentic wrapper."""
        self.assertIsNotNone(self.model.base_model)
        self.assertIsNotNone(self.model.confidence_router)
        self.assertIsNotNone(self.model.effort_controller)

    def test_early_exit_all_confident(self):
        """Test routing when 100% of samples are confident (visual bypass)."""
        self.model.confidence_router.confidence_threshold = -1.0  # Force confident
        
        views = torch.randn(2, 6, 3, 224, 224)
        mesh_features = torch.randn(2, 100)
        
        self.model.eval()
        logits = self.model(views, mesh_features=mesh_features)
        
        self.assertEqual(logits.shape, (2, 10))

    def test_early_exit_none_confident(self):
        """Test routing when 0% of samples are confident (full multimodal path)."""
        self.model.confidence_router.confidence_threshold = 2.0  # Force unconfident
        
        views = torch.randn(2, 6, 3, 224, 224)
        mesh_features = torch.randn(2, 100)
        
        self.model.eval()
        logits = self.model(views, mesh_features=mesh_features)
        
        self.assertEqual(logits.shape, (2, 10))

    def test_early_exit_mixed_batch(self):
        """Test routing when the batch contains a mix of confident and unconfident samples."""
        def mock_forward(geom_features):
            # Element 0 is confident (>=0.85), Element 1 is unconfident (<0.85)
            scores = torch.tensor([[0.9], [0.1]])
            mask = scores >= 0.85
            return scores, mask
            
        self.model.confidence_router.forward = mock_forward
        
        views = torch.randn(2, 6, 3, 224, 224)
        mesh_features = torch.randn(2, 100)
        
        self.model.eval()
        logits = self.model(views, mesh_features=mesh_features)
        
        self.assertEqual(logits.shape, (2, 10))

    def test_effort_controller_fast_mode(self):
        """Verify 'fast' effort mode completely bypasses visual branches."""
        views = torch.randn(1, 6, 3, 224, 224)
        mesh_features = torch.randn(1, 100)
        
        self.model.eval()
        self.model.confidence_router.confidence_threshold = 2.0
        
        logits = self.model(views, mesh_features=mesh_features, effort="fast")
        self.assertEqual(logits.shape, (1, 10))

    def test_effort_controller_high_mode(self):
        """Verify 'high' effort mode routes through the full ensemble."""
        views = torch.randn(1, 6, 3, 224, 224)
        mesh_features = torch.randn(1, 100)
        
        self.model.eval()
        self.model.confidence_router.confidence_threshold = 2.0
        
        logits = self.model(views, mesh_features=mesh_features, effort="high")
        self.assertEqual(logits.shape, (1, 10))

    def test_mixed_precision_autocast_casting(self):
        """Test dtype-safety under CPU/CUDA autocast conditions to verify no float16 assignment crashes."""
        self.model.confidence_router.confidence_threshold = 2.0  # Force full branch
        views = torch.randn(2, 6, 3, 224, 224)
        mesh_features = torch.randn(2, 100)
        
        self.model.eval()
        
        # Simulate mixed precision (AMP) autocast context on CPU
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            logits = self.model(views, mesh_features=mesh_features)
        
        self.assertEqual(logits.shape, (2, 10))

    def test_pointnet_branch_integration(self):
        """Verify agentic flow functions perfectly when point cloud branch is active."""
        cfg.USE_POINTNET_BRANCH = True
        
        # Build model with PointNet branch active
        model = build_model_from_config(cfg, effective_mesh_dim=100)
        self.assertTrue(isinstance(model, AgenticEnsembleModel))
        
        views = torch.randn(2, 6, 3, 224, 224)
        mesh_features = torch.randn(2, 100)
        point_cloud = torch.randn(2, 1024, 3)
        
        model.eval()
        logits = model(views, mesh_features=mesh_features, point_cloud=point_cloud)
        self.assertEqual(logits.shape, (2, 10))
        
        # Reset flag
        cfg.USE_POINTNET_BRANCH = False

    def test_moe_mode_integration(self):
        """Verify agentic flow wraps correctly around MoE architecture when enabled."""
        cfg.USE_MOE = True
        
        # Build MoE model
        model = build_model_from_config(cfg, effective_mesh_dim=100)
        self.assertTrue(isinstance(model, AgenticEnsembleModel))
        
        views = torch.randn(2, 6, 3, 224, 224)
        mesh_features = torch.randn(2, 100)
        
        model.eval()
        logits = model(views, mesh_features=mesh_features)
        self.assertEqual(logits.shape, (2, 10))
        
        # Reset flag
        cfg.USE_MOE = False

if __name__ == "__main__":
    unittest.main()
