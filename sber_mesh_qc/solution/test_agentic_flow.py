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
        # Temporarily enable early exit for build
        cfg.USE_EARLY_EXIT = True
        cfg.EARLY_EXIT_THRESHOLD = 0.85
        cfg.USE_MOE = False
        
        # Build model with wrapper active
        self.model = build_model_from_config(cfg, effective_mesh_dim=100)
        self.assertTrue(isinstance(self.model, AgenticEnsembleModel))

    def test_wrapper_structure(self):
        self.assertIsNotNone(self.model.base_model)
        self.assertIsNotNone(self.model.confidence_router)
        self.assertIsNotNone(self.model.effort_controller)

    def test_early_exit_all_confident(self):
        self.model.confidence_router.confidence_threshold = 0.1
        
        views = torch.randn(2, 6, 3, 224, 224)
        mesh_features = torch.randn(2, 100)
        
        self.model.eval()
        logits = self.model(views, mesh_features=mesh_features)
        
        self.assertEqual(logits.shape, (2, 10))

    def test_early_exit_mixed_batch(self):
        def mock_forward(geom_features):
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
        views = torch.randn(1, 6, 3, 224, 224)
        mesh_features = torch.randn(1, 100)
        
        self.model.eval()
        self.model.confidence_router.confidence_threshold = 2.0
        
        logits = self.model(views, mesh_features=mesh_features, effort="fast")
        self.assertEqual(logits.shape, (1, 10))

    def test_effort_controller_high_mode(self):
        views = torch.randn(1, 6, 3, 224, 224)
        mesh_features = torch.randn(1, 100)
        
        self.model.eval()
        self.model.confidence_router.confidence_threshold = 2.0
        
        logits = self.model(views, mesh_features=mesh_features, effort="high")
        self.assertEqual(logits.shape, (1, 10))

if __name__ == "__main__":
    unittest.main()
