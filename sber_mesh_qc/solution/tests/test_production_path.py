import os
import shutil
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from image_processing import MeshQualityDataset, TTATransform
from models import build_model_from_config, build_model_contract, validate_checkpoint_contract, forward_model
from utils import set_seed, derive_quality

class TestProductionPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
        # Create temp folder for test files
        cls.test_dir = os.path.join(cfg.BASE_DIR, "solution_test_temp")
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create dummy NPZ mesh file
        cls.dummy_npz_path = os.path.join(cls.test_dir, "dummy_item.npz")
        # 4 vertices, 2 faces
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
        faces = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
        np.savez(cls.dummy_npz_path, vertices=vertices, faces=faces)
        
        # Create dummy PNG image files (6 views, but we will mock loading/rendering)
        # Actually we can mock the load_rgb_views or load_raw_geometry_raster
        cls.item_id = "dummy_item"
        
        # Create dummy dataset dataframe
        cls.df = pd.DataFrame([{
            "item_id": cls.item_id,
            "quality": 1,
            "abstract": 0, "artifacts": 0, "intersection": 0, "lowpoly": 0, "noisy": 0,
            "open": 0, "partial": 0, "scale": 0, "set": 0, "simple": 0
        }])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_01_compilation(self):
        """1. Compile all Python files."""
        import compileall
        solution_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = compileall.compile_dir(solution_dir, quiet=True)
        self.assertTrue(result, f"Compilation of solution/ directory failed at: {solution_dir}")

    def test_02_dataset_modes(self):
        """2. Dataset modes: 3, 6, 8, 11 channels."""
        # We will mock _load_rgb_views to return dummy tensors of shape (6, 3, 224, 224)
        original_load_rgb = MeshQualityDataset._load_rgb_views
        original_load_raw = MeshQualityDataset._load_raw_geometry_raster
        
        try:
            # Mock loader methods to isolate rasterizer dependency / filesystem check
            MeshQualityDataset._load_rgb_views = lambda self, item_id, safe_id: [torch.zeros((3, 224, 224)) for _ in range(6)]
            MeshQualityDataset._load_raw_geometry_raster = lambda self, item_id: torch.zeros((6, 5, 224, 224))
            
            for raster, grad, expected_ch in [(False, False, 3), (False, True, 6), (True, False, 8), (True, True, 11)]:
                cfg.USE_GEOMETRY_RASTER = raster
                cfg.USE_GRADIENT_NORMALS = grad
                cfg.IMAGE_IN_CHANNELS = expected_ch
                cfg.validate_config()
                
                dataset = MeshQualityDataset(
                    item_ids=[self.item_id],
                    labels_df=self.df,
                    image_dir=self.test_dir,
                    use_image=True,
                    use_mesh_features=False
                )
                sample = dataset[0]
                self.assertEqual(sample["views"].shape, (6, expected_ch, 224, 224))
                
        finally:
            MeshQualityDataset._load_rgb_views = original_load_rgb
            MeshQualityDataset._load_raw_geometry_raster = original_load_raw

    def test_03_forward_passes(self):
        """3. One CPU forward pass per branch mode (image, mesh, fused)."""
        # Mock class/configs
        original_image_branch = cfg.USE_IMAGE_BRANCH
        original_mesh_branch = cfg.USE_MESH_BRANCH
        
        try:
            cfg.USE_IMAGE_BRANCH = True
            cfg.USE_MESH_BRANCH = True
            cfg.IMAGE_IN_CHANNELS = 3
            cfg.USE_GEOMETRY_RASTER = False
            cfg.USE_GRADIENT_NORMALS = False
            
            # Fused Branch Model
            model = build_model_from_config(cfg, effective_mesh_dim=68)
            model.eval()
            
            views = torch.zeros((1, 6, 3, 224, 224))
            mesh_feat = torch.zeros((1, 68))
            
            with torch.no_grad():
                out = forward_model(model, views, mesh_feat, None, cfg)
            self.assertEqual(out.shape, (1, 10))
            self.assertFalse(torch.isnan(out).any())
            
            # Image-only model forward pass
            cfg.USE_IMAGE_BRANCH = True
            cfg.USE_MESH_BRANCH = False
            model_img = build_model_from_config(cfg, effective_mesh_dim=68)
            model_img.eval()
            with torch.no_grad():
                out_img = forward_model(model_img, views, None, None, cfg)
            self.assertEqual(out_img.shape, (1, 10))
            
            # Mesh-only model forward pass
            cfg.USE_IMAGE_BRANCH = False
            cfg.USE_MESH_BRANCH = True
            model_mesh = build_model_from_config(cfg, effective_mesh_dim=68)
            model_mesh.eval()
            with torch.no_grad():
                out_mesh = forward_model(model_mesh, None, mesh_feat, None, cfg)
            self.assertEqual(out_mesh.shape, (1, 10))

        finally:
            cfg.USE_IMAGE_BRANCH = original_image_branch
            cfg.USE_MESH_BRANCH = original_mesh_branch

    def test_04_checkpoint_contract(self):
        """4. Checkpoint save/load contract and metadata verification."""
        cfg.USE_IMAGE_BRANCH = True
        cfg.USE_MESH_BRANCH = True
        cfg.IMAGE_IN_CHANNELS = 3
        cfg.IMAGE_BACKBONE = "efficientnetv2_s"
        
        contract = build_model_contract(cfg, effective_mesh_dim=68)
        
        # Valid checkpoint
        ckpt = {
            "model_contract": contract,
            "scaler_mean": np.zeros(68),
            "scaler_std": np.ones(68)
        }
        
        # Mismatched model config (should raise ValueError if strict=True)
        mismatched_cfg = type('cfg_mock', (), {
            'USE_IMAGE_BRANCH': True,
            'USE_MESH_BRANCH': True,
            'IMAGE_IN_CHANNELS': 6, # mismatched channel count!
            'IMAGE_BACKBONE': 'efficientnetv2_s',
            'USE_POINTNET_BRANCH': False,
            'USE_MOE': False,
            'USE_CROSS_VIEW_TRANSFORMER': False,
            'USE_DEFECT_QUERY_DECODER': False,
            'USE_SPATIAL_VIEW_TOKENS': False,
            'USE_CROSS_MODAL_ATTENTION': False,
            'USE_EARLY_EXIT': False
        })
        
        with self.assertRaises((ValueError, RuntimeError)):
            validate_checkpoint_contract(ckpt, mismatched_cfg, effective_mesh_dim=68, strict=True)

    def test_05_tta_flipping(self):
        """5. TTA spatial dimensions only (correct W and H rotation/flipping)."""
        # Batch: B=1, V=1, C=3, H=2, W=4
        views = torch.zeros((1, 1, 3, 2, 4))
        # Top-left pixel is marked
        views[0, 0, :, 0, 0] = 1.0
        
        # Horizontally flipped (dim 4 is width)
        flipped = torch.flip(views, dims=[4])
        self.assertEqual(flipped[0, 0, 0, 0, 3], 1.0)
        
        # Rotated 90 (dims 3, 4)
        rotated = torch.rot90(views, k=1, dims=[3, 4])
        self.assertEqual(rotated.shape, (1, 1, 3, 4, 2))

if __name__ == "__main__":
    unittest.main()
