"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 3 Unit Tests: Grad-CAM Explainability Engine
"""

import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.gradcam import GradCAM, overlay_heatmap
from src.model import CLASS_NAMES, MultiModalLungNet, build_model


class TestGradCAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fast, lightweight model initialization for unit testing
        cls.model = MultiModalLungNet(pretrained=False)
        cls.model.eval()

    def setUp(self):
        # Ensure fresh model state before each test
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

    def test_layer_resolution_variants(self):
        """Verify target layer resolution by string name, None, and module reference."""
        # 1. Default 'layer4'
        cam1 = GradCAM(self.model, "layer4")
        self.assertIs(cam1.target_layer, self.model.get_conv_layer())
        cam1.remove_hooks()

        # 2. None (auto-detects layer4 / get_conv_layer)
        cam2 = GradCAM(self.model, None)
        self.assertIs(cam2.target_layer, self.model.get_conv_layer())
        cam2.remove_hooks()

        # 3. Direct module
        cam3 = GradCAM(self.model, self.model.get_conv_layer())
        self.assertIs(cam3.target_layer, self.model.get_conv_layer())
        cam3.remove_hooks()

        # 4. Dot-separated path 'conv_base.7'
        cam4 = GradCAM(self.model, "conv_base.7")
        self.assertIs(cam4.target_layer, self.model.conv_base[7])
        cam4.remove_hooks()

        # 5. Invalid layer name raises ValueError
        with self.assertRaises(ValueError):
            GradCAM(self.model, "non_existent_layer_xyz")

    def test_generate_heatmap_argmax(self):
        """Verify Grad-CAM generation when target_class=None (automatic argmax)."""
        cam = GradCAM(self.model, "layer4")
        x_img = torch.randn(1, 3, 224, 224)
        x_clin = torch.randn(1, 6)

        heatmap = cam.generate_heatmap(x_img, x_clin, target_class=None)

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, (7, 7))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertFalse(np.isnan(heatmap).any())
        cam.remove_hooks()

    def test_generate_heatmap_all_classes(self):
        """Verify Grad-CAM generation for each explicit target class."""
        cam = GradCAM(self.model, "layer4")
        x_img = torch.randn(1, 3, 224, 224)
        x_clin = torch.randn(1, 6)

        for target_cls in range(3):
            heatmap = cam.generate_heatmap(x_img, x_clin, target_class=target_cls)
            self.assertEqual(heatmap.shape, (7, 7))
            self.assertGreaterEqual(float(heatmap.min()), 0.0)
            self.assertLessEqual(float(heatmap.max()), 1.0)

        # Invalid target class index
        with self.assertRaises(ValueError):
            cam.generate_heatmap(x_img, x_clin, target_class=99)

        with self.assertRaises(ValueError):
            cam.generate_heatmap(x_img, x_clin, target_class=-1)

        cam.remove_hooks()

    def test_tensor_dimension_flexibility(self):
        """Verify handling of unbatched [3, H, W] and [6] input tensors."""
        cam = GradCAM(self.model, "layer4")
        x_img_3d = torch.randn(3, 224, 224)
        x_clin_1d = torch.randn(6)

        heatmap = cam.generate_heatmap(x_img_3d, x_clin_1d)
        self.assertEqual(heatmap.shape, (7, 7))
        cam.remove_hooks()

    def test_batch_mismatch_raises_error(self):
        """Verify mismatched batch sizes raise ValueError."""
        cam = GradCAM(self.model, "layer4")
        x_img = torch.randn(2, 3, 224, 224)
        x_clin = torch.randn(1, 6)

        with self.assertRaises(ValueError):
            cam.generate_heatmap(x_img, x_clin)
        cam.remove_hooks()

    def test_frozen_vision_backbone_gradient_propagation(self):
        """Verify Grad-CAM computes valid gradients even when backbone parameters are frozen."""
        cam = GradCAM(self.model, "layer4")
        self.model.freeze_vision_backbone(freeze=True)

        x_img = torch.randn(1, 3, 224, 224)
        x_clin = torch.randn(1, 6)

        heatmap = cam.generate_heatmap(x_img, x_clin)
        self.assertEqual(heatmap.shape, (7, 7))
        self.assertFalse(np.isnan(heatmap).any())

        # Unfreeze for subsequent tests
        self.model.freeze_vision_backbone(freeze=False)
        cam.remove_hooks()

    def test_memory_and_eval_mode_preservation(self):
        """Verify evaluation mode is preserved and no tensors or gradients are leaked."""
        cam = GradCAM(self.model, "layer4")
        self.model.eval()

        x_img = torch.randn(1, 3, 224, 224)
        x_clin = torch.randn(1, 6)

        _ = cam.generate_heatmap(x_img, x_clin)

        # Model must still be in eval mode
        self.assertFalse(self.model.training)

        # Hook buffers must be cleared
        self.assertIsNone(cam.activations)
        self.assertIsNone(cam.gradients)

        # Model parameter gradients should be None/zeroed
        for param in self.model.parameters():
            self.assertIsNone(param.grad)

        cam.remove_hooks()

    def test_overlay_heatmap_pil(self):
        """Verify overlay_heatmap with PIL image input and bilinear resizing."""
        heatmap = np.random.rand(7, 7).astype(np.float32)
        pil_img = Image.new("RGB", (320, 240), color=(100, 150, 200))

        blended = overlay_heatmap(heatmap, pil_img, alpha=0.4)

        self.assertIsInstance(blended, np.ndarray)
        self.assertEqual(blended.shape, (240, 320, 3))
        self.assertEqual(blended.dtype, np.uint8)
        self.assertGreaterEqual(int(blended.min()), 0)
        self.assertLessEqual(int(blended.max()), 255)

    def test_overlay_heatmap_numpy(self):
        """Verify overlay_heatmap with numpy array input."""
        heatmap = np.zeros((7, 7), dtype=np.float32)
        heatmap[3, 3] = 1.0  # Center activation

        orig_np = np.full((100, 100, 3), 128, dtype=np.uint8)
        blended = overlay_heatmap(heatmap, orig_np, alpha=0.5)

        self.assertEqual(blended.shape, (100, 100, 3))
        self.assertEqual(blended.dtype, np.uint8)

    def test_overlay_alpha_limits(self):
        """Verify alpha blending limits (alpha=0.0 retains original image)."""
        heatmap = np.random.rand(7, 7).astype(np.float32)
        orig_np = np.full((50, 50, 3), 77, dtype=np.uint8)

        # Alpha = 0.0 should return pure original image
        overlay_zero = overlay_heatmap(heatmap, orig_np, alpha=0.0)
        np.testing.assert_array_equal(overlay_zero, orig_np)

    def test_context_manager_cleanup(self):
        """Verify context manager automatically removes hooks upon exit."""
        with GradCAM(self.model, "layer4") as cam:
            self.assertIsNotNone(cam._forward_handle)
            x_img = torch.randn(1, 3, 224, 224)
            x_clin = torch.randn(1, 6)
            hm = cam.generate_heatmap(x_img, x_clin)
            self.assertEqual(hm.shape, (7, 7))

        # After exiting context, handles must be cleaned up
        self.assertIsNone(cam._forward_handle)
        self.assertIsNone(cam._backward_handle)

    def test_explain_service(self):
        """Verify high-level explain() method returns complete diagnostic dictionary."""
        cam = GradCAM(self.model, "layer4")
        x_img = torch.randn(1, 3, 224, 224)
        x_clin = torch.randn(1, 6)
        pil_img = Image.new("RGB", (224, 224), color=(50, 50, 50))

        result = cam.explain(x_img, x_clin, original_pil_img=pil_img)

        self.assertIn("target_class", result)
        self.assertIn("predicted_class", result)
        self.assertIn("predicted_label", result)
        self.assertIn("logits", result)
        self.assertIn("probabilities", result)
        self.assertIn("heatmap", result)
        self.assertIn("overlay", result)

        self.assertEqual(result["logits"].shape, (3,))
        self.assertEqual(result["probabilities"].shape, (3,))
        self.assertAlmostEqual(float(result["probabilities"].sum()), 1.0, places=4)
        self.assertIsNotNone(result["overlay"])
        self.assertEqual(result["overlay"].shape, (224, 224, 3))
        cam.remove_hooks()

    def test_checkpoint_weights_integration(self):
        """Verify Grad-CAM execution using the trained project checkpoint weights."""
        repo_root = Path(__file__).resolve().parent.parent
        weights_file = repo_root / "weights" / "best_multimodal_model.pth"

        if weights_file.is_file():
            loaded_model = build_model(pretrained=False, weights_path=weights_file)
            loaded_model.eval()

            with GradCAM(loaded_model, "layer4") as cam:
                x_img = torch.randn(1, 3, 224, 224)
                x_clin = torch.randn(1, 6)
                pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

                explanation = cam.explain(x_img, x_clin, original_pil_img=pil_img)
                self.assertIn(explanation["predicted_class"], [0, 1, 2])
                self.assertEqual(explanation["heatmap"].shape, (7, 7))
                self.assertEqual(explanation["overlay"].shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()

