"""
Unit tests for RayXplain Phase 2 MultiModalLungNet architecture.
"""

import unittest
from pathlib import Path

import torch
from src.model import MultiModalLungNet, build_model, CLASS_NAMES, CLINICAL_FEATURE_NAMES


class TestMultiModalLungNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use pretrained=False for lightweight, fast offline unit test execution
        cls.model = MultiModalLungNet(pretrained=False)
        cls.model.eval()

    def test_forward_batch_2(self):
        """Verify forward pass with batch size of 2 and shape assertions."""
        batch_size = 2
        x_img = torch.randn(batch_size, 3, 224, 224)
        x_clin = torch.randn(batch_size, 6)

        with torch.no_grad():
            logits, conv_features = self.model(x_img, x_clin)

        self.assertEqual(logits.shape, (batch_size, 3))
        self.assertEqual(conv_features.shape, (batch_size, 2048, 7, 7))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isnan(conv_features).any())

    def test_single_sample_inference(self):
        """Verify forward pass with batch size of 1."""
        batch_size = 1
        x_img = torch.randn(batch_size, 3, 224, 224)
        x_clin = torch.randn(batch_size, 6)

        with torch.no_grad():
            logits, conv_features = self.model(x_img, x_clin)

        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(conv_features.shape, (1, 2048, 7, 7))

    def test_batch_mismatch_raises_error(self):
        """Verify that inconsistent image and clinical batch sizes raise ValueError."""
        x_img = torch.randn(3, 3, 224, 224)
        x_clin = torch.randn(2, 6)

        with self.assertRaises(ValueError):
            self.model(x_img, x_clin)

    def test_gradcam_target_layer(self):
        """Verify that get_conv_layer returns layer4 for Grad-CAM hooks."""
        target_layer = self.model.get_conv_layer()
        self.assertIsNotNone(target_layer)
        # Should be the 7th child of conv_base (layer4 Sequential)
        self.assertIs(target_layer, self.model.conv_base[7])

    def test_freeze_backbone(self):
        """Verify vision backbone parameter freezing."""
        self.model.freeze_vision_backbone(freeze=True)
        for p in self.model.conv_base.parameters():
            self.assertFalse(p.requires_grad)

        self.model.freeze_vision_backbone(freeze=False)
        for p in self.model.conv_base.parameters():
            self.assertTrue(p.requires_grad)

    def test_weights_checkpoint_exact_match(self):
        """Verify that weights/best_multimodal_model.pth matches architecture exactly."""
        repo_root = Path(__file__).resolve().parent.parent
        weights_file = repo_root / "weights" / "best_multimodal_model.pth"

        if weights_file.is_file():
            load_result = self.model.load_weights(weights_file, map_location="cpu", strict=True)
            self.assertEqual(len(load_result["missing_keys"]), 0, f"Missing: {load_result['missing_keys']}")
            self.assertEqual(len(load_result["unexpected_keys"]), 0, f"Unexpected: {load_result['unexpected_keys']}")

    def test_constants(self):
        """Verify predefined class names and clinical feature specifications."""
        self.assertEqual(len(CLASS_NAMES), 3)
        self.assertEqual(len(CLINICAL_FEATURE_NAMES), 6)


if __name__ == "__main__":
    unittest.main()
