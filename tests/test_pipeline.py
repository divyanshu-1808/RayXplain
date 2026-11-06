"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 4 Unit Tests: Diagnostic Pipeline
"""

import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.pipeline import TARGET_CLASSES, DiagnosticPipeline


class TestDiagnosticPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force CPU device for fast, portable unit testing
        cls.pipeline = DiagnosticPipeline(device="cpu")

    def test_pipeline_initialization(self):
        """Verify pipeline loads model, scaler, and attaches GradCAM on specified device."""
        self.assertEqual(self.pipeline.device.type, "cpu")
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.scaler)
        self.assertIsNotNone(self.pipeline.gradcam)
        self.assertFalse(self.pipeline.model.training)

    def test_transform_image_pil(self):
        """Verify image transform pipeline with PIL image input."""
        pil_img = Image.new("RGB", (320, 240), color=(120, 130, 140))
        tensor = self.pipeline.transform_image(pil_img)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.device.type, "cpu")

    def test_transform_image_numpy(self):
        """Verify image transform pipeline with numpy array input."""
        np_img = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        tensor = self.pipeline.transform_image(np_img)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_transform_tabular_variations(self):
        """Verify tabular transform supports both strings and numeric inputs."""
        # 1. String inputs with Celsius temperature
        t1 = self.pipeline.transform_tabular(
            age=45,
            sex="Female",
            spo2=96.0,
            smoking_history="No",
            cough_severity="Mild",
            fever=37.2,  # Celsius -> will be converted to ~98.96°F
        )
        self.assertEqual(t1.shape, (1, 6))
        self.assertFalse(torch.isnan(t1).any())

        # 2. Numeric inputs with Fahrenheit temperature
        t2 = self.pipeline.transform_tabular(
            age=62,
            sex=1,
            spo2=90.0,
            smoking_history=25.0,
            cough_severity=2,
            fever=102.5,
        )
        self.assertEqual(t2.shape, (1, 6))
        self.assertFalse(torch.isnan(t2).any())

        # 3. Invalid sex raises ValueError
        with self.assertRaises(ValueError):
            self.pipeline.transform_tabular(40, "InvalidSex", 95, 0, 0, 98.6)

    def test_input_validation(self):
        """Verify validation catches nulls, negatives, and out-of-range parameters."""
        dummy_img = Image.new("RGB", (100, 100))

        # None image
        with self.assertRaises(ValueError):
            self.pipeline.validate_inputs(None, 40, "Male", 95, 0, 0, 98.6)

        # Negative age
        with self.assertRaises(ValueError):
            self.pipeline.validate_inputs(dummy_img, -1, "Male", 95, 0, 0, 98.6)

        # Extreme age > 125
        with self.assertRaises(ValueError):
            self.pipeline.validate_inputs(dummy_img, 150, "Male", 95, 0, 0, 98.6)

        # Invalid SpO2 (< 40%)
        with self.assertRaises(ValueError):
            self.pipeline.validate_inputs(dummy_img, 50, "Male", 30.0, 0, 0, 98.6)

        # Negative smoking pack-years
        with self.assertRaises(ValueError):
            self.pipeline.validate_inputs(dummy_img, 50, "Male", 95.0, -5.0, 0, 98.6)

        # Extreme body temperature
        with self.assertRaises(ValueError):
            self.pipeline.validate_inputs(dummy_img, 50, "Male", 95.0, 0, 0, 150.0)

    def test_predict_end_to_end(self):
        """Verify full predict() pipeline returns (overlay, probabilities_dict, predicted_class_name)."""
        pil_img = Image.new("RGB", (250, 350), color=(80, 80, 80))

        overlay, probs, pred_name = self.pipeline.predict(
            pil_image=pil_img,
            age=55,
            sex="Male",
            spo2=92.0,
            smoking=10.0,
            cough_severity="Moderate",
            fever=101.2,
        )

        # Return types check
        self.assertIsInstance(overlay, np.ndarray)
        self.assertIsInstance(probs, dict)
        self.assertIsInstance(pred_name, str)

        # Overlay shape must match original image size (height=350, width=250, channels=3)
        self.assertEqual(overlay.shape, (350, 250, 3))
        self.assertEqual(overlay.dtype, np.uint8)

        # Probabilities dictionary check
        self.assertEqual(set(probs.keys()), set(TARGET_CLASSES))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        self.assertIn(pred_name, TARGET_CLASSES)

    def test_predict_alias_handling(self):
        """Verify predict supports keyword argument smoking_history."""
        pil_img = Image.new("RGB", (100, 100), color=(60, 60, 60))

        overlay, probs, pred_name = self.pipeline.predict(
            pil_image=pil_img,
            age=30,
            sex="Female",
            spo2=98.0,
            smoking_history=0.0,
            cough_severity="None",
            fever=98.4,
        )

        self.assertIn(pred_name, TARGET_CLASSES)
        self.assertEqual(len(probs), 3)


if __name__ == "__main__":
    unittest.main()

