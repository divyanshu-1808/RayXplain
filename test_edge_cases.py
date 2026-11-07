"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 6: Edge Cases & Validation Test Suite (test_edge_cases.py)

Comprehensive test harness validating:
1. Tabular Sanity Checks:
   - SpO2 clamping in [50, 100].
   - Age strictly in [1, 110].
   - Body temperature strictly in [95.0, 107.0]°F (and Celsius conversion).
   - User-facing warnings for atypical vitals.
2. Image Robustness:
   - 1-channel Grayscale (PIL 'L', 2D NumPy, 3D (H,W,1) NumPy).
   - 4-channel RGBA with transparency alpha compositing.
   - Corrupted byte streams and empty/zero-dimension arrays.
3. System Fallback:
   - Emergency CUDA OOM fallback to CPU without application crash.
4. Gradio Web UI Error/Warning Modal Handlers:
   - gr.Error alerts for missing/invalid inputs.
   - gr.Warning alerts for atypical vitals.
"""

import io
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from app import analyze_patient_case
from src.pipeline import (
    TARGET_CLASSES,
    DiagnosticPipeline,
    check_clinical_vitals,
    sanitize_image,
)


class TestEdgeCasesAndValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize pipeline on CPU for portable, fast edge-case verification
        cls.pipeline = DiagnosticPipeline(device="cpu")
        cls.valid_img = Image.new("RGB", (200, 200), color=(100, 110, 120))

    # --------------------------------------------------------------------------
    # 1. Tabular Sanity & Clamping Checks
    # --------------------------------------------------------------------------
    def test_spo2_clamping(self):
        """Verify SpO2 is clamped within [50.0, 100.0] and atypical warnings are issued."""
        # Value below 50% -> clamped to 50.0%
        vitals_low, warns_low = check_clinical_vitals(
            age=40, sex="Male", spo2=30.0, smoking=0, cough_severity="None", fever=98.6
        )
        self.assertEqual(vitals_low["spo2"], 50.0)
        self.assertTrue(any("clamped to physiological floor" in w for w in warns_low))
        self.assertTrue(any("Critical Hypoxemia" in w for w in warns_low))

        # Value above 100% -> clamped to 100.0%
        vitals_high, warns_high = check_clinical_vitals(
            age=40, sex="Male", spo2=115.0, smoking=0, cough_severity="None", fever=98.6
        )
        self.assertEqual(vitals_high["spo2"], 100.0)
        self.assertTrue(any("clamped to maximum saturation" in w for w in warns_high))

        # Decimal fraction (0.94) -> auto-scaled to 94.0%
        vitals_dec, _ = check_clinical_vitals(
            age=40, sex="Male", spo2=0.94, smoking=0, cough_severity="None", fever=98.6
        )
        self.assertAlmostEqual(vitals_dec["spo2"], 94.0, places=1)

        # Non-numeric SpO2 raises ValueError
        with self.assertRaises(ValueError):
            check_clinical_vitals(age=40, sex="Male", spo2="not_a_number", smoking=0, cough_severity=0, fever=98.6)

    def test_age_boundaries(self):
        """Verify age is strictly enforced within [1, 110] years."""
        # Age = 0 -> rejected
        with self.assertRaises(ValueError):
            check_clinical_vitals(age=0, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=98.6)

        # Negative age -> rejected
        with self.assertRaises(ValueError):
            check_clinical_vitals(age=-5, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=98.6)

        # Age > 110 -> rejected
        with self.assertRaises(ValueError):
            check_clinical_vitals(age=111, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=98.6)

        # Valid boundaries: Age = 1 and Age = 110 -> accepted
        v1, _ = check_clinical_vitals(age=1, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=98.6)
        self.assertEqual(v1["age"], 1.0)

        v110, warns = check_clinical_vitals(age=110, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=98.6)
        self.assertEqual(v110["age"], 110.0)
        self.assertTrue(any("Geriatric" in w for w in warns))

    def test_temperature_boundaries(self):
        """Verify body temperature is strictly enforced within [95.0, 107.0]°F."""
        # Temp below 95.0°F -> rejected
        with self.assertRaises(ValueError):
            check_clinical_vitals(age=40, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=94.8)

        # Temp above 107.0°F -> rejected
        with self.assertRaises(ValueError):
            check_clinical_vitals(age=40, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=107.5)

        # Exact bounds: 95.0°F and 107.0°F -> accepted
        v_min, warns_min = check_clinical_vitals(age=40, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=95.0)
        self.assertEqual(v_min["fever"], 95.0)
        self.assertTrue(any("Hypothermia" in w for w in warns_min))

        v_max, warns_max = check_clinical_vitals(age=40, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=107.0)
        self.assertEqual(v_max["fever"], 107.0)
        self.assertTrue(any("Pyrexia" in w for w in warns_max))

        # Celsius input: 37.0°C (98.6°F) -> accepted
        v_celsius, _ = check_clinical_vitals(age=40, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=37.0)
        self.assertAlmostEqual(v_celsius["fever"], 98.6, places=1)

        # Subnormal Celsius input: 33.0°C (91.4°F < 95.0°F) -> rejected
        with self.assertRaises(ValueError):
            check_clinical_vitals(age=40, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=33.0)

    # --------------------------------------------------------------------------
    # 2. Image Robustness: Grayscale, RGBA, and Corruption
    # --------------------------------------------------------------------------
    def test_image_robustness_grayscale(self):
        """Verify 1-channel Grayscale images are converted to 3-channel RGB."""
        # 1. PIL Grayscale ('L' mode)
        gray_pil = Image.new("L", (180, 180), color=128)
        clean1 = sanitize_image(gray_pil)
        self.assertEqual(clean1.mode, "RGB")
        self.assertEqual(clean1.size, (180, 180))

        # 2. 2D NumPy Array: (H, W)
        np_2d = np.random.randint(20, 220, (150, 150), dtype=np.uint8)
        clean2 = sanitize_image(np_2d)
        self.assertEqual(clean2.mode, "RGB")
        self.assertEqual(clean2.size, (150, 150))

        # 3. 3D 1-channel NumPy Array: (H, W, 1)
        np_3d_single = np.random.randint(20, 220, (160, 160, 1), dtype=np.uint8)
        clean3 = sanitize_image(np_3d_single)
        self.assertEqual(clean3.mode, "RGB")

        # Verify pipeline accepts all grayscale forms end-to-end
        overlay, probs, pred = self.pipeline.predict(
            pil_image=gray_pil, age=40, sex="Male", spo2=96, smoking=0, cough_severity=0, fever=98.6
        )
        self.assertEqual(overlay.shape, (180, 180, 3))
        self.assertIn(pred, TARGET_CLASSES)

    def test_image_robustness_rgba(self):
        """Verify 4-channel RGBA images are alpha-composited with white background."""
        # PIL RGBA with 50% opacity
        rgba_pil = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
        clean_rgba = sanitize_image(rgba_pil)
        self.assertEqual(clean_rgba.mode, "RGB")
        self.assertEqual(clean_rgba.size, (200, 200))

        # 4-channel NumPy Array (H, W, 4)
        np_4ch = np.random.randint(0, 255, (120, 120, 4), dtype=np.uint8)
        clean_np4 = sanitize_image(np_4ch)
        self.assertEqual(clean_np4.mode, "RGB")

        # Full pipeline prediction on RGBA
        overlay, probs, pred = self.pipeline.predict(
            pil_image=rgba_pil, age=45, sex="Female", spo2=97, smoking=0, cough_severity="None", fever=98.6
        )
        self.assertEqual(overlay.shape, (200, 200, 3))

    def test_image_robustness_corrupted_and_empty(self):
        """Verify corrupted or zero-dimension images raise descriptive ValueErrors without crashing."""
        # 1. Corrupted byte buffer
        corrupted_bytes = b"NOT_A_VALID_IMAGE_HEADER_DATA_1234567890"
        with self.assertRaises(ValueError) as ctx:
            sanitize_image(corrupted_bytes)
        self.assertIn("Failed to decode", str(ctx.exception))

        # 2. Empty numpy array
        empty_arr = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            sanitize_image(empty_arr)
        self.assertIn("size is 0", str(ctx.exception))

        # 3. Non-existent file path
        with self.assertRaises(ValueError) as ctx:
            sanitize_image("non_existent_chest_xray_9999.png")
        self.assertIn("not found", str(ctx.exception))

        # 4. Non-image object type
        with self.assertRaises(ValueError) as ctx:
            sanitize_image(12345)
        self.assertIn("Unsupported image object type", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 3. System Fallback: Emergency CUDA OOM Recovery
    # --------------------------------------------------------------------------
    def test_cuda_oom_fallback(self):
        """Verify fallback_to_cpu correctly transfers model, GradCAM, and execution to CPU."""
        self.pipeline.fallback_to_cpu()
        self.assertEqual(self.pipeline.device.type, "cpu")

        # Verify all model parameter tensors reside on CPU
        for p in self.pipeline.model.parameters():
            self.assertEqual(p.device.type, "cpu")

        # Verify inference succeeds seamlessly on CPU after migration
        overlay, probs, pred = self.pipeline.predict(
            pil_image=self.valid_img,
            age=52,
            sex="Male",
            spo2=94,
            smoking=5.0,
            cough_severity="Mild",
            fever=99.5,
        )
        self.assertIn(pred, TARGET_CLASSES)
        self.assertEqual(overlay.shape, (200, 200, 3))

    # --------------------------------------------------------------------------
    # 4. Gradio UI Controller: Error & Warning Handlers
    # --------------------------------------------------------------------------
    def test_app_error_notifications(self):
        """Verify analyze_patient_case raises gr.Error for invalid inputs without crashing."""
        # Missing image triggers gr.Error
        with self.assertRaises(gr.Error) as ctx:
            analyze_patient_case(
                pil_image=None, age=45, sex="Male", spo2=96, smoking=0, cough="None", fever=98.6
            )
        self.assertIn("Missing Chest Radiograph", str(ctx.exception))

        # Invalid age triggers gr.Error
        with self.assertRaises(gr.Error) as ctx:
            analyze_patient_case(
                pil_image=self.valid_img, age=125, sex="Male", spo2=96, smoking=0, cough="None", fever=98.6
            )
        self.assertIn("Validation Error", str(ctx.exception))

        # Invalid temperature triggers gr.Error
        with self.assertRaises(gr.Error) as ctx:
            analyze_patient_case(
                pil_image=self.valid_img, age=45, sex="Male", spo2=96, smoking=0, cough="None", fever=91.0
            )
        self.assertIn("Validation Error", str(ctx.exception))

    def test_app_atypical_vitals_badge_and_warnings(self):
        """Verify atypical clinical vitals trigger alerts in the summary badge HTML."""
        # Patient with critical SpO2 (<90%) and high fever (102.5°F)
        overlay, orig, probs, badge_html = analyze_patient_case(
            pil_image=self.valid_img,
            age=82,
            sex="Male",
            spo2=86.0,
            smoking=35.0,
            cough="Severe",
            fever=102.5,
        )
        self.assertIsNotNone(overlay)
        self.assertIn("Atypical Vitals Alert", badge_html)
        self.assertIn("Critical Hypoxemia", badge_html)
        self.assertIn("Pyrexia Alert", badge_html)
        self.assertIn("Heavy Smoking", badge_html)


if __name__ == "__main__":
    print("=" * 70)
    print(" Running Phase 6 Edge Cases & Defensive Validation Tests")
    print("=" * 70)
    unittest.main()

