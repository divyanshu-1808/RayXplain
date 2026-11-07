"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 5 Unit Tests: Web Application Interface
"""

import unittest
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from app import analyze_patient_case, create_app, demo, ensure_example_cases


class TestAppInterface(unittest.TestCase):
    def test_app_instantiation(self):
        """Verify Gradio app builds cleanly without configuration errors."""
        self.assertIsInstance(demo, gr.Blocks)
        self.assertIsNotNone(demo.children)

    def test_missing_image_validation(self):
        """Verify error is raised when image input is missing."""
        with self.assertRaises(gr.Error):
            analyze_patient_case(
                pil_image=None,
                age=45,
                sex="Male",
                spo2=96,
                smoking=0,
                cough="None",
                fever=98.6,
            )

    def test_analyze_patient_case_success(self):
        """Verify end-to-end prediction through the UI controller."""
        dummy_img = Image.new("RGB", (250, 250), color=(100, 100, 100))
        overlay, orig, probs, badge_html = analyze_patient_case(
            pil_image=dummy_img,
            age=50,
            sex="Female",
            spo2=95,
            smoking=5.0,
            cough="Mild",
            fever=99.2,
        )

        self.assertIsInstance(overlay, np.ndarray)
        self.assertEqual(overlay.shape, (250, 250, 3))
        self.assertIs(orig, dummy_img)
        self.assertIsInstance(probs, dict)
        self.assertEqual(len(probs), 3)
        self.assertIn("<div", badge_html)
        self.assertIn("Clinical Guidance:", badge_html)

    def test_example_cases_exist(self):
        """Verify that synthetic demo scenario files exist and match input schema."""
        examples = ensure_example_cases()
        self.assertGreaterEqual(len(examples), 3)
        for row in examples:
            img_path, age, sex, spo2, smoking, cough, fever = row
            self.assertTrue(Path(img_path).is_file(), f"Example file missing: {img_path}")
            self.assertIn(sex, ["Male", "Female"])
            self.assertIn(cough, ["None", "Mild", "Moderate", "Severe"])
            self.assertGreaterEqual(spo2, 70)


if __name__ == "__main__":
    unittest.main()

