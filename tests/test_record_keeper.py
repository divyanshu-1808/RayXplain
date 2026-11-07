"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Unit Tests: Canonical Cases, Offline Benchmark Samples, and CaseArchiver
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app import archive_current_case
from src.record_keeper import CaseArchiver
from test_samples.canonical_cases import (
    BENCHMARK_CASES,
    get_case_by_name,
    list_benchmark_cases,
)
from test_samples.setup_offline_samples import validate_sample_integrity


class TestCanonicalCases(unittest.TestCase):
    """Test suite for canonical benchmark cases registry."""

    def test_canonical_cases_structure(self):
        """Verify that all canonical benchmark cases are well-formed and complete."""
        self.assertEqual(len(BENCHMARK_CASES), 5, "Expected exactly 5 canonical benchmark cases")

        required_keys = {
            "case_id",
            "title",
            "local_image_path",
            "age",
            "sex",
            "spo2",
            "smoking_pack_years",
            "cough_severity",
            "fever_f",
            "expected_class",
            "clinical_rationale",
        }

        for case_key, case_data in BENCHMARK_CASES.items():
            self.assertTrue(case_data["case_id"].startswith("CASE-"))
            for key in required_keys:
                self.assertIn(key, case_data, f"Missing key '{key}' in case {case_key}")

            self.assertIn(case_data["sex"], ["Male", "Female"])
            self.assertIn(case_data["cough_severity"], ["None", "Mild", "Moderate", "Severe"])
            self.assertIn(case_data["expected_class"], ["Normal", "Pneumonia", "High-Risk / COVID-19"])
            self.assertGreaterEqual(case_data["spo2"], 50)
            self.assertLessEqual(case_data["spo2"], 100)
            self.assertGreaterEqual(case_data["age"], 1)
            self.assertLessEqual(case_data["age"], 110)
            self.assertGreaterEqual(case_data["fever_f"], 95.0)
            self.assertLessEqual(case_data["fever_f"], 107.0)

    def test_get_case_by_name(self):
        """Verify case retrieval by identifier or title query."""
        case = get_case_by_name("CASE-NORM-01")
        self.assertIsNotNone(case)
        self.assertEqual(case["case_id"], "CASE-NORM-01")

        case_ci = get_case_by_name("sample-pneumonia-bacterial")
        self.assertIsNotNone(case_ci)
        self.assertEqual(case_ci["case_id"], "CASE-PNEU-BAC-01")

        case_title = get_case_by_name("Acute Progressive COVID-19")
        self.assertIsNotNone(case_title)
        self.assertEqual(case_title["case_id"], "CASE-COVD-ACU-01")

        case_none = get_case_by_name("NON_EXISTENT_CASE_XYZ")
        self.assertIsNone(case_none)

    def test_list_benchmark_cases(self):
        """Verify list_benchmark_cases returns all 5 entries."""
        cases = list_benchmark_cases()
        self.assertEqual(len(cases), 5)
        self.assertTrue(all(isinstance(c, dict) for c in cases))


class TestOfflineSamplesIntegrity(unittest.TestCase):
    """Verify integrity of permanent local radiograph benchmarks in test_samples/."""

    def test_all_sample_images_valid(self):
        """Verify every registered benchmark image exists, opens, and passes shape checks."""
        for case_id, case_info in BENCHMARK_CASES.items():
            img_path = Path(case_info["local_image_path"])
            self.assertTrue(img_path.is_file(), f"Sample image file not found: {img_path}")

            is_valid, msg = validate_sample_integrity(img_path)
            self.assertTrue(is_valid, f"Validation failed for {case_id} ({img_path}): {msg}")
            with Image.open(str(img_path)) as img:
                w, h = img.size
                self.assertGreaterEqual(w, 224)
                self.assertGreaterEqual(h, 224)
                self.assertEqual(img.convert("RGB").mode, "RGB")


class TestCaseArchiver(unittest.TestCase):
    """Test suite for CaseArchiver automated audit and export system."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rayxplain_test_records_")
        self.archiver = CaseArchiver(records_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_master_csv_initialization(self):
        """Verify master evaluations_history.csv is initialized with headers."""
        csv_path = Path(self.test_dir) / "evaluations_history.csv"
        self.assertTrue(csv_path.is_file())

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, CaseArchiver.CSV_HEADER)

    def test_save_evaluation_artifacts(self):
        """Verify that save_evaluation exports original image, GradCAM overlay, JSON, and CSV row."""
        orig_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        overlay_arr = np.full((300, 300, 3), 200, dtype=np.uint8)

        vitals = {
            "age": 45,
            "sex": "Male",
            "spo2": 96.0,
            "smoking": 10.0,
            "cough": "Mild",
            "fever": 99.1,
        }

        preds = {
            "predicted_class": "Pneumonia",
            "confidence": 0.942,
            "probabilities": {
                "Normal": 0.02,
                "Pneumonia": 0.942,
                "High-Risk / COVID-19": 0.038,
            },
        }

        saved_path = self.archiver.save_evaluation(
            original_pil_img=orig_img,
            overlay_pil_img=overlay_arr,
            vitals_dict=vitals,
            prediction_dict=preds,
            patient_tag="UnitTest_Patient_01",
            latency_sec=0.085,
        )

        # 1. Directory exists
        self.assertTrue(saved_path.is_dir())
        self.assertTrue(saved_path.name.startswith("TEST_"))
        self.assertIn("UnitTest_Patient_01", saved_path.name)

        # 2. Artifact files exist
        orig_file = saved_path / "original_xray.png"
        gradcam_file = saved_path / "gradcam_explanation.png"
        json_file = saved_path / "clinical_summary.json"

        self.assertTrue(orig_file.is_file())
        self.assertTrue(gradcam_file.is_file())
        self.assertTrue(json_file.is_file())

        # 3. Validate JSON summary content
        with open(json_file, mode="r", encoding="utf-8") as jf:
            summary = json.load(jf)

        self.assertEqual(summary["patient_tag"], "UnitTest_Patient_01")
        self.assertEqual(summary["clinical_vitals"]["age"], 45)
        self.assertEqual(summary["diagnostic_assessment"]["predicted_class"], "Pneumonia")
        self.assertAlmostEqual(summary["diagnostic_assessment"]["confidence_percent"], 94.2, places=1)
        self.assertEqual(summary["saved_artifacts"]["original_radiograph"], "original_xray.png")

        # 4. Validate CSV row appended
        csv_path = Path(self.test_dir) / "evaluations_history.csv"
        with open(csv_path, mode="r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            self.assertEqual(len(rows), 2)  # Header + 1 row
            data_row = rows[1]
            self.assertEqual(data_row[1], saved_path.name)
            self.assertEqual(data_row[2], "UnitTest_Patient_01")
            self.assertEqual(data_row[3], "Pneumonia")
            self.assertIn("94.2", data_row[4])
            self.assertEqual(data_row[6], "45")
            self.assertEqual(data_row[7], "Male")

    def test_rapid_succession_naming_collision_prevention(self):
        """Verify that rapid succession evaluations generate distinct folder names."""
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        vitals = {"age": 30, "sex": "Female", "spo2": 99.0, "smoking": 0, "cough": "None", "fever": 98.6}
        preds = {"predicted_class": "Normal", "confidence": 0.99}

        path1 = self.archiver.save_evaluation(img, overlay, vitals, preds, patient_tag="CollisionTest")
        path2 = self.archiver.save_evaluation(img, overlay, vitals, preds, patient_tag="CollisionTest")

        self.assertNotEqual(path1, path2)
        self.assertTrue(path1.is_dir())
        self.assertTrue(path2.is_dir())

    def test_app_archiving_controller_flow(self):
        """Verify the UI helper archive_current_case handles missing and valid inputs cleanly."""
        # 1. Missing images aborts gracefully
        status_empty = archive_current_case(
            raw_input_img=None,
            original_out_img=None,
            overlay_out_img=None,
            risk_label_data={},
            age=40,
            sex="Male",
            spo2=98,
            smoking=0,
            cough="None",
            fever=98.6,
            patient_tag="Test_Tag",
        )
        self.assertIn("Export Aborted", status_empty)

        # 2. Valid assessment archives properly
        img = Image.new("RGB", (200, 200), color=(80, 80, 80))
        overlay = np.zeros((200, 200, 3), dtype=np.uint8)
        risk_data = {"confidences": [{"label": "Pneumonia", "confidence": 0.88}]}

        status_success = archive_current_case(
            raw_input_img=img,
            original_out_img=img,
            overlay_out_img=overlay,
            risk_label_data=risk_data,
            age=55,
            sex="Male",
            spo2=92,
            smoking=15,
            cough="Moderate",
            fever=101.5,
            patient_tag="UI_Viva_Test",
        )
        self.assertIn("Case Successfully Archived", status_success)
        self.assertIn("clinical_summary.json", status_success)


if __name__ == "__main__":
    unittest.main()

