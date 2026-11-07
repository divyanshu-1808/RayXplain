"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Manual Test Case Archiver (src/record_keeper.py)

This module provides academic documentation and viva verification record-keeping:
Whenever a clinical evaluation is executed, CaseArchiver exports:
1. original_xray.png       - The exact uploaded radiograph.
2. gradcam_explanation.png - The computed Grad-CAM saliency overlay.
3. clinical_summary.json   - Machine-readable metadata of vitals, predictions, and latency.
4. evaluations_history.csv - Master audit trail table readable in Microsoft Excel.
"""

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

_DEFAULT_RECORDS_DIR = Path(__file__).resolve().parent.parent / "patient_records"

logger = logging.getLogger("RayXplain.RecordKeeper")


class CaseArchiver:
    """
    Automated case exporter and record keeper for academic evaluation and clinical audits.

    Attributes:
        records_dir (Path): Root folder storing patient record directories.
        master_csv_path (Path): Master audit log file (evaluations_history.csv).
    """

    CSV_HEADER = [
        "Timestamp",
        "Folder_Name",
        "Patient_Tag",
        "Predicted_Class",
        "Confidence_Percent",
        "Latency_Seconds",
        "Age",
        "Sex",
        "SpO2",
        "Smoking_Pack_Years",
        "Cough_Severity",
        "Body_Temperature_F",
        "GradCAM_Saved",
    ]

    def __init__(self, records_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the archiver and ensure the destination and master CSV log exist.

        Args:
            records_dir: Target directory for storing patient records.
                         Defaults to 'patient_records/' in project root.
        """
        self.records_dir = Path(records_dir) if records_dir is not None else _DEFAULT_RECORDS_DIR
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.master_csv_path = self.records_dir / "evaluations_history.csv"

        # Initialize master CSV if not present
        if not self.master_csv_path.is_file():
            self._initialize_master_csv()

    def _initialize_master_csv(self) -> None:
        """Create the evaluations_history.csv audit file with headers."""
        try:
            with open(self.master_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADER)
            logger.info(f"Initialized master audit CSV at {self.master_csv_path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to initialize master CSV: {e}")

    def save_evaluation(
        self,
        original_pil_img: Union[Image.Image, np.ndarray, str, Path],
        overlay_pil_img: Union[Image.Image, np.ndarray, str, Path],
        vitals_dict: Dict[str, Any],
        prediction_dict: Dict[str, Any],
        patient_tag: Optional[str] = None,
        latency_sec: Optional[float] = None,
    ) -> Path:
        """
        Save a complete diagnostic evaluation session to patient_records/.

        Args:
            original_pil_img: Raw uploaded radiograph image.
            overlay_pil_img: Blended Grad-CAM saliency overlay image.
            vitals_dict: Dict with age, sex, spo2, smoking, cough, fever.
            prediction_dict: Dict with predicted_class, confidence, probabilities.
            patient_tag: Optional custom identifier (e.g. 'Patient_42' or 'Viva_Demo_1').
            latency_sec: Inference duration in seconds.

        Returns:
            Path: The directory where the patient evaluation was archived.

        Raises:
            ValueError: If images or inputs are invalid.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()

        # 1. Normalize Predicted Class & Confidence
        pred_class = str(prediction_dict.get("predicted_class", "Diagnosis"))
        confidence_val = prediction_dict.get("confidence", prediction_dict.get("confidence_percent", 0.0))
        # Handle decimal (0.85) vs percent (85.0)
        conf_pct = float(confidence_val) * 100.0 if float(confidence_val) <= 1.0 and float(confidence_val) > 0.0 else float(confidence_val)

        # 2. Generate Unique Folder Name
        tag_source = patient_tag if patient_tag and str(patient_tag).strip() else pred_class
        clean_tag = re.sub(r"[^a-zA-Z0-9_\-]", "", str(tag_source).replace(" ", "_").replace("/", "_"))
        clean_pred = re.sub(r"[^a-zA-Z0-9_\-]", "", pred_class.replace(" ", "_").replace("/", "_"))

        folder_name = f"TEST_{timestamp_str}_{clean_tag}"
        case_dir = self.records_dir / folder_name

        # Ensure uniqueness in rapid succession runs
        counter = 1
        while case_dir.exists():
            folder_name = f"TEST_{timestamp_str}_{clean_tag}_{counter}"
            case_dir = self.records_dir / folder_name
            counter += 1

        case_dir.mkdir(parents=True, exist_ok=True)

        # 3. Standardize and Save Images
        # Original Image
        orig_img = self._to_pil(original_pil_img)
        orig_path = case_dir / "original_xray.png"
        orig_img.save(str(orig_path), format="PNG")

        # Grad-CAM Overlay Image
        overlay_img = self._to_pil(overlay_pil_img)
        overlay_path = case_dir / "gradcam_explanation.png"
        overlay_img.save(str(overlay_path), format="PNG")

        # 4. Standardize Vitals Data
        age = vitals_dict.get("age", "")
        sex = vitals_dict.get("sex", "")
        spo2 = vitals_dict.get("spo2", "")
        smoking = vitals_dict.get("smoking", vitals_dict.get("smoking_pack_years", vitals_dict.get("smoking_history", 0.0)))
        cough = vitals_dict.get("cough", vitals_dict.get("cough_severity", ""))
        fever = vitals_dict.get("fever", vitals_dict.get("fever_f", vitals_dict.get("body_temperature", 98.6)))

        # 5. Build and Save clinical_summary.json
        summary_data = {
            "case_id": folder_name,
            "patient_tag": patient_tag or clean_tag,
            "timestamp": iso_timestamp,
            "formatted_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "clinical_vitals": {
                "age": age,
                "sex": sex,
                "spo2_percent": spo2,
                "smoking_pack_years": smoking,
                "cough_severity": cough,
                "body_temperature_f": fever,
            },
            "diagnostic_assessment": {
                "predicted_class": pred_class,
                "confidence_percent": round(conf_pct, 2),
                "class_probabilities": prediction_dict.get("probabilities", {}),
                "inference_latency_seconds": round(float(latency_sec or 0.0), 4),
            },
            "saved_artifacts": {
                "original_radiograph": "original_xray.png",
                "gradcam_explanation": "gradcam_explanation.png",
            },
        }

        json_path = case_dir / "clinical_summary.json"
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 6. Append Row to Master evaluations_history.csv
        csv_row = [
            now.strftime("%Y-%m-%d %H:%M:%S"),
            folder_name,
            patient_tag or clean_tag,
            pred_class,
            round(conf_pct, 2),
            round(float(latency_sec or 0.0), 3),
            age,
            sex,
            spo2,
            smoking,
            cough,
            fever,
            True,
        ]

        with open(self.master_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

        logger.info(f"Successfully archived case evaluation to: {case_dir.resolve()}")
        return case_dir

    @staticmethod
    def _to_pil(img: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """Convert any supported image format to a clean RGB PIL Image."""
        if isinstance(img, (str, Path)):
            return Image.open(str(img)).convert("RGB")
        if isinstance(img, np.ndarray):
            arr = img
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L").convert("RGB")
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return Image.fromarray(arr[:, :, :3], mode="RGB")
            return Image.fromarray(arr).convert("RGB")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        raise ValueError(f"Cannot convert object of type {type(img)} to PIL Image.")


if __name__ == "__main__":
    print("=" * 70)
    print(" RayXplain: CaseArchiver Self-Test Suite")
    print("=" * 70)

    archiver = CaseArchiver()
    print(f"Records Directory : {archiver.records_dir}")
    print(f"Master Audit File : {archiver.master_csv_path}")

    # Generate dummy evaluation
    test_orig = Image.new("RGB", (256, 256), color=(80, 80, 90))
    test_overlay = Image.new("RGB", (256, 256), color=(180, 50, 50))

    saved_path = archiver.save_evaluation(
        original_pil_img=test_orig,
        overlay_pil_img=test_overlay,
        vitals_dict={
            "age": 55,
            "sex": "Male",
            "spo2": 93.0,
            "smoking": 14.0,
            "cough_severity": "Moderate",
            "fever": 101.5,
        },
        prediction_dict={
            "predicted_class": "Pneumonia",
            "confidence": 88.4,
            "probabilities": {"Normal": 0.06, "Pneumonia": 0.884, "High-Risk / COVID-19": 0.056},
        },
        patient_tag="Viva_Test_Sample",
        latency_sec=0.128,
    )

    print(f"\n[PASSED] Archived test case saved at: {saved_path}")
    assert (saved_path / "original_xray.png").is_file(), "Missing original_xray.png"
    assert (saved_path / "gradcam_explanation.png").is_file(), "Missing gradcam_explanation.png"
    assert (saved_path / "clinical_summary.json").is_file(), "Missing clinical_summary.json"
    assert archiver.master_csv_path.is_file(), "Missing evaluations_history.csv"

    print("CaseArchiver operational and ready for deployment.\n")

