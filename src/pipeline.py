"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 4: End-to-End Diagnostic Pipeline (Model + Preprocessing + Grad-CAM)

This module provides the complete inference pipeline for RayXplain:
1. Loads the trained MultiModalLungNet model weights (auto-detecting CUDA/CPU device).
2. Loads and applies the fitted clinical StandardScaler.
3. Implements standard radiograph preprocessing (Resize, ToTensor, ImageNet normalization).
4. Integrates GradCAM visual explainability to produce saliency heatmaps overlaid on radiographs.
5. Provides a unified, validated `predict()` method returning class probabilities, predictions, and overlays.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Suppress harmless sklearn unpickling warnings if minor version discrepancies exist
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from .model import CLASS_NAMES, CLINICAL_FEATURE_NAMES, MultiModalLungNet, build_model
    from .gradcam import GradCAM, overlay_heatmap
except ImportError:
    from src.model import CLASS_NAMES, CLINICAL_FEATURE_NAMES, MultiModalLungNet, build_model
    from src.gradcam import GradCAM, overlay_heatmap


# Diagnostic Target Class Names formatted for web & clinical reporting
TARGET_CLASSES: List[str] = [
    "Normal",
    "Pneumonia",
    "High-Risk / COVID-19",
]

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DiagnosticPipeline:
    """
    Unified multimodal diagnostic assessment pipeline for pulmonary disease prediction.

    Coordinates chest X-ray image transforms, clinical tabular parameter scaling,
    multimodal model inference, and Grad-CAM saliency explanation generation.

    Attributes:
        device (torch.device): Compute device ('cuda' if available, otherwise 'cpu').
        model (MultiModalLungNet): The trained deep learning network in eval mode.
        scaler (Any): The fitted scikit-learn StandardScaler for tabular features.
        gradcam (GradCAM): Hooked Grad-CAM explainability engine.
        image_transform (transforms.Compose): Image preprocessing pipeline.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the diagnostic pipeline, load model checkpoint, scaler, and GradCAM engine.

        Args:
            model_path: Path to model checkpoint file (.pth). Defaults to weights/best_multimodal_model.pth.
            scaler_path: Path to serialized scaler (.joblib). Defaults to data/scaler.joblib.
            device: Compute device ('cuda', 'cpu', or torch.device). Auto-detected if None.
        """
        # 1. Device configuration
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Resolve default artifact paths
        default_model = _REPO_ROOT / "weights" / "best_multimodal_model.pth"
        default_scaler = _REPO_ROOT / "data" / "scaler.joblib"

        self.model_path = Path(model_path) if model_path is not None else default_model
        self.scaler_path = Path(scaler_path) if scaler_path is not None else default_scaler

        # 3. Load Model
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path.resolve()}")

        print(f"[DiagnosticPipeline] Loading model on device: {self.device} from {self.model_path.name}")
        self.model = build_model(
            pretrained=False,
            weights_path=self.model_path,
            device=self.device,
        )
        self.model.eval()

        # 4. Load Scaler
        if not self.scaler_path.is_file():
            raise FileNotFoundError(f"Clinical scaler artifact not found at: {self.scaler_path.resolve()}")

        print(f"[DiagnosticPipeline] Loading scaler from: {self.scaler_path.name}")
        self.scaler = joblib.load(str(self.scaler_path))

        # Expected feature names from scaler or fallback standard
        self.feature_names = getattr(
            self.scaler,
            "feature_names_in_",
            ["age", "sex", "spo2", "smoking_history", "cough_severity", "fever"],
        )

        # 5. Initialize Image Preprocessing Transform
        # Resize to (224, 224) -> ToTensor() -> ImageNet Normalization
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # 6. Attach GradCAM instance to target convolutional stage (layer4)
        self.gradcam = GradCAM(model=self.model, target_layer_name="layer4")
        print(f"[DiagnosticPipeline] Attached GradCAM to {self.gradcam.target_layer_name}")

    def transform_image(self, pil_image: Union[Image.Image, np.ndarray, str, Path]) -> torch.Tensor:
        """
        Preprocess a chest radiograph for model ingestion.

        Process:
        1. Convert image to PIL RGB format.
        2. Resize to (224, 224).
        3. Convert pixel values to [0.0, 1.0] tensor.
        4. Apply ImageNet channel standardization.
        5. Add batch dimension and transfer to device.

        Args:
            pil_image: PIL Image, RGB numpy array, or file path.

        Returns:
            torch.Tensor: Normalized tensor of shape [1, 3, 224, 224] on self.device.
        """
        if isinstance(pil_image, (str, Path)):
            img = Image.open(str(pil_image)).convert("RGB")
        elif isinstance(pil_image, np.ndarray):
            if pil_image.ndim == 2:
                img = Image.fromarray(pil_image).convert("RGB")
            elif pil_image.ndim == 3 and pil_image.shape[2] == 1:
                img = Image.fromarray(pil_image.squeeze(2)).convert("RGB")
            else:
                img = Image.fromarray(pil_image[:, :, :3]).convert("RGB")
        elif isinstance(pil_image, Image.Image):
            img = pil_image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(pil_image)}")

        tensor = self.image_transform(img)  # Shape: [3, 224, 224]
        tensor = tensor.unsqueeze(0).to(self.device)  # Shape: [1, 3, 224, 224]
        return tensor

    def transform_tabular(
        self,
        age: Union[int, float],
        sex: Union[str, int, float],
        spo2: Union[int, float],
        smoking_history: Union[int, float, str],
        cough_severity: Union[int, float, str],
        fever: Union[int, float, bool],
    ) -> torch.Tensor:
        """
        Preprocess and standardize clinical tabular input parameters.

        Process:
        1. Parse and standardize categorical and numeric values.
        2. Construct structured 2D DataFrame matching scaler feature schema.
        3. Transform values via loaded StandardScaler.
        4. Convert output to float32 tensor on self.device.

        Args:
            age: Patient age in years (0 - 120).
            sex: Sex category ("Male", "Female", 1, 0, etc.).
            spo2: Blood oxygen saturation percentage (0 - 100).
            smoking_history: Cumulative smoking pack-years (nonnegative float).
            cough_severity: Cough severity scale (0 - 3 or "None", "Mild", "Moderate", "Severe").
            fever: Body temperature (in °F, or °C converted to °F, or binary).

        Returns:
            torch.Tensor: Standardized tabular tensor of shape [1, 6] on self.device.
        """
        # Encode sex: Female=0.0, Male=1.0
        if isinstance(sex, str):
            sex_clean = sex.strip().lower()
            if sex_clean in ("male", "m", "1", "1.0", "true"):
                sex_val = 1.0
            elif sex_clean in ("female", "f", "0", "0.0", "false"):
                sex_val = 0.0
            else:
                raise ValueError(f"Unrecognized sex category '{sex}'. Supported: 'Male', 'Female', 1, 0.")
        else:
            sex_val = 1.0 if float(sex) > 0.5 else 0.0

        # Encode cough severity
        if isinstance(cough_severity, str):
            cough_clean = cough_severity.strip().lower()
            cough_map = {
                "none": 0.0,
                "no": 0.0,
                "normal": 0.0,
                "mild": 1.0,
                "moderate": 2.0,
                "severe": 3.0,
                "high": 3.0,
            }
            if cough_clean in cough_map:
                cough_val = cough_map[cough_clean]
            else:
                try:
                    cough_val = float(cough_clean)
                except ValueError:
                    raise ValueError(
                        f"Unrecognized cough severity '{cough_severity}'. "
                        "Supported: 'None', 'Mild', 'Moderate', 'Severe' or numeric 0-3."
                    )
        else:
            cough_val = float(cough_severity)

        # Parse smoking pack-years
        if isinstance(smoking_history, str):
            smk_clean = smoking_history.strip().lower()
            if smk_clean in ("no", "never", "none"):
                smoking_val = 0.0
            else:
                try:
                    smoking_val = float(smk_clean)
                except ValueError:
                    raise ValueError(f"Invalid smoking exposure '{smoking_history}'. Expected nonnegative number.")
        else:
            smoking_val = float(smoking_history)

        # Parse temperature / fever
        fever_num = float(fever)
        # If entered in Celsius (< 48°C), convert to Fahrenheit
        if 25.0 <= fever_num <= 48.0:
            temp_f = fever_num * (9.0 / 5.0) + 32.0
        elif fever_num in (0.0, 1.0):
            # Binary flag: 0 -> normal baseline (98.6°F), 1 -> elevated fever (101.5°F)
            temp_f = 101.5 if fever_num == 1.0 else 98.6
        else:
            temp_f = fever_num

        age_val = float(age)
        spo2_val = float(spo2)

        # Construct single-sample DataFrame with exact fitted feature column names
        cols = list(self.feature_names)
        df_row = pd.DataFrame(
            [[age_val, sex_val, spo2_val, smoking_val, cough_val, temp_f]],
            columns=cols,
        )

        scaled_np = self.scaler.transform(df_row).astype(np.float32)
        tabular_tensor = torch.tensor(scaled_np, dtype=torch.float32, device=self.device)
        return tabular_tensor

    def validate_inputs(
        self,
        pil_image: Any,
        age: Any,
        sex: Any,
        spo2: Any,
        smoking: Any,
        cough_severity: Any,
        fever: Any,
    ) -> None:
        """
        Validate nulls, types, and domain range boundaries for all input modalities.

        Raises:
            ValueError: If any input is None, invalid, or out of clinical bounds.
        """
        # 1. Image validation
        if pil_image is None:
            raise ValueError("Input image is required and cannot be None.")

        # 2. Null check for clinical inputs
        params = {
            "age": age,
            "sex": sex,
            "spo2": spo2,
            "smoking": smoking,
            "cough_severity": cough_severity,
            "fever": fever,
        }
        for name, val in params.items():
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError(f"Clinical input '{name}' cannot be None or empty.")

        # 3. Numeric bounds validation
        try:
            age_f = float(age)
            if not (0.0 <= age_f <= 125.0):
                raise ValueError(f"Age must be between 0 and 125 years (got {age_f}).")
        except (ValueError, TypeError) as e:
            if "between 0 and 125" in str(e):
                raise
            raise ValueError(f"Invalid age value: '{age}'. Must be a valid number.")

        try:
            spo2_f = float(spo2)
            # If percentage given in 0.0 - 1.0, allow or check 0 - 100
            if spo2_f <= 1.0 and spo2_f > 0.0:
                spo2_f *= 100.0
            if not (40.0 <= spo2_f <= 100.0):
                raise ValueError(f"SpO2 oxygen saturation must be between 40% and 100% (got {spo2_f}%).")
        except (ValueError, TypeError) as e:
            if "between 40%" in str(e):
                raise
            raise ValueError(f"Invalid SpO2 value: '{spo2}'. Must be a valid percentage.")

        try:
            if not isinstance(smoking, str) or smoking.strip().lower() not in ("no", "never", "none"):
                smk_f = float(smoking)
                if smk_f < 0.0:
                    raise ValueError(f"Smoking pack-years cannot be negative (got {smk_f}).")
        except (ValueError, TypeError) as e:
            if "negative" in str(e):
                raise
            raise ValueError(f"Invalid smoking exposure value: '{smoking}'.")

        try:
            fever_f = float(fever)
            # Valid temperatures: 25 - 48 °C or 80 - 118 °F, or binary 0/1
            if fever_f in (0.0, 1.0):
                pass
            elif 25.0 <= fever_f <= 48.0 or 80.0 <= fever_f <= 118.0:
                pass
            else:
                raise ValueError(f"Body temperature {fever_f} is outside plausible clinical range (80-118°F or 25-48°C).")
        except (ValueError, TypeError) as e:
            if "plausible clinical range" in str(e):
                raise
            raise ValueError(f"Invalid body temperature value: '{fever}'. Must be a valid number.")

    def predict(
        self,
        pil_image: Union[Image.Image, np.ndarray, str, Path],
        age: Union[int, float],
        sex: Union[str, int, float],
        spo2: Union[int, float],
        smoking: Union[int, float, str] = 0.0,
        cough_severity: Union[int, float, str] = 0.0,
        fever: Union[int, float, bool] = 98.6,
        alpha: float = 0.4,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, float], str]:
        """
        Execute full multimodal diagnostic inference and Grad-CAM explanation.

        Args:
            pil_image: Input chest radiograph (PIL Image, RGB numpy array, or file path).
            age: Patient age in years.
            sex: Sex category ("Male", "Female", 1, 0).
            spo2: Blood oxygen saturation percentage (SpO2).
            smoking: Cumulative smoking pack-years (or smoking_history alias).
            cough_severity: Severity rating of cough symptoms (0-3 or 'Mild'/'Moderate'/'Severe').
            fever: Body temperature (in °F, °C, or binary).
            alpha: Transparency weight for Grad-CAM overlay (default: 0.4).

        Returns:
            Tuple[np.ndarray, Dict[str, float], str]:
                1. overlay_image: Blended RGB numpy array of shape (H, W, 3), dtype uint8.
                2. probabilities_dict: Mapping class names to softmax probabilities.
                   e.g. {'Normal': 0.85, 'Pneumonia': 0.10, 'High-Risk / COVID-19': 0.05}.
                3. predicted_class_name: The highest probability diagnosis name.

        Raises:
            ValueError: If inputs fail validation checks.
        """
        # Support keyword alias 'smoking_history' if passed
        actual_smoking = kwargs.get("smoking_history", smoking)

        # 1. Validate inputs
        self.validate_inputs(
            pil_image=pil_image,
            age=age,
            sex=sex,
            spo2=spo2,
            smoking=actual_smoking,
            cough_severity=cough_severity,
            fever=fever,
        )

        # Keep original image for overlay
        if isinstance(pil_image, (str, Path)):
            orig_pil = Image.open(str(pil_image)).convert("RGB")
        elif isinstance(pil_image, np.ndarray):
            orig_pil = Image.fromarray(pil_image).convert("RGB")
        else:
            orig_pil = pil_image.convert("RGB")

        # 2. Transform Image and Clinical inputs
        img_tensor = self.transform_image(orig_pil)
        clin_tensor = self.transform_tabular(
            age=age,
            sex=sex,
            spo2=spo2,
            smoking_history=actual_smoking,
            cough_severity=cough_severity,
            fever=fever,
        )

        # 3. Model Forward Pass & Probability Computation
        with torch.no_grad():
            logits, _ = self.model(img_tensor, clin_tensor)
            probs = F.softmax(logits, dim=1)[0]

        # 4. Extract Diagnostic Predictions
        pred_idx = int(torch.argmax(probs).item())
        predicted_class_name = TARGET_CLASSES[pred_idx]

        probabilities_dict: Dict[str, float] = {
            TARGET_CLASSES[i]: round(float(probs[i].item()), 4)
            for i in range(len(TARGET_CLASSES))
        }

        # 5. Compute Grad-CAM Saliency & Alpha Overlay
        heatmap = self.gradcam.generate_heatmap(
            img_tensor=img_tensor,
            clin_tensor=clin_tensor,
            target_class=pred_idx,
        )

        overlay_image = self.gradcam.overlay_heatmap(
            heatmap=heatmap,
            original_pil_img=orig_pil,
            alpha=alpha,
            colormap=cv2.COLORMAP_JET,
        )

        return overlay_image, probabilities_dict, predicted_class_name


if __name__ == "__main__":
    print("=" * 70)
    print(" RayXplain Phase 4: DiagnosticPipeline Verification Suite")
    print("=" * 70)

    # 1. Instantiate Pipeline
    print("\n[1/4] Instantiating DiagnosticPipeline...")
    pipeline = DiagnosticPipeline()
    print(f"      Execution Device    : {pipeline.device}")
    print(f"      Model Parameters    : {sum(p.numel() for p in pipeline.model.parameters()):,}")
    print(f"      Scaler Features     : {list(pipeline.feature_names)}")

    # 2. Generate Synthetic Chest Radiograph & Clinical Data
    print("\n[2/4] Generating synthetic patient inputs...")
    synthetic_img = Image.new("RGB", (384, 384), color=(80, 85, 90))
    test_age = 58
    test_sex = "Male"
    test_spo2 = 91.5
    test_smoking = 15.0
    test_cough = "Moderate"
    test_fever = 101.8  # Elevated fever in °F

    print(f"      Simulated Radiograph: {synthetic_img.size} (W, H)")
    print(f"      Age: {test_age}, Sex: '{test_sex}', SpO2: {test_spo2}%, Smoking: {test_smoking} pk-yrs")
    print(f"      Cough: '{test_cough}', Fever: {test_fever}°F")

    # 3. Execute Pipeline Predict
    print("\n[3/4] Running predict() end-to-end...")
    overlay, probs, pred_name = pipeline.predict(
        pil_image=synthetic_img,
        age=test_age,
        sex=test_sex,
        spo2=test_spo2,
        smoking=test_smoking,
        cough_severity=test_cough,
        fever=test_fever,
    )

    print(f"      Predicted Diagnosis : {pred_name}")
    print(f"      Class Probabilities : {probs}")
    print(f"      Overlay Image Shape : {overlay.shape}")
    print(f"      Overlay Image Dtype : {overlay.dtype}")
    print(f"      Overlay Pixel Range : [{overlay.min()}, {overlay.max()}]")

    # 4. Strict Assertions
    print("\n[4/4] Validating outputs & constraints...")
    assert pred_name in TARGET_CLASSES, f"Unknown prediction: {pred_name}"
    assert set(probs.keys()) == set(TARGET_CLASSES), "Mismatched probability keys"
    assert abs(sum(probs.values()) - 1.0) < 1e-2, f"Probabilities don't sum to 1.0: {sum(probs.values())}"
    assert overlay.shape == (384, 384, 3), f"Overlay shape mismatch: {overlay.shape}"
    assert overlay.dtype == np.uint8, f"Expected uint8 overlay, got {overlay.dtype}"
    assert overlay.min() >= 0 and overlay.max() <= 255, "Overlay pixel values outside [0, 255]"

    # Test input validation errors
    try:
        pipeline.predict(pil_image=None, age=50, sex="Male", spo2=95, smoking=0, cough_severity=0, fever=98.6)
        assert False, "Failed to catch None image"
    except ValueError:
        pass

    try:
        pipeline.predict(pil_image=synthetic_img, age=-5, sex="Male", spo2=95, smoking=0, cough_severity=0, fever=98.6)
        assert False, "Failed to catch negative age"
    except ValueError:
        pass

    print("      [PASSED] Output shapes, dtypes, and probability constraints verified")
    print("      [PASSED] Input range & null validation verified")
    print("      [PASSED] Grad-CAM overlay dimensions perfectly match original radiograph")

    print("\n" + "=" * 70)
    print(" DiagnosticPipeline Phase 4 Implementation VERIFIED SUCCESSFULLY!")
    print("=" * 70)

