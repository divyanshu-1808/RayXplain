"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 4 & Phase 6: Hardened End-to-End Diagnostic Pipeline

This module provides the complete, production-grade inference pipeline for RayXplain:
1. Loads the trained MultiModalLungNet model weights (auto-detecting CUDA/CPU device).
2. Auto-recovers from CUDA Out-of-Memory (OOM) by automatically falling back to CPU.
3. Deserializes and applies the fitted clinical StandardScaler.
4. Robust Image Ingestion: handles 1-channel Grayscale, 4-channel RGBA, 3-channel RGB,
   and catches corrupted/empty files gracefully.
5. Strict Tabular Sanity Checks: SpO2 clamped to [50, 100], Age validated in [1, 110],
   Body Temperature validated in [95.0, 107.0]°F.
6. Proactive Clinical Alerting: flags atypical vitals (critical hypoxemia, high pyrexia, etc.).
7. Integrates GradCAM visual explainability to produce saliency heatmaps overlaid on radiographs.
"""

import io
import logging
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
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Setup module logger
logger = logging.getLogger("RayXplain.Pipeline")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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


def sanitize_image(pil_image: Any) -> Image.Image:
    """
    Validate, sanitize, and convert any valid 2D/3D image into a standardized 3-channel RGB PIL Image.

    Robustly handles:
    - 1-channel Grayscale images (PIL 'L', '1', 'I', 'F', and numpy 2D arrays).
    - 4-channel RGBA images (compositing with a neutral white background to eliminate transparency artifacts).
    - Palette ('P') and CMYK images.
    - Raw numpy arrays (2D grayscale, 3D 1-channel, 3D 3-channel, 3D 4-channel).
    - File paths with corrupted byte streams or unsupported file formats.

    Args:
        pil_image: Image representation (PIL Image, numpy array, or file path).

    Returns:
        Image.Image: Standardized 3-channel RGB PIL Image.

    Raises:
        ValueError: If image is None, empty, corrupted, or unsupported.
    """
    if pil_image is None:
        raise ValueError("Chest radiograph input is required and cannot be None.")

    # 1. Handle file paths (str or Path)
    if isinstance(pil_image, (str, Path)):
        img_path = Path(pil_image)
        if not img_path.is_file():
            raise ValueError(f"Image file not found at path: {img_path}")
        try:
            with Image.open(str(img_path)) as loaded:
                loaded.load()  # Force decode to catch truncated or corrupted files
                pil_image = loaded.copy()
        except (UnidentifiedImageError, OSError, ValueError) as err:
            raise ValueError(f"Corrupted or unsupported image file '{img_path.name}': {err}")

    # 2. Handle raw bytes / bytearrays
    elif isinstance(pil_image, (bytes, bytearray)):
        if len(pil_image) == 0:
            raise ValueError("Empty image byte buffer provided.")
        try:
            with Image.open(io.BytesIO(pil_image)) as loaded:
                loaded.load()
                pil_image = loaded.copy()
        except Exception as err:
            raise ValueError(f"Failed to decode image from byte buffer: {err}")

    # 3. Handle NumPy Arrays
    if isinstance(pil_image, np.ndarray):
        arr = pil_image
        if arr.size == 0:
            raise ValueError("Empty image array provided (size is 0).")

        if arr.ndim == 2:
            # 2D Grayscale: (H, W)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3:
            channels = arr.shape[2]
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)

            if channels == 1:
                # 3D 1-channel Grayscale: (H, W, 1)
                img = Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
            elif channels == 3:
                # 3D 3-channel RGB: (H, W, 3)
                img = Image.fromarray(arr, mode="RGB")
            elif channels == 4:
                # 3D 4-channel RGBA: (H, W, 4) -> composite with white background
                rgba = Image.fromarray(arr, mode="RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[-1])
                img = bg
            else:
                raise ValueError(f"Unsupported image channel count: {channels}. Expected 1, 3, or 4 channels.")
        else:
            raise ValueError(f"Invalid image array dimensions: expected 2D or 3D array, got {arr.ndim}D with shape {arr.shape}.")
        pil_image = img

    # 4. Handle PIL Image Instances
    if isinstance(pil_image, Image.Image):
        w, h = pil_image.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: width={w}, height={h}.")

        # Handle transparency (RGBA, LA, or palette with transparency)
        if pil_image.mode in ("RGBA", "LA") or ("transparency" in pil_image.info) or ("A" in pil_image.getbands()):
            rgba = pil_image.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            return bg

        # 1-channel Grayscale ('L', '1', 'I', 'F') or Palette ('P') or CMYK -> standard 3-channel RGB
        return pil_image.convert("RGB")

    raise ValueError(f"Unsupported image object type: {type(pil_image)}. Expected PIL Image, NumPy array, or file path.")


def check_clinical_vitals(
    age: Any,
    sex: Any,
    spo2: Any,
    smoking: Any,
    cough_severity: Any,
    fever: Any,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Validate and clamp tabular clinical inputs, producing sanitized numeric values and clinical warnings.

    Enforces Phase 6 requirements:
    - SpO2 is clamped between [50.0, 100.0]%.
    - Age is strictly validated within [1, 110] years.
    - Temperature is strictly validated within [95.0, 107.0]°F (auto-converting Celsius if needed).
    - Identifies atypical clinical vitals and returns formatted user-facing alert messages.

    Returns:
        Tuple[Dict[str, float], List[str]]: Sanitized feature dictionary and list of warning messages.

    Raises:
        ValueError: If values are null, empty, non-numeric, or outside valid physiological bounds.
    """
    warnings_list: List[str] = []

    # 1. Null / Empty Checks
    raw_dict = {
        "age": age,
        "sex": sex,
        "spo2": spo2,
        "smoking": smoking,
        "cough_severity": cough_severity,
        "fever": fever,
    }
    for key, val in raw_dict.items():
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError(f"Clinical parameter '{key}' cannot be null or empty.")

    # 2. Age Validation: [1, 110]
    try:
        age_f = float(age)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid age value '{age}'. Must be a valid number.")

    if not (1.0 <= age_f <= 110.0):
        raise ValueError(f"Patient age must be between 1 and 110 years (got {age_f}).")

    if age_f > 75.0:
        warnings_list.append("ℹ️ Geriatric Profile: Patient age > 75 increases susceptibility to severe pulmonary complications.")

    # 3. Biological Sex Parsing
    if isinstance(sex, str):
        sex_clean = sex.strip().lower()
        if sex_clean in ("male", "m", "1", "1.0", "true"):
            sex_val = 1.0
        elif sex_clean in ("female", "f", "0", "0.0", "false"):
            sex_val = 0.0
        else:
            raise ValueError(f"Unrecognized biological sex '{sex}'. Supported: 'Male', 'Female', 1, 0.")
    else:
        sex_val = 1.0 if float(sex) > 0.5 else 0.0

    # 4. SpO2 Validation & Clamping: [50.0, 100.0]
    try:
        raw_spo2 = float(spo2)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid SpO2 value '{spo2}'. Must be a valid numeric percentage.")

    # Auto-scale decimals (e.g. 0.94 -> 94.0%)
    if 0.0 < raw_spo2 <= 1.0:
        raw_spo2 *= 100.0

    # Clamping between [50.0, 100.0]
    spo2_clamped = float(np.clip(raw_spo2, 50.0, 100.0))

    if raw_spo2 < 50.0:
        warnings_list.append(f"⚠️ Atypical SpO2 ({raw_spo2:.1f}%): Value was clamped to physiological floor (50.0%).")
    elif raw_spo2 > 100.0:
        warnings_list.append(f"⚠️ Atypical SpO2 ({raw_spo2:.1f}%): Value was clamped to maximum saturation (100.0%).")

    if spo2_clamped < 90.0:
        warnings_list.append("🚨 Critical Hypoxemia Alert: SpO2 is critically low (<90%). Immediate oxygenation therapy indicated.")

    # 5. Smoking Pack-Years
    if isinstance(smoking, str):
        smk_clean = smoking.strip().lower()
        if smk_clean in ("no", "never", "none", "0"):
            smoking_val = 0.0
        else:
            try:
                smoking_val = float(smk_clean)
            except ValueError:
                raise ValueError(f"Invalid smoking exposure '{smoking}'. Expected nonnegative number or 'None'.")
    else:
        smoking_val = float(smoking)

    if smoking_val < 0.0:
        raise ValueError(f"Smoking pack-years cannot be negative (got {smoking_val}).")

    if smoking_val > 30.0:
        warnings_list.append("ℹ️ Heavy Smoking Exposure: >30 pack-years indicates elevated chronic lung pathology risk.")

    # 6. Cough Severity
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
                raise ValueError(f"Unrecognized cough severity '{cough_severity}'. Supported: 'None', 'Mild', 'Moderate', 'Severe' or 0-3.")
    else:
        cough_val = float(cough_severity)

    # 7. Body Temperature: [95.0, 107.0]°F
    try:
        raw_temp = float(fever)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid temperature value '{fever}'. Must be a valid number.")

    # Celsius conversion (25.0 - 45.0°C)
    if 25.0 <= raw_temp <= 45.0:
        temp_f = raw_temp * 1.8 + 32.0
    elif raw_temp in (0.0, 1.0):
        temp_f = 101.5 if raw_temp == 1.0 else 98.6
    else:
        temp_f = raw_temp

    # Strict boundary check [95.0, 107.0]°F
    if not (95.0 <= temp_f <= 107.0):
        raise ValueError(f"Body temperature {temp_f:.1f}°F is outside the valid physiological range of [95.0, 107.0]°F.")

    if temp_f >= 101.0:
        warnings_list.append(f"⚠️ Pyrexia Alert: Elevated core body temperature detected ({temp_f:.1f}°F >= 101.0°F).")
    elif temp_f <= 96.0:
        warnings_list.append(f"⚠️ Hypothermia Alert: Core body temperature is abnormally subnormal ({temp_f:.1f}°F <= 96.0°F).")

    sanitized = {
        "age": age_f,
        "sex": sex_val,
        "spo2": spo2_clamped,
        "smoking_history": smoking_val,
        "cough_severity": cough_val,
        "fever": temp_f,
    }

    return sanitized, warnings_list


class DiagnosticPipeline:
    """
    Hardened multimodal diagnostic assessment pipeline for pulmonary disease prediction.

    Coordinates chest X-ray image transforms, clinical tabular parameter scaling,
    multimodal model inference, and Grad-CAM saliency explanation generation.
    Equipped with automatic CUDA OOM fallback to CPU.

    Attributes:
        device (torch.device): Compute device ('cuda' if available, otherwise 'cpu').
        model (MultiModalLungNet): The trained deep learning network in eval mode.
        scaler (Any): The fitted scikit-learn StandardScaler for tabular features.
        gradcam (GradCAM): Hooked Grad-CAM explainability engine.
        image_transform (transforms.Compose): Image preprocessing pipeline.
        last_warnings (List[str]): Cached clinical warning messages from most recent prediction.
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

        logger.info(f"Loading model on device: {self.device} from {self.model_path.name}")
        self.model = build_model(
            pretrained=False,
            weights_path=self.model_path,
            device=self.device,
        )
        self.model.eval()

        # 4. Load Scaler
        if not self.scaler_path.is_file():
            raise FileNotFoundError(f"Clinical scaler artifact not found at: {self.scaler_path.resolve()}")

        logger.info(f"Loading scaler from: {self.scaler_path.name}")
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
        logger.info(f"Attached GradCAM to {self.gradcam.target_layer_name}")

        self.last_warnings: List[str] = []

    def fallback_to_cpu(self) -> None:
        """
        Gracefully migrate model, GradCAM, and execution state from CUDA to CPU upon Out-of-Memory (OOM).
        """
        logger.warning("[DiagnosticPipeline] CUDA Out-of-Memory detected! Initiating emergency CPU migration...")
        warnings.warn("[DiagnosticPipeline] CUDA Out of Memory! Automatically falling back to CPU.", RuntimeWarning)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Re-initialize GradCAM on CPU
        if hasattr(self, "gradcam") and self.gradcam is not None:
            self.gradcam.remove_hooks()
            self.gradcam = GradCAM(model=self.model, target_layer_name="layer4")

        logger.info("[DiagnosticPipeline] Emergency migration complete: pipeline operational on CPU.")

    def transform_image(self, pil_image: Any) -> torch.Tensor:
        """
        Preprocess any valid radiograph input into a normalized model tensor.

        Args:
            pil_image: PIL Image, RGB/RGBA/Grayscale numpy array, or file path.

        Returns:
            torch.Tensor: Normalized tensor of shape [1, 3, 224, 224] on self.device.
        """
        clean_pil = sanitize_image(pil_image)
        tensor = self.image_transform(clean_pil)  # Shape: [3, 224, 224]
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

        Returns:
            torch.Tensor: Standardized tabular tensor of shape [1, 6] on self.device.
        """
        sanitized, _ = check_clinical_vitals(
            age=age,
            sex=sex,
            spo2=spo2,
            smoking=smoking_history,
            cough_severity=cough_severity,
            fever=fever,
        )

        cols = list(self.feature_names)
        df_row = pd.DataFrame(
            [[sanitized["age"], sanitized["sex"], sanitized["spo2"], sanitized["smoking_history"], sanitized["cough_severity"], sanitized["fever"]]],
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
    ) -> Tuple[Image.Image, Dict[str, float], List[str]]:
        """
        Validate nulls, types, domain boundaries, and image integrity for all modalities.

        Returns:
            Tuple[Image.Image, Dict[str, float], List[str]]:
                1. Sanitized RGB PIL Image.
                2. Sanitized numeric clinical parameters dictionary.
                3. List of atypical clinical alert messages.
        """
        clean_pil = sanitize_image(pil_image)
        sanitized_vitals, warnings_list = check_clinical_vitals(
            age=age,
            sex=sex,
            spo2=spo2,
            smoking=smoking,
            cough_severity=cough_severity,
            fever=fever,
        )
        return clean_pil, sanitized_vitals, warnings_list

    def predict(
        self,
        pil_image: Any,
        age: Union[int, float],
        sex: Union[str, int, float],
        spo2: Union[int, float],
        smoking: Union[int, float, str] = 0.0,
        cough_severity: Union[int, float, str] = 0.0,
        fever: Union[int, float, bool] = 98.6,
        alpha: float = 0.4,
        return_warnings: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[np.ndarray, Dict[str, float], str], Tuple[np.ndarray, Dict[str, float], str, List[str]]]:
        """
        Execute full multimodal diagnostic inference and Grad-CAM explanation with CUDA OOM recovery.

        Args:
            pil_image: Input chest radiograph (PIL Image, RGB/RGBA/Grayscale array, or file path).
            age: Patient age in years ([1, 110]).
            sex: Sex category ("Male", "Female", 1, 0).
            spo2: Blood oxygen saturation percentage (clamped to [50, 100]).
            smoking: Cumulative smoking pack-years (or smoking_history alias).
            cough_severity: Severity rating of cough symptoms (0-3 or 'Mild'/'Moderate'/'Severe').
            fever: Body temperature in [95.0, 107.0]°F (or °C auto-converted).
            alpha: Transparency weight for Grad-CAM overlay (default: 0.4).
            return_warnings: If True, returns (overlay, probs, pred_name, warnings_list).

        Returns:
            Tuple[np.ndarray, Dict[str, float], str] (or 4-tuple with warnings):
                1. overlay_image: Blended RGB numpy array of shape (H, W, 3), dtype uint8.
                2. probabilities_dict: Mapping class names to softmax probabilities.
                3. predicted_class_name: The highest probability diagnosis name.
                (4. Optional warnings_list: List of atypical vitals alert messages).

        Raises:
            ValueError: If inputs fail validation checks.
        """
        # Support keyword alias 'smoking_history' if passed
        actual_smoking = kwargs.get("smoking_history", smoking)

        # 1. Comprehensive Input Validation & Sanitization
        clean_pil, sanitized_vitals, warnings_list = self.validate_inputs(
            pil_image=pil_image,
            age=age,
            sex=sex,
            spo2=spo2,
            smoking=actual_smoking,
            cough_severity=cough_severity,
            fever=fever,
        )
        self.last_warnings = warnings_list

        # 2. Transform Image and Clinical inputs
        img_tensor = self.transform_image(clean_pil)
        clin_tensor = self.transform_tabular(
            age=sanitized_vitals["age"],
            sex=sanitized_vitals["sex"],
            spo2=sanitized_vitals["spo2"],
            smoking_history=sanitized_vitals["smoking_history"],
            cough_severity=sanitized_vitals["cough_severity"],
            fever=sanitized_vitals["fever"],
        )

        # 3. Model Forward Pass & Probability Computation with CUDA OOM Fallback
        try:
            with torch.no_grad():
                logits, _ = self.model(img_tensor, clin_tensor)
                probs = F.softmax(logits, dim=1)[0]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            is_oom = "out of memory" in str(err).lower() or isinstance(err, getattr(torch.cuda, "OutOfMemoryError", ()))
            if is_oom and self.device.type == "cuda":
                self.fallback_to_cpu()
                img_tensor = img_tensor.to("cpu")
                clin_tensor = clin_tensor.to("cpu")
                with torch.no_grad():
                    logits, _ = self.model(img_tensor, clin_tensor)
                    probs = F.softmax(logits, dim=1)[0]
            else:
                raise

        # 4. Extract Diagnostic Predictions
        pred_idx = int(torch.argmax(probs).item())
        predicted_class_name = TARGET_CLASSES[pred_idx]

        probabilities_dict: Dict[str, float] = {
            TARGET_CLASSES[i]: round(float(probs[i].item()), 4)
            for i in range(len(TARGET_CLASSES))
        }

        # 5. Compute Grad-CAM Saliency & Alpha Overlay with CUDA OOM Fallback
        try:
            heatmap = self.gradcam.generate_heatmap(
                img_tensor=img_tensor,
                clin_tensor=clin_tensor,
                target_class=pred_idx,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            is_oom = "out of memory" in str(err).lower() or isinstance(err, getattr(torch.cuda, "OutOfMemoryError", ()))
            if is_oom and self.device.type == "cuda":
                self.fallback_to_cpu()
                img_tensor = img_tensor.to("cpu")
                clin_tensor = clin_tensor.to("cpu")
                heatmap = self.gradcam.generate_heatmap(
                    img_tensor=img_tensor,
                    clin_tensor=clin_tensor,
                    target_class=pred_idx,
                )
            else:
                raise

        overlay_image = self.gradcam.overlay_heatmap(
            heatmap=heatmap,
            original_pil_img=clean_pil,
            alpha=alpha,
            colormap=cv2.COLORMAP_JET,
        )

        if return_warnings:
            return overlay_image, probabilities_dict, predicted_class_name, warnings_list

        return overlay_image, probabilities_dict, predicted_class_name


if __name__ == "__main__":
    print("=" * 70)
    print(" RayXplain Phase 6: Hardened DiagnosticPipeline Verification")
    print("=" * 70)

    # 1. Instantiate Pipeline
    print("\n[1/5] Instantiating Pipeline...")
    pipeline = DiagnosticPipeline()
    print(f"      Initial Device: {pipeline.device}")

    # 2. Test Image Robustness: Grayscale & RGBA
    print("\n[2/5] Testing Image Robustness (Grayscale & 4-channel RGBA)...")
    gray_arr = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
    rgba_arr = np.random.randint(50, 200, (300, 300, 4), dtype=np.uint8)

    t_gray = pipeline.transform_image(gray_arr)
    t_rgba = pipeline.transform_image(rgba_arr)
    assert t_gray.shape == (1, 3, 224, 224), "Failed grayscale transform"
    assert t_rgba.shape == (1, 3, 224, 224), "Failed RGBA transform"
    print("      [PASSED] Grayscale and RGBA standardized to 3-channel RGB [1, 3, 224, 224]")

    # 3. Test Tabular Clamping & Boundaries
    print("\n[3/5] Testing Tabular Clamping & Clinical Sanity...")
    # Test SpO2 clamping: 35% -> clamped to 50%
    sanitized, warns = check_clinical_vitals(
        age=50,
        sex="Male",
        spo2=35.0,
        smoking=0,
        cough_severity="None",
        fever=98.6,
    )
    assert sanitized["spo2"] == 50.0, f"Expected SpO2 clamped to 50.0, got {sanitized['spo2']}"
    assert any("clamped" in w.lower() for w in warns), "Missing clamping warning"
    print(f"      [PASSED] SpO2 35% clamped to {sanitized['spo2']}% with warning")

    # 4. Test Error Boundaries: Age & Temperature
    print("\n[4/5] Testing Strict Boundary Exceptions (Age & Temp)...")
    try:
        check_clinical_vitals(age=120, sex="Male", spo2=95, smoking=0, cough_severity=0, fever=98.6)
        assert False, "Failed to catch Age > 110"
    except ValueError:
        print("      [PASSED] Caught invalid Age > 110")

    try:
        check_clinical_vitals(age=50, sex="Male", spo2=95, smoking=0, cough_severity=0, fever=92.0)
        assert False, "Failed to catch Temp < 95.0°F"
    except ValueError:
        print("      [PASSED] Caught invalid Temp < 95.0°F")

    # 5. Test CUDA Fallback
    print("\n[5/5] Testing System Fallback to CPU...")
    pipeline.fallback_to_cpu()
    assert pipeline.device.type == "cpu", "Failed to switch device to CPU"
    res_overlay, res_probs, res_pred = pipeline.predict(
        pil_image=gray_arr,
        age=45,
        sex="Female",
        spo2=96,
        smoking=0,
        cough_severity="None",
        fever=98.6,
    )
    assert res_pred in TARGET_CLASSES
    print("      [PASSED] CPU fallback executed successfully without crash")

    print("\n" + "=" * 70)
    print(" Phase 6 Hardened Pipeline VERIFIED SUCCESSFULLY!")
    print("=" * 70)
