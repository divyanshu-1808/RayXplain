"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 3: Explainable AI Engine - Gradient-Weighted Class Activation Mapping (Grad-CAM)

This module implements the Grad-CAM explainability pipeline (Selvaraju et al., 2017)
tailored for the dual-branch MultiModalLungNet architecture. It hooks into the final
convolutional stage (ResNet-50 `layer4`) to capture spatial feature activations and their
associated class-specific gradients, generating localized saliency heatmaps and visual
alpha-blended overlays for clinical transparency.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path if not present for standalone execution
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Import class names and model definitions from sibling module
try:
    from .model import CLASS_NAMES, MultiModalLungNet
except ImportError:
    try:
        from src.model import CLASS_NAMES, MultiModalLungNet
    except ImportError:
        from model import CLASS_NAMES, MultiModalLungNet


def _locate_target_layer(
    model: nn.Module,
    target_layer_name: Union[str, nn.Module, None] = "layer4",
) -> Tuple[nn.Module, str]:
    """
    Resolve and locate the target convolutional layer within a neural network.

    Supports:
    - Direct nn.Module reference.
    - None (defaults to model.get_conv_layer() or conv_base[7] / layer4).
    - String layer names ("layer4", "conv_base.7", "conv_base[7]", etc.).
    - Dot-separated submodule paths.

    Args:
        model: The PyTorch neural network module.
        target_layer_name: Layer identifier or nn.Module instance.

    Returns:
        Tuple[nn.Module, str]: The target layer module and its canonical name.

    Raises:
        ValueError: If the target layer cannot be found in the model architecture.
    """
    if isinstance(target_layer_name, nn.Module):
        # Scan named modules to find its canonical name
        for name, module in model.named_modules():
            if module is target_layer_name:
                return module, name
        return target_layer_name, "custom_module"

    # Default / 'layer4' resolution for MultiModalLungNet
    if target_layer_name is None or str(target_layer_name).strip().lower() in ("layer4", "none"):
        if hasattr(model, "get_conv_layer") and callable(getattr(model, "get_conv_layer")):
            return model.get_conv_layer(), "conv_base.7 (layer4)"
        if hasattr(model, "conv_base") and isinstance(model.conv_base, nn.Sequential) and len(model.conv_base) > 7:
            return model.conv_base[7], "conv_base.7 (layer4)"
        if hasattr(model, "layer4"):
            return getattr(model, "layer4"), "layer4"

    name_str = str(target_layer_name).strip()

    # Direct match in named_modules dict
    named_modules = dict(model.named_modules())
    if name_str in named_modules:
        return named_modules[name_str], name_str

    # MultiModalLungNet conv_base alias handling (e.g. 'conv_base.7' or 'conv_base[7]')
    if name_str in ("conv_base[7]", "conv_base.7", "conv_base.layer4"):
        if hasattr(model, "conv_base") and len(model.conv_base) > 7:
            return model.conv_base[7], "conv_base.7"

    # Try prepending 'conv_base.'
    candidate = f"conv_base.{name_str}"
    if candidate in named_modules:
        return named_modules[candidate], candidate

    # Suffix matching (e.g. '.layer4' matching any submodule named layer4)
    suffix_matches = [
        (k, m) for k, m in named_modules.items()
        if k == name_str or k.endswith(f".{name_str}") or k.endswith(name_str)
    ]
    if suffix_matches:
        # Return the deepest match
        chosen_name, chosen_mod = suffix_matches[-1]
        return chosen_mod, chosen_name

    # Attribute dot-traversal fallback
    try:
        curr = model
        for part in name_str.split("."):
            if "[" in part and part.endswith("]"):
                attr_name, idx_str = part[:-1].split("[")
                curr = getattr(curr, attr_name)[int(idx_str)]
            else:
                curr = getattr(curr, part)
        if isinstance(curr, nn.Module):
            return curr, name_str
    except Exception:
        pass

    # Fallback: Locate last Conv2d layer in the model ONLY when no target name was specified
    if target_layer_name is None:
        last_conv_name, last_conv_module = None, None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name, last_conv_module = name, module

        if last_conv_module is not None:
            return last_conv_module, f"{last_conv_name} (auto-fallback Conv2d)"

    available_layers = [name for name, _ in model.named_modules() if name]
    raise ValueError(
        f"Target layer '{target_layer_name}' not found in model hierarchy. "
        f"Available modules include: {available_layers[:15]} ... (total {len(available_layers)} modules)."
    )


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) Engine for MultiModalLungNet.

    Captures forward activation maps and backpropagates class-specific gradients to produce
    coarse 2D visual localization heatmaps highlighting discriminative pulmonary regions.

    Attributes:
        model (nn.Module): The multimodal neural network under inspection.
        target_layer (nn.Module): The target convolutional module (e.g. ResNet50 layer4).
        target_layer_name (str): Identifier name of the hooked layer.
        activations (Optional[torch.Tensor]): Cached forward feature maps [B, C, H, W].
        gradients (Optional[torch.Tensor]): Cached backward gradient maps [B, C, H, W].
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: Union[str, nn.Module, None] = "layer4",
    ) -> None:
        """
        Initialize the GradCAM explainability engine and register forward/backward hooks.

        Args:
            model (nn.Module): Pretrained or initialized neural network.
            target_layer_name (Union[str, nn.Module, None]): Layer name or module to hook into.
                Defaults to 'layer4' (final ResNet50 convolutional stage).
        """
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._backward_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self.target_layer, self.target_layer_name = _locate_target_layer(model, target_layer_name)

        # Hook registration
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target convolutional layer."""
        self.remove_hooks()

        def _forward_hook(module: nn.Module, input_tensor: Any, output_tensor: Any) -> None:
            # Handle tuple/list returns if applicable
            out = output_tensor[0] if isinstance(output_tensor, (tuple, list)) else output_tensor
            self.activations = out.detach()

            # Register tensor-level backward hook on the activation tensor
            if out.requires_grad:
                out.register_hook(self._save_gradient)

        def _backward_hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(_forward_hook)
        try:
            self._backward_handle = self.target_layer.register_full_backward_hook(_backward_hook)
        except Exception:
            # Fallback for environments where full backward hook registration on container is unsupported
            self._backward_handle = None

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Tensor backward hook callback storing incoming gradients."""
        self.gradients = grad.detach()

    def remove_hooks(self) -> None:
        """Safely unregister existing forward and backward hooks."""
        forward_handle = getattr(self, "_forward_handle", None)
        if forward_handle is not None:
            forward_handle.remove()
            self._forward_handle = None

        backward_handle = getattr(self, "_backward_handle", None)
        if backward_handle is not None:
            backward_handle.remove()
            self._backward_handle = None

    def __enter__(self) -> "GradCAM":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: automatically clean up hooks."""
        self.remove_hooks()

    def __del__(self) -> None:
        """Destructor cleanup."""
        try:
            self.remove_hooks()
        except Exception:
            pass

    def generate_heatmap(
        self,
        img_tensor: torch.Tensor,
        clin_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a normalized 2D Grad-CAM saliency heatmap for the specified class.

        Process:
        1. Forward pass through model in evaluation mode with gradient tracking enabled.
        2. Target class logit extraction (model predicted argmax if None).
        3. Backward pass of target logit score to layer4 convolutional feature maps.
        4. Global Average Pooling (GAP) on gradients to compute channel-wise weights alpha_k.
        5. Linear combination of activation maps weighted by alpha_k, followed by ReLU.
        6. Min-max normalization of heatmap values into [0.0, 1.0].
        7. Strict cleanup of gradient buffers and preservation of evaluation mode.

        Args:
            img_tensor (torch.Tensor): Preprocessed chest radiograph tensor of shape
                [3, H, W] or [1, 3, H, W].
            clin_tensor (torch.Tensor): Standardized clinical tabular feature tensor of shape
                [6] or [1, 6].
            target_class (Optional[int]): Target diagnostic class index (0, 1, or 2).
                If None, the model's predicted class (argmax of logits) is used.

        Returns:
            np.ndarray: 2D float32 numpy array of shape (H_feat, W_feat) with values in [0.0, 1.0].

        Raises:
            ValueError: If input tensor dimensions or target_class are invalid.
            RuntimeError: If hooks failed to capture feature maps or gradients.
        """
        # 1. Format and validate input batch dimensions
        if not isinstance(img_tensor, torch.Tensor):
            raise ValueError(f"img_tensor must be a torch.Tensor, got {type(img_tensor)}")
        if not isinstance(clin_tensor, torch.Tensor):
            raise ValueError(f"clin_tensor must be a torch.Tensor, got {type(clin_tensor)}")

        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.dim() != 4:
            raise ValueError(f"img_tensor must have 3 or 4 dimensions, got shape {list(img_tensor.shape)}")

        if clin_tensor.dim() == 1:
            clin_tensor = clin_tensor.unsqueeze(0)
        elif clin_tensor.dim() != 2:
            raise ValueError(f"clin_tensor must have 1 or 2 dimensions, got shape {list(clin_tensor.shape)}")

        if img_tensor.size(0) != clin_tensor.size(0):
            raise ValueError(
                f"Batch size mismatch: img_tensor batch={img_tensor.size(0)} vs "
                f"clin_tensor batch={clin_tensor.size(0)}"
            )

        # 2. Align device with model
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        img_input = img_tensor.to(device)
        clin_input = clin_tensor.to(device)

        # Ensure image input tracks gradients so backprop flows to target_layer
        # even if vision backbone parameters have requires_grad=False (frozen backbone).
        img_input = img_input.clone().detach().requires_grad_(True)

        # 3. Preserve model training state and enforce eval mode
        was_training = self.model.training
        self.model.eval()

        # Reset buffers and gradients
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        # 4. Forward pass within gradient-enabled context
        try:
            with torch.enable_grad():
                model_out = self.model(img_input, clin_input)
                if isinstance(model_out, tuple):
                    logits = model_out[0]
                else:
                    logits = model_out

                if logits.dim() != 2 or logits.size(0) < 1:
                    raise RuntimeError(f"Expected 2D logits tensor [B, Num_Classes], got {logits.shape}")

                # 5. Extract target class
                num_classes = logits.size(1)
                if target_class is None:
                    target_class_idx = int(torch.argmax(logits, dim=1)[0].item())
                else:
                    target_class_idx = int(target_class)
                    if target_class_idx < 0 or target_class_idx >= num_classes:
                        raise ValueError(
                            f"target_class index {target_class_idx} out of range for {num_classes} classes."
                        )

                # Target class unnormalized score (logit) for single sample
                target_score = logits[0, target_class_idx]

                # 6. Backward pass to compute gradients d(y_c) / d(A_k)
                self.model.zero_grad(set_to_none=True)
                target_score.backward(retain_graph=False)

            # 7. Validate captured activations and gradients
            if self.activations is None:
                raise RuntimeError(
                    f"GradCAM forward hook failed to capture feature activations from '{self.target_layer_name}'."
                )
            if self.gradients is None:
                raise RuntimeError(
                    f"GradCAM backward hook failed to capture gradients from '{self.target_layer_name}'. "
                    "Ensure target layer is in the forward compute path of logits."
                )

            # Extract sample 0 feature maps and gradients [1, C, H_feat, W_feat]
            act = self.activations[0:1]
            grad = self.gradients[0:1]

            # 8. Channel-wise importance weights via Global Average Pooling (GAP)
            # alpha_k = (1 / Z) * sum_i sum_j (d y_c / d A_i,j^k)
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # Shape: [1, C, 1, 1]

            # 9. Weighted combination followed by ReLU
            # L_GradCAM = ReLU( sum_k alpha_k * A^k )
            cam = torch.sum(weights * act, dim=1, keepdim=True)  # Shape: [1, 1, H_feat, W_feat]
            cam = F.relu(cam)
            cam = cam.squeeze()  # Shape: [H_feat, W_feat]

            # 10. Min-Max Normalization to [0.0, 1.0]
            cam_min = cam.min()
            cam_max = cam.max()
            denom = cam_max - cam_min
            if denom > 1e-8:
                heatmap = (cam - cam_min) / denom
            else:
                heatmap = torch.zeros_like(cam)

            heatmap_np = heatmap.detach().cpu().numpy().astype(np.float32)

        finally:
            # 11. Strict Cleanup & State Restoration (Requirement 4)
            self.model.zero_grad(set_to_none=True)
            self.activations = None
            self.gradients = None

            # Restore original model training mode
            if was_training:
                self.model.train()
            else:
                self.model.eval()

            # Evict residual CUDA caching if active
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return heatmap_np

    @staticmethod
    def overlay_heatmap(
        heatmap: Union[np.ndarray, torch.Tensor],
        original_pil_img: Union[Image.Image, np.ndarray, str, Path],
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay a 2D Grad-CAM heatmap onto an original image using alpha blending.

        Process:
        1. Resize heatmap to match original image dimensions using bilinear interpolation.
        2. Convert normalized heatmap to 8-bit unsigned integer [0, 255].
        3. Apply colormap (cv2.COLORMAP_JET) and convert BGR output to RGB.
        4. Blend RGB heatmap with original RGB image: alpha * heatmap + (1 - alpha) * original.
        5. Return clean uint8 RGB numpy array ready for web display.

        Args:
            heatmap (Union[np.ndarray, torch.Tensor]): 2D saliency heatmap with values in [0, 1].
            original_pil_img (Union[Image.Image, np.ndarray, str, Path]): Original chest radiograph.
                Can be a PIL Image, uint8 RGB numpy array, or filesystem path.
            alpha (float): Transparency factor for the heatmap overlay (default: 0.4).
                0.0 shows only original image; 1.0 shows only the colormap heatmap.
            colormap (int): OpenCV colormap identifier (default: cv2.COLORMAP_JET).

        Returns:
            np.ndarray: Blended RGB image of shape (H, W, 3), dtype uint8, values in [0, 255].

        Raises:
            ValueError: If heatmap or image formats are incompatible.
        """
        # 1. Standardize heatmap input to 2D numpy float32
        if isinstance(heatmap, torch.Tensor):
            heatmap_arr = heatmap.detach().cpu().numpy()
        else:
            heatmap_arr = np.asarray(heatmap)

        if heatmap_arr.ndim != 2:
            heatmap_arr = np.squeeze(heatmap_arr)
            if heatmap_arr.ndim != 2:
                raise ValueError(f"Heatmap must be 2D, got shape {heatmap_arr.shape}")

        heatmap_arr = heatmap_arr.astype(np.float32)

        # Clip values to [0.0, 1.0]
        heatmap_arr = np.clip(heatmap_arr, 0.0, 1.0)

        # 2. Standardize original image to RGB uint8 numpy array
        if isinstance(original_pil_img, (str, Path)):
            pil_img = Image.open(str(original_pil_img)).convert("RGB")
            orig_rgb = np.array(pil_img, dtype=np.uint8)
        elif isinstance(original_pil_img, Image.Image):
            pil_img = original_pil_img.convert("RGB")
            orig_rgb = np.array(pil_img, dtype=np.uint8)
        elif isinstance(original_pil_img, np.ndarray):
            orig_arr = original_pil_img
            if orig_arr.dtype != np.uint8:
                if orig_arr.max() <= 1.0:
                    orig_arr = (orig_arr * 255.0).astype(np.uint8)
                else:
                    orig_arr = orig_arr.astype(np.uint8)

            if orig_arr.ndim == 2:
                # Grayscale to RGB
                orig_rgb = cv2.cvtColor(orig_arr, cv2.COLOR_GRAY2RGB)
            elif orig_arr.ndim == 3 and orig_arr.shape[2] == 1:
                orig_rgb = cv2.cvtColor(orig_arr, cv2.COLOR_GRAY2RGB)
            elif orig_arr.ndim == 3 and orig_arr.shape[2] >= 3:
                orig_rgb = orig_arr[:, :, :3]
            else:
                raise ValueError(f"Unsupported numpy image shape: {orig_arr.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(original_pil_img)}")

        orig_h, orig_w = orig_rgb.shape[:2]

        # 3. Bilinear interpolation resize of heatmap to original image dimensions
        # cv2.resize expects dsize=(width, height)
        resized_heatmap = cv2.resize(
            heatmap_arr,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

        # 4. Colormap mapping: float [0, 1] -> uint8 [0, 255] -> ColorMap
        heatmap_uint8 = np.uint8(255.0 * resized_heatmap)
        # Note: OpenCV applyColorMap outputs BGR format
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        # Convert BGR to RGB for consistent blending and web rendering
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 5. Alpha blending: overlay = alpha * heatmap_rgb + (1 - alpha) * orig_rgb
        alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
        blended = cv2.addWeighted(
            src1=heatmap_rgb,
            alpha=alpha_clamped,
            src2=orig_rgb,
            beta=1.0 - alpha_clamped,
            gamma=0.0,
        )

        return blended

    def explain(
        self,
        img_tensor: torch.Tensor,
        clin_tensor: torch.Tensor,
        original_pil_img: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
    ) -> Dict[str, Any]:
        """
        High-level end-to-end diagnostic explanation service.

        Computes model prediction, probabilities, Grad-CAM heatmap, and optionally
        blends an RGB overlay if the original image is supplied.

        Args:
            img_tensor: Preprocessed image tensor [3, H, W] or [1, 3, H, W].
            clin_tensor: Preprocessed clinical tensor [6] or [1, 6].
            original_pil_img: Optional PIL Image or numpy image for overlay rendering.
            target_class: Optional diagnostic class index (None for predicted argmax).
            alpha: Transparency factor for overlay (default: 0.4).

        Returns:
            Dict containing:
                - 'target_class': int index
                - 'predicted_class': int index
                - 'predicted_label': str class name
                - 'logits': np.ndarray of shape (3,)
                - 'probabilities': np.ndarray of shape (3,)
                - 'heatmap': np.ndarray 2D float32 [H_feat, W_feat]
                - 'overlay': Optional np.ndarray uint8 RGB [H, W, 3]
        """
        # Format inputs
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if clin_tensor.dim() == 1:
            clin_tensor = clin_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits_tensor, _ = self.model(img_tensor.to(device), clin_tensor.to(device))
            probs_tensor = F.softmax(logits_tensor, dim=1)

        logits_np = logits_tensor[0].cpu().numpy()
        probs_np = probs_tensor[0].cpu().numpy()
        predicted_idx = int(np.argmax(logits_np))

        selected_class = predicted_idx if target_class is None else int(target_class)

        # Generate Grad-CAM heatmap
        heatmap = self.generate_heatmap(img_tensor, clin_tensor, target_class=selected_class)

        overlay = None
        if original_pil_img is not None:
            overlay = self.overlay_heatmap(heatmap, original_pil_img, alpha=alpha)

        label_name = CLASS_NAMES[selected_class] if selected_class < len(CLASS_NAMES) else f"Class_{selected_class}"

        return {
            "target_class": selected_class,
            "predicted_class": predicted_idx,
            "predicted_label": label_name,
            "logits": logits_np,
            "probabilities": probs_np,
            "heatmap": heatmap,
            "overlay": overlay,
        }


# Module-level alias matching requirement signature
overlay_heatmap = GradCAM.overlay_heatmap


if __name__ == "__main__":
    print("=" * 70)
    print(" RayXplain Phase 3: GradCAM Explainability Verification Suite")
    print("=" * 70)

    # 1. Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Instantiate Model in Eval Mode
    print("\n[1/5] Instantiating MultiModalLungNet & GradCAM Engine...")
    model = MultiModalLungNet(pretrained=False)
    model.eval()

    gradcam = GradCAM(model=model, target_layer_name="layer4")
    print(f"      Hooked Target Layer : {gradcam.target_layer_name}")
    print(f"      Target Layer Type   : {type(gradcam.target_layer).__name__}")

    # 3. Dummy Inputs Generation
    print("\n[2/5] Creating simulated inputs...")
    dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    dummy_clin = torch.randn(1, 6, dtype=torch.float32)
    dummy_pil = Image.fromarray(np.random.randint(40, 220, (512, 512, 3), dtype=np.uint8))
    print(f"      Input Image Tensor    : {list(dummy_img.shape)}")
    print(f"      Input Clinical Tensor : {list(dummy_clin.shape)}")
    print(f"      Simulated PIL Image   : {dummy_pil.size} (W, H)")

    # 4. Test Heatmap Generation (Argmax & Specific Class)
    print("\n[3/5] Generating Grad-CAM heatmaps...")
    heatmap_pred = gradcam.generate_heatmap(dummy_img, dummy_clin, target_class=None)
    print(f"      Heatmap shape (predicted argmax) : {heatmap_pred.shape}")
    print(f"      Heatmap min / max                : {heatmap_pred.min():.4f} / {heatmap_pred.max():.4f}")

    assert heatmap_pred.shape == (7, 7), f"Expected shape (7, 7), got {heatmap_pred.shape}"
    assert 0.0 <= heatmap_pred.min() <= heatmap_pred.max() <= 1.0, "Heatmap not in [0, 1] range"

    # Specific class test (Class 2: High-Risk Pathological / COVID-19)
    heatmap_cls2 = gradcam.generate_heatmap(dummy_img, dummy_clin, target_class=2)
    assert heatmap_cls2.shape == (7, 7), f"Expected shape (7, 7), got {heatmap_cls2.shape}"
    print(f"      Heatmap shape (Class 2 target)   : {heatmap_cls2.shape}")

    # 5. Test Overlay Heatmap Method
    print("\n[4/5] Testing overlay_heatmap method...")
    overlay_out = gradcam.overlay_heatmap(heatmap_pred, dummy_pil, alpha=0.4)
    print(f"      Overlay Output shape : {overlay_out.shape}")
    print(f"      Overlay Output dtype : {overlay_out.dtype}")
    print(f"      Overlay Output range : [{overlay_out.min()}, {overlay_out.max()}]")

    assert overlay_out.shape == (512, 512, 3), f"Expected (512, 512, 3), got {overlay_out.shape}"
    assert overlay_out.dtype == np.uint8, f"Expected uint8 dtype, got {overlay_out.dtype}"
    assert overlay_out.min() >= 0 and overlay_out.max() <= 255

    # 6. Test Frozen Backbone Compatibility & Memory Cleanup
    print("\n[5/5] Testing frozen vision backbone & graph cleanup...")
    model.freeze_vision_backbone(freeze=True)
    frozen_heatmap = gradcam.generate_heatmap(dummy_img, dummy_clin, target_class=1)
    assert frozen_heatmap.shape == (7, 7), "Failed under frozen backbone"
    assert model.training is False, "Model training mode leaked"
    assert gradcam.activations is None, "Activations buffer not cleared"
    assert gradcam.gradients is None, "Gradients buffer not cleared"

    print("      [PASSED] Frozen backbone gradient flow verified")
    print("      [PASSED] Evaluation mode preserved strictly")
    print("      [PASSED] Buffers cleared and computational graph freed")

    # Clean up hooks
    gradcam.remove_hooks()
    print("\n" + "=" * 70)
    print(" GradCAM Phase 3 Implementation VERIFIED SUCCESSFULLY!")
    print("=" * 70)
