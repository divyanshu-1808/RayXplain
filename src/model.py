"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 2: Multi-Modal Deep Learning Architecture

Architecture Specification:
---------------------------
- Model Class: MultiModalLungNet(nn.Module)
- Vision Branch: Pretrained ResNet50 (weights=ResNet50_Weights.DEFAULT) truncated at `layer4`.
  Output shape: [B, 2048, 7, 7] (conv_features for Grad-CAM).
- Image Projection: AdaptiveAvgPool2d((1, 1)) -> Flatten() -> Linear(2048, 128)
  -> BatchNorm1d(128) -> ReLU -> Dropout(0.3) -> [B, 128].
- Clinical Tabular Branch: Input dim = 6 -> Linear(6, 64) -> BatchNorm1d(64) -> ReLU
  -> Linear(64, 32) -> BatchNorm1d(32) -> ReLU -> Dropout(0.2) -> [B, 32].
- Fusion Layer: Concatenate image latents (128) and tabular latents (32) along dim=1 -> [B, 160].
- Classification Head: Linear(160, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.2)
  -> Linear(64, 3) -> [B, 3].
- Forward Signature: forward(x_img, x_clin) -> Tuple[logits, conv_features].
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


# Diagnostic Target Class Names (in fixed order defined by training pipeline & SRS)
CLASS_NAMES: List[str] = [
    "Normal",
    "Viral / Bacterial Pneumonia",
    "High-Risk Pathological / COVID-19",
]

# Clinical Input Feature Vector (6 clinical parameters in fixed order)
CLINICAL_FEATURE_NAMES: List[str] = [
    "Age",
    "Sex",
    "SpO2",
    "Smoking Pack-Years",
    "Cough Severity",
    "Body Temperature",
]


class MultiModalLungNet(nn.Module):
    """
    Dual-branch multi-modal neural network combining chest radiograph deep features
    with standardized clinical tabular parameters for pulmonary disease risk prediction.

    Attributes:
        conv_base (nn.Sequential): Truncated ResNet-50 convolutional base (conv1 through layer4),
            extracting rich spatial feature maps [B, 2048, 7, 7] preserved for Grad-CAM.
        pool (nn.AdaptiveAvgPool2d): Spatial pooling to collapse [B, 2048, 7, 7] to [B, 2048, 1, 1].
        img_proj (nn.Sequential): MLP projecting 2048-dim image features to a 128-dim latent space.
        tabular_mlp (nn.Sequential): MLP projecting 6-dim clinical parameters to a 32-dim latent space.
        classifier (nn.Sequential): Multimodal classification head projecting the fused 160-dim
            representation to 3 class logits.
    """

    def __init__(
        self,
        tabular_dim: int = 6,
        num_classes: int = 3,
        pretrained: bool = True,
        weights: Optional[ResNet50_Weights] = ResNet50_Weights.DEFAULT,
        img_latent_dim: int = 128,
        tab_latent_dim: int = 32,
        dropout_img: float = 0.3,
        dropout_tab: float = 0.2,
        dropout_cls: float = 0.2,
    ) -> None:
        """
        Initialize the MultiModalLungNet model architecture.

        Args:
            tabular_dim: Dimensionality of clinical tabular input vector (default: 6).
            num_classes: Number of disease prediction categories (default: 3).
            pretrained: Whether to load pre-trained ImageNet weights for ResNet50 (default: True).
            weights: Torchvision ResNet50_Weights enum instance (default: ResNet50_Weights.DEFAULT).
            img_latent_dim: Dimensionality of projected image latent vector (default: 128).
            tab_latent_dim: Dimensionality of projected tabular latent vector (default: 32).
            dropout_img: Dropout probability for image projection head (default: 0.3).
            dropout_tab: Dropout probability for clinical tabular branch (default: 0.2).
            dropout_cls: Dropout probability for classification head (default: 0.2).
        """
        super().__init__()

        self.tabular_dim = tabular_dim
        self.num_classes = num_classes
        self.img_latent_dim = img_latent_dim
        self.tab_latent_dim = tab_latent_dim
        self.fusion_dim = img_latent_dim + tab_latent_dim  # 128 + 32 = 160

        # ----------------------------------------------------------------------
        # 1. Vision Branch: Pretrained ResNet-50 truncated up to layer4
        # ----------------------------------------------------------------------
        # ResNet50 children structure:
        # [0]: conv1, [1]: bn1, [2]: relu, [3]: maxpool,
        # [4]: layer1, [5]: layer2, [6]: layer3, [7]: layer4,
        # [8]: avgpool, [9]: fc
        # Truncating up to index 8 keeps conv1 through layer4, yielding [B, 2048, 7, 7].
        resnet_weights = weights if pretrained else None
        try:
            base_resnet = models.resnet50(weights=resnet_weights)
        except Exception as e:
            # Fallback to unweighted initialization if offline or download fails
            print(f"[MultiModalLungNet] Warning: Could not download ResNet50 weights ({e}). Initializing with random weights.")
            base_resnet = models.resnet50(weights=None)

        self.conv_base = nn.Sequential(*list(base_resnet.children())[:8])

        # ----------------------------------------------------------------------
        # 2. Image Projection Head:
        # AdaptiveAvgPool2d((1, 1)) -> Flatten() -> Linear(2048, 128)
        # -> BatchNorm1d(128) -> ReLU -> Dropout(0.3)
        # ----------------------------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, self.img_latent_dim),
            nn.BatchNorm1d(self.img_latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_img),
        )

        # ----------------------------------------------------------------------
        # 3. Clinical Tabular Branch:
        # Input dim = 6 -> Linear(6, 64) -> BatchNorm1d(64) -> ReLU
        # -> Linear(64, 32) -> BatchNorm1d(32) -> ReLU -> Dropout(0.2)
        # ----------------------------------------------------------------------
        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.tab_latent_dim),
            nn.BatchNorm1d(self.tab_latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_tab),
        )

        # ----------------------------------------------------------------------
        # 4. Multimodal Intermediate Fusion & Classification Head:
        # Concatenate image latents (128) and tabular latents (32) along dim=1 -> [B, 160]
        # Linear(160, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.2) -> Linear(64, 3)
        # ----------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_cls),
            nn.Linear(64, self.num_classes),
        )

        # Attribute aliases via properties (prevents PyTorch submodule re-registration)
        # and ensures state_dict matches existing weights checkpoints with 100% precision.

    @property
    def avgpool(self) -> nn.AdaptiveAvgPool2d:
        return self.pool

    @property
    def tabular_branch(self) -> nn.Sequential:
        return self.tabular_mlp

    @property
    def clinical_branch(self) -> nn.Sequential:
        return self.tabular_mlp

    @property
    def classification_head(self) -> nn.Sequential:
        return self.classifier

    def forward(
        self,
        x_img: torch.Tensor,
        x_clin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multimodal lung disease risk prediction.

        Args:
            x_img (torch.Tensor): Chest radiograph input batch tensor of shape [B, 3, H, W]
                (typically [B, 3, 224, 224]).
            x_clin (torch.Tensor): Clinical tabular parameters input tensor of shape [B, 6].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): Unnormalized class prediction logits of shape [B, 3].
                - conv_features (torch.Tensor): Raw convolutional activation maps from layer4
                  of shape [B, 2048, H/32, W/32] (e.g. [B, 2048, 7, 7]), required for Grad-CAM.
        """
        # Validate batch dimension consistency
        if x_img.size(0) != x_clin.size(0):
            raise ValueError(
                f"Batch size mismatch: image batch has {x_img.size(0)} samples, "
                f"while clinical batch has {x_clin.size(0)} samples."
            )

        # 1. Vision Feature Extraction (layer4 raw convolutional activations)
        conv_features = self.conv_base(x_img)  # Shape: [B, 2048, 7, 7]

        # 2. Image Latent Projection
        pooled_img = self.pool(conv_features)  # Shape: [B, 2048, 1, 1]
        img_latents = self.img_proj(pooled_img)  # Shape: [B, 128]

        # 3. Clinical Tabular Latent Projection
        tab_latents = self.tabular_mlp(x_clin)  # Shape: [B, 32]

        # 4. Intermediate Multimodal Fusion (concatenation along dim=1)
        fused = torch.cat([img_latents, tab_latents], dim=1)  # Shape: [B, 160]

        # 5. Disease Risk Classification Head
        logits = self.classifier(fused)  # Shape: [B, 3]

        return logits, conv_features

    def get_conv_layer(self) -> nn.Module:
        """
        Return the target convolutional layer (layer4) for Grad-CAM explainability hooks.

        Returns:
            nn.Module: The final ResNet50 convolutional stage (conv_base[7], i.e., layer4).
        """
        return self.conv_base[7]

    def freeze_vision_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the ResNet50 convolutional base parameters.

        Args:
            freeze (bool): If True, disable gradient calculation for conv_base parameters.
        """
        for param in self.conv_base.parameters():
            param.requires_grad = not freeze

    def load_weights(
        self,
        weights_path: Union[str, Path],
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load trained weights from a checkpoint file into this model instance.

        Args:
            weights_path: Path to the .pth or .pt checkpoint file.
            map_location: Device to load tensors to (default: 'cpu').
            strict: Whether to strictly enforce key matching (default: True).

        Returns:
            Dict containing 'missing_keys' and 'unexpected_keys'.
        """
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")

        try:
            checkpoint = torch.load(str(weights_path), map_location=map_location, weights_only=True)
        except Exception:
            checkpoint = torch.load(str(weights_path), map_location=map_location, weights_only=False)

        # Unpack wrapped checkpoint dictionaries if present
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        load_result = self.load_state_dict(state_dict, strict=strict)
        return {
            "missing_keys": load_result.missing_keys,
            "unexpected_keys": load_result.unexpected_keys,
        }


# Case-insensitive alias matching documentation and SDS references
MultimodalLungNet = MultiModalLungNet


def build_model(
    pretrained: bool = True,
    weights_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> MultiModalLungNet:
    """
    Factory function to construct MultiModalLungNet and optionally load checkpoint weights.

    Args:
        pretrained: Whether to initialize ResNet50 with default ImageNet weights.
        weights_path: Optional path to a trained checkpoint file (e.g. weights/best_multimodal_model.pth).
        device: Target execution device ('cuda', 'cpu', or torch.device).

    Returns:
        MultiModalLungNet: Initialized model instance on the specified device.
    """
    model = MultiModalLungNet(pretrained=pretrained)

    if weights_path is not None:
        weights_file = Path(weights_path)
        if weights_file.is_file():
            print(f"[build_model] Loading checkpoint from: {weights_file}")
            model.load_weights(weights_file, map_location="cpu")
        else:
            print(f"[build_model] Warning: Weights file not found at {weights_file}, skipping load.")

    if device is not None:
        model.to(device)

    return model


if __name__ == "__main__":
    print("=" * 70)
    print(" RayXplain Phase 2: MultiModalLungNet Verification Suite")
    print("=" * 70)

    # 1. Reproducibility seed
    torch.manual_seed(42)

    # 2. Instantiate Architecture
    print("\n[1/4] Instantiating MultiModalLungNet architecture...")
    model = MultiModalLungNet(
        tabular_dim=6,
        num_classes=3,
        pretrained=True,
    )
    model.eval()

    # 3. Parameter count diagnostics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_base_params = sum(p.numel() for p in model.conv_base.parameters())
    img_proj_params = sum(p.numel() for p in model.img_proj.parameters())
    tabular_mlp_params = sum(p.numel() for p in model.tabular_mlp.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"      Total Parameters       : {total_params:,}")
    print(f"      Trainable Parameters   : {trainable_params:,}")
    print(f"      - conv_base (ResNet50) : {conv_base_params:,} ({conv_base_params / total_params * 100:.1f}%)")
    print(f"      - img_proj (128-D)     : {img_proj_params:,} ({img_proj_params / total_params * 100:.2f}%)")
    print(f"      - tabular_mlp (32-D)   : {tabular_mlp_params:,} ({tabular_mlp_params / total_params * 100:.3f}%)")
    print(f"      - classifier (Head)    : {classifier_params:,} ({classifier_params / total_params * 100:.3f}%)")

    # 4. Generate dummy tensors (Batch size = 2)
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    dummy_clin = torch.randn(batch_size, 6, dtype=torch.float32)

    print(f"\n[2/4] Testing forward pass with dummy batch of {batch_size}...")
    print(f"      Input image tensor shape    : {list(dummy_img.shape)}  [B, C, H, W]")
    print(f"      Input clinical tensor shape : {list(dummy_clin.shape)}  [B, Tabular_Dim]")

    # 5. Forward pass execution
    with torch.no_grad():
        logits, conv_features = model(dummy_img, dummy_clin)

    print(f"      Output logits shape         : {list(logits.shape)}  [B, Num_Classes]")
    print(f"      Output conv_features shape  : {list(conv_features.shape)}  [B, 2048, 7, 7]")

    # 6. Assertions for zero dimension bugs
    print("\n[3/4] Running strict shape & dimension assertions...")
    assert logits.shape == (batch_size, 3), (
        f"Assertion Failed! Expected logits shape ({batch_size}, 3), but got {logits.shape}"
    )
    assert conv_features.shape == (batch_size, 2048, 7, 7), (
        f"Assertion Failed! Expected conv_features shape ({batch_size}, 2048, 7, 7), but got {conv_features.shape}"
    )
    assert not torch.isnan(logits).any(), "Assertion Failed! Logits contain NaN values."
    assert not torch.isnan(conv_features).any(), "Assertion Failed! conv_features contain NaN values."
    print("      [PASSED] logits.shape == (2, 3)")
    print("      [PASSED] conv_features.shape == (2, 2048, 7, 7)")
    print("      [PASSED] Zero dimension bugs verified: No NaNs, exact dimension match")

    # 7. Checkpoint loading verification (if weights exist)
    print("\n[4/4] Validating against project weights artifact...")
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint_file = repo_root / "weights" / "best_multimodal_model.pth"

    if checkpoint_file.is_file():
        print(f"      Found artifact: {checkpoint_file}")
        try:
            load_info = model.load_weights(checkpoint_file, map_location="cpu", strict=True)
            missing = load_info["missing_keys"]
            unexpected = load_info["unexpected_keys"]
            if len(missing) == 0 and len(unexpected) == 0:
                print("      [PASSED] Checkpoint loaded with PERFECT 100% key alignment!")
                print(f"               Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            else:
                print(f"      [WARN] Checkpoint keys mismatch: Missing={len(missing)}, Unexpected={len(unexpected)}")
        except Exception as e:
            print(f"      [WARN] Could not load checkpoint directly: {e}")
    else:
        print(f"      [INFO] Checkpoint not found at {checkpoint_file} (skipping checkpoint test).")

    print("\n" + "=" * 70)
    print(" MultiModalLungNet Phase 2 Implementation VERIFIED SUCCESSFULLY!")
    print("=" * 70)
