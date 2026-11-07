"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 5: Primary Web User Interface (Gradio Blocks Dashboard)

This module implements the primary web interface for RayXplain:
- Responsive dual-column clinical layout (Intake & Clinical Vitals vs. Diagnostic Saliency Assessment).
- Connected to the end-to-end MultiModal DiagnosticPipeline.
- Real-time Grad-CAM explainability overlay with tabbed visual comparison.
- Top-3 class risk probabilities bar plot with percentage indicators.
- Clinical boundary validation and defensive alert handling.
- Integrated one-click faculty demo scenarios.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr
import numpy as np
from PIL import Image, ImageFilter

from src.pipeline import TARGET_CLASSES, DiagnosticPipeline

# Initialize the shared multimodal diagnostic pipeline (loads model, scaler, and GradCAM)
print("[RayXplain Web UI] Initializing DiagnosticPipeline backend...")
pipeline = DiagnosticPipeline()
print(f"[RayXplain Web UI] Backend ready on compute device: {pipeline.device}")


# ------------------------------------------------------------------------------
# Example Case Generation & Management
# ------------------------------------------------------------------------------
def ensure_example_cases() -> List[List[Any]]:
    """
    Ensure synthetic radiograph case assets exist on disk for quick one-click faculty demos.
    Returns list of example rows: [image_path, age, sex, spo2, smoking, cough, fever].
    """
    examples_dir = _REPO_ROOT / "examples"
    examples_dir.mkdir(exist_ok=True)

    def _generate_synthetic_cxr(variant: str) -> Path:
        target_path = examples_dir / f"case_{variant}.png"
        if target_path.is_file():
            return target_path

        img = np.zeros((400, 400), dtype=np.float32)
        y, x = np.ogrid[:400, :400]

        # Outer torso contour
        torso = ((x - 200) / 160) ** 2 + ((y - 200) / 190) ** 2 < 1.0
        img[torso] = 68.0

        # Bilateral lung fields (darker air cavities)
        left_lung = ((x - 140) / 45) ** 2 + ((y - 180) / 110) ** 2 < 1.0
        right_lung = ((x - 260) / 45) ** 2 + ((y - 180) / 110) ** 2 < 1.0
        img[left_lung] = 25.0
        img[right_lung] = 25.0

        # Cardiac silhouette & mediastinum
        heart = ((x - 215) / 50) ** 2 + ((y - 230) / 60) ** 2 < 1.0
        img[heart] = 135.0
        spine = np.abs(x - 200) < 18
        img[spine & torso] += 40.0

        # Anatomic rib contours
        for rib_y in range(100, 320, 35):
            rib = np.abs(y - rib_y - 0.05 * (x - 200) ** 2 / 10) < 6
            img[rib & torso] += 25.0

        # Pathological opacities
        if variant == "pneumonia":
            infiltrate = ((x - 265) / 32) ** 2 + ((y - 220) / 38) ** 2 < 1.0
            img[infiltrate] += 65.0
        elif variant == "highrisk_covid":
            opac1 = ((x - 125) / 28) ** 2 + ((y - 200) / 45) ** 2 < 1.0
            opac2 = ((x - 275) / 28) ** 2 + ((y - 200) / 45) ** 2 < 1.0
            img[opac1] += 58.0
            img[opac2] += 58.0

        # Natural Poisson/Gaussian noise simulation
        noise = np.random.normal(0, 4.5, (400, 400)).astype(np.float32)
        final_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(final_img).convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.2))
        pil_img.save(target_path)
        return target_path

    path_norm = str(_generate_synthetic_cxr("normal"))
    path_pneu = str(_generate_synthetic_cxr("pneumonia"))
    path_covi = str(_generate_synthetic_cxr("highrisk_covid"))

    return [
        [path_norm, 28, "Female", 98.5, 0.0, "None", 98.4],
        [path_pneu, 54, "Male", 93.0, 12.0, "Moderate", 101.4],
        [path_covi, 69, "Male", 88.0, 26.0, "Severe", 102.8],
    ]


EXAMPLE_CASES = ensure_example_cases()


# ------------------------------------------------------------------------------
# Diagnostic Assessment Controller
# ------------------------------------------------------------------------------
def analyze_patient_case(
    pil_image: Optional[Image.Image],
    age: float,
    sex: str,
    spo2: float,
    smoking: float,
    cough: str,
    fever: float,
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], Dict[str, float], str]:
    """
    Controller function invoking the MultiModal DiagnosticPipeline with input validation.

    Returns:
        Tuple:
            1. overlay_image (np.ndarray): Grad-CAM saliency overlay.
            2. original_image (Image.Image): Uploaded radiograph.
            3. probabilities (Dict[str, float]): Top-3 class risk probabilities.
            4. status_badge (str): Styled HTML badge representing diagnostic outcome.
    """
    # 1. Image Upload Validation
    if pil_image is None:
        raise gr.Error("Missing Chest Radiograph: Please upload a valid chest X-ray image (.png, .jpg, .jpeg) before analysis.")

    # 2. Execute Multimodal Diagnostic Pipeline
    try:
        overlay_img, probs_dict, predicted_class = pipeline.predict(
            pil_image=pil_image,
            age=age,
            sex=sex,
            spo2=spo2,
            smoking=smoking,
            cough_severity=cough,
            fever=fever,
            alpha=0.42,
        )
    except Exception as e:
        raise gr.Error(f"Diagnostic Assessment Error: {str(e)}")

    # 3. Format Diagnostic Summary HTML Badge
    top_prob = probs_dict.get(predicted_class, 0.0) * 100.0

    if predicted_class == "Normal":
        badge_color = "#10b981"
        badge_bg = "#ecfdf5"
        badge_border = "#a7f3d0"
        icon = "✅"
        clinical_note = "Features indicate low pulmonary risk. Maintain clinical correlation."
    elif predicted_class == "Pneumonia":
        badge_color = "#f59e0b"
        badge_bg = "#fffbeb"
        badge_border = "#fde68a"
        icon = "⚠️"
        clinical_note = "Consolidation / infiltrative pattern detected. Sputum & antibiotic evaluation suggested."
    else:  # High-Risk / COVID-19
        badge_color = "#ef4444"
        badge_bg = "#fef2f2"
        badge_border = "#fecaca"
        icon = "🚨"
        clinical_note = "Bilateral ground-glass / acute pathology markers detected. Immediate isolation and confirmatory PCR recommended."

    badge_html = f"""
    <div style="background-color: {badge_bg}; border: 1.5px solid {badge_border}; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.6rem;">{icon}</span>
                <div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">Primary Assessment</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: {badge_color};">{predicted_class}</div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">Confidence</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #1e293b;">{top_prob:.1f}%</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 0.875rem; color: #334155; border-top: 1px dashed {badge_border}; padding-top: 8px;">
            <strong>Clinical Guidance:</strong> {clinical_note}
        </div>
    </div>
    """

    return overlay_img, pil_image, probs_dict, badge_html


# ------------------------------------------------------------------------------
# Custom Clinical Dashboard CSS
# ------------------------------------------------------------------------------
CUSTOM_CSS = """
/* Dashboard Container & Clinical Typography */
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #f8fafc !important;
}

/* Primary CTA Button - Medical Blue */
#btn-analyze {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.28) !important;
    transition: all 0.2s ease-in-out !important;
}
#btn-analyze:hover {
    background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
    box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4) !important;
    transform: translateY(-1px);
}

/* Card Column Containers */
.clinical-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

/* Disclaimer Callout */
.disclaimer-banner {
    background: #fffbeb !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    font-size: 0.825rem !important;
    color: #92400e !important;
    margin-top: 14px !important;
    line-height: 1.4 !important;
}
"""


# ------------------------------------------------------------------------------
# Gradio Blocks Application Layout
# ------------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    """Build and assemble the RayXplain Gradio Blocks application interface."""

    blocks_kwargs: Dict[str, Any] = {"title": "RayXplain | Multi-Modal Diagnostic Assistant"}
    # In Gradio < 6, css is passed to Blocks; in Gradio >= 6, css is passed to launch()
    if hasattr(gr, "__version__") and int(gr.__version__.split(".")[0]) < 6:
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        # Header Section
        gr.Markdown(
            """
            # 🩺 RayXplain: Multi-Modal Lung Disease Diagnostic Assistant
            **Dual-Branch Deep Learning (ResNet-50 + Clinical MLP) with Explainable AI via Grad-CAM Saliency Localization**
            """
        )

        # Responsive Two-Column Layout
        with gr.Row(equal_height=False):
            # ------------------------------------------------------------------
            # Left Column: Patient Case Intake & Clinical Vitals
            # ------------------------------------------------------------------
            with gr.Column(scale=5, elem_classes=["clinical-card"]):
                gr.Markdown("### 📋 Patient Case Intake & Clinical Vitals")

                image_input = gr.Image(
                    type="pil",
                    label="Chest Radiograph (PA / AP Projection)",
                    sources=["upload", "clipboard"],
                    height=280,
                )

                with gr.Row():
                    age_input = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=45,
                        step=1,
                        label="Patient Age (Years)",
                    )
                    sex_input = gr.Radio(
                        choices=["Male", "Female"],
                        value="Male",
                        label="Biological Sex",
                    )

                with gr.Row():
                    spo2_input = gr.Slider(
                        minimum=70,
                        maximum=100,
                        value=96,
                        step=1,
                        label="SpO2 Oxygen Saturation (%)",
                    )
                    fever_input = gr.Number(
                        value=98.6,
                        minimum=85.0,
                        maximum=115.0,
                        step=0.1,
                        label="Body Temperature (°F)",
                    )

                with gr.Row():
                    smoking_input = gr.Number(
                        value=0.0,
                        minimum=0.0,
                        step=0.5,
                        label="Smoking History (Pack-Years)",
                    )
                    cough_input = gr.Dropdown(
                        choices=["None", "Mild", "Moderate", "Severe"],
                        value="None",
                        label="Cough Severity",
                    )

                analyze_btn = gr.Button(
                    "🔍 Analyze Patient Case",
                    variant="primary",
                    size="lg",
                    elem_id="btn-analyze",
                )

            # ------------------------------------------------------------------
            # Right Column: Diagnostic Assessment & Saliency Maps
            # ------------------------------------------------------------------
            with gr.Column(scale=6, elem_classes=["clinical-card"]):
                gr.Markdown("### 🔬 Diagnostic Assessment & Explainability")

                # Diagnostic Summary Badge (Populated upon inference)
                badge_output = gr.HTML(
                    value="""
                    <div style="background-color: #f1f5f9; border: 1.5px dashed #cbd5e1; border-radius: 10px; padding: 14px 18px; text-align: center; color: #64748b; font-size: 0.95rem;">
                        Awaiting patient case submission. Upload a chest radiograph and enter clinical vitals to run the assessment.
                    </div>
                    """
                )

                # Class Risk Probabilities Bar Plot
                risk_label_output = gr.Label(
                    num_top_classes=3,
                    label="Top-3 Class Risk Probabilities",
                )

                # Tabbed Radiograph Visualization (Grad-CAM Overlay vs Original)
                with gr.Tabs():
                    with gr.TabItem("🔥 Grad-CAM Saliency Overlay"):
                        overlay_image_output = gr.Image(
                            label="Class-Specific Activation Overlay (ResNet-50 layer4)",
                            type="numpy",
                            height=340,
                        )
                    with gr.TabItem("🖼️ Original Radiograph"):
                        original_image_output = gr.Image(
                            label="Raw Uploaded Chest Radiograph",
                            type="pil",
                            height=340,
                        )

                # Clinical Disclaimer Callout Banner
                gr.Markdown(
                    """
                    <div class="disclaimer-banner">
                        ⚠️ <strong>Research & Clinical Decision-Support Prototype:</strong> Not certified for autonomous diagnosis.
                        All multi-modal risk probabilities and Grad-CAM localized heatmaps are provided exclusively as assistive decision-support tools
                        and must be verified by a licensed radiologist or medical professional.
                    </div>
                    """
                )

        # ----------------------------------------------------------------------
        # Faculty Review Demo Scenarios (One-Click Quick Load)
        # ----------------------------------------------------------------------
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🧪 Pre-Configured Demonstration Scenarios (One-Click Faculty Review)")
                gr.Examples(
                    examples=EXAMPLE_CASES,
                    inputs=[
                        image_input,
                        age_input,
                        sex_input,
                        spo2_input,
                        smoking_input,
                        cough_input,
                        fever_input,
                    ],
                    outputs=[
                        overlay_image_output,
                        original_image_output,
                        risk_label_output,
                        badge_output,
                    ],
                    fn=analyze_patient_case,
                    cache_examples=False,
                )

        # ----------------------------------------------------------------------
        # Event Wiring
        # ----------------------------------------------------------------------
        analyze_btn.click(
            fn=analyze_patient_case,
            inputs=[
                image_input,
                age_input,
                sex_input,
                spo2_input,
                smoking_input,
                cough_input,
                fever_input,
            ],
            outputs=[
                overlay_image_output,
                original_image_output,
                risk_label_output,
                badge_output,
            ],
        )

    return demo


# Create top-level demo app instance for standard WSGI/ASGI/Gradio CLI hosting
demo = create_app()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" RayXplain Phase 5: Launching Web Dashboard Interface...")
    print("=" * 70)

    launch_kwargs: Dict[str, Any] = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
    }
    if hasattr(gr, "__version__") and int(gr.__version__.split(".")[0]) >= 6:
        launch_kwargs["css"] = CUSTOM_CSS

    # Launch on default local port 7860
    demo.launch(**launch_kwargs)
