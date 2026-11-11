"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 5 (Refined): Clinical Dashboard Web Interface (Gradio Blocks)

A polished, high-contrast medical AI dashboard featuring:
- Cohesive Medical Slate colour palette with full WCAG readability.
- Dual-column layout: Patient Case Intake ↔ Diagnostic Assessment & Explainability.
- Dynamic KPI badge (Emerald / Amber / Rose) with confidence percentage.
- Horizontal probability bars for all 3 diagnostic classes.
- Tabbed Grad-CAM saliency overlay vs. original radiograph.
- Silent background archiving via CaseArchiver (no history UI elements).
- One-click real clinical benchmark presets from test_samples/.
- Statutory clinical decision-support disclaimer.
"""

import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline import TARGET_CLASSES, DiagnosticPipeline
from src.record_keeper import CaseArchiver

logger = logging.getLogger("RayXplain.App")

# ---------------------------------------------------------------------------
# Backend Initialization
# ---------------------------------------------------------------------------
print("[RayXplain] Initializing DiagnosticPipeline backend...")
pipeline = DiagnosticPipeline()
print(f"[RayXplain] Backend ready — compute device: {pipeline.device}")

archiver = CaseArchiver()
print(f"[RayXplain] CaseArchiver ready — records dir: {archiver.records_dir}")

# Compute badge
_DEVICE_BADGE = "CUDA GPU" if str(pipeline.device) == "cuda" else "CPU"


# ---------------------------------------------------------------------------
# Load Real Clinical Example Cases (no synthetic phantoms)
# ---------------------------------------------------------------------------
def load_example_cases() -> List[List[Any]]:
    """
    Load pre-configured real clinical benchmark scenarios for quick faculty evaluation.
    Uses genuine open-access human chest radiographs from test_samples/.
    Returns list of example rows: [image_path, age, sex, spo2, smoking, cough, fever].
    """
    try:
        from test_samples.canonical_cases import BENCHMARK_CASES
        from test_samples.setup_offline_samples import setup_offline_samples

        # Ensure real benchmark files are present on disk
        missing = [c for c in BENCHMARK_CASES.values() if not Path(c["local_image_path"]).is_file()]
        if missing:
            setup_offline_samples(force_redownload=False)

        # Only include the three representative cases for clean demo presets
        selected_keys = ["sample_normal_1", "sample_pneumonia_bacterial", "sample_covid_acute"]
        cases_list = []
        for key in selected_keys:
            c = BENCHMARK_CASES.get(key)
            if c and Path(c["local_image_path"]).is_file():
                cases_list.append([
                    c["local_image_path"],
                    c["age"],
                    c["sex"],
                    c["spo2"],
                    c["smoking_pack_years"],
                    c["cough_severity"],
                    c["fever_f"],
                ])

        # Fallback: include any remaining valid cases to guarantee ≥3 rows for tests
        if len(cases_list) < 3:
            for c in BENCHMARK_CASES.values():
                if Path(c["local_image_path"]).is_file():
                    row = [
                        c["local_image_path"],
                        c["age"],
                        c["sex"],
                        c["spo2"],
                        c["smoking_pack_years"],
                        c["cough_severity"],
                        c["fever_f"],
                    ]
                    if row not in cases_list:
                        cases_list.append(row)

        return cases_list
    except Exception as exc:
        print(f"[RayXplain] Warning: Could not load canonical cases: {exc}")

    return []


# Backward-compatible alias used by test suites
ensure_example_cases = load_example_cases

EXAMPLE_CASES = load_example_cases()


# ---------------------------------------------------------------------------
# Backward-Compatible Manual Archive Function (used by test suites only)
# ---------------------------------------------------------------------------
def archive_current_case(
    raw_input_img: Optional[Image.Image],
    original_out_img: Optional[Image.Image],
    overlay_out_img: Optional[np.ndarray],
    risk_label_data: Any,
    age: float,
    sex: str,
    spo2: float,
    smoking: float,
    cough: str,
    fever: float,
    patient_tag: str,
) -> str:
    """
    Export current diagnostic assessment into a permanent, audit-ready folder
    under patient_records/. Retained for backward compatibility with test suites.

    Returns:
        str: Markdown-formatted status message.
    """
    active_img = original_out_img if original_out_img is not None else raw_input_img
    if active_img is None or overlay_out_img is None:
        gr.Warning("No assessment available to export. Please analyze a patient case first.")
        return (
            "⚠️ **Export Aborted:** Please upload a radiograph and click "
            "**Analyze Patient Case** before archiving."
        )

    # Parse risk_label_data
    probs: Dict[str, float] = {}
    predicted_class = "Assessment"
    confidence = 0.0

    if isinstance(risk_label_data, dict):
        if "confidences" in risk_label_data and isinstance(risk_label_data["confidences"], list):
            for item in risk_label_data["confidences"]:
                probs[str(item.get("label", "Unknown"))] = float(item.get("confidence", 0.0))
            predicted_class = str(
                risk_label_data.get("label", max(probs, key=probs.get) if probs else "Diagnosis")
            )
            confidence = probs.get(predicted_class, 0.0)
        else:
            for k, v in risk_label_data.items():
                try:
                    probs[str(k)] = float(v)
                except (ValueError, TypeError):
                    pass
            if probs:
                predicted_class = max(probs, key=probs.get)
                confidence = probs[predicted_class]
    elif isinstance(risk_label_data, str) and risk_label_data.strip():
        predicted_class = risk_label_data.strip()
        confidence = 1.0
        probs = {predicted_class: 1.0}

    vitals = {
        "age": age, "sex": sex, "spo2": spo2,
        "smoking": smoking, "cough": cough, "fever": fever,
    }
    pred_dict = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probs,
    }

    try:
        tag_str = patient_tag.strip() if patient_tag and patient_tag.strip() else "Manual_Case"
        saved_dir = archiver.save_evaluation(
            original_pil_img=active_img,
            overlay_pil_img=overlay_out_img,
            vitals_dict=vitals,
            prediction_dict=pred_dict,
            patient_tag=tag_str,
        )
        return (
            f"✅ **Case Successfully Archived!**<br>"
            f"📁 **Destination:** `{saved_dir.name}`<br>"
            f"📄 **Saved Artifacts:** `original_xray.png`, `gradcam_explanation.png`, `clinical_summary.json`<br>"
            f"📊 **Master Audit Log:** Appended to `patient_records/evaluations_history.csv`"
        )
    except Exception as e:
        gr.Warning(f"Failed to archive patient record: {str(e)}")
        return f"❌ **Archive Failed:** {str(e)}"


# ---------------------------------------------------------------------------
# Silent Background Archiving (no UI elements)
# ---------------------------------------------------------------------------
def _archive_silently(
    original_img: Image.Image,
    overlay_img: np.ndarray,
    vitals: Dict[str, Any],
    prediction: Dict[str, Any],
) -> None:
    """Fire-and-forget archiving on a background thread. Never blocks the UI."""
    try:
        archiver.save_evaluation(
            original_pil_img=original_img,
            overlay_pil_img=overlay_img,
            vitals_dict=vitals,
            prediction_dict=prediction,
            patient_tag="Auto_Archive",
        )
    except Exception:
        logger.debug("Silent archiving failed: %s", traceback.format_exc())


# ---------------------------------------------------------------------------
# Diagnostic Assessment Controller
# ---------------------------------------------------------------------------
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
            1. overlay_image (np.ndarray): Grad-CAM saliency overlay (uint8 RGB).
            2. original_image (Image.Image): Uploaded radiograph.
            3. probabilities (Dict[str, float]): Top-3 class risk probabilities.
            4. status_html (str): Styled HTML card representing diagnostic outcome.
    """
    # 1. Image Upload Validation
    if pil_image is None:
        raise gr.Error(
            "Missing Chest Radiograph: Please upload a valid chest X-ray image "
            "(.png, .jpg, .jpeg) before analysis."
        )

    # 2. Execute Multimodal Diagnostic Pipeline
    try:
        overlay_img, probs_dict, predicted_class, warnings_list = pipeline.predict(
            pil_image=pil_image,
            age=age,
            sex=sex,
            spo2=spo2,
            smoking=smoking,
            cough_severity=cough,
            fever=fever,
            alpha=0.42,
            return_warnings=True,
        )
    except ValueError as val_err:
        raise gr.Error(f"Validation Error: {str(val_err)}")
    except Exception as e:
        raise gr.Error(f"Diagnostic Assessment Error: {str(e)}")

    # 3. Trigger user-facing notifications for atypical vitals
    for warn_msg in warnings_list:
        gr.Warning(warn_msg)

    # 4. Ensure overlay is clean uint8 RGB numpy array (fixes h11 content-length error)
    if isinstance(overlay_img, np.ndarray):
        if overlay_img.dtype != np.uint8:
            overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
    else:
        overlay_img = np.array(overlay_img, dtype=np.uint8)

    # 5. Format KPI Badge + Probability Bars + Clinical Summary
    top_prob = probs_dict.get(predicted_class, 0.0) * 100.0

    # Colour scheme per diagnosis
    BADGE_STYLES = {
        "Normal": {
            "color": "#059669", "bg": "#ecfdf5", "border": "#a7f3d0",
            "icon": "✅", "bar": "#10b981",
            "note": "Lung fields appear clear. No acute consolidation or effusion detected. "
                    "Maintain routine clinical correlation.",
        },
        "Pneumonia": {
            "color": "#d97706", "bg": "#fffbeb", "border": "#fde68a",
            "icon": "⚠️", "bar": "#f59e0b",
            "note": "Consolidation and/or infiltrative pattern detected. Sputum culture, CBC, "
                    "and antibiotic evaluation advised.",
        },
        "High-Risk / COVID-19": {
            "color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca",
            "icon": "🚨", "bar": "#ef4444",
            "note": "Bilateral ground-glass opacities / acute pathology markers detected. "
                    "Immediate isolation and confirmatory RT-PCR recommended.",
        },
    }

    s = BADGE_STYLES.get(predicted_class, BADGE_STYLES["Pneumonia"])

    # Build horizontal probability bars HTML
    bar_rows = ""
    for cls_name in TARGET_CLASSES:
        p = probs_dict.get(cls_name, 0.0) * 100.0
        cls_style = BADGE_STYLES.get(cls_name, s)
        bar_color = cls_style["bar"]
        bar_rows += f"""
        <div style="margin-bottom:6px;">
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;
                      font-weight:600;color:#334155;margin-bottom:2px;">
            <span>{cls_name}</span><span>{p:.1f}%</span>
          </div>
          <div style="background:#e2e8f0;border-radius:6px;height:10px;overflow:hidden;">
            <div style="width:{p:.1f}%;height:100%;background:{bar_color};
                        border-radius:6px;transition:width 0.5s ease;"></div>
          </div>
        </div>"""

    # Vitals alerts sub-section
    alerts_html = ""
    if warnings_list:
        items = "".join(
            f'<li style="margin-bottom:2px;color:#92400e;font-size:0.8rem;">{w}</li>'
            for w in warnings_list
        )
        alerts_html = f"""
        <div style="margin-top:10px;background:#fffbeb;border:1px solid #fde68a;
                    border-radius:6px;padding:8px 12px;">
          <strong style="color:#92400e;font-size:0.8rem;">Atypical Vitals Alert</strong>
          <ul style="margin:4px 0 0 16px;padding:0;">{items}</ul>
        </div>"""

    badge_html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
      <!-- KPI Status Card -->
      <div style="background:{s['bg']};border:1.5px solid {s['border']};border-radius:10px;
                  padding:16px 20px;margin-bottom:14px;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.7rem;">{s['icon']}</span>
            <div>
              <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;
                          color:#64748b;font-weight:700;">Primary Assessment</div>
              <div style="font-size:1.4rem;font-weight:800;color:{s['color']};">
                {predicted_class}</div>
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;
                        color:#64748b;font-weight:700;">Confidence</div>
            <div style="font-size:1.4rem;font-weight:800;color:#0f172a;">{top_prob:.1f}%</div>
          </div>
        </div>
      </div>

      <!-- Risk Distribution Bars -->
      <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
                  padding:14px 18px;margin-bottom:14px;">
        <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;
                    color:#64748b;font-weight:700;margin-bottom:8px;">
          Risk Distribution</div>
        {bar_rows}
      </div>

      <!-- Clinical Summary Note -->
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                  padding:14px 18px;">
        <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;
                    color:#64748b;font-weight:700;margin-bottom:6px;">
          Clinical Guidance:</div>
        <p style="margin:0;font-size:0.88rem;color:#334155;line-height:1.5;">
          {s['note']}</p>
        {alerts_html}
      </div>
    </div>
    """

    # 6. Silent background archiving (no UI clutter)
    vitals_record = {
        "age": age, "sex": sex, "spo2": spo2,
        "smoking": smoking, "cough": cough, "fever": fever,
    }
    pred_record = {
        "predicted_class": predicted_class,
        "confidence": probs_dict.get(predicted_class, 0.0),
        "probabilities": probs_dict,
    }
    threading.Thread(
        target=_archive_silently,
        args=(pil_image, overlay_img, vitals_record, pred_record),
        daemon=True,
    ).start()

    return overlay_img, pil_image, probs_dict, badge_html


# ---------------------------------------------------------------------------
# Custom Clinical Dashboard CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────────── */
footer { display: none !important; }
.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #f1f5f9 !important;
}

/* ── Header Bar ─────────────────────────────────────────────────── */
#header-bar {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    border-radius: 12px !important;
    padding: 20px 28px !important;
    margin-bottom: 16px !important;
    color: #f8fafc !important;
}
#header-bar h1, #header-bar p, #header-bar span {
    color: #f8fafc !important;
}

/* ── Column Cards ───────────────────────────────────────────────── */
.intake-card, .results-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px 22px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* ── Section Headers (fix white-on-white contrast bug) ──────────── */
.section-title {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-bottom: 4px !important;
}
.section-title * {
    color: #0f172a !important;
}

/* Force all Gradio label text to high contrast */
.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .block label span {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Slider value readout */
.gradio-container input[type=range] + span,
.gradio-container .rangeSlider .value {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* Input borders */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .dropdown {
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 6px !important;
}

/* ── Primary CTA Button ─────────────────────────────────────────── */
#btn-analyze {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
    padding: 12px 0 !important;
}
#btn-analyze:hover {
    background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
    transform: translateY(-1px);
}

/* ── Disclaimer Banner ──────────────────────────────────────────── */
.disclaimer-banner {
    background: #fffbeb !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    font-size: 0.82rem !important;
    color: #92400e !important;
    line-height: 1.5 !important;
    margin-top: 10px !important;
}

/* ── Examples Section ───────────────────────────────────────────── */
.examples-section {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-top: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.examples-section * {
    color: #334155 !important;
}

/* ── Tabs Styling ───────────────────────────────────────────────── */
.gradio-container .tabs .tab-nav button {
    color: #475569 !important;
    font-weight: 600 !important;
}
.gradio-container .tabs .tab-nav button.selected {
    color: #0284c7 !important;
    border-bottom-color: #0284c7 !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio Blocks Application Layout
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    """Build and assemble the RayXplain Gradio Blocks application interface."""

    _THEME = gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
    )

    blocks_kwargs: Dict[str, Any] = {
        "title": "RayXplain | Multi-Modal Diagnostic Assistant",
    }

    # Gradio < 6 passes css and theme to Blocks; >= 6 passes them to launch()
    try:
        major = int(gr.__version__.split(".")[0])
    except Exception:
        major = 4

    if major < 6:
        blocks_kwargs["css"] = CUSTOM_CSS
        blocks_kwargs["theme"] = _THEME

    with gr.Blocks(**blocks_kwargs) as demo:
        # ── Header Bar ────────────────────────────────────────────
        gr.HTML(
            f"""
            <div id="header-bar" style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
                 border-radius:12px;padding:20px 28px;margin-bottom:14px;">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
                <div>
                  <h1 style="margin:0;font-size:1.55rem;font-weight:800;color:#f8fafc;letter-spacing:-0.01em;">
                    🩺 RayXplain</h1>
                  <p style="margin:4px 0 0;font-size:0.88rem;color:#94a3b8;font-weight:500;">
                    Multi-Modal Pulmonary Diagnostic Assistant &mdash;
                    Dual-Branch ResNet-50 + Clinical MLP with Grad-CAM Explainability</p>
                </div>
                <div style="background:#1e3a5f;border:1px solid #334155;border-radius:8px;
                            padding:6px 14px;display:flex;align-items:center;gap:6px;margin-top:4px;">
                  <span style="width:8px;height:8px;border-radius:50%;background:#34d399;
                              display:inline-block;"></span>
                  <span style="font-size:0.78rem;font-weight:600;color:#cbd5e1;">
                    System Ready &bull; {_DEVICE_BADGE}</span>
                </div>
              </div>
            </div>
            """
        )

        # ── Two-Column Layout ─────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT COLUMN: Patient Case Intake ──────────────────
            with gr.Column(scale=5, elem_classes=["intake-card"]):
                gr.HTML(
                    '<p class="section-title" style="color:#0f172a !important;'
                    'font-weight:700;font-size:1.1rem;margin:0 0 8px;">📋 '
                    "Patient Case Intake &amp; Clinical Vitals</p>"
                )

                image_input = gr.Image(
                    type="pil",
                    label="Chest Radiograph (PA / AP Projection)",
                    sources=["upload", "clipboard"],
                    height=280,
                )

                with gr.Row():
                    age_input = gr.Slider(
                        minimum=1, maximum=100, value=45, step=1,
                        label="Patient Age (Years)",
                    )
                    sex_input = gr.Radio(
                        choices=["Male", "Female"], value="Male",
                        label="Biological Sex",
                    )

                with gr.Row():
                    spo2_input = gr.Slider(
                        minimum=70, maximum=100, value=96, step=1,
                        label="SpO₂ Oxygen Saturation (%)",
                    )
                    fever_input = gr.Number(
                        value=98.6, minimum=85.0, maximum=115.0, step=0.1,
                        label="Body Temperature (°F)",
                    )

                with gr.Row():
                    smoking_input = gr.Number(
                        value=0.0, minimum=0.0, step=0.5,
                        label="Smoking History (Pack-Years)",
                    )
                    cough_input = gr.Dropdown(
                        choices=["None", "Mild", "Moderate", "Severe"],
                        value="None",
                        label="Cough Severity",
                    )

                analyze_btn = gr.Button(
                    "🔍 Analyze Patient Radiograph",
                    variant="primary",
                    size="lg",
                    elem_id="btn-analyze",
                )

            # ── RIGHT COLUMN: Diagnostic Assessment ───────────────
            with gr.Column(scale=6, elem_classes=["results-card"]):
                gr.HTML(
                    '<p class="section-title" style="color:#0f172a !important;'
                    'font-weight:700;font-size:1.1rem;margin:0 0 8px;">🔬 '
                    "Diagnostic Assessment &amp; Explainability</p>"
                )

                # Dynamic KPI + Probability Bars + Clinical Summary (all in one HTML output)
                badge_output = gr.HTML(
                    value="""
                    <div style="background:#f8fafc;border:1.5px dashed #cbd5e1;border-radius:10px;
                                padding:28px 20px;text-align:center;color:#64748b;font-size:0.92rem;">
                        <span style="font-size:1.6rem;display:block;margin-bottom:6px;">📤</span>
                        Awaiting patient case submission.<br>
                        Upload a chest radiograph and enter clinical vitals to begin analysis.
                    </div>
                    """
                )

                # Label output for Gradio's native bar-chart rendering
                risk_label_output = gr.Label(
                    num_top_classes=3,
                    label="Class Risk Probabilities",
                )

                # Tabbed image comparison: Grad-CAM Overlay vs Original
                with gr.Tabs():
                    with gr.TabItem("🔥 Grad-CAM Saliency Overlay"):
                        overlay_image_output = gr.Image(
                            label="Class-Specific Activation Overlay (ResNet-50 layer4)",
                            type="numpy",
                            height=360,
                        )
                    with gr.TabItem("🖼️ Original Radiograph"):
                        original_image_output = gr.Image(
                            label="Uploaded Chest Radiograph",
                            type="pil",
                            height=360,
                        )

                # Clinical Disclaimer
                gr.HTML(
                    """
                    <div class="disclaimer-banner" style="background:#fffbeb;border-left:4px solid #f59e0b;
                         border-radius:6px;padding:12px 16px;margin-top:12px;font-size:0.82rem;
                         color:#92400e;line-height:1.5;">
                        ⚠️ <strong>Research &amp; Clinical Decision-Support Prototype.</strong>
                        Not certified for autonomous diagnosis.
                        All risk probabilities and Grad-CAM heatmaps are provided exclusively
                        as assistive tools and must be verified by a licensed radiologist or
                        medical professional.
                    </div>
                    """
                )

        # ── Quick Presets (Examples) ──────────────────────────────
        if EXAMPLE_CASES:
            with gr.Row():
                with gr.Column(elem_classes=["examples-section"]):
                    gr.HTML(
                        '<p style="color:#0f172a !important;font-weight:700;'
                        'font-size:1.0rem;margin:0 0 6px;">🧪 Quick Demo Presets '
                        "<span style='font-weight:400;font-size:0.82rem;color:#64748b;'>"
                        "(Real Clinical Radiographs)</span></p>"
                    )
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

        # ── Event Wiring ──────────────────────────────────────────
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


# Create top-level demo instance for WSGI/ASGI/Gradio CLI hosting
demo = create_app()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RayXplain: Launching Clinical Dashboard...")
    print("=" * 70 + "\n")

    launch_kwargs: Dict[str, Any] = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
    }

    try:
        major = int(gr.__version__.split(".")[0])
    except Exception:
        major = 4
    if major >= 6:
        launch_kwargs["css"] = CUSTOM_CSS
        launch_kwargs["theme"] = gr.themes.Soft(
            primary_hue="sky",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[
                gr.themes.GoogleFont("Inter"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
        )
    launch_kwargs["share"] = True
    demo.launch(**launch_kwargs)
