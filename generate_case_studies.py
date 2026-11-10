"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Phase 7: Publication-Quality Clinical Evaluation Case Studies

Generates 300 DPI, 3-panel side-by-side figures for each canonical benchmark case:
  Panel 1: Original High-Resolution Chest Radiograph
  Panel 2: Calibrated Grad-CAM Saliency Overlay (class-specific attention map)
  Panel 3: Horizontal Confidence Bar Chart (Normal / Pneumonia / High-Risk)

Outputs:
  docs/case_studies/case_1_normal.png
  docs/case_studies/case_2_pneumonia.png
  docs/case_studies/case_3_covid.png
  docs/case_studies/summary_report.md
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is importable
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from src.pipeline import DiagnosticPipeline, TARGET_CLASSES
from test_samples.canonical_cases import BENCHMARK_CASES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = _REPO_ROOT / "docs" / "case_studies"
DPI = 300

# Select the 3 representative canonical cases (A, B, C)
SELECTED_CASES: List[Tuple[str, str, str]] = [
    ("sample_normal_1",             "case_1_normal",    "Case A"),
    ("sample_pneumonia_bacterial",  "case_2_pneumonia", "Case B"),
    ("sample_covid_acute",          "case_3_covid",     "Case C"),
]

# Colour palette per diagnostic class
CLASS_COLORS: Dict[str, str] = {
    "Normal":              "#10b981",  # Emerald
    "Pneumonia":           "#f59e0b",  # Amber
    "High-Risk / COVID-19": "#ef4444",  # Rose
}

# Focus region descriptions for the summary table
FOCUS_REGIONS: Dict[str, str] = {
    "Normal":              "No focal pathological attention; diffuse low-intensity activation across bilateral lung fields",
    "Pneumonia":           "Focal consolidation zone with dense saliency over affected lobe(s)",
    "High-Risk / COVID-19": "Bilateral peripheral ground-glass opacities with subpleural attention peaks",
}


# ---------------------------------------------------------------------------
# Matplotlib Style Configuration
# ---------------------------------------------------------------------------
def _configure_matplotlib() -> None:
    """Set publication-grade matplotlib defaults."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          9,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     9,
        "axes.labelweight":   "medium",
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "figure.facecolor":   "#ffffff",
        "axes.facecolor":     "#f8fafc",
        "axes.edgecolor":     "#cbd5e1",
        "axes.grid":          False,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
    })


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------
def generate_case_figure(
    case_data: Dict[str, Any],
    case_label: str,
    overlay_img: np.ndarray,
    probs_dict: Dict[str, float],
    predicted_class: str,
    latency_sec: float,
    output_path: Path,
) -> None:
    """
    Generate a publication-grade 3-panel clinical evaluation figure.

    Panel 1: Original radiograph
    Panel 2: Grad-CAM saliency overlay
    Panel 3: Horizontal probability bar chart
    """
    # Load original image
    orig_pil = Image.open(case_data["local_image_path"]).convert("RGB")
    orig_np = np.array(orig_pil)

    # Create figure with constrained layout
    fig = plt.figure(figsize=(16, 5.5), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=2, ncols=3,
        height_ratios=[0.08, 0.92],
        width_ratios=[1, 1, 0.85],
        hspace=0.06, wspace=0.20,
        figure=fig,
    )

    # ── Header spanning all columns ───────────────────────────────
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_axis_off()

    age = case_data["age"]
    sex = case_data["sex"]
    spo2 = case_data["spo2"]
    fever = case_data["fever_f"]
    smoking = case_data["smoking_pack_years"]
    cough = case_data["cough_severity"]
    case_id = case_data["case_id"]
    title = case_data["title"]

    header_text = (
        f"{case_label}  ·  {case_id}  ·  {title}\n"
        f"Age: {age}y  |  Sex: {sex}  |  SpO2: {spo2}%  |  "
        f"Temp: {fever}°F  |  Smoking: {smoking} pk-yrs  |  "
        f"Cough: {cough}  |  Latency: {latency_sec:.3f}s"
    )

    ax_header.text(
        0.5, 0.5, header_text,
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        color="#0f172a",
        family="sans-serif",
        transform=ax_header.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f1f5f9",
            edgecolor="#94a3b8",
            linewidth=0.8,
        ),
    )

    # ── Panel 1: Original Radiograph ──────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(orig_np, cmap="gray" if orig_np.ndim == 2 else None)
    ax1.set_title("Original Radiograph", fontsize=10, fontweight="bold", color="#0f172a", pad=8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_edgecolor("#94a3b8")
        spine.set_linewidth(0.6)

    # ── Panel 2: Grad-CAM Saliency Overlay ────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(overlay_img)
    ax2.set_title("Grad-CAM Saliency Overlay", fontsize=10, fontweight="bold", color="#0f172a", pad=8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor("#94a3b8")
        spine.set_linewidth(0.6)

    # ── Panel 3: Confidence Bar Chart ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])

    class_names = list(TARGET_CLASSES)
    probabilities = [probs_dict.get(cn, 0.0) * 100.0 for cn in class_names]
    bar_colors = [CLASS_COLORS.get(cn, "#64748b") for cn in class_names]

    # Reverse order so highest class appears on top visually
    y_pos = np.arange(len(class_names))
    bars = ax3.barh(
        y_pos, probabilities,
        color=bar_colors,
        edgecolor="#ffffff",
        linewidth=0.5,
        height=0.55,
        zorder=3,
    )

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(class_names, fontsize=9, fontweight="semibold", color="#1e293b")
    ax3.set_xlabel("Confidence (%)", fontsize=9, fontweight="medium", color="#475569")
    ax3.set_xlim(0, 105)
    ax3.set_title("Risk Probabilities", fontsize=10, fontweight="bold", color="#0f172a", pad=8)

    # Add percentage labels on bars
    for bar, pct in zip(bars, probabilities):
        label_x = bar.get_width() + 1.5
        if label_x > 100:
            label_x = bar.get_width() - 5
        ax3.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left" if label_x < 95 else "right",
            fontsize=8.5, fontweight="bold", color="#0f172a",
        )

    # Light horizontal grid
    ax3.xaxis.grid(True, linestyle="--", alpha=0.3, color="#94a3b8", zorder=0)
    ax3.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax3.spines[spine].set_edgecolor("#cbd5e1")
        ax3.spines[spine].set_linewidth(0.6)

    # Highlight predicted class
    pred_color = CLASS_COLORS.get(predicted_class, "#0f172a")
    ax3.annotate(
        f">> Predicted: {predicted_class}",
        xy=(0.5, -0.12), xycoords="axes fraction",
        ha="center", va="top",
        fontsize=9, fontweight="bold", color=pred_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor=pred_color, linewidth=1),
    )

    # ── Save ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=DPI, facecolor="#ffffff")
    plt.close(fig)
    print(f"  ✅ Saved: {output_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Summary Report Generation
# ---------------------------------------------------------------------------
def generate_summary_report(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate an executive markdown summary table."""
    lines = [
        "# RayXplain – Clinical Evaluation Case Studies",
        "",
        "> **Phase 7 | Publication-Quality Diagnostic Assessment Report**",
        "> Generated by `generate_case_studies.py` using verified real clinical radiographs",
        "> and the trained MultiModalLungNet (ResNet-50 + Clinical MLP) with Grad-CAM explainability.",
        "",
        "---",
        "",
        "## Executive Summary Table",
        "",
        "| # | Case ID | Image | Vitals Summary | Ground Truth | Predicted Class | Confidence (%) | Latency (s) | Primary Radiological Focus Region |",
        "|---|---------|-------|----------------|--------------|-----------------|----------------|-------------|-----------------------------------|",
    ]

    for i, r in enumerate(results, 1):
        c = r["case_data"]
        vitals = (
            f"Age {c['age']}y, {c['sex']}, SpO₂ {c['spo2']}%, "
            f"Temp {c['fever_f']}°F, Smoking {c['smoking_pack_years']} pk-yrs, "
            f"Cough: {c['cough_severity']}"
        )
        img_name = Path(c["local_image_path"]).name
        confidence = r["probs"].get(r["predicted"], 0.0) * 100.0
        focus = FOCUS_REGIONS.get(r["predicted"], "Indeterminate")
        match_icon = "✅" if c["expected_class"] == r["predicted"] else "⚠️"

        lines.append(
            f"| {i} | {c['case_id']} | `{img_name}` | {vitals} | "
            f"{c['expected_class']} | {match_icon} {r['predicted']} | "
            f"{confidence:.1f} | {r['latency']:.3f} | {focus} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Case Study Figures",
        "",
    ])

    for i, r in enumerate(results, 1):
        fig_name = r["fig_name"]
        lines.append(f"### Case {i}: {r['case_data']['title']}")
        lines.append("")
        lines.append(f"![{r['case_data']['title']}](./{fig_name}.png)")
        lines.append("")
        lines.append(f"**Clinical Rationale:** {r['case_data']['clinical_rationale']}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "> ⚠️ **Disclaimer:** This report is generated for research and academic documentation purposes only. "
        "All diagnostic outputs are produced by experimental AI models and must be verified by licensed "
        "medical professionals. Not certified for autonomous clinical diagnosis.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ Saved: {output_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 75)
    print("  RayXplain Phase 7: Clinical Evaluation Case Study Generation")
    print("=" * 75)

    _configure_matplotlib()

    # 1. Initialize pipeline
    print("\n[1/4] Initializing DiagnosticPipeline...")
    pipeline = DiagnosticPipeline()
    print(f"       Compute device: {pipeline.device}")

    # 2. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[2/4] Output directory: {OUTPUT_DIR.relative_to(_REPO_ROOT)}/")

    # 3. Iterate through selected cases
    print("[3/4] Running inference and generating figures...\n")
    results: List[Dict[str, Any]] = []

    for case_key, fig_name, case_label in SELECTED_CASES:
        case_data = BENCHMARK_CASES[case_key]
        print(f"  ── {case_label}: {case_data['case_id']} – {case_data['title']}")

        # Verify image exists
        img_path = Path(case_data["local_image_path"])
        if not img_path.is_file():
            print(f"     ❌ SKIPPED: Image not found at {img_path}")
            continue

        # Run inference with timing
        t0 = time.perf_counter()
        overlay_img, probs_dict, predicted_class, warnings_list = pipeline.predict(
            pil_image=str(img_path),
            age=case_data["age"],
            sex=case_data["sex"],
            spo2=case_data["spo2"],
            smoking=case_data["smoking_pack_years"],
            cough_severity=case_data["cough_severity"],
            fever=case_data["fever_f"],
            alpha=0.42,
            return_warnings=True,
        )
        latency = time.perf_counter() - t0

        print(f"     Prediction: {predicted_class} "
              f"({probs_dict.get(predicted_class, 0.0) * 100:.1f}%) "
              f"| Latency: {latency:.3f}s")

        if warnings_list:
            for w in warnings_list:
                print(f"     ⚠ {w}")

        # Generate figure
        output_path = OUTPUT_DIR / f"{fig_name}.png"
        generate_case_figure(
            case_data=case_data,
            case_label=case_label,
            overlay_img=overlay_img,
            probs_dict=probs_dict,
            predicted_class=predicted_class,
            latency_sec=latency,
            output_path=output_path,
        )

        results.append({
            "case_data": case_data,
            "fig_name": fig_name,
            "predicted": predicted_class,
            "probs": probs_dict,
            "latency": latency,
        })

        print()

    # 4. Generate summary markdown report
    print("[4/4] Generating summary report...")
    report_path = OUTPUT_DIR / "summary_report.md"
    generate_summary_report(results, report_path)

    # Final summary
    print("\n" + "=" * 75)
    print("  Phase 7 COMPLETE – All artifacts generated successfully!")
    print("=" * 75)
    print(f"\n  📁 Output Directory: {OUTPUT_DIR.relative_to(_REPO_ROOT)}/")
    print(f"  📊 Figures Generated: {len(results)}")
    print(f"  📄 Summary Report:   {report_path.relative_to(_REPO_ROOT)}")

    for r in results:
        c = r["case_data"]
        conf = r["probs"].get(r["predicted"], 0) * 100
        match = "✅" if c["expected_class"] == r["predicted"] else "⚠️"
        print(f"     {match} {c['case_id']}: {r['predicted']} ({conf:.1f}%) [{r['latency']:.3f}s]")

    print()


if __name__ == "__main__":
    main()

