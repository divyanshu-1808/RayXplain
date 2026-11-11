# RayXplain: Multi-Modal Pulmonary Disease Diagnostic Assistant
### Dual-Branch Deep Learning (ResNet-50 + Clinical MLP) with Grad-CAM Explainable AI (XAI)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio Dashboard](https://img.shields.io/badge/UI-Gradio%20Blocks-orange.svg)](https://gradio.app/)
[![Offline First](https://img.shields.io/badge/Architecture-100%25%20Offline%20Air--Gapped-green.svg)](#offline-audit-logging-and-record-keeping)

---

## 1. Executive Summary & Problem Formulation

### The Clinical Dilemma: Unimodal Radiology Blind Spots
In emergency department triage and pulmonology clinics, chest radiographs (CXRs) are the primary diagnostic tool for respiratory diseases. However, **unimodal automated CXR models suffer from critical clinical blind spots**:

1. **Atypical Presentations & Early Consolidation:** Early-stage bacterial pneumonia or atypical viral pneumonias frequently present with subtle, faint, or equivocal radiological findings that computer vision backbones fail to distinguish from normal aeration.
2. **Visual Ambiguity Across Pathologies:** COVID-19 pneumonitis and non-COVID viral pneumonias both exhibit ground-glass opacities (GGOs) and bilateral infiltrates. An image-only classifier cannot reliably distinguish between them without systemic clinical context.
3. **Overlooked Vital Signs:** Real-world clinicians never interpret a radiograph in isolation. An equivocal subpleural haze in an asymptomatic patient ($SpO_2 = 99\%$, Temp = $98.4^\circ\text{F}$) has an entirely different differential diagnosis than the identical radiological finding in a hypoxemic patient ($SpO_2 = 84\%$, Temp = $103.2^\circ\text{F}$, heavy smoking history).

### The RayXplain Solution
**RayXplain** resolves this challenge through **Intermediate Multi-Modal Fusion**. By coupling a truncated deep convolutional backbone (ResNet-50) with an asymmetric multi-layer perceptron (MLP) for patient vitals, RayXplain mirrors clinical reasoning. The system cross-references radiological feature maps with 6 key physiological biomarkers to deliver high-confidence tri-class predictions (`Normal`, `Viral / Bacterial Pneumonia`, `High-Risk Pathological / COVID-19`) accompanied by **calibrated Grad-CAM saliency heatmaps** for transparent clinician verification.

---

## 2. System Architecture & Mathematical Specification

```
====================================================================================================
                                  RAYXPLAIN MULTI-MODAL ARCHITECTURE
====================================================================================================

      [ Modality 1: Chest Radiograph ]                    [ Modality 2: Clinical Vitals (6) ]
      Shape: (B, 3, 224, 224)                             [Age, Sex, SpO2, Smoking, Cough, Temp]
                 │                                                          │
                 ▼                                                          ▼
      ┌──────────────────────────────┐                           ┌──────────────────────────────┐
      │  ResNet-50 Feature Extractor │                           │    StandardScaler Transform  │
      │   (conv1 through layer4)     │                           │  Z = (X - mu) / sigma        │
      └──────────────┬───────────────┘                           └──────────────┬───────────────┘
                     │                                                          │
             conv_features [B, 2048, 7, 7]                                      ▼
                     │ (Hooked for Grad-CAM)                     ┌──────────────────────────────┐
                     ▼                                           │      Tabular MLP Branch      │
      ┌──────────────────────────────┐                           │  Linear(6 -> 64)             │
      │  Spatial AdaptiveAvgPool2d   │                           │  BatchNorm1d + ReLU          │
      │        (1, 1) -> Flatten     │                           │  Linear(64 -> 32)            │
      └──────────────┬───────────────┘                           │  BatchNorm1d + ReLU          │
                     │ [B, 2048]                                 │  Dropout(p=0.2)              │
                     ▼                                           └──────────────┬───────────────┘
      ┌──────────────────────────────┐                                          │
      │   Vision Projection Head     │                                          │
      │  Linear(2048 -> 128)         │                                          │
      │  BatchNorm1d + ReLU          │                                          │
      │  Dropout(p=0.3)              │                                          │
      └──────────────┬───────────────┘                                          │
                     │ [B, 128]                                                 │ [B, 32]
                     └──────────────────────────┬───────────────────────────────┘
                                                │
                                                ▼ (Intermediate Feature Concatenation)
                                 ┌──────────────────────────────┐
                                 │     Fused Latent Vector      │
                                 │         [B, 160]             │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │   Multimodal Classification  │
                                 │  Linear(160 -> 64)           │
                                 │  BatchNorm1d + ReLU          │
                                 │  Dropout(p=0.2)              │
                                 │  Linear(64 -> 3)             │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼ [B, 3] (Logits)
                               ┌────────────────┴────────────────┐
                               │                                 │
                               ▼                                 ▼
                   ┌───────────────────────┐         ┌───────────────────────┐
                   │  Softmax Probability  │         │  Grad-CAM Backprop    │
                   │  P(c|X_img, X_clin)   │         │  d(y^c) / d(layer4)   │
                   └───────────────────────┘         └───────────────────────┘
                               │                                 │
                               ▼                                 ▼
                   Top-3 Class Risk Scores            Localized Saliency Map
                   [Normal, Pneumonia, COVID]         Overlay on Original CXR
====================================================================================================
```

### Mathematical Formulation of Grad-CAM in Fused Space
Grad-CAM computes the spatial importance of each convolutional unit in ResNet-50's final residual block (`layer4`) with respect to the target class logit $y^c$.

1. **Class-Specific Neuron Importance Weights ($\alpha_k^c$):**
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}$$
   where $A^k$ is the $k$-th feature activation map of shape $U \times V$ ($7 \times 7$), $Z = U \times V = 49$, and $y^c$ is the unnormalized logit score of class $c$.

2. **Rectified Localization Map ($L_{\text{Grad-CAM}}^c$):**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{2048} \alpha_k^c A^k \right)$$
   The $\text{ReLU}$ non-linearity ensures the heatmap filters out features contributing negatively to class $c$.

3. **Min-Max Normalization & Calibrated Blending:**
   $$S_{i,j} = \frac{L_{i,j} - \min(L)}{\max(L) - \min(L) + \epsilon}$$
   $$I_{\text{blended}} = \text{uint8}\Big( \alpha \cdot I_{\text{colormap}}(255 \cdot S) + (1 - \alpha) \cdot I_{\text{original}} \Big)$$
   where $\alpha = 0.42$ defines transparency, $I_{\text{colormap}}$ utilizes OpenCV's `COLORMAP_JET` converted to RGB, and the explicit $\text{uint8}$ cast guarantees zero pitch-black image serialization artifacts.

---

## 3. Repository Directory Structure

```
d:\project 1\
├── .venv/                         # Local Python 3.11 virtual environment
├── data/
│   └── scaler.joblib              # Fitted StandardScaler for 6 clinical features
├── docs/
│   └── case_studies/              # Phase 7 publication figures & summary report
│       ├── case_1_normal.png      # 300 DPI 3-panel figure: Normal Thorax
│       ├── case_2_pneumonia.png   # 300 DPI 3-panel figure: Bacterial Pneumonia
│       ├── case_3_covid.png       # 300 DPI 3-panel figure: Acute COVID-19
│       └── summary_report.md      # Executive clinical evaluation markdown table
├── patient_records/               # Local audit trail directory (100% offline)
│   ├── .gitkeep                   # Directory placeholder
│   └── evaluations_history.csv    # Master CSV audit trail (auto-initialized)
├── src/
│   ├── __init__.py                # Package exports
│   ├── gradcam.py                 # Grad-CAM engine with backward hooks & overlay
│   ├── model.py                   # MultiModalLungNet dual-branch PyTorch module
│   ├── pipeline.py                # DiagnosticPipeline with edge cases & CPU fallback
│   └── record_keeper.py           # CaseArchiver for background audit logging
├── tests/
│   ├── test_app.py                # Gradio Blocks UI test suite
│   ├── test_gradcam.py            # Grad-CAM forward/backward hook verification
│   ├── test_model.py              # Architecture, tensor shapes, and freeze tests
│   ├── test_pipeline.py           # Preprocessing, validation, and OOM recovery tests
│   └── test_record_keeper.py      # CaseArchiver and canonical case integrity tests
├── test_samples/                  # Verified real clinical radiographs & registry
│   ├── canonical_cases.py         # Canonical benchmark cases master registry
│   ├── sample_covid_acute.jpeg    # Real COVID-19 CXR (ground-glass opacities)
│   ├── sample_covid_bilateral.jpeg# Real severe bilateral COVID-19 CXR
│   ├── sample_normal_1.jpeg       # Real normal adult thorax CXR
│   ├── sample_normal_2.jpeg       # Real healthy senior thorax CXR
│   ├── sample_pneumonia_bacterial.jpeg # Real lobar consolidation CXR
│   └── setup_offline_samples.py   # Downloader & sample integrity validator
├── weights/
│   └── best_multimodal_model.pth  # Trained model checkpoint weights (~95.5 MB)
├── app.py                         # Primary clinical dashboard (Gradio Blocks)
├── generate_case_studies.py       # Phase 7 evaluation artifact generator script
├── requirements.txt               # Pinned production dependencies
├── run_app.bat                    # One-click Windows launcher
├── test_edge_cases.py             # Phase 6 standalone edge-case testing script
└── verify_env.py                  # Hardware, dependency, and artifact verification
```

---

## 4. Installation & Local Execution Guide

### Prerequisites
- **Operating System:** Windows 10/11 (x64), Linux, or macOS.
- **Python Version:** Python 3.11.x (tested on Python 3.11.9).
- **Compute Hardware:** Compatible with any modern x86_64 CPU (inference $\sim 350\text{ms}$) or NVIDIA GPU with CUDA 12.1+ ($\sim 45\text{ms}$).

### Step 1: Clone the Repository
```bash
git clone https://github.com/divyanshu-1808/RayXplain.git
cd RayXplain
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell/CMD):
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### Step 3: Install Production Dependencies
```bash
# For GPU acceleration (NVIDIA CUDA 12.1):
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# OR for CPU-only execution:
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4: Verify Environment & Weights
Run the environment self-diagnostic script:
```bash
python verify_env.py
```
Expected output:
```
[PASS] All core dependencies verified
[PASS] Compute device detected: CUDA GPU (or CPU)
[PASS] weights/best_multimodal_model.pth verified
[PASS] data/scaler.joblib verified
```

### Step 5: Launch the Application

#### Option A: One-Click Windows Launcher (Recommended)
Double-click `run_app.bat` or run:
```cmd
run_app.bat
```
This automatically activates `.venv`, detects GPU/CPU hardware acceleration, launches the Gradio server on `http://127.0.0.1:7860`, and opens your default browser.

#### Option B: Direct Python Execution
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:7860`**

---

## 5. Offline Audit Logging & Record Keeping

RayXplain features a clinical audit logging subsystem via `src/record_keeper.py` designed for academic viva review and hospital traceability:

- **100% Background Execution:** Case archiving is delegated to an asynchronous background worker (`threading.Thread`), introducing **zero UI latency or blocking**.
- **Self-Contained Evaluation Folders:** Every manual evaluation creates a timestamped folder under `patient_records/`:
  ```
  patient_records/TEST_20261110_144804_Auto_Archive/
  ├── original_xray.png          # Exact uploaded chest radiograph
  ├── gradcam_explanation.png    # Computed Grad-CAM saliency overlay
  └── clinical_summary.json      # Structured JSON metadata (vitals, predictions, latency)
  ```
- **Master Excel-Compatible Log:** Automatically logs every case to `patient_records/evaluations_history.csv` with headers:
  `Timestamp`, `Folder_Name`, `Patient_Tag`, `Predicted_Class`, `Confidence_Percent`, `Latency_Seconds`, `Age`, `Sex`, `SpO2`, `Smoking_Pack_Years`, `Cough_Severity`, `Body_Temperature_F`, `GradCAM_Saved`.

---

## 6. Verification & Automated Test Suite

The project includes an end-to-end unit and edge-case testing harness covering all 6 development phases:

```bash
# Run the complete test suite (39 comprehensive unit tests):
python -m unittest discover tests -v
```

Test suite coverage:
- `tests/test_model.py`: Dimension assertions, batch processing, parameter freezing, and layer verification.
- `tests/test_gradcam.py`: Tensor hook execution, GAP weights, ReLU activation, context manager memory cleanup.
- `tests/test_pipeline.py`: Tabular transformation, physiological clamping, image sanitization, CUDA OOM fallback.
- `tests/test_app.py`: Gradio Blocks interface construction, validation errors, end-to-end controller flows.
- `tests/test_record_keeper.py`: Offline sample integrity, case registry query validation, CSV audit persistence.
- `test_edge_cases.py`: Standalone resilience verification against boundary-breaking inputs.

---

## 7. Generating Publication Case Studies (Phase 7)

To regenerate the high-resolution, 300 DPI evaluation figures and summary table:
```bash
python generate_case_studies.py
```
Outputs are stored in `docs/case_studies/`:
- `case_1_normal.png`: Normal adult thorax baseline (**98.7%** confidence)
- `case_2_pneumonia.png`: Community-acquired bacterial pneumonia (**64.8%** confidence)
- `case_3_covid.png`: Acute progressive COVID-19 (**98.0%** confidence)
- `summary_report.md`: Executive clinical review table

---

## 8. Statutory Clinical Disclaimer

> **IMPORTANT CLINICAL & REGULATORY NOTICE**
> 
> **RayXplain is a research and clinical decision-support prototype.** It is designed exclusively for academic demonstration, algorithmic benchmarking, and assistive exploration.
> - This software has **not been certified by the FDA, EMA, or CDSCO** as a medical device for autonomous diagnosis.
> - Diagnostic outputs, risk probabilities, and Grad-CAM localized heatmaps **must never supersede the clinical judgment of a licensed radiologist or pulmonologist**.
> - The model should not be used as the sole determinant for administering medication, oxygenation therapy, patient isolation, or intensive care intervention.

