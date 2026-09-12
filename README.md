# 🩺 diabeticRetino — AI-Assisted Diabetic Retinopathy Screening for Rural India

> **SIH 2026 | Problem Statement: SIH26038 | Vision for Every Village**

An offline-first, AI-assisted retinal screening and tele-ophthalmology prototype designed for **rural and resource-constrained healthcare environments**.

The system combines retinal image quality assessment, fundus preprocessing, lesion analysis, DR severity staging, DME/CSME proximity analysis, uncertainty-aware guardrails, multi-agent clinical validation, risk-based triage, multilingual reporting, offline patient records, edge inference, and large-scale rural workflow simulation.

> ⚠️ **Medical disclaimer:** This repository is a research/prototype screening system. It is not a certified medical device and must not be used as a substitute for examination or diagnosis by a qualified ophthalmologist.

---

## 🎯 Problem

Diabetic Retinopathy (DR) can cause preventable vision loss when detection and referral are delayed. Rural screening introduces additional constraints:

- Limited access to ophthalmologists
- Variable-quality fundus photographs
- Low-cost imaging hardware
- Intermittent internet connectivity
- Specialist capacity bottlenecks
- Language and health-literacy barriers

**diabeticRetino** is designed around an edge-first workflow: perform as much screening and evidence generation as possible at the local PHC, while routing uncertain or high-risk cases to human specialists.

---

## 🧠 What the System Does

```text
Fundus Image
     │
     ▼
Low-Cost Hardware Correction
     │
     ▼
Retinal FOV + Preprocessing
     │
     ▼
Image Quality Assessment (IQA)
     │
     ├── Poor Quality ──► Recapture Guidance
     │
     ▼
DR Stage Estimation + Lesion Analysis
     │
     ├── Hemorrhage Detection
     ├── Exudate Detection
     ├── DME / CSME Proximity
     └── Uncertainty Estimation
     │
     ▼
Explainable / Multimodal Evidence
     │
     ▼
Multi-Agent Validation
     │
     ├── ACCEPT
     ├── ABSTAIN
     └── HUMAN REVIEW
     │
     ▼
Clinical Risk Triage
     │
     ├── Routine
     ├── Referable
     └── Urgent
     │
     ▼
Multilingual Clinical PDF Report
     │
     ▼
Offline SQLite Record + Sync Queue
```

---

# ✨ Core Features

## 1. 📷 Low-Cost Hardware Adaptation

The application supports hardware profiles intended for rural fundus acquisition:

- **Remidio Fundus on Phone (FOP)**
- **MII RetCam / 20D Lens Attachment**
- **Standard Desktop Fundus Camera**

The hardware adaptation layer includes:

- Barrel distortion correction
- Central LED illumination hotspot suppression
- Chromatic aberration correction
- Optical field alignment

This allows the software pipeline to account for differences between acquisition devices.

---

## 2. 🧹 Retinal Image Preprocessing

`modules/preprocessing.py`

The preprocessing pipeline performs:

- Retinal circular Field-of-View (FOV) extraction
- Green-channel extraction
- CLAHE contrast enhancement
- Illumination/color normalization
- Bilateral noise reduction

The goal is to isolate the clinically relevant retinal region before downstream analysis.

---

## 3. 🚦 Image Quality Assessment (IQA)

`modules/iqa.py`

The **Quality Gate** prevents poor-quality images from proceeding blindly through the diagnostic pipeline.

It evaluates:

- Image sharpness / blur
- Under-exposure
- Over-exposure
- Retinal FOV coverage
- Optic disc visibility
- Fovea/macula landmark visibility

A composite quality score is generated and the system provides actionable recapture guidance for a rural health worker.

### Example workflow

```text
Image
  │
  ▼
Quality Score
  │
  ├── Acceptable ──► Continue
  │
  └── Suboptimal ──► Recapture
```

---

## 4. 🔬 DR Severity Staging

The system uses a five-level DR severity representation:

| Stage | Classification | Clinical Representation |
|---:|---|---|
| 0 | No DR | No abnormalities |
| 1 | Mild NPDR | Microaneurysms |
| 2 | Moderate NPDR | Microaneurysms, hemorrhages and/or hard exudates |
| 3 | Severe NPDR | Severe hemorrhages, venous beading / IRMA pattern |
| 4 | Proliferative DR | Neovascularization and/or vitreous/preretinal hemorrhage |

The repository contains a PyTorch convolutional backbone and an ONNX representation for edge inference. The current screening orchestrator additionally applies **lesion-derived clinical calibration rules** to generate stage probabilities.

> **Implementation note:** The current repository should be considered a prototype/hybrid screening implementation rather than a clinically validated trained DR classifier. The included CNN/ONNX model architecture is present for the edge/MATLAB workflow, while the active `predict_dr_pipeline()` combines quantitative lesion evidence with rule-based severity calibration.

---

## 5. 🩸 Retinal Lesion Analysis

`modules/ai_models.py`

The lesion-analysis component uses classical computer-vision techniques to identify candidate:

- Hemorrhages / microaneurysm-like dark lesions
- Hard exudate-like bright lesions

It calculates:

- Hemorrhage count
- Exudate count
- Total lesion count
- Hemorrhage area percentage
- Exudate area percentage
- Lesion masks

The optic disc is excluded from candidate exudate regions to reduce false positives.

---

## 6. 🎯 DME / CSME Risk Analysis

The system estimates the proximity of exudates to the fovea using **Disc Diameter (DD)** as the reference scale.

```text
Fovea
  │
  ├── ≤ 1 DD  → High / CSME risk
  │
  ├── 1–2 DD  → Moderate risk
  │
  └── > 2 DD  → Low / peripheral risk
```

The interface also visualizes a **1-DD macular danger zone** to make the proximity evidence easier to verify.

---

## 7. 🧠 Uncertainty-Aware Screening

The system decomposes uncertainty into two prototype components:

### Epistemic uncertainty
Represents ambiguity around the predicted DR stage.

### Aleatoric uncertainty
Represents image/data-related noise such as:

- Blur
- Exposure problems
- Optical/media quality

The pipeline combines these values into a total uncertainty estimate and can require human verification when confidence or uncertainty crosses configured safety thresholds.

---

## 8. 👁️ Multi-Agent Clinical Guardrail

`modules/multi_agent.py`

Two logical agents participate in the validation layer:

### Agent 1 — Vision Analyst

Converts computer-vision output into structured evidence, including:

- Predicted DR stage
- Confidence
- Lesion counts
- Lesion burden
- DME status
- Image-quality information

### Agent 2 — Clinical Validator

Acts as a rule-based clinical consistency checker and looks for contradictions between the predicted stage and quantitative biomarkers.

### Consensus states

```text
                 ┌──────────────┐
                 │ AI Evidence  │
                 └──────┬───────┘
                        │
              ┌─────────▼─────────┐
              │ Clinical Validator│
              └─────────┬─────────┘
                        │
             ┌──────────▼──────────┐
             │ Consensus Guardrail │
             └──────┬──────┬───────┘
                    │      │
                 ACCEPT  ABSTAIN
                           │
                    HUMAN REVIEW
```

The guardrail is intentionally designed so that the system can **abstain instead of forcing a confident-looking answer** when evidence is insufficient or contradictory.

---

## 9. 🩺 Clinical Risk Triage

`modules/triage.py`

The triage engine combines:

- DR severity
- DME/CSME status
- Consensus/guardrail state
- Patient information

It produces a risk/urgency category and a recommended referral window.

Typical categories include:

- **Low Risk / Routine follow-up**
- **Referable DR**
- **High Risk / Urgent**
- **Inconclusive / Human Review**

The system is designed to prioritize specialist attention for cases that actually need it.

---

## 10. 🔎 Explainable & Multimodal Evidence

`modules/explainability.py`

The UI provides visual evidence including:

- Grad-CAM-style attention visualization
- Lesion overlays
- Hemorrhage masks
- Exudate masks
- 1-DD macular danger zone
- Composite multimodal visualization
- Quantitative biomarker cards

The objective is to let a clinician inspect **what evidence contributed to the screening result**, rather than presenting only a classification label.

> Note: the current Grad-CAM-style implementation is an image-processing/evidence visualization prototype; it is not a gradient extraction from the live CNN weights.

---

## 11. 📴 Offline-First Rural Database

`modules/database.py`

The system uses **SQLite** for local PHC operation.

Stored information includes:

### Patient information

- Patient ID
- Name
- Age
- Gender
- PHC location
- Diabetes duration
- HbA1c
- Systolic BP
- Contact information
- Timestamp

### Screening information

- Image-quality results
- DR prediction
- Confidence/uncertainty
- Lesion evidence
- DME status
- Consensus result
- Triage result
- Sync status

The database also contains a **priority sync queue** so locally stored cases can be synchronized when connectivity becomes available.

---

## 12. 🌐 Telemedicine / Connectivity Model

The application supports an offline-first workflow:

```text
              Rural PHC
                  │
        ┌─────────▼─────────┐
        │ Local Edge AI     │
        │ + SQLite          │
        └─────────┬─────────┘
                  │
           No Internet?
                  │
             Save Locally
                  │
          Connectivity Returns
                  │
                  ▼
          Priority Sync Queue
                  │
                  ▼
          Specialist Workflow
```

Urgent cases can be prioritized for transmission when the network becomes available.

---

## 13. 📄 Multilingual Clinical Reports

`modules/reports.py`

The system generates structured PDF screening reports using **ReportLab**.

Supported patient-advice languages:

- 🇬🇧 English
- 🇮🇳 Hindi
- 🇮🇳 Tamil
- 🇮🇳 Telugu

Reports can contain:

- Patient details
- Vitals
- DR stage
- Confidence
- Uncertainty
- Lesion biomarkers
- DME/CSME status
- Consensus/guardrail result
- Clinical triage
- Referral recommendation
- Local-language patient advice

---

## 14. ⚡ ONNX Edge Inference

`modules/onnx_engine.py`

The repository includes an ONNX Runtime CPU engine designed for local inference without a cloud dependency.

Capabilities include:

- PyTorch → ONNX export
- ONNX model validation
- CPU inference using ONNX Runtime
- Runtime latency measurement
- 256 × 256 model input
- MATLAB-compatible ONNX export

The Streamlit UI allows switching between:

- **ONNX Runtime Edge CPU**
- **PyTorch Neural Backbone**

This supports experimentation with lightweight deployment on rural laptops/tablets.

---

## 15. 📊 Rural Population Scalability Simulation

`modules/simulation.py`

The project contains a discrete-event-style rural workflow simulation for large screening cohorts.

The simulation models:

- Patient arrivals
- Multiple PHCs
- Specialist capacity
- AI triage
- Abstention
- Image recapture
- Specialist referrals
- Waiting times
- Workload reduction
- Telemedicine prioritization

The UI can simulate cohorts up to **250,000 patients/year**, with a default scenario of **100,000 patients/year**.

It compares:

```text
Traditional Workflow
Every case → Specialist Queue

             VS

AI-Assisted Workflow
Every case → Local AI
               │
       ┌───────┴────────┐
       ▼                ▼
 Local clearance    Refer / Urgent /
                    Human review
```

---

# 🧮 MathWorks Integration

The repository includes MATLAB integration for the SIH/MathWorks evaluation workflow.

## `matlab/load_dr_onnx_matlab.m`

Demonstrates:

- Loading the ONNX DR model
- MATLAB Deep Learning Toolbox integration
- Retinal image processing
- Model evaluation
- MATLAB-side workflow experimentation

## `matlab/dr_rural_simulink.m`

Provides a MATLAB/Simulink-oriented scalability model for:

- Rural PHC screening
- Specialist queues
- Stateflow-style guardrails
- Telemedicine workflow
- Large-scale annual patient volumes

---

# 🗂️ Project Structure

```text
diabeticRetino/
│
├── app.py
├── test_pipeline.py
│
├── models/
│   └── dr_classifier.onnx
│
├── modules/
│   ├── ai_models.py
│   ├── database.py
│   ├── explainability.py
│   ├── hardware_profiles.py
│   ├── iqa.py
│   ├── multi_agent.py
│   ├── onnx_engine.py
│   ├── preprocessing.py
│   ├── reports.py
│   ├── sample_data.py
│   ├── simulation.py
│   └── triage.py
│
├── matlab/
│   ├── dr_rural_simulink.m
│   └── load_dr_onnx_matlab.m
│
├── data/
│   ├── rural_screening.db
│   └── generated/sample reports
│
├── sample_cases/
│   └── generated retinal demo cases
│
└── reports/
```

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Application | Python |
| UI | Streamlit |
| Image Processing | OpenCV, NumPy |
| AI/ML Framework | PyTorch |
| Edge Runtime | ONNX Runtime |
| Model Format | ONNX |
| Data | SQLite, Pandas |
| Visualization | Altair |
| PDF Reports | ReportLab |
| Medical Image Handling | Pillow / OpenCV |
| Simulation | NumPy / Pandas |
| MATLAB Integration | MATLAB / Deep Learning Toolbox / Simulink-oriented workflow |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/mdanfaz-gif/diabeticRetino.git
cd diabeticRetino
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

The current repository does not include a `requirements.txt`, so install the runtime dependencies manually:

```bash
pip install streamlit opencv-python numpy pandas pillow altair reportlab torch onnx onnxruntime
```

If your local Python/PyTorch environment requires a platform-specific PyTorch installation, install the appropriate PyTorch build first.

---

# ▶️ Run the Application

From the repository root:

```bash
streamlit run app.py
```

Then open the Streamlit URL displayed in the terminal.

---

# 🧪 Run the Test Suite

The repository contains an end-to-end verification script:

```bash
python test_pipeline.py
```

The test suite exercises the major architectural blocks including:

- Preprocessing
- IQA
- DR pipeline
- Lesion analysis
- Multi-agent validation
- Explainability
- Triage
- Simulation
- PDF generation
- Database
- Sample-case generation
- Hardware correction
- ONNX export/inference

---

# 🧪 Demo / Sample Cases

The project contains a sample-data generator that creates synthetic fundus-style demonstration cases representing scenarios such as:

- Normal retina
- Mild NPDR
- Moderate NPDR
- Severe NPDR with macular involvement
- Proliferative DR
- Low-quality/blurry image
- Diagnostic contradiction / guardrail case

These are useful for **demonstrations and pipeline testing** without downloading an external retinal dataset.

> These generated images are demonstration inputs, not clinically validated patient data.

---

# 🔐 Privacy & Security Considerations

For real deployment, patient data must be protected using appropriate healthcare/privacy controls.

Recommended production requirements include:

- Patient de-identification
- Encryption at rest
- Secure network transmission
- Role-based access control
- Authentication
- Audit logging
- Secure backup
- Retention policies
- Explicit consent and applicable regulatory compliance

The current repository is a prototype and should not be considered production-grade clinical infrastructure.

---

# ⚠️ Important Implementation & Validation Notes

This project intentionally combines **AI concepts, computer vision, clinical rules, and systems engineering** into a working prototype.

However, several components are research/prototype implementations and require clinical validation before real-world use:

- The included DR CNN architecture is not demonstrated here as a clinically trained/validated model.
- The active DR staging pipeline uses lesion-derived heuristic calibration rules.
- Lesion detection uses classical image-processing methods rather than a trained U-Net segmentation network.
- The Grad-CAM-style visualization is an evidence visualization prototype rather than true gradient-based attribution from a trained production model.
- DME/CSME assessment is based on geometric proximity rules.
- Simulation results are model assumptions, not measured healthcare outcomes.
- The system has not been established as a medical device.

These limitations are important because **a convincing prototype should not be presented as clinically validated AI**.

---

# 🔮 Future Development

Potential next steps include:

- Training and validating a real DR classifier on representative retinal datasets
- Training a dedicated lesion-segmentation model
- Calibrating uncertainty using validated probabilistic methods
- External clinical validation across different cameras and populations
- Federated / privacy-preserving learning
- Secure cloud synchronization
- Mobile/edge deployment
- Real tele-ophthalmologist integration
- Robust authentication and authorization
- Automated monitoring and model drift detection
- Clinical regulatory evaluation
- Expanded retinal disease screening

---

# 🏆 Project Vision

**Vision for Every Village**

The long-term goal is to create a scalable screening workflow where a patient in a resource-constrained rural setting can receive:

```text
Affordable Capture
       ↓
Quality Check
       ↓
Local AI Screening
       ↓
Evidence + Explainability
       ↓
Safety Guardrails
       ↓
Risk-Based Referral
       ↓
Specialist Review
```

The key design principle is simple:

> **AI should reduce the specialist bottleneck — not remove the specialist from the loop.**

---

## 📜 License

No license has currently been specified for this repository.

If you intend to make the project open source, add an appropriate license such as MIT, Apache-2.0, or another license matching your project's requirements.

---

## 👥 Team

**Vision for Every Village**

Built for **Smart India Hackathon 2026 — SIH26038**.
