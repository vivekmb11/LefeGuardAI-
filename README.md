# LeafGuard AI
### AI-Powered Plant Disease Detection & Out-of-Distribution Safety System

[![Python](https://img.shields.io/badge/Python-3.10+-123D2A.svg?style=flat&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-16A36A.svg?style=flat&logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV2-FF6F00.svg?style=flat&logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Quality_Guard-5C3EE8.svg?style=flat&logo=opencv)](https://opencv.org)
[![Tests](https://img.shields.io/badge/Official%20Tests-10%2F10%20Passed%20(100%25)-success.svg?style=flat)]()

> **"Don't Just Detect Diseases. Know When NOT to Predict."**

---

## 1. Project Objective & Problem Statement

**LeafGuard AI** is an agricultural vision platform engineered for potato leaf disease detection featuring Out-of-Distribution (OOD) and photometric quality defense.

Traditional deep learning classifiers operate under a naive *closed-world assumption*: given an image of a dog, car, phone, cup, or blurry photo, they forcefully output one of their known disease classes with high false confidence. In agriculture, these hallucinations lead to devastating misuse of chemical fungicides and crop damage.

**LeafGuard AI solves this** by executing a four-tier fail-safe validation pipeline:
1. **Photometric & Sharpness Guard**: Rejects blurry, dark, overexposed, or tiny images using normalized Laplacian edge variance and pixel intensity histograms.
2. **Feature Manifold OOD Detection**: Extracts 1280-dimensional deep embeddings via MobileNetV2 and calculates cosine distance to known potato leaf centroids. Rejects non-leaf / unsupported objects.
3. **Calibrated Disease Classification**: Evaluates in-distribution potato leaves across supported classes (`Healthy`, `Early Blight`, `Late Blight`).
4. **Confidence Bounds Enforcement**: Rejects uncertain predictions if softmax confidence is below safety thresholds.

**Core Safety Guarantee:** *LeafGuard AI will NEVER return a disease prediction label when an image is rejected.*

---

## 2. Supported Classes

1. **Potato Healthy**
2. **Potato Early Blight** (*Alternaria solani*)
3. **Potato Late Blight** (*Phytophthora infestans*)

---

## 3. Official 10-Item Verification Results

The complete system was evaluated against the 10 mandatory verification test categories. **All 10 tests passed (100.0%)**:

| # | Test Category | Input Image | Status | Prediction | OOD Score / Reason | Verdict |
|---|---|---|---|---|---|---|
| **1** | Healthy potato leaf | `01_healthy_potato_leaf.jpg` | **ACCEPTED** | `Healthy` | OOD: 0.2374 (< 0.39) | **PASS** |
| **2** | Early blight potato leaf | `02_early_blight_potato_leaf.jpg` | **ACCEPTED** | `Early Blight` | OOD: 0.2028 (< 0.39) | **PASS** |
| **3** | Late blight potato leaf | `03_late_blight_potato_leaf.jpg` | **ACCEPTED** | `Late Blight` | OOD: 0.2216 (< 0.39) | **PASS** |
| **4** | Car image | `04_car_image.jpg` | **REJECTED** | `None` *(Protected)* | OOD: 0.5565 (> 0.39) | **PASS** |
| **5** | Animal image | `05_animal_image.jpg` | **REJECTED** | `None` *(Protected)* | OOD: 0.6437 (> 0.39) | **PASS** |
| **6** | Mobile phone image | `06_mobile_phone_image.jpg` | **REJECTED** | `None` *(Protected)* | OOD: 0.6501 (> 0.39) | **PASS** |
| **7** | Coffee mug image | `07_coffee_mug_image.jpg` | **REJECTED** | `None` *(Protected)* | OOD: 0.5686 (> 0.39) | **PASS** |
| **8** | Other plant leaf | `08_other_plant_leaf.jpg` | **REJECTED** | `None` *(Protected)* | Exposure / OOD Guard | **PASS** |
| **9** | Blurry image | `09_blurry_image.jpg` | **REJECTED** | `None` *(Protected)* | `BLUR_DETECTED` (Var: 1.05) | **PASS** |
| **10**| Dark image | `10_dark_image.jpg` | **REJECTED** | `None` *(Protected)* | `TOO_DARK` (Mean: 14.19) | **PASS** |

**Zero Violations:** No out-of-distribution or degraded image was assigned a false disease diagnosis.

---

## 4. Architecture & Directory Structure

```text
LeafGuard/
├── app.py                      # Flask server (/predict, /stats, /health, /)
├── config.py                   # Centralized safety bounds & calibrated thresholds
├── requirements.txt            # Python dependencies
├── train_real_model.py         # MobileNetV2 transfer learning & OOD calibration script
├── prepare_10_test_cases.py    # Test case packager
├── run_10_verification_tests.py# Automated 10-item test verification runner
├── test_real_evaluation.py     # Evaluation on real held-out test splits
│
├── AI_Module/
│   ├── __init__.py
│   ├── quality_check.py        # Laplacian variance blur & photometric exposure checks
│   ├── ood_detector.py         # MobileNetV2 1280-dim embedding & cosine centroid OOD
│   ├── predict.py              # 4-stage safety inference pipeline
│   ├── model.keras             # Trained MobileNetV2 potato disease classifier
│   ├── class_names.json        # Aligned class names
│   └── ood_thresholds.json     # Calibrated class centroids and safety distance threshold
│
├── datasets/
│   ├── Potato/PLD_3_Classes_256/ # Real Potato Leaf Disease Dataset (Train, Val, Test)
│   └── ODD/                    # Real OOD Datasets (car, animals, cup, random_objects)
│
├── templates/
│   └── index.html              # Responsive AgTech SaaS dashboard
│
├── static/
│   ├── css/style.css           # Dark Forest Green (#123D2A) & Emerald (#16A36A)
│   ├── js/script.js            # Drag-and-drop, Fetch API, 5-state result cards, live stats
│   └── images/samples/         # 10 ready-to-test sample images
│
├── test_cases/                 # Standardized 10 verification test images
├── tests/
│   ├── conftest.py             # Pytest root path resolution
│   ├── test_api.py             # REST API endpoint tests
│   └── test_validation.py      # Unit tests for Quality & OOD detection
│
└── README.md                   # Project documentation
```

---

## 5. REST API Specification

### `POST /predict`
Accepts `multipart/form-data` with an image file (`image` or `file`).

#### Case 1: Accepted Diagnosis (Healthy or Diseased)
```json
{
  "status": "ACCEPTED",
  "reason": null,
  "prediction": "Healthy",
  "confidence": 1.0,
  "quality": "GOOD",
  "ood_score": 0.2374,
  "message": "Potato leaf is healthy and free of disease symptoms."
}
```

#### Case 2: Rejected (Out-of-Distribution / Unknown Object)
```json
{
  "status": "REJECTED",
  "reason": "OOD_DETECTED",
  "prediction": null,
  "confidence": null,
  "quality": "GOOD",
  "ood_score": 0.5565,
  "message": "Unknown object detected. The image does not belong to the supported potato leaf categories."
}
```

#### Case 3: Rejected (Poor Image Quality / Blur)
```json
{
  "status": "REJECTED",
  "reason": "BLUR_DETECTED",
  "prediction": null,
  "confidence": null,
  "quality": "POOR",
  "ood_score": null,
  "message": "Image is blurry or out of focus. Please upload a crisp, clear leaf image."
}
```

#### Case 4: Rejected (Uncertain Prediction)
```json
{
  "status": "REJECTED",
  "reason": "UNCERTAIN_PREDICTION",
  "prediction": null,
  "confidence": 0.54,
  "quality": "GOOD",
  "ood_score": 0.32,
  "message": "The AI model cannot confidently classify this image."
}
```

### `GET /stats`
Returns live platform session telemetry:
```json
{
  "total_analyzed": 24,
  "accepted": 19,
  "rejected": 5,
  "avg_confidence": 0.93
}
```

### `GET /health`
Returns system availability and loaded model status:
```json
{
  "status": "HEALTHY",
  "backend": "RUNNING",
  "model_loaded": true,
  "ood_calibrated": true,
  "supported_classes": ["Potato Healthy", "Potato Early Blight", "Potato Late Blight"]
}
```

---

## 6. Installation & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Run the Official 10-Item Verification Suite
```bash
python run_10_verification_tests.py
```

### 4. Run Pytest Suite
```bash
pytest -v
```
All 16 unit and API integration tests will run.
