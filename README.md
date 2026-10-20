# 🔬 Endometriosis Lesion Segmentation AI

An end-to-end deep learning pipeline for multi-class endometriosis lesion segmentation from laparoscopic video frames using U-Net.

---

## 📌 Project Overview
Endometriosis is a complex gynecological condition characterized by endometrium-like tissue growing outside the uterine cavity. Automated lesion segmentation assists surgeons in identifying difficult-to-spot lesions across anatomical sites in real time.

This project delivers:
- **PyTorch U-Net Architecture** for multi-class semantic segmentation.
- **Combined Dice + Weighted Cross-Entropy Loss** to tackle severe medical class imbalance (>98% background pixels).
- **Patient/Video Leakage-Free Splitting** grouping frames by video source.
- **Interactive Streamlit Web Dashboard** for live frame upload, translucent mask overlay, and quantitative lesion pixel analytics.
- **Single-Image CLI & COCO Visualization Tools**.

---

## 🏷️ Endometriosis Target Classes
| Class ID | Anatomical Category | Marker Color | Description |
|---|---|---|---|
| **0** | `Background` | ⚫ Black | Healthy non-lesion tissue & background |
| **1** | `Endo-Peritoneum` | 🟣 Purple (`#be3ecc`) | Peritoneal superficial endometriosis lesions |
| **2** | `Endo-Ovar` | 🔵 Blue (`#5d59fe`) | Ovarian endometriomas |
| **3** | `Endo-TIE` | 🟢 Green (`#91d28a`) | Deep Infiltrating Endometriosis (TIE) |
| **4** | `Endo-Uterus` | 🟡 Yellow (`#eeec32`) | Uterine/adenomyosis-associated lesions |

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yuvaingit/endometriosis_detection.git
cd endometriosis_detection
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application
Run the interactive web interface:
```bash
streamlit run app.py
```
Open (https://endometriosisdetection-ap8zzyicuani2jfnccjh8i.streamlit.app/) in your browser to:
- Upload any laparoscopic image (`.jpg`, `.png`).
- Select pre-configured validation frames.
- Adjust segmentation overlay transparency in real-time.
- View pixel-level metrics and lesion coverage percentages.

---

## 🛠️ Pipeline Tools & Scripts

### Single-Image CLI Inference
Run predictions on an individual laparoscopic frame:
```bash
python predict_lesion.py "frames/c_170_v_(video_5222.mp4)_f_64.jpg"
```
Outputs a 3-panel side-by-side comparison (`Input Frame` | `Predicted Mask` | `Lesion Overlay`).

### Train U-Net
Retrain or fine-tune the model:
```bash
python train_unet.py
```
- Trains on **Apple Silicon MPS**, **CUDA**, or **CPU**.
- Standardizes frames to `256×256` using bilinear image interpolation and nearest-neighbor mask interpolation.
- Automatically saves the best model checkpoint to `checkpoints/glenda_unet_best.pth`.

### Validation Set Evaluation
Evaluate the model and generate multi-view validation panels:
```bash
python evaluate_unet.py
```
Saves side-by-side evaluation comparisons to `evaluation_results/`.

### COCO Dataset Visualizer
Generate dataset statistics and an interactive dark-mode HTML gallery:
```bash
python visualize_coco.py --num-samples 15
```

---

## 📁 Repository Structure
```
endometriosis_detection/
├── app.py                     # Streamlit web dashboard
├── train_unet.py              # U-Net architecture & training loop
├── dataset_loader.py          # PyTorch Dataset with video-group split
├── evaluate_unet.py           # Multi-panel evaluation generator
├── predict_lesion.py          # Single-frame CLI prediction script
├── visualize_coco.py          # COCO annotation visualizer & HTML gallery
├── requirements.txt           # Python dependencies
├── checkpoints/
│   ├── glenda_unet_best.pth   # Best trained model weights
│   └── glenda_unet.pth        # Final trained model weights
├── coco.json                  # COCO annotation metadata
├── colors.txt                 # RGB color definitions per class
├── labels.txt                 # Class labels
├── config.json                # Dataset configuration
├── frames/                    # Laparoscopic video frames
├── annots/                    # Expert ground-truth segmentation masks
└── evaluation_results/        # Evaluation output panels & app scripts
```

---

## ⚖️ Disclaimer
*This model is developed for research, education, and algorithmic demonstration only. It is not an FDA-approved medical device and must not be used as a substitute for professional clinical diagnosis or surgical judgment.*
