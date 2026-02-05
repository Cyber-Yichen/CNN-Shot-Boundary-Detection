# 🎬 CNN-Shot-Boundary-Detection

English | [简体中文](README.zh-CN.md)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

A lightweight CNN-based system for automatic shot boundary detection in videos, designed for efficient training, evaluation, and integration into computational film analysis and intelligent editing workflows.

基于卷积神经网络的轻量级视频镜头边界检测系统，专为计算电影分析和智能剪辑工作流设计。

---

## 📖 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Evaluation Metrics](#-evaluation-metrics)
- [Tools](#-tools)
- [License](#-license)

---

## ✨ Features

- 🎯 **CNN-based Architecture**: Lightweight CNN model designed for efficient shot boundary detection
- 📊 **Multi-Channel Input**: Support for 9-channel (2FPS) and 21-channel (4FPS) inputs
- ⚖️ **Class Imbalance Handling**: Weighted loss function with configurable pos_weight
- 📈 **Comprehensive Metrics**: Precision, Recall, F1, PR-AUC, ROC-AUC, and per-video analysis
- 🛠️ **Complete Toolchain**: Dataset creation, training, evaluation, and report visualization
- 🌐 **Web-based Tools**: Flask and Streamlit applications for data preparation and result viewing
- 🔬 **Baseline Comparisons**: Includes MLP and Linear models for performance benchmarking

---

## 📁 Project Structure

```
CNN-Shot-Boundary-Detection/
├── Dataset Tool/           # Dataset creation tool (Flask)
│   ├── app.py              # Main Flask application
│   └── requirements.txt    # Dependencies
├── Movie Cut/              # Video frame extraction tool (Streamlit)
│   ├── movie.py            # Frame sampler application
│   └── requirements.txt    # Dependencies
├── Reports System/         # Training report visualization (Flask)
│   ├── app.py              # Report viewer application
│   ├── rsys/               # Report system modules
│   └── requirements.txt    # Dependencies
├── Traning Files/          # Training notebooks and code
│   ├── CNN_2FPS_9CH.ipynb  # Main: CNN model with 9-channel input
│   ├── CNN_4FPS_21CH.ipynb # Main: CNN model with 21-channel input
│   ├── MLP_2FPS_9CH.ipynb  # Baseline: MLP model (for comparison)
│   ├── MLP_4FPS_21CH.ipynb # Baseline: MLP model (for comparison)
│   ├── Linear_2FPS_9CH.ipynb   # Baseline: Linear model (for comparison)
│   ├── Linear_4FPS_21CH.ipynb  # Baseline: Linear model (for comparison)
│   ├── code/               # Additional training code
│   └── movie/              # Video dataset directory
├── Traning Reports/        # Generated training reports
└── LICENSE                 # MIT License
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Cyber-Yichen/CNN-Shot-Boundary-Detection.git
cd CNN-Shot-Boundary-Detection

# Install core dependencies
pip install torch torchvision
pip install opencv-python numpy openpyxl

# For Dataset Tool
pip install flask

# For Movie Cut
pip install streamlit

# For Reports System
pip install flask openpyxl
```

---

## 🚀 Quick Start

### 1. Prepare Dataset

Use the **Movie Cut** tool to extract frames from videos:

```bash
cd "Movie Cut"
streamlit run movie.py
```

### 2. Create Training Data

Use the **Dataset Tool** to generate cut/non-cut samples:

```bash
cd "Dataset Tool"
python app.py
```

### 3. Train Model

Open the Jupyter notebook and run training:

```bash
cd "Traning Files"
jupyter notebook CNN_2FPS_9CH.ipynb
```

### 4. View Results

Launch the **Reports System** to visualize training metrics:

```bash
cd "Reports System"
python app.py
```

---

## 🏗️ Model Architecture

### BoundaryCNN

The core CNN model uses a simple yet effective architecture:

```python
class BoundaryCNN(nn.Module):
    def __init__(self):
        super(BoundaryCNN, self).__init__()  
        self.features = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )  
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary: Cut / Non-Cut
        )
```

### Input Construction

| Mode | Channels | Description |
|------|----------|-------------|
| **2FPS - 9CH** | 9 | `[Frame_A (3)] + [Frame_B (3)] + [Diff (3)]` |
| **4FPS - 21CH** | 21 | Multiple consecutive frames with differences |

The model takes concatenated adjacent frames and their pixel-wise difference as input to detect shot boundaries.

---

## 🎯 Training

### Configuration

Key hyperparameters in the training notebooks:

```python
# Data settings
DATA_VERSION = "v13"
FRAME_SIZE = (224, 224)

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 1024
LR_INIT = 1e-5

# Class imbalance handling
POS_WEIGHT = 40
POS_WEIGHT_MODE = "fixed"  # "fixed" or "epoch"

# Dynamic negative sampling
USE_DYNAMIC_NEG_SAMPLING = False
NEG_SAMPLING_MODE = "ratio"  # "ratio" or "per_pos"
NEG_SAMPLE_RATIO = 0.20
```

### Data Annotation Format

Training data is annotated using Excel files:
- **Row 1**: FPS value (e.g., 24.0)
- **Subsequent rows**: One row per video, columns contain cut point timecodes

Timecode formats supported:
- Frame number: `51`
- Seconds:Frames: `02:14` (2 seconds, 14 frames)
- Hours:Minutes:Seconds: `01:23:45`

---

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

| Metric | Description |
|--------|-------------|
| **Precision** | True positives / Predicted positives |
| **Recall** | True positives / Actual positives |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Accuracy** | Overall correct predictions |
| **PR-AUC (AP)** | Area under Precision-Recall curve |
| **ROC-AUC** | Area under ROC curve |
| **TP/FP/TN/FN** | Confusion matrix components |

### Per-Video Analysis

The evaluation includes per-video breakdown:
- Ground truth cut indices
- Predicted cut indices
- TP/FP/FN counts per video
- Top-K suspect frames (for FP/FN analysis)

---

## 🛠️ Tools

### Dataset Tool

A Flask-based web application for dataset creation:
- Extract frames from videos
- Parse XML annotation files
- Generate cut/non-cut frame pairs
- Support for multiple video formats (MP4, MOV, AVI, MKV)

### Movie Cut

A Streamlit application for video frame sampling:
- Load videos from directory
- Configure sampling interval
- Save frames as PNG images
- Progress tracking

### Reports System

A Flask-based web application for viewing training reports:
- Parse training metrics from Excel files
- Display precision, recall, F1, AUC curves
- Show per-video test results
- Environment information display

---

## 📋 Model Comparison

> **Note**: MLP and Linear models are included for **performance comparison** purposes only. The CNN model is the main focus of this project.

| Model | Input | Parameters | Use Case |
|-------|-------|------------|----------|
| **CNN** | 9CH / 21CH | ~1.5M | **Main model** - Best accuracy, recommended for production |
| **MLP** | 9CH / 21CH | ~800K | Baseline comparison - Faster training, moderate accuracy |
| **Linear** | 9CH / 21CH | ~400K | Baseline comparison - Simplest model, fastest |

---

## 🔬 Technical Details

### Training Environment

Based on actual training runs:

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA H800 PCIe |
| **CUDA** | CUDA 11.8 |
| **PyTorch** | 2.0.0+cu118 |
| **Python** | 3.8.10 |
| **Platform** | Linux (x86_64) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 50 |
| **Training Time** | ~78 minutes (~4657 seconds) |
| **Optimizer** | Adam |
| **Learning Rate** | 1e-5 |
| **Loss Function** | CrossEntropyLoss (weighted) |
| **Class Weight (Cut)** | 40.0 |
| **Class Weight (Non-Cut)** | 1.0 |
| **Threshold** | 0.95 |

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.0+ | Deep learning framework |
| OpenCV | 4.10+ | Video/image processing |
| NumPy | 1.24+ | Numerical operations |
| openpyxl | 3.1+ | Excel file handling |
| Flask | 3.1+ | Web applications |
| Streamlit | 1.53+ | Interactive UI |

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only (slow training)
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Tested on**: NVIDIA RTX 3090, NVIDIA A100, NVIDIA H800

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Cyber-Yichen** - [@Cyber-Yichen](https://github.com/Cyber-Yichen)

Project Link: [https://github.com/Cyber-Yichen/CNN-Shot-Boundary-Detection](https://github.com/Cyber-Yichen/CNN-Shot-Boundary-Detection)

---

## 🤖 AI Contributors

This project was developed with assistance from:

- **ChatGPT** (OpenAI)
- **Gemini** (Google)

---

<p align="center">
  Made with ❤️ for Computational Film Analysis
</p>
