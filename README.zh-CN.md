# 🎬 CNN-Shot-Boundary-Detection

简体中文 | [English](README.md)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

一个轻量级的基于卷积神经网络的视频镜头边界检测系统，专为高效训练、评估以及在计算电影分析和智能剪辑工作流中的集成而设计。

---

## 📖 目录

- [功能亮点](#-功能亮点)
- [项目结构](#-项目结构)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [模型结构](#-模型结构)
- [训练](#-训练)
- [评估指标](#-评估指标)
- [工具](#-工具)
- [许可证](#-许可证)

---

## ✨ 功能亮点

- 🎯 **CNN 架构**：轻量级 CNN 模型，专为高效镜头边界检测设计
- 📊 **多通道输入**：支持 9 通道（2FPS）和 21 通道（4FPS）输入
- ⚖️ **类别不平衡处理**：带可配置 pos_weight 的加权损失函数
- 📈 **全面评估指标**：Precision、Recall、F1、PR-AUC、ROC-AUC 以及按视频统计
- 🛠️ **完整工具链**：数据集创建、训练、评估与报告可视化
- 🌐 **Web 工具**：用于数据准备和结果查看的 Flask 与 Streamlit 应用
- 🔬 **基线对比**：包含 MLP 与 Linear 模型用于性能基准对比

---

## 📁 项目结构

```
CNN-Shot-Boundary-Detection/
├── Dataset Tool/           # 数据集创建工具（Flask）
│   ├── app.py              # 主 Flask 应用
│   └── requirements.txt    # 依赖
├── Movie Cut/              # 视频帧提取工具（Streamlit）
│   ├── movie.py            # 帧采样应用
│   └── requirements.txt    # 依赖
├── Reports System/         # 训练报告可视化（Flask）
│   ├── app.py              # 报告查看应用
│   ├── rsys/               # 报告系统模块
│   └── requirements.txt    # 依赖
├── Traning Files/          # 训练笔记本与代码
│   ├── CNN_2FPS_9CH.ipynb  # 主模型：9 通道 CNN
│   ├── CNN_4FPS_21CH.ipynb # 主模型：21 通道 CNN
│   ├── MLP_2FPS_9CH.ipynb  # 基线：MLP（用于对比）
│   ├── MLP_4FPS_21CH.ipynb # 基线：MLP（用于对比）
│   ├── Linear_2FPS_9CH.ipynb   # 基线：Linear（用于对比）
│   ├── Linear_4FPS_21CH.ipynb  # 基线：Linear（用于对比）
│   ├── code/               # 额外训练代码
│   └── movie/              # 视频数据集目录
├── Traning Reports/        # 训练报告输出
└── LICENSE                 # MIT License
```

---

## 🔧 安装

### 前置条件

- Python 3.8+
- 支持 CUDA 的 GPU（推荐）

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Cyber-Yichen/CNN-Shot-Boundary-Detection.git
cd CNN-Shot-Boundary-Detection

# 安装核心依赖
pip install torch torchvision
pip install opencv-python numpy openpyxl

# Dataset Tool 依赖
pip install flask

# Movie Cut 依赖
pip install streamlit

# Reports System 依赖
pip install flask openpyxl
```

---

## 🚀 快速开始

### 1. 准备数据集

使用 **Movie Cut** 工具从视频中抽帧：

```bash
cd "Movie Cut"
streamlit run movie.py
```

### 2. 创建训练数据

使用 **Dataset Tool** 生成切换/非切换样本：

```bash
cd "Dataset Tool"
python app.py
```

### 3. 训练模型

打开 Jupyter Notebook 并执行训练：

```bash
cd "Traning Files"
jupyter notebook CNN_2FPS_9CH.ipynb
```

### 4. 查看结果

启动 **Reports System** 可视化训练指标：

```bash
cd "Reports System"
python app.py
```

---

## 🏗️ 模型结构

### BoundaryCNN

核心 CNN 模型使用简洁有效的结构：

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

### 输入构造

| 模式 | 通道数 | 描述 |
|------|--------|------|
| **2FPS - 9CH** | 9 | `[Frame_A (3)] + [Frame_B (3)] + [Diff (3)]` |
| **4FPS - 21CH** | 21 | 多帧连续采样并包含差分 |

模型通过拼接相邻帧及其像素级差分作为输入来检测镜头边界。

---

## 🎯 训练

### 配置

训练笔记本中的关键超参数：

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

### 数据标注格式

训练数据使用 Excel 文件标注：
- **第 1 行**：FPS 值（如 24.0）
- **后续行**：每个视频一行，列中包含镜头切点时间码

支持的时间码格式：
- 帧编号：`51`
- 秒:帧：`02:14`（2 秒 14 帧）
- 时:分:秒：`01:23:45`

---

## 📊 评估指标

系统提供完整的评估指标：

| 指标 | 描述 |
|------|------|
| **Precision** | 真阳性 / 预测为正 |
| **Recall** | 真阳性 / 实际为正 |
| **F1-Score** | Precision 与 Recall 的调和均值 |
| **Accuracy** | 总体准确率 |
| **PR-AUC (AP)** | PR 曲线面积 |
| **ROC-AUC** | ROC 曲线面积 |
| **TP/FP/TN/FN** | 混淆矩阵指标 |

### 按视频分析

评估包含按视频的拆分结果：
- 真实切点索引
- 预测切点索引
- 每个视频的 TP/FP/FN 统计
- Top-K 可疑帧（用于 FP/FN 分析）

---

## 🛠️ 工具

### Dataset Tool

用于数据集创建的 Flask Web 应用：
- 从视频中提取帧
- 解析 XML 标注文件
- 生成切换/非切换帧对
- 支持多种视频格式（MP4、MOV、AVI、MKV）

### Movie Cut

用于视频帧采样的 Streamlit 应用：
- 从目录加载视频
- 配置采样间隔
- 保存帧为 PNG 图像
- 进度跟踪

### Reports System

用于查看训练报告的 Flask Web 应用：
- 从 Excel 文件解析训练指标
- 展示 Precision、Recall、F1、AUC 曲线
- 显示按视频的测试结果
- 环境信息展示

---

## 📋 模型对比

> **说明**：MLP 和 Linear 模型仅用于 **性能对比**，CNN 模型为本项目主要模型。

| 模型 | 输入 | 参数量 | 用途 |
|------|------|--------|------|
| **CNN** | 9CH / 21CH | ~1.5M | **主模型** - 最佳精度，推荐生产使用 |
| **MLP** | 9CH / 21CH | ~800K | 基线对比 - 训练更快、精度中等 |
| **Linear** | 9CH / 21CH | ~400K | 基线对比 - 最简模型、速度最快 |

---

## 🔬 技术细节

### 训练环境

基于真实训练运行：

| 组件 | 规格 |
|------|------|
| **GPU** | NVIDIA H800 PCIe |
| **CUDA** | CUDA 11.8 |
| **PyTorch** | 2.0.0+cu118 |
| **Python** | 3.8.10 |
| **Platform** | Linux (x86_64) |

### 训练配置

| 参数 | 值 |
|------|------|
| **Epochs** | 50 |
| **Training Time** | ~78 分钟（~4657 秒） |
| **Optimizer** | Adam |
| **Learning Rate** | 1e-5 |
| **Loss Function** | CrossEntropyLoss（加权） |
| **Class Weight (Cut)** | 40.0 |
| **Class Weight (Non-Cut)** | 1.0 |
| **Threshold** | 0.95 |

### 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| PyTorch | 2.0+ | 深度学习框架 |
| OpenCV | 4.10+ | 视频/图像处理 |
| NumPy | 1.24+ | 数值计算 |
| openpyxl | 3.1+ | Excel 文件处理 |
| Flask | 3.1+ | Web 应用 |
| Streamlit | 1.53+ | 交互式界面 |

### 硬件需求

- **最低**：8GB RAM，CPU-only（训练较慢）
- **推荐**：16GB RAM，NVIDIA GPU（8GB+ 显存）
- **测试通过**：NVIDIA RTX 3090、NVIDIA A100、NVIDIA H800

---

## 📄 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE)。

---

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

1. Fork 仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交修改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 打开 Pull Request

---

## 📧 联系方式

**Cyber-Yichen** - [@Cyber-Yichen](https://github.com/Cyber-Yichen)

项目链接：[https://github.com/Cyber-Yichen/CNN-Shot-Boundary-Detection](https://github.com/Cyber-Yichen/CNN-Shot-Boundary-Detection)

---

## 🤖 AI 贡献者

本项目在以下工具协助下完成：

- **ChatGPT**（OpenAI）
- **Gemini**（Google）

---

<p align="center">
  Made with ❤️ for Computational Film Analysis
</p>
