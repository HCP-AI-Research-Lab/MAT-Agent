# MAT-Agent: Adaptive Multi-Agent Training Optimization

> Official implementation for the NeurIPS 2025 paper **["MAT-Agent: Adaptive Multi-Agent Training Optimization"](https://arxiv.org/abs/2510.17845)**.

**[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-orange.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/117440) | [![Arxiv](https://img.shields.io/badge/Arxiv-PDF-b31b1b.svg)](https://arxiv.org/abs/2510.17845)** | [中文](README_zh.md)

---

## 📖 Introduction

**MAT-Agent** is a novel multi-agent framework that reimagines the training optimization process for Multi-Label Image Classification (MLIC). Instead of relying on static configurations, MAT-Agent treats training as a collaborative, real-time optimization process.

**Key Features:**

* **🤖 Multi-Agent Collaboration:** Deploys four autonomous agents to dynamically tune Data Augmentation, Optimizers, Learning Rates, and Loss Functions in real-time.
* **⚖️ Dynamic Trade-off:** leverages non-stationary multi-armed bandit algorithms to balance exploration and exploitation.
* **🎯 Composite Reward:** Guided by a reward system that harmonizes accuracy, rare-class performance, and training stability.
* **🚀 SOTA Performance:** Achieves **97.4 mAP** on Pascal VOC 2007 and **92.8 mAP** on MS-COCO, significantly outperforming conventional static training methods.

For more details, please refer to our [**paper**](https://arxiv.org/abs/2510.17845).

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/Basicname/MAT-Agent
```
### 2. Environment Setup

We recommend using Conda to manage the environment.

```bash
# Create environment
conda create -n mat-agent python==3.10
conda activate mat-agent

# Install PyTorch 1.31.1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip3 install Pillow randaugment numpy opencv-python tensorboard pycocotools scikit-learn pandas
```

This project requires `inplace_abn`. Please install it from source ensuring you checkout version `v1.1.0`.

```bash
cd MAT-Agent # change directory to root
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
git checkout v1.1.0
python setup.py install
cd scripts
pip3 install -r requirements.txt
cd ../..
```

---

## 📂 Data Preparation

### 1. Download Datasets

Please download the Pascal VOC 2007 and MS-COCO 2014 datasets.

**For Pascal VOC 2007:**

```bash
# Download and extract
wget https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/resolve/main/voc2007.tar.gz -O datasets/voc2007.tar.gz
cd datasets
tar xvzf voc2007.tar.gz

# Convert to COCO format
python3 convert_to_coco.py

# Cleanup (Optional)
rm voc2007.tar.gz && rm -rf VOC2007

```

**For MS-COCO 2014:**
Download the dataset (Train/Val/Test 2014) and annotations from [this link](https://cocodataset.org/#download), then organize them into `datasets/MSCOCO-2014`.

### 2. Directory Structure

Ensure your `datasets/` directory looks exactly like this:

```text
datasets/
├── convert_to_coco.py
├── PascalVOC2COCO.py
├── VOC2007_COCO/              # Processed VOC data
├── MSCOCO-2014/
│   ├── annotations/
│   ├── train2014/
│   ├── val2014/
└── └── test2014/
```

---

## 🚀 Usage

### Training & Evaluating on MS-COCO

To train and evaluate the model on the MS-COCO dataset using MAT-Agent:

```bash
python3 run_coco.py
```

### Training on Pascal VOC

To train and evaluate on Pascal VOC 2007:

```bash
python3 run_voc.py
```

---

## 📊 Results

MAT-Agent demonstrates superior performance compared to state-of-the-art methods across multiple benchmarks.

| Method | Backbone | Pascal VOC (mAP) | MS-COCO (mAP) | VG-256 (mAP) |
| --- | --- | --- | --- | --- |
| ML-GCN | ResNet-101 | 94.0 | 83.0 | 52.3 |
| ASL | ResNet-101 | 95.8 | 86.6 | 56.3 |
| PAT-T | ResNet-101 | 96.2 | 91.8 | 59.5 |
| **MAT-Agent (Ours)** | **ResNet-101** | **97.4** | **92.8** | **60.9** |

---

## 🔗 Citation

If you find our work or code helpful, please cite our paper:

```bibtex
@misc{zhang2025matagentadaptivemultiagenttraining,
      title={MAT-Agent: Adaptive Multi-Agent Training Optimization}, 
      author={Jusheng Zhang and Kaitong Cai and Yijia Fan and Ningyuan Liu and Keze Wang},
      year={2025},
      eprint={2510.17845},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.17845}, 
}

```

---

## 📄 License

This project is released under the [MIT License](LICENSE).