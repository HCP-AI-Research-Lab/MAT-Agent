# MAT-Agent: 自适应多智能体训练优化

> NeurIPS 2025 论文 **["MAT-Agent: Adaptive Multi-Agent Training Optimization"](https://arxiv.org/abs/2510.17845)** 的官方实现。

**[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-orange.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/117440) | [![Arxiv](https://img.shields.io/badge/Arxiv-PDF-b31b1b.svg)](https://arxiv.org/abs/2510.17845)** | [English](README.md)

---

## 📖 简介 (Introduction)

**MAT-Agent** 是一个新颖的多智能体框架，它重新构想了多标签图像分类（MLIC）的训练优化过程。MAT-Agent 不再依赖静态配置，而是将训练视为一个协作的、实时的优化过程。

**主要特性：**

* **🤖 多智能体协作：** 部署四个自主智能体，实时动态调整数据增强（Data Augmentation）、优化器（Optimizers）、学习率（Learning Rates）和损失函数（Loss Functions）。
* **⚖️ 动态权衡：** 利用非平稳多臂老虎机（non-stationary multi-armed bandit）算法来平衡探索（exploration）与利用（exploitation）。
* **🎯 复合奖励：** 由一个协调准确率、稀有类别表现和训练稳定性的奖励系统引导。
* **🚀 SOTA 性能：** 在 Pascal VOC 2007 上达到 **97.4 mAP**，在 MS-COCO 上达到 **92.8 mAP**，显著优于传统的静态训练方法。

更多详情，请参阅我们的 **[论文](https://arxiv.org/abs/2510.17845)**。

---

## 🛠️ 安装 (Installation)

### 1. 克隆仓库

```bash
git clone https://github.com/Basicname/MAT-Agent

```

### 2. 环境设置

我们建议使用 Conda 来管理环境。

```bash
# 创建环境
conda create -n mat-agent python==3.10
conda activate mat-agent

# 安装 PyTorch 1.31.1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装依赖
pip3 install Pillow randaugment numpy opencv-python tensorboard pycocotools scikit-learn pandas

```

本项目需要 `inplace_abn`。请从源码安装，并确保检出 `v1.1.0` 版本。

```bash
cd MAT-Agent # 切换到根目录
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
git checkout v1.1.0
python setup.py install
cd scripts
pip3 install -r requirements.txt
cd ../..

```

---

## 📂 数据准备 (Data Preparation)

### 1. 下载数据集

请下载 Pascal VOC 2007 和 MS-COCO 2014 数据集。

**对于 Pascal VOC 2007：**

```bash
# 下载并解压
wget https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/resolve/main/voc2007.tar.gz -O datasets/voc2007.tar.gz
cd datasets
tar xvzf voc2007.tar.gz

# 转换为 COCO 格式
python3 convert_to_coco.py

# 清理（可选）
rm voc2007.tar.gz && rm -rf VOC2007


```

**对于 MS-COCO 2014：**
从 [此链接](https://cocodataset.org/#download) 下载数据集（Train/Val/Test 2014）和标注，然后将它们整理到 `datasets/MSCOCO-2014` 目录下。

### 2. 目录结构

确保你的 `datasets/` 目录结构如下所示：

```text
datasets/
├── convert_to_coco.py
├── PascalVOC2COCO.py
├── VOC2007_COCO/              # 处理后的 VOC 数据
├── MSCOCO-2014/
│   ├── annotations/
│   ├── train2014/
│   ├── val2014/
└── └── test2014/

```

---

## 🚀 使用方法 (Usage)

### 在 MS-COCO 上训练与评估

使用 MAT-Agent 在 MS-COCO 数据集上训练和评估模型：

```bash
python3 run_coco.py

```

### 在 Pascal VOC 上训练

在 Pascal VOC 2007 上训练和评估：

```bash
python3 run_voc.py

```

---

## 📊 结果 (Results)

MAT-Agent 在多个基准测试中展现出了优于现有最先进（SOTA）方法的性能。

| 方法 (Method) | 主干网络 (Backbone) | Pascal VOC (mAP) | MS-COCO (mAP) | VG-256 (mAP) |
| --- | --- | --- | --- | --- |
| ML-GCN | ResNet-101 | 94.0 | 83.0 | 52.3 |
| ASL | ResNet-101 | 95.8 | 86.6 | 56.3 |
| PAT-T | ResNet-101 | 96.2 | 91.8 | 59.5 |
| **MAT-Agent (Ours)** | **ResNet-101** | **97.4** | **92.8** | **60.9** |

---

## 🔗 引用 (Citation)

如果你觉得我们的工作或代码对你有帮助，请引用我们的论文：

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

## 📄 许可

本项目基于 [MIT License](https://www.google.com/search?q=LICENSE) 发布。