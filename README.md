# Task Segmentation for Robot Manipulation

**A unified framework for multi-modal task segmentation combining visual (DINOv3) and proprioceptive (Fast-DROID) features.**

This codebase merges the best components from two task segmentation projects, providing:
- Advanced multi-modal neural architecture
- Ray-parallelized feature extraction
- Comprehensive evaluation metrics (F1@IoU, temporal, confidence)
- Proper train/val/test splitting
- Production-ready code with extensive documentation

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Citation](#citation)

---

## ✨ Features

### Multi-Modal Learning
- **Visual Features**: DINOv3-ViT-L16 vision transformer (1024-dim per frame)
- **Proprioceptive Features**: Fast-DROID tokenization of joint trajectories
- **Fusion**: Transformer-based proprioception encoder + per-frame DINO encoder

### Efficient Processing
- **Ray Parallelization**: Multi-GPU feature extraction with 4 workers
- **Feature Caching**: MD5-based caching for preprocessed features
- **Batch Processing**: Efficient batch DINO extraction (256 frames/batch)

### Robust Training
- **Callback System**: WandB logging, checkpointing, early stopping, TensorBoard
- **Loss Functions**: Focal Loss, Soft Focal Loss, BCE variants, Weighted BCE
- **Label Smoothing**: Gaussian, Triangular, Uniform, Exponential smoothers
- **Trainer Classes**: Modular `Trainer` and `BoundaryTrainer` with extensible hooks
- **Feature Caching**: Automatic caching of DINO features for instant re-runs
- **Scripts**: Complete CLI scripts for data prep, training, evaluation, inference

### Comprehensive Evaluation
- **Frame-level**: Accuracy, precision, recall, F1
- **Segment-level**: F1@IoU with Hungarian algorithm matching
- **Temporal**: Boundary detection error, tolerance-based metrics
- **Confidence**: AUC-ROC, AUC-PR, optimal threshold finding

### Production-Ready
- **Type Hints**: Full type annotations throughout
- **Documentation**: Comprehensive docstrings with examples
- **Testing**: Unit tests for all modules (target: >80% coverage)
- **Reproducibility**: Seed management, experiment state saving
- **Modular**: Clean separation of concerns, easy to extend

---

## 📦 Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- 32GB+ RAM recommended
- 4+ GPUs recommended for parallel feature extraction

### Install from Source

```bash
# Clone repository
git clone https://github.com/KeplerC/RoboSegNet.git
cd RoboSegNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

Core dependencies:
```
torch>=2.0.0
transformers>=4.30.0
ray>=2.5.0
av  # PyAV for video processing
numpy
Pillow
scikit-learn
scipy  # For Hungarian algorithm
h5py  # For proprioception data
tqdm
```

Optional dependencies:
```
wandb  # For experiment tracking
tensorboard  # For logging
pytest  # For testing
hydra-core  # For configuration (coming soon)
```

---

## 🚀 Quick Start

### Complete Workflow (Using Scripts)

The easiest way to get started is using our command-line scripts:

```bash
# 1. Prepare data splits
  # Multi-task ablation
  python scripts/run_multi_task_ablation.py \
      --base-config configs/experiments/evaluation.yaml 

  # Single-task ablation  
  python scripts/run_ablation_study.py \
      --base-config configs/experiments/evaluation.yaml
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{task_segmentation_2025,
  title={Task Segmentation for Robot Manipulation},
  author={Your Team},
  year={2025},
  url={https://github.com/your-org/task_seg}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **DINOv3**: Meta AI's vision transformer
- **Fast-DROID**: Physical Intelligence's proprioceptive tokenizer
- **AgiBotWorld**: Dataset and annotations
- **Ray**: Distributed computing framework

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your-email@example.com

---

**Status**: 🚧 Active Development

**Latest Version**: 0.1.0 (October 2025)
