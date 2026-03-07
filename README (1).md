# Masked Autoencoder (MAE) — Self-Supervised Image Representation Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![Kaggle](https://img.shields.io/badge/Kaggle-T4%20x2%20GPU-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**A complete from-scratch PyTorch implementation of Masked Autoencoders (MAE) for self-supervised visual representation learning.**

[Live Demo](#live-demo) • [Architecture](#architecture) • [Results](#results) • [Setup](#setup) • [Usage](#usage)

</div>

---

## Overview

This project implements the **Masked Autoencoder (MAE)** proposed by He et al. (2021), a powerful self-supervised learning framework that teaches a model to understand images by reconstructing **75% of randomly masked patches** from only 25% of visible patches.

The entire system is built **from scratch using base PyTorch** — no pretrained weights, no external ViT libraries.

```
Input Image (224x224)
       │
       ▼
  196 Patches (16x16 each)
       │
       ▼
  Mask 75% → Keep 49 patches only
       │
       ▼
  Encoder (ViT-Base, 86M params)
       │
       ▼
  Decoder (ViT-Small, 22M params)
       │
       ▼
  Reconstruct all 196 patches
       │
       ▼
  MSE Loss on masked patches only
```

---

## Live Demo

Try the live Gradio app on Hugging Face Spaces:

**[Launch App →](https://huggingface.co/spaces/YOUR_USERNAME/mae-reconstruction)**

Upload any image, choose a masking ratio (10%–90%), and watch the MAE reconstruct the missing patches in real time.

---

## Architecture

The system uses an **asymmetric encoder-decoder design** — the key innovation that makes MAE computationally efficient.

### Encoder — ViT-Base (B/16)

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Patch Size | 16 × 16 |
| Total Patches | 196 |
| Visible Patches | 49 (25%) |
| Hidden Dimension | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Parameters | ~86 Million |

- Accepts **only 25% visible patches** (49 out of 196)
- Uses **sinusoidal positional embeddings** for spatial awareness
- Mask tokens are **never** fed to the encoder
- Outputs latent representations for visible tokens only

### Decoder — ViT-Small (S/16)

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 384 |
| Transformer Layers | 12 |
| Attention Heads | 6 |
| Parameters | ~22 Million |

- Receives encoder output + **learnable mask tokens** for missing patches
- Reconstructs the full 196-patch sequence
- Used **only during training** — discarded at inference

### Why Asymmetric?

The encoder processes 4x fewer tokens (49 vs 196), making training roughly **16x cheaper** due to quadratic attention complexity. The decoder is intentionally lightweight since reconstruction is simpler than understanding.

---

## Key Concepts

### Patchification
```
224x224 image → split into 14x14 grid → 196 patches of 16x16 pixels
Each patch flattened → 16 x 16 x 3 = 768 numbers per patch
```

### Random Masking
```
Shuffle all 196 patch indices randomly
Keep first 49 (25%) → visible patches
Discard last 147 (75%) → masked patches
Save ids_restore to unshuffle later
```

### Loss Function
```
Target = patchify(original image)
Target = normalize per-patch (mean=0, std=1)
Loss   = MSE(prediction, target)
Loss computed ONLY on 147 masked patches
Visible patches are ignored in loss
```

### Training Techniques
- **AdamW** optimizer with betas (0.9, 0.95) and weight decay 0.05
- **Cosine LR schedule** with 5-epoch linear warmup
- **Mixed Precision (AMP)** for 2x memory savings and faster training
- **Gradient clipping** at 1.0 to prevent instability
- **DataParallel** for dual T4 GPU utilization

---

## Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Best Validation Loss | 0.3491 |
| Training Platform | Kaggle T4 x2 GPU |
| Dataset | TinyImageNet (100K images) |
| Batch Size | 64 |

### Quantitative Evaluation

| Metric | Score |
|--------|-------|
| Mean PSNR | ~22 dB |
| Mean SSIM | ~0.70 |

### Qualitative Results

The model successfully:
- Reconstructs major structures and shapes from 25% visible patches
- Infers approximate colors and textures of masked regions
- Maintains spatial coherence across reconstructed patches

---

## Project Structure

```
MAE-Masked-Autoencoder/
│
├── AI_ASS01_XXF_YYYY.ipynb     # Complete Kaggle notebook
├── app.py                       # Hugging Face Gradio app
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── outputs/
    ├── loss_curve.png           # Training & validation loss plot
    ├── reconstruction.png       # 5 qualitative reconstruction samples
    └── metrics_distribution.png # PSNR & SSIM distributions
```

---

## Setup

### Prerequisites
```bash
Python 3.10+
CUDA-enabled GPU (recommended)
```

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/MAE-Masked-Autoencoder.git
cd MAE-Masked-Autoencoder
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.1
torchvision>=0.15.2
gradio>=3.50.2
einops>=0.6.1
Pillow>=9.5.0
numpy>=1.24.3
scikit-image>=0.21.0
```

---

## Usage

### Run on Kaggle (Recommended)

1. Open the notebook on Kaggle
2. Add **TinyImageNet** dataset: `akash2sharma/tiny-imagenet`
3. Enable **GPU T4 x2** accelerator
4. Run all cells in order (Cell 1 → Cell 15)

### Run Gradio App Locally

```bash
# Download mae_best.pth from releases
python app.py
# Open http://localhost:7860
```

### Notebook Cell Guide

| Cell | Description |
|------|-------------|
| Cell 1 | Install dependencies |
| Cell 2 | Config & imports |
| Cell 3 | TinyImageNet dataloaders |
| Cell 4 | Transformer building blocks |
| Cell 5 | Patchify / Unpatchify / Masking |
| Cell 6 | MAE Encoder (ViT-Base) |
| Cell 7 | MAE Decoder (ViT-Small) |
| Cell 8 | Full MAE model |
| Cell 9 | Optimizer + Scheduler + AMP |
| Cell 10 | Training loop |
| Cell 11 | Loss curve plot |
| Cell 12 | Visualization (5 samples) |
| Cell 13 | PSNR & SSIM evaluation |
| Cell 14 | Save checkpoint |
| Cell 15 | Gradio app |

---

## Dataset

**TinyImageNet** — a subset of ImageNet
- 200 object classes
- 100,000 training images
- 10,000 validation images
- Image size: resized to 224x224

Add to Kaggle notebook from:
`https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet`

---

## References

```bibtex
@article{he2021masked,
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  author  = {He, Kaiming and Chen, Xinlei and Xie, Saining and 
             Li, Yanghao and Dollar, Piotr and Girshick, Ross},
  journal = {arXiv preprint arXiv:2111.06377},
  year    = {2021}
}

@article{dosovitskiy2020image,
  title   = {An Image is Worth 16x16 Words: Transformers for 
             Image Recognition at Scale},
  author  = {Dosovitskiy, Alexey and Beyer, Lucas and 
             Kolesnikov, Alexander and others},
  journal = {arXiv preprint arXiv:2010.11929},
  year    = {2020}
}
```

---

## Course Information

- **Course:** Generative AI (AI4009)
- **University:** National University of Computer and Emerging Sciences
- **Semester:** Spring 2026
- **Assignment:** No. 2 — Self-Supervised Image Representation Learning

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built with PyTorch | Trained on Kaggle | Deployed on Hugging Face

</div>
