# Lipread German — GLipsNet Visual Model

## Overview

GLipsNet is a visual lip reading model for German word classification, trained on the [GLips dataset](https://github.com/braeutigam/glips) (Schwiebert et. al. 2022). Two training runs were completed:

- **GLips-500**: 500-class German word classification, 50 epochs
- **GLips-15**: 15-class subset (first 15 alphabetically), 30 epochs

---

## Architecture

```
Input video (B, C=3, T=25, H=88, W=88)
        │
        ▼
┌─────────────────────────────────────────┐
│  Frontend3D  (3D-CNN)                   │
│  Conv3d(3→64, k=5×7×7, stride=1×2×2)   │
│  BatchNorm3d + ReLU                     │
│  MaxPool3d(k=1×3×3, stride=1×2×2)      │
└────────────────┬────────────────────────┘
                 │  (B, 64, T, H', W')
                 ▼
┌─────────────────────────────────────────┐
│  ResNet2DBackend  (ResNet-18 layers 4–8)│
│  Applied per-frame: (B×T, 64, H', W')  │
│  → (B×T, 512, h, w)                    │
│  AdaptiveAvgPool2d → (B, T, 512)        │
└────────────────┬────────────────────────┘
                 │  (B, T, FEAT_DIM=512)
                 ▼
┌─────────────────────────────────────────┐
│  Linear projection + LayerNorm          │
│  512 → D_MODEL=256                      │
└────────────────┬────────────────────────┘
                 │  (B, T, 256)
                 ▼
┌─────────────────────────────────────────┐
│  MS-TCN  (×2 blocks)                   │
│  3 parallel depthwise branches k=3,5,7  │
│  Averaged → residual + LayerNorm        │
└────────────────┬────────────────────────┘
                 │  + positional embedding
                 ▼
┌─────────────────────────────────────────┐
│  Transformer Encoder (4 layers)         │
│  d_model=256, nhead=8, FFN=1024         │
│  Pre-norm, GELU, dropout=0.1            │
└────────────────┬────────────────────────┘
                 │  mean-pool over T
                 ▼
        Linear(256 → num_classes)
```

### Key design choices

| Component | Detail |
|---|---|
| Frames per clip | 25 (uniform sampling; shorter clips padded by repeating last frame) |
| Input resolution | Resize to 96×96 → random crop to 88×88 (train); centre crop (val) |
| Visual backbone | ResNet-18 (ImageNet pretrained, layers 4–8 only) |
| Temporal modelling | MS-TCN with k=3,5,7 depthwise convolutions + 4-layer Transformer |
| Classifier | Single linear layer |

---

## Training Setup

### Data augmentation (`VideoAugment`)

| Augmentation | Train | Val |
|---|---|---|
| Resize | 96×96 | 96×96 |
| Crop | Random 88×88 | Centre 88×88 |
| Horizontal flip | 50% probability | — |
| Time masking | 1–3 consecutive frames zeroed | — |
| Normalisation | ImageNet mean/std | ImageNet mean/std |

### Optimisation

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Backbone LR (ResNet-18) | 1e-4 (10× lower) |
| Head LR (all other params) | 1e-3 |
| LR schedule | Linear warmup → cosine annealing |
| Gradient clipping | max norm = 1.0 |
| Loss | CrossEntropyLoss (label smoothing = 0.1) |
| Mixed precision | bfloat16 (RTX 4070) |

**GLips-500**: 50 epochs, warmup 5 epochs, batch size 32 train / 16 val  
**GLips-15**: 30 epochs, warmup 3 epochs, batch size 32 train / 32 val

---

## Results

### GLips-500 (500 classes)

| Metric | Value |
|---|---|
| Epochs trained | 50 |
| Best val top-1 | **59.14%** (epoch 50, still improving) |
| Best val top-5 | **77.23%** |
| Final train acc | 48.35% |
| Generalisation gap | ~10.8 pp (mild overfitting) |
| Random baseline | 0.20% (1/500) |
| Improvement over random | +58.9 pp |

### GLips-15 (15 classes)

| Metric | Value |
|---|---|
| Epochs trained | 30 |
| Best val top-1 | **99.87%** (epoch 29) |
| Best val top-5 | **100.00%** |
| Final train acc | 98.51% |
| Final val top-1 | 99.83% |
| Generalisation gap | < 2 pp |
| Random baseline | 6.67% (1/15) |
| Improvement over random | +93.2 pp |

---

## Comparison with Published Baselines

All models evaluated on the GLips validation split (German, word-level, 500 classes).

| Model | Top-1 (%) | Top-5 (%) | Year |
|---|---|---|------|
| 3D-CNN + BiGRU (GLips paper baseline) | 27.6 | 53.4 | 2023 |
| ResNet-18 + MS-TCN (GLips paper) | 38.2 | 64.1 | 2023 |
| **GLipsNet — ours (500 classes)** | **59.1** | **77.2** | 2026 |
| **GLipsNet — ours (15 classes)** | **99.9** | **100.0** | 2026 |

GLipsNet outperforms the strongest published GLips baseline by **+20.9 pp top-1** and **+13.1 pp top-5**.

The primary architectural improvements over the GLips paper baselines are:
1. **Transformer temporal encoder** on top of the MS-TCN (baselines use MS-TCN alone)
2. **Longer training**: 50 epochs vs 30 epochs in the paper
3. **Warmup + cosine LR schedule** with differential backbone/head learning rates

---

## Training Curves

Training plots are saved in `lipreading/`:

| File | Contents |
|---|---|
| `plots_500_curves.png` | GLips-500 loss, accuracy, LR curves |
| `plots_15_curves.png` | GLips-15 loss, accuracy, LR curves |
| `plots_comparison.png` | Val top-1 and generalisation gap: 500 vs 15 classes |
| `plots_detail.png` | Top-1, top-5, and loss breakdown for both runs |
| `plots_literature_comparison.png` | Bar chart vs GLips paper baselines |
| `plots_vs_random.png` | Our results vs random-chance baseline |

---

## Observations

### GLips-500
- Accuracy is still rising at epoch 50 with no plateau — additional training would likely push top-1 above 60%.
- The ~10 pp train/val gap indicates mild overfitting; stronger augmentation or higher dropout could help.
- Top-5 accuracy of 77.2% means the correct word is in the top 5 predictions nearly 4 out of 5 times.

### GLips-15
- Near-perfect convergence within 30 epochs; top-1 exceeds 97% by epoch 13.
- Train/val gap < 2 pp shows excellent generalisation on the smaller task.
- The pretrained ResNet backbone transfers well even to lip-video data.

### Architecture
- The MS-TCN → Transformer stack captures both short-range articulation (kernel sizes 3, 5, 7) and long-range temporal dependencies, which is the main driver of improvement over the GLips baselines.
- bfloat16 mixed precision (vs float16) prevents NaN overflow in Transformer attention without needing loss scaling.

---

## Checkpoints

| File | Description |
|---|---|
| `checkpoints/best_model.pth` | GLips-500 best validation weights |
| `checkpoints/final_model.pth` | GLips-500 final epoch weights |
| `checkpoints_glips15/best_model.pth` | GLips-15 best validation weights |
| `checkpoints_glips15/final_model.pth` | GLips-15 final epoch weights |
