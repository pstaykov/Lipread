import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'figure.dpi': 120,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#cccccc',
    'grid.alpha': 0.6,
    'grid.linestyle': '--',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
})

BLUE   = '#4C72B0'
ORANGE = '#DD8452'
GREEN  = '#55A868'
RED    = '#C44E52'
GRAY   = '#8c8c8c'

SAVE_KW = dict(bbox_inches='tight', facecolor='white', dpi=150)

df500 = pd.read_csv('checkpoints/metrics.csv')
df15  = pd.read_csv('checkpoints_glips15/metrics.csv')


# --- 1: GLips-500 loss + accuracy ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.set_facecolor('white')
fig.suptitle('GLips-500  (500 German words, 50 epochs)', fontsize=14, fontweight='bold', y=1.02)
e = df500['epoch']

ax = axes[0]
ax.plot(e, df500['train_loss'], color=BLUE,   label='Train loss', linewidth=2)
ax.plot(e, df500['val_loss'],   color=ORANGE, label='Val loss',   linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss')
ax.set_title('Loss'); ax.legend()

ax = axes[1]
ax.plot(e, df500['train_acc'] * 100, color=BLUE,   label='Train top-1', linewidth=2)
ax.plot(e, df500['val_top1']  * 100, color=ORANGE, label='Val top-1',   linewidth=2)
ax.plot(e, df500['val_top5']  * 100, color=GREEN,  label='Val top-5',   linewidth=2, linestyle='--')
best_epoch = df500['val_top1'].idxmax()
ax.axvline(best_epoch + 1, color=RED, linestyle=':', alpha=0.7, label=f'Best epoch ({best_epoch+1})')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy'); ax.legend()

plt.tight_layout()
plt.savefig('plots_500_curves.png', **SAVE_KW)
plt.close()
print('Saved plots_500_curves.png')


# --- 2: Literature comparison — top-1 only ---
german_models = [
    ('3D-CNN + BiGRU\n(GLips paper)',      27.6, 'GLips 500', 2023),
    ('ResNet-18 + MS-TCN\n(GLips paper)',  38.2, 'GLips 500', 2023),
    ('GLipsNet ours\n(500-class)',          round(df500['val_top1'].max() * 100, 1), 'GLips 500', 2025),
    ('GLipsNet ours\n(15-class)',           round(df15['val_top1'].max() * 100, 1),  'GLips 15',  2025),
]
labels     = [m[0] for m in german_models]
top1       = [m[1] for m in german_models]
bar_colors = [GRAY, GRAY, BLUE, ORANGE]
x          = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(9, 5))
fig.set_facecolor('white')
fig.suptitle('German Lip Reading on GLips Dataset — Top-1 Accuracy', fontsize=13, fontweight='bold')

bars = ax.bar(x, top1, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.55, zorder=3)
for bar, val in zip(bars, top1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_ylim(0, 110)
ax.axhline(1 / 500 * 100, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.text(3.3, 1 / 500 * 100 + 0.5, 'random (500)', fontsize=8, color=GRAY, va='bottom')

legend_elements = [
    Patch(facecolor=GRAY,   edgecolor='black', label='GLips paper baselines (2023)'),
    Patch(facecolor=BLUE,   edgecolor='black', label='GLipsNet ours — 500 classes (2025)'),
    Patch(facecolor=ORANGE, edgecolor='black', label='GLipsNet ours — 15 classes (2025)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('plots_literature_comparison.png', **SAVE_KW)
plt.close()
print('Saved plots_literature_comparison.png')
