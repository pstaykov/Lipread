"""Paper figures for GNet, rendered in serif to match the LaTeX body text.

Produces three PDFs into paper/figures/:

  fig_curves.pdf     validation top-1 vs epoch, GNet vs MS-TCN, three settings
  fig_ablation.pdf   test top-1 for each ablated configuration
  fig_confusion.pdf  row-normalised confusion matrix, GNet + transfer, 15 classes

Text is rendered by pdflatex (text.usetex) so the figures use the same Computer
Modern as the document. If no working LaTeX is found the script falls back to
DejaVu Serif, which keeps every label serif but will not match the body face.

    python paper/make_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LIPREAD = os.path.join(REPO, 'lipreading')
OUTDIR = os.path.join(HERE, 'figures')
CONF_DIR = os.path.join(LIPREAD, 'analysis', 'figures', 'confusion')

# Categorical slots 1 and 2 of the validated palette. Adjacent-pair CVD separation
# passes for protan/deutan but sits in the floor band for tritan, so both series
# also carry a distinct line style and marker as secondary encoding.
MSTCN = '#2a78d6'
GNET = '#008300'
INK = '#1a1a19'
MUTED = '#6b6b66'
GRID = '#d8d8d4'


def use_serif():
    """Prefer real Computer Modern via pdflatex; fall back to DejaVu Serif."""
    preamble = r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}'
    try:
        matplotlib.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': preamble,
            'font.family': 'serif',
        })
        fig = plt.figure(figsize=(0.5, 0.5))
        fig.text(0.1, 0.1, r'über 5\%')          # umlaut + escaped percent
        fig.canvas.draw()
        plt.close(fig)
        return 'latex'
    except Exception as e:                        # no latex, or a broken install
        plt.close('all')
        matplotlib.rcParams.update({
            'text.usetex': False,
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif'],
            'mathtext.fontset': 'dejavuserif',
        })
        print(f'  (usetex unavailable, falling back to DejaVu Serif: '
              f'{type(e).__name__})', file=sys.stderr)
        return 'dejavu'


def style():
    matplotlib.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.edgecolor': MUTED,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'axes.linewidth': 0.6,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.5,
        'axes.axisbelow': True,
        'figure.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def esc(s):
    """Escape a label for whichever text renderer is active."""
    if matplotlib.rcParams['text.usetex']:
        return s.replace('%', r'\%').replace('&', r'\&').replace('_', r'\_')
    return s


def metrics(*parts):
    return pd.read_csv(os.path.join(LIPREAD, *parts, 'metrics.csv'))


# --------------------------------------------------------------------------
# Figure 1 — validation curves
# --------------------------------------------------------------------------
def fig_curves():
    panels = [
        ('GLips-498', metrics('mstcn_baseline', 'checkpoints'),
         metrics('Transformer_based', 'checkpoints')),
        ('GLips-15, from scratch', metrics('mstcn_baseline', 'checkpoints_15'),
         metrics('Transformer_based', 'checkpoints_15')),
        ('GLips-15, in-domain transfer', metrics('mstcn_baseline', 'checkpoints_15_transfer'),
         metrics('Transformer_based', 'checkpoints_15_transfer')),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for ax, (title, tcn, tf) in zip(axes, panels):
        # peak labels are placed above/below their marker so the two series'
        # annotations cannot collide when the curves finish close together
        for df, color, ls, marker, label, dy, va in (
                (tcn, MSTCN, '--', 's', 'MS-TCN', -11, 'top'),
                (tf, GNET, '-', 'o', 'GNet', 7, 'bottom')):
            y = df['val_top1'] * 100
            ax.plot(df['epoch'], y, color=color, linestyle=ls, linewidth=1.4,
                    label=esc(label), zorder=3)
            i = int(np.argmax(y.values))
            ax.plot(df['epoch'].values[i], y.values[i], marker=marker, color=color,
                    markersize=4.5, markeredgecolor='white', markeredgewidth=0.7,
                    zorder=4, linestyle='none')
            ax.annotate(esc(f'{y.values[i]:.1f}%'),
                        (df['epoch'].values[i], y.values[i]),
                        textcoords='offset points', xytext=(0, dy),
                        ha='center', va=va, fontsize=7, color=color, zorder=5)
        ax.set_title(esc(title), pad=6)
        ax.set_xlabel(esc('epoch'))
        ax.set_ylim(0, 100)
        ax.set_xlim(left=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(esc('validation top-1 (%)'))
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    # upper left is the only empty quadrant of the 498 panel; lower right collides
    # with the MS-TCN peak label
    axes[0].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'fig_curves.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(out, REPO)}')


# --------------------------------------------------------------------------
# Figure 2 — ablation bars (test split)
# --------------------------------------------------------------------------
def fig_ablation():
    rows = [
        ('GNet (full)', 59.33),
        ('mean pool instead of attentive', 60.53),
        ('no multi-scale conv stem', 56.93),
        ('no mouth-ROI crop', 30.13),
        ('baseline input regime', 16.53),
    ]
    labels = [esc(r[0]) for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.62, color=GNET, zorder=3)
    ax.axvline(rows[0][1], color=MUTED, linewidth=0.7, linestyle=':', zorder=2)
    # labels sit inside the bar end: outside placement collides with the dotted
    # reference line for values close to the full model
    for yi, v in zip(y, vals):
        ax.annotate(esc(f'{v:.1f}'), (v, yi), textcoords='offset points',
                    xytext=(-4, 0), ha='right', va='center', fontsize=7.5,
                    color='white', zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(esc('test top-1 (%)'))
    ax.set_xlim(0, 72)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'fig_ablation.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(out, REPO)}')


# --------------------------------------------------------------------------
# Figure 3 — confusion matrix
# --------------------------------------------------------------------------
def fig_confusion():
    sys.path.insert(0, LIPREAD)
    from dataset15 import CLASSES

    path = os.path.join(CONF_DIR, 'glips15_GLipsNet_transfer.npy')
    if not os.path.exists(path):
        print(f'  ! skipped confusion: {path} missing (run analysis/ablation.py)',
              file=sys.stderr)
        return
    cm = np.load(path).astype(float)
    # row-normalized (per true label), matching analysis/ablation.py::_save_confusion
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    n = len(CLASSES)

    # Same style as the diagnostic PNGs in ablation.py: annotated cells, white text
    # on dark cells, Blues 0-1. Uses that function's own figsize formula so the
    # cell-to-type ratio is identical; LaTeX scales the whole figure to the text
    # width, which keeps adjacent cell labels from touching.
    side = max(6, n * 0.62)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    fs = max(6, min(11, int(150 / n)))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, esc(f'{round(float(norm[i, j]), 2):g}'),
                    ha='center', va='center', fontsize=fs,
                    color='white' if norm[i, j] > 0.5 else '#1a1a6e')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([esc(c) for c in CLASSES], rotation=90, fontsize=10)
    ax.set_yticklabels([esc(c) for c in CLASSES], fontsize=10)
    ax.set_xlabel(esc('Predicted label'), fontsize=11)
    ax.set_ylabel(esc('True label'), fontsize=11)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'fig_confusion.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(out, REPO)}')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    backend = use_serif()
    style()
    print(f'Rendering figures (text backend: {backend})')
    fig_curves()
    fig_ablation()
    fig_confusion()


if __name__ == '__main__':
    main()
