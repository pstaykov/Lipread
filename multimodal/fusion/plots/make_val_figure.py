"""Val-split reference figure: panel A clean-audio top-1 per system, panel B noise-robustness curves."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'val_reference.png')

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e7e6e2'
BASE = '#9a9892'   # baseline gray (audio/visual references)

clean = [
    ('joint_tf  (AV-HuBERT-style)', 0.7286, BLUE),
    ('cross_attn  (finetuned)',     0.7166, ORANGE),
    ('cross_attn  (frozen)',        0.7141, ORANGE),
    ('audio_only',                  0.6137, BASE),
    ('concat',                      0.6113, AQUA),
    ('late',                        0.6055, YELLOW),
    ('visual_only',                 0.3332, BASE),
]

scen = ['clean', 'white\n15', 'white\n10', 'white\n5', 'babble\n15', 'babble\n10', 'babble\n5']
multimodal = [0.7163, 0.6998, 0.6792, 0.6421, 0.6951, 0.6603, 0.5702]
audio_only = [0.6118, 0.5162, 0.4546, 0.3508, 0.5626, 0.4825, 0.2932]
visual_flat = 0.3332

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans',
                     'axes.edgecolor': MUTED, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1, 1.15]})

names = [n for n, _, _ in clean][::-1]
vals = [v for _, v, _ in clean][::-1]
cols = [c for _, _, c in clean][::-1]
y = range(len(names))
axA.barh(list(y), vals, color=cols, height=0.62, zorder=3)
for yi, v in zip(y, vals):
    axA.text(v + 0.008, yi, f'{v:.3f}', va='center', ha='left', fontsize=10, color=INK)
axA.set_yticks(list(y)); axA.set_yticklabels(names, fontsize=10)
axA.set_xlim(0, 0.85); axA.set_xlabel('val top-1 accuracy')
axA.set_title('A · Clean-audio accuracy (val, 498-class)', fontsize=12, weight='bold', loc='left', pad=10)
axA.xaxis.grid(True, color=GRID, zorder=0); axA.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    axA.spines[s].set_visible(False)
axA.tick_params(length=0)

x = range(len(scen))
axB.plot(x, multimodal, '-o', color=BLUE, lw=2.4, ms=6, zorder=4, label='multimodal (shipped)')
axB.plot(x, audio_only, '-o', color=ORANGE, lw=2.4, ms=6, zorder=4, label='audio-only')
axB.axhline(visual_flat, ls='--', lw=1.8, color=BASE, zorder=2, label='visual-only (flat)')
axB.text(len(scen) - 1 + 0.12, multimodal[-1], 'multimodal', color=BLUE, va='center', fontsize=10, weight='bold')
axB.text(len(scen) - 1 + 0.12, audio_only[-1], 'audio', color=ORANGE, va='center', fontsize=10, weight='bold')
axB.text(0.05, visual_flat + 0.012, 'visual-only', color=MUTED, va='bottom', fontsize=9)
axB.set_xticks(list(x)); axB.set_xticklabels(scen, fontsize=9)
axB.set_ylim(0.25, 0.78); axB.set_xlim(-0.3, len(scen) + 0.9)
axB.set_ylabel('val top-1 accuracy')
axB.set_title('B · Noise robustness (val)', fontsize=12, weight='bold', loc='left', pad=10)
axB.yaxis.grid(True, color=GRID, zorder=0); axB.set_axisbelow(True)
for s in ('top', 'right'):
    axB.spines[s].set_visible(False)
axB.tick_params(length=0)

fig.suptitle('GLips multimodal — architecture & robustness reference (validation split)',
             fontsize=13.5, weight='bold', x=0.02, ha='left')
fig.text(0.02, 0.005, 'Frozen fusion heads (concat/late/joint_tf/cross_attn-frozen) have clean-val points only; '
         'full noise curves are in the test-split figure.', fontsize=8.5, color=MUTED, ha='left')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}')
