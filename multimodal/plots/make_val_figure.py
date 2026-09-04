"""Val-split reference figure: panel A clean-audio top-1 per system, panel B noise-robustness curves."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'val_reference.png')

BLUE, ORANGE, GREEN, GOLD = '#2f5f8a', '#a1592c', '#4c724c', '#a3821b'
INK, MUTED, GRID = '#1a1a1a', '#5a5a5a', '#dcdcdc'
BASE = '#8c8c8c'   # baseline gray (audio/visual references)

clean = [
    ('joint_tf',                 0.7286, BLUE),
    ('cross_attn (finetuned)',   0.7166, ORANGE),
    ('cross_attn (frozen)',      0.7141, ORANGE),
    ('audio_only',               0.6137, BASE),
    ('concat',                   0.6113, GREEN),
    ('late',                     0.6055, GOLD),
    ('visual_only',              0.3332, BASE),
]

scen = ['rein', 'weiß\n15', 'weiß\n10', 'weiß\n5', 'gewirr\n15', 'gewirr\n10', 'gewirr\n5']
multimodal = [0.7163, 0.6998, 0.6792, 0.6421, 0.6951, 0.6603, 0.5702]
audio_only = [0.6118, 0.5162, 0.4546, 0.3508, 0.5626, 0.4825, 0.2932]
visual_flat = 0.3332

plt.rcParams.update({'font.size': 10, 'font.family': 'serif',
                     'axes.edgecolor': MUTED, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1, 1.15]})

names = [n for n, _, _ in clean][::-1]
vals = [v for _, v, _ in clean][::-1]
cols = [c for _, _, c in clean][::-1]
y = range(len(names))
axA.barh(list(y), vals, color=cols, height=0.62, zorder=3)
axA.set_yticks(list(y)); axA.set_yticklabels(names, fontsize=9)
axA.set_xlim(0, 0.85); axA.set_xlabel('Top-1-Genauigkeit (Val)')
axA.set_title('A', fontsize=11, loc='left')
axA.xaxis.grid(True, color=GRID, zorder=0); axA.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    axA.spines[s].set_visible(False)
axA.tick_params(length=0)

x = range(len(scen))
axB.plot(x, multimodal, '-o', color=BLUE, lw=1.6, ms=4.5, zorder=4, label='multimodal')
axB.plot(x, audio_only, '-o', color=ORANGE, lw=1.6, ms=4.5, zorder=4, label='nur Audio')
axB.axhline(visual_flat, ls='--', lw=1.2, color=BASE, zorder=2, label='nur visuell')
axB.set_xticks(list(x)); axB.set_xticklabels(scen, fontsize=8.5)
axB.set_ylim(0.25, 0.78); axB.set_xlim(-0.3, len(scen) - 0.7)
axB.set_ylabel('Top-1-Genauigkeit (Val)')
axB.set_title('B', fontsize=11, loc='left')
axB.yaxis.grid(True, color=GRID, zorder=0); axB.set_axisbelow(True)
for s in ('top', 'right'):
    axB.spines[s].set_visible(False)
axB.tick_params(length=0)
axB.legend(frameon=False, fontsize=8.5, loc='upper right')

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}')
