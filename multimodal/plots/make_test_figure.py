"""Test-split figure: every architecture's top-1 across every noise scenario, from master_scenarios.csv."""
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'test_scenarios.png')

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e7e6e2'
STYLE = {  # system -> (color, linestyle, marker, label, zorder)
    'joint_tf (AV-HuBERT-style)': ('#2a78d6', '-', 'o', 'joint_tf (AV-HuBERT-style)', 6),
    'cross_attn (frozen)':        ('#eb6834', '-', 's', 'cross_attn', 6),
    'concat':                     ('#1baf7a', '-', '^', 'concat', 5),
    'late':                       ('#eda100', '-', 'D', 'late', 5),
    'audio_only':                 ('#e87ba4', '-', 'v', 'audio-only', 5),
    'visual_only':                ('#9a9892', '--', None, 'visual-only', 3),
}
PLOT_ORDER = ['joint_tf (AV-HuBERT-style)', 'cross_attn (frozen)', 'concat', 'late',
              'audio_only', 'visual_only']

data = {}
with open(os.path.join(ROOT, 'results', 'master_scenarios_500.csv')) as f:
    for r in csv.DictReader(f):
        data[r['system']] = {k: (float(v) if v else None) for k, v in r.items() if k != 'system'}

XLAB = ['clean', '15 dB', '10 dB', '5 dB']
WHITE = ['clean', 'white 15', 'white 10', 'white 5']
BABBLE = ['clean', 'babble 15', 'babble 10', 'babble 5']

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.edgecolor': MUTED})
fig, (axW, axB) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)

for ax, cols, title in [(axW, WHITE, 'White noise'), (axB, BABBLE, 'Babble noise')]:
    x = range(4)
    for sysname in PLOT_ORDER:
        color, ls, mk, lab, z = STYLE[sysname]
        y = [data[sysname][c] for c in cols]
        ax.plot(x, y, ls, color=color, marker=mk, lw=2.4, ms=6, zorder=z,
                label=lab if ax is axW else None,
                markeredgecolor='white', markeredgewidth=0.8 if mk else 0)
    ax.set_xticks(list(x)); ax.set_xticklabels(XLAB)
    ax.set_xlabel('audio SNR  (clean → 5 dB)')
    ax.set_title(title, fontsize=12, weight='bold', loc='left', pad=8)
    ax.set_ylim(0.27, 0.77)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axW.set_ylabel('test top-1 accuracy')

handles, labels = axW.get_legend_handles_labels()
fig.legend(handles, labels, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           frameon=False, fontsize=10, columnspacing=1.4, handletextpad=0.5)

fig.suptitle('GLips multimodal — architecture robustness across noise (test split, 500-class)',
             fontsize=13.5, weight='bold', x=0.02, ha='left')
fig.text(0.02, 0.045, 'concat/late tie audio-only at clean but stay far above it under noise '
         '(visual fallback); audio-only collapses toward the visual-only floor.',
         fontsize=9, color=MUTED, ha='left')
fig.tight_layout(rect=[0, 0.09, 1, 0.95])
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}')
