"""Test-split figure: every architecture's top-1 across every noise scenario, from master_scenarios.csv."""
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'test_scenarios.png')

INK, MUTED, GRID = '#1a1a1a', '#5a5a5a', '#dcdcdc'
STYLE = {  # system -> (color, linestyle, marker, label, zorder)
    'joint_tf (AV-HuBERT-style)': ('#2f5f8a', '-', 'o', 'joint_tf', 6),
    'cross_attn (frozen)':        ('#a1592c', '-', 's', 'cross_attn', 6),
    'concat':                     ('#4c724c', '-', '^', 'concat', 5),
    'late':                       ('#a3821b', '-', 'D', 'late', 5),
    'audio_only':                 ('#7a4a6a', '-', 'v', 'nur Audio', 5),
    'visual_only':                ('#8c8c8c', '--', None, 'nur visuell', 3),
}
PLOT_ORDER = ['joint_tf (AV-HuBERT-style)', 'cross_attn (frozen)', 'concat', 'late',
              'audio_only', 'visual_only']

data = {}
with open(os.path.join(ROOT, 'results', 'master_scenarios.csv')) as f:
    for r in csv.DictReader(f):
        data[r['system']] = {k: (float(v) if v else None) for k, v in r.items() if k != 'system'}

XLAB = ['rein', '15 dB', '10 dB', '5 dB']
WHITE = ['clean', 'white 15', 'white 10', 'white 5']
BABBLE = ['clean', 'babble 15', 'babble 10', 'babble 5']

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.edgecolor': MUTED})
fig, (axW, axB) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

for ax, cols, title in [(axW, WHITE, 'Weißes Rauschen'), (axB, BABBLE, 'Stimmengewirr')]:
    x = range(4)
    for sysname in PLOT_ORDER:
        color, ls, mk, lab, z = STYLE[sysname]
        y = [data[sysname][c] for c in cols]
        ax.plot(x, y, ls, color=color, marker=mk, lw=1.6, ms=4.5, zorder=z,
                label=lab if ax is axW else None)
    ax.set_xticks(list(x)); ax.set_xticklabels(XLAB)
    ax.set_xlabel('Signal-Rausch-Verhältnis')
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylim(0.27, 0.77)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axW.set_ylabel('Top-1-Genauigkeit (Test)')

handles, labels = axW.get_legend_handles_labels()
fig.legend(handles, labels, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           frameon=False, fontsize=9, columnspacing=1.3, handletextpad=0.4)

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}')
