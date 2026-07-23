"""Assemble the systems x scenarios test-split matrix (top-1) into one wide CSV.

Scenarios: clean + white/babble at 15/10/5 dB. Systems: the four frozen fusion
heads, the shipped fine-tuned cross_attn, audio-only (Whisper probe), visual-only
(standalone model, audio-independent so flat across scenarios).

Sources (all test split, 24900 clips):
  snr_fusion_<v>_test.csv  -> frozen head <v> + audio_only rows
  snr_results_test.csv     -> shipped model (system 'multimodal')
  visual_only              -> standalone model on cached test tokens (constant)
"""
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
VISUAL_ONLY_TEST = 0.3356   # standalone GLipsNet on cached test tokens (audio-independent)

SCENARIOS = [('none', 'clean'), ('white', '15'), ('white', '10'), ('white', '5'),
             ('babble', '15'), ('babble', '10'), ('babble', '5')]
SCEN_LABEL = ['clean', 'white 15', 'white 10', 'white 5', 'babble 15', 'babble 10', 'babble 5']


def read_snr_csv(path):
    """path -> {system: {(noise, snr): top1}}"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(r['system'], {})[(r['noise'], r['snr_db'])] = float(r['top1'])
    return out


def main():
    # frozen fusion heads + audio_only come from the per-variant fusion sweeps
    frozen = {}
    audio = None
    for v in ['late', 'concat', 'cross_attn', 'joint_tf']:
        d = read_snr_csv(os.path.join(HERE, 'results', f'snr_fusion_{v}_test.csv'))
        if v in d:
            frozen[v] = d[v]
        if audio is None and 'audio_only' in d:
            audio = d['audio_only']

    shipped = read_snr_csv(os.path.join(HERE, 'results', 'snr_results_test.csv')).get('multimodal', {})

    # display name -> per-scenario dict (or constant for visual_only)
    systems = [
        ('joint_tf (AV-HuBERT-style)', frozen.get('joint_tf', {})),
        ('cross_attn (frozen)',        frozen.get('cross_attn', {})),
        ('cross_attn (finetuned)',     shipped),
        ('concat',                     frozen.get('concat', {})),
        ('late',                       frozen.get('late', {})),
        ('audio_only',                 audio or {}),
        ('visual_only',                {s: VISUAL_ONLY_TEST for s in SCENARIOS}),
    ]

    rows = []
    for name, d in systems:
        vals = [d.get(s) for s in SCENARIOS]
        rows.append((name, vals))
    # sort by clean accuracy desc (visual_only/audio_only keep their spot by value)
    rows.sort(key=lambda nv: (nv[1][0] if nv[1][0] is not None else -1), reverse=True)

    out = os.path.join(HERE, 'results', 'master_scenarios.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system'] + SCEN_LABEL)
        for name, vals in rows:
            w.writerow([name] + [f'{v:.4f}' if v is not None else '' for v in vals])

    print(f'master_scenarios.csv (test, 24900 clips, 498-class, top-1):\n')
    hdr = f'{"system":28s}' + ''.join(f'{s:>11}' for s in SCEN_LABEL)
    print(hdr)
    for name, vals in rows:
        print(f'{name:28s}' + ''.join(f'{v:>11.4f}' if v is not None else f'{"—":>11}' for v in vals))
    missing = [n for n, v in rows if any(x is None for x in v)]
    if missing:
        print(f'\nWARNING incomplete rows (sweep not done?): {missing}')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
