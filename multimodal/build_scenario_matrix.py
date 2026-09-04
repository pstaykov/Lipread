"""Assemble the systems x scenarios test-split matrix (top-1) into one wide CSV."""
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
VISUAL_ONLY_TEST = float(os.environ.get('VISUAL_ONLY_TEST', '0'))  # audio-independent, constant across scenarios

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
    frozen = {}
    audio = None
    visual = None
    for v in ['late', 'concat', 'cross_attn', 'joint_tf']:
        d = read_snr_csv(os.path.join(HERE, 'results', f'snr_fusion_{v}_test_500.csv'))
        if v in d:
            frozen[v] = d[v]
        if audio is None and 'audio_only' in d:
            audio = d['audio_only']
        if visual is None and 'visual_only' in d:
            visual = d['visual_only']

    shipped = read_snr_csv(os.path.join(HERE, 'results', 'snr_results_500.csv')).get('multimodal', {})

    visual_only_vals = visual or {s: VISUAL_ONLY_TEST for s in SCENARIOS}

    systems = [
        ('joint_tf (AV-HuBERT-style)', frozen.get('joint_tf', {})),
        ('cross_attn (frozen)',        frozen.get('cross_attn', {})),
        ('cross_attn (finetuned)',     shipped),
        ('concat',                     frozen.get('concat', {})),
        ('late',                       frozen.get('late', {})),
        ('audio_only',                 audio or {}),
        ('visual_only',                visual_only_vals),
    ]

    rows = []
    for name, d in systems:
        vals = [d.get(s) for s in SCENARIOS]
        rows.append((name, vals))
    rows.sort(key=lambda nv: (nv[1][0] if nv[1][0] is not None else -1), reverse=True)

    out = os.path.join(HERE, 'results', 'master_scenarios_500.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system'] + SCEN_LABEL)
        for name, vals in rows:
            w.writerow([name] + [f'{v:.4f}' if v is not None else '' for v in vals])

    print(f'master_scenarios_500.csv (test, 500-class, top-1):\n')
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
