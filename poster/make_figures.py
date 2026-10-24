"""Erzeugt die Bild- und Datendateien für main.tex in poster/figures/.

Ausführen aus poster/:  python make_figures.py
- Filmstreifen (Mund-ROI) für drei Wörter, je 12 gleichverteilte Frames
- Augmentierungsbeispiele eines Frames
- Ausschnitte (Gesicht mit Landmarken, Mundausschnitt) aus roi_grid_2x2.png
- Trainingskurven als .dat (aus metrics.csv)
- Konfusionsmatrix GLips-15 (GNet + Transfer) als TikZ-Datei
"""
import csv
import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LIP = os.path.join(REPO, 'lipreading')
ROI_ROOT = os.path.join(LIP, 'GLips_mouth', 'lipread_files')
OUT = os.path.join(HERE, 'figures')

STRIP_WORDS = ['digitalisierung', 'möglichkeit', 'aufgaben']
N_STRIP = 12
WORDS15 = ['aber', 'aufgaben', 'bleibt', 'darüber', 'digitalisierung', 'einmal', 'finden', 'gegen',
           'geworden', 'herrn', 'investitionen', 'kommunen', 'linke', 'möglichkeit', 'passiert']


def read_frames(path):
    # cv2 kann keine Nicht-ASCII-Pfade öffnen -> über Bytes einlesen ist bei mp4 nicht möglich,
    # daher Kurzpfad über das Arbeitsverzeichnis
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    cap = cv2.VideoCapture(os.path.basename(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    os.chdir(cwd)
    return frames


def first_clip(word, split='test'):
    d = os.path.join(ROI_ROOT, word, split)
    return os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith('.mp4'))[0])


def save(img, name, size=192):
    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    img.resize((size, size), Image.BICUBIC).save(os.path.join(OUT, name))


def strips():
    for wi, word in enumerate(STRIP_WORDS):
        frames = read_frames(first_clip(word))
        idx = np.linspace(0, len(frames) - 1, N_STRIP).round().astype(int)
        for k, i in enumerate(idx):
            save(frames[i], f'strip{wi}_{k}.png')
        print(f'{word}: {len(frames)} Frames, {frames[0].shape}')


def augmentations():
    frames = read_frames(first_clip('aber'))
    base = Image.fromarray(frames[len(frames) // 2]).resize((96, 96), Image.BICUBIC)
    crop = base.crop((4, 4, 92, 92))
    save(base, 'aug_base.png')
    save(crop, 'aug_crop.png')
    save(ImageOps.mirror(crop), 'aug_flip.png')
    save(ImageEnhance.Contrast(ImageEnhance.Brightness(crop).enhance(1.35)).enhance(0.75), 'aug_bright.png')
    save(ImageOps.grayscale(crop).convert('RGB'), 'aug_gray.png')
    save(crop.rotate(10, resample=Image.BICUBIC), 'aug_affine.png')
    erased = np.array(crop)
    erased[20:50, 46:80] = 0
    save(erased, 'aug_erase.png')


def roi_tiles():
    g = Image.open(os.path.join(HERE, 'roi_grid_2x2.png')).convert('RGB')
    w, h = g.size
    half = w // 2 - 5
    g.crop((0, 0, half, half)).save(os.path.join(OUT, 'face_landmarks.png'))
    g.crop((w - half, 0, w, half)).save(os.path.join(OUT, 'face_crop.png'))


def curves():
    runs = {
        'gnet15':   'Transformer_based/checkpoints_15',
        'mstcn15':  'mstcn_baseline/checkpoints_15',
        'gnet15t':  'Transformer_based/checkpoints_15_transfer',
        'mstcn15t': 'mstcn_baseline/checkpoints_15_transfer',
        'gnet500':  'Transformer_based/checkpoints_500',
        'mstcn500': 'mstcn_baseline/checkpoints_500',
        'gnet15d':  'Transformer_based/checkpoints_15_disjoint',
    }
    for name, rel in runs.items():
        with open(os.path.join(LIP, rel, 'metrics.csv'), newline='') as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(OUT, f'curve_{name}.dat'), 'w') as f:
            f.write('epoch train val top5\n')
            for r in rows:
                f.write(f"{r['epoch']} {100 * float(r['train_acc']):.2f} "
                        f"{100 * float(r['val_top1']):.2f} {100 * float(r['val_top5']):.2f}\n")


def confusion():
    cm = np.load(os.path.join(LIP, 'analysis', 'figures', 'confusion', 'glips15_GLipsNet_transfer.npy'))
    cm = cm / cm.sum(axis=1, keepdims=True)
    n = len(cm)
    lines = []
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            p = int(round(100 * min(v / 0.9, 1)))
            y = n - 1 - i
            lines.append(f'\\fill[bwkiblau!{p}!white] ({j},{y}) rectangle +(1,1);')
            if v >= 0.10:
                col = 'white' if v > 0.45 else 'black!70'
                lines.append(f'\\node[cmval, text={col}] at ({j + 0.5},{y + 0.5}) {{{round(100 * v):d}}};')
    for k, w in enumerate(WORDS15):
        lines.append(f'\\node[cmlab, anchor=east] at (-0.15,{n - 1 - k + 0.5}) {{{w}}};')
    with open(os.path.join(OUT, 'cm15.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('Diagonale:', np.round(np.diag(cm), 2))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    strips()
    augmentations()
    roi_tiles()
    curves()
    confusion()
