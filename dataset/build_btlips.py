"""Turn the selected clips (bundestag_data/clips/dataset.jsonl, split_clips.py) into the GLips-style dataset.

    python dataset/build_btlips.py

BTLips/lipread_files/<word>/<split>/<word>_<speaker>-<instance>.{mp4,wav,txt}   (clips are moved, not copied)
BTLips/speakerlist.txt, titlelist.txt, wordlist.csv, README.txt
BTLips_facemesh/<split>.npy (N, 29, 478, 3) float16 + <split>.txt (clip names in the same order)
"""
import json
import shutil
import sys
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_clips import CLIP, FPS, OUT  # noqa: E402
from word_stats import read_jsonl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DS, FM = REPO / "BTLips", REPO / "BTLips_facemesh"
SPLITS = ("train", "val", "test")


def main():
    clips = read_jsonl(OUT / "dataset.jsonl")
    ids = {c["id"] for c in clips}
    cat = {c["id"]: c for c in read_jsonl(REPO / "bundestag_data" / "catalog.jsonl") if c["id"] in ids}
    spk = {s: i + 1 for i, s in enumerate(sorted({c["speaker_id"] for c in clips}, key=lambda s: int(s[1:])))}
    sp_ids = sorted({c["id"] for c in clips})
    speech = {s: i + 1 for i, s in enumerate(sp_ids)}
    inst = defaultdict(int)
    for c in sorted(clips, key=lambda c: (c["word"], c["speaker_id"], c["id"], c["f0"])):
        inst[c["word"], c["speaker_id"]] += 1
        c["name"] = f"{c['word']}_{spk[c['speaker_id']]:04d}-{inst[c['word'], c['speaker_id']]:04d}"

    DS.mkdir(exist_ok=True)
    FM.mkdir(exist_ok=True)
    first = {}
    for c in clips:
        first.setdefault(c["speaker_id"], c)
    with open(DS / "speakerlist.txt", "w", encoding="utf-8") as f:
        for s, i in spk.items():
            e = cat[first[s]["id"]]
            f.write(f"{i:04d}\t{s}\t{e['speaker']}\t{e.get('faction') or ''}\n")
    with open(DS / "titlelist.txt", "w", encoding="utf-8") as f:
        for s in sp_ids:
            e = cat[s]
            f.write(f"{speech[s]:05d}\t{s}\t{e['date'][:10]}\t{spk[e['speaker_id']]:04d}\t{e['agenda']}\n")
    shutil.copy(REPO / "bundestag_data" / "words" / "vocab.csv", DS / "wordlist.csv")

    def one(c):
        d = DS / "lipread_files" / c["word"] / c["split"]
        src, dst = OUT / (c["path"] + ".mp4"), d / (c["name"] + ".mp4")
        z = np.load(OUT / (c["path"] + ".npz"))
        if not dst.exists():
            d.mkdir(parents=True, exist_ok=True)
            with wave.open(str(d / (c["name"] + ".wav")), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(z["audio"].tobytes())
            s0 = c["f0"] / FPS
            e = cat[c["id"]]
            (d / (c["name"] + ".txt")).write_text(
                f"Spoken word: {c['word']}\n"
                f"Starttime of utterance in seconds: {c['t0']:.3f}\n"
                f"Endtime of utterance in seconds: {c['t1']:.3f}\n"
                f"Duration of utterance in seconds: {c['t1'] - c['t0']:.3f}\n"
                f"Starttime of clip in seconds: {s0:.3f}\n"
                f"Endtime of clip in seconds: {s0 + CLIP / FPS:.3f}\n"
                f"Original speech: {c['id']} - {e['date'][:10]} - {e['speaker']}\n"
                f"Corresponding number in titlelist.txt: {speech[c['id']]:05d}\n"
                f"Speaker number in speakerlist.txt: {spk[c['speaker_id']]:04d}\n"
                f"Mouth width in source pixels: {c['mouth_px']}\n", encoding="utf-8")
            shutil.move(str(src), str(dst))
        return z["landmarks"]

    for split in SPLITS:
        part = sorted((c for c in clips if c["split"] == split), key=lambda c: c["name"])
        arr = np.lib.format.open_memmap(FM / f"{split}.npy", "w+", np.float16, (len(part), CLIP, 478, 3))
        with ThreadPoolExecutor(12) as ex:
            for i, lm in enumerate(ex.map(one, part)):
                arr[i] = lm
                if i % 20000 == 0:
                    print(split, i, len(part), flush=True)
        arr.flush()
        del arr
        (FM / f"{split}.txt").write_text("".join(f"{c['word']}/{split}/{c['name']}\n" for c in part), encoding="utf-8")
    print("done", len(clips), flush=True)


if __name__ == "__main__":
    main()
