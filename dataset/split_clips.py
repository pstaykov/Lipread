"""Speaker-disjoint train/val/test split and the 500-clip selection per word.

    python dataset/split_clips.py

1. Speakers start in model/data/clips.py's hash split (80/10/10 of the speakers). A few speakers say a word
   often, so some words have too few val/test clips there; speakers are then moved one at a time between the
   splits while that lowers the total shortfall below NEED clips per word and split (hill climbing, fixed seed).
2. Per word and split, NEED clips are picked round-robin over the speakers (each speaker's clips by alignment
   score), so many speakers are covered.

Writes bundestag_data/clips/split.json {speaker_id: split} and dataset.jsonl (clips.jsonl lines of the selected
clips with a "split" field).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.data.clips import ROOT, read_index, split_of  # noqa: E402

SPLITS = ("train", "val", "test")
NEED = np.array([400, 50, 50])


def assign(L, seed=0):
    spk = sorted({c["speaker_id"] for c in L})
    words = sorted({c["word"] for c in L})
    si, wi = {s: i for i, s in enumerate(spk)}, {w: i for i, w in enumerate(words)}
    M = np.zeros((len(spk), len(words)), np.int64)
    for c in L:
        M[si[c["speaker_id"]], wi[c["word"]]] += 1
    a = np.array([SPLITS.index(split_of(s)) for s in spk])
    T = np.stack([M[a == k].sum(0) for k in range(3)])
    cost = lambda T: np.maximum(NEED[:, None] - T, 0).sum()
    rng = np.random.default_rng(seed)
    best, improved = cost(T), True
    print(f"start shortfall {best}", flush=True)
    while improved and best:
        improved = False
        for s in rng.permutation(len(spk)):
            for k in range(3):
                if k == a[s]:
                    continue
                T2 = T.copy()
                T2[a[s]] -= M[s]
                T2[k] += M[s]
                c2 = cost(T2)
                if c2 < best:
                    T, best, a[s], improved = T2, c2, k, True
        if not improved and best:                          # stuck: move a speaker of a short word plus any other one
            short = (T < NEED[:, None]).any(0)
            for s in np.flatnonzero(M[:, short].sum(1)):
                for k in range(3):
                    if k == a[s]:
                        continue
                    T1 = T.copy()
                    T1[a[s]] -= M[s]
                    T1[k] += M[s]
                    a1 = a.copy()
                    a1[s] = k
                    for k2 in range(3):
                        T2 = np.repeat(T1[None], len(spk), 0)
                        T2[np.arange(len(spk)), a1] -= M
                        T2[:, k2] += M
                        c2 = np.maximum(NEED[None, :, None] - T2, 0).sum((1, 2))
                        c2[a1 == k2] = best
                        t = int(c2.argmin())
                        if c2[t] < best:
                            T, best, improved = T2[t], c2[t], True
                            a[s], a[t] = k, k2
                            break
                    if improved:
                        break
                if improved:
                    break
        print(f"shortfall {best}", flush=True)
    return {s: SPLITS[a[i]] for i, s in enumerate(spk)}, T, words


def main():
    L = read_index()
    split, T, words = assign(L)
    for k, name in enumerate(SPLITS):
        print(f"{name}: {sum(v == name for v in split.values())} speakers, per word min {T[k].min()}, "
              f"words short {(T[k] < NEED[k]).sum()}")
    score = {r["path"]: float(r["a_score"]) for r in csv.DictReader(open(ROOT / "scan.csv", encoding="utf-8"))}
    groups = defaultdict(lambda: defaultdict(list))
    for c in L:
        groups[(c["word"], split[c["speaker_id"]])][c["speaker_id"]].append(c)
    out = []
    for (w, name), by_spk in sorted(groups.items()):
        queues = [sorted(v, key=lambda c: -score.get(c["path"], 0)) for _, v in sorted(by_spk.items())]
        picked, r = [], 0
        while len(picked) < NEED[SPLITS.index(name)] and any(len(q) > r for q in queues):
            picked += [q[r] for q in queues if len(q) > r][:NEED[SPLITS.index(name)] - len(picked)]
            r += 1
        out += [{**c, "split": name} for c in picked]
    with open(ROOT / "split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=0, sort_keys=True)
    with open(ROOT / "dataset.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in out)
    print(f"{len(out)} clips -> dataset.jsonl", flush=True)


if __name__ == "__main__":
    main()
