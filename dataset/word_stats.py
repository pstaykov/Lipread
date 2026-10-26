"""Word statistics for vocabulary selection: counts, speakers, visibility, phonemes, visemes, homophenes.

    python dataset/word_stats.py            # reads bundestag_data/catalog.jsonl + index.jsonl (HD speeches only)

Writes bundestag_data/words/{words.csv, homophenes.csv, near_pairs.csv}.
Visemes: Weiss & Aschenberner 2005 (15 classes); pronunciations: MFA german_mfa.dict.
"""
import csv
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bundestag_data"
OUT = DATA / "words"
DICT_URL = "https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/german/mfa/german_mfa.dict"
STOP_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-de/master/stopwords-de.txt"

# MFA (IPA) phone -> Weiss & Aschenberner viseme. Diphthongs/affricates = two visemes (as in the paper).
VISEME = {
    "p": "P", "pʰ": "P", "b": "P",
    "t": "T", "tʰ": "T", "d": "T", "k": "T", "kʰ": "T", "ɡ": "T", "ɟ": "T", "c": "T", "cʰ": "T",
    "n": "N", "n̩": "N", "l": "N", "l̩": "N", "ɲ": "N",
    "m": "M", "m̩": "M",
    "f": "F", "v": "F",
    "s": "S", "z": "S",
    "ʃ": "Z", "ʒ": "Z", "tʃ": "Z", "dʒ": "Z",
    "h": "R", "ʁ": "R", "x": "R", "ŋ": "R",
    "j": "C", "ç": "C",
    "iː": "E", "ɪ": "E", "i": "E", "eː": "E", "e": "E", "ɛ": "E", "ɛː": "E",
    "a": "A", "aː": "A",
    "oː": "O", "o": "O", "ɔ": "O",
    "uː": "U", "u": "U", "ʊ": "U",
    "ə": "Q", "ɐ": "Q",
    "yː": "Y", "y": "Y", "ʏ": "Y", "øː": "Y", "ø": "Y", "œ": "Y",
    "aj": "AE", "aw": "AU", "ɔʏ": "OE", "ts": "TS", "pf": "PF",
}
TOKEN = re.compile(r"[A-Za-zÄÖÜäöüß]+(?:-[A-Za-zÄÖÜäöüß]+)*")
MIN_LEN = 4
VISIBLE_FRAC = 0.8          # sentence counts as visible if >= 80 % of its frames are valid face frames
GROUP_MIN_COUNT = 100       # words considered for homophene groups / near pairs
HD_FROM = "2016-08-01"      # Mediathek has 1920x1080 from Sept 2016 on; older video is 720x400 (mouth ~19 px)


def fetch(url, path):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def load_lexicon():
    lex = {}
    for line in open(fetch(DICT_URL, OUT / "german_mfa.dict"), encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0] not in lex:      # first variant
            lex[parts[0]] = parts[-1].split()
    return lex


def to_visemes(phones):
    try:
        return "".join(VISEME[p] for p in phones)
    except KeyError:
        return None


def read_jsonl(path):
    out = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:                  # partial last line while a writer is running
                pass
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stop = set(open(fetch(STOP_URL, OUT / "stopwords-de.txt"), encoding="utf-8").read().split())
    lex = load_lexicon()

    catalog = {e["id"]: e for e in read_jsonl(DATA / "catalog.jsonl") if e["date"] >= HD_FROM}
    kept = {r["id"] for r in read_jsonl(DATA / "index.jsonl") if r["status"] == "kept"}

    count, vis_count, mined_count = Counter(), Counter(), Counter()
    speakers, vis_speakers = defaultdict(set), defaultdict(set)
    surface = defaultdict(Counter)
    for sid, e in catalog.items():
        valid = None
        if sid in kept:
            p = DATA / "speeches" / sid / "facemesh.npz"
            if p.exists():
                valid = np.load(p)["valid"]
        for s, t, text in e["sentences"]:
            visible = False
            if valid is not None:
                a, b = int(s * 25), min(int(t * 25), len(valid))
                visible = b > a and valid[a:b].mean() >= VISIBLE_FRAC
            for tok in TOKEN.findall(text):
                w = tok.lower()
                count[w] += 1
                speakers[w].add(e["speaker_id"])
                surface[w][tok] += 1
                if valid is not None:
                    mined_count[w] += 1
                if visible:
                    vis_count[w] += 1
                    vis_speakers[w].add(e["speaker_id"])

    rows = []
    for w, n in count.items():
        if len(w) < MIN_LEN or w in stop or "-" in w:
            continue
        phones = lex.get(w) or lex.get(w.replace("ß", "ss"))
        vis = to_visemes(phones) if phones else None
        rows.append({"word": surface[w].most_common(1)[0][0], "key": w, "count": n,
                     "speakers": len(speakers[w]), "mined_count": mined_count[w], "vis_count": vis_count[w],
                     "vis_speakers": len(vis_speakers[w]),
                     "phonemes": " ".join(phones) if phones else "", "visemes": vis or "",
                     "n_visemes": len(vis) if vis else 0})
    rows.sort(key=lambda r: -r["count"])

    # homophene groups: same viseme string, different phonemes
    groups = defaultdict(list)
    for r in rows:
        if r["visemes"] and r["count"] >= GROUP_MIN_COUNT:
            groups[r["visemes"]].append(r)
    gid = {}
    hom = []
    for v, members in groups.items():
        if len({m["phonemes"] for m in members}) < 2:
            continue
        for m in members:
            gid[m["key"]] = v
        keys = [m["key"] for m in members]
        stem = min(len(k) for k in keys) - 2           # inflection-only group, e.g. geehrte/geehrter
        hom.append({"visemes": v, "size": len(members),
                    "inflection_only": all(k[:stem] == keys[0][:stem] for k in keys),
                    "min_count": min(m["count"] for m in members),
                    "min_speakers": min(m["speakers"] for m in members),
                    "members": " | ".join(f"{m['word']} ({m['count']}, {m['speakers']} spk, /{m['phonemes']}/)"
                                          for m in members)})
    hom.sort(key=lambda g: -g["min_count"])
    for r in rows:
        r["homophene_group"] = gid.get(r["key"], "")
        r["group_size"] = len(groups[gid[r["key"]]]) if r["key"] in gid else 1

    # near pairs: viseme strings differing by exactly one substituted viseme
    buckets = defaultdict(list)
    cand = [r for r in rows if r["visemes"] and r["count"] >= GROUP_MIN_COUNT]
    for r in cand:
        v = r["visemes"]
        for i in range(len(v)):
            buckets[v[:i] + "*" + v[i + 1:]].append(r)
    near = {}
    for key, members in buckets.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a["visemes"] != b["visemes"]:
                    pair = tuple(sorted((a["key"], b["key"])))
                    near[pair] = {"a": a["word"], "b": b["word"], "visemes_a": a["visemes"],
                                  "visemes_b": b["visemes"], "min_count": min(a["count"], b["count"]),
                                  "min_speakers": min(a["speakers"], b["speakers"])}
    near = sorted(near.values(), key=lambda p: -p["min_count"])

    def write(name, items, fields):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fields, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(items)

    write("words.csv", rows, ["word", "count", "speakers", "mined_count", "vis_count", "vis_speakers",
                              "phonemes", "visemes", "n_visemes", "group_size", "homophene_group"])
    write("homophenes.csv", hom, ["visemes", "size", "inflection_only", "min_count", "min_speakers", "members"])
    write("near_pairs.csv", near, ["a", "b", "visemes_a", "visemes_b", "min_count", "min_speakers"])

    no_pron = sum(1 for r in rows if not r["phonemes"] and r["count"] >= GROUP_MIN_COUNT)
    print(f"{len(catalog)} speeches ({len(kept)} mined), {sum(count.values())} tokens, "
          f"{len(rows)} content words, {sum(r['count'] >= GROUP_MIN_COUNT for r in rows)} with >= "
          f"{GROUP_MIN_COUNT} occurrences ({no_pron} without pronunciation), "
          f"{len(hom)} homophene groups, {len(near)} near pairs")


if __name__ == "__main__":
    main()
