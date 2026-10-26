"""Final vocabulary for the HD pool (run after word_stats.py).

Words are ranked by estimated usable clips if every HD speech is mined (see `est` below).

    python dataset/select_vocab.py   ->  bundestag_data/words/vocab.csv

"Relevant" word = not a stopword, >= 4 letters (word_stats.py), majority POS in context is
NOUN/PROPN/VERB/ADJ (spaCy de_core_news_sm), not a filler or a fixed address formula (BLOCK).

Roles:
    homophene  same viseme string, >= 2 different lemmas (inflection-only groups drop out)
    near_pair  one viseme substituted, different lemmas
    control    unique viseme string, most visible (fills the vocabulary up to TOTAL)
    other      shares its viseme string with a relevant word outside the vocabulary (fill as well)
Fill words: one form per word family, so the 500 classes are distinct words, not inflections.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import spacy

from word_stats import TOKEN, read_jsonl

DATA = Path(__file__).resolve().parent.parent / "bundestag_data"
WORDS = DATA / "words"
POS_OK = {"NOUN", "PROPN", "VERB", "ADJ"}
CONTEXTS = 30                 # sentences tagged per word
EST_MIN, MIN_SPK = 500, 100   # estimated usable clips in the HD pool, speakers in pool
SHRINK = 50                   # per-word visible rate is shrunk towards the global rate with this many pseudo-counts
ALIGN_KEEP = 0.85             # assumed share of instances that survive forced alignment
NEAR_MAX = 100                # pairs
TOTAL = 500
BLOCK = {
    # fillers / discourse words
    "genau", "einfach", "eigentlich", "nämlich", "glaube", "gerne", "bitte", "leider", "wirklich",
    "natürlich", "tatsächlich", "ganz", "ehrlich", "übrigens", "letztlich", "sozusagen", "sodass",
    "bisschen", "übrigen", "sogenannten", "sogenannte", "müsste", "weiß", "zitiere", "zitat",
    # address formulas
    "präsident", "präsidentin", "verehrte", "verehrten", "kollege", "kollegin", "kollegen", "kolleginnen", "geehrte",
    "geehrter", "geehrten", "damen", "herren", "liebe", "lieber", "lieben", "dank", "danke",
    "herzlichen", "zuschauer", "zuschauerinnen", "tribüne", "frau", "herr", "herrn",
}


def related(a, b):
    """Same word family: Hilfe/hilft, Rede/reden, Bildung/Weiterbildung (but not Staat/Stadt)."""
    n = min(len(a), len(b))
    common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), n)
    return a in b or b in a or (n >= 4 and common >= n - 1) or (common >= 5 and common >= 0.75 * n)


def tag(cands):
    """Majority POS and lemma of each candidate word over up to CONTEXTS sentences."""
    ctx = defaultdict(list)
    for e in read_jsonl(DATA / "catalog.jsonl"):
        for _, _, text in e["sentences"]:
            for tok in {t.lower() for t in TOKEN.findall(text)}:
                if tok in cands and len(ctx[tok]) < CONTEXTS:
                    ctx[tok].append(text)
    sents = list({s for v in ctx.values() for s in v})
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    pos, lemma = defaultdict(Counter), defaultdict(Counter)
    for doc in nlp.pipe(sents, batch_size=256):
        for t in doc:
            w = t.text.lower()
            if w in cands:
                pos[w][t.pos_] += 1
                lemma[w][t.lemma_.lower()] += 1
    return ({w: c.most_common(1)[0][0] for w, c in pos.items()},
            {w: c.most_common(1)[0][0] for w, c in lemma.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-est", type=int, default=EST_MIN)
    ap.add_argument("--out", default="vocab.csv")
    args = ap.parse_args()
    words = {r["word"].lower(): r for r in csv.DictReader(open(WORDS / "words.csv", encoding="utf-8"))}
    for r in words.values():
        for k in ("count", "speakers", "mined_count", "vis_count", "vis_speakers", "n_visemes", "group_size"):
            r[k] = int(r[k])

    # estimated usable clips if the whole HD pool is mined:
    # transcript count x visible rate (measured per word in mined speeches) x speech keep rate x alignment
    status = Counter(r["status"] for r in read_jsonl(DATA / "index.jsonl"))
    keep = status["kept"] / (status["kept"] + status["rejected"])
    rate = sum(r["vis_count"] for r in words.values()) / sum(r["mined_count"] for r in words.values())
    for r in words.values():
        r["est"] = round(r["count"] * (r["vis_count"] + SHRINK * rate) / (r["mined_count"] + SHRINK)
                         * keep * ALIGN_KEEP)
    print(f"visible rate {rate:.2f}, speech keep rate {keep:.2f}, alignment keep {ALIGN_KEEP} (assumed)")

    cands = {w for w, r in words.items() if r["visemes"] and r["est"] >= args.min_est
             and r["speakers"] >= MIN_SPK and w not in BLOCK}
    pos, lemma = tag(cands)
    rel = {w for w in cands if pos.get(w) in POS_OK}
    print(f"{len(cands)} candidates (>= {args.min_est} estimated clips, >= {MIN_SPK} speakers), {len(rel)} relevant")

    out = {}

    # homophene groups: relevant members, one form per word family (most visible), >= 2 families;
    # the same lemma set in another inflection (setzen/sitzen vs setzt/sitzt) is kept only once
    vis = lambda w: words[w]["est"]
    groups = defaultdict(list)
    for w in sorted(rel, key=vis, reverse=True):
        g = groups[words[w]["visemes"]]
        if not any(lemma[w] == lemma[o] or related(w, o) for o in g):
            g.append(w)
    seen = set()
    for g, members in sorted(groups.items(), key=lambda x: -min(map(vis, x[1]))):
        key = frozenset(lemma[w] for w in members)
        if len(members) >= 2 and key not in seen:
            seen.add(key)
            for w in members:
                out[w] = ("homophene", g)

    # near pairs: both relevant, not the same word family, one pair per lemma pair, strongest first
    pairs = []
    for p in csv.DictReader(open(WORDS / "near_pairs.csv", encoding="utf-8")):
        a, b = p["a"].lower(), p["b"].lower()
        if a in rel and b in rel and lemma[a] != lemma[b] and not related(a, b):
            pairs.append((min(vis(a), vis(b)), a, b))
    n_pairs = 0
    for v, a, b in sorted(pairs, reverse=True):
        key = frozenset((lemma[a], lemma[b]))
        if n_pairs >= NEAR_MAX or key in seen \
                or any(lemma[w] == lemma[o] or related(w, o) for w in (a, b) for o in out if o not in (a, b)):
            continue
        seen.add(key)
        n_pairs += 1
        for w, o in ((a, b), (b, a)):
            out.setdefault(w, ("near_pair", f"{words[w]['word']}~{words[o]['word']}"))

    # fill up to TOTAL with the most visible relevant words, one form per word family
    for w in sorted(rel, key=vis, reverse=True):
        if len(out) >= TOTAL:
            break
        if w in out or any(lemma[w] == lemma[o] or related(w, o) for o in out):
            continue
        r = words[w]
        out[w] = ("control", r["visemes"]) if len(groups[r["visemes"]]) <= 1 else ("other", r["visemes"])

    rows = [{"word": words[w]["word"], "role": role, "group": grp, "pos": pos[w], "lemma": lemma[w],
             **{k: words[w][k] for k in ("visemes", "phonemes", "count", "speakers", "est", "vis_count", "vis_speakers")}}
            for w, (role, grp) in sorted(out.items(), key=lambda x: (x[1][0], x[1][1], x[0]))]
    with open(WORDS / args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    for role in ("homophene", "near_pair", "control", "other"):
        v = sorted(r["est"] for r in rows if r["role"] == role)
        if v:
            print(f"{role:10s} {len(v):4d} words, estimated clips min {v[0]} / median {v[len(v) // 2]}")
    print(f"{len(rows)} words, homophene groups: {len({r['group'] for r in rows if r['role'] == 'homophene'})}, "
          f"near pairs: {n_pairs}")


if __name__ == "__main__":
    main()
