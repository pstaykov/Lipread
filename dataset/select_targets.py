"""Build the broad target-word set that steers mining (run after word_stats.py).

    python dataset/select_targets.py   ->  bundestag_data/words/targets.csv

Roles:
    homophene  members of real homophene groups (not inflection-only), every member >= 500 occ / 300 speakers
    near_pair  strongest one-viseme-apart pairs (different words, not just inflections)
    control    frequent words with a unique viseme string and no strong near pair
"""
import csv
from pathlib import Path

WORDS = Path(__file__).resolve().parent.parent / "bundestag_data" / "words"
HOM_MIN_COUNT, HOM_MIN_SPK = 500, 300
NEAR_MIN_COUNT, NEAR_MIN_SPK, NEAR_MAX = 1000, 400, 40
CTRL_MIN_COUNT, CTRL_MIN_SPK, CTRL_MAX, CTRL_MIN_LEN = 2000, 600, 40, 5


def same_stem(a, b):
    a, b = a.lower(), b.lower()
    n = min(len(a), len(b)) - 2
    return n > 2 and a[:n] == b[:n]


def main():
    words = {r["word"].lower(): r for r in csv.DictReader(open(WORDS / "words.csv", encoding="utf-8"))}
    out = {}

    for g in csv.DictReader(open(WORDS / "homophenes.csv", encoding="utf-8")):
        if g["inflection_only"] == "True" or int(g["min_count"]) < HOM_MIN_COUNT \
                or int(g["min_speakers"]) < HOM_MIN_SPK:
            continue
        for m in g["members"].split(" | "):
            out[m.split(" (")[0].lower()] = ("homophene", g["visemes"])

    near_words = set()
    n = 0
    for p in csv.DictReader(open(WORDS / "near_pairs.csv", encoding="utf-8")):
        if n >= NEAR_MAX or int(p["min_count"]) < NEAR_MIN_COUNT:
            break
        if int(p["min_speakers"]) < NEAR_MIN_SPK or same_stem(p["a"], p["b"]):
            continue
        for w, other in ((p["a"], p["b"]), (p["b"], p["a"])):
            near_words.add(w.lower())
            out.setdefault(w.lower(), ("near_pair", f"{w}~{other}"))
        n += 1

    n = 0
    for w, r in words.items():            # words.csv is sorted by count
        if n >= CTRL_MAX:
            break
        if (w in out or w in near_words or not r["visemes"] or r["group_size"] != "1"
                or len(w) < CTRL_MIN_LEN or int(r["count"]) < CTRL_MIN_COUNT
                or int(r["speakers"]) < CTRL_MIN_SPK):
            continue
        out[w] = ("control", r["visemes"])
        n += 1

    rows = [{"word": words[w]["word"], "role": role, "group": grp, "visemes": words[w]["visemes"],
             "count": words[w]["count"], "speakers": words[w]["speakers"]}
            for w, (role, grp) in out.items() if w in words]
    with open(WORDS / "targets.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    by = {}
    for r in rows:
        by[r["role"]] = by.get(r["role"], 0) + 1
    print(f"{len(rows)} target words: {by}")


if __name__ == "__main__":
    main()
