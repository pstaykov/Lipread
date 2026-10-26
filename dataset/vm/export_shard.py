"""Write the catalog part a second machine needs: dataset/vm/catalog_<k>of<n>.jsonl.gz + vocab.csv.

    python dataset/vm/export_shard.py --shard 1/2

Keeps HD speeches of the shard's speakers that this machine has not mined yet, and only the
sentences containing a vocab word (all mine_clips.py uses), so the file is small enough for git.
"""
import argparse
import csv
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mine_clips import OUT, WORD, shard_of                   # noqa: E402
from mine_bundestag import DATA                              # noqa: E402
from word_stats import HD_FROM, read_jsonl                   # noqa: E402

KEEP = ("id", "origin_media_id", "date", "duration", "period", "session", "speaker_id", "speaker", "party", "faction")

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
args = ap.parse_args()
k, n = map(int, args.shard.split("/"))
here = Path(__file__).resolve().parent
vocab_csv = DATA / "words" / "vocab.csv"
vocab = {r["word"].lower() for r in csv.DictReader(open(vocab_csv, encoding="utf-8"))}
done = {r["id"] for r in read_jsonl(OUT / "clips_index.jsonl")}

out = here / f"catalog_{k}of{n}.jsonl.gz"
kept = 0
with gzip.open(out, "wt", encoding="utf-8") as f:
    for e in read_jsonl(DATA / "catalog.jsonl"):
        if e["date"] < HD_FROM or e["id"] in done or shard_of(e["speaker_id"], n) != k:
            continue
        sents = [s for s in e["sentences"] if any(w.lower() in vocab for w in WORD.findall(s[2]))]
        if sents:
            f.write(json.dumps({**{x: e.get(x) for x in KEEP}, "sentences": sents}, ensure_ascii=False) + "\n")
            kept += 1
shutil.copy(vocab_csv, here / "vocab.csv")
print(f"{kept} speeches -> {out} ({out.stat().st_size / 1e6:.1f} MB), vocab.csv copied")
