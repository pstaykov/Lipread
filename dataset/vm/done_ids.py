"""Print the speeches this machine has finished for the given speaker shard, for skip_ids.txt on the other machine.

    python dataset/vm/done_ids.py 7,9/10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mine_clips import OUT, shard_of                         # noqa: E402
from mine_bundestag import DATA                              # noqa: E402
from word_stats import read_jsonl                            # noqa: E402

ks, n = sys.argv[1].split("/")
ks, n = {int(k) for k in ks.split(",")}, int(n)
spk = {e["id"]: e["speaker_id"] for e in read_jsonl(DATA / "catalog.jsonl")}
ids = sorted({r["id"] for r in read_jsonl(OUT / "clips_index.jsonl") if r["id"] in spk and shard_of(spk[r["id"]], n) in ks})
print(" ".join(ids))
print(f"{len(ids)} speeches", file=sys.stderr)
