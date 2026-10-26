"""Merge clips mined on another machine into bundestag_data/clips (run while the local miner is stopped).

    tar xf clips_vm.tar -C some/dir          # -> some/dir/clips/...
    python dataset/vm/merge_clips.py some/dir/clips

Copies clip files and appends their clips.jsonl / clips_index.jsonl lines; clips or speeches already
present locally are skipped. Caps are re-applied later, when the train/val/test split is made.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mine_clips import OUT                                   # noqa: E402
from word_stats import read_jsonl                            # noqa: E402

src = Path(sys.argv[1])
have_clips = {c["path"] for c in read_jsonl(OUT / "clips.jsonl")}
have_ids = {r["id"] for r in read_jsonl(OUT / "clips_index.jsonl")}
new_clips = [c for c in read_jsonl(src / "clips.jsonl") if c["path"] not in have_clips]
new_ids = [r for r in read_jsonl(src / "clips_index.jsonl") if r["id"] not in have_ids]

with open(OUT / "clips.jsonl", "a", encoding="utf-8") as f:
    for c in new_clips:
        dst = OUT / c["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".mp4", ".npz"):
            shutil.copy2(src / (c["path"] + ext), dst.with_suffix(ext))
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
with open(OUT / "clips_index.jsonl", "a", encoding="utf-8") as f:
    for r in new_ids:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"merged {len(new_clips)} clips, {len(new_ids)} speeches")
