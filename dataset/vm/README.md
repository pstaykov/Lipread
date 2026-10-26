# Mining on a second machine (Ubuntu, CPU only)

Speakers are split by `md5(speaker_id) % 2`: the main PC mines shard 0, this machine shard 1
(`mine_clips.py --shard k/2`, each up to 400 clips per word, 800 together). Alignment runs on the CPU.

```
git clone -b Jufo-2026/27 https://github.com/pstaykov/Lipread.git && cd Lipread
bash dataset/vm/setup.sh     # once: uv + Python 3.11 env (.venv-mine), catalog + vocab -> bundestag_data/
bash dataset/vm/run.sh       # start / resume in the background (WORKERS=<n>, default: all cores)
bash dataset/vm/status.sh    # progress
bash dataset/vm/stop.sh      # stop (resumable)
bash dataset/vm/pack.sh      # -> clips_vm.tar for the main PC
```

Back on the main PC (miner stopped): `tar xf clips_vm.tar -C <dir>` and
`python dataset/vm/merge_clips.py <dir>/clips`.

`catalog_1of2.jsonl.gz` is made on the main PC by `python dataset/vm/export_shard.py --shard 1/2`
(HD speeches of shard 1 not mined yet, only sentences with vocab words); `vocab.csv` is a copy of
`bundestag_data/words/vocab.csv`.
