# Mining on a second machine (Windows, CPU only)

Speakers are split by `md5(speaker_id) % 10` in proportion to speed: this machine mines remainders 1,3,5
(`--shard 1,3,5/10`, up to 240 clips per word), the main PC 0,2,4,6,7,8,9 (up to 560); 800 together.
Until 2026-10-27 the split was `% 2` (odd = this machine); speeches of remainders 7,9 it had already
mined are listed with `python dataset/vm/done_ids.py 7,9/10` and go into the PC's
`bundestag_data/clips/skip_ids.txt`. Alignment runs on the CPU.
Needs ~65-70 GB free disk (clips ~32 GB + the miner's 30 GB safety margin).

In PowerShell (Git: `winget install Git.Git`, then open a new window):
```
git clone -b Jufo-2026/27 https://github.com/pstaykov/Lipread.git
cd Lipread
powershell -ExecutionPolicy Bypass -File dataset\vm\setup.ps1     # once: uv, Python 3.11 env, data, self-check
powershell -ExecutionPolicy Bypass -File dataset\vm\run.ps1       # start / resume (-Workers 14; default: cores - 2)
powershell -ExecutionPolicy Bypass -File dataset\vm\status.ps1    # progress
powershell -ExecutionPolicy Bypass -File dataset\vm\stop.ps1      # stop (resumable)
powershell -ExecutionPolicy Bypass -File dataset\vm\pack.ps1      # -> clips_vm.tar for the main PC
```
The miner keeps running when the Remote Desktop window is closed, but not after signing out.

Back on the main PC (miner stopped): `tar -xf clips_vm.tar -C <dir>` and
`python dataset/vm/merge_clips.py <dir>/clips`.

`catalog_1of2.jsonl.gz` is made on the main PC by `python dataset/vm/export_shard.py --shard 1/2`
(HD speeches of shard 1 not mined yet, only sentences with vocab words); `vocab.csv` is a copy of
`bundestag_data/words/vocab.csv`.
