# BTLips: the Bundestag word dataset

GLips-style German word-level lipreading dataset from Bundestag Mediathek HD video: **500 words × 500 clips,
400 / 50 / 50 clips per word, globally speaker-disjoint** (each speaker in exactly one of train / val / test across all words). Layout and file formats: [`BTLips/README.txt`](../BTLips/README.txt)
(gitignored, with `BTLips_facemesh/`). Loader: `model/data/clips.py`.

| | Speakers | Clips per word | Speakers per word (median) |
|---|---|---|---|
| train | 1,039 | 400 | 242 |
| val | 163 | 50 | 36 |
| test | 170 | 50 | 36 |

250,000 clips from 18,826 speeches, 06.09.2016 – 25.09.2026 (periods 18–21). Mouth width 36–87 px in the source (median 51; GLips ≈ 45).

## Pipeline (`dataset/`)
The intermediate data (`bundestag_data/`) was deleted after the build; the scripts document how BTLips was made.

| Step | Script | What |
|---|---|---|
| 1 | `mine_bundestag.py catalog` | Speech catalog from the Open Parliament TV API: one main speaker, transcript sentences with times. |
| 2 | `mine_bundestag.py mine`, `select_targets.py` | Pilot mine at 576p/720p: per-frame lectern gate (FaceMesh geometry, one face, wall strips against the speech's median ⇒ cutaways rejected). Gives each word's visible rate. |
| 3 | `word_stats.py`, `select_vocab.py` | Word counts in the HD pool (1080p from Sept 2016), visemes (Weiss & Aschenberner) from the MFA dictionary. 500 content words, no stopwords, ≥ 100 speakers: 94 homophene-group words, 94 near-pair words, 312 controls (`wordlist.csv`). |
| 4 | `mine_clips.py` | Per word occurrence: forced alignment (MMS_FA) on the Mediathek audio, 29 frames around the word centre decoded from the HD video, same lectern gate; kept if ≥ 27/29 frames pass and the mouth is ≥ 36 px. 256 px face crop (2 × face width). Caps 600 per word, 25 per word and speaker, 5 per word and speech; `--topup` refills words. |
| 5 | `scan_clips.py`, `clean_clips.py` | Each clip re-aligned on its own audio, landmark checks. Rejected: camera cut inside the clip, static or hidden mouth, profile face, word > 0.2 s off centre, mouth leaving the crop (18,106 of 331,651). Problems only in the first / last 3 frames are repaired with the nearest good frame. |
| 6 | `split_clips.py` | Speaker-disjoint split. A speaker hash alone left some words with 15 test clips, so speakers are moved between splits (hill climbing) until every word has ≥ 400 / 50 / 50; then per word and split the clips are taken round-robin over speakers. |
| 7 | `build_btlips.py` | Writes `BTLips/` (mp4, wav, txt, lists) and `BTLips_facemesh/`. |

Notes:
- Only Bundestag video is used (permission). The full video is never stored; only the 29-frame windows are decoded.
- Period 21 is 50 fps interlaced and is deinterlaced (`bwdif`) to 25 fps; 18–20 are progressive 25 fps.
- Mining ran on two machines, speakers split by `md5(speaker_id) % 10`.
