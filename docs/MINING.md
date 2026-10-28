# Mining the Bundestag dataset

Script: `dataset/mine_bundestag.py` (resumable; output in `bundestag_data/`, gitignored).

```
python dataset/mine_bundestag.py catalog   # API only
python dataset/mine_bundestag.py mine      # video
```

## Source
- Only Deutscher Bundestag (`parliament=DE`) via the [Open Parliament TV API](https://de.openparliament.tv/api), electoral periods 19–21 (2017+; older video is too low-res for lips).
- One media item = one speech. Gives the main speaker's Wikidata ID, party/faction, the MP4 URL, and transcript sentences with timestamps (`aligned=true`).

## Catalog (which speeches)
- Enumerate sessions `DE-0PPSSSS`, page through `search/media?sessionID=…`.
- Keep: exactly one `main-speaker`, 60 s–30 min, ≥ 30 s of main-speaker sentences.
- Keep only main-speaker `speech` sentences; drop president interjections and `(Beifall …)` comments.

## Mining (per speech)
- One ffmpeg pass streams the MP4 from the CDN; the full video is never stored.
  - Resample to 25 fps.
  - Fixed centre crop (ROI) `x 0.30–0.70, y 0.00–0.50` of the frame → `roi.mp4` (x264 CRF 20). The camera is static on the lectern shot, so the speaker's face always falls inside it.
  - Audio → `audio.flac` (16 kHz mono).
  - Downscaled copy of the ROI piped to MediaPipe FaceMesh (478 landmarks, refined lips/iris).
- Talking ≠ face on screen (cutaways to the hall, the president, the audience), so each frame is gated:
  - **Geometry:** face centre `x 0.38–0.62, y 0.08–0.40`, width `0.06–0.125` (full-frame units; measured on P19/P21). Close-ups of the president or audience are larger/lower → rejected.
  - **Scene:** wall strips left/right of the head are compared to the speech's median strip (static camera ⇒ identical on the lectern shot). Distance < 12; lectern ≈ 1–8, other shots ≈ 40–50.
  - Exactly one face may pass; two faces → frame dropped.
  - Deliberately rejected: the occasional *wide* lectern shot (face ≈ 4–5 % of frame width, mouth ≈ 20 px, MPs seated behind visible in the crop). Too little lip detail; ~2 of 15 speeches in a P21 test.
- Clean-up: bridge ≤ 2-frame detector dropouts, drop runs < 0.5 s, drop runs where the face centre drifts (std > 0.05 → camera moved).
- `usable` = valid face frames ∩ main-speaker sentence time. Speech kept only if ≥ 20 s and ≥ 25 % of its speech time is usable; otherwise the files are deleted.

## Selection / budget
- First ~240 speeches: round-robin over speakers (everyone's 1st speech before anyone's 2nd).
- Since 2026-10-25: `--targets bundestag_data/words/targets.csv` (from `dataset/select_targets.py`).
  - Speeches are ordered by target-word occurrences per minute, with rare targets weighted 1/√count.
  - Target set: 179 words (96 homophene-group members, 43 near-pair words, 40 controls).
  - Projected at 250 GB: min 143 / median ~300 visible instances per target word, vs 50 / 137 for round-robin.
- Max 10 speeches per speaker in both modes (speaker-disjoint split).
- Stops at 250 GB of kept data or < 60 GB free disk (~8 MB per minute of speech).

## Output per speech (`bundestag_data/speeches/<media-id>/`)
- `roi.mp4`: centre crop at source resolution (never downscaled; 512×360 for 720p, ~410×288 for P19), 25 fps, frame i = time i/25 s of the original video. Size stored as `roi_px` in `meta.json` and `facemesh.npz`.
- `audio.flac`
- `facemesh.npz`:
  - `landmarks` (K×478×3, ROI-normalized) and `landmark_frame` (K)
  - `bbox` (cx, cy, w, h, full-frame normalized)
  - `scene_dist`, `valid`, `speech` (per frame)
- `meta.json`: speaker, party, faction, session, date, source URLs, sentences, gate params, stats.
- `bundestag_data/index.jsonl`: one line per processed speech (kept / rejected / error).

## Verified
- Checked by eye on a 2019 and a 2026 speech: hall, president and audience cutaways are rejected; 48/48 random accepted frames show the speaker at the lectern.

## Vocabulary statistics (`dataset/word_stats.py`)
- Counts every main-speaker word in the catalog: occurrences, distinct speakers, and `vis_*` (only sentences in mined speeches that are ≥ 80 % valid face frames).
- Drops stopwords (stopwords-iso de) and words < 4 letters.
- Phonemes from the MFA German dictionary (first variant, ß→ss fallback) → 15 Weiss & Aschenberner visemes (diphthongs/affricates = 2 visemes, as in the paper).
- Words with ≥ 100 occurrences:
  - `homophenes.csv`: same viseme string, different phonemes; `inflection_only` flags e.g. geehrte/geehrter.
  - `near_pairs.csv`: one viseme substituted.
- Output in `bundestag_data/words/`. Rerun after catalog/mining progress (~15 min).

## Result (2026-10-26)
- Stopped at the 250 GB budget: 7,162 speeches kept, 1,169 speakers, ~560 h video, ~419 h usable.
- 1,163 rejected (wide shot / cutaways); 191 errors (184 missing on the CDN or no video stream, 7 retryable but over budget).

## Final vocabulary (`dataset/select_vocab.py` → `words/vocab.csv`)
- "Relevant" word:
  - no stopword, ≥ 4 letters;
  - majority POS in context is noun / proper noun / full verb / adjective (spaCy `de_core_news_sm`, 30 sentences per word);
  - not on a short blocklist of fillers (genau, eigentlich, …) and address formulas (Präsidentin, Kolleginnen, geehrte, Damen, …).
- ≥ 100 visible instances from ≥ 50 speakers.
- Homophene groups need ≥ 2 different word families: same lemma, shared prefix or substring counts as one family (Hilfe/hilft, alte/Alter). The same contrast in another inflection is kept once.
- Fill words, after the homophene groups and up to 100 near pairs: the most visible relevant words, one form per word family, up to 500 in total.
- Result: 500 words. 48 homophene groups (106 words), 100 near pairs (90 words), 304 fill words. Every word has ≥ 100 visible instances; ~307k visible instances in total, before alignment.

## HD pool and video quality (2026-10-26)
- The OPTV video links are small versions (1024×576 / 1280×720, mouth ≈ 27 / 34 px). The Mediathek share data (`webtv.bundestag.de/player/macros/_x_s-144277506/shareData.json?contentId=<originMediaID>`) also lists 1920×1080 downloads on the same Bundestag CDN.
- 1080p exists from the 2016 summer break on: the last SD speech is from 2016-07-08, the first HD one from 2016-09-06. Older video is 720×400 (mouth ≈ 19 px) and is excluded (`HD_FROM` in `word_stats.py`).
- Periods 18 (from Sept 2016), 19 and 20: progressive 25 fps. Period 21: 50 fps with interlaced fields, so `bwdif` deinterlacing is applied.
- URL labels are unreliable (`720_400` files are often 1024×576), so the resolution is always probed from the file.
- HD pool: 50,474 speeches, 31.1 M words. Mouth in HD ≈ 46–61 px (GLips clips ≈ 45 px).

## Vocabulary for the HD pool
- `word_stats.py` counts only the HD pool; `mined_count` = occurrences inside the mined speeches.
- `est` = transcript count × the word's visible rate in mined speeches (shrunk towards the global 0.72) × speech keep rate 0.86 × assumed alignment keep 0.85.
- `select_vocab.py` (default `--min-est 500`, ≥ 100 speakers): 500 words. 41 homophene groups (94 words), 100 near pairs, 312 fill words.
- Homophones (identical phonemes) barely exist among frequent German content words (only höre/höher), so homophenes are the category used.

## Clip mining (`dataset/mine_clips.py`)
- Speeches ordered rare-words-first; only sentences with still-needed words are processed.
- Two stages:
  - The main process prepares speeches in 3 threads (links, MP3, alignment). It holds the only GPU model, in fp16.
  - CPU worker processes (`--workers`, no torch) decode, gate and write. Decode segments are ≤ 10 s, so memory stays bounded.
- Stop with `taskkill /PID <bundestag_data/clips/miner.pid> /T /F`, which kills the workers too. Interrupted speeches are retried on the next start.
- Candidates of the same word < 0.2 s apart are merged (OPTV sometimes repeats a sentence).
- Audio: Mediathek MP3, forced alignment with torchaudio MMS_FA (GPU).
  - Each sentence is padded by ±1.5 s with `*` tokens (OPTV sentence times are ~1 s early).
  - Word kept if mean frame probability ≥ 0.5.
- Video: only the 29-frame window around each word is decoded from the HD file (seek), using the same centre ROI and lectern gate as above (geometry, one face, scene strips against the speech's median).
- Clip kept if ≥ 27/29 frames pass (gaps ≤ 2), the face doesn't drift, and the mouth is ≥ 36 px in source pixels.
- Caps: 800 clips per word, 25 per word per speaker, 5 per word per speech.
- Output `bundestag_data/clips/<word>/<speech>_<frame>`:
  - `.mp4`: 256×256 face crop (side = 2 × face width, GLips framing), 25 fps, 29 frames.
  - `.npz`: `landmarks` (29×478×3, crop-normalized, NaN where missing), `audio` (16 kHz int16), `crop_box`, `roi_px`.
- `clips.jsonl`: one line per clip (word, speaker, times, score, mouth_px, source). `clips_index.jsonl`: one line per speech.
- Test on 4 speeches (periods 18–21): 106 clips, mouth 37–85 px (median 50), all checked by eye.

## Result of the clip mining (2026-10-27)
- Two machines, speakers split by `md5(speaker_id) % 10` (`--shard`, see `dataset/vm/`): PC 0,2,4,6,7,8,9, a CPU-only Windows VM 1,3,5. Total cap 600 per word, split 420 / 180.
- Merged with `dataset/vm/merge_clips.py`: 330,271 clips.
- Top-up (`--topup 500`): all speeches again for the 9 words still below 500, skipping instances already mined.
- Bug found there: the scene reference needs >= 2 x 29 face frames, so a speech with a single needed occurrence could never pass. Fixed: such speeches sample extra frames across the speech for the reference (`REF_SAMPLES`).
- Final: 330,649 clips, all 500 words >= 500 (minimum exactly 500 for the 9 topped-up words).

## Cleanup (2026-10-28)
- `dataset/scan_clips.py` -> `scan.csv`: re-aligns the word on each clip's own audio (offset from the clip centre) and computes lip opening, yaw and landmark jumps.
- `dataset/clean_clips.py` (`--dry` first) rejects clips with a camera cut inside (frames 3..25), a static mouth, a profile face, the word more than 0.2 s off centre, or the mouth leaving the crop. Cuts, missing faces or a mouth outside the crop in the 3 edge frames are repaired with the nearest good frame. Rejects go to `clips/_rejected/` and `rejected.jsonl`; the miner never mines them again.
- 18,106 rejected (outside 8,235, static 5,298, offword 3,965, profile 607, cut 1), 9,135 repaired.
- 16 words fell below 500 -> `--topup 530 --words ...` -> scanned and cleaned again.
- Final: 313,545 clips, 1,372 speakers, all 500 words >= 503.

## Split (2026-10-28)
- `dataset/split_clips.py`: speaker-disjoint, 400 train / 50 val / 50 test clips per word (250,000 clips, `clips/dataset.jsonl` with a `split` field; speaker -> split in `clips/split.json`).
- The plain speaker hash (`model/data/clips.py:split_of`, 80/10/10 of the speakers) left some words with only 15 test clips, since a few speakers say a word often. Speakers are moved between the splits until every word reaches 400/50/50: 1,039 / 163 / 170 speakers.
- Per word and split the clips are taken round-robin over the speakers (best alignment score first): median 242 / 36 / 36 speakers per word, no test speaker above 4 % of the test set.

## Old mine
- After the switch to HD, `roi.mp4` and `audio.flac` of the 576p/720p mine were deleted, along with the per-frame landmarks. `meta.json` and the `facemesh.npz` masks remain. Sample speech DE-0210004073 is kept whole for `lipreading/landmarks/smoke_test.py`.

## Not yet done
- Banner OCR cross-check.
