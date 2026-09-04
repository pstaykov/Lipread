# GLips Lip-Reading — Training Runs & Metrics

Reference for every trainer in `lipreading/`: what it trains, why it exists, and
where its metrics CSV lives. All runs are sequenced by `run_all.py` (resumable;
skips a run once its `final_model.pth` exists).

Every run writes a `metrics.csv` into its own checkpoint dir, all with the **same
schema** (one row per epoch):

```
epoch, train_loss, train_acc, val_loss, val_top1, val_top5, val_f1, lr
```

Alongside each CSV the trainer saves `best_model.pth` (best val top-1, EMA weights
where EMA is used), `best_top5_model.pth`, `final_model.pth` (last epoch), and
`checkpoint_latest.pth` (exact-resume state).

> **Val curves ≠ test numbers.** These CSVs are the per-epoch *validation* curves.
> Verified **test-split** results live in `analysis/ablation.py`.

---

## 1. 500-class backbones (transfer sources)

| Script | Trains | Why | Checkpoint dir | CSV |
|---|---|---|---|---|
| `Transformer_based/train.py` (+ `train_500_finetune.py`) | GLipsNet (3D-conv+ResNet18 → MS-TCN stem → Transformer → attentive pool) on all **500** GLips classes, stock split, plain recipe (no EMA/Mixup). | The Transformer backbone that the 15-class transfer runs warm-start from; plain recipe makes it directly comparable to the MS-TCN 500 baseline. | `Transformer_based/checkpoints_500/` | `Transformer_based/checkpoints_500/metrics.csv` |
| `mstcn_baseline/train.py` (+ `train_500_finetune.py`) | MS-TCN baseline (TCNLipNet, width 384) on all **500** classes, identical frontend/data/recipe. | MS-TCN counterpart of the above; the only difference is the temporal back-end. Transfer source for the MS-TCN 15-class run. | `mstcn_baseline/checkpoints_500/` | `mstcn_baseline/checkpoints_500/metrics.csv` |

## 2. 15-class from-scratch (our full-augmentation recipe)

| Script | Trains | Why | Checkpoint dir | CSV |
|---|---|---|---|---|
| `Transformer_based/train_15.py` | GLipsNet from scratch on Ameer's **15** classes, mouth-ROI, stock split. Full regularization: attentive pool, Mixup+CutMix, random grayscale + random-erasing, temporal speed-perturb, weight-EMA, early stopping. | Our best from-scratch Transformer number on the 15-word benchmark. **Currently being cleanly retrained** (old dir → `checkpoints_15_bak_*`). | `Transformer_based/checkpoints_15/` | `Transformer_based/checkpoints_15/metrics.csv` |
| `mstcn_baseline/train_15.py` | TCNLipNet (width 384) from scratch on the 15 classes — mirrors `train_15.py` exactly. | Isolates the temporal back-end (MS-TCN vs Transformer) under an identical recipe. | `mstcn_baseline/checkpoints_15/` | `mstcn_baseline/checkpoints_15/metrics.csv` |

## 3. Ameer-matched 15-word (head-to-head with Ameer et al.)

| Script | Trains | Why | Checkpoint dir | CSV |
|---|---|---|---|---|
| `Transformer_based/train_15_ameer.py` | GLipsNet under Ameer's **exact** data regime: uncropped full face, 128×128, 16 frames, min-max norm, flip-only aug, same-class interpolation, plain recipe. | Only the data/aug matches Ameer (NASNetMobile, 0.484 acc / 0.485 F1) — so the comparison isolates *architecture* under an identical data regime. | `Transformer_based/checkpoints_15_ameer/` | `Transformer_based/checkpoints_15_ameer/metrics.csv` |
| `mstcn_baseline/train_15_ameer.py` | TCNLipNet (width 384) under the same Ameer-matched regime. | MS-TCN counterpart of the Ameer-matched comparison. | `mstcn_baseline/checkpoints_15_ameer/` | `mstcn_baseline/checkpoints_15_ameer/metrics.csv` |

## 4. In-domain transfer (500-class backbone → 15 classes)

| Script | Trains | Why | Checkpoint dir | CSV |
|---|---|---|---|---|
| `Transformer_based/train_15_transfer.py` | GLipsNet fine-tuned on the 15 classes, warm-started from the 500-class backbone (whole feature extractor transfers; only the 15-way head is fresh). Recipe held identical to `train_15.py`. | Measures GLips-500 → GLips-15 transfer vs from-scratch, holding everything else fixed. Run **after** `train.py`. | `Transformer_based/checkpoints_15_transfer/` | `Transformer_based/checkpoints_15_transfer/metrics.csv` |
| `mstcn_baseline/train_15_transfer.py` | TCNLipNet fine-tuned on 15 classes from the 500-class MS-TCN backbone; same recipe. | Same transfer question for the MS-TCN back-end; run **after** `mstcn_baseline/train.py`. | `mstcn_baseline/checkpoints_15_transfer/` | `mstcn_baseline/checkpoints_15_transfer/metrics.csv` |

## 5. GLipsNet ablations (all vs `train_15.py`)

| Script | Trains | Why | Checkpoint dir | CSV |
|---|---|---|---|---|
| `Transformer_based/train_15_meanpool.py` | GLipsNet with plain temporal **mean** pooling instead of attentive pool; everything else identical. | Isolates the contribution of the attentive-pool head. | `Transformer_based/checkpoints_15_meanpool/` | `Transformer_based/checkpoints_15_meanpool/metrics.csv` |
| `Transformer_based/train_15_nostem.py` | GLipsNet **without** the multi-scale temporal conv stem (`use_stem=False`); per-frame features go straight into the Transformer. | Isolates how much the local multi-scale temporal convs add on top of global attention. | `Transformer_based/checkpoints_15_nostem/` | `Transformer_based/checkpoints_15_nostem/metrics.csv` |
| `Transformer_based/train_15_noroi.py` | GLipsNet on **uncropped** full-face frames (no mouth-ROI crop), same architecture/recipe. | Measures how much accuracy comes from the mouth-ROI preprocessing vs the architecture itself. | `Transformer_based/checkpoints_15_noroi/` | `Transformer_based/checkpoints_15_noroi/metrics.csv` |

---

## Non-canonical dirs

Stale backups (`*__pre_ameer/`, `_my_glips500_bak/`, `_my_clean_partial_bak/`)
have been deleted — only the current `run_all` outputs remain. One temporary dir
may still be present:

- `checkpoints_15_bak_*/` — rollback of the pre-retrain TF-15 run, kept only while
  the clean retrain is in flight; safe to delete once `checkpoints_15/` finishes.

## Support files (not standalone trainers)

- `train_loop.py` — shared `run_training`/`make_loaders`/`EMA`/`mixup_cutmix`; writes every `metrics.csv`.
- `Transformer_based/train_utils.py`, `Transformer_based/model.py`, `dataset*.py` — model + data plumbing.

---

## Load every CSV as a pandas DataFrame

```python
from pathlib import Path
import pandas as pd

LIP = Path(__file__).resolve().parent  # or Path("lipreading")

# --- Canonical runs only (skips *_bak_* and *__pre_ameer backups) --------------
RUNS = {
    # 500-class backbones
    "tf_500":            "Transformer_based/checkpoints_500/metrics.csv",
    "mstcn_500":         "mstcn_baseline/checkpoints_500/metrics.csv",
    # 15-class from scratch
    "tf_15":             "Transformer_based/checkpoints_15/metrics.csv",
    "mstcn_15":          "mstcn_baseline/checkpoints_15/metrics.csv",
    # Ameer-matched
    "tf_15_ameer":       "Transformer_based/checkpoints_15_ameer/metrics.csv",
    "mstcn_15_ameer":    "mstcn_baseline/checkpoints_15_ameer/metrics.csv",
    # transfer (500 -> 15)
    "tf_15_transfer":    "Transformer_based/checkpoints_15_transfer/metrics.csv",
    "mstcn_15_transfer": "mstcn_baseline/checkpoints_15_transfer/metrics.csv",
    # GLipsNet ablations
    "tf_15_meanpool":    "Transformer_based/checkpoints_15_meanpool/metrics.csv",
    "tf_15_nostem":      "Transformer_based/checkpoints_15_nostem/metrics.csv",
    "tf_15_noroi":       "Transformer_based/checkpoints_15_noroi/metrics.csv",
}

dfs = {name: pd.read_csv(LIP / rel) for name, rel in RUNS.items()
       if (LIP / rel).exists()}

# Best epoch (by val_top1) per run — a quick leaderboard
summary = pd.DataFrame({
    name: df.loc[df["val_top1"].idxmax(), ["epoch", "val_top1", "val_top5", "val_f1"]]
    for name, df in dfs.items()
}).T.sort_values("val_top1", ascending=False)
print(summary)

# --- OR: auto-discover EVERY metrics.csv, backups included ----------------------
all_csvs = {
    p.parent.relative_to(LIP).as_posix(): pd.read_csv(p)
    for p in LIP.glob("**/checkpoints*/metrics.csv")
}
# access e.g. all_csvs["Transformer_based/checkpoints_15"]
```

Both blocks skip missing files, so a run that hasn't finished yet (e.g. the
in-progress `checkpoints_15` retrain) simply won't appear until its CSV exists.