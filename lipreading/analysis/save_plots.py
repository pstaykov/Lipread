"""Unified plotting for GLipsNet vs. MS-TCN comparisons: 15-class, 500-class, transfer, and summary bar, all in one shared style.

Usage:
    python plot_results.py

Edit CSV_PATHS below to point at your real metrics.csv files.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV_PATHS = {  # edit these to match your machine
    "mstcn_15":            r"C:\Users\pstay\code\BWKI\lipreading\mstcn_baseline\checkpoints_15\metrics.csv",
    "transformer_15":      r"C:\Users\pstay\code\BWKI\lipreading\Transformer_based\checkpoints_15\metrics.csv",
    "mstcn_500":           r"C:\Users\pstay\code\BWKI\lipreading\mstcn_baseline\checkpoints_500\metrics.csv",
    "transformer_500":     r"C:\Users\pstay\code\BWKI\lipreading\Transformer_based\checkpoints_500\metrics.csv",
    "mstcn_transfer":      r"C:\Users\pstay\code\BWKI\lipreading\mstcn_baseline\checkpoints_15_transfer\metrics.csv",
    "transformer_transfer": r"C:\Users\pstay\code\BWKI\lipreading\Transformer_based\checkpoints_15_transfer\metrics.csv",
}

OUTDIR = "figures"  # where PNGs get saved

sns.set_theme(style="whitegrid")

MSTCN_COLOR = "C0"
TRANSFORMER_COLOR = "C1"
MSTCN_LABEL = "MS-TCN"
TRANSFORMER_LABEL = "GNet"

ANNOT_KW = dict(
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    color="black",
    zorder=6,
)

FIGSIZE = (20, 6)
TITLE_PAD = 12


def _annotate_max(ax, df, col, color, offset, fmt="max {v:.1f}% @ ep{e}"):
    """Mark the epoch of peak value with a black dot + bordered label box."""
    idx = df[col].idxmax()
    epoch, value = df["epoch"][idx], df[col][idx] * 100
    ax.scatter([epoch], [value], color="black", zorder=6, s=55)
    ax.annotate(
        fmt.format(v=value, e=int(epoch)),
        (epoch, value),
        xytext=offset,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            alpha=0.95,
        ),
        **ANNOT_KW,
    )
    return epoch, value


def _align_epochs(df_mstcn, df_transformer, title_prefix):
    """Trim both runs to the epoch budget they both completed, so a head-to-head panel never compares mismatched run lengths."""
    last = int(min(df_mstcn["epoch"].max(), df_transformer["epoch"].max()))
    if df_mstcn["epoch"].max() != df_transformer["epoch"].max():
        print(f"  ! {title_prefix}: unequal run lengths "
              f"(MS-TCN {int(df_mstcn['epoch'].max())} ep, "
              f"GLipsNet {int(df_transformer['epoch'].max())} ep) -- "
              f"both trimmed to {last} for a like-for-like curve.")
    return (df_mstcn[df_mstcn["epoch"] <= last].reset_index(drop=True),
            df_transformer[df_transformer["epoch"] <= last].reset_index(drop=True),
            last)


def plot_run(
    df_mstcn: pd.DataFrame,
    df_transformer: pd.DataFrame,
    *,
    title_prefix: str,
    xlim: int = None,
    xtick_step: int = None,
    save_path: str,
):
    """Render the standard two-panel (val/train) comparison figure for one run; returns the exact peak values used."""
    df_mstcn, df_transformer, last_epoch = _align_epochs(
        df_mstcn, df_transformer, title_prefix)
    if xlim is None:
        xlim = last_epoch
    if xtick_step is None:
        xtick_step = max(5, round(xlim / 10 / 5) * 5 or 5)
    fig, (ax_val, ax_train) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    ax_val.plot(df_mstcn["epoch"], df_mstcn["val_top1"] * 100,
                label=MSTCN_LABEL, color=MSTCN_COLOR)
    ax_val.plot(df_transformer["epoch"], df_transformer["val_top1"] * 100,
                label=TRANSFORMER_LABEL, color=TRANSFORMER_COLOR)
    ax_val.plot(df_mstcn["epoch"], df_mstcn["val_top5"] * 100,
                label=f"{MSTCN_LABEL} (top-5)", linestyle="--", color=MSTCN_COLOR)
    ax_val.plot(df_transformer["epoch"], df_transformer["val_top5"] * 100,
                label=f"{TRANSFORMER_LABEL} (top-5)", linestyle="--", color=TRANSFORMER_COLOR)

    mstcn_val_e, mstcn_val_v = _annotate_max(ax_val, df_mstcn, "val_top1", MSTCN_COLOR, (10, -22))
    tf_val_e, tf_val_v = _annotate_max(ax_val, df_transformer, "val_top1", TRANSFORMER_COLOR, (10, 12))

    mstcn_top5_v = df_mstcn["val_top5"].max() * 100  # numeric only, no marker
    tf_top5_v = df_transformer["val_top5"].max() * 100

    ax_val.set_title(f"{title_prefix}: Validation Accuracy over Epochs", pad=TITLE_PAD)
    ax_val.set_ylabel("Accuracy (%)")
    ax_val.set_xlabel("Epoch")
    ax_val.legend(loc="lower right", framealpha=0.9)
    ax_val.grid(True)

    ax_train.plot(df_mstcn["epoch"], df_mstcn["train_acc"] * 100,
                  label=MSTCN_LABEL, color=MSTCN_COLOR)
    ax_train.plot(df_transformer["epoch"], df_transformer["train_acc"] * 100,
                  label=TRANSFORMER_LABEL, color=TRANSFORMER_COLOR)

    mstcn_tr_e, mstcn_tr_v = _annotate_max(ax_train, df_mstcn, "train_acc", MSTCN_COLOR, (10, -22))
    tf_tr_e, tf_tr_v = _annotate_max(ax_train, df_transformer, "train_acc", TRANSFORMER_COLOR, (10, 12))

    ax_train.set_title(f"{title_prefix}: Train Accuracy over Epochs", pad=TITLE_PAD)
    ax_train.set_xlabel("Epoch")
    ax_train.legend(loc="lower right", framealpha=0.9)
    ax_train.grid(True)

    xticks = list(np.arange(0, xlim + 1, xtick_step))
    if xlim - xticks[-1] >= xtick_step / 2:
        xticks.append(xlim)

    for ax in (ax_val, ax_train):
        ax.set_ylim(0, 100)
        ax.set_xlim(1, xlim)          # data starts at epoch 1; no dead margin
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(0, 101, 10))

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "mstcn_val_top1_max": mstcn_val_v, "mstcn_val_top1_epoch": int(mstcn_val_e),
        "transformer_val_top1_max": tf_val_v, "transformer_val_top1_epoch": int(tf_val_e),
        "mstcn_val_top5_max": mstcn_top5_v,
        "transformer_val_top5_max": tf_top5_v,
        "mstcn_train_acc_max": mstcn_tr_v, "mstcn_train_acc_epoch": int(mstcn_tr_e),
        "transformer_train_acc_max": tf_tr_v, "transformer_train_acc_epoch": int(tf_tr_e),
        "mstcn_train_val_gap_pts": mstcn_val_v - mstcn_tr_v,
        "transformer_train_val_gap_pts": tf_val_v - tf_tr_v,
    }


def plot_summary_bar(results: dict, save_path: str):
    """Single bar chart of best validation top-1 across all three runs."""
    plot_data = pd.DataFrame({
        "Model": ["MS-TCN", "Transformer", "MS-TCN", "Transformer", "MS-TCN", "Transformer"],
        "Configuration": [
            "15 Classes (Transfer)", "15 Classes (Transfer)",
            "15 Classes", "15 Classes",
            "500 Classes", "500 Classes",
        ],
        "Best Val Top-1": [
            results["transfer"]["mstcn_val_top1_max"] / 100,
            results["transfer"]["transformer_val_top1_max"] / 100,
            results["15"]["mstcn_val_top1_max"] / 100,
            results["15"]["transformer_val_top1_max"] / 100,
            results["500"]["mstcn_val_top1_max"] / 100,
            results["500"]["transformer_val_top1_max"] / 100,
        ],
    })

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=plot_data, x="Configuration", y="Best Val Top-1", hue="Model",
        palette={"MS-TCN": MSTCN_COLOR, "Transformer": TRANSFORMER_COLOR},
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    plt.title("Lipreading Model Comparison: Best Validation Top-1 Accuracy", pad=TITLE_PAD)
    plt.ylabel("Validation Top-1 Score")
    plt.xlabel("Configuration")
    plt.ylim(0, min(1.0, plot_data["Best Val Top-1"].max() * 1.15))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)

    df_mstcn_15 = pd.read_csv(CSV_PATHS["mstcn_15"])
    df_transformer_15 = pd.read_csv(CSV_PATHS["transformer_15"])
    df_mstcn_500 = pd.read_csv(CSV_PATHS["mstcn_500"])
    df_transformer_500 = pd.read_csv(CSV_PATHS["transformer_500"])
    df_mstcn_transfer = pd.read_csv(CSV_PATHS["mstcn_transfer"])
    df_transformer_transfer = pd.read_csv(CSV_PATHS["transformer_transfer"])

    results = {}
    results["15"] = plot_run(
        df_mstcn_15, df_transformer_15,
        title_prefix="15-Class GLips",
        save_path=f"{OUTDIR}/glips15_metrics.png",
    )
    results["500"] = plot_run(
        df_mstcn_500, df_transformer_500,
        title_prefix="500-Class GLips",
        save_path=f"{OUTDIR}/glips500_metrics.png",
    )
    results["transfer"] = plot_run(
        df_mstcn_transfer, df_transformer_transfer,
        title_prefix="GLips-500 to GLips-15 Transfer",
        save_path=f"{OUTDIR}/transfer_metrics.png",
    )
    plot_summary_bar(results, save_path=f"{OUTDIR}/summary_bar.png")

    print("\n=== Exact values for the paper (copy into tables/text) ===\n")
    for run_name, r in results.items():
        print(f"--- {run_name} ---")
        for k, v in r.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()