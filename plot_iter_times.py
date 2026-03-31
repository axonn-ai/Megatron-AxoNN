import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

TB_DIR = "tensorboard"

RUNS = [
    ("50708058_5B_fsdp_dense_stock",       "dense\nstock NCCL"),
    ("50708114_5B_fsdp_dense_ncclx_ch64",  "dense\nncclx ch64"),
    ("50708072_5B_fsdp_sp0.99_dense_rs",   "pruned\ndense RS"),
    ("50708070_5B_fsdp_sp0.99_ch8",        "sparse RS\nch8"),
    ("50708059_5B_fsdp_sp0.99_ch16",       "sparse RS\nch16"),
    ("50708063_5B_fsdp_sp0.99_ch32",       "sparse RS\nch32"),
    ("50708077_5B_fsdp_sp0.99_sparse_rs_ch64", "sparse RS\nch64"),
]

medians, p25, p75, labels = [], [], [], []
for dirname, label in RUNS:
    path = os.path.join(TB_DIR, dirname)
    ea = EventAccumulator(path)
    ea.Reload()
    vals = np.array([e.value for e in ea.Scalars("iteration-time") if e.step > 1])
    medians.append(np.median(vals))
    p25.append(np.percentile(vals, 25))
    p75.append(np.percentile(vals, 75))
    labels.append(label)
    print(f"{label.replace(chr(10),' '):<28}  n={len(vals)}  "
          f"median={np.median(vals):.3f}s  "
          f"p25={np.percentile(vals,25):.3f}  p75={np.percentile(vals,75):.3f}")

medians = np.array(medians)
err_lo  = medians - np.array(p25)
err_hi  = np.array(p75) - medians

# ── colour coding ──────────────────────────────────────────────────────────
COLORS = [
    "#4C72B0",  # dense stock       (blue)
    "#4C72B0",  # dense ncclx       (blue, same family)
    "#DD8452",  # pruned dense RS   (orange — ablation)
    "#55A868",  # sparse RS ch8     (green sweep)
    "#55A868",  # sparse RS ch16
    "#55A868",  # sparse RS ch32
    "#55A868",  # sparse RS ch64
]
ALPHAS = [1.0, 0.6, 1.0, 0.5, 0.65, 0.8, 1.0]

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(labels))
bars = ax.bar(x, medians, color=COLORS,
              alpha=1.0,  # set per-bar below
              width=0.6, zorder=3,
              yerr=[err_lo, err_hi],
              error_kw=dict(elinewidth=1.5, capsize=5, capthick=1.5,
                            ecolor="black", zorder=4))

# apply per-bar alpha manually
for bar, alpha in zip(bars, ALPHAS):
    bar.set_alpha(alpha)

# annotate median value above each bar
for i, (m, hi) in enumerate(zip(medians, err_hi)):
    ax.text(i, m + hi + 0.05, f"{m:.2f}s",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Iteration time (s)", fontsize=11)
ax.set_title("5B FSDP — median iteration time  (error bars = IQR, step 1 excluded)",
             fontsize=11)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2, zorder=0)
ax.set_ylim(0, max(medians) * 1.25)

# legend
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#4C72B0", label="Dense baseline"),
    Patch(facecolor="#DD8452", label="Pruned, dense RS (ablation)"),
    Patch(facecolor="#55A868", label="Sparse RS (channel sweep)"),
]
ax.legend(handles=legend_els, fontsize=9, loc="upper right")

plt.tight_layout()
out = "iter_times_5B.png"
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
