#!/usr/bin/env python3
"""Plot per-iteration sparsity with min/max error bars for pruner logs."""
import argparse
import matplotlib.pyplot as plt
from parse_pruner import parse


def series(per_iter):
    iters = sorted(per_iter)
    mean, lo, hi = [], [], []
    for it in iters:
        sps = [r[1] * 100.0 for r in per_iter[it]]
        m = sum(sps) / len(sps)
        mean.append(m)
        lo.append(m - min(sps))
        hi.append(max(sps) - m)
    return iters, mean, [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="sparsity.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    label_map = {"out.asd": "With EF", "out_noea.asd": "Without EF"}
    for p in args.files:
        iters, mean, err = series(parse(p))
        ax.errorbar(iters, mean, yerr=err, marker="o", markersize=3,
                    linewidth=1.2, capsize=3, capthick=1, elinewidth=1,
                    label=label_map.get(p, p))
    ax.set_xlabel("iteration")
    ax.set_ylabel("sparsity (%)")
    ax.set_title("Per-iteration gradient sparsity (min/max over 16 entries)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
