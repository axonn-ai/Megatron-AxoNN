#!/usr/bin/env python3
"""Parse [TritonGradientPruner] log lines and aggregate per-iteration sparsity.

Lines look like:
  [TritonGradientPruner] key=140077573341184 prev sparsity=0.988223 (nnz=18335145/1556867200)

Iteration boundaries are inferred from Megatron's "iteration N/M" log lines.
Pruner lines that appear after iteration N's log line are attributed to
iteration N+1 (they're printed during the next step). Pruner lines before
the first iteration log line are attributed to iteration 1.
"""
import argparse
import re
import sys
from collections import defaultdict

PRUNER_RE = re.compile(
    r"\[TritonGradientPruner\] key=(\d+) prev sparsity=([\d.]+) \(nnz=(\d+)/(\d+)\)"
)
ITER_RE = re.compile(r"^\s*iteration\s+(\d+)/\s*\d+\s*\|")


def parse(path):
    # iter -> list of (key, sparsity, nnz, total)
    per_iter = defaultdict(list)
    cur_iter = 1
    with open(path) as f:
        for line in f:
            m = PRUNER_RE.search(line)
            if m:
                key = int(m.group(1))
                sp = float(m.group(2))
                nnz = int(m.group(3))
                tot = int(m.group(4))
                per_iter[cur_iter].append((key, sp, nnz, tot))
                continue
            m = ITER_RE.match(line)
            if m:
                # pruner lines printed after this log belong to the next iter
                cur_iter = int(m.group(1)) + 1
    return per_iter


def summarize(per_iter, label):
    print(f"=== {label} ===")
    print(f"{'iter':>5} {'entries':>8} {'mean_sp':>10} {'min_sp':>10} {'max_sp':>10} "
          f"{'total_nnz':>14} {'total_params':>14} {'global_sp':>10}")
    for it in sorted(per_iter):
        rows = per_iter[it]
        n = len(rows)
        sps = [r[1] for r in rows]
        tot_nnz = sum(r[2] for r in rows)
        tot_par = sum(r[3] for r in rows)
        g = 1.0 - tot_nnz / tot_par if tot_par else 0.0
        print(f"{it:>5} {n:>8} {sum(sps)/n:>10.6f} {min(sps):>10.6f} "
              f"{max(sps):>10.6f} {tot_nnz:>14} {tot_par:>14} {g:>10.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="log files (e.g. out.asd out_noea.asd)")
    ap.add_argument("--csv", help="optional CSV output path")
    args = ap.parse_args()

    all_data = {}
    for p in args.files:
        all_data[p] = parse(p)
        summarize(all_data[p], p)
        print()

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("file,iter,key,sparsity,nnz,total\n")
            for p, per_iter in all_data.items():
                for it, rows in sorted(per_iter.items()):
                    for k, sp, nnz, tot in rows:
                        f.write(f"{p},{it},{k},{sp},{nnz},{tot}\n")
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
