#!/usr/bin/env python3
"""
plot_bench.py - plot every kernel saved by bench_matmul.py against cuBLAS.

Reads results/<label>.csv (the latest run of each kernel; history/ is ignored
unless you pass files explicitly), so the plots grow as you add kernels.

Usage:
  python plot_bench.py                          # all kernels in results/
  python plot_bench.py -k naive -k coalesced    # only these labels
  python plot_bench.py results/history/naive_20261026-101500.csv results/naive.csv
  python plot_bench.py --at 4096                # size used for the progress table

Writes plots/overview.png, plots/{tflops,pct_cublas,time,bandwidth}.png and
plots/progress.md, a table of where each kernel stands at one size. Those are
always the latest; every run also keeps a copy that is never overwritten:

  plots/history/<timestamp>_<kernels>/   e.g. plots/history/20261112-093000_naive+coalesced/

so the step-by-step figures (naive alone, then + coalesced, then + shared_mem)
survive later runs. Pass --no-snapshot to skip the copy.
"""
import argparse
import datetime as dt
import shutil
from pathlib import Path

import matplotlib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # repo root
RESULTS_DIR = ROOT / "results"

# Categorical slots in fixed order: a kernel keeps its color as more are added.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]
CUBLAS_COLOR = "#0b0b0b"
PEAK_COLOR = "#8a8984"
GRID_COLOR = "#e4e3df"
TEXT_SECONDARY = "#52514e"


def parse_args():
    p = argparse.ArgumentParser(description="Plot GEMM benchmark results")
    p.add_argument("csvs", nargs="*", help="CSV files (default: every results/*.csv)")
    p.add_argument("-k", "--kernel", action="append", default=[], help="Only plot these labels")
    p.add_argument("--at", type=int, default=None,
                   help="Size for the progress table (default: largest size every kernel has)")
    p.add_argument("--out-dir", default=str(ROOT / "plots"))
    p.add_argument("--no-peaks", action="store_true", help="Hide the GPU peak lines")
    p.add_argument("--show", action="store_true", help="Open an interactive window too")
    p.add_argument("--no-snapshot", action="store_true",
                   help="Don't keep a copy in plots/history/ (only update the latest plots)")
    return p.parse_args()


def load(args):
    paths = [Path(c) for c in args.csvs] or sorted(RESULTS_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit("[error] no results yet; run bench_matmul.py -k <kernel> first")

    frames = []
    for path in paths:
        df = pd.read_csv(path).sort_values("size")
        if path.parent.name == "history":  # tell repeated runs of one kernel apart
            df["kernel"] = path.stem
        frames.append(df)
    if args.kernel:
        frames = [df for df in frames if df["kernel"].iloc[0] in args.kernel]
        if not frames:
            raise SystemExit(f"[error] none of {args.kernel} found")

    # Progression order: slowest first, so colors and legend read naive -> fastest.
    frames.sort(key=lambda df: df["tflops"].max())
    if len(frames) > len(SERIES_COLORS):
        print(f"[warn] {len(frames)} kernels but {len(SERIES_COLORS)} colors; "
              f"plotting the {len(SERIES_COLORS)} fastest (use -k to pick)")
        frames = frames[-len(SERIES_COLORS):]
    return frames


def newest(frames):
    """cuBLAS reference: the most recent run, since every run re-times cuBLAS."""
    return max(frames, key=lambda df: str(df["timestamp"].iloc[0]))


def style_axis(ax, sizes, title, ylabel, logy=False):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("matrix size N  (M = N = K)", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def series(ax, frames, col, base=None, base_col=None):
    for color, df in zip(SERIES_COLORS, frames):
        ax.plot(df["size"], df[col], "o-", color=color, lw=2, ms=4, label=df["kernel"].iloc[0])
    if base is not None:
        ax.plot(base["size"], base[base_col], "s--", color=CUBLAS_COLOR, lw=1.5, ms=4, label="cuBLAS")


def draw(ax, kind, frames, base, sizes, show_peaks):
    if kind == "tflops":
        series(ax, frames, "tflops", base, "cublas_tflops")
        peak = base["peak_tflops"].iloc[0]
        if show_peaks and pd.notna(peak):
            ax.axhline(peak, color=PEAK_COLOR, ls=":", lw=1, label=f"FP32 peak ({peak:g})")
        style_axis(ax, sizes, "Throughput", "TFLOPS")
    elif kind == "pct_cublas":
        series(ax, frames, "pct_cublas")
        ax.axhline(100, color=CUBLAS_COLOR, ls="--", lw=1.5, label="cuBLAS = 100%")
        style_axis(ax, sizes, "Percent of cuBLAS (same run, same inputs)", "% of cuBLAS (log)", logy=True)
    elif kind == "time":
        series(ax, frames, "median_ms", base, "cublas_median_ms")
        style_axis(ax, sizes, "Runtime", "median ms (log)", logy=True)
    elif kind == "bandwidth":
        series(ax, frames, "gbs", base, "cublas_gbs")
        peak = base["peak_gbs"].iloc[0]
        if show_peaks and pd.notna(peak):
            ax.axhline(peak, color=PEAK_COLOR, ls=":", lw=1, label=f"DRAM peak ({peak:g} GB/s)")
        style_axis(ax, sizes, "Effective bandwidth (minimum bytes / time)", "GB/s")
    ax.legend(fontsize=8, frameon=False)


def progress_table(frames, at):
    """Markdown table of each kernel at one size, with step-over-step speedup."""
    lines = [f"Progress at N = {at}", "",
             "| kernel | ms | TFLOPS | % cuBLAS | % peak | speedup vs previous | correct |",
             "|---|---:|---:|---:|---:|---:|---|"]
    prev = None
    for df in frames:
        r = df[df["size"] == at]
        if r.empty:
            lines.append(f"| {df['kernel'].iloc[0]} | - | - | - | - | - | not run at {at} |")
            continue
        r = r.iloc[0]
        step = f"{prev / r['median_ms']:.2f}x" if prev else "-"
        prev = r["median_ms"]
        lines.append(f"| {r['kernel']} | {r['median_ms']:.3f} | {r['tflops']:.2f} | "
                     f"{r['pct_cublas']:.1f}% | {r['pct_peak']:.1f}% | {step} | {r.get('correct', '')} |")
    return "\n".join(lines)


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = load(args)
    base = newest(frames)
    sizes = sorted(set().union(*[set(df["size"]) for df in frames]))
    common = sorted(set.intersection(*[set(df["size"]) for df in frames]))
    at = args.at or (common[-1] if common else sizes[-1])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["tflops", "pct_cublas", "time", "bandwidth"]
    show_peaks = not args.no_peaks

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, kind in zip(axes.flat, kinds):
        draw(ax, kind, frames, base, sizes, show_peaks)
    fig.suptitle(f"FP32 GEMM vs cuBLAS - {base['gpu'].iloc[0]}", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)

    for kind in kinds:
        f, ax = plt.subplots(figsize=(8, 5.5))
        draw(ax, kind, frames, base, sizes, show_peaks)
        f.tight_layout()
        f.savefig(out / f"{kind}.png", dpi=150)
        if not args.show:
            plt.close(f)

    table = progress_table(frames, at)
    (out / "progress.md").write_text(table + "\n")
    print(table)
    print(f"\nsaved plots to {out}/")

    if not args.no_snapshot:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        labels = "+".join(df["kernel"].iloc[0] for df in frames)
        snap = out / "history" / f"{stamp}_{labels}"
        snap.mkdir(parents=True, exist_ok=True)
        for name in ["overview.png", *[f"{k}.png" for k in kinds], "progress.md"]:
            shutil.copy2(out / name, snap / name)
        print(f"kept a copy in {snap}/")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
