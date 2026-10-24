"""
Phase 17: figures and tables for the paper, from saved results only (no GPU, no audio).

Run from the project root, after scripts/evaluate.py:
    python scripts/make_plots.py                 # test set figures (default)
    python scripts/make_plots.py --split dev     # same figures on the dev set

Reads
    results/eval/<split>/predictions_<model>.csv   per utterance errors (evaluate.py)
    experiments/<model>/log.csv                    per epoch training logs
    experiments/stage8_*/log.csv                   pilot runs (SpecAugment strength ablation)

Writes to results/plots/<split>/ (PNG for slides / README, PDF for the paper)
    fig1_wer_vs_snr        WER vs SNR, 3 models, 95% bootstrap CIs, SNRs outside the
                           0 to 20 dB training range shaded
    fig2_cer_vs_snr        the same for CER
    fig3_error_types       substitutions / deletions / insertions vs SNR (small multiples)
    fig4_c_minus_b         WER difference C - B per condition with 95% CI
    fig5_training_curves   dev mean WER and training loss per epoch (two panels, one axis each)
    fig6_pilot_ablation    pilot runs: no SpecAugment vs mild vs strong masking
and to results/eval/<split>/
    table_wer_ci.csv / .md          WER and CER per model x condition with 95% CIs
    table_differences.csv / .md     paired differences B-A, C-A, C-B with CIs

Statistics
    Corpus level WER (total word errors / total reference words). 95% confidence
    intervals from a PAIRED bootstrap over utterances: the same 1000 resamples
    (seed 0) are used for every model and condition, so differences between
    models are tested on identical utterance sets.

Design (documented for the paper)
    Colors: validated categorical slots blue / orange / aqua (colorblind safe in
    every pair); every series also has its own marker and a direct label, so
    the figures stay readable in grayscale print. One y axis per chart, light
    hairline grid, text in neutral ink (never in the series color).
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [  # experiment name, short label, long label, color, marker
    ("model_a_clean", "A", "A: clean training", "#2a78d6", "o"),
    ("model_b_noise", "B", "B: noise augmentation", "#eb6834", "s"),
    ("model_c_lb", "C", "C: noise + SpecAugment", "#1baf7a", "^"),
]
STANDARD = ["clean", "20", "10", "5", "0"]
TRAIN_SNR = (0.0, 20.0)       # noise.snr_min_db / snr_max_db used for Models B and C


def outside_training(c):
    """True for SNRs below or above the training range (e.g. -5 dB); clean is not an SNR."""
    return c != "clean" and not TRAIN_SNR[0] <= float(c) <= TRAIN_SNR[1]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
N_BOOT, SEED = 1000, 0

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


# ------------------------------------------------------------------ data
def read_predictions(path):
    """condition -> {utt_id: (word_errors, ref_words, sub, del, ins, char_errors, ref_chars)}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s, d, i = int(r["sub"]), int(r["dele"]), int(r["ins"])
            out.setdefault(r["condition"], {})[r["utt_id"]] = (
                s + d + i, int(r["ref_words"]), s, d, i, int(r["char_errors"]), int(r["ref_chars"]))
    return out


def read_log(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def snr_key(c):
    """Order on the x axis: lowest SNR left, clean at the far right."""
    return 1e9 if c == "clean" else float(c)


class Results:
    def __init__(self, split):
        self.split = split
        self.preds = {}
        for name, *_ in MODELS:
            p = ROOT / "results" / "eval" / split / f"predictions_{name}.csv"
            if not p.exists():
                sys.exit(f"missing {p}: run scripts/evaluate.py --split {split} first")
            self.preds[name] = read_predictions(p)
        first = self.preds[MODELS[0][0]]
        self.conditions = sorted(first, key=snr_key)
        self.utts = sorted(first["clean"])
        for name in self.preds:                    # all models scored on identical sets
            for c in self.conditions:
                assert sorted(self.preds[name][c]) == self.utts, (name, c)
        # arrays [n_utts, 7] per model x condition
        self.arr = {(n, c): np.array([self.preds[n][c][u] for u in self.utts], dtype=np.int64)
                    for n in self.preds for c in self.conditions}
        self.label = f"{split}-clean ({len(self.utts):,} utterances)"
        rng = np.random.default_rng(SEED)
        self.boot = rng.integers(0, len(self.utts), size=(N_BOOT, len(self.utts)))

    def rate(self, name, cond, err_col, ref_col):
        a = self.arr[name, cond]
        point = 100 * a[:, err_col].sum() / a[:, ref_col].sum()
        e, r = a[:, err_col][self.boot].sum(1), a[:, ref_col][self.boot].sum(1)
        return point, 100 * e / r                 # point estimate, bootstrap samples

    def wer(self, name, cond):
        return self.rate(name, cond, 0, 1)

    def cer(self, name, cond):
        return self.rate(name, cond, 5, 6)


def ci(samples):
    return np.percentile(samples, 2.5), np.percentile(samples, 97.5)


# ------------------------------------------------------------------ helpers
def snr_axis(ax, conditions, unseen):
    x = np.arange(len(conditions))
    ax.set_xticks(x, ["clean" if c == "clean" else f"{c} dB" for c in conditions])
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    for k, c in enumerate(conditions):
        if c in unseen:
            ax.axvspan(k - 0.45, k + 0.45, color="#f1f0ec", zorder=0, lw=0)
    ax.set_xlabel("signal to noise ratio (evaluation noise files, never heard in training)")
    return x


def unseen_note(ax, conditions, unseen):
    if unseen:
        k = conditions.index(sorted(unseen, key=snr_key)[0])
        ax.text(k, 0.03, "below the\ntraining range\n(0 to 20 dB)", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=MUTED)


def end_labels(ax, x_end, values, labels, min_gap, dx=0.18):
    """Direct labels right of the last point, nudged apart so they never collide."""
    order = np.argsort(values)
    placed = []
    for k in order:
        y = values[k]
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(labels[k], (x_end, values[k]), xytext=(x_end + dx, y), fontsize=8.5,
                    color=INK2, va="center", annotation_clip=False)


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def write_table(path_csv, rows):
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
    path_csv.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  {path_csv.name} / .md")


# ------------------------------------------------------------------ figures
def fig_rate_vs_snr(R, out, metric, stem):
    fn = R.wer if metric == "WER" else R.cer
    conds = R.conditions
    unseen = [c for c in conds if outside_training(c)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = snr_axis(ax, conds, unseen)
    ends = []
    for name, short, long, color, marker in MODELS:
        pts, lo, hi = [], [], []
        for c in conds:
            p, s = fn(name, c)
            a, b = ci(s)
            pts.append(p), lo.append(p - a), hi.append(b - p)
        ax.errorbar(x, pts, yerr=[lo, hi], color=color, marker=marker, markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=2.5, elinewidth=1,
                    label=long, zorder=3)
        ends.append(pts[-1])
    ax.set_ylabel(f"{metric} (%)")
    ax.set_ylim(0, None)
    gap = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])      # label spacing in this axis' units
    end_labels(ax, x[-1], np.array(ends), [m[1] for m in MODELS], min_gap=gap)
    unseen_note(ax, conds, unseen)
    ax.legend(loc="upper right", fontsize=8.5, handlelength=2.2)
    ax.set_title(f"{metric} vs noise level, {R.label}, 95% bootstrap CIs", loc="left")
    save(fig, out, stem)


def fig_error_types(R, out):
    conds = R.conditions
    unseen = [c for c in conds if outside_training(c)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, (col, title) in zip(axes, [(2, "substitutions"), (3, "deletions"), (4, "insertions")]):
        x = snr_axis(ax, conds, unseen)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8, rotation=0)
        for name, short, long, color, marker in MODELS:
            ys = [R.rate(name, c, col, 1)[0] for c in conds]
            ax.plot(x, ys, color=color, marker=marker, markersize=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=long)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("% of reference words")
    h, l = axes[0].get_legend_handles_labels()
    fig.supxlabel("signal to noise ratio" + (" (shaded: below the 0 to 20 dB training range)"
                                             if unseen else ""), fontsize=9, color=INK2)
    fig.suptitle(f"Error types vs noise level, {R.label}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=8.5, bbox_to_anchor=(0.995, 1.0))
    save(fig, out, "fig3_error_types")


def fig_c_minus_b(R, out):
    conds = R.conditions
    unseen = [c for c in conds if outside_training(c)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = snr_axis(ax, conds, unseen)
    b, c_ = MODELS[1][0], MODELS[2][0]
    for k, cond in enumerate(conds):
        pb, sb = R.wer(b, cond)
        pc, sc = R.wer(c_, cond)
        d, lo, hi = pc - pb, *ci(sc - sb)
        sig = hi < 0 or lo > 0
        ax.plot([k, k], [lo, hi], color=INK2, lw=1.2, zorder=2)
        ax.plot(k, d, marker="D", markersize=7, zorder=3, markeredgecolor=INK2,
                markerfacecolor=INK2 if sig else SURFACE, markeredgewidth=1.3)
        ax.annotate(f"{d:+.2f}", (k, d), xytext=(9, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("WER difference, C minus B\n(percentage points)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, -1.8), max(ymax, 1.0))
    ax.text(0.01, 0.03, "below 0: SpecAugment (C) better", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    ax.text(0.01, 0.92, "above 0: noise only (B) better", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    ax.text(0.99, 0.03, "filled: 95% CI excludes 0", transform=ax.transAxes, ha="right",
            fontsize=8, color=MUTED)
    ax.set_title(f"Effect of adding SpecAugment, paired bootstrap 95% CIs ({R.split})", loc="left")
    save(fig, out, "fig4_c_minus_b")


def fig_training(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    wer_ends, wer_labels = [], []
    for name, short, long, color, marker in MODELS:
        log = ROOT / "experiments" / name / "log.csv"
        if not log.exists():
            continue
        rows = read_log(log)
        ep = [int(r["epoch"]) for r in rows]
        a1.plot(ep, [100 * float(r["mean_wer"]) for r in rows], color=color, label=long)
        a2.plot(ep, [float(r["train_loss"]) for r in rows], color=color, label=long)
        wer_ends.append(100 * float(rows[-1]["mean_wer"]))
        wer_labels.append(short)
        a2.annotate(short, (ep[-1], float(rows[-1]["train_loss"])), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    end_labels(a1, 20, np.array(wer_ends), wer_labels, min_gap=2.5, dx=0.4)
    a1.set_title("dev mean WER (clean, 20, 10, 5, 0 dB)", loc="left")
    a1.set_ylabel("WER (%)")
    a2.set_title("training loss (CTC per character, with augmentation)", loc="left")
    a2.set_ylabel("loss")
    a2.set_yscale("log")
    for ax in (a1, a2):
        ax.set_xlabel("epoch")
        ax.set_xticks(range(0, 21, 5))
    a1.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig5_training_curves")


def fig_pilots(out):
    runs = [("stage8_a5000", "A: clean"), ("stage8_b5000", "B: noise"),
            ("stage8_c5000_lb", "C mild\n(1 + 1 masks)"), ("stage8_c5000", "C strong\n(2 + 2 masks)")]
    vals, labels = [], []
    for run, lab in runs:
        log = ROOT / "experiments" / run / "log.csv"
        if log.exists():
            vals.append(100 * min(float(r["mean_wer"]) for r in read_log(log)))
            labels.append(lab)
    if not vals:
        return
    colors = {"A": MODELS[0][3], "B": MODELS[1][3], "C": MODELS[2][3]}
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, width=0.55, color=[colors[l[0]] for l in labels],
                  edgecolor=SURFACE, linewidth=2)
    for k, l in enumerate(labels):              # strong masking: hatch as secondary cue
        if "strong" in l:
            bars[k].set_hatch("///")
            bars[k].set_edgecolor(SURFACE)
    for k, v in enumerate(vals):
        ax.text(k, v + 0.8, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("best mean dev WER (%)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("Pilot runs: 5,000 utterances, 15 epochs, 500 dev utterances", loc="left")
    save(fig, out, "fig6_pilot_ablation")


# ------------------------------------------------------------------ tables
def tables(R, split):
    out = ROOT / "results" / "eval" / split
    rows = []
    for name, short, long, *_ in MODELS:
        for c in R.conditions:
            pw, sw = R.wer(name, c)
            pc, sc = R.cer(name, c)
            lw, hw = ci(sw)
            lc, hc = ci(sc)
            rows.append(dict(model=short, condition=c, outside_training_snr=outside_training(c), used_for_selection=c in STANDARD,
                             wer=round(pw, 2), wer_ci=f"[{lw:.2f}, {hw:.2f}]",
                             cer=round(pc, 2), cer_ci=f"[{lc:.2f}, {hc:.2f}]"))
        mw = np.mean([R.wer(name, c)[1] for c in STANDARD], axis=0)
        point = np.mean([R.wer(name, c)[0] for c in STANDARD])
        lo, hi = ci(mw)
        rows.append(dict(model=short, condition="mean (standard 5)", outside_training_snr=False, used_for_selection=True,
                         wer=round(point, 2), wer_ci=f"[{lo:.2f}, {hi:.2f}]", cer="", cer_ci=""))
    write_table(out / "table_wer_ci.csv", rows)

    diffs = []
    pairs = [(MODELS[1], MODELS[0]), (MODELS[2], MODELS[0]), (MODELS[2], MODELS[1])]
    for (n1, s1, *_), (n0, s0, *_) in pairs:
        for c in R.conditions + ["mean"]:
            cs = STANDARD if c == "mean" else [c]
            d = np.mean([R.wer(n1, cc)[0] - R.wer(n0, cc)[0] for cc in cs])
            ds = np.mean([R.wer(n1, cc)[1] - R.wer(n0, cc)[1] for cc in cs], axis=0)
            lo, hi = ci(ds)
            diffs.append(dict(comparison=f"{s1} - {s0}", condition=c, diff_points=round(d, 2),
                              ci95=f"[{lo:.2f}, {hi:.2f}]",
                              p_first_better=round(float((ds < 0).mean()), 3),
                              significant=bool(hi < 0 or lo > 0)))
    write_table(out / "table_differences.csv", diffs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    args = ap.parse_args()
    R = Results(args.split)
    out = ROOT / "results" / "plots" / args.split
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.split}: {len(R.utts)} utterances, conditions {R.conditions}, "
          f"{N_BOOT} paired bootstrap resamples")
    print(f"figures -> {out}")
    fig_rate_vs_snr(R, out, "WER", "fig1_wer_vs_snr")
    fig_rate_vs_snr(R, out, "CER", "fig2_cer_vs_snr")
    fig_error_types(R, out)
    fig_c_minus_b(R, out)
    fig_training(out)
    fig_pilots(out)
    print(f"tables -> results/eval/{args.split}")
    tables(R, args.split)
    print("done")


if __name__ == "__main__":
    main()
