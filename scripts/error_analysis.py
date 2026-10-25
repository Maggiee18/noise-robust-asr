"""
Phase 18: error analysis of Models A, B and C, from saved transcripts only (no GPU).

Run from the project root, after scripts/evaluate.py:
    python scripts/error_analysis.py                 # test set (default)
    python scripts/error_analysis.py --split dev
    python scripts/error_analysis.py --decoding lm   # results of scripts/decode_lm.py

Reads
    results/eval/<split>/predictions_<model>.csv     references and hypotheses (evaluate.py)
    data/manifests/train-clean-100.csv              training transcripts (word frequencies)
    data/manifests/<split>-clean.csv                utterance durations

Questions it answers (every number goes into report.md)
  1. Overall: share of perfectly recognized utterances, near empty outputs.
  2. Deletions: are words dropped one at a time or in long silent stretches,
     and where in the utterance (start / middle / end)?
  3. Substitutions: near misses (a spelling slip, e.g. BARTLEY -> BARTLY) or
     completely different words?
  4. Which words: error rate by how often the word occurs in the TRAINING
     transcripts (never seen, rare ... very frequent) and by word length.
  5. Utterance length: are short or long utterances harder?
  6. Most frequent confusions, deleted and inserted words per model.
  7. Example transcripts chosen automatically by fixed rules (no cherry picking).

Writes results/analysis/<split>/ (or <split>_lm/ with --decoding lm)
    report.md                    every table below, ready for the paper
    word_frequency.csv           error rate per model x condition x frequency bin
    utterance_length.csv         WER per model x condition x duration bin
    deletion_runs.csv            deleted words by run length per model x condition
    substitution_types.csv       near miss vs different word per model x condition
    top_confusions.csv           top substitutions / deletions / insertions
    examples.md                  rule based example transcripts
    fig7_word_frequency.png/pdf  error rate vs training frequency (clean and 0 dB)
    fig8_deletion_runs.png/pdf   deleted words by run length, as % of all reference words

Definitions
    Word level alignment = jiwer (same as the WER). A reference word is an ERROR
    if it is substituted or deleted; insertions have no reference word and are
    counted separately. "Word error rate of a word class" = (substituted +
    deleted words of that class) / (reference words of that class).
    Near miss: character error rate between the reference word and the word
    that replaced it is <= 1/3 (for example 1 wrong letter in a 3+ letter word).
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.vocabulary import Vocabulary  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from make_plots import INK2, MODELS, SURFACE, save  # noqa: E402  (shared style)

FREQ_BINS = [("never seen", 0, 0), ("1 to 9", 1, 9), ("10 to 99", 10, 99),
             ("100 to 999", 100, 999), ("1000+", 1000, 10 ** 12)]
DUR_BINS = [("under 4 s", 0, 4), ("4 to 8 s", 4, 8), ("8 to 15 s", 8, 15), ("15 s+", 15, 1e9)]
RUN_BINS = [("1 word", 1, 1), ("2 words", 2, 2), ("3 to 5", 3, 5), ("6+", 6, 10 ** 9)]
NEAR_MISS = 1 / 3


def snr_key(c):
    return 1e9 if c == "clean" else float(c)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bin_of(value, bins):
    for name, lo, hi in bins:
        if lo <= value <= hi:
            return name
    return bins[-1][0]


def md_table(rows, cols=None):
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ analysis of one utterance
def analyse(ref, hyp):
    """Alignment based facts about one reference / hypothesis pair."""
    r, h = ref.split(), hyp.split()
    if not r:
        return None
    out = jiwer.process_words(ref, hyp if hyp else "")
    wrong = [False] * len(r)                       # per reference word: substituted / deleted
    subs, dels, ins, del_runs, del_pos = [], [], [], [], []
    for a in out.alignments[0]:
        if a.type == "substitute":
            for k in range(a.ref_end_idx - a.ref_start_idx):
                wrong[a.ref_start_idx + k] = True
                subs.append((r[a.ref_start_idx + k], h[a.hyp_start_idx + k]))
        elif a.type == "delete":
            n = a.ref_end_idx - a.ref_start_idx
            del_runs.append(n)
            for k in range(a.ref_start_idx, a.ref_end_idx):
                wrong[k] = True
                dels.append(r[k])
                del_pos.append(k / max(len(r) - 1, 1))
        elif a.type == "insert":
            ins.extend(h[a.hyp_start_idx:a.hyp_end_idx])
    errors = out.substitutions + out.deletions + out.insertions
    return dict(words=r, wrong=wrong, subs=subs, dels=dels, ins=ins, del_runs=del_runs,
                del_pos=del_pos, errors=errors, n=len(r), hyp_words=len(h))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--decoding", default="greedy", choices=["greedy", "lm"])
    args = ap.parse_args()
    vocab = Vocabulary()
    man = ROOT / "data" / "manifests"
    src_dir = args.split + ("_lm" if args.decoding == "lm" else "")
    dec_label = ", 3-gram LM decoding" if args.decoding == "lm" else ""
    out = ROOT / "results" / "analysis" / src_dir
    out.mkdir(parents=True, exist_ok=True)

    train_freq = Counter(w for r in read_csv(man / "train-clean-100.csv")
                         for w in vocab.normalize(r["transcript"]).split())
    duration = {r["utt_id"]: float(r["duration_s"]) for r in read_csv(man / f"{args.split}-clean.csv")}

    # A[model][condition][utt] = analysis
    A, conds = {}, None
    for name, *_ in MODELS:
        p = ROOT / "results" / "eval" / src_dir / f"predictions_{name}.csv"
        if not p.exists():
            sys.exit(f"missing {p}: run scripts/{'decode_lm' if args.decoding == 'lm' else 'evaluate'}.py --split {args.split} first")
        A[name] = defaultdict(dict)
        for row in read_csv(p):
            res = analyse(row["reference"], row["hypothesis"])
            if res is not None:
                res["ref"], res["hyp"] = row["reference"], row["hypothesis"]
                A[name][row["condition"]][row["utt_id"]] = res
        conds = sorted(A[name], key=snr_key)
        print(f"  analysed {name}: {sum(len(v) for v in A[name].values())} utterance x condition pairs")
    utts = sorted(A[MODELS[0][0]]["clean"])
    lowest = conds[0]
    short = {m[0]: m[1] for m in MODELS}
    report = [f"# Error analysis, {args.split} set ({len(utts):,} utterances){dec_label}\n",
              "Generated by scripts/error_analysis.py from results/eval/"
              f"{src_dir}/predictions_*.csv. Training word counts from train-clean-100 "
              f"({sum(train_freq.values()):,} words, {len(train_freq):,} distinct).\n"]

    # 1. utterance level overview ------------------------------------------------
    rows = []
    for name, *_ in MODELS:
        for c in conds:
            d = A[name][c]
            n = len(d)
            perfect = sum(1 for x in d.values() if x["errors"] == 0)
            silent = sum(1 for x in d.values() if x["hyp_words"] <= 0.25 * x["n"])
            rows.append(dict(model=short[name], condition=c,
                             perfect_utts_pct=round(100 * perfect / n, 1),
                             near_empty_utts_pct=round(100 * silent / n, 2),
                             empty_utts=sum(1 for x in d.values() if x["hyp_words"] == 0)))
    report += ["## 1. Utterances recognized perfectly, and near empty outputs\n",
               "near_empty = the hypothesis has at most a quarter as many words as the reference.\n",
               md_table(rows), ""]

    # 2. deletions: runs and position -------------------------------------------
    run_rows, pos_rows = [], []
    for name, *_ in MODELS:
        for c in conds:
            runs = [n for x in A[name][c].values() for n in x["del_runs"]]
            words = sum(runs)
            total = sum(x["n"] for x in A[name][c].values())
            row = dict(model=short[name], condition=c, deleted_words=words,
                       deletion_rate_pct=round(100 * words / total, 2))
            for label, lo, hi in RUN_BINS:
                row[f"in runs of {label}"] = (round(100 * sum(n for n in runs if lo <= n <= hi) /
                                                    words, 1) if words else 0.0)
            row["longest_run"] = max(runs) if runs else 0
            run_rows.append(row)
            pos = np.array([p for x in A[name][c].values() for p in x["del_pos"]])
            if len(pos):
                pos_rows.append(dict(model=short[name], condition=c,
                                     start_third_pct=round(100 * (pos < 1 / 3).mean(), 1),
                                     middle_third_pct=round(100 * ((pos >= 1 / 3) & (pos <= 2 / 3)).mean(), 1),
                                     end_third_pct=round(100 * (pos > 2 / 3).mean(), 1)))
    write_csv(out / "deletion_runs.csv", run_rows)
    report += ["## 2. Deletions: one word at a time, or whole stretches?\n",
               "Share of deleted words that belong to a run of N consecutive deleted reference words.\n",
               md_table(run_rows), "",
               "Where deleted words sit in the utterance (relative position of the word).\n",
               md_table(pos_rows), ""]

    # 3. substitutions: near miss vs different word ------------------------------
    sub_rows = []
    for name, *_ in MODELS:
        for c in conds:
            pairs = [p for x in A[name][c].values() for p in x["subs"]]
            near = sum(1 for r, h in pairs if jiwer.cer(r, h) <= NEAR_MISS)
            total = sum(x["n"] for x in A[name][c].values())
            sub_rows.append(dict(model=short[name], condition=c, substitutions=len(pairs),
                                 substitution_rate_pct=round(100 * len(pairs) / total, 2),
                                 near_miss_pct_of_subs=round(100 * near / max(len(pairs), 1), 1),
                                 different_word_pct_of_subs=round(100 * (len(pairs) - near) /
                                                                  max(len(pairs), 1), 1)))
    write_csv(out / "substitution_types.csv", sub_rows)
    report += ["## 3. Substitutions: spelling slip or a different word?\n",
               f"near miss = character error rate between the two words <= {NEAR_MISS:.2f}.\n",
               md_table(sub_rows), ""]

    # 4. error rate by training frequency and by word length ----------------------
    freq_rows, len_rows = [], []
    ref_words = [w for u in utts for w in A[MODELS[0][0]]["clean"][u]["words"]]
    share = Counter(bin_of(train_freq[w], FREQ_BINS) for w in ref_words)
    for name, *_ in MODELS:
        for c in conds:
            by_f, by_l = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
            for x in A[name][c].values():
                for w, bad in zip(x["words"], x["wrong"]):
                    fb = bin_of(train_freq[w], FREQ_BINS)
                    lb = "1 to 3" if len(w) <= 3 else "4 to 6" if len(w) <= 6 else "7+"
                    by_f[fb][0] += bad
                    by_f[fb][1] += 1
                    by_l[lb][0] += bad
                    by_l[lb][1] += 1
            for label, *_ in FREQ_BINS:
                e, n = by_f[label]
                freq_rows.append(dict(model=short[name], condition=c, train_count=label,
                                      ref_words=n, word_error_pct=round(100 * e / n, 2) if n else ""))
            for label in ["1 to 3", "4 to 6", "7+"]:
                e, n = by_l[label]
                len_rows.append(dict(model=short[name], condition=c, letters=label, ref_words=n,
                                     word_error_pct=round(100 * e / n, 2) if n else ""))
    write_csv(out / "word_frequency.csv", freq_rows)
    pivot = lambda rows, key, cs: [  # noqa: E731
        dict(**{key: k}, **{f"{short[m[0]]} {c}": next(r["word_error_pct"] for r in rows
                                                     if r["model"] == short[m[0]] and r["condition"] == c
                                                     and r[key] == k)
                           for c in cs for m in MODELS})
        for k in dict.fromkeys(r[key] for r in rows)]
    show = ["clean", "0"] + ([lowest] if lowest not in ("clean", "0") else [])
    report += ["## 4. Which words fail\n",
               "Reference words by how often they occur in the training transcripts: " +
               ", ".join(f"{k} {100 * share[k] / len(ref_words):.1f}%" for k, *_ in FREQ_BINS) + ".\n",
               "Word error rate (% of reference words of that class substituted or deleted):\n",
               md_table(pivot(freq_rows, "train_count", show)), "",
               "By word length (letters):\n", md_table(pivot(len_rows, "letters", show)), ""]

    # 5. utterance length ----------------------------------------------------------
    dur_rows = []
    for name, *_ in MODELS:
        for c in conds:
            agg = defaultdict(lambda: [0, 0, 0])
            for u, x in A[name][c].items():
                b = bin_of(duration[u], DUR_BINS)
                agg[b][0] += x["errors"]
                agg[b][1] += x["n"]
                agg[b][2] += 1
            for label, *_ in DUR_BINS:
                e, n, k = agg[label]
                dur_rows.append(dict(model=short[name], condition=c, duration=label, utterances=k,
                                     wer_pct=round(100 * e / n, 2) if n else ""))
    write_csv(out / "utterance_length.csv", dur_rows)
    piv = [dict(duration=d, utterances=next(r["utterances"] for r in dur_rows if r["duration"] == d),
                **{f"{short[m[0]]} {c}": next(r["wer_pct"] for r in dur_rows if r["model"] == short[m[0]]
                                             and r["condition"] == c and r["duration"] == d)
                   for c in show for m in MODELS}) for d, *_ in DUR_BINS]
    report += ["## 5. Utterance length\n", "WER (%) by utterance duration:\n", md_table(piv), ""]

    # 6. top confusions --------------------------------------------------------------
    top_rows, lines = [], ["## 6. Most frequent errors\n"]
    for c in ["clean", "0"] + ([lowest] if lowest not in ("clean", "0") else []):
        lines.append(f"### {c}{'' if c == 'clean' else ' dB'}\n")
        tab = []
        for name, *_ in MODELS:
            S = Counter(p for x in A[name][c].values() for p in x["subs"])
            D = Counter(w for x in A[name][c].values() for w in x["dels"])
            I = Counter(w for x in A[name][c].values() for w in x["ins"])
            for kind, cnt in [("substitution", S), ("deletion", D), ("insertion", I)]:
                for item, k in cnt.most_common(10):
                    top_rows.append(dict(model=short[name], condition=c, kind=kind,
                                         item=f"{item[0]} -> {item[1]}" if kind == "substitution" else item,
                                         count=k))
            tab.append(dict(model=short[name],
                            substitutions=", ".join(f"{a}->{b} ({k})" for (a, b), k in S.most_common(5)),
                            deletions=", ".join(f"{w} ({k})" for w, k in D.most_common(5)),
                            insertions=", ".join(f"{w} ({k})" for w, k in I.most_common(5))))
        lines += [md_table(tab), ""]
    write_csv(out / "top_confusions.csv", top_rows)
    report += lines

    # 7. examples by fixed rules -----------------------------------------------------
    a, b, c_ = (m[0] for m in MODELS)

    def wer_u(model, cond, u):
        x = A[model][cond][u]
        return x["errors"] / x["n"]

    def mid_len(u):
        return 6 <= A[a]["clean"][u]["n"] <= 14

    ex = ["# Example transcripts (chosen by fixed rules, not by hand)\n"]

    def show_utt(title, rule, u, cs):
        ex.append(f"## {title}\n\n*Rule: {rule}*  utterance `{u}`\n")
        ex.append(f"- **reference**: {A[a]['clean'][u]['ref']}")
        for cond in cs:
            for m in MODELS:
                x = A[m[0]][cond][u]
                ex.append(f"- **{m[1]} @ {cond}{'' if cond == 'clean' else ' dB'}** "
                          f"(WER {100 * x['errors'] / x['n']:.0f}%): {x['hyp'] or '(empty)'}")
        ex.append("")

    cand = [u for u in utts if mid_len(u)]
    u1 = min(cand, key=lambda u: (wer_u(b, "0", u) - wer_u(a, "0", u), u))
    show_utt("Noise training rescues an utterance at 0 dB",
             "6 to 14 words; largest WER drop from A to B at 0 dB", u1, ["clean", "0"])
    u2 = min(cand, key=lambda u: (wer_u(c_, lowest, u) - wer_u(b, lowest, u), u))
    show_utt(f"SpecAugment helps at {lowest} dB",
             f"6 to 14 words; largest WER drop from B to C at {lowest} dB", u2, [lowest])
    u3 = max(cand, key=lambda u: (sum(A[a][lowest][u]["del_runs"] or [0]) / A[a][lowest][u]["n"], u))
    show_utt(f"Model A goes silent at {lowest} dB",
             f"6 to 14 words; largest share of deleted words for A at {lowest} dB", u3, [lowest])
    shared = Counter()
    for u in utts:
        sets = [{r for r, _ in A[m[0]]["clean"][u]["subs"]} for m in MODELS]
        for w in set.intersection(*sets):
            if train_freq[w] == 0:
                shared[w] += 1
    if shared:
        w0 = shared.most_common(1)[0][0]
        hits = [u for u in utts if all(w0 in {r for r, _ in A[m[0]]["clean"][u]["subs"]}
                                       for m in MODELS)]
        u4 = min(hits, key=lambda u: (A[a]["clean"][u]["n"] > 20, A[a]["clean"][u]["n"], u))
        show_utt("A word no model has seen in training (clean)",
                 "most frequent never seen word that all three models get wrong on clean "
                 "speech; shortest utterance containing it",
                 u4, ["clean"])
    (out / "examples.md").write_text("\n".join(ex) + "\n", encoding="utf-8")

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    labels = [k for k, *_ in FREQ_BINS]
    for ax, c in zip(axes, ["clean", lowest]):
        x = np.arange(len(labels))
        for name, s, long, color, marker in MODELS:
            ys = [next(r["word_error_pct"] for r in freq_rows if r["model"] == s and r["condition"] == c
                       and r["train_count"] == k) for k in labels]
            ax.plot(x, ys, color=color, marker=marker, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.3, label=long)
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.set_xlabel("times the word occurs in the training transcripts")
        ax.set_title(f"{'clean speech' if c == 'clean' else c + ' dB SNR'}", loc="left")
    axes[0].set_ylabel("words substituted or deleted (%)")
    axes[0].legend(fontsize=8.5)
    fig.suptitle(f"Rare words fail most, {args.split}-clean{dec_label}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig7_word_frequency")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for ax, c in zip(axes, ["0", lowest] if lowest != "0" else ["5", "0"]):
        x = np.arange(len(RUN_BINS))
        w = 0.26
        for k, (name, s, long, color, marker) in enumerate(MODELS):
            r = next(r for r in run_rows if r["model"] == s and r["condition"] == c)
            vals = [r[f"in runs of {lab}"] * r["deletion_rate_pct"] / 100 for lab, *_ in RUN_BINS]
            ax.bar(x + (k - 1) * w, vals, width=w - 0.02, color=color, label=long,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.set_xticks(x, [lab for lab, *_ in RUN_BINS])
        ax.set_xlabel("length of the run of consecutive deleted words")
        rates = ", ".join(f"{s} {next(r['deletion_rate_pct'] for r in run_rows if r['model'] == s and r['condition'] == c):.1f}%"
                          for _, s, *_ in MODELS)
        ax.set_title(f"{c} dB  (total deletion rate: {rates})", loc="left", fontsize=10)
    axes[0].set_ylabel("deleted words, % of all\nreference words")
    axes[0].legend(fontsize=8.5)
    fig.suptitle(f"How deletions are grouped, {args.split}-clean{dec_label}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig8_deletion_runs")

    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nreport -> {out / 'report.md'}\nexamples -> {out / 'examples.md'}")


if __name__ == "__main__":
    main()
