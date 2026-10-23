"""
Phase 3: build and verify LibriSpeech manifests.

Run from the project root:
    python scripts/build_manifests.py
    python scripts/build_manifests.py --subsets dev-clean          (quick test)

For every subset it writes data/manifests/<subset>.csv with one row per
utterance (id, speaker, chapter, relative path, sample rate, length, duration,
raw transcript) and checks:
  * utterance and speaker counts match the published LibriSpeech numbers
  * every audio file exists, is mono and is 16 kHz
  * no duplicate utterance ids, no empty transcripts
  * the character set of the transcripts (reported for Phase 4)
  * NO speaker appears in more than one subset (train / dev / test leakage)

Statistics for the paper go to data/manifests/librispeech_stats.json.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import SAMPLE_RATE, audio_info  # noqa: E402
from src.librispeech import EXPECTED, iter_utterances, write_manifest  # noqa: E402

DEFAULT_SUBSETS = ["train-clean-100", "dev-clean", "test-clean"]


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def build(raw_dir, man_dir, subset):
    rows, problems = [], []
    utts = list(iter_utterances(raw_dir, subset))
    for u in tqdm(utts, desc=f"{subset}: reading headers", unit="utt"):
        audio = raw_dir / u["path"]
        if not audio.exists():
            problems.append(f"missing audio {u['path']}")
            continue
        sr, frames, ch = audio_info(audio)
        if sr != SAMPLE_RATE:
            problems.append(f"{u['utt_id']}: {sr} Hz")
        if ch != 1:
            problems.append(f"{u['utt_id']}: {ch} channels")
        if not u["transcript"]:
            problems.append(f"{u['utt_id']}: empty transcript")
        rows.append({**u, "sample_rate": sr, "num_samples": frames,
                     "duration_s": round(frames / sr, 4)})

    dup = [k for k, v in Counter(r["utt_id"] for r in rows).items() if v > 1]
    if dup:
        problems.append(f"{len(dup)} duplicate utterance ids, e.g. {dup[:3]}")

    speakers = sorted({r["speaker_id"] for r in rows}, key=int)
    exp = EXPECTED.get(subset)
    if exp and (len(rows) != exp["utterances"] or len(speakers) != exp["speakers"]):
        problems.append(f"counts {len(rows)} utt / {len(speakers)} spk, expected "
                        f"{exp['utterances']} / {exp['speakers']}")

    write_manifest(rows, man_dir / f"{subset}.csv")

    durs = [r["duration_s"] for r in rows]
    chars = [len(r["transcript"]) for r in rows]
    # characters per second of audio: needed later to check that the CNN does
    # not shrink the time axis below what CTC needs (Phase 9/10)
    cps = [len(r["transcript"]) / r["duration_s"] for r in rows if r["duration_s"] > 0]
    charset = Counter(c for r in rows for c in r["transcript"])
    stats = dict(
        utterances=len(rows), speakers=len(speakers), hours=round(sum(durs) / 3600, 2),
        duration_s=dict(min=min(durs), mean=round(statistics.mean(durs), 2),
                        median=round(statistics.median(durs), 2),
                        p95=pct(durs, 0.95), max=max(durs)),
        transcript_chars=dict(min=min(chars), mean=round(statistics.mean(chars), 1),
                              max=max(chars)),
        chars_per_second=dict(mean=round(statistics.mean(cps), 2),
                              max=round(max(cps), 2)),
        charset={c: charset[c] for c in sorted(charset)},
    )
    return stats, set(speakers), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    raw_dir, man_dir = args.data_dir / "raw", args.data_dir / "manifests"

    all_stats, spk_sets, failed = {}, {}, False
    for subset in args.subsets:
        print("\n" + "=" * 70 + f"\n{subset}\n" + "=" * 70)
        stats, spk, problems = build(raw_dir, man_dir, subset)
        all_stats[subset], spk_sets[subset] = stats, spk
        d = stats["duration_s"]
        print(f"utterances {stats['utterances']}, speakers {stats['speakers']}, "
              f"{stats['hours']} h")
        print(f"duration s: min {d['min']}  median {d['median']}  p95 {d['p95']}  "
              f"max {d['max']}")
        print(f"transcript chars: mean {stats['transcript_chars']['mean']}  "
              f"max {stats['transcript_chars']['max']}   "
              f"chars/s: mean {stats['chars_per_second']['mean']}  "
              f"max {stats['chars_per_second']['max']}")
        print("characters used: " + "".join(
            "<space>" if c == " " else c for c in stats["charset"]))
        print(f"manifest: {man_dir / (subset + '.csv')}")
        if problems:
            failed = True
            print(f"[FAIL] {len(problems)} problem(s):")
            for p in problems[:10]:
                print("   ", p)
        else:
            print("[PASS] counts, audio format, ids and transcripts")

    # ---- speaker leakage across subsets
    print("\n" + "=" * 70 + "\nSPEAKER OVERLAP CHECK\n" + "=" * 70)
    names = list(spk_sets)
    overlap_found = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = spk_sets[names[i]] & spk_sets[names[j]]
            status = "PASS" if not common else "FAIL"
            overlap_found |= bool(common)
            print(f"[{status}] {names[i]} vs {names[j]}: {len(common)} shared speakers")
    failed |= overlap_found

    stats_path = man_dir / "librispeech_stats.json"
    old = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    old.update(all_stats)
    stats_path.write_text(json.dumps(old, indent=2))
    print(f"\nStatistics: {stats_path}")
    print("RESULT: " + ("FAILED" if failed else "ALL CHECKS PASSED"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
