"""
Phase 6a: scan MUSAN, measure every file, split into disjoint pools.

Run from the project root:
    python scripts/build_noise_manifest.py

Reads every MUSAN WAV once (about 12 GB, several minutes) to measure its
length, loudness (RMS dBFS) and peak. Writes
    data/manifests/musan.csv          one row per file, with pool and any exclusion reason
    data/manifests/musan_stats.json   files and hours per category and pool
and checks
    * all files are 16 kHz mono
    * every usable file is in exactly one pool; no pool is empty
    * the split is reproducible (recomputing it gives identical pools)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.musan import (CATEGORIES, MIN_DURATION_S, MIN_FILE_RMS_DB, POOLS,  # noqa: E402
                       SAMPLE_RATE, assign_pools, file_stats, write_manifest)

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    raw = ROOT / "data" / "raw"
    musan = raw / "musan"
    files = sorted(p for c in CATEGORIES for p in (musan / c).rglob("*.wav"))
    if not files:
        print(f"No WAV files under {musan}. Run scripts/download_data.py --datasets musan")
        sys.exit(1)

    rows, bad_format = [], []
    for p in tqdm(files, desc="Measuring MUSAN files", unit="file"):
        rel = p.relative_to(raw).as_posix()
        category, source = rel.split("/")[1], rel.split("/")[2]
        sr, frames, ch, rms, peak = file_stats(p)
        dur = frames / sr
        reason = ""
        if sr != SAMPLE_RATE or ch != 1:
            reason = f"format {sr} Hz {ch} ch"
            bad_format.append(rel)
        elif dur < MIN_DURATION_S:
            reason = f"shorter than {MIN_DURATION_S} s"
        elif rms < MIN_FILE_RMS_DB:
            reason = f"quieter than {MIN_FILE_RMS_DB} dBFS"
        rows.append(dict(category=category, source=source, path=rel, sample_rate=sr,
                         num_samples=frames, duration_s=round(dur, 4),
                         rms_dbfs=round(rms, 2), peak=round(peak, 4),
                         pool="excluded" if reason else "", exclude_reason=reason))

    assign_pools(rows)
    # reproducibility: recompute from a shuffled copy and compare
    shuffled = [dict(r, pool=("excluded" if r["exclude_reason"] else "")) for r in reversed(rows)]
    again = {r["path"]: r["pool"] for r in assign_pools(shuffled)}
    out = ROOT / "data" / "manifests" / "musan.csv"
    write_manifest(rows, out)

    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    stats = defaultdict(lambda: defaultdict(lambda: dict(files=0, hours=0.0)))
    for r in rows:
        s = stats[r["category"]][r["pool"]]
        s["files"] += 1
        s["hours"] += r["duration_s"] / 3600
    print(f"{'category':<8} " + " ".join(f"{p:>16}" for p in POOLS + ("excluded",)))
    for c in CATEGORIES:
        cells = []
        for p in POOLS + ("excluded",):
            s = stats[c][p]
            cells.append(f"{s['files']:>5} f {s['hours']:>6.2f} h")
        print(f"{c:<8} " + " ".join(f"{x:>16}" for x in cells))

    excluded = [r for r in rows if r["pool"] == "excluded"]
    if excluded:
        print(f"\nExcluded {len(excluded)} file(s):")
        for r in excluded[:15]:
            print(f"   {r['path']}  ({r['duration_s']:.2f} s, {r['rms_dbfs']:.1f} dBFS): "
                  f"{r['exclude_reason']}")
        if len(excluded) > 15:
            print(f"   ... and {len(excluded) - 15} more (see musan.csv)")

    print("\n" + "=" * 70 + "\nCHECKS\n" + "=" * 70)
    check("all files 16 kHz mono", not bad_format, f"{len(bad_format)} bad, e.g. {bad_format[:2]}")
    check("every file has exactly one pool",
          all(r["pool"] in POOLS + ("excluded",) for r in rows))
    for c in CATEGORIES:
        for p in POOLS:
            check(f"{c}/{p} pool not empty", stats[c][p]["files"] > 0)
    check("split is reproducible", all(again[r["path"]] == r["pool"] for r in rows))

    stats_out = {c: {p: dict(files=v["files"], hours=round(v["hours"], 3))
                     for p, v in stats[c].items()} for c in stats}
    (ROOT / "data" / "manifests" / "musan_stats.json").write_text(json.dumps(stats_out, indent=2))
    print(f"\nManifest: {out}")
    print("RESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
