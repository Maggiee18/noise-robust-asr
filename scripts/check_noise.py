"""
Phase 6b: check the noise sampler.

Run from the project root (after build_noise_manifest.py):
    python scripts/check_noise.py

Checks
  1. pools are disjoint (no file in two pools)
  2. segments have exactly the requested length (short, medium, longer than
     most files) and contain finite, non silent audio; at least 99% of
     segments are accepted without falling back to a quiet one
  3. same seed -> identical segment; different seed -> different segment
  4. a sampler for the test pool only ever returns test pool files
  5. timing per segment
Saves results/plots/phase6_noise_examples.png (4 noise segments + log Mel).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.features import LogMel  # noqa: E402
from src.musan import MIN_SEGMENT_RMS_DB, POOLS, NoiseBank, read_manifest, rms_db  # noqa: E402

MANIFEST = ROOT / "data" / "manifests" / "musan.csv"
RAW = ROOT / "data" / "raw"
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    rows = read_manifest(MANIFEST)
    print("=" * 70 + "\n1. POOLS\n" + "=" * 70)
    sets = {p: {r["path"] for r in rows if r["pool"] == p} for p in POOLS}
    for i, a in enumerate(POOLS):
        for b in POOLS[i + 1:]:
            check(f"{a} and {b} share no files", not (sets[a] & sets[b]),
                  f"{len(sets[a] & sets[b])} shared")

    banks = {p: NoiseBank(MANIFEST, RAW, p, categories=("noise",)) for p in POOLS}
    print("noise files per pool: " + ", ".join(f"{p} {len(b)}" for p, b in banks.items()))

    print("\n" + "=" * 70 + "\n2. SEGMENTS\n" + "=" * 70)
    rng = np.random.default_rng(0)
    for seconds in (0.5, 5.0, 35.0):
        n = int(seconds * 16000)
        levels, ok_len, ok_finite = [], True, True
        for _ in range(20):
            seg, meta = banks["train"].sample(n, rng)
            ok_len &= seg.shape == (n,) and seg.dtype == torch.float32
            ok_finite &= bool(torch.isfinite(seg).all())
            levels.append(meta["rms_dbfs"])
        check(f"{seconds:>4} s segments: exact length, float32", ok_len)
        check(f"{seconds:>4} s segments: finite", ok_finite)
        quiet = sum(level < MIN_SEGMENT_RMS_DB for level in levels)
        check(f"{seconds:>4} s segments: not silent", quiet <= 1,
              f"RMS {min(levels):.1f} to {max(levels):.1f} dBFS, {quiet}/20 below "
              f"{MIN_SEGMENT_RMS_DB} dBFS")

    rng = np.random.default_rng(11)
    metas = [banks["train"].sample(2 * 16000, rng)[1] for _ in range(1000)]
    rate = sum(m["accepted"] for m in metas) / len(metas)
    redrawn = sum(m["attempts"] > 1 for m in metas)
    check("2 s segments: at least 99% accepted (not near silent)", rate >= 0.99,
          f"{rate:.1%} accepted, {redrawn}/1000 needed a redraw")

    print("\n" + "=" * 70 + "\n3. DETERMINISM\n" + "=" * 70)
    a, ma = banks["train"].sample(80000, np.random.default_rng(123))
    b, mb = banks["train"].sample(80000, np.random.default_rng(123))
    c, mc = banks["train"].sample(80000, np.random.default_rng(124))
    check("same seed gives identical segment", torch.equal(a, b) and ma == mb)
    check("different seed gives a different segment", not torch.equal(a, c))

    print("\n" + "=" * 70 + "\n4. POOL ISOLATION\n" + "=" * 70)
    rng = np.random.default_rng(7)
    used = {banks["test"].sample(16000, rng)[1]["path"] for _ in range(300)}
    check("test sampler only returns test pool files", used <= sets["test"],
          f"{len(used)} distinct test files drawn")
    check("test sampler never returns a train file", not (used & sets["train"]))

    print("\n" + "=" * 70 + "\n5. TIMING\n" + "=" * 70)
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    for _ in range(50):
        banks["train"].sample(15 * 16000, rng)
    print(f"15 s segment: {(time.perf_counter() - t0) / 50 * 1000:.1f} ms each")

    # ---- figure
    fe = LogMel(normalize=False)
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(4, 2, figsize=(12, 8))
    for i in range(4):
        seg, meta = banks["train"].sample(5 * 16000, rng)
        t = np.arange(seg.numel()) / 16000
        ax[i, 0].plot(t, seg.numpy(), linewidth=0.4)
        ax[i, 0].set_title(f"{Path(meta['path']).name}  ({rms_db(seg.numpy()):.1f} dBFS)",
                           fontsize=8)
        ax[i, 0].set_ylim(-1, 1)
        mel = fe(seg)[0][0].numpy()
        ax[i, 1].imshow(mel, origin="lower", aspect="auto", cmap="magma",
                        extent=[0, mel.shape[-1] / 100, 0, 80])
        ax[i, 1].set_title("log Mel", fontsize=8)
    ax[3, 0].set_xlabel("time (s)")
    ax[3, 1].set_xlabel("time (s)")
    out = ROOT / "results" / "plots" / "phase6_noise_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nFigure: {out}")
    print("RESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
