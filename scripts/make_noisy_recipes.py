"""
Phase 8a: create the FIXED noisy evaluation conditions.

Run once, from the project root:
    python scripts/make_noisy_recipes.py

For every utterance of the dev set (dev noise pool) and the test set (test
noise pool) it draws ONE noise segment and records it in a recipe:
    data/manifests/recipes/<subset>_noise.csv
        utt_id, noise_path, start, noise_file_samples, noise_rms_dbfs

The SAME segment is used at every SNR (20, 10, 5, 0 dB). Only the SNR changes
between conditions, so the WER vs SNR curve measures the effect of the noise
LEVEL and not of a different noise draw per condition.

Recipes are tiny CSV files: the noisy audio is rebuilt identically on the fly
(on the laptop or on Kaggle) instead of storing gigabytes of WAV files. The
script is deterministic: running it again gives byte identical recipes, which
it verifies.
"""

import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.musan import NoiseBank  # noqa: E402

FIELDS = ["utt_id", "noise_path", "start", "noise_file_samples", "noise_rms_dbfs"]


def build(rows, bank, seed, file_samples):
    out = []
    for k, r in enumerate(rows):
        rng = np.random.default_rng([seed, k])
        for _ in range(6):
            _, meta = bank.sample(r["num_samples"], rng)
            if meta["accepted"]:
                break
        else:
            raise RuntimeError(f"no usable noise segment for {r['utt_id']}")
        out.append(dict(utt_id=r["utt_id"], noise_path=meta["path"], start=meta["start"],
                        noise_file_samples=file_samples[meta["path"]],
                        noise_rms_dbfs=meta["rms_dbfs"]))
    return out


def to_csv_bytes(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main():
    cfg = load_config("configs/base.yaml")
    man = resolve(cfg, "manifests_dir")
    raw = resolve(cfg, "raw_dir")
    ncfg = cfg["noise"]
    out_dir = man / "recipes"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    for subset, pool, offset in [(cfg["data"]["dev_subset"], "dev", 1),
                                 (cfg["data"]["test_subset"], "test", 2)]:
        rows = read_manifest(man / f"{subset}.csv")
        bank = NoiseBank(man / ncfg["manifest"], raw, pool=pool,
                         categories=tuple(ncfg["categories"]))
        pool_files = {f["path"] for f in bank.files}
        file_samples = {f["path"]: f["num_samples"] for f in bank.files}
        seed = ncfg["recipe_seed"] * 10 + offset

        print(f"{subset}: drawing {len(rows)} noise segments from the {pool} pool "
              f"({len(bank)} files)...")
        recipe = build(rows, bank, seed, file_samples)
        data = to_csv_bytes(recipe)
        again = to_csv_bytes(build(rows, bank, seed, file_samples))
        path = out_dir / f"{subset}_noise.csv"
        path.write_bytes(data)

        used = {x["noise_path"] for x in recipe}
        checks = [
            ("one recipe row per utterance", len(recipe) == len(rows) ==
             len({x["utt_id"] for x in recipe})),
            (f"all noise from the {pool} pool", used <= pool_files),
            ("deterministic (rebuilt recipe is byte identical)", data == again),
        ]
        for name, cond in checks:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
            ok &= cond
        levels = [x["noise_rms_dbfs"] for x in recipe]
        print(f"  {len(used)} distinct noise files used; segment RMS "
              f"{min(levels):.1f} to {max(levels):.1f} dBFS")
        print(f"  saved {path}  (md5 {hashlib.md5(data).hexdigest()})")

    print("RESULT: " + ("ALL CHECKS PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
