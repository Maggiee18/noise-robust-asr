"""
MUSAN noise: manifest, train/dev/test pools, and a sampler for noise segments.

Layout: data/raw/musan/<category>/<source>/<file>.wav
    category: noise | music | speech      (all 16 kHz mono WAV)

Design decisions (documented for the paper)
-------------------------------------------
* Disjoint noise pools. Every usable file goes to exactly one pool:
      train 70%  -> mixed into training audio (Models B and C)
      dev   10%  -> mixed into the validation set
      test  20%  -> mixed into the test set
  The test set is therefore corrupted by noise recordings that no model ever
  heard in training, so noise robustness is measured on unseen noise.
* The split is stratified by category AND source (e.g. noise/free-sound,
  noise/sound-bible) and is deterministic without any random generator:
  files are ordered by a SHA1 hash of their relative path and cut at 70/80%.
  The same files land in the same pool on every machine.
* Files are excluded (and listed in the manifest with the reason) if they are
  shorter than MIN_DURATION_S (looping a very short clip produces an
  artificial periodic buzz rather than realistic noise) or quieter than
  MIN_FILE_RMS_DB (near silent files cannot be scaled to a target SNR).
* Sampling picks a file uniformly at random from the pool, then a random
  segment of the requested length. Files shorter than the request are looped.
  Segments quieter than MIN_SEGMENT_RMS_DB (e.g. a silent stretch of an
  impulsive recording) are redrawn, because Phase 7 would otherwise amplify
  near silence by a huge factor to reach the target SNR.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
CATEGORIES = ("noise", "music", "speech")
POOLS = ("train", "dev", "test")
SPLIT = (0.70, 0.10, 0.20)
MIN_DURATION_S = 1.0
MIN_FILE_RMS_DB = -60.0
MIN_SEGMENT_RMS_DB = -50.0

MANIFEST_FIELDS = ["category", "source", "path", "sample_rate", "num_samples",
                   "duration_s", "rms_dbfs", "peak", "pool", "exclude_reason"]


def rms_db(x):
    x = np.asarray(x, dtype=np.float64)
    return float(10 * np.log10(np.mean(x * x) + 1e-20))


def file_stats(path, block=16000 * 60):
    """Read a whole file in blocks; return sr, frames, channels, rms dBFS, peak."""
    info = sf.info(str(path))
    sq, peak, n = 0.0, 0.0, 0
    for blk in sf.blocks(str(path), blocksize=block, dtype="float32", always_2d=True):
        sq += float(np.square(blk, dtype=np.float64).sum())
        peak = max(peak, float(np.abs(blk).max()) if blk.size else 0.0)
        n += blk.size
    rms = 10 * np.log10(sq / max(n, 1) + 1e-20)
    return info.samplerate, info.frames, info.channels, float(rms), peak


def _hash(rel_path):
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()


def assign_pools(rows):
    """Set row['pool'] for usable rows: stratified by (category, source),
    ordered by path hash, cut at 70% / 80%. Excluded rows keep pool='excluded'."""
    groups = {}
    for r in rows:
        if r["pool"] != "excluded":
            groups.setdefault((r["category"], r["source"]), []).append(r)
    for members in groups.values():
        members.sort(key=lambda r: _hash(r["path"]))
        n = len(members)
        cut1 = round(n * SPLIT[0])
        cut2 = round(n * (SPLIT[0] + SPLIT[1]))
        for i, r in enumerate(members):
            r["pool"] = "train" if i < cut1 else ("dev" if i < cut2 else "test")
    return rows


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["sample_rate"] = int(r["sample_rate"])
        r["num_samples"] = int(r["num_samples"])
        r["duration_s"] = float(r["duration_s"])
        r["rms_dbfs"] = float(r["rms_dbfs"])
        r["peak"] = float(r["peak"])
    return rows


def load_segment(path, start, num_samples, file_num_samples):
    """
    Read num_samples of noise beginning at `start`. Files shorter than the
    request are looped. Used both by NoiseBank (random segments) and by the
    fixed evaluation recipes (stored path + start), so both read identically.
    Returns a float32 numpy array.
    """
    n = file_num_samples
    if n >= num_samples:
        x, _ = sf.read(str(path), start=start, frames=num_samples, dtype="float32")
    else:
        whole, _ = sf.read(str(path), dtype="float32")
        reps = int(np.ceil((num_samples + start) / n))
        x = np.tile(whole, reps)[start:start + num_samples]
    if x.shape[0] != num_samples:
        raise ValueError(f"{path}: read {x.shape[0]} samples, expected {num_samples}")
    return x


class NoiseBank:
    """
    Draws noise segments of a requested length from ONE pool.

        bank = NoiseBank("data/manifests/musan.csv", "data/raw", pool="train",
                         categories=("noise",))
        rng = np.random.default_rng(seed)
        seg, meta = bank.sample(num_samples, rng)    # seg: float32 tensor [num_samples]
    """

    def __init__(self, manifest, raw_dir, pool, categories=("noise",),
                 min_segment_rms_db=MIN_SEGMENT_RMS_DB, max_tries=10):
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}")
        self.raw_dir = Path(raw_dir)
        self.pool = pool
        self.files = [r for r in read_manifest(manifest)
                      if r["pool"] == pool and r["category"] in categories]
        if not self.files:
            raise ValueError(f"no files for pool={pool} categories={categories}")
        self.min_segment_rms_db = min_segment_rms_db
        self.max_tries = max_tries

    def __len__(self):
        return len(self.files)

    def _segment(self, r, num_samples, rng):
        n = r["num_samples"]
        hi = n - num_samples + 1 if n >= num_samples else n
        start = int(rng.integers(0, hi))
        return load_segment(self.raw_dir / r["path"], start, num_samples, n), start

    def sample(self, num_samples, rng):
        """
        Return (float32 tensor [num_samples], metadata dict).

        Each attempt draws a new file and a new segment. The first segment at
        or above min_segment_rms_db is returned with meta['accepted'] = True.
        If all max_tries attempts are too quiet, the LOUDEST one is returned
        with meta['accepted'] = False, so the caller (Phase 7 mixing) can
        decide what to do instead of silently amplifying near silence.
        """
        best = None
        for attempt in range(1, self.max_tries + 1):
            r = self.files[int(rng.integers(0, len(self.files)))]
            x, start = self._segment(r, num_samples, rng)
            level = rms_db(x)
            if best is None or level > best[2]:
                best = (x, dict(path=r["path"], category=r["category"], start=start,
                                rms_dbfs=round(level, 2)), level)
            if level >= self.min_segment_rms_db:
                break
        x, meta, level = best
        meta["attempts"] = attempt
        meta["accepted"] = level >= self.min_segment_rms_db
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), meta
