"""
Dataset, batching and collation.

One ASRDataset item
-------------------
    utt_id      str
    wav         float32 [N]        16 kHz waveform (clean or noisy)
    labels      int64   [L]        character ids, never the blank (0)
    transcript  str
    snr_db      float or None      None means clean

Three modes, chosen at construction
-----------------------------------
1. clean            no noise                         (Model A training, clean eval)
2. random noise     training augmentation            (Models B and C)
       with probability p_noise, mix a TRAIN pool noise segment at an SNR
       drawn from Uniform[snr_min_db, snr_max_db]. The random generator is
       seeded by (seed, epoch, index), so an item is identical no matter which
       worker loads it or in which order, and changes from epoch to epoch.
3. fixed recipe     evaluation at one SNR            (all models, dev and test)
       noise file, start sample and SNR come from a recipe CSV made once by
       scripts/make_noisy_recipes.py, so every model is scored on exactly the
       same noisy audio.

Batching
--------
DurationBatchSampler groups utterances of similar length and caps the PADDED
audio per batch (batch size x longest utterance) at max_batch_seconds. This
bounds GPU memory regardless of utterance length (dev/test go up to 35 s) and
wastes far less computation on padding than random batches.

The sampler yields (index, epoch) pairs, so the epoch reaches the dataset even
inside persistent DataLoader worker processes.

Collated batch
--------------
    wav          [B, N_max]   zero padded      wav_lens     [B]
    targets      [B, L_max]   zero padded      target_lens  [B]
    utt_ids, transcripts, snr_db   lists of length B
"""

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.audio import load_audio
from src.mixing import NoiseAdder, mix_at_snr
from src.musan import load_segment


def read_recipe(path):
    """utt_id -> dict(noise_path, start, noise_file_samples)."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["utt_id"]: dict(noise_path=r["noise_path"], start=int(r["start"]),
                                  noise_file_samples=int(r["noise_file_samples"]))
                for r in csv.DictReader(f)}


class ASRDataset(Dataset):
    def __init__(self, rows, raw_dir, vocab, *, noise_bank=None, p_noise=0.0,
                 snr_range=(0.0, 20.0), seed=0, recipe=None, snr_db=None):
        if (recipe is None) != (snr_db is None):
            raise ValueError("recipe and snr_db must be given together")
        if recipe is not None and noise_bank is not None:
            raise ValueError("use either a recipe (evaluation) or a noise bank (training)")
        if recipe is not None:
            missing = [r["utt_id"] for r in rows if r["utt_id"] not in recipe]
            if missing:
                raise ValueError(f"{len(missing)} utterances missing from recipe, e.g. {missing[:3]}")
        self.rows = rows
        self.raw_dir = Path(raw_dir)
        self.vocab = vocab
        self.adder = NoiseAdder(noise_bank) if noise_bank is not None else None
        self.p_noise = p_noise
        self.snr_range = snr_range
        self.seed = seed
        self.recipe = recipe
        self.snr_db = snr_db

    def __len__(self):
        return len(self.rows)

    def durations(self):
        return [r["duration_s"] for r in self.rows]

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        r = self.rows[index]
        wav = load_audio(self.raw_dir / r["path"])
        snr = None

        if self.recipe is not None:
            rec = self.recipe[r["utt_id"]]
            noise = torch.from_numpy(load_segment(self.raw_dir / rec["noise_path"], rec["start"],
                                                  wav.numel(), rec["noise_file_samples"]))
            wav, _ = mix_at_snr(wav, noise, self.snr_db)
            snr = float(self.snr_db)
        elif self.adder is not None:
            rng = np.random.default_rng([self.seed, epoch, index])
            if rng.random() < self.p_noise:
                snr = float(rng.uniform(*self.snr_range))
                wav, _ = self.adder(wav, snr, rng)

        labels = torch.tensor(self.vocab.encode(self.vocab.normalize(r["transcript"])),
                              dtype=torch.long)
        return dict(utt_id=r["utt_id"], wav=wav, labels=labels,
                    transcript=r["transcript"], snr_db=snr)


def collate(batch):
    B = len(batch)
    wav_lens = torch.tensor([b["wav"].numel() for b in batch], dtype=torch.long)
    target_lens = torch.tensor([b["labels"].numel() for b in batch], dtype=torch.long)
    wav = torch.zeros(B, int(wav_lens.max()))
    targets = torch.zeros(B, int(target_lens.max()), dtype=torch.long)
    for i, b in enumerate(batch):
        wav[i, :wav_lens[i]] = b["wav"]
        targets[i, :target_lens[i]] = b["labels"]
    return dict(wav=wav, wav_lens=wav_lens, targets=targets, target_lens=target_lens,
                utt_ids=[b["utt_id"] for b in batch],
                transcripts=[b["transcript"] for b in batch],
                snr_db=[b["snr_db"] for b in batch])


class DurationBatchSampler(Sampler):
    """
    Batches whose padded audio (count x longest duration) stays under
    max_batch_seconds. An utterance longer than the cap gets its own batch.

    shuffle=True (training): each epoch, durations are jittered by up to
    +/- jitter (10%) before sorting, so batch membership varies between epochs,
    and the batch order is shuffled. Seeded by (seed, epoch).
    shuffle=False (evaluation): fixed, sorted by duration.
    Call set_epoch(e) before iterating each training epoch.
    """

    def __init__(self, durations, max_batch_seconds, shuffle, seed=0, jitter=0.1):
        self.durations = np.asarray(durations, dtype=np.float64)
        self.max_batch_seconds = max_batch_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.jitter = jitter
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = np.random.default_rng([self.seed, epoch])
        key = self.durations
        if self.shuffle:
            key = key * (1 + rng.uniform(-self.jitter, self.jitter, size=len(key)))
        order = np.argsort(key, kind="stable")
        batches, current, longest = [], [], 0.0
        for i in order:
            d = self.durations[i]
            new_longest = max(longest, d)
            if current and new_longest * (len(current) + 1) > self.max_batch_seconds:
                batches.append(current)
                current, new_longest = [], d
            current.append(int(i))
            longest = new_longest
        if current:
            batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for b in self.batches:
            yield [(i, self.epoch) for i in b]


def padding_efficiency(batches, durations):
    """Fraction of padded audio that is real speech (1.0 = no padding)."""
    real = sum(durations[i] for b in batches for i in b)
    padded = sum(len(b) * max(durations[i] for i in b) for b in batches)
    return real / padded if padded else math.nan
