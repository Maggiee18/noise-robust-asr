"""
Phase 5 check: 80 band log Mel features on real LibriSpeech audio.

Run from the project root:
    python scripts/check_features.py

Checks
  1. frame count formula T = floor(N / 160) + 1 for edge case lengths
  2. single utterance shapes ([1, 80, T]) and frame coverage of the audio
  3. batching: each utterance gives identical features alone and padded
     inside a batch; padded frames are exactly 0
  4. normalization: every Mel bin has mean ~0 and std ~1 over valid frames
  5. a silent (all zero) input gives finite values (no NaN / inf)
  6. GPU and CPU agree (if a GPU is present), plus timing for a realistic batch
Saves results/plots/phase5_logmel.png (waveform, raw log Mel, normalized).
"""

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.audio import load_audio  # noqa: E402
from src.features import HOP_LENGTH, N_MELS, SAMPLE_RATE, LogMel, num_frames  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def pad_batch(wavs):
    lens = torch.tensor([w.numel() for w in wavs])
    out = torch.zeros(len(wavs), int(lens.max()))
    for i, w in enumerate(wavs):
        out[i, :w.numel()] = w
    return out, lens


def main():
    torch.manual_seed(0)
    fe = LogMel()

    print("=" * 70 + "\n1. FRAME COUNT FORMULA\n" + "=" * 70)
    for n in [1, 159, 160, 161, 400, 15999, 16000, 16001]:
        T = fe(torch.randn(n))[0].shape[-1]
        check(f"N={n:>5} samples -> T={T}", T == num_frames(n), f"formula {num_frames(n)}")

    print("\n" + "=" * 70 + "\n2. REAL UTTERANCES (dev-clean)\n" + "=" * 70)
    rows = read_manifest(ROOT / "data" / "manifests" / "dev-clean.csv")
    picks = random.Random(0).sample(rows, 4)
    wavs = [load_audio(ROOT / "data" / "raw" / r["path"]) for r in picks]
    singles = []
    for r, w in zip(picks, wavs):
        f, fl = fe(w)
        singles.append(f[0])
        print(f"{r['utt_id']}: waveform {tuple(w.shape)} -> features {tuple(f.shape)}"
              f"  ({fl.item()} frames x 10 ms = {fl.item() / 100:.2f} s, audio {r['duration_s']:.2f} s)")
        check("  shape [1, 80, T] with T = N//160 + 1",
              f.shape == (1, N_MELS, num_frames(w.numel())))
        # centered STFT: frames cover the audio plus at most one extra hop
        n = w.numel()
        check("  frames cover the audio: N/160 < T <= N/160 + 1",
              n < fl.item() * HOP_LENGTH <= n + HOP_LENGTH)

    print("\n" + "=" * 70 + "\n3. BATCHING\n" + "=" * 70)
    batch, lens = pad_batch(wavs)
    bf, bl = fe(batch, lens)
    print(f"padded waveforms [B, N] = {tuple(batch.shape)}, lengths {lens.tolist()}")
    print(f"batch features [B, 80, T] = {tuple(bf.shape)}, lengths {bl.tolist()}")
    max_diff, pad_ok = 0.0, True
    for i, s in enumerate(singles):
        T = s.shape[-1]
        max_diff = max(max_diff, (bf[i, :, :T] - s).abs().max().item())
        pad_ok &= bool((bf[i, :, T:] == 0).all())
    check("each utterance identical alone vs in batch", max_diff < 1e-4,
          f"max abs difference {max_diff:.2e}")
    check("padded frames are exactly 0", pad_ok)

    print("\n" + "=" * 70 + "\n4. NORMALIZATION\n" + "=" * 70)
    worst_mean, worst_std = 0.0, 0.0
    for i in range(len(wavs)):
        valid = bf[i, :, :bl[i]]
        worst_mean = max(worst_mean, valid.mean(-1).abs().max().item())
        worst_std = max(worst_std, (valid.std(-1, unbiased=False) - 1).abs().max().item())
    check("per bin mean ~ 0 over valid frames", worst_mean < 1e-3, f"worst |mean| {worst_mean:.1e}")
    check("per bin std ~ 1 over valid frames", worst_std < 1e-2, f"worst |std-1| {worst_std:.1e}")
    raw = LogMel(normalize=False)(wavs[0])[0][0]          # [80, T]
    print(f"raw log Mel range for {picks[0]['utt_id']}: "
          f"{raw.min().item():.1f} to {raw.max().item():.1f}")

    print("\n" + "=" * 70 + "\n5. SILENCE\n" + "=" * 70)
    sf, _ = fe(torch.zeros(SAMPLE_RATE))
    check("all zero input gives finite features", bool(torch.isfinite(sf).all()))

    print("\n" + "=" * 70 + "\n6. DEVICE\n" + "=" * 70)
    if torch.cuda.is_available():
        gfe = LogMel().cuda()
        gf, _ = gfe(batch.cuda(), lens.cuda())
        diff = (gf.cpu() - bf).abs().max().item()
        check("GPU matches CPU", diff < 1e-3, f"max abs difference {diff:.2e}")
        big = torch.randn(16, 15 * SAMPLE_RATE, device="cuda")
        big_lens = torch.full((16,), big.shape[1], device="cuda")
        gfe(big, big_lens)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            gfe(big, big_lens)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"batch of 16 x 15 s on GPU: {ms:.1f} ms")
    else:
        print("[SKIP] no GPU")

    # ---- figure
    w, r = wavs[0], picks[0]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7))
    t = torch.arange(w.numel()) / SAMPLE_RATE
    ax[0].plot(t.numpy(), w.numpy(), linewidth=0.4)
    ax[0].set_xlim(0, t[-1].item())
    ax[0].set_title(f"{r['utt_id']}: waveform", fontsize=9)
    im1 = ax[1].imshow(raw.numpy(), origin="lower", aspect="auto", cmap="magma",
                       extent=[0, raw.shape[-1] / 100, 0, N_MELS])
    ax[1].set_title("log Mel (80 bands), before normalization", fontsize=9)
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(singles[0].numpy(), origin="lower", aspect="auto", cmap="magma",
                       extent=[0, singles[0].shape[-1] / 100, 0, N_MELS])
    ax[2].set_title("after per utterance normalization (model input)", fontsize=9)
    fig.colorbar(im2, ax=ax[2])
    for a in ax[1:]:
        a.set_ylabel("Mel bin")
    ax[2].set_xlabel("time (s)")
    out = ROOT / "results" / "plots" / "phase5_logmel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nFigure: {out}")
    print("RESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
