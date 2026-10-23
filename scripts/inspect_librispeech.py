"""
Phase 3 check: actually load a few utterances and look at them.

Run from the project root (after build_manifests.py):
    python scripts/inspect_librispeech.py
    python scripts/inspect_librispeech.py --subset train-clean-100 --n 5

Picks N utterances with a fixed seed, loads the audio with src.audio.load_audio
and checks that the decoded length matches the manifest. Prints shape, dtype,
peak and loudness, and saves a waveform figure to
results/plots/phase3_waveforms_<subset>.png.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.audio import SAMPLE_RATE, load_audio  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev-clean")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    rows = read_manifest(args.data_dir / "manifests" / f"{args.subset}.csv")
    picks = random.Random(args.seed).sample(rows, args.n)
    ok = True

    fig, axes = plt.subplots(args.n, 1, figsize=(10, 2.2 * args.n), squeeze=False)
    for ax, r in zip(axes[:, 0], picks):
        wav = load_audio(args.data_dir / "raw" / r["path"])
        peak = wav.abs().max().item()
        rms = wav.pow(2).mean().sqrt().item()
        dbfs = 20 * torch.log10(torch.tensor(max(rms, 1e-12))).item()
        match = wav.numel() == r["num_samples"]
        ok &= match and wav.dtype == torch.float32 and wav.dim() == 1 and peak <= 1.0
        print(f"\n{r['utt_id']}  (speaker {r['speaker_id']})")
        print(f"  tensor  shape {tuple(wav.shape)}  dtype {wav.dtype}  "
              f"{wav.numel() / SAMPLE_RATE:.2f} s")
        print(f"  length matches manifest: {match}")
        print(f"  peak {peak:.3f}   RMS {dbfs:.1f} dBFS")
        print(f"  text: {r['transcript']}")
        t = torch.arange(wav.numel()) / SAMPLE_RATE
        ax.plot(t.numpy(), wav.numpy(), linewidth=0.4)
        ax.set_title(f"{r['utt_id']}: {r['transcript'][:70]}", fontsize=8)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylim(-1, 1)

    out = ROOT / "results" / "plots" / f"phase3_waveforms_{args.subset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nFigure: {out}")
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
