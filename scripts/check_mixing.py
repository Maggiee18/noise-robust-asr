"""
Phase 7 check: SNR controlled mixing.

Run from the project root (after Phases 3 and 6):
    python scripts/check_mixing.py

Checks
  1. exact SNR on synthetic signals, -5 to 30 dB (error < 0.01 dB), measured
     independently from (mixture - speech)
  2. clipping protection: peak <= 0.99 and SNR unchanged
  3. errors for silent speech, silent noise and length mismatch
  4. real dev-clean speech + DEV pool MUSAN noise at 20/10/5/0 dB:
     achieved SNR, finite output, exact length, never clipped
  5. determinism: same seed -> identical mixture
  6. the corruption is graded: the log Mel features move further from the
     clean features as SNR drops (20 dB < 10 dB < 5 dB < 0 dB)
Saves listening examples to results/audio/ and a spectrogram figure to
results/plots/phase7_snr_examples.png.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from src.audio import load_audio  # noqa: E402
from src.features import LogMel  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.mixing import NoiseAdder, mix_at_snr, snr_db  # noqa: E402
from src.musan import NoiseBank  # noqa: E402

SNRS = [20, 10, 5, 0]
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    torch.manual_seed(0)
    print("=" * 70 + "\n1. EXACT SNR (synthetic)\n" + "=" * 70)
    t = torch.arange(32000) / 16000
    speech = 0.1 * torch.sin(2 * torch.pi * 220 * t)
    noise = 0.05 * torch.randn(32000)
    for target in [-5, 0, 5, 10, 20, 30]:
        mix, info = mix_at_snr(speech, noise, target)
        s = speech.double() * info["output_scale"]
        measured = snr_db(s, mix.double() - s)
        check(f"target {target:>3} dB", abs(measured - target) < 0.01,
              f"measured {measured:.4f} dB from (mixture - speech)")

    print("\n" + "=" * 70 + "\n2. CLIPPING PROTECTION\n" + "=" * 70)
    loud = 0.9 * torch.sin(2 * torch.pi * 220 * t)
    mix, info = mix_at_snr(loud, torch.randn(32000), 0)
    check("peak limited to 0.99", mix.abs().max().item() <= 0.99 + 1e-6,
          f"peak {mix.abs().max().item():.3f}, output scale {info['output_scale']:.3f}")
    check("SNR unchanged by peak limiting", abs(info["achieved_snr_db"]) < 0.01,
          f"{info['achieved_snr_db']} dB")

    print("\n" + "=" * 70 + "\n3. ERROR HANDLING\n" + "=" * 70)
    for name, a, b in [("silent speech", torch.zeros(100), torch.randn(100)),
                       ("silent noise", torch.randn(100), torch.zeros(100)),
                       ("length mismatch", torch.randn(100), torch.randn(99))]:
        try:
            mix_at_snr(a, b, 10)
            check(f"{name} raises an error", False)
        except ValueError as e:
            check(f"{name} raises an error", True, str(e))

    print("\n" + "=" * 70 + "\n4. REAL SPEECH + DEV POOL NOISE\n" + "=" * 70)
    rows = read_manifest(ROOT / "data" / "manifests" / "dev-clean.csv")
    picks = random.Random(1).sample(rows, 10)
    bank = NoiseBank(ROOT / "data" / "manifests" / "musan.csv", ROOT / "data" / "raw",
                     pool="dev", categories=("noise",))
    adder = NoiseAdder(bank)
    rng = np.random.default_rng(0)
    errs, ok_len, ok_fin, ok_peak, scaled = [], True, True, True, 0
    for r in picks:
        wav = load_audio(ROOT / "data" / "raw" / r["path"])
        for snr in SNRS:
            noisy, meta = adder(wav, snr, rng)
            errs.append(abs(meta["achieved_snr_db"] - snr))
            ok_len &= noisy.shape == wav.shape
            ok_fin &= bool(torch.isfinite(noisy).all())
            ok_peak &= noisy.abs().max().item() <= 0.99 + 1e-6
            scaled += meta["output_scale"] < 1
    check("achieved SNR within 0.01 dB (40 mixtures)", max(errs) < 0.01,
          f"worst error {max(errs):.5f} dB")
    check("mixture length equals speech length", ok_len)
    check("mixtures finite", ok_fin)
    check("mixtures never exceed 0.99", ok_peak, f"{scaled}/40 needed peak limiting")

    print("\n" + "=" * 70 + "\n5. DETERMINISM\n" + "=" * 70)
    wav = load_audio(ROOT / "data" / "raw" / picks[0]["path"])
    a, ma = adder(wav, 5, np.random.default_rng(42))
    b, mb = adder(wav, 5, np.random.default_rng(42))
    check("same seed gives identical mixture", torch.equal(a, b) and ma["path"] == mb["path"],
          f"noise {Path(ma['path']).name} @ {ma['start']}")

    print("\n" + "=" * 70 + "\n6. GRADED CORRUPTION (features)\n" + "=" * 70)
    fe = LogMel()
    clean_f = fe(wav)[0]
    dists, mixes = [], {}
    for snr in SNRS:
        noisy, _ = mix_at_snr(wav, a_noise(bank, wav), snr)
        mixes[snr] = noisy
        dists.append((fe(noisy)[0] - clean_f).abs().mean().item())
    print("mean |noisy - clean| in normalized log Mel: " +
          ", ".join(f"{s} dB: {d:.3f}" for s, d in zip(SNRS, dists)))
    check("distance grows as SNR drops", all(x < y for x, y in zip(dists, dists[1:])))

    # ---- listening examples and figure
    out_audio = ROOT / "results" / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)
    uid = picks[0]["utt_id"]
    sf.write(out_audio / f"phase7_{uid}_clean.wav", wav.numpy(), 16000, subtype="PCM_16")
    for snr, m in mixes.items():
        sf.write(out_audio / f"phase7_{uid}_{snr:02d}dB.wav", m.numpy(), 16000, subtype="PCM_16")
    fe_raw = LogMel(normalize=False)
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    panels = [("clean", wav)] + [(f"{s} dB SNR", mixes[s]) for s in SNRS]
    for axis, (title, x) in zip(ax, panels):
        mel = fe_raw(x)[0][0].numpy()
        axis.imshow(mel, origin="lower", aspect="auto", cmap="magma", vmin=-12, vmax=6,
                    extent=[0, mel.shape[-1] / 100, 0, 80])
        axis.set_title(f"{uid}: {title}", fontsize=9)
        axis.set_ylabel("Mel bin")
    ax[-1].set_xlabel("time (s)")
    out = ROOT / "results" / "plots" / "phase7_snr_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nAudio:  {out_audio / f'phase7_{uid}_*.wav'}")
    print(f"Figure: {out}")
    print(f"Transcript: {picks[0]['transcript']}")
    print("RESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


_NOISE_CACHE = {}


def a_noise(bank, wav):
    """One fixed noise segment (same for every SNR) so only the SNR varies."""
    if wav.numel() not in _NOISE_CACHE:
        seg, meta = bank.sample(wav.numel(), np.random.default_rng(5))
        _NOISE_CACHE[wav.numel()] = seg
    return _NOISE_CACHE[wav.numel()]


if __name__ == "__main__":
    main()
