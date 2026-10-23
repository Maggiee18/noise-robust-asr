"""
SNR controlled noise mixing.

Definition used throughout the project
--------------------------------------
    P(x)      = mean(x^2) over the whole utterance          (signal power)
    SNR_dB    = 10 * log10( P(speech) / P(noise_scaled) )
    gain g    = sqrt( P(speech) / ( P(noise) * 10^(SNR_dB / 10) ) )
    mixture   = speech + g * noise

So the noise is scaled to hit the requested SNR exactly; the speech is never
rescaled relative to the noise.

Design decisions (documented for the paper)
-------------------------------------------
* Global (whole utterance) power for both signals. It is the most common
  definition in augmentation work and needs no voice activity detector.
  LibriSpeech utterances are segmented with little leading or trailing
  silence, so the global level is close to the active speech level.
  Consequence: the SNR describes the utterance as a whole. A noise that is
  only present for part of the utterance is locally louder than the nominal
  SNR while it plays (and absent elsewhere), which is realistic.
* Power is computed in float64 so that the achieved SNR matches the target to
  well under 0.01 dB.
* Clipping protection: if the mixture's peak exceeds 0.99, the WHOLE mixture
  (speech and noise together) is scaled down. This keeps the SNR exactly
  unchanged and keeps the mixture safe to save as 16 bit audio. The absolute
  level does not matter to the model because Phase 5 normalizes each
  utterance's features.
* A speech or noise signal with power below MIN_POWER (-100 dBFS) raises an
  error instead of producing an undefined or exploding gain.
"""

import math

import torch

MIN_POWER = 1e-10          # -100 dBFS
PEAK_LIMIT = 0.99


def power(x):
    """Mean square in float64 (as a Python float)."""
    x = x.detach().double() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float64)
    return float((x * x).mean())


def snr_db(clean, noise):
    """SNR in dB of a clean signal against an additive noise signal."""
    return 10 * math.log10(power(clean) / power(noise))


def mix_at_snr(speech, noise, target_snr_db, peak_limit=PEAK_LIMIT):
    """
    speech, noise: 1D float tensors of equal length.
    Returns (mixture float32 [N], info dict).
    info: target_snr_db, achieved_snr_db, noise_gain, output_scale,
          speech_power_db, noise_power_db
    """
    if speech.dim() != 1 or noise.dim() != 1:
        raise ValueError("speech and noise must be 1D tensors")
    if speech.numel() != noise.numel():
        raise ValueError(f"length mismatch: speech {speech.numel()}, noise {noise.numel()}")
    ps, pn = power(speech), power(noise)
    if ps < MIN_POWER:
        raise ValueError(f"speech is (near) silent: power {ps:.2e}")
    if pn < MIN_POWER:
        raise ValueError(f"noise is (near) silent: power {pn:.2e}")

    gain = math.sqrt(ps / (pn * 10 ** (target_snr_db / 10)))
    s = speech.double()
    n = noise.double() * gain
    mix = s + n

    peak = float(mix.abs().max())
    scale = peak_limit / peak if peak > peak_limit else 1.0
    mix = mix * scale

    info = dict(
        target_snr_db=float(target_snr_db),
        achieved_snr_db=round(snr_db(s * scale, n * scale), 4),
        noise_gain=gain,
        output_scale=scale,
        speech_power_db=round(10 * math.log10(ps), 2),
        noise_power_db=round(10 * math.log10(pn), 2),
    )
    return mix.float(), info


class NoiseAdder:
    """
    Draw a noise segment from a NoiseBank pool and mix it at a given SNR.

        adder = NoiseAdder(NoiseBank(..., pool="train"))
        noisy, meta = adder(speech, snr_db=10.0, rng=np.random.default_rng(seed))

    If the bank cannot find a non silent segment (meta['accepted'] False) it
    redraws up to `redraws` times; if that still fails it raises, so near
    silence is never amplified by accident.
    """

    def __init__(self, bank, redraws=5):
        self.bank = bank
        self.redraws = redraws

    def __call__(self, speech, snr_db, rng):
        for _ in range(self.redraws + 1):
            noise, nmeta = self.bank.sample(speech.numel(), rng)
            if nmeta["accepted"]:
                break
        else:
            raise RuntimeError("could not draw a non silent noise segment; "
                               f"last: {nmeta}")
        mixture, info = mix_at_snr(speech, noise, snr_db)
        return mixture, {**nmeta, **info}

