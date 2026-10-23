"""
SpecAugment (Park et al., 2019) for Model C: frequency and time masking of the
log Mel features, training batches only.

    augment = SpecAugment.from_config(cfg["specaugment"])
    feats = augment(feats, feat_lens)       # [B, 80, T] -> [B, 80, T]

It plugs into the existing hook in src/trainer.py: TrainStep(augment=...)
calls it right after feature extraction, before the model. evaluate() never
receives it, so dev and test audio are never masked.

What it does, independently for every utterance in the batch
------------------------------------------------------------
* freq_masks frequency masks: width f ~ U{0 .. freq_width} Mel bins, start
  f0 ~ U{0 .. 80 - f}; bins f0 .. f0+f-1 are masked over the whole utterance.
* time_masks time masks: width t ~ U{0 .. w_max} frames, with
  w_max = min(time_width, floor(time_ratio * T_i)), start t0 ~ U{0 .. T_i - t};
  frames t0 .. t0+t-1 are masked across all Mel bins. Masks never reach into
  the padding, and the time cap scales with the utterance's own length T_i.
* Masked values are set to 0. The features are normalized per utterance
  (src/features.py), so 0 is that utterance's mean, the usual choice; it is
  also the value the padding already has.

Design decisions (documented for the paper)
-------------------------------------------
* Frequency masking follows the paper's LibriSpeech policies: F = 27 on 80 Mel
  bins, two masks (LD policy). Each mask hides up to a third of the band.
* Time masking is milder than the paper's T = 100: our model is small, the
  budget is 20 epochs, and many LibriSpeech utterances are short (dev has
  one word utterances like "YES"). Two masks of at most 40 frames (0.4 s),
  each at most 10% of the utterance, hide at most 20% of any utterance, which
  keeps every transcript recoverable (Park et al. use the same p cap idea).
* No time warping: the paper reports it as the least important part, and it
  is the most expensive one.
* Applied on the GPU to the whole batch with vectorized masks (no Python loop
  over utterances), so it adds negligible time per step.
* Model C = Model B + SpecAugment: noise mixing stays on, so the B -> C
  comparison isolates SpecAugment alone.
* Random numbers come from torch's global generator, which train.py reseeds
  every epoch (set_seed(seed + epoch)), so runs are reproducible.
"""

import torch
from torch import nn


class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, freq_width=27, time_masks=2, time_width=40,
                 time_ratio=0.1):
        super().__init__()
        for k, v in dict(freq_masks=freq_masks, freq_width=freq_width, time_masks=time_masks,
                         time_width=time_width).items():
            if int(v) != v or v < 0:
                raise ValueError(f"{k} must be a non negative integer, got {v}")
        if not 0.0 <= time_ratio <= 1.0:
            raise ValueError(f"time_ratio must be in [0, 1], got {time_ratio}")
        self.freq_masks, self.freq_width = int(freq_masks), int(freq_width)
        self.time_masks, self.time_width = int(time_masks), int(time_width)
        self.time_ratio = float(time_ratio)

    @classmethod
    def from_config(cls, scfg):
        """None when SpecAugment is disabled (Models A and B)."""
        if not scfg or not scfg.get("enabled", False):
            return None
        return cls(**{k: v for k, v in scfg.items() if k != "enabled"})

    def extra_repr(self):
        return (f"freq_masks={self.freq_masks}, freq_width={self.freq_width}, "
                f"time_masks={self.time_masks}, time_width={self.time_width}, "
                f"time_ratio={self.time_ratio}")

    @staticmethod
    def _bands(n_masks, max_width, size, length, device):
        """
        Boolean [B, size] mask that is the union of n_masks random bands per row.
        max_width: [B] integer tensor (largest allowed width per row)
        length:    [B] integer tensor (valid positions per row; bands stay inside)
        """
        B = length.shape[0]
        pos = torch.arange(size, device=device)[None, :]
        mask = torch.zeros(B, size, dtype=torch.bool, device=device)
        for _ in range(n_masks):
            width = (torch.rand(B, device=device) * (max_width + 1).float()).long()
            width = torch.minimum(width, max_width)                 # guard rand() == 1
            start = (torch.rand(B, device=device) * (length - width + 1).float()).long()
            start = torch.minimum(start, length - width)
            mask |= (pos >= start[:, None]) & (pos < (start + width)[:, None])
        return mask

    @torch.no_grad()
    def masks(self, feats, feat_lens):
        """(freq_mask [B, F], time_mask [B, T]) for this batch."""
        B, F, T = feats.shape
        dev = feats.device
        lens = feat_lens.to(dev).long()
        f_max = torch.full((B,), min(self.freq_width, F), dtype=torch.long, device=dev)
        t_max = torch.clamp((self.time_ratio * lens.float()).floor().long(), max=self.time_width)
        freq = self._bands(self.freq_masks, f_max, F, torch.full_like(lens, F), dev)
        time = self._bands(self.time_masks, t_max, T, lens, dev)
        return freq, time

    @torch.no_grad()
    def forward(self, feats, feat_lens):
        freq, time = self.masks(feats, feat_lens)
        T = feats.shape[-1]
        valid = torch.arange(T, device=feats.device)[None, :] < feat_lens.to(feats.device)[:, None]
        mask = (freq[:, :, None] & valid[:, None, :]) | time[:, None, :]
        return feats.masked_fill(mask, 0.0)                         # new tensor; input untouched
