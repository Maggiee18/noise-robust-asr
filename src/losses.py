"""
CTC loss.

Conventions (fixed for the whole project)
-----------------------------------------
* The model outputs BATCH FIRST log probabilities [B, T', C].
  torch.nn.CTCLoss wants [T', B, C]; the transpose happens here and only here.
* blank = 0 (src/vocabulary.py). Targets never contain the blank.
* Targets arrive zero padded [B, L_max] with target_lens [B]; PyTorch reads
  only the first target_lens[b] entries of row b, so padding is ignored.
* Computed in float32 even under mixed precision.

Normalization
-------------
    loss = sum over utterances of CTC(utterance) / total number of target characters

i.e. the average negative log likelihood PER CHARACTER over the batch. With
duration batching the batch size varies (1 to 86 utterances), so a per batch
sum would change scale from step to step; dividing by the character count
keeps the loss on the same scale for every batch and every experiment.

Safety
------
CTC is impossible when an utterance has fewer output frames than
(characters + repeated adjacent characters); PyTorch would then return inf.
Phase 9 verified this never happens for our data, so instead of silently
zeroing such losses (zero_infinity=True), check_lengths() raises an error
naming the utterance: an impossible example means a bug, not noise.

Note for the paper: the CUDA backward pass of CTC uses atomic additions and is
not bit for bit deterministic; repeated GPU runs can differ in the last digits.
"""

import torch
from torch import nn


def required_frames(targets, target_lens):
    """Minimum output frames per utterance: length + number of adjacent repeats."""
    need = []
    for t, n in zip(targets, target_lens):
        t = t[:n]
        repeats = int((t[1:] == t[:-1]).sum()) if n > 1 else 0
        need.append(int(n) + repeats)
    return torch.tensor(need)


def check_lengths(out_lens, targets, target_lens, utt_ids=None):
    need = required_frames(targets.cpu(), target_lens.cpu())
    bad = (out_lens.cpu() < need).nonzero().flatten().tolist()
    if bad:
        k = bad[0]
        who = utt_ids[k] if utt_ids else f"batch item {k}"
        raise ValueError(f"CTC impossible for {len(bad)} utterance(s), e.g. {who}: "
                         f"{int(out_lens[k])} output frames < {int(need[k])} required")


class CTCLoss(nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank
        self.ctc = nn.CTCLoss(blank=blank, reduction="sum", zero_infinity=False)

    def forward(self, log_probs, out_lens, targets, target_lens):
        """
        log_probs:   [B, T', C] batch first log probabilities (from the model)
        out_lens:    [B]  valid output frames
        targets:     [B, L_max] zero padded character ids
        target_lens: [B]
        returns      scalar: negative log likelihood per target character
        """
        lp = log_probs.float().transpose(0, 1)          # [T', B, C]
        total = self.ctc(lp, targets, out_lens, target_lens)
        loss = total / target_lens.sum().clamp(min=1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non finite CTC loss: {loss.item()}")
        return loss
