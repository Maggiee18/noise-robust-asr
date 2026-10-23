"""
Training and evaluation building blocks shared by the overfit test (Phase 12)
and the full experiments (Phases 13 to 15).

    set_seed(seed)
    step = TrainStep(model, features, optimizer, clip_norm, use_amp)
    loss, grad_norm = step(batch, device)            # one optimizer update
    out = evaluate(model, features, batches, device, vocab)   # loss + transcripts

Mixed precision: the model runs under float16 autocast on the GPU; features,
log_softmax and the CTC loss stay in float32 (see features.py / model.py /
losses.py). GradScaler protects small float16 gradients from underflow.
Gradients are clipped to clip_norm (global L2 norm) after unscaling, which
guards the LSTM against occasional exploding gradients.
"""

import random

import numpy as np
import torch

from src.decoder import blank_fraction, greedy_decode
from src.losses import CTCLoss, check_lengths


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward(model, features, batch, device, augment=None):
    feats, feat_lens = features(batch["wav"].to(device, non_blocking=True),
                                batch["wav_lens"].to(device))
    if augment is not None:                      # SpecAugment (Phase 15), training only
        feats = augment(feats, feat_lens)
    log_probs, out_lens = model(feats, feat_lens)
    return log_probs, out_lens


class TrainStep:
    def __init__(self, model, features, optimizer, clip_norm=5.0, use_amp=True,
                 scheduler=None, augment=None):
        self.model, self.features, self.opt = model, features, optimizer
        self.scheduler, self.augment = scheduler, augment
        self.clip_norm = clip_norm
        self.device_type = next(model.parameters()).device.type
        self.use_amp = use_amp and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)
        self.crit = CTCLoss()
        self.skipped = 0                         # optimizer updates skipped by GradScaler

    def __call__(self, batch, device):
        self.model.train()
        with torch.autocast(self.device_type, dtype=torch.float16, enabled=self.use_amp):
            log_probs, out_lens = forward(self.model, self.features, batch, device, self.augment)
        check_lengths(out_lens, batch["targets"], batch["target_lens"], batch["utt_ids"])
        loss = self.crit(log_probs, out_lens, batch["targets"].to(device),
                         batch["target_lens"].to(device))
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.opt)              # skipped automatically if grads are inf/nan
        self.scaler.update()
        # A float16 overflow makes GradScaler skip the update and lower its scale
        # (normal, mostly in the first steps). The schedule only advances when the
        # optimizer really stepped; skipped updates are counted for the log.
        skipped = self.use_amp and self.scaler.get_scale() < scale_before
        self.skipped += int(skipped)
        if self.scheduler is not None and not skipped:
            self.scheduler.step()
        return loss.item(), grad_norm.item()


@torch.no_grad()
def evaluate(model, features, batches, device, vocab, use_amp=True):
    """Loss per character, transcripts and blank fraction over a list/loader of batches."""
    model.eval()
    crit = CTCLoss()
    amp = use_amp and device.type == "cuda"
    total_loss, total_chars, blanks, frames = 0.0, 0, 0.0, 0
    hyps, refs, ids = [], [], []
    for batch in batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            log_probs, out_lens = forward(model, features, batch, device)
        n_chars = int(batch["target_lens"].sum())
        total_loss += crit(log_probs, out_lens, batch["targets"].to(device),
                           batch["target_lens"].to(device)).item() * n_chars
        total_chars += n_chars
        n_frames = int(out_lens.sum())
        blanks += blank_fraction(log_probs, out_lens, vocab.blank_id) * n_frames
        frames += n_frames
        hyps += greedy_decode(log_probs, out_lens, vocab)
        refs += batch["transcripts"]
        ids += batch["utt_ids"]
    return dict(loss=total_loss / max(total_chars, 1), hyps=hyps, refs=refs, utt_ids=ids,
                blank_fraction=blanks / max(frames, 1))
