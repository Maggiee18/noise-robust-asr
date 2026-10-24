"""
Inference on arbitrary audio with trained checkpoints (used by the Gradio demo).

    asr = Transcriber.from_checkpoint("experiments/model_b_noise/best.pt")
    wav = prepare_audio(samples, sample_rate)          # any rate, mono or stereo
    text = asr(wav)

Everything the model needs (feature settings, architecture) is rebuilt from the
config saved inside the checkpoint, exactly as in scripts/evaluate.py, so the
demo uses the same front end and greedy CTC decoder as the experiments.

Also here:
    prepare_audio(x, sr)      -> float32 mono 16 kHz tensor in [-1, 1]
    fit_noise(noise, n)       -> loop / cut a noise signal to n samples
    word_diff(ref, hyp)       -> markdown that marks substitutions, deletions, insertions
"""

import numpy as np
import torch
import torchaudio.functional as AF
import jiwer

from src.decoder import greedy_decode
from src.features import LogMel
from src.model import CNNBiLSTM
from src.vocabulary import Vocabulary

SAMPLE_RATE = 16000


def prepare_audio(x, sr):
    """
    numpy or tensor audio, shape [N] or [N, C] (or [C, N]), int or float,
    any sample rate -> float32 mono tensor [N'] at 16 kHz, values in [-1, 1].
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("the audio is empty")
    if np.issubdtype(x.dtype, np.integer):              # int16 / int32 from a browser
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if x.ndim == 2:                                     # stereo: average the channels
        x = x.mean(axis=1) if x.shape[0] > x.shape[1] else x.mean(axis=0)
    elif x.ndim != 1:
        raise ValueError(f"unexpected audio shape {x.shape}")
    wav = torch.from_numpy(np.ascontiguousarray(x))
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, int(sr), SAMPLE_RATE)
    peak = float(wav.abs().max())
    if peak > 1.0:                                      # float input outside [-1, 1]
        wav = wav / peak
    return wav.contiguous()


def fit_noise(noise, n):
    """Loop (if shorter) or cut (if longer) a 1D noise tensor to exactly n samples."""
    if noise.numel() >= n:
        return noise[:n].clone()
    reps = int(np.ceil(n / noise.numel()))
    return noise.repeat(reps)[:n].clone()


class Transcriber:
    def __init__(self, model, features, cfg, epoch=None, device="cpu"):
        self.model, self.features, self.cfg = model.eval(), features, cfg
        self.epoch, self.device = epoch, torch.device(device)
        self.vocab = Vocabulary()

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        vocab = Vocabulary()
        model = CNNBiLSTM.from_config(cfg["model"], cfg["features"]["n_mels"], len(vocab))
        model.load_state_dict(ck["model"])
        features = LogMel(**cfg["features"])
        return cls(model.to(device), features.to(device), cfg, ck.get("epoch"), device)

    @torch.no_grad()
    def __call__(self, wav):
        """wav: float32 mono 16 kHz tensor [N] -> transcript (str).
        Mixed precision on the GPU, exactly like evaluation (src/trainer.evaluate)."""
        x = wav.to(self.device)[None]
        lens = torch.tensor([wav.numel()], device=self.device)
        amp = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp):
            feats, flens = self.features(x, lens)
            log_probs, out_lens = self.model(feats, flens)
        return greedy_decode(log_probs, out_lens, self.vocab)[0]


def word_diff(ref, hyp):
    """
    Markdown view of the word alignment (no color needed):
        correct words plain, ~~deleted~~, ~~reference~~ **substitute**, **+inserted**
    Returns (markdown, dict with wer, sub, dele, ins, ref_words).
    """
    r, h = ref.split(), hyp.split()
    if not r:
        return hyp, None
    out = jiwer.process_words(ref, hyp if hyp else "")
    parts = []
    for a in out.alignments[0]:
        rs, hs = r[a.ref_start_idx:a.ref_end_idx], h[a.hyp_start_idx:a.hyp_end_idx]
        if a.type == "equal":
            parts += rs
        elif a.type == "substitute":
            parts += [f"~~{x}~~ **{y}**" for x, y in zip(rs, hs)]
        elif a.type == "delete":
            parts += [f"~~{x}~~" for x in rs]
        elif a.type == "insert":
            parts += [f"**+{y}**" for y in hs]
    stats = dict(wer=100 * out.wer, sub=out.substitutions, dele=out.deletions,
                 ins=out.insertions, ref_words=len(r))
    return " ".join(parts), stats
