"""
Audio loading shared by every phase.

We read audio with soundfile (libsndfile), not torchaudio.load, because since
torchaudio 2.9 torchaudio.load depends on TorchCodec/FFmpeg. LibriSpeech is FLAC
and MUSAN is WAV; both are 16 kHz mono, which load_audio enforces by default.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000


def audio_info(path):
    """Header only read (fast): returns (sample_rate, num_frames, channels)."""
    info = sf.info(str(path))
    return info.samplerate, info.frames, info.channels


def load_audio(path, expected_sr=SAMPLE_RATE):
    """
    Load an audio file as a float32 torch tensor of shape [num_samples],
    values in [-1, 1].

    Raises ValueError if the file is not mono or not at expected_sr, instead of
    silently resampling: for LibriSpeech and MUSAN both conditions should
    always hold, so a violation means something is wrong with the data.
    Resampling for arbitrary user audio is handled later in the demo.
    """
    data, sr = sf.read(str(Path(path)), dtype="float32", always_2d=True)  # [N, C]
    if data.shape[1] != 1:
        raise ValueError(f"{path}: expected mono, got {data.shape[1]} channels")
    if expected_sr is not None and sr != expected_sr:
        raise ValueError(f"{path}: expected {expected_sr} Hz, got {sr} Hz")
    return torch.from_numpy(np.ascontiguousarray(data[:, 0]))
