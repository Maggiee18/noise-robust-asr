"""
80 band log Mel feature extraction.

Pipeline (one batch at a time, on the GPU if available):

    waveform     [B, N]          float32, 16 kHz, zero padded to the longest
    wav_lens     [B]             true number of samples per utterance
        |  STFT: 25 ms Hann window (400), 10 ms hop (160), n_fft 512
        |  power spectrum -> 80 triangular Mel filters (0 to 8000 Hz, HTK scale)
        |  log(mel + 1e-6)
        |  per utterance mean/variance normalization, per Mel bin,
        |  computed over that utterance's VALID frames only
        v
    features     [B, 80, T]      padded frames set to 0
    feat_lens    [B]             T_i = floor(N_i / 160) + 1

Design decisions (documented for the paper)
-------------------------------------------
* Computed on the fly, per batch. Noise is mixed into the waveform with a new
  random noise clip and SNR every time a training utterance is used, so
  features cannot be precomputed. Doing it on the GPU is much faster than in
  DataLoader workers (which are slow to start on Windows).
* n_fft 512 with a 400 sample (25 ms) window, as in Kaldi and ESPnet. With
  n_fft 400 the FFT bins are 40 Hz apart and 9 of the narrow low frequency Mel
  filters cover a single bin each; zero padding to 512 (31.25 Hz bins) reduces
  that to 3 without changing time resolution or the frame count.
* Zero padding at the signal edges (pad_mode="constant") instead of the
  default reflection. With zero padding, an utterance gives EXACTLY the same
  features alone or inside a padded batch, which check_features.py verifies.
* Per utterance normalization rather than global dataset statistics. It needs
  no statistics file (which would differ between clean and noisy training),
  removes level and channel differences between recordings, and treats every
  experiment identically.
* Always computed in float32, even when the model later runs in mixed
  precision: the log of very small Mel energies is not reliable in float16.
* Output [B, 80, T]. The CNN (Phase 9) adds a channel axis: [B, 1, 80, T].
"""

import torch
import torchaudio

SAMPLE_RATE = 16000
N_FFT = 512        # 25 ms window zero padded to 512 (31.25 Hz bins)
WIN_LENGTH = 400   # 25 ms
HOP_LENGTH = 160   # 10 ms  ->  100 frames per second
N_MELS = 80
LOG_EPS = 1e-6
NORM_EPS = 1e-5


def num_frames(num_samples, hop_length=HOP_LENGTH):
    """Frames produced by a centered STFT: floor(N / hop) + 1. Works on ints
    and on integer tensors."""
    return num_samples // hop_length + 1


class LogMel(torch.nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
                 hop_length=HOP_LENGTH, n_mels=N_MELS, normalize=True):
        super().__init__()
        self.hop_length = hop_length
        self.normalize = normalize
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, f_min=0.0, f_max=sample_rate / 2, n_mels=n_mels,
            window_fn=torch.hann_window, power=2.0, center=True,
            pad_mode="constant", mel_scale="htk", norm=None)

    @torch.no_grad()
    def forward(self, wav, wav_lens=None):
        """
        wav:      [B, N] (or [N] for a single utterance)
        wav_lens: [B] number of valid samples; defaults to N for every row
        returns   (features [B, n_mels, T], feat_lens [B])
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        B, N = wav.shape
        if wav_lens is None:
            wav_lens = torch.full((B,), N, dtype=torch.long, device=wav.device)
        wav_lens = wav_lens.to(wav.device)

        # zero any padding so it cannot leak into the last frames
        sample_mask = torch.arange(N, device=wav.device)[None, :] < wav_lens[:, None]
        wav = wav.float() * sample_mask

        with torch.autocast(device_type=wav.device.type, enabled=False):
            feats = torch.log(self.mel(wav) + LOG_EPS)         # [B, n_mels, T]

        feat_lens = num_frames(wav_lens, self.hop_length)
        T = feats.shape[-1]
        assert T == num_frames(N, self.hop_length), (T, N)
        frame_mask = (torch.arange(T, device=wav.device)[None, :]
                      < feat_lens[:, None]).unsqueeze(1)        # [B, 1, T]

        if self.normalize:
            n = feat_lens.view(B, 1, 1).float()
            mean = (feats * frame_mask).sum(-1, keepdim=True) / n
            var = (((feats - mean) * frame_mask) ** 2).sum(-1, keepdim=True) / n
            feats = (feats - mean) / torch.sqrt(var + NORM_EPS)

        feats = feats * frame_mask                               # padded frames = 0
        return feats, feat_lens
