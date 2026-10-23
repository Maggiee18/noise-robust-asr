"""
CNN BiLSTM acoustic model for character CTC.

Tensor shapes (default configuration)
-------------------------------------
    features       [B, 80, T]           log Mel from src/features.py
    unsqueeze      [B, 1, 80, T]        one input "image" channel
    conv block 1   [B, 32, 40, T/2]     3x3 conv, stride (2, 2): halves frequency AND time
    conv block 2   [B, 32, 20, T/2]     3x3 conv, stride (2, 1): halves frequency only
    reshape        [B, T/2, 640]        channels x frequency become the feature vector
    BiLSTM x3      [B, T/2, 2H]         packed by length, so padding never enters
    linear         [B, T/2, 29]         one score per character class
    log_softmax    [B, T/2, 29]         log probabilities for CTC

    exact output length: T' = (T - 1) // 2 + 1  (= ceil(T / 2))

The model returns BATCH FIRST log probabilities [B, T', C] plus T' per
utterance. torch.nn.CTCLoss expects [T', B, C]; the transpose is done in the
loss (Phase 10), in one place.

Design decisions (documented for the paper)
-------------------------------------------
* Time is downsampled by exactly 2: 50 output frames per second, one every
  20 ms. Phase 3 measured the fastest training utterance at 23
  characters per second; CTC needs at least (characters + repeated letters)
  frames, which 2x leaves with a wide margin while 4x would leave almost none.
  check_model.py verifies this for EVERY utterance in the dataset.
* Frequency is downsampled 4x (80 -> 20) to keep the LSTM input small.
* Each conv block: Conv2d -> BatchNorm2d -> ReLU, then frames beyond each
  utterance's length are set to zero, so padding behaves exactly like the
  convolution's own zero padding. Consequence (verified): an utterance gets
  the same output alone or inside a padded batch.
* The BiLSTM uses packed sequences. Without packing, the backward direction
  would start reading from the padding at the end of short utterances.
* Layer counts, channels, strides, hidden size and dropout all come from the
  config, so the architecture can be varied without code changes.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def conv_out_len(length, kernel, stride, padding):
    """Standard convolution output length (works on ints and tensors)."""
    return (length + 2 * padding - kernel) // stride + 1


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super().__init__()
        self.kernel, self.stride, self.padding = kernel, stride, kernel // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=self.padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def out_len(self, t):
        return conv_out_len(t, self.kernel, self.stride[1], self.padding)

    def forward(self, x, lens):
        x = self.act(self.bn(self.conv(x)))            # [B, C, F', T']
        lens = self.out_len(lens)
        mask = torch.arange(x.shape[-1], device=x.device)[None, :] < lens[:, None]
        return x * mask[:, None, None, :].to(x.dtype), lens


class CNNBiLSTM(nn.Module):
    def __init__(self, n_mels=80, n_classes=29, cnn_channels=(32, 32), cnn_kernel=3,
                 cnn_strides=((2, 2), (2, 1)), lstm_hidden=256, lstm_layers=3, dropout=0.2):
        super().__init__()
        if len(cnn_channels) != len(cnn_strides):
            raise ValueError("cnn_channels and cnn_strides must have the same length")
        blocks, c_in, freq = [], 1, n_mels
        for c_out, stride in zip(cnn_channels, cnn_strides):
            blocks.append(ConvBlock(c_in, c_out, cnn_kernel, tuple(stride)))
            freq = conv_out_len(freq, cnn_kernel, stride[0], cnn_kernel // 2)
            c_in = c_out
        self.cnn = nn.ModuleList(blocks)
        self.lstm_input = c_in * freq
        self.lstm = nn.LSTM(self.lstm_input, lstm_hidden, num_layers=lstm_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, n_classes)

    @classmethod
    def from_config(cls, mcfg, n_mels, n_classes):
        return cls(n_mels=n_mels, n_classes=n_classes,
                   cnn_channels=tuple(mcfg["cnn_channels"]), cnn_kernel=mcfg["cnn_kernel"],
                   cnn_strides=tuple(tuple(s) for s in mcfg["cnn_strides"]),
                   lstm_hidden=mcfg["lstm_hidden"], lstm_layers=mcfg["lstm_layers"],
                   dropout=mcfg["dropout"])

    def output_lengths(self, feat_lens):
        """Number of output frames for given input frame counts."""
        for block in self.cnn:
            feat_lens = block.out_len(feat_lens)
        return feat_lens

    def forward(self, feats, feat_lens):
        """
        feats:     [B, n_mels, T] float
        feat_lens: [B] int64
        returns    log_probs [B, T', n_classes], out_lens [B]
        """
        # zero any padding first: the first convolution looks one frame past
        # each utterance's end, so non zero padding would leak into real frames
        in_mask = torch.arange(feats.shape[-1], device=feats.device)[None, :] < feat_lens[:, None]
        x, lens = (feats * in_mask[:, None, :].to(feats.dtype)).unsqueeze(1), feat_lens
        for block in self.cnn:
            x, lens = block(x, lens)                    # [B, C, F', T']
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)   # [B, T', C*F']

        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(out, batch_first=True, total_length=T)  # [B, T', 2H]

        logits = self.fc(self.dropout(x))               # [B, T', n_classes]
        return logits.float().log_softmax(dim=-1), lens


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
