"""
Phase 1: environment verification for the noise robust ASR project.

Run from the project root:
    python scripts/check_env.py

What it does
  1. Prints Python, PyTorch, torchaudio, CUDA, cuDNN, GPU and package versions.
  2. Runs small functional smoke tests for every library feature the project
     will rely on (FLAC I/O, log Mel features, SpecAugment masks, resampling,
     CTC loss, a BiLSTM forward/backward pass with mixed precision, WER/CER).
  3. Saves everything to results/env_report.json so the exact environment can
     be quoted in the paper's Experimental Setup section.

Exit code is 0 only if every required check passes.
"""

import importlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import traceback
from datetime import datetime, timezone

RESULTS = []          # (name, status, detail)
REPORT = {}           # written to results/env_report.json


def record(name, ok, detail="", required=True):
    status = "PASS" if ok else ("FAIL" if required else "WARN")
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  ->  {detail}" if detail else ""))


def run_check(name, fn, required=True):
    try:
        detail = fn()
        record(name, True, detail or "", required)
    except Exception as e:  # noqa: BLE001  we want to report every failure
        record(name, False, f"{type(e).__name__}: {e}", required)
        if os.environ.get("CHECK_ENV_DEBUG"):
            traceback.print_exc()


# --------------------------------------------------------------------------
# 1. Versions
# --------------------------------------------------------------------------
print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

py = sys.version_info
REPORT["python"] = platform.python_version()
REPORT["platform"] = platform.platform()
REPORT["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Python      : {REPORT['python']}  ({sys.executable})")
print(f"Platform    : {REPORT['platform']}")
record("Python version is 3.10 to 3.13", (3, 10) <= (py.major, py.minor) <= (3, 13),
       REPORT["python"])

try:
    import torch
    import torchaudio
except ImportError as e:
    record("import torch / torchaudio", False, str(e))
    print("\nCannot continue without torch and torchaudio. See the install steps.")
    sys.exit(1)

REPORT["torch"] = torch.__version__
REPORT["torchaudio"] = torchaudio.__version__
REPORT["torch_cuda_build"] = torch.version.cuda
print(f"PyTorch     : {torch.__version__}  (built for CUDA {torch.version.cuda})")
print(f"torchaudio  : {torchaudio.__version__}")

# torchaudio is in maintenance mode and its last release is 2.11, while torch
# keeps moving. A matched pair (2.11 with 2.11) is preferred. A mismatch is only
# a warning because the project uses torchaudio's pure Python transforms; the
# functional checks below decide whether the pair actually works.
t_rel = torch.__version__.split("+")[0].rsplit(".", 1)[0]
a_rel = torchaudio.__version__.split("+")[0].rsplit(".", 1)[0]
record("torch and torchaudio release match", t_rel == a_rel,
       f"torch {t_rel} vs torchaudio {a_rel}", required=False)

cuda_ok = torch.cuda.is_available()
REPORT["cuda_available"] = cuda_ok
if cuda_ok:
    props = torch.cuda.get_device_properties(0)
    REPORT["gpu_name"] = props.name
    REPORT["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
    REPORT["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    REPORT["cudnn"] = torch.backends.cudnn.version()
    REPORT["bf16_supported"] = torch.cuda.is_bf16_supported()
    print(f"GPU         : {props.name}")
    print(f"VRAM        : {REPORT['gpu_vram_gb']} GB")
    print(f"Compute cap : {REPORT['gpu_compute_capability']}")
    print(f"cuDNN       : {REPORT['cudnn']}")
    print(f"bf16        : {REPORT['bf16_supported']}")
# A missing GPU is a warning, not a failure: CPU is enough for phases 2 to 11.
record("CUDA GPU available", cuda_ok,
       REPORT.get("gpu_name", "no GPU visible to PyTorch"), required=False)

print("\nPackages:")
# (import name, distribution name on PyPI)
PACKAGES = [("numpy", "numpy"), ("pandas", "pandas"), ("soundfile", "soundfile"),
            ("jiwer", "jiwer"), ("matplotlib", "matplotlib"), ("yaml", "PyYAML"),
            ("tqdm", "tqdm")]
REPORT["packages"] = {}
for name, dist in PACKAGES:
    try:
        importlib.import_module(name)
        ver = importlib.metadata.version(dist)
        REPORT["packages"][name] = ver
        print(f"  {name:<11} {ver}")
    except ImportError as e:
        REPORT["packages"][name] = None
        record(f"import {name}", False, str(e))

# --------------------------------------------------------------------------
# 2. Functional smoke tests
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FUNCTIONAL CHECKS")
print("=" * 70)

device = torch.device("cuda" if cuda_ok else "cpu")
SR = 16000          # project sample rate (LibriSpeech and MUSAN are both 16 kHz)
N_MELS = 80
HOP = 160           # 10 ms hop at 16 kHz
N_FFT = 400         # 25 ms window at 16 kHz


def check_flac_io():
    # LibriSpeech ships as FLAC, MUSAN as WAV. We read both with soundfile so
    # the project does not depend on FFmpeg / TorchCodec.
    import numpy as np
    import soundfile as sf
    x = (0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype("float32")
    with tempfile.TemporaryDirectory() as d:
        for ext in ("flac", "wav"):
            p = os.path.join(d, f"t.{ext}")
            sf.write(p, x, SR)
            y, sr = sf.read(p, dtype="float32")
            assert sr == SR and y.shape == x.shape, (sr, y.shape)
            assert np.max(np.abs(y - x)) < 1e-3
    return f"FLAC and WAV round trip OK, libsndfile {sf.__libsndfile_version__}"


def check_logmel():
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS).to(device)
    wav = torch.randn(2, SR, device=device)           # [B, samples], 1 second
    feats = torch.log(mel(wav) + 1e-6)                # [B, n_mels, frames]
    expected_frames = SR // HOP + 1                   # center=True -> 101
    assert feats.shape == (2, N_MELS, expected_frames), feats.shape
    assert torch.isfinite(feats).all()
    return f"[B, Mel, T] = {tuple(feats.shape)}"


def check_specaugment():
    fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    tm = torchaudio.transforms.TimeMasking(time_mask_param=35)
    x = torch.ones(2, N_MELS, 200)
    y = tm(fm(x))
    assert y.shape == x.shape
    return f"masked fraction {1 - y.mean().item():.2f}"


def check_resample():
    wav = torch.randn(1, 44100)
    out = torchaudio.functional.resample(wav, orig_freq=44100, new_freq=SR)
    assert abs(out.shape[-1] - SR) <= 1, out.shape
    return f"44100 Hz -> {SR} Hz, {out.shape[-1]} samples"


def check_ctc():
    # Convention used in the project: log_probs [T, B, C], blank index 0.
    T, B, C, S = 50, 3, 30, 12
    log_probs = torch.randn(T, B, C, device=device, requires_grad=True).log_softmax(-1)
    targets = torch.randint(1, C, (B, S), device=device)   # never the blank id
    in_len = torch.full((B,), T, dtype=torch.long)
    tg_len = torch.tensor([12, 8, 5], dtype=torch.long)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(log_probs, targets, in_len, tg_len)
    loss.backward()
    assert torch.isfinite(loss), loss
    return f"loss {loss.item():.3f}, convention [T, B, C], blank=0"


def check_bilstm_amp():
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(input_size=N_MELS, hidden_size=256, num_layers=3,
                         bidirectional=True, batch_first=True).to(device)
    head = torch.nn.Linear(512, 30).to(device)
    opt = torch.optim.AdamW(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)
    x = torch.randn(4, 300, N_MELS, device=device)   # [B, T, F]
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        out, _ = lstm(x)
        logits = head(out)                            # [B, T, C]
    loss = logits.float().pow(2).mean()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    peak = ""
    if use_amp:
        peak = f", peak GPU memory {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB"
    return f"output {tuple(logits.shape)}, AMP={'on' if use_amp else 'off (CPU)'}{peak}"


def check_metrics():
    import jiwer
    w = jiwer.wer("the cat is running", "the dog is running")
    c = jiwer.cer("hello", "helo")
    assert abs(w - 0.25) < 1e-9, w
    assert abs(c - 0.20) < 1e-9, c
    return f"WER={w:.2f} (expected 0.25), CER={c:.2f} (expected 0.20)"


def check_seed():
    torch.manual_seed(1234)
    a = torch.randn(5)
    torch.manual_seed(1234)
    b = torch.randn(5)
    assert torch.equal(a, b)
    return "same seed gives identical tensors"


run_check("soundfile reads FLAC/WAV", check_flac_io)
run_check("80 band log Mel spectrogram", check_logmel)
run_check("SpecAugment masks (torchaudio)", check_specaugment)
run_check("Resampling", check_resample)
run_check("CTC loss forward/backward", check_ctc)
run_check("3 layer BiLSTM + AMP training step", check_bilstm_amp)
run_check("jiwer WER/CER", check_metrics)
run_check("Seed reproducibility", check_seed)

# --------------------------------------------------------------------------
# 3. Summary and report
# --------------------------------------------------------------------------
fails = [r for r in RESULTS if r[1] == "FAIL"]
warns = [r for r in RESULTS if r[1] == "WARN"]
REPORT["checks"] = [{"name": n, "status": s, "detail": d} for n, s, d in RESULTS]

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "env_report.json"), "w") as f:
    json.dump(REPORT, f, indent=2)

print("\n" + "=" * 70)
print(f"SUMMARY: {len(RESULTS) - len(fails) - len(warns)} passed, "
      f"{len(warns)} warnings, {len(fails)} failed")
print("Report saved to results/env_report.json")
print("=" * 70)
sys.exit(1 if fails else 0)
