"""
Phase 15 check: SpecAugment (src/specaugment.py) and its wiring into training.

Run from the project root:
    python scripts/check_specaugment.py

Sections
  1. configs: Models A and B have SpecAugment OFF, Model C ON; the ONLY
     difference between the resolved model_b and model_c configs is
     specaugment.enabled (plus the experiment name)
  2. basic contract: same shape / dtype / device, input tensor never modified,
     masked values exactly 0, every other value unchanged
  3. mask structure on 64 random utterances: frequency masks are whole Mel
     bands over the valid frames, time masks are whole frames, nothing lands
     in the padding, widths respect freq_width / time_width / time_ratio
  4. edge cases: 1 frame, 5 frames (time cap 0), 3000 frames (cap 40),
     all masks disabled = identity
  5. statistics over 20000 draws: widths uniform from 0 to the cap, starts
     cover the whole range, average fraction masked
  6. reproducibility: same seed -> same masks, new seed -> new masks,
     utterances in one batch get different masks
  7. trainer wiring: TrainStep applies the augmentation on training steps,
     evaluate() never does (dev / test audio is never masked)
  8. real data: a dev utterance (clean and 10 dB) before / after masking,
     saved as results/specaugment_example.png; GPU time per training batch
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.specaugment import SpecAugment  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def header(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def runs(row):
    """Number of contiguous True runs and their lengths in a 1D bool tensor."""
    lengths, cur = [], 0
    for v in row.tolist():
        if v:
            cur += 1
        elif cur:
            lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths


def random_batch(lens, n_mels=80):
    """Normalized looking features with no exact zeros inside the valid frames."""
    lens = torch.tensor(lens)
    x = torch.randn(len(lens), n_mels, int(lens.max()))
    x = torch.where(x.abs() < 1e-3, torch.full_like(x, 1e-3), x)
    for k, L in enumerate(lens):
        x[k, :, L:] = 0
    return x, lens


def main():
    cfgs = {m: load_config(f"configs/model_{m}.yaml") for m in "abc"}
    scfg = cfgs["c"]["specaugment"]
    aug = SpecAugment.from_config(scfg)

    # ------------------------------------------------------------------ 1
    header("1. CONFIGS")
    for m in "abc":
        on = SpecAugment.from_config(cfgs[m].get("specaugment")) is not None
        print(f"model_{m}: noise {'ON ' if cfgs[m]['noise']['train_noise'] else 'OFF'}  "
              f"SpecAugment {'ON' if on else 'OFF'}")
    check("Model A: SpecAugment OFF", SpecAugment.from_config(cfgs["a"]["specaugment"]) is None)
    check("Model B: SpecAugment OFF", SpecAugment.from_config(cfgs["b"]["specaugment"]) is None)
    check("Model C: SpecAugment ON", aug is not None, repr(aug))
    check("Model C keeps noise training ON", cfgs["c"]["noise"]["train_noise"] is True)
    fb, fc = flatten(cfgs["b"]), flatten(cfgs["c"])
    diff = sorted(k for k in set(fb) | set(fc) if fb.get(k) != fc.get(k))
    check("model_b vs model_c differ ONLY in specaugment.enabled and the name",
          diff == ["experiment.name", "specaugment.enabled"], str(diff))
    check("old configs without a specaugment section -> OFF", SpecAugment.from_config(None) is None)

    # ------------------------------------------------------------------ 2
    header("2. BASIC CONTRACT")
    torch.manual_seed(0)
    x, lens = random_batch([1600, 1200, 700, 300])
    x_copy = x.clone()
    y = aug(x, lens)
    check("same shape, dtype, device", y.shape == x.shape and y.dtype == x.dtype
          and y.device == x.device, f"{tuple(y.shape)} {y.dtype}")
    check("input tensor not modified", torch.equal(x, x_copy))
    changed = y != x
    check("changed values are exactly 0", bool((y[changed] == 0).all()))
    check("something was masked", int(changed.sum()) > 0,
          f"{changed.float().mean().item():.1%} of all values")
    xh = x.half()
    check("float16 input stays float16", aug(xh, lens).dtype == torch.float16)

    # ------------------------------------------------------------------ 3
    header("3. MASK STRUCTURE (64 random utterances)")
    torch.manual_seed(1)
    L = torch.randint(50, 3000, (64,)).tolist()
    x, lens = random_batch(L)
    y = aug(x, lens)
    zero = (y == 0)
    bad_pad = bad_shape = bad_freq = bad_time = 0
    for k, n in enumerate(lens.tolist()):
        z = zero[k, :, :n]                                  # valid region
        if not torch.equal(y[k, :, n:], x[k, :, n:]):       # padding untouched (all 0)
            bad_pad += 1
        rows = z.all(1)                                     # bins masked for every frame
        cols = z.all(0)                                     # frames masked for every bin
        # every zero must be explained by a full row or a full column
        if not torch.equal(z, rows[:, None] | cols[None, :]):
            bad_shape += 1
        r = runs(rows)
        if len(r) > aug.freq_masks or sum(r) > aug.freq_masks * aug.freq_width:
            bad_freq += 1
        cap = min(aug.time_width, int(aug.time_ratio * n))
        c = runs(cols)
        if len(c) > aug.time_masks or sum(c) > aug.time_masks * cap:
            bad_time += 1
    check("padding frames stay exactly as they were", bad_pad == 0, f"{bad_pad} bad")
    check("every masked value is part of a full band or a full frame", bad_shape == 0,
          f"{bad_shape} bad")
    check(f"frequency: <= {aug.freq_masks} bands, <= {aug.freq_masks}x{aug.freq_width} bins",
          bad_freq == 0, f"{bad_freq} bad")
    check(f"time: <= {aug.time_masks} blocks, each <= min({aug.time_width}, "
          f"{aug.time_ratio} x T)", bad_time == 0, f"{bad_time} bad")
    frac = sum(zero[k, :, :n].sum().item() for k, n in enumerate(lens.tolist()))
    valid = sum(80 * n for n in lens.tolist())
    print(f"masked share of valid values in this batch: {frac / valid:.1%}")

    # ------------------------------------------------------------------ 4
    header("4. EDGE CASES")
    torch.manual_seed(2)
    for T in [1, 5, 9, 10, 400, 3000]:
        x, lens = random_batch([T])
        ok = True
        for _ in range(200):
            f, t = aug.masks(x, lens)
            cap = min(aug.time_width, int(aug.time_ratio * T))
            if int(t.sum()) > aug.time_masks * cap or int(t[:, T:].sum()) > 0:
                ok = False
            if int(f.sum()) > aug.freq_masks * aug.freq_width:
                ok = False
        check(f"T = {T}: time cap {min(aug.time_width, int(aug.time_ratio * T))} frames/mask, "
              "masks stay in range", ok)
    off = SpecAugment(freq_masks=0, time_masks=0)
    x, lens = random_batch([800, 300])
    check("0 frequency masks and 0 time masks = identity", torch.equal(off(x, lens), x))
    zero_w = SpecAugment(freq_width=0, time_width=0)
    check("width 0 = identity", torch.equal(zero_w(x, lens), x))
    for bad in [dict(freq_masks=-1), dict(time_width=2.5), dict(time_ratio=1.5)]:
        try:
            SpecAugment(**bad)
            check(f"rejects {bad}", False)
        except ValueError:
            check(f"rejects {bad}", True)

    # ------------------------------------------------------------------ 5
    header("5. STATISTICS (20000 single mask draws)")
    torch.manual_seed(3)
    one = SpecAugment(freq_masks=1, freq_width=aug.freq_width, time_masks=1,
                      time_width=aug.time_width, time_ratio=aug.time_ratio)
    N, T = 20000, 1000                                      # cap = min(40, 100) = 40
    shape_only = torch.zeros(1, 1, 1).expand(N, 80, T)     # masks() only reads the shape
    f, t = one.masks(shape_only, torch.full((N,), T))
    fw, tw = f.sum(1), t.sum(1)
    fstart = torch.where(fw > 0, f.float().argmax(1), torch.full_like(fw, -1))
    tstart = torch.where(tw > 0, t.float().argmax(1), torch.full_like(tw, -1))
    print(f"freq width: min {fw.min().item()} max {fw.max().item()} mean {fw.float().mean():.2f}"
          f" (uniform 0..{aug.freq_width} -> {aug.freq_width / 2:.1f})")
    print(f"time width: min {tw.min().item()} max {tw.max().item()} mean {tw.float().mean():.2f}"
          f" (uniform 0..{aug.time_width} -> {aug.time_width / 2:.1f})")
    check("frequency width takes every value 0..F",
          set(fw.tolist()) == set(range(aug.freq_width + 1)))
    check("frequency width mean within 0.3 of F/2",
          abs(fw.float().mean().item() - aug.freq_width / 2) < 0.3)
    check("time width takes every value 0..cap", set(tw.tolist()) == set(range(aug.time_width + 1)))
    check("time width mean within 0.4 of cap/2",
          abs(tw.float().mean().item() - aug.time_width / 2) < 0.4)
    fs, ts = fstart[fstart >= 0], tstart[tstart >= 0]
    check("frequency masks can start at bin 0 and reach bin 79",
          fs.min().item() == 0 and (fs + fw[fstart >= 0]).max().item() == 80)
    check("time masks can start at frame 0 and reach the last frame",
          ts.min().item() == 0 and (ts + tw[tstart >= 0]).max().item() == T)
    # expected share masked with the real policy on a typical 12 s utterance
    torch.manual_seed(4)
    x, lens = random_batch([1200] * 500)
    share = (aug(x, lens) == 0).float().mean().item()
    print(f"policy on 12 s utterances: {share:.1%} of values masked on average")
    check("average masked share is moderate (10% to 40%)", 0.10 < share < 0.40, f"{share:.1%}")

    # ------------------------------------------------------------------ 6
    header("6. REPRODUCIBILITY")
    x, lens = random_batch([900, 900, 900, 900])
    torch.manual_seed(1234)
    a1 = aug(x, lens)
    torch.manual_seed(1234)
    a2 = aug(x, lens)
    torch.manual_seed(1235)
    a3 = aug(x, lens)
    check("same seed -> identical masks", torch.equal(a1, a2))
    check("different seed -> different masks", not torch.equal(a1, a3))
    z = (a1 == 0)
    check("each utterance in a batch gets its own masks",
          len({z[k].flatten().nonzero().flatten().tolist().__repr__() for k in range(4)}) == 4)

    # ------------------------------------------------------------------ 7
    header("7. TRAINER WIRING")
    from src.features import LogMel
    from src.model import CNNBiLSTM
    from src.trainer import TrainStep, evaluate
    from src.vocabulary import Vocabulary

    class Counting(SpecAugment):
        calls = 0

        def forward(self, feats, feat_lens):
            Counting.calls += 1
            return super().forward(feats, feat_lens)

    torch.manual_seed(5)
    vocab = Vocabulary()
    mcfg = dict(cfgs["c"]["model"], lstm_hidden=32, lstm_layers=1)
    model = CNNBiLSTM.from_config(mcfg, 80, len(vocab))
    feats = LogMel(**cfgs["c"]["features"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    counting = Counting(**{k: v for k, v in scfg.items() if k != "enabled"})
    step = TrainStep(model, feats, opt, use_amp=False, augment=counting)
    texts = ["HELLO WORLD", "YES"]
    tg = [torch.tensor(vocab.encode(t)) for t in texts]
    wav_lens = torch.tensor([24000, 9000])
    batch = dict(wav=torch.randn(2, 24000) * 0.1, wav_lens=wav_lens,
                 targets=torch.nn.utils.rnn.pad_sequence(tg, batch_first=True),
                 target_lens=torch.tensor([len(t) for t in tg]), utt_ids=["u1", "u2"],
                 transcripts=texts)
    batch["wav"][1, 9000:] = 0
    losses = [step(batch, torch.device("cpu"))[0] for _ in range(3)]
    check("TrainStep calls SpecAugment once per training step", Counting.calls == 3,
          f"{Counting.calls} calls")
    check("training loss is finite with SpecAugment", all(torch.isfinite(torch.tensor(losses))),
          str([round(v, 3) for v in losses]))
    before = Counting.calls
    e1 = evaluate(model, feats, [batch], torch.device("cpu"), vocab, use_amp=False)
    e2 = evaluate(model, feats, [batch], torch.device("cpu"), vocab, use_amp=False)
    check("evaluate() never calls SpecAugment", Counting.calls == before)
    check("evaluation is deterministic (no masking at eval time)",
          e1["loss"] == e2["loss"] and e1["hyps"] == e2["hyps"])
    src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
    check("train.py builds SpecAugment from the config and passes it to TrainStep",
          "SpecAugment.from_config(cfg.get(\"specaugment\"))" in src and "augment=augment" in src)

    # ------------------------------------------------------------------ 8
    header("8. REAL DATA + GPU TIME")
    man, raw = resolve(cfgs["c"], "manifests_dir"), resolve(cfgs["c"], "raw_dir")
    dev_csv = man / f"{cfgs['c']['data']['dev_subset']}.csv"
    if not dev_csv.exists():
        print(f"{dev_csv} not found: skipping the real data section")
    else:
        from src.dataset import ASRDataset, read_recipe
        from src.librispeech import read_manifest
        rows = read_manifest(dev_csv)
        row = min(rows, key=lambda r: abs(r["duration_s"] - 8.0))     # about 8 s
        recipe = read_recipe(man / "recipes" / f"{cfgs['c']['data']['dev_subset']}_noise.csv")
        clean = ASRDataset([row], raw, vocab)[0]
        noisy = ASRDataset([row], raw, vocab, recipe=recipe, snr_db=10.0)[0]
        lm = LogMel(**cfgs["c"]["features"])
        torch.manual_seed(7)
        panels = []
        for name, item in [("clean", clean), ("10 dB", noisy)]:
            f_, l_ = lm(item["wav"][None], torch.tensor([item["wav"].numel()]))
            a_ = aug(f_, l_)
            panels += [(f"{name}", f_[0]), (f"{name} + SpecAugment", a_[0])]
            m = (a_ == 0) & (f_ != 0)
            print(f"{name}: {l_.item()} frames, masked {m.float().mean().item():.1%}")
        print(f"utterance {row['utt_id']} ({row['duration_s']:.1f} s): {row['transcript'][:70]}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
            for ax, (title, im) in zip(axes, panels):
                ax.imshow(im.numpy(), origin="lower", aspect="auto", cmap="magma",
                          vmin=-2.5, vmax=2.5, extent=[0, im.shape[1] / 100, 0, 80])
                ax.set_title(title, fontsize=10)
                ax.set_ylabel("Mel bin")
            axes[-1].set_xlabel("time (s)")
            fig.suptitle(f"SpecAugment on dev utterance {row['utt_id']}", fontsize=11)
            fig.tight_layout()
            out = ROOT / "results" / "specaugment_example.png"
            out.parent.mkdir(exist_ok=True)
            fig.savefig(out, dpi=120)
            print(f"saved {out}")
        except ImportError:
            print("matplotlib not installed: figure skipped")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        g = SpecAugment.from_config(scfg).to(dev)
        lens = torch.tensor([1700] * 11 + [1500], device=dev)    # ~200 s of audio
        x = torch.randn(12, 80, 1700, device=dev)
        for _ in range(5):
            g(x, lens)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            g(x, lens)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50 * 1000
        print(f"GPU ({torch.cuda.get_device_name(0)}): {ms:.2f} ms per 200 s batch")
        check("SpecAugment costs < 5 ms per training batch on the GPU", ms < 5, f"{ms:.2f} ms")
    else:
        print("no GPU: timing skipped")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
