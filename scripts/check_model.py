"""
Phase 9 check: the CNN BiLSTM model.

Run from the project root:
    python scripts/check_model.py

Sections
  1. tensor shapes through every layer
  2. output length formula T' = (T - 1) // 2 + 1 matches the real output
  3. an utterance gives the same output alone or in a padded batch, and
     changing the padding values changes nothing (eval mode)
  4. outputs are valid log probabilities; gradients reach every parameter
  5. parameter count per component
  6. CTC feasibility for EVERY utterance in train, dev and test:
     output frames >= characters + repeated letters
  7. GPU benchmark (forward + backward, mixed precision) for the hardest
     batches the loader produces, at three LSTM sizes: time and peak memory
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.features import num_frames  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.model import CNNBiLSTM, count_parameters  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def header(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    torch.manual_seed(0)
    cfg = load_config("configs/base.yaml")
    vocab = Vocabulary()
    n_mels = cfg["features"]["n_mels"]
    model = CNNBiLSTM.from_config(cfg["model"], n_mels, len(vocab)).eval()

    header("1. SHAPES")
    lens = torch.tensor([1814, 1190, 429])
    feats = torch.randn(3, n_mels, int(lens.max()))
    for k, L in enumerate(lens):
        feats[k, :, L:] = 0
    print(f"features             {tuple(feats.shape)}   lengths {lens.tolist()}")
    x, l_ = feats.unsqueeze(1), lens
    print(f"unsqueeze            {tuple(x.shape)}")
    with torch.no_grad():
        for k, block in enumerate(model.cnn, 1):
            x, l_ = block(x, l_)
            print(f"conv block {k}         {tuple(x.shape)}   lengths {l_.tolist()}")
        B, C, F, T = x.shape
        print(f"reshape              ({B}, {T}, {C * F})")
        log_probs, out_lens = model(feats, lens)
    print(f"BiLSTM + linear      {tuple(log_probs.shape)}   lengths {out_lens.tolist()}")
    check("output [B, T', 29] with T' = ceil(T / 2)",
          log_probs.shape == (3, (int(lens.max()) - 1) // 2 + 1, len(vocab)))

    header("2. OUTPUT LENGTH FORMULA")
    bad = []
    with torch.no_grad():
        for T in [1, 2, 3, 4, 5, 100, 101, 999, 1000, 1001, 1525]:
            lp, ol = model(torch.randn(1, n_mels, T), torch.tensor([T]))
            if not (lp.shape[1] == ol.item() == (T - 1) // 2 + 1
                    == model.output_lengths(torch.tensor([T])).item()):
                bad.append((T, lp.shape[1], ol.item()))
    check("T' = (T - 1) // 2 + 1 for T in 1..1525 (11 cases)", not bad, f"mismatches {bad}")

    header("3. PADDING INVARIANCE (eval mode)")
    with torch.no_grad():
        batch_lp, batch_lens = model(feats, lens)
        worst = 0.0
        for k, L in enumerate(lens):
            alone, alone_len = model(feats[k:k + 1, :, :L], lens[k:k + 1])
            n = alone_len.item()
            worst = max(worst, (batch_lp[k, :n] - alone[0]).abs().max().item())
        check("same output alone vs in padded batch", worst < 1e-4, f"max difference {worst:.1e}")
        junk = feats.clone()
        for k, L in enumerate(lens):
            junk[k, :, L:] = 99.0
        junk_lp, _ = model(junk, lens)
        diff = max((junk_lp[k, :batch_lens[k]] - batch_lp[k, :batch_lens[k]]).abs().max().item()
                   for k in range(3))
        check("garbage in the padding changes nothing", diff < 1e-4, f"max difference {diff:.1e}")

    header("4. OUTPUTS AND GRADIENTS")
    probs_sum = batch_lp.exp().sum(-1)
    valid = torch.cat([probs_sum[k, :batch_lens[k]] for k in range(3)])
    check("each frame's probabilities sum to 1", (valid - 1).abs().max().item() < 1e-4)
    model.train()
    lp, ol = model(feats, lens)
    lp.sum().backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    check("every parameter receives a finite, non zero gradient", not no_grad, f"{no_grad[:3]}")
    model.zero_grad()

    header("5. PARAMETERS")
    parts = {"CNN": model.cnn, "BiLSTM": model.lstm, "output layer": model.fc}
    for name, m in parts.items():
        print(f"{name:<14} {count_parameters(m):>10,}")
    print(f"{'total':<14} {count_parameters(model):>10,}")

    header("6. CTC FEASIBILITY (every utterance)")
    man = resolve(cfg, "manifests_dir")
    for subset in [cfg["data"]["train_subset"], cfg["data"]["dev_subset"], cfg["data"]["test_subset"]]:
        rows = read_manifest(man / f"{subset}.csv")
        T = num_frames(torch.tensor([r["num_samples"] for r in rows]))
        Tp = model.output_lengths(T)
        need = torch.tensor([len(r["transcript"]) + sum(a == b for a, b in
                                                         zip(r["transcript"], r["transcript"][1:]))
                             for r in rows])
        slack = Tp.float() / need.float()
        k = int(slack.argmin())
        check(f"{subset}: output frames >= required for all {len(rows)} utterances",
              bool((Tp >= need).all()),
              f"tightest {rows[k]['utt_id']}: {Tp[k]} frames for {need[k]} needed "
              f"({slack[k]:.2f}x)")

    header("7. GPU BENCHMARK (forward + backward, mixed precision)")
    if not torch.cuda.is_available():
        print("[SKIP] no GPU")
    else:
        dev = torch.device("cuda")
        cap = cfg["loader"]["max_batch_seconds"]
        shapes = {"8 x 24.5 s (longest train)": (8, 24.5), "86 x 2.3 s (most utterances)": (86, 2.3),
                  "14 x 14 s (typical)": (14, 14.0)}
        print(f"batches hold at most {cap} s of padded audio")
        print(f"{'hidden':>6} {'params':>8}   " + "   ".join(f"{k:>28}" for k in shapes))
        for hidden in [256, 320, 384]:
            mcfg = dict(cfg["model"], lstm_hidden=hidden)
            m = CNNBiLSTM.from_config(mcfg, n_mels, len(vocab)).to(dev).train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            cells = []
            for B, secs in shapes.values():
                T = int(secs * 100) + 1
                x = torch.randn(B, n_mels, T, device=dev)
                xl = torch.full((B,), T, device=dev)
                try:
                    torch.cuda.reset_peak_memory_stats()
                    for rep in range(4):
                        if rep == 1:
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.float16):
                            lp, _ = m(x, xl)
                        scaler.scale(lp.mean()).backward()
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) / 3 * 1000
                    gb = torch.cuda.max_memory_allocated() / 1024 ** 3
                    cells.append(f"{ms:7.0f} ms {gb:5.2f} GB")
                except torch.cuda.OutOfMemoryError:
                    cells.append("        out of memory")
                    torch.cuda.empty_cache()
            print(f"{hidden:>6} {count_parameters(m) / 1e6:>7.1f}M   " +
                  "   ".join(f"{c:>28}" for c in cells))
            del m, opt
            torch.cuda.empty_cache()
        print("(time and peak memory per training step; ~2018 steps per epoch)")

    print("\nRESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
