"""
Phase 12: the 50 sample overfitting test.

Run from the project root:
    python scripts/overfit_test.py
    python scripts/overfit_test.py --n 50 --max-epochs 400      (defaults)

Trains the REAL pipeline (log Mel -> CNN BiLSTM -> CTC, mixed precision,
gradient clipping) on 50 clean training utterances, and checks that it can
memorize them: greedy transcripts must match the references (CER <= 1%).

If a correct pipeline is given only 50 examples, it will learn them by heart.
If it cannot, something is broken (labels, lengths, blank index, decoder,
features ...) and full training would silently waste hours.

Settings that differ from real training, on purpose:
  * dropout 0 and weight decay 0: we WANT memorization here
  * no noise, no SpecAugment
  * the same 50 utterances every epoch (only batch order shuffled)
  * 30 s batches instead of 200 s, so there are ~10 updates per epoch

Outputs
  results/phase12_overfit_log.csv      one row per evaluation
  results/plots/phase12_overfit.png    loss, CER and blank fraction curves
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.dataset import ASRDataset, DurationBatchSampler, collate  # noqa: E402
from src.features import LogMel  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.model import CNNBiLSTM, count_parameters  # noqa: E402
from src.trainer import TrainStep, evaluate, set_seed  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-seconds", type=float, default=10.0,
                    help="only pick utterances up to this long (keeps the test fast)")
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-seconds", type=float, default=30.0,
                    help="smaller than real training (200 s) so the 50 utterances give ~10 "
                         "updates per epoch instead of 3")
    ap.add_argument("--hidden", type=int, default=None, help="override lstm_hidden")
    args = ap.parse_args()

    cfg = load_config("configs/base.yaml")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary()

    # ---- the 50 utterances (seeded choice among those <= max_seconds)
    rows = read_manifest(resolve(cfg, "manifests_dir") / f"{cfg['data']['train_subset']}.csv")
    pool = [r for r in rows if r["duration_s"] <= args.max_seconds]
    picks = random.Random(cfg["seed"]).sample(pool, args.n)
    secs = sum(r["duration_s"] for r in picks)
    print(f"{args.n} utterances from {cfg['data']['train_subset']} "
          f"({secs / 60:.1f} min of audio, {sum(len(r['transcript']) for r in picks)} characters)")

    # ---- load once, batch by duration, keep in memory
    ds = ASRDataset(picks, resolve(cfg, "raw_dir"), vocab)
    items = [ds[k] for k in range(len(ds))]
    sampler = DurationBatchSampler([r["duration_s"] for r in picks],
                                   args.batch_seconds, shuffle=True, seed=cfg["seed"])
    eval_batches = [collate([items[i] for i in b])
                    for b in DurationBatchSampler([r["duration_s"] for r in picks],
                                                  cfg["loader"]["max_batch_seconds"],
                                                  shuffle=False).batches]

    # ---- model: real architecture, dropout 0 for memorization
    mcfg = dict(cfg["model"], dropout=0.0)
    if args.hidden:
        mcfg["lstm_hidden"] = args.hidden
    model = CNNBiLSTM.from_config(mcfg, cfg["features"]["n_mels"], len(vocab)).to(device)
    features = LogMel(**cfg["features"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    step = TrainStep(model, features, opt, clip_norm=5.0, use_amp=True)
    print(f"model {count_parameters(model) / 1e6:.1f}M parameters, device {device}, "
          f"AMP {step.use_amp}, lr {args.lr}, {len(sampler)} batches per epoch")

    log_path = ROOT / "results" / "phase12_overfit_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = []
    print(f"\n{'epoch':>5} {'train loss':>10} {'eval loss':>9} {'CER':>7} {'WER':>7} "
          f"{'exact':>6} {'blank':>6} {'time':>6}")
    t_start, passed_streak, first_hit = time.perf_counter(), 0, None
    for epoch in range(1, args.max_epochs + 1):
        sampler.set_epoch(epoch)
        losses = [step(collate([items[i] for i in b]), device)[0] for b in sampler.batches]
        if epoch % args.eval_every and epoch != 1:
            continue
        ev = evaluate(model, features, eval_batches, device, vocab)
        cer = jiwer.cer(ev["refs"], ev["hyps"])
        wer = jiwer.wer(ev["refs"], ev["hyps"])
        exact = sum(h == r for h, r in zip(ev["hyps"], ev["refs"]))
        elapsed = time.perf_counter() - t_start
        row = dict(epoch=epoch, train_loss=sum(losses) / len(losses), eval_loss=ev["loss"],
                   cer=cer, wer=wer, exact=exact, blank_fraction=ev["blank_fraction"],
                   seconds=round(elapsed, 1))
        log.append(row)
        print(f"{epoch:>5} {row['train_loss']:>10.4f} {ev['loss']:>9.4f} {cer:>7.2%} {wer:>7.2%} "
              f"{exact:>3}/{args.n} {ev['blank_fraction']:>6.3f} {elapsed:>5.0f}s")
        if epoch in (1, 25, 50) or (cer <= 0.01 and first_hit is None):
            k = 0
            print(f"        ref: {ev['refs'][k][:75]}")
            print(f"        hyp: {ev['hyps'][k][:75]}")
        if cer <= 0.01:
            first_hit = first_hit or epoch
            passed_streak += 1
            if passed_streak >= 3:
                break
        else:
            passed_streak = 0

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)

    # ---- figure
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ep, [r["train_loss"] for r in log], label="train")
    ax[0].plot(ep, [r["eval_loss"] for r in log], label="eval (same 50)")
    ax[0].set_yscale("log")
    ax[0].set_title("CTC loss per character")
    ax[0].legend()
    ax[1].plot(ep, [100 * r["cer"] for r in log], label="CER")
    ax[1].plot(ep, [100 * r["wer"] for r in log], label="WER")
    ax[1].set_title("error rate on the 50 utterances (%)")
    ax[1].legend()
    ax[2].plot(ep, [r["blank_fraction"] for r in log])
    ax[2].set_ylim(0, 1)
    ax[2].set_title("fraction of frames decoded as blank")
    for a in ax:
        a.set_xlabel("epoch")
    out = ROOT / "results" / "plots" / "phase12_overfit.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)

    final = log[-1]
    print(f"\nLog: {log_path}\nFigure: {out}")
    print("\nFinal transcripts (first 5):")
    for h, r in list(zip(ev["hyps"], ev["refs"]))[:5]:
        print(f"  {'OK ' if h == r else 'ERR'} {h[:90]}")
    ok = final["cer"] <= 0.01
    print("\nRESULT: " + (f"PASS - memorized {final['exact']}/{args.n} exactly, CER {final['cer']:.2%} "
                          f"(first reached CER <= 1% at epoch {first_hit})" if ok else
                          f"FAIL - CER {final['cer']:.2%} after {final['epoch']} epochs. STOP and debug "
                          "before Phase 13; paste the full output."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
