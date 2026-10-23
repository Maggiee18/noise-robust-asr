"""
Train one experiment (Model A, B or C) from a config file.

Run from the project root:
    python src/train.py --config configs/model_a.yaml                     # full run
    python src/train.py --config configs/model_a.yaml --limit 500 --epochs 40 --dev-limit 300 --name stage7_a500
    python src/train.py --config configs/model_a.yaml --resume            # continue after a stop

What happens
  * training data: the config's train subset (optionally a seeded --limit subset),
    clean (Model A) or with training pool noise (Models B, C);
    Model C also gets SpecAugment masks on the training features (src/specaugment.py)
  * every epoch: evaluation on the dev set in every condition of
    evaluation.conditions (clean + fixed noisy recipes at 20/10/5/0 dB)
  * the best epoch is chosen by evaluation.selection (mean dev WER over all
    conditions) - the same rule for every model; the TEST set is never touched

Everything goes to experiments/<name>/:
    config.yaml           the fully resolved configuration used
    environment.json      library versions, GPU, command line
    train_utterances.txt  ids used for training (when --limit is set)
    log.csv               one row per epoch: losses, lr, time, dev WER/CER per condition
    steps.csv             training loss / grad norm / lr every log_every_steps
    last.pt, best.pt      checkpoints (model, optimizer, scheduler, scaler, epoch)
    dev_predictions_best.csv   dev transcripts of the best epoch, every condition
"""

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.dataset import ASRDataset, DurationBatchSampler, collate, read_recipe  # noqa: E402
from src.features import LogMel  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.metrics import error_rates  # noqa: E402
from src.model import CNNBiLSTM, count_parameters  # noqa: E402
from src.musan import NoiseBank  # noqa: E402
from src.specaugment import SpecAugment  # noqa: E402
from src.trainer import TrainStep, evaluate, set_seed  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402


# ---------------------------------------------------------------- helpers
def warmup_cosine(total_steps, warmup_fraction, min_ratio):
    warm = max(1, int(total_steps * warmup_fraction))

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total_steps - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def make_loader(dataset, sampler, workers, pin, persistent):
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                      pin_memory=pin, persistent_workers=persistent and workers > 0)


def build_dev(cfg, vocab, limit, workers, pin):
    man, raw = resolve(cfg, "manifests_dir"), resolve(cfg, "raw_dir")
    rows = read_manifest(man / f"{cfg['data']['dev_subset']}.csv")
    if limit:
        keep = set(random.Random(cfg["seed"]).sample(range(len(rows)), limit))
        rows = [r for k, r in enumerate(rows) if k in keep]
    recipe = read_recipe(man / "recipes" / f"{cfg['data']['dev_subset']}_noise.csv")
    loaders = {}
    for cond in cfg["evaluation"]["conditions"]:
        if str(cond) == "clean":
            ds = ASRDataset(rows, raw, vocab)
        else:
            ds = ASRDataset(rows, raw, vocab, recipe=recipe, snr_db=float(cond))
        sampler = DurationBatchSampler(ds.durations(), cfg["loader"]["max_batch_seconds"],
                                       shuffle=False)
        loaders[str(cond)] = make_loader(ds, sampler, workers, pin, persistent=False)
    return loaders, len(rows)


def save_checkpoint(path, model, opt, sched, scaler, epoch, step, best, cfg):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                    epoch=epoch, global_step=step, best=best, config=cfg), tmp)
    tmp.replace(path)                     # never leave a half written checkpoint


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", help="run name (default: experiment.name from the config)")
    ap.add_argument("--limit", type=int, help="train on a seeded subset of N utterances")
    ap.add_argument("--epochs", type=int, help="override training.epochs")
    ap.add_argument("--dev-limit", type=int, help="evaluate on a seeded subset of the dev set")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.dev_limit:
        cfg["evaluation"]["dev_limit"] = args.dev_limit
    name = args.name or cfg["experiment"]["name"]
    run_dir = ROOT / "experiments" / name
    if run_dir.exists() and (run_dir / "log.csv").exists() and not args.resume:
        sys.exit(f"{run_dir} already has results. Use --resume to continue, or a new --name.")
    run_dir.mkdir(parents=True, exist_ok=True)

    tcfg, ncfg, lcfg = cfg["training"], cfg["noise"], cfg["loader"]
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = lcfg["pin_memory"] and device.type == "cuda"
    vocab = Vocabulary()
    man, raw = resolve(cfg, "manifests_dir"), resolve(cfg, "raw_dir")

    # ---- training data
    rows = read_manifest(man / f"{cfg['data']['train_subset']}.csv")
    if args.limit:
        rows = random.Random(cfg["seed"]).sample(rows, args.limit)
        (run_dir / "train_utterances.txt").write_text("\n".join(r["utt_id"] for r in rows))
    bank = None
    if ncfg["train_noise"]:
        bank = NoiseBank(man / ncfg["manifest"], raw, pool="train",
                         categories=tuple(ncfg["categories"]))
    train_ds = ASRDataset(rows, raw, vocab, noise_bank=bank, p_noise=ncfg["p_noise"],
                          snr_range=(ncfg["snr_min_db"], ncfg["snr_max_db"]), seed=cfg["seed"])
    sampler = DurationBatchSampler(train_ds.durations(), lcfg["max_batch_seconds"],
                                   shuffle=True, seed=cfg["seed"])
    train_loader = make_loader(train_ds, sampler, lcfg["num_workers"], pin, persistent=True)
    dev_loaders, n_dev = build_dev(cfg, vocab, cfg["evaluation"]["dev_limit"],
                                   lcfg["num_workers"], pin)

    # ---- model and optimization
    model = CNNBiLSTM.from_config(cfg["model"], cfg["features"]["n_mels"], len(vocab)).to(device)
    features = LogMel(**cfg["features"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(total_steps, tcfg["warmup_fraction"], tcfg["min_lr_ratio"]))
    augment = SpecAugment.from_config(cfg.get("specaugment"))     # None for Models A and B
    if augment is not None:
        augment = augment.to(device)
    step = TrainStep(model, features, opt, clip_norm=tcfg["clip_norm"], use_amp=tcfg["amp"],
                     scheduler=sched, augment=augment)

    start_epoch, global_step, best = 1, 0, None
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step.scaler.load_state_dict(ck["scaler"])
        start_epoch, global_step, best = ck["epoch"] + 1, ck["global_step"], ck["best"]
        print(f"Resumed from epoch {ck['epoch']} (best so far: {best})")

    # ---- record the setup
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(command=" ".join(sys.argv), python=platform.python_version(),
               torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               parameters=count_parameters(model), train_utterances=len(rows),
               train_hours=round(sum(r["duration_s"] for r in rows) / 3600, 2),
               dev_utterances=n_dev, steps_per_epoch=len(sampler), total_steps=total_steps)
    env_path = run_dir / "environment.json"
    if env_path.exists():                         # resumed run: keep the original record
        old = json.loads(env_path.read_text())
        old.setdefault("resumed", []).append(dict(command=env["command"], epoch=start_epoch,
                                                  time=time.strftime("%Y-%m-%d %H:%M")))
        env = old
    env_path.write_text(json.dumps(env, indent=2))
    print(f"run {name}: {len(rows)} training utterances ({env['train_hours']} h), "
          f"noise {'ON' if bank else 'OFF'}, SpecAugment {'ON' if augment is not None else 'OFF'}, "
          f"{len(sampler)} steps/epoch x {tcfg['epochs']} epochs")
    if augment is not None:
        print(f"SpecAugment({augment.extra_repr()})")
    print(f"model {env['parameters'] / 1e6:.1f}M params on {device}; dev {n_dev} utterances x "
          f"{len(dev_loaders)} conditions; outputs in {run_dir}")

    # ---- training loop
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        set_seed(cfg["seed"] + epoch)
        sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        skipped_before = step.skipped
        losses, window, wgrad = [], [], []
        for k, batch in enumerate(train_loader, 1):
            loss, grad = step(batch, device)
            global_step += 1
            losses.append(loss)
            window.append(loss)
            wgrad.append(grad)
            if k % tcfg["log_every_steps"] == 0 or k == len(sampler):
                el = time.perf_counter() - t0
                lr = sched.get_last_lr()[0]
                print(f"  epoch {epoch} step {k}/{len(sampler)}  loss {sum(window) / len(window):.4f}"
                      f"  grad {max(wgrad):.2f}  lr {lr:.2e}  {el / k:.2f}s/step"
                      f"  eta {el / k * (len(sampler) - k) / 60:.1f} min", flush=True)
                append_csv(run_dir / "steps.csv", dict(
                    epoch=epoch, step=global_step, loss=round(sum(window) / len(window), 5),
                    max_grad=round(max(wgrad), 3), lr=lr))
                window, wgrad = [], []
        train_time = time.perf_counter() - t0

        # ---- dev evaluation, every condition
        t1 = time.perf_counter()
        row = dict(epoch=epoch, train_loss=round(sum(losses) / len(losses), 5),
                   lr=sched.get_last_lr()[0], train_min=round(train_time / 60, 2),
                   sec_per_step=round(train_time / len(losses), 3),
                   skipped_steps=step.skipped - skipped_before)
        preds = {}
        for cond, loader in dev_loaders.items():
            ev = evaluate(model, features, loader, device, vocab, use_amp=tcfg["amp"])
            er = error_rates(ev["refs"], ev["hyps"], vocab)
            row[f"{cond}_loss"] = round(ev["loss"], 4)
            row[f"{cond}_wer"] = round(er["wer"], 4)
            row[f"{cond}_cer"] = round(er["cer"], 4)
            row[f"{cond}_blank"] = round(ev["blank_fraction"], 3)
            preds[cond] = ev
        conds = list(dev_loaders)
        row["mean_wer"] = round(sum(row[f"{c}_wer"] for c in conds) / len(conds), 4)
        row["eval_min"] = round((time.perf_counter() - t1) / 60, 2)
        append_csv(run_dir / "log.csv", row)

        improved = best is None or row["mean_wer"] < best["mean_wer"]
        if improved:
            best = dict(epoch=epoch, mean_wer=row["mean_wer"])
        save_checkpoint(run_dir / "last.pt", model, opt, sched, step.scaler, epoch, global_step,
                        best, cfg)
        if improved:
            save_checkpoint(run_dir / "best.pt", model, opt, sched, step.scaler, epoch,
                            global_step, best, cfg)
            with open(run_dir / "dev_predictions_best.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["condition", "utt_id", "reference", "hypothesis"])
                for cond, ev in preds.items():
                    for u, r_, h in zip(ev["utt_ids"], ev["refs"], ev["hyps"]):
                        w.writerow([cond, u, r_, h])

        print(f"epoch {epoch}: train loss {row['train_loss']:.4f} ({row['train_min']} min) | dev WER "
              + "  ".join(f"{c}: {row[f'{c}_wer']:.1%}" for c in conds)
              + f" | mean {row['mean_wer']:.1%}{'  * best' if improved else ''}"
              f" | eval {row['eval_min']} min", flush=True)
        ex = preds[conds[0]]
        print(f"    ref: {ex['refs'][0][:80]}\n    hyp: {ex['hyps'][0][:80]}", flush=True)

    print(f"\nDone. Best epoch {best['epoch']} (mean dev WER {best['mean_wer']:.2%}). "
          f"Results in {run_dir}")


if __name__ == "__main__":        # required on Windows for DataLoader workers
    main()
