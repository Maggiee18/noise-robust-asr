"""
Phase 16: evaluate trained models on the dev or test set, in every noise condition.

Run from the project root:
    # 1. verification on DEV: must reproduce the dev numbers logged during training
    python scripts/evaluate.py --split dev --models model_a_clean model_b_noise

    # 2. the final evaluation on TEST (once, after all models are trained)
    python scripts/evaluate.py --split test --models model_a_clean model_b_noise model_c_lb --extra-snrs -5 15

What it does, for every model
  * loads experiments/<model>/best.pt (the epoch chosen on dev) and rebuilds the
    model and features from the config saved INSIDE the checkpoint
  * decodes every utterance of the split in each condition: clean plus the fixed
    noisy recipes at 20 / 10 / 5 / 0 dB (dev: dev noise pool, test: test noise
    pool, never heard in training). --extra-snrs adds SNRs no model was trained
    or selected on (e.g. -5 and 15 dB) using the SAME noise segments
  * scores with the same normalization as training (Vocabulary.normalize):
    corpus level WER and CER, plus substitution / deletion / insertion rates
    (% of reference words or characters), loss per character, blank fraction
    and the number of empty hypotheses
  * also splits noisy results by MUSAN noise source (free-sound / sound-bible)

Outputs in results/eval/<split>/  (a --limit run goes to <split>_limit<N>/)
    summary.csv               one row per model x condition (all numbers above)
    by_noise_source.csv       WER / CER per model x condition x noise source
    predictions_<model>.csv   every reference / hypothesis with per utterance errors
    metadata.json             checkpoints (epoch, md5), data files (md5), versions, command
Every TEST run is also appended to results/eval/test_runs.log, so it is visible
how often the test set was used.

Design decisions (documented for the paper)
-------------------------------------------
* Same decoding, batching, mixed precision and scoring code as the dev
  evaluation during training (src/trainer.evaluate, greedy CTC). With
  --split dev the script therefore checks itself: the WER per condition must
  match log.csv at the best epoch.
* All models must share the same feature, data and noise recipe settings;
  the script refuses to compare models that do not.
* Noise segments are fixed per utterance and identical at every SNR, so the
  WER vs SNR curve measures the noise LEVEL only, and all models hear byte
  identical audio.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import resolve  # noqa: E402
from src.dataset import ASRDataset, DurationBatchSampler, collate, read_recipe  # noqa: E402
from src.features import LogMel  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.metrics import error_rates  # noqa: E402
from src.model import CNNBiLSTM  # noqa: E402
from src.trainer import evaluate  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

# settings that must be identical for a fair comparison between models
SHARED_KEYS = [("features",), ("data",), ("paths",), ("noise", "eval_snrs_db"),
               ("noise", "recipe_seed"), ("noise", "manifest"), ("seed",)]


def md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def get(cfg, keys):
    for k in keys:
        cfg = cfg[k]
    return cfg


def label(cond):
    """'clean' stays 'clean'; numbers become '20', '-5', '7.5'."""
    return "clean" if str(cond) == "clean" else f"{float(cond):g}"


def pct(x):
    return round(100 * x, 2)


def score(refs, hyps):
    """Corpus level word and character error breakdown on normalized text."""
    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    nw = w.substitutions + w.deletions + w.hits
    nc = c.substitutions + c.deletions + c.hits
    return dict(ref_words=nw, wer=pct(w.wer), sub=pct(w.substitutions / nw),
                dele=pct(w.deletions / nw), ins=pct(w.insertions / nw),
                ref_chars=nc, cer=pct(c.cer), char_sub=pct(c.substitutions / nc),
                char_del=pct(c.deletions / nc), char_ins=pct(c.insertions / nc))


def per_utterance(ref, hyp):
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    return dict(ref_words=len(ref.split()), sub=w.substitutions, dele=w.deletions,
                ins=w.insertions, ref_chars=len(ref),
                char_errors=c.substitutions + c.deletions + c.insertions)


def build_loaders(cfg, split, conditions, limit, workers, pin):
    man, raw = resolve(cfg, "manifests_dir"), resolve(cfg, "raw_dir")
    subset = cfg["data"][f"{split}_subset"]
    manifest = man / f"{subset}.csv"
    recipe_path = man / "recipes" / f"{subset}_noise.csv"
    rows = read_manifest(manifest)
    if limit:                                   # same seeded subset rule as train.py
        keep = set(random.Random(cfg["seed"]).sample(range(len(rows)), limit))
        rows = [r for k, r in enumerate(rows) if k in keep]
    recipe = read_recipe(recipe_path)
    loaders = {}
    for cond in conditions:
        if cond == "clean":
            ds = ASRDataset(rows, raw, Vocabulary())
        else:
            ds = ASRDataset(rows, raw, Vocabulary(), recipe=recipe, snr_db=float(cond))
        sampler = DurationBatchSampler(ds.durations(), cfg["loader"]["max_batch_seconds"],
                                       shuffle=False)
        loaders[cond] = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                                   num_workers=workers, pin_memory=pin)
    source = {u: r["noise_path"].split("/")[2] for u, r in recipe.items()}
    files = {"manifest": manifest, "recipe": recipe_path}
    return loaders, rows, source, files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--models", required=True, nargs="+",
                    help="experiment names, e.g. model_a_clean model_b_noise model_c_lb")
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--extra-snrs", nargs="*", type=float, default=[],
                    help="additional SNRs in dB beyond the standard conditions, e.g. -5 15")
    ap.add_argument("--limit", type=int, help="quick run on a seeded subset of N utterances")
    ap.add_argument("--workers", type=int, help="DataLoader workers (default: from the config)")
    ap.add_argument("--no-amp", action="store_true", help="float32 instead of mixed precision")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary()

    # ---- load every checkpoint first and check the models are comparable
    ckpts = {}
    for name in args.models:
        path = ROOT / "experiments" / name / f"{args.checkpoint}.pt"
        if not path.exists():
            sys.exit(f"missing checkpoint: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ckpts[name] = dict(path=path, ck=ck, cfg=ck["config"])
    ref_name, ref_cfg = args.models[0], ckpts[args.models[0]]["cfg"]
    for name, c in ckpts.items():
        for keys in SHARED_KEYS:
            if get(c["cfg"], keys) != get(ref_cfg, keys):
                sys.exit(f"{name} and {ref_name} differ in {'.'.join(keys)}: "
                         f"{get(c['cfg'], keys)} vs {get(ref_cfg, keys)}. Not comparable.")

    standard = [label(c) for c in ref_cfg["evaluation"]["conditions"]]
    conditions = standard + [label(s) for s in args.extra_snrs if label(s) not in standard]
    workers = ref_cfg["loader"]["num_workers"] if args.workers is None else args.workers
    pin = ref_cfg["loader"]["pin_memory"] and device.type == "cuda"
    use_amp = not args.no_amp
    loaders, rows, source, files = build_loaders(ref_cfg, args.split, conditions, args.limit,
                                                 workers, pin)

    out = ROOT / "results" / "eval" / (args.split + (f"_limit{args.limit}" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.split}: {len(rows)} utterances x {len(conditions)} conditions "
          f"({', '.join(conditions)}) | models: {', '.join(args.models)} | {device}"
          f"{' | AMP' if use_amp and device.type == 'cuda' else ''}")
    print(f"outputs in {out}")

    summary, by_source, meta_models = [], [], {}
    t_all = time.perf_counter()
    for name in args.models:
        c = ckpts[name]
        cfg, ck = c["cfg"], c["ck"]
        model = CNNBiLSTM.from_config(cfg["model"], cfg["features"]["n_mels"], len(vocab))
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        features = LogMel(**cfg["features"]).to(device)
        epoch, best = ck["epoch"], ck.get("best")
        print(f"\n== {name}: {args.checkpoint}.pt, epoch {epoch}"
              + (f", dev mean WER at selection {best['mean_wer']:.2%}" if best else ""))

        pred_rows = []
        for cond in conditions:
            t0 = time.perf_counter()
            ev = evaluate(model, features, loaders[cond], device, vocab, use_amp=use_amp)
            refs = [vocab.normalize(r) for r in ev["refs"]]
            hyps = [vocab.normalize(h) for h in ev["hyps"]]
            s = score(refs, hyps)
            # cross check against the scorer used during training
            er = error_rates(ev["refs"], ev["hyps"], vocab)
            assert abs(100 * er["wer"] - s["wer"]) < 0.01 and abs(100 * er["cer"] - s["cer"]) < 0.01
            empty = sum(1 for h in hyps if not h)
            summary.append(dict(model=name, epoch=epoch, split=args.split, condition=cond,
                                standard=cond in standard, utterances=len(refs), **s,
                                loss=round(ev["loss"], 4),
                                blank_fraction=round(ev["blank_fraction"], 3),
                                empty_hyps=empty))
            groups = defaultdict(lambda: ([], []))
            for u, r, h in zip(ev["utt_ids"], refs, hyps):
                pu = per_utterance(r, h)
                src = "" if cond == "clean" else source[u]
                pred_rows.append(dict(condition=cond, utt_id=u, noise_source=src,
                                      reference=r, hypothesis=h, **pu))
                if cond != "clean":
                    groups[src][0].append(r)
                    groups[src][1].append(h)
            for src, (rr, hh) in sorted(groups.items()):
                g = score(rr, hh)
                by_source.append(dict(model=name, condition=cond, noise_source=src,
                                      utterances=len(rr), wer=g["wer"], cer=g["cer"],
                                      sub=g["sub"], dele=g["dele"], ins=g["ins"]))
            print(f"  {cond:>5}: WER {s['wer']:6.2f}%  (S {s['sub']:5.2f} D {s['dele']:5.2f} "
                  f"I {s['ins']:4.2f})  CER {s['cer']:6.2f}%  empty {empty}  "
                  f"[{time.perf_counter() - t0:.0f} s]", flush=True)

        with open(out / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
            w.writeheader()
            w.writerows(pred_rows)
        meta_models[name] = dict(checkpoint=str(c["path"].relative_to(ROOT)), epoch=epoch,
                                 selection=best, md5=md5(c["path"]),
                                 train_noise=cfg["noise"]["train_noise"],
                                 specaugment=cfg.get("specaugment", {}).get("enabled", False))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- write the tables
    for fname, table in [("summary.csv", summary), ("by_noise_source.csv", by_source)]:
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    meta = dict(split=args.split, limit=args.limit, conditions=conditions, standard=standard,
                command=" ".join(sys.argv), time=time.strftime("%Y-%m-%d %H:%M"),
                minutes=round((time.perf_counter() - t_all) / 60, 1), device=str(device),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                amp=use_amp, python=platform.python_version(), torch=torch.__version__,
                jiwer=importlib.metadata.version("jiwer"),
                data={k: dict(path=str(v.relative_to(ROOT)), md5=md5(v)) for k, v in files.items()},
                models=meta_models)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    if args.split == "test":
        with open(ROOT / "results" / "eval" / "test_runs.log", "a", encoding="utf-8") as f:
            f.write(f"{meta['time']}  {meta['command']}  limit={args.limit}\n")

    # ---- overview table
    print("\nWER % by condition" + (" (mean over the standard conditions)"))
    head = f"{'model':<18}" + "".join(f"{c:>8}" for c in conditions) + f"{'mean':>8}"
    print(head + "\n" + "-" * len(head))
    for name in args.models:
        r = {x["condition"]: x for x in summary if x["model"] == name}
        mean = sum(r[c]["wer"] for c in standard) / len(standard)
        print(f"{name:<18}" + "".join(f"{r[c]['wer']:8.2f}" for c in conditions) + f"{mean:8.2f}")

    # ---- self check: dev numbers must match the training log at the best epoch
    fails = 0
    if args.split == "dev" and not args.limit:
        print("\nself check against experiments/<model>/log.csv at the evaluated epoch:")
        for name in args.models:
            log = ROOT / "experiments" / name / "log.csv"
            if not log.exists():
                continue
            ep = meta_models[name]["epoch"]
            row = next((x for x in csv.DictReader(open(log)) if int(x["epoch"]) == ep), None)
            if row is None:
                continue
            for x in summary:
                if x["model"] != name or f"{x['condition']}_wer" not in row:
                    continue
                logged = 100 * float(row[f"{x['condition']}_wer"])
                ok = abs(logged - x["wer"]) <= 0.05
                fails += not ok
                print(f"  [{'PASS' if ok else 'FAIL'}] {name} {x['condition']:>5}: "
                      f"now {x['wer']:.2f}%  logged {logged:.2f}%")
        print("ALL MATCH" if not fails else f"{fails} MISMATCHES")
    print(f"\nDone in {meta['minutes']} min. Results in {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":        # required on Windows for DataLoader workers
    main()
