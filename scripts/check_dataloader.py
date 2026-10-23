"""
Phase 8b: check the Dataset, sampler, collation and DataLoader.

Run from the project root (after make_noisy_recipes.py):
    python scripts/check_dataloader.py

Sections
  1. clean items: labels decode back to the transcript
  2. fixed recipe items: identical to mixing by hand, reproducible, same
     noise segment at every SNR
  3. training noise items: about p_noise of them noisy, SNR inside the range,
     identical for the same (epoch, index), different across epochs
  4. collation: shapes, lengths, zero padding
  5. duration batching on the full training set: every utterance exactly
     once per epoch, padded audio under the cap, padding efficiency
  6. end to end DataLoader with worker processes -> GPU log Mel features,
     throughput and estimated epoch time
"""

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.dataset import (ASRDataset, DurationBatchSampler, collate,  # noqa: E402
                         padding_efficiency, read_recipe)
from src.features import LogMel, num_frames  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.mixing import mix_at_snr  # noqa: E402
from src.musan import NoiseBank, load_segment  # noqa: E402
from src.audio import load_audio  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def header(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    cfg = load_config("configs/base.yaml")
    man, raw = resolve(cfg, "manifests_dir"), resolve(cfg, "raw_dir")
    ncfg, lcfg = cfg["noise"], cfg["loader"]
    vocab = Vocabulary()
    dev_rows = read_manifest(man / f"{cfg['data']['dev_subset']}.csv")
    train_rows = read_manifest(man / f"{cfg['data']['train_subset']}.csv")
    print(f"config: seed {cfg['seed']}, p_noise {ncfg['p_noise']}, SNR "
          f"{ncfg['snr_min_db']} to {ncfg['snr_max_db']} dB, eval {ncfg['eval_snrs_db']}, "
          f"max_batch_seconds {lcfg['max_batch_seconds']}, workers {lcfg['num_workers']}")

    header("1. CLEAN ITEMS")
    clean = ASRDataset(dev_rows, raw, vocab)
    ok = True
    for i in random.Random(0).sample(range(len(clean)), 20):
        item = clean[i]
        ok &= vocab.decode(item["labels"].tolist()) == item["transcript"]
        ok &= item["snr_db"] is None and item["wav"].dtype == torch.float32
        ok &= bool((item["labels"] != vocab.blank_id).all())
    check("20 dev items: labels decode to transcript, no blank, float32 audio", ok)

    header("2. FIXED RECIPE ITEMS (dev set)")
    recipe = read_recipe(man / "recipes" / f"{cfg['data']['dev_subset']}_noise.csv")
    check("recipe covers every dev utterance", set(recipe) == {r["utt_id"] for r in dev_rows},
          f"{len(recipe)} rows")
    i = 7
    r = dev_rows[i]
    rec = recipe[r["utt_id"]]
    s = load_audio(raw / r["path"])
    n = torch.from_numpy(load_segment(raw / rec["noise_path"], rec["start"], s.numel(),
                                      rec["noise_file_samples"]))
    for snr in ncfg["eval_snrs_db"]:
        ds = ASRDataset(dev_rows, raw, vocab, recipe=recipe, snr_db=snr)
        by_hand, info = mix_at_snr(s, n, snr)
        a, b = ds[i], ds[i]
        check(f"{snr:>2} dB: dataset equals mixing by hand, reproducible",
              torch.equal(a["wav"], by_hand) and torch.equal(a["wav"], b["wav"]),
              f"achieved {info['achieved_snr_db']} dB, noise {Path(rec['noise_path']).name}")

    header("3. TRAINING NOISE ITEMS (train set, train pool)")
    bank = NoiseBank(man / ncfg["manifest"], raw, pool="train",
                     categories=tuple(ncfg["categories"]))
    subset = [train_rows[k] for k in random.Random(1).sample(range(len(train_rows)), 300)]
    tr = ASRDataset(subset, raw, vocab, noise_bank=bank, p_noise=ncfg["p_noise"],
                    snr_range=(ncfg["snr_min_db"], ncfg["snr_max_db"]), seed=cfg["seed"])
    t0 = time.perf_counter()
    snrs = [tr[(k, 0)]["snr_db"] for k in range(len(tr))]
    per_item = (time.perf_counter() - t0) / len(tr) * 1000
    noisy = [x for x in snrs if x is not None]
    frac = len(noisy) / len(snrs)
    check(f"fraction noisy close to p_noise={ncfg['p_noise']}", abs(frac - ncfg["p_noise"]) < 0.07,
          f"{frac:.2f} of 300")
    check("all SNRs inside the training range",
          all(ncfg["snr_min_db"] <= x <= ncfg["snr_max_db"] for x in noisy),
          f"{min(noisy):.1f} to {max(noisy):.1f} dB")
    a, b, c = tr[(5, 0)], tr[(5, 0)], tr[(5, 1)]
    check("same (index, epoch) -> identical item", torch.equal(a["wav"], b["wav"]))
    differs = any(not torch.equal(tr[(k, 0)]["wav"], tr[(k, 1)]["wav"]) for k in range(5, 10))
    check("different epoch -> different noise", differs)
    check("labels unchanged by noise", torch.equal(a["labels"], c["labels"]))
    print(f"loading + mixing one training item: {per_item:.1f} ms")

    header("4. COLLATION")
    items = [clean[k] for k in range(4)]
    batch = collate(items)
    B = len(items)
    shapes_ok = (batch["wav"].shape == (B, int(batch["wav_lens"].max()))
                 and batch["targets"].shape == (B, int(batch["target_lens"].max())))
    pad_ok = all(bool((batch["wav"][k, batch["wav_lens"][k]:] == 0).all())
                 and bool((batch["targets"][k, batch["target_lens"][k]:] == 0).all())
                 for k in range(B))
    data_ok = all(torch.equal(batch["wav"][k, :batch["wav_lens"][k]], items[k]["wav"])
                  for k in range(B))
    print(f"wav {tuple(batch['wav'].shape)}  wav_lens {batch['wav_lens'].tolist()}")
    print(f"targets {tuple(batch['targets'].shape)}  target_lens {batch['target_lens'].tolist()}")
    check("shapes [B, N_max] and [B, L_max]", shapes_ok)
    check("padding is zero, real data intact", pad_ok and data_ok)

    header("5. DURATION BATCHING (full train set)")
    durs = np.array([r["duration_s"] for r in train_rows])
    cap = lcfg["max_batch_seconds"]
    sampler = DurationBatchSampler(durs, cap, shuffle=True, seed=cfg["seed"])
    e0 = [list(b) for b in sampler.batches]
    seen = sorted(i for b in e0 for i in b)
    check("every utterance exactly once per epoch", seen == list(range(len(train_rows))),
          f"{len(e0)} batches")
    over = [b for b in e0 if len(b) > 1 and len(b) * durs[b].max() > cap]
    check(f"padded audio <= {cap} s per batch", not over, f"{len(over)} violations")
    sizes = [len(b) for b in e0]
    print(f"batch size: min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}")
    sampler.set_epoch(1)
    e1 = [list(b) for b in sampler.batches]
    sampler.set_epoch(0)
    check("epochs differ, same epoch reproducible", e1 != e0 and sampler.batches == e0)
    eff = padding_efficiency(e0, durs)
    rnd = np.random.default_rng(0).permutation(len(durs))
    med = int(np.median(sizes))
    rand_batches = [rnd[k:k + med].tolist() for k in range(0, len(rnd), med)]
    check("duration batching wastes less than random batching",
          eff > padding_efficiency(rand_batches, durs),
          f"real speech fraction {eff:.1%} vs {padding_efficiency(rand_batches, durs):.1%} random")

    header("6. DATALOADER -> GPU FEATURES")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = ASRDataset(train_rows, raw, vocab, noise_bank=bank, p_noise=ncfg["p_noise"],
                      snr_range=(ncfg["snr_min_db"], ncfg["snr_max_db"]), seed=cfg["seed"])
    workers = lcfg["num_workers"]
    loader = DataLoader(full, batch_sampler=sampler, collate_fn=collate,
                        num_workers=workers, pin_memory=lcfg["pin_memory"] and device.type == "cuda",
                        persistent_workers=workers > 0)
    fe = LogMel(**cfg["features"]).to(device)
    n_batches, n_utts, t0 = 30, 0, None
    lens_ok = True
    for k, batch in enumerate(loader):
        if k == 2:
            t0 = time.perf_counter()                 # skip worker start up
        wav = batch["wav"].to(device, non_blocking=True)
        lens = batch["wav_lens"].to(device)
        feats, feat_lens = fe(wav, lens)
        lens_ok &= bool((feat_lens.cpu() == num_frames(batch["wav_lens"])).all())
        lens_ok &= feats.shape[:2] == (wav.shape[0], cfg["features"]["n_mels"])
        if k >= 2:
            n_utts += wav.shape[0]
        if k == 2:
            print(f"example batch: wav {tuple(wav.shape)} -> features {tuple(feats.shape)}, "
                  f"targets {tuple(batch['targets'].shape)}")
        if k + 1 >= n_batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    rate = n_utts / secs
    check("feature lengths = wav_lens // 160 + 1, shape [B, 80, T]", lens_ok)
    print(f"throughput with {workers} worker(s): {rate:.1f} utterances/s "
          f"-> about {len(train_rows) / rate / 60:.1f} min of data loading per epoch")

    print("\nRESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":      # required on Windows for DataLoader workers
    main()
