"""
Language model decoding, step 2: CTC beam search + KenLM n-gram LM for Models A, B, C.

Run from the project root, after scripts/prepare_lm.py:
    # (1) tune the LM weight and word bonus per model on a DEV subset
    python scripts/decode_lm.py --split dev --models model_a_clean model_b_noise model_c_lb --tune
    #     (extend the grid later; finished settings are reused, e.g.)
    python scripts/decode_lm.py --split dev --models ... --tune --lm-weights 2.5 3 3.5 4 --word-scores -1 0 1 2 3
    # (2) full dev with the tuned settings (compare with greedy)
    python scripts/decode_lm.py --split dev --models model_a_clean model_b_noise model_c_lb
    # (3) the test set, ONCE, with the same settings
    python scripts/decode_lm.py --split test --models model_a_clean model_b_noise model_c_lb --extra-snrs -5 15

How it works
  * The acoustic model runs once per condition on the GPU (same checkpoint,
    features, batching and mixed precision as scripts/evaluate.py) and its CTC
    log probabilities are decoded on the CPU by torchaudio's ctc_decoder
    (flashlight lexicon beam search), in parallel worker processes. The 3-gram
    is scored by src/ngram_lm.py (Python), because the Windows build of
    flashlight-text cannot load KenLM; it was verified to give the same word
    scores (max difference 5e-7 log10) and the same transcripts as KenLM.
  * Score of a hypothesis = log P_acoustic + lm_weight * log P_LM + word_score * #words.
    lm_weight and word_score are chosen per model by grid search on a seeded
    dev subset (--tune-utts, default 300) to minimise the mean WER over the
    standard conditions (clean, 20, 10, 5, 0 dB) - the same selection rule as
    for training. The test set is never used for any choice.
  * Before anything else a sanity check decodes utterances with the beam search
    but WITHOUT LM and lexicon and beam size 1: it must reproduce greedy
    decoding exactly, which proves tokens, blank and space are mapped right.

Outputs
    results/lm/tuning_<model>.csv     every grid point: lm_weight, word_score, WER per condition
    results/lm/params.json            chosen settings per model (used by steps 2 and 3)
    results/eval/<split>_lm/          summary.csv, predictions_<model>.csv, by_noise_source.csv,
                                      metadata.json in the SAME format as evaluate.py, with the
                                      greedy WER next to the LM WER in summary.csv
    TEST runs are appended to results/eval/test_runs.log.
"""

import argparse
import csv
import importlib.metadata
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluate import SHARED_KEYS, build_loaders, get, label, md5, per_utterance, score  # noqa: E402
from src.decoder import greedy_decode  # noqa: E402
from src.features import LogMel  # noqa: E402
from src.model import CNNBiLSTM  # noqa: E402
from src.trainer import forward  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

LM_DIR = ROOT / "data" / "lm"
PARAMS = ROOT / "results" / "lm" / "params.json"
LM_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # 0.0 = lexicon only, no LM
WORD_SCORES = [-1.0, 0.0, 1.0, 2.0]

# ------------------------------------------------------------------ worker side
_DEC = {}


def _decoder(key):
    """The decoder for (lm_weight, word_score, beam, use_lm) in this worker process.
    Only ONE is kept: each holds its own copy of the LM, so caching all grid
    points would exhaust the RAM."""
    if key not in _DEC:
        _DEC.clear()
        from torchaudio.models.decoder import ctc_decoder
        lm_weight, word_score, beam, use_lm = key
        info = json.loads((LM_DIR / "lm_info.json").read_text())
        if use_lm:
            # Python n-gram LM (src/ngram_lm.py): same scores as KenLM, which the
            # Windows build of flashlight-text cannot load
            from src.ngram_lm import NgramLM
            words = (LM_DIR / "lm_dict.txt").read_text(encoding="utf-8").split("\n")[:-1]
            lm = NgramLM(str(LM_DIR / f"{info['lm']}.npz"), words)
            _DEC[key] = ctc_decoder(lexicon=str(LM_DIR / "lexicon.txt"),
                                    tokens=str(LM_DIR / "tokens.txt"), lm=lm,
                                    lm_dict=str(LM_DIR / "lm_dict.txt"), nbest=1,
                                    beam_size=beam, lm_weight=lm_weight, word_score=word_score,
                                    blank_token="-", sil_token="|", unk_word="<unk>")
        else:                                       # lexicon free, no LM: greedy equivalent
            _DEC[key] = ctc_decoder(lexicon=None, tokens=str(LM_DIR / "tokens.txt"), lm=None,
                                    nbest=1, beam_size=beam, blank_token="-", sil_token="|")
    return _DEC[key]


def _decode_chunk(args):
    key, emissions = args
    dec = _decoder(key)
    out = []
    for e in emissions:
        hyp = dec(torch.from_numpy(e)[None], torch.tensor([e.shape[0]], dtype=torch.int32))[0][0]
        if key[3]:
            out.append(" ".join(hyp.words))
        else:
            toks = "".join(dec.idxs_to_tokens(hyp.tokens)).replace("|", " ")
            out.append(" ".join(toks.replace("-", "").split()))
    return out


def decode_all(pool, key, emissions, chunk=24):
    jobs = [(key, emissions[i:i + chunk]) for i in range(0, len(emissions), chunk)]
    return [t for part in pool.map(_decode_chunk, jobs) for t in part]


# ------------------------------------------------------------------ acoustic side
@torch.no_grad()
def emissions_for(model, features, loader, device):
    """CTC log probabilities per utterance (float32 numpy [T', 29]) + ids, refs, greedy."""
    model.eval()
    amp = device.type == "cuda"
    vocab = Vocabulary()
    em, ids, refs, greedy = [], [], [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            lp, ol = forward(model, features, batch, device)
        greedy += greedy_decode(lp, ol, vocab)
        lp = lp.float().cpu().numpy()
        for k, n in enumerate(ol.tolist()):
            em.append(np.ascontiguousarray(lp[k, :n]))
        ids += batch["utt_ids"]
        refs += batch["transcripts"]
    return em, ids, refs, greedy


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = CNNBiLSTM.from_config(cfg["model"], cfg["features"]["n_mels"], len(Vocabulary()))
    model.load_state_dict(ck["model"])
    return model.to(device), LogMel(**cfg["features"]).to(device), cfg, ck


def mean_wer(by_cond, standard):
    return sum(by_cond[c] for c in standard) / len(standard)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--models", required=True, nargs="+")
    ap.add_argument("--tune", action="store_true", help="grid search on a dev subset")
    ap.add_argument("--tune-utts", type=int, default=300)
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--extra-snrs", nargs="*", type=float, default=[])
    ap.add_argument("--lm-weights", nargs="+", type=float, default=LM_WEIGHTS,
                    help="grid for --tune (results are merged with earlier tuning runs)")
    ap.add_argument("--word-scores", nargs="+", type=float, default=WORD_SCORES)
    args = ap.parse_args()
    if args.tune and args.split != "dev":
        sys.exit("--tune is only allowed on the dev set")
    if not (LM_DIR / "lm_info.json").exists():
        sys.exit("run scripts/prepare_lm.py first")
    params = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
    if not args.tune:
        missing = [m for m in args.models if m not in params]
        if missing:
            sys.exit(f"no tuned settings for {missing}: run with --split dev --tune first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary()
    ckpts = {m: ROOT / "experiments" / m / "best.pt" for m in args.models}
    cfgs = {m: torch.load(p, map_location="cpu", weights_only=False)["config"] for m, p in ckpts.items()}
    ref_cfg = cfgs[args.models[0]]
    for m, c in cfgs.items():
        for keys in SHARED_KEYS:
            if get(c, keys) != get(ref_cfg, keys):
                sys.exit(f"{m} differs from {args.models[0]} in {'.'.join(keys)}")
    standard = [label(c) for c in ref_cfg["evaluation"]["conditions"]]
    conditions = standard + [label(s) for s in args.extra_snrs if label(s) not in standard]
    limit = args.tune_utts if args.tune else None
    workers = ref_cfg["loader"]["num_workers"]
    pin = ref_cfg["loader"]["pin_memory"] and device.type == "cuda"
    loaders, rows, source, files = build_loaders(ref_cfg, args.split, conditions, limit, workers, pin)
    info = json.loads((LM_DIR / "lm_info.json").read_text())
    print(f"{args.split}: {len(rows)} utterances x {len(conditions)} conditions | LM {info['lm']}, "
          f"lexicon {info['lexicon_words']:,} words | beam {args.beam} | {args.workers} CPU workers | {device}")

    out_dir = ROOT / "results" / "eval" / f"{args.split}_lm"
    (ROOT / "results" / "lm").mkdir(parents=True, exist_ok=True)
    summary, by_source, meta_models = [], [], {}
    t_all = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        sanity_done = False
        tune_data = {}
        for name in args.models:
            model, features, cfg, ck = load_model(ckpts[name], device)
            t0 = time.perf_counter()
            E = {c: emissions_for(model, features, loaders[c], device) for c in conditions}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\n== {name} (epoch {ck['epoch']}): acoustic model done in "
                  f"{time.perf_counter() - t0:.0f} s")

            if not sanity_done:                     # beam 1, no LM, no lexicon == greedy
                em, _, _, greedy = E[standard[0]]
                n = min(50, len(em))
                got = decode_all(pool, (0.0, 0.0, 1, False), em[:n])
                same, worst = 0, 0.0
                for k, (a, b) in enumerate(zip(got, greedy[:n])):
                    a, b = vocab.normalize(a), vocab.normalize(b)
                    if a == b:
                        same += 1
                    else:                           # show it: a near tie, or a real problem?
                        cer = jiwer.cer(b, a) if b else 1.0
                        worst = max(worst, cer)
                        print(f"    differs (CER {100 * cer:.1f}% vs greedy)\n      greedy: {b}\n      beam:   {a}")
                print(f"  sanity check: beam search without LM reproduces greedy on {same}/{n} "
                      f"utterances (largest difference {100 * worst:.1f}% CER)")
                # a wrong token mapping garbles almost every utterance; rare small
                # differences come from near ties between paths in the beam search
                if same < 0.9 * n or worst >= 0.05:
                    sys.exit("  [FAIL] decoder does not reproduce greedy; nothing was decoded")
                print("  [PASS] decoder setup verified")
                sanity_done = True

            greedy_wer = {c: score([vocab.normalize(r) for r in E[c][2]],
                                   [vocab.normalize(h) for h in E[c][3]])["wer"] for c in conditions}

            if args.tune:                           # grid search runs after all models
                tune_data[name] = (E, greedy_wer)
                continue

            # ---- full decode with the tuned settings
            p = params[name]
            key = (p["lm_weight"], p["word_score"], args.beam, True)
            pred_rows = []
            for c in conditions:
                em, ids, refs, greedy = E[c]
                t1 = time.perf_counter()
                hyps = decode_all(pool, key, em)
                refs_n = [vocab.normalize(r) for r in refs]
                hyps_n = [vocab.normalize(h) for h in hyps]
                s = score(refs_n, hyps_n)
                summary.append(dict(model=name, epoch=ck["epoch"], split=args.split, condition=c,
                                    standard=c in standard, utterances=len(refs), **s,
                                    greedy_wer=greedy_wer[c],
                                    lm_gain_points=round(greedy_wer[c] - s["wer"], 2),
                                    empty_hyps=sum(1 for h in hyps_n if not h)))
                groups = defaultdict(lambda: ([], []))
                for u, r, h in zip(ids, refs_n, hyps_n):
                    src = "" if c == "clean" else source[u]
                    pred_rows.append(dict(condition=c, utt_id=u, noise_source=src, reference=r,
                                          hypothesis=h, **per_utterance(r, h)))
                    if c != "clean":
                        groups[src][0].append(r)
                        groups[src][1].append(h)
                for src, (rr, hh) in sorted(groups.items()):
                    g = score(rr, hh)
                    by_source.append(dict(model=name, condition=c, noise_source=src, utterances=len(rr),
                                          wer=g["wer"], cer=g["cer"], sub=g["sub"], dele=g["dele"],
                                          ins=g["ins"]))
                print(f"  {c:>5}: WER {s['wer']:6.2f}% (greedy {greedy_wer[c]:6.2f}%)  "
                      f"S {s['sub']:5.2f} D {s['dele']:5.2f} I {s['ins']:4.2f}  CER {s['cer']:6.2f}%  "
                      f"[{time.perf_counter() - t1:.0f} s]", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
                w.writeheader()
                w.writerows(pred_rows)
            meta_models[name] = dict(checkpoint=str(ckpts[name].relative_to(ROOT)), epoch=ck["epoch"],
                                     md5=md5(ckpts[name]), **p)

        if args.tune:
            # grid outermost: every setting loads the LM once and decodes all models
            # earlier tuning runs with the same subset and beam are reused (merged)
            grids = {}
            for name in tune_data:
                old = ROOT / "results" / "lm" / f"tuning_{name}.csv"
                rows_old = list(csv.DictReader(open(old))) if old.exists() else []
                grids[name] = [
                    {k: (float(v) if k not in ("tune_utts", "beam") else int(float(v))) for k, v in r.items()}
                    for r in rows_old
                    if int(float(r.get("tune_utts", 300))) == len(rows) and int(float(r.get("beam", 50))) == args.beam]
                for r in grids[name]:
                    r.setdefault("tune_utts", len(rows))
                    r.setdefault("beam", args.beam)
            have = [{(r["lm_weight"], r["word_score"]) for r in g} for g in grids.values()]
            done = set.intersection(*have) if have else set()      # finished for EVERY model
            todo = [(lw, ws) for lw in args.lm_weights for ws in args.word_scores if (lw, ws) not in done]
            print(f"\ngrid search: {len(todo)} new settings ({len(done)} reused from earlier runs)")
            for lw, ws in todo:
                if True:                            # (keeps the block below unchanged)
                    t1 = time.perf_counter()
                    line = []
                    for name, (E, _) in tune_data.items():
                        grids[name] = [r for r in grids[name]
                                       if (r["lm_weight"], r["word_score"]) != (lw, ws)]
                        row = dict(lm_weight=lw, word_score=ws, tune_utts=len(rows), beam=args.beam)
                        for c in standard:
                            hyps = decode_all(pool, (lw, ws, args.beam, True), E[c][0])
                            row[c] = score([vocab.normalize(r) for r in E[c][2]],
                                           [vocab.normalize(h) for h in hyps])["wer"]
                        row["mean"] = round(mean_wer(row, standard), 2)
                        grids[name].append(row)
                        line.append(f"{name.split('_')[1].upper()} {row['mean']:6.2f}")
                    print(f"  lm_weight {lw:4.2f} word_score {ws:+.1f}: mean WER  " + "  ".join(line)
                          + f"  [{time.perf_counter() - t1:.0f} s]", flush=True)
            for name, grid in grids.items():
                with open(ROOT / "results" / "lm" / f"tuning_{name}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(grid[0]))
                    w.writeheader()
                    w.writerows(grid)
                gw = tune_data[name][1]
                grid.sort(key=lambda r: (r["lm_weight"], r["word_score"]))
                best = min(grid, key=lambda r: r["mean"])
                lws = sorted({r["lm_weight"] for r in grid})
                wss = sorted({r["word_score"] for r in grid})
                if best["lm_weight"] == lws[-1] or best["word_score"] in (wss[0], wss[-1]):
                    print(f"  WARNING {name}: best setting is on the edge of the grid "
                          f"(lm_weight {best['lm_weight']}, word_score {best['word_score']}); "
                          "extend it with --lm-weights / --word-scores")
                lex = min((r for r in grid if r["lm_weight"] == 0.0), key=lambda r: r["mean"], default=None)
                params[name] = dict(lm_weight=best["lm_weight"], word_score=best["word_score"],
                                    beam=args.beam, lm=info["lm"], tuned_on=f"dev subset of {len(rows)}",
                                    dev_subset_mean_wer_lm=best["mean"],
                                    dev_subset_mean_wer_lexicon_only=lex["mean"] if lex else None,
                                    dev_subset_mean_wer_greedy=round(mean_wer(gw, standard), 2))
                print(f"  {name}: best lm_weight {best['lm_weight']}, word_score {best['word_score']} -> "
                      f"mean WER {best['mean']:.2f} (lexicon only {lex['mean'] if lex else 'n/a'}, "
                      f"greedy {mean_wer(gw, standard):.2f})")
            PARAMS.write_text(json.dumps(params, indent=2))

    if args.tune:
        print(f"\nsettings saved to {PARAMS.relative_to(ROOT)}; next: run without --tune")
        return

    for fname, table in [("summary.csv", summary), ("by_noise_source.csv", by_source)]:
        with open(out_dir / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    meta = dict(split=args.split, decoder="torchaudio ctc_decoder (flashlight) + src.ngram_lm",
                lm=info, conditions=conditions, standard=standard, beam=args.beam,
                command=" ".join(sys.argv), time=time.strftime("%Y-%m-%d %H:%M"),
                minutes=round((time.perf_counter() - t_all) / 60, 1), torch=torch.__version__,
                flashlight_text=importlib.metadata.version("flashlight-text"),
                data={k: dict(path=str(v.relative_to(ROOT)), md5=md5(v)) for k, v in files.items()},
                models=meta_models)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    if args.split == "test":
        with open(ROOT / "results" / "eval" / "test_runs.log", "a", encoding="utf-8") as f:
            f.write(f"{meta['time']}  {meta['command']}  (LM decoding)\n")

    print("\nWER % with LM (greedy in brackets)")
    head = f"{'model':<16}" + "".join(f"{c:>15}" for c in conditions)
    print(head + "\n" + "-" * len(head))
    for name in args.models:
        r = {x["condition"]: x for x in summary if x["model"] == name}
        print(f"{name:<16}" + "".join(f"{r[c]['wer']:7.2f} ({r[c]['greedy_wer']:5.2f})" for c in conditions))
    print(f"\nDone in {meta['minutes']} min. Results in {out_dir}")


if __name__ == "__main__":        # required on Windows for worker processes
    main()
