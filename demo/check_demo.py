"""
Phase 19 check: the demo's inference path gives the same transcripts as the
evaluation (scripts/evaluate.py) for the same audio.

Run from the project root, after scripts/evaluate.py --split test:
    python demo/check_demo.py

For 20 seeded test utterances, clean and at 0 dB (the fixed test recipe, i.e.
byte identical audio to the evaluation), every model's demo transcript
(src.inference.Transcriber, one utterance at a time) is compared with the
hypothesis saved in results/eval/test/predictions_<model>.csv (batched).
Batching cannot change the output (verified in Phase 9), so they should be
identical; with mixed precision on the GPU an occasional one letter difference
is tolerated. PASS: >= 90% identical and any difference below 5% CER.
"""

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402
import torch  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.dataset import ASRDataset, read_recipe  # noqa: E402
from src.inference import Transcriber  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

MODELS = ["model_a_clean", "model_b_noise", "model_c_lb"]
N, CONDITIONS = 20, ["clean", "0"]


def main():
    cfg = load_config("configs/base.yaml")
    man, raw = resolve(cfg, "manifests_dir"), resolve(cfg, "raw_dir")
    vocab = Vocabulary()
    rows = read_manifest(man / f"{cfg['data']['test_subset']}.csv")
    rows = random.Random(0).sample(rows, N)
    recipe = read_recipe(man / "recipes" / f"{cfg['data']['test_subset']}_noise.csv")
    data = {"clean": ASRDataset(rows, raw, vocab),
            "0": ASRDataset(rows, raw, vocab, recipe=recipe, snr_db=0.0)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = same = 0
    worst = 0.0
    for name in MODELS:
        ck = ROOT / "experiments" / name / "best.pt"
        pred = ROOT / "results" / "eval" / "test" / f"predictions_{name}.csv"
        if not ck.exists() or not pred.exists():
            print(f"skipping {name}: checkpoint or predictions missing")
            continue
        saved = {(r["condition"], r["utt_id"]): r["hypothesis"]
                 for r in csv.DictReader(open(pred, encoding="utf-8"))}
        asr = Transcriber.from_checkpoint(ck, device=device)
        n_same = 0
        for cond in CONDITIONS:
            for k in range(N):
                item = data[cond][k]
                demo = vocab.normalize(asr(item["wav"]))
                ref = saved[(cond, item["utt_id"])]
                if demo == ref:
                    n_same += 1
                else:
                    cer = jiwer.cer(ref, demo) if ref else 1.0
                    worst = max(worst, cer)
                    print(f"  differs {name} {cond} {item['utt_id']} (CER {100 * cer:.1f}%)\n"
                          f"    eval: {ref}\n    demo: {demo}")
        print(f"{name}: {n_same}/{2 * N} demo transcripts identical to the evaluation")
        total += 2 * N
        same += n_same
    if not total:
        sys.exit("nothing compared")
    ok = same / total >= 0.9 and worst < 0.05
    print(f"\n{same}/{total} identical, largest difference {100 * worst:.1f}% CER on {device}")
    print("[PASS] demo matches the evaluation" if ok else "[FAIL] demo differs from the evaluation")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
