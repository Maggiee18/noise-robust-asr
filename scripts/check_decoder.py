"""
Phase 11 check: greedy CTC decoder.

Run from the project root:
    python scripts/check_decoder.py

Sections
  1. worked examples, including _HH_ELL_O_ -> HELO and _HH_EL_LO_ -> HELLO
  2. edge cases: all blanks, zero frames, spaces, apostrophes
  3. padding frames are ignored
  4. round trip: for EVERY dev-clean transcript, build a frame path the way
     CTC allows (random stretching, blanks, a blank between repeats), turn it
     into log probabilities, decode -> must give the transcript exactly
  5. agreement with CTC: the decoded string is at least as likely under CTC
     as the single best path (P(string) sums over many paths, including it)
  6. untrained model on real dev audio: output is gibberish (expected), plus
     blank fraction and decoding speed
"""

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.dataset import ASRDataset, collate  # noqa: E402
from src.decoder import blank_fraction, greedy_decode  # noqa: E402
from src.features import LogMel  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.model import CNNBiLSTM  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

FAILS = []
V = Vocabulary()


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def header(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def from_path(symbols):
    """'_HH_ELL_O_' -> one hot log probs [1, T, C] and lengths ('_' = blank, '^' = space)."""
    ids = [0 if s == "_" else V.char2id[" " if s == "^" else s] for s in symbols]
    lp = torch.full((1, len(ids), len(V)), -20.0)
    for t, i in enumerate(ids):
        lp[0, t, i] = 0.0
    return lp, torch.tensor([len(ids)])


def decode_path(symbols, clean=True):
    lp, n = from_path(symbols)
    return greedy_decode(lp, n, V, clean=clean)[0]


def ctc_path(text, rng):
    """A random valid CTC frame path for text: each character repeated 1-3
    times, random blanks in between, a blank forced between repeats."""
    out = []
    for k, c in enumerate(text):
        if rng.random() < 0.5 or (k > 0 and text[k - 1] == c):
            out += [0] * rng.randint(1, 2)
        out += [V.char2id[c]] * rng.randint(1, 3)
    out += [0] * rng.randint(0, 2)
    return out


def main():
    header("1. WORKED EXAMPLES  ('_' = blank, '^' = space)")
    for path, want in [("_HH_ELL_O_", "HELO"), ("_HH_EL_LO_", "HELLO"),
                       ("CC_AAT", "CAT"), ("C_A_T", "CAT"), ("I^^T'SS", "I T'S"),
                       ("II_T'^S", "IT' S")]:
        got = decode_path(path)
        check(f"{path:<12} -> {want!r}", got == want, f"got {got!r}")

    header("2. EDGE CASES")
    check("all blanks -> empty string", decode_path("_____") == "")
    lp = torch.zeros(1, 0, len(V))
    check("zero frames -> empty string", greedy_decode(lp, torch.tensor([0]), V) == [""])
    check("leading/trailing/double spaces cleaned", decode_path("^A_^_^B^") == "A B",
          f"raw {decode_path('^A_^_^B^', clean=False)!r}")
    check("raw mode keeps the exact best path string", decode_path("^A_^_^B^", clean=False) == " A  B ")

    header("3. PADDING IGNORED")
    lp, _ = from_path("_HI___XXXX")
    check("frames beyond out_lens never decoded",
          greedy_decode(lp, torch.tensor([6]), V) == ["HI"], "(XXXX sits in the padding)")

    header("4. ROUND TRIP ON EVERY DEV TRANSCRIPT")
    cfg = load_config("configs/base.yaml")
    rows = read_manifest(resolve(cfg, "manifests_dir") / f"{cfg['data']['dev_subset']}.csv")
    rng = random.Random(0)
    wrong, frames = [], 0
    for k in range(0, len(rows), 64):
        chunk = rows[k:k + 64]
        paths = [ctc_path(r["transcript"], rng) for r in chunk]
        T = max(len(p) for p in paths)
        lp = torch.full((len(paths), T, len(V)), -20.0)
        for b, p in enumerate(paths):
            lp[b, torch.arange(len(p)), torch.tensor(p)] = 0.0
            lp[b, len(p):, V.char2id["Q"]] = 30.0        # garbage in the padding
        lens = torch.tensor([len(p) for p in paths])
        frames += int(lens.sum())
        for r, hyp in zip(chunk, greedy_decode(lp, lens, V)):
            if hyp != r["transcript"]:
                wrong.append((r["utt_id"], hyp[:40]))
    check(f"all {len(rows)} dev transcripts recovered exactly from random CTC paths",
          not wrong, f"{len(wrong)} wrong {wrong[:2]}; {frames} frames decoded")

    header("5. AGREEMENT WITH CTC")
    torch.manual_seed(0)
    ok, n = True, 0
    for _ in range(200):
        T = random.Random(n).randint(5, 40)
        lp = torch.randn(1, T, len(V)).mul(3).log_softmax(-1)
        raw = greedy_decode(lp, torch.tensor([T]), V, clean=False)[0]
        ids = V.encode(raw)
        best = lp.max(-1).values.sum().item()                   # log P(best path)
        nll = torch.nn.functional.ctc_loss(lp.transpose(0, 1), torch.tensor([ids or [0]])[:, :len(ids)],
                                           torch.tensor([T]), torch.tensor([len(ids)]),
                                           blank=0, reduction="sum").item()
        ok &= -nll >= best - 1e-4
        n += 1
    check("log P_ctc(decoded string) >= log P(best path) in 200 random cases", ok)

    header("6. UNTRAINED MODEL ON REAL DEV AUDIO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = collate([ASRDataset(rows[:4], resolve(cfg, "raw_dir"), V)[k] for k in range(4)])
    torch.manual_seed(0)
    model = CNNBiLSTM.from_config(cfg["model"], cfg["features"]["n_mels"], len(V)).to(device).eval()
    fe = LogMel(**cfg["features"]).to(device)
    with torch.no_grad():
        feats, fl = fe(batch["wav"].to(device), batch["wav_lens"].to(device))
        lp, ol = model(feats, fl)
    hyps = greedy_decode(lp, ol, V)
    print(f"reference : {batch['transcripts'][0][:70]}")
    print(f"untrained : {hyps[0][:70]!r}")
    print(f"blank fraction of best path: {blank_fraction(lp, ol):.3f}")
    check("decoder runs on real model output, one string per utterance", len(hyps) == 4)
    big = torch.randn(86, 1250, len(V), device=device).log_softmax(-1)
    big_lens = torch.full((86,), 1250)
    t0 = time.perf_counter()
    greedy_decode(big, big_lens, V)
    ms = (time.perf_counter() - t0) * 1000
    print(f"decoding 86 utterances x 1250 frames: {ms:.0f} ms")

    print("\nRESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
