"""
Phase 4 check: vocabulary and transcript normalization.

Run from the project root (after build_manifests.py):
    python scripts/check_vocab.py

Part 1 runs unit tests on hand written cases.
Part 2 runs every transcript in the manifests through normalize -> encode ->
decode and requires that
  * normalization changes nothing (LibriSpeech is already normalized)
  * decoding returns the exact original text (lossless round trip)
  * no id is the blank
It also reports the label length statistics, and saves the vocabulary to
data/manifests/vocab.json so every later phase can confirm it uses the same ids.
"""

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.librispeech import read_manifest  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

SUBSETS = ["train-clean-100", "dev-clean", "test-clean"]
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def unit_tests(v):
    print("=" * 70 + "\nUNIT TESTS\n" + "=" * 70)
    check("29 classes", len(v) == 29, f"{len(v)}")
    check("blank is id 0", v.blank_id == 0 and v.tokens[0] == "<blank>")
    check("space is id 1, apostrophe id 2, A id 3, Z id 28",
          v.char2id[" "] == 1 and v.char2id["'"] == 2
          and v.char2id["A"] == 3 and v.char2id["Z"] == 28)
    check("ids are deterministic across instances",
          Vocabulary().tokens == v.tokens)

    cases = [
        ("hello world", "HELLO WORLD", []),
        ("  It’s   a  test\n", "IT'S A TEST", []),
        ("well-known", "WELL KNOWN", []),
        ("Hello, world!", "HELLO WORLD", ["!", ","]),
        ("café naïve", "CAFE NAIVE", []),
        ("5 O'CLOCK", "O'CLOCK", ["5"]),
    ]
    for raw, want, want_removed in cases:
        got, removed = v.normalize(raw, return_removed=True)
        check(f"normalize {raw!r}", got == want and removed == want_removed,
              f"{got!r}, removed {removed}")

    ids = v.encode("HE SAID IT'S")
    check("encode/decode round trip", v.decode(ids) == "HE SAID IT'S", f"{ids}")
    check("encode never emits blank", v.blank_id not in ids)
    check("decode drops blanks", v.decode([0, 10, 0, 7, 0]) == "HE")
    try:
        v.encode("hello")
        check("encode rejects unnormalized text", False)
    except ValueError:
        check("encode rejects unnormalized text", True)


def corpus_tests(v):
    print("\n" + "=" * 70 + "\nCORPUS PASS\n" + "=" * 70)
    for subset in SUBSETS:
        path = ROOT / "data" / "manifests" / f"{subset}.csv"
        if not path.exists():
            check(f"{subset} manifest exists", False, str(path))
            continue
        rows = read_manifest(path)
        changed, mismatched, lengths = [], 0, []
        for r in rows:
            t = r["transcript"]
            n = v.normalize(t)
            if n != t:
                changed.append((r["utt_id"], t, n))
            ids = v.encode(n)
            mismatched += v.decode(ids) != n or v.blank_id in ids
            lengths.append(len(ids))
        check(f"{subset}: normalization leaves all {len(rows)} transcripts unchanged",
              not changed, f"{len(changed)} changed, e.g. {changed[:1]}" if changed else "")
        check(f"{subset}: lossless encode/decode", mismatched == 0, f"{mismatched} mismatches")
        print(f"       label length: min {min(lengths)}, "
              f"mean {statistics.mean(lengths):.1f}, max {max(lengths)}")


def main():
    v = Vocabulary()
    unit_tests(v)
    corpus_tests(v)
    out = ROOT / "data" / "manifests" / "vocab.json"
    v.to_json(out)
    check("vocab.json written and matches code", Vocabulary.check_json(out), str(out))
    print("\nTokens: " + " ".join(f"{i}:{'<sp>' if t == ' ' else t}" for i, t in enumerate(v.tokens)))
    print("RESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
