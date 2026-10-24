"""
Language model decoding, step 1: download the official LibriSpeech n-gram LM
and build the files the beam search decoder needs.

Run from the project root:
    python scripts/prepare_lm.py                      # 3-gram, pruned 1e-7 (~34 MB)
    python scripts/prepare_lm.py --lm 4-gram          # full 4-gram (~1.3 GB download)

Source: OpenSLR 11 (https://www.openslr.org/11), the language model released
with LibriSpeech. It was trained on ~800 M words of normalized book text from
which the books in dev and test were EXCLUDED, so it does not leak the
evaluation transcripts. Downloaded from the same OpenSLR mirrors as
scripts/download_data.py; the MD5 is checked against the mirror's md5sum.txt.

Writes data/lm/ (kept out of git, reproducible with this script)
    <lm>.arpa.gz      the download
    <lm>.arpa         decompressed (what KenLM reads)
    lexicon.txt       every LM word spelled with our characters:  HELLO  H E L L O |
    tokens.txt        our 29 output symbols in id order, blank = "-", space = "|"
    lm_dict.txt       the decoder's word list in index order (lexicon words + <unk>)
    <lm>.npz          the n-grams as sorted arrays for the Python LM (src/ngram_lm.py),
                      self checked here against the ARPA file
    lm_info.json      sizes, MD5, n-gram counts, lexicon size, OOV rates on dev / test

Design decisions (documented for the paper)
    * Lexicon = the LM's own unigram vocabulary (~200 k words), restricted to
      words written with our symbols (A to Z and the apostrophe). With a
      lexicon the decoder can only output words in that list: rare names
      outside it cannot be produced, so the OOV rate of the references is
      reported here (it bounds what the LM decoder can get right).
    * The word separator of the decoder ("|") is our space symbol, so the
      decoder sees exactly the model's 29 classes.
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.librispeech import read_manifest  # noqa: E402
from src.ngram_lm import NgramArrays, build_arrays  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from download_data import MIRRORS, download  # noqa: E402

LMS = ["3-gram.pruned.1e-7", "3-gram.pruned.3e-7", "3-gram", "4-gram"]
WORD = re.compile(r"^[A-Z']+$")


def md5_of(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_md5(name):
    for m in MIRRORS:
        try:
            req = urllib.request.Request(f"{m}/11/md5sum.txt",
                                         headers={"User-Agent": "noise-robust-asr/1.0"})
            text = urllib.request.urlopen(req, timeout=30).read().decode()
            for line in text.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].lstrip("*") == name:
                    return parts[0]
        except Exception as e:                      # try the next mirror
            print(f"  md5sum.txt from {m}: {e}")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--lm", default=LMS[0], choices=LMS)
    args = ap.parse_args()
    out = ROOT / "data" / "lm"
    out.mkdir(parents=True, exist_ok=True)
    gz, arpa = out / f"{args.lm}.arpa.gz", out / f"{args.lm}.arpa"

    # ---- download + verify
    if not gz.exists():
        for m in MIRRORS:
            print(f"downloading {gz.name} from {m}")
            if download(f"{m}/11/{gz.name}", gz):
                break
        else:
            sys.exit("download failed on every mirror")
    want, have = expected_md5(gz.name), md5_of(gz)
    if want and want != have:
        gz.unlink()
        sys.exit(f"MD5 mismatch for {gz.name}: got {have}, expected {want}. Deleted; run again.")
    print(f"[{'PASS' if want else 'INFO'}] {gz.name}: {gz.stat().st_size / 1e6:.1f} MB, md5 {have}"
          + ("" if want else " (no published checksum found)"))

    # ---- decompress
    if not arpa.exists():
        print(f"decompressing to {arpa.name} ...")
        with gzip.open(gz, "rb") as fi, open(arpa.with_suffix(".tmp"), "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 22)
        arpa.with_suffix(".tmp").replace(arpa)

    # ---- n-gram counts and unigram vocabulary
    counts, words, section = {}, [], None
    with open(arpa, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ngram "):
                n, c = line[6:].split("=")
                counts[int(n)] = int(c)
            elif line.startswith("\\") and line.endswith("-grams:"):
                section = int(line[1:line.index("-")])
            elif line == "\\end\\":
                break
            elif section == 1 and line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    words.append(parts[1])
            elif section and section > 1:
                break                               # only the unigrams are needed
    lexicon = sorted({w for w in words if WORD.match(w)})
    skipped = len(words) - len(lexicon)
    with open(out / "lexicon.txt", "w", encoding="utf-8", newline="\n") as f:
        for w in lexicon:
            f.write(f"{w}\t{' '.join(w)} |\n")
    print(f"[PASS] lexicon.txt: {len(lexicon):,} words ({skipped} LM entries such as <s>, </s>, <unk> skipped)")

    # ---- tokens in model id order
    vocab = Vocabulary()
    toks = ["-" if i == vocab.blank_id else "|" if t == " " else t for i, t in enumerate(vocab.tokens)]
    assert len(set(toks)) == len(toks) == len(vocab)
    (out / "tokens.txt").write_text("\n".join(toks) + "\n", encoding="utf-8")
    print(f"[PASS] tokens.txt: {len(toks)} symbols, blank '-' = id 0, space '|' = id {vocab.space_id}")
    (out / "lm_dict.txt").write_text("\n".join(lexicon + ["<unk>"]) + "\n", encoding="utf-8")

    # ---- n-gram arrays for the Python LM, checked line by line against the ARPA file
    npz = out / f"{args.lm}.npz"
    if not npz.exists():
        print("building n-gram arrays for the Python LM ...")
        build_arrays(arpa, npz)
    m = NgramArrays(npz)
    checked = worst = 0
    section = None
    with open(arpa, encoding="utf-8", errors="replace") as f:
        for k, line in enumerate(f):
            line = line.strip()
            if line.startswith("\\") and line.endswith("-grams:"):
                section = int(line[1:line.index("-")])
                continue
            if not line or section is None or line == "\\end\\" or k % 997:
                continue                            # every 997th n-gram line
            lp, ws = float(line.split("\t")[0]), [m.id[w] for w in line.split("\t")[1].split(" ")]
            if ws[-1] == m.bos:
                continue                            # <s> is never predicted
            got = (float(m.uni_lp[ws[0]]) if section == 1 else
                   m.score(-1, ws[0], ws[1]) if section == 2 else m.score(ws[0], ws[1], ws[2]))
            worst = max(worst, abs(got - lp))
            checked += 1
    if worst > 1e-5:
        sys.exit(f"[FAIL] n-gram arrays disagree with the ARPA file (max {worst:.2e})")
    print(f"[PASS] {npz.name}: {m.V:,} words; {checked:,} sampled n-grams match the ARPA file "
          f"(max difference {worst:.1e})")

    # ---- OOV rates of the references
    lex, oov = set(lexicon), {}
    man = ROOT / "data" / "manifests"
    for split in ["dev-clean", "test-clean"]:
        p = man / f"{split}.csv"
        if p.exists():
            ws = [w for r in read_manifest(p) for w in vocab.normalize(r["transcript"]).split()]
            miss = [w for w in ws if w not in lex]
            oov[split] = dict(words=len(ws), oov_words=len(miss),
                              oov_rate_pct=round(100 * len(miss) / len(ws), 2),
                              examples=sorted(set(miss))[:15])
            print(f"  {split}: {oov[split]['oov_rate_pct']}% of reference words are not in the lexicon "
                  f"(e.g. {', '.join(oov[split]['examples'][:6])})")
    info = dict(lm=args.lm, source="https://www.openslr.org/11", archive=gz.name,
                archive_mb=round(gz.stat().st_size / 1e6, 1), md5=have, md5_verified=bool(want),
                ngram_counts=counts, lexicon_words=len(lexicon), oov=oov)
    (out / "lm_info.json").write_text(json.dumps(info, indent=2))
    print(f"n-gram counts: {counts}\ndone -> {out}")


if __name__ == "__main__":
    main()
