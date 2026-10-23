"""
LibriSpeech reading and manifest utilities.

Directory layout (as extracted by scripts/download_data.py):
    data/raw/LibriSpeech/<subset>/<speaker>/<chapter>/
        <speaker>-<chapter>.trans.txt     one line per utterance:
                                          "<speaker>-<chapter>-<utt> TRANSCRIPT"
        <speaker>-<chapter>-<utt>.flac

A manifest is one CSV row per utterance. Paths are stored relative to the raw
data folder with forward slashes, so the same manifest works on Windows,
Linux and Kaggle.
"""

import csv
from pathlib import Path

# Published statistics, used as hard checks.
EXPECTED = {
    "train-clean-100": dict(utterances=28539, speakers=251),
    "dev-clean": dict(utterances=2703, speakers=40),
    "test-clean": dict(utterances=2620, speakers=40),
    "test-other": dict(utterances=2939, speakers=33),
}

MANIFEST_FIELDS = ["utt_id", "speaker_id", "chapter_id", "path",
                   "sample_rate", "num_samples", "duration_s", "transcript"]


def iter_utterances(raw_dir, subset):
    """
    Yield one dict per utterance in deterministic order (sorted by speaker,
    chapter, utterance id). The transcript is returned exactly as written in
    the corpus; normalization happens in Phase 4.
    """
    raw_dir = Path(raw_dir)
    root = raw_dir / "LibriSpeech" / subset
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found. Run scripts/download_data.py first.")

    def numeric(p):
        return int(p.name) if p.name.isdigit() else p.name

    for spk_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=numeric):
        for chap_dir in sorted((p for p in spk_dir.iterdir() if p.is_dir()), key=numeric):
            trans = chap_dir / f"{spk_dir.name}-{chap_dir.name}.trans.txt"
            lines = trans.read_text(encoding="utf-8").splitlines()
            for line in sorted(l for l in lines if l.strip()):
                utt_id, _, text = line.partition(" ")
                audio = chap_dir / f"{utt_id}.flac"
                yield dict(
                    utt_id=utt_id,
                    speaker_id=spk_dir.name,
                    chapter_id=chap_dir.name,
                    path=audio.relative_to(raw_dir).as_posix(),
                    transcript=text.strip(),
                )


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path):
    """Return a list of dicts with numeric fields converted."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["sample_rate"] = int(r["sample_rate"])
        r["num_samples"] = int(r["num_samples"])
        r["duration_s"] = float(r["duration_s"])
    return rows
