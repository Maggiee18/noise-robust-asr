"""
Character vocabulary and transcript normalization.

Design decisions (documented for the paper)
-------------------------------------------
* Character level output, as specified. The symbol inventory is FIXED in code
  rather than collected from the data, so token ids are identical on every
  machine and in every run.
* Index 0 is the CTC blank. torch.nn.CTCLoss(blank=0) and the greedy decoder
  must use this same id; it is exposed as Vocabulary.blank_id.
* The other 28 symbols are exactly the characters that occur in LibriSpeech
  (verified in Phase 3): space, apostrophe and A to Z. Total: 29 classes.
* Apostrophes are KEPT because they distinguish words (IT'S vs ITS,
  WE'LL vs WELL); removing them would change words in the reference.
* No <unk> output class. The model could never learn to predict it (it never
  appears in the training targets), so it would only waste an output unit.
  Instead, normalize() maps common variants onto the vocabulary (lower case,
  accented letters, curly apostrophes, hyphens, repeated whitespace) and
  REMOVES anything still outside it, reporting what was removed. For LibriSpeech nothing is removed;
  this path only matters for free text such as demo references.
* encode() is strict: it raises on an unknown character, so a missing
  normalization step becomes a loud error instead of a silent label change.
* The same normalize() is applied to references and hypotheses before WER/CER.
"""

import json
import re
import unicodedata
from pathlib import Path

BLANK = "<blank>"
SYMBOLS = [" ", "'"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# Variants mapped onto the vocabulary before filtering.
_APOSTROPHES = {"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"}
_TO_SPACE = re.compile(r"[-‐-―_/]")   # hyphens, dashes, slash, underscore
_SPACES = re.compile(r"\s+")


class Vocabulary:
    def __init__(self):
        self.tokens = [BLANK] + SYMBOLS              # id -> token
        self.char2id = {c: i for i, c in enumerate(self.tokens) if i != 0}
        self.blank_id = 0
        self.space_id = self.char2id[" "]

    def __len__(self):
        return len(self.tokens)

    # ---------------------------------------------------------------- text
    def normalize(self, text, return_removed=False):
        """Map free text onto the vocabulary's character set."""
        t = unicodedata.normalize("NFKC", text)
        # strip accents: decompose (e.g. E + combining acute) and drop the marks
        t = "".join(c for c in unicodedata.normalize("NFKD", t)
                    if unicodedata.category(c) != "Mn")
        for k, v in _APOSTROPHES.items():
            t = t.replace(k, v)
        t = t.upper()
        t = _TO_SPACE.sub(" ", t)
        removed = sorted({c for c in t if c not in self.char2id and not c.isspace()})
        t = "".join(c if c in self.char2id or c.isspace() else "" for c in t)
        t = _SPACES.sub(" ", t).strip()
        return (t, removed) if return_removed else t

    # ---------------------------------------------------------------- ids
    def encode(self, text):
        """Normalized text -> list of ids (never contains the blank)."""
        try:
            return [self.char2id[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} not in vocabulary; "
                             f"call normalize() first. Text: {text!r}") from None

    def decode(self, ids):
        """ids -> text, dropping blanks. Does NOT collapse repeats: that is
        CTC decoding, implemented separately in Phase 11."""
        return "".join(self.tokens[i] for i in ids if i != self.blank_id)

    # ---------------------------------------------------------------- io
    def to_json(self, path):
        Path(path).write_text(json.dumps(
            {"blank_id": self.blank_id, "tokens": self.tokens}, indent=2))

    @staticmethod
    def check_json(path):
        """Verify a saved vocabulary matches this code exactly."""
        saved = json.loads(Path(path).read_text())
        v = Vocabulary()
        return saved["tokens"] == v.tokens and saved["blank_id"] == v.blank_id
