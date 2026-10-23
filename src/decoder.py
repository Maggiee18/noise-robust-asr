"""
Greedy (best path) CTC decoding.

For each utterance:
  1. take the most probable class at every valid output frame (argmax)
  2. merge consecutive repeats of the same class
  3. remove blanks
  4. map ids to characters
  5. (clean=True) collapse runs of spaces and strip the ends

Order matters: repeats are merged BEFORE blanks are removed, so a blank
between two identical characters keeps both:
      _ H H _ E L L _ O _   ->  HELO
      _ H H _ E L _ L O _   ->  HELLO

Frames beyond out_lens (padding) are ignored.

Step 5 exists because space is an ordinary output class: the model can emit
two spaces separated by a blank, or a leading or trailing space. Evaluation
(Phase 16) normalizes references and hypotheses identically anyway; cleaning
here keeps printed examples readable. clean=False returns the exact best path
string, which check_decoder.py uses to verify the decoder against CTC.
"""

import re

import torch

_SPACES = re.compile(r" +")


def best_paths(log_probs, out_lens):
    """[B, T', C] -> list of 1D int64 tensors (argmax ids over valid frames)."""
    ids = log_probs.argmax(dim=-1).cpu()                 # [B, T']
    return [ids[b, :int(out_lens[b])] for b in range(ids.shape[0])]


def collapse(path, blank_id=0):
    """Merge repeats, then drop blanks. path: 1D int tensor -> list of ids."""
    if path.numel() == 0:
        return []
    merged = torch.unique_consecutive(path)
    return merged[merged != blank_id].tolist()


def greedy_decode(log_probs, out_lens, vocab, clean=True):
    """
    log_probs: [B, T', C]  (any device)
    out_lens:  [B]
    returns    list of B strings
    """
    texts = []
    for path in best_paths(log_probs, out_lens):
        text = "".join(vocab.tokens[i] for i in collapse(path, vocab.blank_id))
        if clean:
            text = _SPACES.sub(" ", text).strip()
        texts.append(text)
    return texts


def blank_fraction(log_probs, out_lens, blank_id=0):
    """Fraction of valid frames whose best class is the blank. A model stuck in
    the 'all blank' failure mode shows values near 1.0 and empty outputs."""
    paths = best_paths(log_probs, out_lens)
    total = sum(p.numel() for p in paths)
    blanks = sum(int((p == blank_id).sum()) for p in paths)
    return blanks / max(total, 1)
