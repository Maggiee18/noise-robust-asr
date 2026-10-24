"""
A backoff n-gram language model (up to 3-grams) in Python, for torchaudio's CTC
beam search decoder.

Why it exists: the Windows build of flashlight-text ships its KenLM wrapper
without kenlm.dll, so torchaudio cannot load an ARPA file there. This class
gives the decoder the SAME scores KenLM would (standard ARPA backoff, log10),
checked against KenLM itself on Linux (see scripts/check_ngram_lm.py).

    build_arrays("data/lm/3-gram.pruned.1e-7.arpa", "data/lm/3-gram.pruned.1e-7.npz")
    lm = NgramLM("data/lm/3-gram.pruned.1e-7.npz", lm_dict_words)   # words in decoder index order
    decoder = ctc_decoder(lexicon=..., tokens=..., lm=lm, lm_dict="data/lm/lm_dict.txt", ...)

Storage: n-grams are integer keys in sorted numpy arrays (unigram id a,
bigram a*V+b, trigram (a*V+b)*V+c with V = vocabulary size), looked up with
binary search (~60 MB in RAM per decoder process for the pruned 3-gram).

Scoring (ARPA backoff, log10), for history (a, b) and next word w:
    log P(w | a b) = trigram(a, b, w)                          if it exists
                   = bow(a, b) + log P(w | b)                   otherwise
    log P(w | b)   = bigram(b, w)  if it exists, else bow(b) + unigram(w)
bow() of an n-gram that is not in the model is 0. Words outside the LM
vocabulary score as <unk>.
"""

import numpy as np
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState


def build_arrays(arpa_path, out_path):
    """Parse an ARPA file (orders 1 to 3) into sorted arrays saved as .npz."""
    vocab, uni_lp, uni_bo = {}, [], []
    bi, tri = [], []
    section = None
    with open(arpa_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("\\") and line.endswith("-grams:"):
                section = int(line[1:line.index("-")])
                if section > 3:
                    raise ValueError("only orders up to 3 are supported")
                continue
            if line == "\\end\\":
                break
            if section is None:
                continue
            parts = line.split("\t")
            lp = float(parts[0])
            words = parts[1].split(" ")
            bo = float(parts[2]) if len(parts) > 2 else 0.0
            if section == 1:
                vocab[words[0]] = len(uni_lp)
                uni_lp.append(lp)
                uni_bo.append(bo)
            elif section == 2:
                bi.append((vocab[words[0]], vocab[words[1]], lp, bo))
            else:
                tri.append((vocab[words[0]], vocab[words[1]], vocab[words[2]], lp))
    V = len(uni_lp)
    bi_k = np.array([a * V + b for a, b, _, _ in bi], dtype=np.int64)
    o = np.argsort(bi_k)
    tri_k = np.array([(a * V + b) * V + c for a, b, c, _ in tri], dtype=np.int64)
    t = np.argsort(tri_k)
    np.savez(out_path,
             words=np.array(list(vocab), dtype=str),
             uni_lp=np.array(uni_lp, dtype=np.float32), uni_bo=np.array(uni_bo, dtype=np.float32),
             bi_k=bi_k[o], bi_lp=np.array([x[2] for x in bi], dtype=np.float32)[o],
             bi_bo=np.array([x[3] for x in bi], dtype=np.float32)[o],
             tri_k=tri_k[t], tri_lp=np.array([x[3] for x in tri], dtype=np.float32)[t])
    return dict(unigrams=V, bigrams=len(bi), trigrams=len(tri))


class NgramArrays:
    """The model itself: log10 P(w | history) from the arrays."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.words = list(z["words"])
        self.id = {w: i for i, w in enumerate(self.words)}
        self.V = len(self.words)
        self.uni_lp, self.uni_bo = np.asarray(z["uni_lp"]), np.asarray(z["uni_bo"])
        self.bi_k, self.bi_lp, self.bi_bo = z["bi_k"], z["bi_lp"], z["bi_bo"]
        self.tri_k, self.tri_lp = z["tri_k"], z["tri_lp"]
        self.unk = self.id.get("<unk>")
        self.bos, self.eos = self.id["<s>"], self.id["</s>"]

    @staticmethod
    def _find(keys, k):
        i = int(np.searchsorted(keys, k))
        return i if i < len(keys) and keys[i] == k else -1

    def bigram(self, b, w):
        """log10 P(w | b)."""
        i = self._find(self.bi_k, b * self.V + w)
        if i >= 0:
            return float(self.bi_lp[i])
        return float(self.uni_bo[b]) + float(self.uni_lp[w])

    def score(self, a, b, w):
        """log10 P(w | a b); a = -1 means only one word of history."""
        if a >= 0:
            i = self._find(self.tri_k, (a * self.V + b) * self.V + w)
            if i >= 0:
                return float(self.tri_lp[i])
            j = self._find(self.bi_k, a * self.V + b)
            bow = float(self.bi_bo[j]) if j >= 0 else 0.0
            return bow + self.bigram(b, w)
        return self.bigram(b, w)

    def sentence(self, words):
        """Total log10 probability of a sentence including </s> (for checks)."""
        a, b, total = -1, self.bos, 0.0
        for w in list(words) + ["</s>"]:
            wid = self.id.get(w, self.unk)
            total += self.score(a, b, wid)
            a, b = b, wid
        return total


class NgramLM(CTCDecoderLM):
    """Adapter for torchaudio's ctc_decoder (custom LM interface)."""

    def __init__(self, npz_path, decoder_words):
        """decoder_words: the words in the decoder's index order (the lm_dict file)."""
        CTCDecoderLM.__init__(self)
        self.m = NgramArrays(npz_path)
        unk = self.m.unk if self.m.unk is not None else 0
        self.to_lm = [self.m.id.get(w, unk) for w in decoder_words]
        self.ctx, self.cache = {}, {}

    def start(self, start_with_nothing):
        state = CTCDecoderLMState()
        self.ctx = {state: (-1, self.m.bos)}       # a new utterance: forget old states
        self.cache = {}
        return state

    def score(self, state, usr_token_idx):
        out = state.child(usr_token_idx)           # same child object for the same word
        hit = self.cache.get(out)
        if hit is None:
            a, b = self.ctx[state]
            w = self.to_lm[usr_token_idx]
            hit = self.m.score(a, b, w)
            self.ctx[out] = (b, w)
            self.cache[out] = hit
        return out, hit

    def finish(self, state):
        out = state.child(-1)
        hit = self.cache.get(out)
        if hit is None:
            a, b = self.ctx[state]
            hit = self.m.score(a, b, self.m.eos)
            self.cache[out] = hit
        return out, hit
