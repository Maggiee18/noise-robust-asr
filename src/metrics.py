"""
Word and character error rates.

References and hypotheses go through the SAME normalization
(Vocabulary.normalize) before scoring, so formatting differences never count
as errors. Scores are corpus level: total edits / total reference words (or
characters) over all utterances, which is how WER and CER are normally
reported (not an average of per utterance rates).
"""

import jiwer


def error_rates(refs, hyps, vocab):
    refs = [vocab.normalize(r) for r in refs]
    hyps = [vocab.normalize(h) for h in hyps]
    return dict(wer=jiwer.wer(refs, hyps), cer=jiwer.cer(refs, hyps))
