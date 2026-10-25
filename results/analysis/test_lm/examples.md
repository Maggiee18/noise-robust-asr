# Example transcripts (chosen by fixed rules, not by hand)

## Noise training rescues an utterance at 0 dB

*Rule: 6 to 14 words; largest WER drop from A to B at 0 dB*  utterance `237-134500-0017`

- **reference**: IF I FEEL THAT WAY I FEEL THAT WAY
- **A @ clean** (WER 0%): IF I FEEL THAT WAY I FEEL THAT WAY
- **B @ clean** (WER 0%): IF I FEEL THAT WAY I FEEL THAT WAY
- **C @ clean** (WER 0%): IF I FEEL THAT WAY I FEEL THAT WAY
- **A @ 0 dB** (WER 100%): IT IS ONLY NEAL AND LAY
- **B @ 0 dB** (WER 0%): IF I FEEL THAT WAY I FEEL THAT WAY
- **C @ 0 dB** (WER 22%): IF I FEEL THAT WAY I FEEL AWAY

## SpecAugment helps at -5 dB

*Rule: 6 to 14 words; largest WER drop from B to C at -5 dB*  utterance `61-70968-0021`

- **reference**: SURELY WE CAN SUBMIT WITH GOOD GRACE
- **A @ -5 dB** (WER 43%): SURELY WEAKEN SOME MIT WITH GOOD GRACE
- **B @ -5 dB** (WER 57%): SHORTLY WEAKENS MET WITH GOOD GRACE
- **C @ -5 dB** (WER 0%): SURELY WE CAN SUBMIT WITH GOOD GRACE

## Model A goes silent at -5 dB

*Rule: 6 to 14 words; largest share of deleted words for A at -5 dB*  utterance `8224-274384-0007`

- **reference**: HAVE MERCY LORD ON ME I PRAY FOR MEN WOULD ME DEVOUR
- **A @ -5 dB** (WER 100%): (empty)
- **B @ -5 dB** (WER 92%): THESE TO SOON WOULD METE
- **C @ -5 dB** (WER 83%): IT LAY ON TO THE NOON WOULD MAYBE BE

## A word no model has seen in training (clean)

*Rule: most frequent never seen word that all three models get wrong on clean speech; shortest utterance containing it*  utterance `8555-284449-0014`

- **reference**: THE FORMER BOOLOOROO GROANED
- **A @ clean** (WER 75%): THE FORMER BILER IT GROUND
- **B @ clean** (WER 50%): THE FORMER VILLARI GROUND
- **C @ clean** (WER 75%): THE FARMER BILLET GROUND

