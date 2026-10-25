# Example transcripts (chosen by fixed rules, not by hand)

## Noise training rescues an utterance at 0 dB

*Rule: 6 to 14 words; largest WER drop from A to B at 0 dB*  utterance `174-50561-0019`

- **reference**: YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL
- **A @ clean** (WER 12%): YOU SEE THE TREATMENT IS A TRUTHFUL FANCIFUL
- **B @ clean** (WER 0%): YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL
- **C @ clean** (WER 0%): YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL
- **A @ 0 dB** (WER 100%): USE HERMAN HURRY FOR HIM
- **B @ 0 dB** (WER 0%): YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL
- **C @ 0 dB** (WER 0%): YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL

## SpecAugment helps at 0 dB

*Rule: 6 to 14 words; largest WER drop from B to C at 0 dB*  utterance `2902-9008-0008`

- **reference**: WHAT DO I CARE FOR FOOD
- **A @ 0 dB** (WER 100%): OLD WAT WI HER
- **B @ 0 dB** (WER 83%): TO WHY CAREFUL FOOD
- **C @ 0 dB** (WER 0%): WHAT DO I CARE FOR FOOD

## Model A goes silent at 0 dB

*Rule: 6 to 14 words; largest share of deleted words for A at 0 dB*  utterance `84-121123-0027`

- **reference**: I ONLY WISH TO BE ALONE YOU WILL EXCUSE ME WILL YOU NOT
- **A @ 0 dB** (WER 100%): BETWEEN
- **B @ 0 dB** (WER 92%): ONE IS ALONE ALEXANDER
- **C @ 0 dB** (WER 85%): I WISH POLO

## A word no model has seen in training (clean)

*Rule: most frequent never seen word that all three models get wrong on clean speech; shortest utterance containing it*  utterance `3536-8226-0007`

- **reference**: OF COURSE IT AIN'T SAID MISSUS BOZZLE
- **A @ clean** (WER 29%): OF COURSE I AIN'T SAID MISSUS BAL
- **B @ clean** (WER 14%): OF COURSE IT AIN'T SAID MISSUS BOTTLE
- **C @ clean** (WER 29%): OF COURSE IT SAID MISSUS BASLE

