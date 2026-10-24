# Example transcripts (chosen by fixed rules, not by hand)

## Noise training rescues an utterance at 0 dB

*Rule: 6 to 14 words; largest WER drop from A to B at 0 dB*  utterance `5895-34622-0012`

- **reference**: THE CURIOSITY OF ONE PLACE EXHAUSTED THEY PASSED ON TO ANOTHER
- **A @ clean** (WER 0%): THE CURIOSITY OF ONE PLACE EXHAUSTED THEY PASSED ON TO ANOTHER
- **B @ clean** (WER 0%): THE CURIOSITY OF ONE PLACE EXHAUSTED THEY PASSED ON TO ANOTHER
- **C @ clean** (WER 0%): THE CURIOSITY OF ONE PLACE EXHAUSTED THEY PASSED ON TO ANOTHER
- **A @ 0 dB** (WER 127%): T CURYOT IT BE DETH WENT TI SEINSOLS IN WHO AT MASSDOUNT I WHINATHER
- **B @ 0 dB** (WER 27%): TE CURIOSITY ONE PLACED EXHAUSTED THEY PASSED ON TO ANOTHER
- **C @ 0 dB** (WER 9%): THE CURIOSITY ONE PLACE EXHAUSTED THEY PASSED ON TO ANOTHER

## SpecAugment helps at 0 dB

*Rule: 6 to 14 words; largest WER drop from B to C at 0 dB*  utterance `3752-4944-0019`

- **reference**: I HAVE THESE ATTACKS AT TIMES
- **A @ 0 dB** (WER 83%): I HEAVS ATTANKS T TO
- **B @ 0 dB** (WER 83%): I HADE THIS ATTAXIT TEIS
- **C @ 0 dB** (WER 33%): I HAVE THIS ATTACKS OT TIMES

## Model A goes silent at 0 dB

*Rule: 6 to 14 words; largest share of deleted words for A at 0 dB*  utterance `1988-24833-0022`

- **reference**: I TOLD TOM WE SHOULDN'T COME SO LATE SAYS HILDA
- **A @ 0 dB** (WER 100%): ISHOLD MOMICHANLADLELANTE'LSALA
- **B @ 0 dB** (WER 80%): I TOLD TAMISH INCOS HE LATES AS ETE
- **C @ 0 dB** (WER 80%): I TOLD TOAS SHOIDCOPS THE LATES AS EO

## A word no model has seen in training (clean)

*Rule: most frequent never seen word that all three models get wrong on clean speech; shortest utterance containing it*  utterance `3536-8226-0017`

- **reference**: THEN BOZZLE CAME FORWARD AND INTRODUCED HIS WIFE
- **A @ clean** (WER 12%): THEN BAZZLE CAME FORWARD AND INTRODUCED HIS WIFE
- **B @ clean** (WER 12%): THEN BOZEL CAME FORWARD AND INTRODUCED HIS WIFE
- **C @ clean** (WER 12%): THEN BOZAL CAME FORWARD AND INTRODUCED HIS WIFE

