# Example transcripts (chosen by fixed rules, not by hand)

## Noise training rescues an utterance at 0 dB

*Rule: 6 to 14 words; largest WER drop from A to B at 0 dB*  utterance `4446-2271-0015`

- **reference**: A LITTLE ATTACK OF NERVES POSSIBLY
- **A @ clean** (WER 33%): A LITTLE ATTAC OF NEURRERES POSSIBLY
- **B @ clean** (WER 17%): A LITTLE ATTACK OF NERRVES POSSIBLY
- **C @ clean** (WER 0%): A LITTLE ATTACK OF NERVES POSSIBLY
- **A @ 0 dB** (WER 117%): ALL RITLE A TANC OF NURSE MOSSO LAKE
- **B @ 0 dB** (WER 0%): A LITTLE ATTACK OF NERVES POSSIBLY
- **C @ 0 dB** (WER 17%): A LITTLE ATTARK OF NERVES POSSIBLY

## SpecAugment helps at -5 dB

*Rule: 6 to 14 words; largest WER drop from B to C at -5 dB*  utterance `237-126133-0006`

- **reference**: AT THIS THE BUNDLE OPENED SUDDENLY AND OUT POPPED PHRONSIE
- **A @ -5 dB** (WER 190%): SH HUT NIT HI LETIC UT A IME LO OPETE UNTE THE CONNIC ADANE HAD AT CLIPING OUT GLOMB ELLENTHN EELECAP
- **B @ -5 dB** (WER 140%): TAT TI I TE FIN NO OPEN SUC TE AN E OU TOPPED CRANSY
- **C @ -5 dB** (WER 80%): AT TINS BUT FIN NO OEN SUCAND OUT TOP TRANS

## Model A goes silent at -5 dB

*Rule: 6 to 14 words; largest share of deleted words for A at -5 dB*  utterance `260-123440-0008`

- **reference**: I'LL TRY IF I KNOW ALL THE THINGS I USED TO KNOW
- **A @ -5 dB** (WER 100%): (empty)
- **B @ -5 dB** (WER 92%): OE CO WIT AMOD AN THE GI AS HE S
- **C @ -5 dB** (WER 92%): A O WITH ANO AN THE GI AR YUSO

## A word no model has seen in training (clean)

*Rule: most frequent never seen word that all three models get wrong on clean speech; shortest utterance containing it*  utterance `2300-131720-0014`

- **reference**: MISTER EDISON WAS A LEADER FAR AHEAD OF THE TIME
- **A @ clean** (WER 30%): MISTER EDDISON WAS A LEADER FAR A HEAD OF THE TIME
- **B @ clean** (WER 10%): MISTER ADDISON WAS A LEADER FAR AHEAD OF THE TIME
- **C @ clean** (WER 30%): MISTER ADDISON WAS A LEADER FAR A HEAD OF THE TIME

