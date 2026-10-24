"""
Phase 19: Gradio demo. Compare Models A, B and C on your own audio, clean or noisy.

Run from the project root (opens in the browser at http://127.0.0.1:7860):
    python demo/app.py
    python demo/app.py --device cpu          # keep the GPU free
    python demo/app.py --share               # temporary public link (gradio.live)

What it does
  1. Audio in: upload a file, record with the microphone, or pick a LibriSpeech
     test utterance (its reference transcript is filled in for you). Any sample
     rate, mono or stereo; it is converted to 16 kHz mono like the training data.
  2. Optional noise: mix MUSAN TEST pool noise (never heard in training) or your
     own noise recording at an exact SNR from -5 to 20 dB, with the same
     mix_at_snr code as the experiments. The seed picks the noise segment, so a
     result can be reproduced.
  3. Each model transcribes the same audio. If a reference is given, every word
     error is marked and the WER per model is shown.
  4. You can listen to exactly what the models heard.

Needs: pip install gradio   (tested with gradio 6.30)
Checkpoints: experiments/<model>/best.pt (kept out of git; see README).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.audio import load_audio  # noqa: E402
from src.config import load_config, resolve  # noqa: E402
from src.inference import SAMPLE_RATE, Transcriber, fit_noise, prepare_audio, word_diff  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.mixing import mix_at_snr  # noqa: E402
from src.musan import NoiseBank  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

MODELS = [("model_a_clean", "A: clean training"),
          ("model_b_noise", "B: noise augmentation"),
          ("model_c_lb", "C: noise + SpecAugment")]
# test utterances shown as ready made examples (from results/analysis/test/examples.md)
EXAMPLE_IDS = ["4446-2271-0015", "2300-131720-0014", "260-123440-0008",
               "237-126133-0006", "61-70968-0000"]
MAX_SECONDS = 60
NOISE_MUSAN = "MUSAN test noise (never heard in training)"
NOISE_OWN = "my own noise recording"


class Demo:
    def __init__(self, device):
        self.vocab = Vocabulary()
        self.cfg = load_config("configs/base.yaml")
        self.models = {}
        for name, label in MODELS:
            ck = ROOT / "experiments" / name / "best.pt"
            if ck.exists():
                self.models[label] = Transcriber.from_checkpoint(ck, device=device)
                print(f"loaded {name} (epoch {self.models[label].epoch}) on {device}")
            else:
                print(f"skipping {name}: {ck} not found")
        if not self.models:
            sys.exit("no checkpoints found in experiments/*/best.pt")
        man, raw = resolve(self.cfg, "manifests_dir"), resolve(self.cfg, "raw_dir")
        self.raw = raw
        try:
            self.bank = NoiseBank(man / self.cfg["noise"]["manifest"], raw, pool="test",
                                  categories=tuple(self.cfg["noise"]["categories"]))
            print(f"MUSAN test noise pool: {len(self.bank)} files")
        except Exception as e:                      # data not downloaded: demo still works
            self.bank = None
            print(f"MUSAN test noise not available ({e}); only your own noise can be mixed")
        self.examples = {}
        test_csv = man / f"{self.cfg['data']['test_subset']}.csv"
        if test_csv.exists():
            rows = {r["utt_id"]: r for r in read_manifest(test_csv)}
            for u in EXAMPLE_IDS:
                if u in rows and (raw / rows[u]["path"]).exists():
                    r = rows[u]
                    self.examples[f"{u}  ({r['duration_s']:.1f} s)"] = r

    # ------------------------------------------------------------------ callbacks
    def load_example(self, key):
        if not key or key not in self.examples:
            return gr.update(), gr.update()
        r = self.examples[key]
        wav = load_audio(self.raw / r["path"]).numpy()
        return (SAMPLE_RATE, wav), r["transcript"]

    def run(self, audio, reference, add_noise, noise_source, own_noise, snr, seed):
        if audio is None:
            raise gr.Error("Upload, record or pick an example first.")
        sr, data = audio
        wav = prepare_audio(data, sr)
        if wav.numel() > MAX_SECONDS * SAMPLE_RATE:
            wav = wav[:MAX_SECONDS * SAMPLE_RATE]
            gr.Warning(f"Audio longer than {MAX_SECONDS} s: only the first {MAX_SECONDS} s are used.")
        if float(wav.abs().max()) < 1e-4:
            raise gr.Error("The audio is (almost) silent. Check the microphone.")

        status = [f"input: {wav.numel() / SAMPLE_RATE:.1f} s at 16 kHz"]
        if add_noise:
            if noise_source == NOISE_OWN:
                if own_noise is None:
                    raise gr.Error("Upload or record a noise sample, or choose MUSAN noise.")
                noise = fit_noise(prepare_audio(own_noise[1], own_noise[0]), wav.numel())
                where = "your noise recording"
            else:
                if self.bank is None:
                    raise gr.Error("MUSAN is not downloaded here; use your own noise recording.")
                noise, meta = self.bank.sample(wav.numel(), np.random.default_rng(int(seed)))
                where = f"{Path(meta['path']).name}, from {meta['start'] / SAMPLE_RATE:.1f} s"
            try:
                wav, info = mix_at_snr(wav, noise, float(snr))
            except ValueError as e:
                raise gr.Error(str(e))
            status.append(f"noise: {where}; requested {snr:g} dB, achieved "
                          f"{info['achieved_snr_db']:.2f} dB")
        else:
            status.append("no noise added")

        ref = self.vocab.normalize(reference or "")
        rows = ["| model | transcript | WER |", "|---|---|---|"]
        for label, asr in self.models.items():
            t0 = time.perf_counter()
            hyp = asr(wav)
            ms = 1000 * (time.perf_counter() - t0)
            if ref:
                shown, st = word_diff(ref, hyp)
                wer = (f"**{st['wer']:.1f}%** ({st['sub']} sub, {st['dele']} del, {st['ins']} ins "
                       f"/ {st['ref_words']} words)")
            else:
                shown, wer = hyp or "*(empty)*", "n/a (no reference)"
            rows.append(f"| {label} | {shown or '*(empty)*'} | {wer} |")
            status.append(f"{label.split(':')[0]}: {ms:.0f} ms")
        legend = ("\n\n~~struck out~~ = reference word missed or replaced, **bold** = what the model "
                  "wrote instead, **+word** = inserted word." if ref else "")
        return (SAMPLE_RATE, wav.numpy()), "\n".join(rows) + legend, " | ".join(status)


def build(demo):
    with gr.Blocks(title="Noise robust ASR demo") as ui:
        gr.Markdown(
            "# Noise robust speech recognition: CNN BiLSTM + CTC\n"
            "Three models with the same architecture (10.3 M parameters, trained on 100 h of "
            "LibriSpeech): **A** clean training, **B** noise augmentation, **C** noise + "
            "SpecAugment. Add noise and see how each one copes. Greedy decoding, no language "
            "model, so rare names get spelled by sound.")
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="numpy",
                                 label="speech (upload or record)")
                example = gr.Dropdown(choices=list(demo.examples), value=None,
                                      label="or pick a LibriSpeech test utterance")
                reference = gr.Textbox(label="reference transcript (optional, for WER)",
                                       placeholder="type what was said to score each model")
            with gr.Column(scale=1):
                add_noise = gr.Checkbox(value=True, label="add noise")
                snr = gr.Slider(-5, 20, value=0, step=1, label="SNR (dB): lower = louder noise")
                noise_source = gr.Radio([NOISE_MUSAN, NOISE_OWN], value=NOISE_MUSAN,
                                        label="noise")
                own_noise = gr.Audio(sources=["upload", "microphone"], type="numpy",
                                     label="your noise (only for 'my own noise recording')")
                seed = gr.Number(value=0, precision=0, label="noise seed (picks the MUSAN segment)")
                go = gr.Button("Transcribe with all models", variant="primary")
        results = gr.Markdown()
        with gr.Row():
            heard = gr.Audio(label="what the models heard (16 kHz)", interactive=False)
            status = gr.Markdown()
        example.change(demo.load_example, inputs=example, outputs=[audio, reference])
        go.click(demo.run, inputs=[audio, reference, add_noise, noise_source, own_noise, snr, seed],
                 outputs=[heard, results, status])
        gr.Markdown(
            "Test set results (WER %, 2,620 utterances): clean A 20.8 / B 18.2 / C 18.4; "
            "0 dB A 68.3 / B 38.1 / C 37.7; -5 dB A 77.6 / B 50.1 / C 49.0. "
            "Code: github.com/Maggiee18/noise-robust-asr")
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    args = ap.parse_args()
    ui = build(Demo(args.device))
    ui.launch(server_name="127.0.0.1", server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
