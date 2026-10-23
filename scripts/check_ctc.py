"""
Phase 10 check: CTC loss.

Run from the project root:
    python scripts/check_ctc.py

Sections
  1. hand calculated examples (with the arithmetic printed)
  2. brute force: enumerate EVERY alignment path of a small problem, collapse
     each one (merge repeats, drop blanks) and sum the probabilities of the
     paths that give the target; must equal exp(-loss)
  3. conventions: batch first wrapper = direct call on [T, B, C];
     normalization = sum / total characters; padded target values ignored
  4. impossible lengths are caught with an error naming the utterance
  5. optimization sanity: gradient descent on one example drives the loss down
  6. real pipeline: dev-clean batch -> log Mel -> untrained model -> loss,
     forward and backward, with mixed precision on the GPU; CPU and GPU agree
"""

import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.dataset import ASRDataset, collate  # noqa: E402
from src.features import LogMel  # noqa: E402
from src.librispeech import read_manifest  # noqa: E402
from src.losses import CTCLoss, check_lengths  # noqa: E402
from src.model import CNNBiLSTM  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def header(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def collapse(path, blank=0):
    out, prev = [], None
    for s in path:
        if s != prev and s != blank:
            out.append(s)
        prev = s
    return out


def raw_ctc(log_probs_tbc, target, T):
    """Unnormalized loss for one utterance via torch (sum reduction)."""
    return torch.nn.functional.ctc_loss(
        log_probs_tbc, torch.tensor([target]), torch.tensor([T]),
        torch.tensor([len(target)]), blank=0, reduction="sum").item()


def main():
    torch.manual_seed(0)
    header("1. HAND CALCULATED EXAMPLES  (classes: 0 = blank, 1 = 'a')")
    uniform = torch.full((2, 1, 2), math.log(0.5))
    # T=2, target 'a': valid paths a a / a _ / _ a  -> 3 x 0.5^2 = 0.75
    got = raw_ctc(uniform, [1], 2)
    check("T=2, target 'a', p=0.5 everywhere: 3 paths x 0.25 = 0.75", abs(got + math.log(0.75)) < 1e-5,
          f"loss {got:.4f}, expected -ln 0.75 = {-math.log(0.75):.4f}")
    uniform3 = torch.full((3, 1, 2), math.log(0.5))
    got = raw_ctc(uniform3, [1, 1], 3)
    check("T=3, target 'aa': only path a _ a = 0.125 (blank must separate repeats)",
          abs(got - math.log(8)) < 1e-5, f"loss {got:.4f}, expected ln 8 = {math.log(8):.4f}")
    got = raw_ctc(uniform, [1, 1], 2)
    check("T=2, target 'aa': impossible -> infinite loss", math.isinf(got), f"loss {got}")

    header("2. BRUTE FORCE OVER ALL PATHS")
    for T, C, target in [(5, 3, [1, 2]), (6, 3, [2, 2]), (6, 4, [1, 3, 1]), (7, 3, [1, 1, 2])]:
        lp = torch.randn(T, 1, C).log_softmax(-1)
        total = 0.0
        for path in itertools.product(range(C), repeat=T):
            if collapse(path) == target:
                total += math.exp(sum(lp[t, 0, s].item() for t, s in enumerate(path)))
        got = raw_ctc(lp, target, T)
        check(f"T={T}, C={C}, target {target}: {C ** T} paths enumerated",
              abs(math.exp(-got) - total) < 1e-6 * max(total, 1e-12) + 1e-9,
              f"sum of valid paths {total:.6e}, exp(-loss) {math.exp(-got):.6e}")

    header("3. CONVENTIONS")
    B, T, C = 3, 40, 29
    lp_bf = torch.randn(B, T, C).log_softmax(-1)
    tl = torch.tensor([7, 3, 12])
    targets = torch.zeros(B, 12, dtype=torch.long)
    for b in range(B):
        targets[b, :tl[b]] = torch.randint(1, C, (int(tl[b]),))
    ol = torch.tensor([40, 25, 33])
    crit = CTCLoss()
    ours = crit(lp_bf, ol, targets, tl).item()
    direct = torch.nn.functional.ctc_loss(lp_bf.transpose(0, 1), targets, ol, tl, blank=0,
                                          reduction="sum").item() / tl.sum().item()
    check("batch first wrapper = direct call on [T, B, C], divided by total characters",
          abs(ours - direct) < 1e-5, f"{ours:.5f} vs {direct:.5f}")
    junk = targets.clone()
    for b in range(B):
        junk[b, tl[b]:] = 17
    check("values in the target padding are ignored",
          abs(crit(lp_bf, ol, junk, tl).item() - ours) < 1e-6)

    header("4. IMPOSSIBLE LENGTHS")
    t_rep = torch.tensor([[5, 5, 5, 0]])
    try:
        check_lengths(torch.tensor([4]), t_rep, torch.tensor([3]), ["utt-X"])
        check("'EEE' in 4 frames is rejected (needs 5: 3 letters + 2 separators)", False)
    except ValueError as e:
        check("'EEE' in 4 frames is rejected (needs 5: 3 letters + 2 separators)", True, str(e))
    check_lengths(torch.tensor([5]), t_rep, torch.tensor([3]))
    check("'EEE' in 5 frames is accepted", True)

    header("5. OPTIMIZATION SANITY")
    logits = torch.zeros(1, 30, C, requires_grad=True)
    tgt = torch.tensor([[10, 7, 14, 14, 17]])                      # H E L L O
    opt = torch.optim.Adam([logits], lr=0.1)
    first = None
    for step in range(200):
        loss = crit(logits.log_softmax(-1), torch.tensor([30]), tgt, torch.tensor([5]))
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    best = logits.argmax(-1)[0].tolist()
    decoded = "".join(Vocabulary().tokens[i] for i in collapse(best))
    check("loss falls and best path spells the target", loss.item() < 0.05 * first and decoded == "HELLO",
          f"{first:.3f} -> {loss.item():.4f}, best path decodes to {decoded!r}")

    header("6. REAL PIPELINE (dev-clean batch, untrained model)")
    cfg = load_config("configs/base.yaml")
    vocab = Vocabulary()
    rows = read_manifest(resolve(cfg, "manifests_dir") / f"{cfg['data']['dev_subset']}.csv")
    batch = collate([ASRDataset(rows[:6], resolve(cfg, "raw_dir"), vocab)[k] for k in range(6)])
    model = CNNBiLSTM.from_config(cfg["model"], cfg["features"]["n_mels"], len(vocab))
    results = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        torch.manual_seed(0)
        m = CNNBiLSTM.from_config(cfg["model"], cfg["features"]["n_mels"], len(vocab))
        m.load_state_dict(model.state_dict())
        m = m.to(dev).eval()
        fe = LogMel(**cfg["features"]).to(dev)
        feats, fl = fe(batch["wav"].to(dev), batch["wav_lens"].to(dev))
        with torch.no_grad():
            lp, ol = m(feats, fl)
        check_lengths(ol, batch["targets"], batch["target_lens"], batch["utt_ids"])
        results[dev] = CTCLoss()(lp, ol.cpu(), batch["targets"].to(dev), batch["target_lens"].to(dev)).item()
    print(f"batch: {len(batch['utt_ids'])} utterances, {int(batch['target_lens'].sum())} characters, "
          f"features {tuple(feats.shape)} -> log probs {tuple(lp.shape)}")
    print(f"untrained loss per character: {results['cpu']:.3f}")
    blank_p = lp.exp()[..., vocab.blank_id].mean().item()
    print(f"untrained mean blank probability: {blank_p:.3f}  (uniform would be {1 / len(vocab):.3f})")
    check("untrained loss finite and positive", 0 < results["cpu"] < float("inf"))
    if "cuda" in results:
        check("GPU loss matches CPU", abs(results["cuda"] - results["cpu"]) < 1e-3 * results["cpu"],
              f"{results['cuda']:.4f} vs {results['cpu']:.4f}")
        m = m.train()
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            lp, ol = m(feats, fl)
        loss = CTCLoss()(lp, ol.cpu(), batch["targets"].cuda(), batch["target_lens"].cuda())
        scaler.scale(loss).backward()
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        check("mixed precision forward + backward on GPU: finite loss and gradients",
              math.isfinite(loss.item()) and grads_ok, f"loss {loss.item():.3f}")
    else:
        print("[SKIP] no GPU")

    print("\nRESULT: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
