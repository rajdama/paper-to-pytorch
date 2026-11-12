"""
Fine-tune a pretrained TinyGPT with LoRA to write Shakespeare in ALL CAPITALS.

A small, visible change of style: the pretrained model already knows Shakespeare; LoRA
teaches it a new "dialect" while ~97% of the weights stay frozen.

First pretrain:  python -m bonus.step13_gpt_decoder_only.train_gpt
Then run:        python -m bonus.step16_lora.finetune_caps
"""

import argparse
import sys

import torch

from bonus.data import ROOT, tiny_shakespeare
from bonus.step13_gpt_decoder_only.generate import load
from bonus.step13_gpt_decoder_only.sampling import generate
from bonus.step16_lora.lora import apply_lora, count_trainable, merge_lora

CKPT = ROOT / "checkpoints" / "tinygpt_char.pt"


def caps_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / max(1, len(letters))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    if not CKPT.exists():
        sys.exit("No pretrained model. Run:  python -m bonus.step13_gpt_decoder_only.train_gpt")
    torch.manual_seed(0)
    model, tok = load(CKPT)
    block = model.cfg.block_size

    text = tiny_shakespeare().upper()                       # the new "style"
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def batch(src):
        ix = torch.randint(len(src) - block - 1, (args.batch_size,))
        return (torch.stack([src[i:i + block] for i in ix]),
                torch.stack([src[i + 1:i + block + 1] for i in ix]))

    @torch.no_grad()
    def val_loss():
        model.eval()
        g = torch.Generator().manual_seed(123)
        losses = []
        for _ in range(20):
            ix = torch.randint(len(val_data) - block - 1, (args.batch_size,), generator=g)
            x = torch.stack([val_data[i:i + block] for i in ix])
            y = torch.stack([val_data[i + 1:i + block + 1] for i in ix])
            losses.append(model(x, y)[1].item())
        return sum(losses) / len(losses)

    def sample(seed=0):
        model.eval()
        out = generate(model, torch.tensor([tok.encode("ROMEO:")]), 200, temperature=0.8,
                       top_k=30, generator=torch.Generator().manual_seed(seed))
        return tok.decode(out[0].tolist())

    before_text = sample()
    before_loss = val_loss()

    apply_lora(model, targets=("w_q", "w_v"), rank=args.rank, alpha=args.alpha)
    trainable, total = count_trainable(model)
    print(f"LoRA rank {args.rank} on every query and value projection: "
          f"training {trainable:,} of {total:,} parameters ({trainable / total:.2%})\n")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    for step in range(1, args.steps + 1):
        _, loss = model(*batch(train_data))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:4d}  loss on ALL-CAPS text {loss.item():.3f}")
            model.train()

    after_loss = val_loss()
    merge_lora(model)                                     # fold B @ A into W: a normal TinyGPT again
    after_text = sample()

    print(f"\nvalidation loss on ALL-CAPS Shakespeare: {before_loss:.3f} -> {after_loss:.3f}")
    print(f"share of generated letters that are capitals: "
          f"{caps_fraction(before_text):.0%} -> {caps_fraction(after_text):.0%}")
    print("\n----- before LoRA -----\n" + before_text)
    print("\n----- after LoRA (merged) -----\n" + after_text)


if __name__ == "__main__":
    main()
