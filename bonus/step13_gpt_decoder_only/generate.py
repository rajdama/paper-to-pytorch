"""
Generate text with a trained TinyGPT.

    python -m bonus.step13_gpt_decoder_only.generate --prompt "JULIET:" --tokens 300
    python -m bonus.step13_gpt_decoder_only.generate --temperature 1.2 --top-p 0.9
    python -m bonus.step13_gpt_decoder_only.generate --checkpoint checkpoints/tinygpt_bpe.pt
"""

import argparse

import torch

from bonus.data import ROOT
from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step13_gpt_decoder_only.sampling import generate


def load_tokenizer(state: dict):
    if state["type"] == "char":
        from bonus.step13_gpt_decoder_only.char_tokenizer import CharTokenizer
        return CharTokenizer.from_state(state)
    from bonus.step14_bpe_tokenizer.bpe import BPETokenizer
    return BPETokenizer.from_state(state)


def load(path):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, load_tokenizer(ckpt["tokenizer"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "tinygpt_char.pt"))
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--kv-cache", action="store_true", help="use the KV cache from bonus Step 15")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, tok = load(args.checkpoint)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    g = torch.Generator().manual_seed(args.seed)
    if args.kv_cache:
        from bonus.step15_kv_cache.kv_cache import generate_with_kv_cache
        n = min(args.tokens, model.cfg.block_size - idx.size(1))
        out = generate_with_kv_cache(model, idx, n, args.temperature, args.top_k, args.top_p, g)
    else:
        out = generate(model, idx, args.tokens, args.temperature, args.top_k, args.top_p, g)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
