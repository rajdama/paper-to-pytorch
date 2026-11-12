# Bonus 14 · Byte Pair Encoding tokenizer

**Papers:** Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016); byte-level BPE from GPT-2 (Radford et al., 2019). The original Transformer paper also used BPE (Section 5.1, ~37,000 tokens).

## The idea

Characters make sequences very long; whole words need a gigantic vocabulary and still
fail on new words. BPE sits in between and *learns* its vocabulary from data:

1. Start with the **256 possible bytes**, so any text in any language (and emoji) can be encoded. No `<unk>` token is ever needed.
2. Count every adjacent pair of tokens in the training text.
3. Merge the **most frequent pair** into a new token.
4. Repeat until the vocabulary is the size you want.

```
"aaabdaaabac"   merge (a, a) -> Z    "ZabdZabac"
                merge (Z, a) -> Y    "YbdYbac"
                merge (Y, b) -> X    "XdXac"        11 bytes -> 5 tokens
```

To encode new text, replay the learned merges **in the order they were learned**.

### Pre-splitting (GPT-2's trick)
Before training, the text is cut into chunks: words with their leading space, numbers,
punctuation, and whitespace. Merges never cross a chunk boundary, so `"dog."`, `"dog!"` and
`"dog?"` all share one token `" dog"` instead of wasting three vocabulary slots.

### Making it fast
Training doesn't loop over 1.1M characters for each merge. It counts **unique chunks** once
(`" the"` appears ~15,000 times but is stored once with weight 15,000), so each merge only
touches a few thousand short sequences. Training 1,024 tokens on Tiny Shakespeare takes
seconds in pure Python.

## Results on Tiny Shakespeare

From [`results/tinygpt/bpe_log.txt`](../../results/tinygpt/bpe_log.txt):

| vocabulary | tokens for all 1,115,394 bytes | bytes per token | training time |
|---:|---:|---:|---:|
| 256 (bytes only) | 1,115,394 | 1.00 | – |
| 512 | 575,345 | 1.94 | 8 s |
| 1024 | 459,760 | 2.43 | 21 s |

The first merges are the most common pairs in English: `' t'`, `'he'`, `' a'`, `'ou'`, then
`' t' + 'he' → ' the'` at merge 12. With 1,024 tokens, the longest are character names that
appear in nearly every scene: `'GLOUCESTER'`, `' VINCENTIO'`, `'CORIOLANUS'`, `'PETRUCHIO'`.

How one line gets split:

```
vocab  256 (58 tokens): R|O|M|E|O|:| |B|u|t|,| |s|o|f|t|!| |w|h|a|t| |l|i|g|h|t| ...
vocab  512 (33 tokens): R|O|M|E|O|:| |But|,| so|f|t|!| what| li|ght| th|r|ou|gh| y|o|nd|er| w|ind|ow| b|re|a|k|s|?
vocab 1024 (25 tokens): ROMEO|:| |But|,| so|f|t|!| what| li|ght| th|r|ough| y|ond|er| w|ind|ow| bre|ak|s|?
```

### Does it help the language model?
TinyGPT trained on 512 BPE tokens (Bonus 13, same settings) reaches a validation loss of
2.948 per **token**. At 1.94 characters per token that's about **1.52 per character**, better
than the 1.67 the character-level model reaches. Each 128-token context also covers about
twice as much text. (This conversion is approximate: it uses the average over the whole text.)

## What's implemented

| File | Contents |
|---|---|
| `bpe.py` | `get_pair_counts`, `merge`, `BPETokenizer` (train / encode / decode / save / load), GPT-2-style split pattern |
| `test_bpe.py` | the Wikipedia example, **round-trips any Unicode text**, split pattern never drops characters, merges never cross word boundaries, save/load, determinism |
| `demo.py` | compression at different vocabulary sizes, the first merges learned, how a line gets split |

Pure Python: no PyTorch needed for this step.

## Run it

```bash
python -m bonus.step14_bpe_tokenizer.test_bpe
python -m bonus.step14_bpe_tokenizer.demo
python -m bonus.step13_gpt_decoder_only.train_gpt --tokenizer bpe --vocab-size 512   # train TinyGPT on BPE tokens
```

## Try this

- Encode `" hello"` and `"hello"`. Why are they different tokens? (This is why LLMs are bad at spelling and counting letters.)
- Train with `BPETokenizer(pattern=None)` on Tiny Shakespeare. What strange tokens appear without pre-splitting?
- Encode a sentence in Hindi. How many tokens does each character cost, and why? (Hint: UTF-8 and an English-only training text.)
