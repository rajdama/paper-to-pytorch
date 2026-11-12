# Bonus 16 · LoRA: fine-tuning with 2% of the parameters

**Paper:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

## The idea

Fine-tuning a pretrained model normally updates every weight. For a 7B-parameter model,
that means storing gradients and optimizer state for 7 billion numbers, and saving a full
copy of the model for every task.

LoRA's observation: the **change** a fine-tune makes to a weight matrix has low rank. So
instead of learning a full $\Delta W$, freeze $W$ and learn two thin matrices:

$$W' = W + \frac{\alpha}{r}\, B A, \qquad A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r},\; r \ll d$$

```
x ──► [ W (frozen) ] ─────────────┐
 │                                ├─► (+) ──► out
 └──► [ A: d→r ] ─► [ B: r→d ] ───┘
        trained       trained, starts at 0
```

- **B starts at zero**, so before any training the model is *exactly* the pretrained one.
- **Merging:** after training, $W + \frac{\alpha}{r}BA$ is just a matrix. Fold it in and inference
  costs nothing extra. A test checks the merged model gives identical outputs.
- **Where:** the paper's default adapts the attention **query** and **value** projections.

## The demo: teaching TinyGPT to SHOUT

`finetune_caps.py` takes the TinyGPT pretrained in Bonus 13 and fine-tunes it on Tiny
Shakespeare converted to ALL CAPITALS: a new style, same language. Only the LoRA matrices on
`w_q` and `w_v` train; everything else is frozen.

## Results

Rank 8 on every query and value projection: **16,384 trainable parameters out of 834,432
(1.96%)**, 400 steps, about 2 minutes on a CPU.

| | before LoRA | after LoRA (merged) |
|---|---:|---:|
| validation loss on ALL-CAPS Shakespeare | 4.298 | **2.068** |
| share of generated letters that are capitals | 9% | **99%** |

```
before:  ROMEO:
         The most kind self it one means by this toster.

after:   ROMEO:
         THOS WOULY RETHTSER THHE MEeT AND ON SIN
         TO SEUM FERD BREOO AND TO MOTARY SOR HORTABTELT.
```

It learned the new style almost perfectly but the spelling got worse. To the model,
`"THE"` and `"the"` are completely different character sequences, so it has to relearn every
word's spelling through 2% of its weights in 400 steps. The validation loss (2.07, against
1.67 for the original model on normal text) reflects that. More steps or a higher rank close
the gap; the "Try this" section below has experiments.

Log: [`results/tinygpt/lora_log.txt`](../../results/tinygpt/lora_log.txt).

## What's implemented

| File | Contents |
|---|---|
| `lora.py` | `LoRALinear` (with `merged()`), `apply_lora(model, targets, rank, alpha)`, `merge_lora`, `count_trainable` |
| `test_lora.py` | starts identical to the base, **only A and B get gradients**, merge gives the same function, the update has rank r, parameter count, **a rank-2 LoRA learns a rank-2 change** |
| `finetune_caps.py` | the ALL-CAPS fine-tuning demo, before/after loss and samples |

## Run it

```bash
python -m bonus.step16_lora.test_lora
python -m bonus.step13_gpt_decoder_only.train_gpt          # pretrain first
python -m bonus.step16_lora.finetune_caps
```

## Try this

- Try `--rank 1`. Is one dimension of change enough to learn "capitalise everything"?
- Adapt only `w_v`, or add `w_o` and the feed-forward layers (`linear1`, `linear2`). Which matters most?
- Full fine-tuning would train all parameters. Count the optimizer memory saved (AdamW keeps 2 extra numbers per trained parameter).
