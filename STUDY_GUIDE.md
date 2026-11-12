# Study guide

Questions to check you understand each step, the kind an interviewer asks about
Transformers. Try to answer before opening the answer. Every answer points to the code
or test in this repo that demonstrates it.

## Paper 1 · Attention Is All You Need

### Embeddings and positions (Steps 01–02)

<details>
<summary>Why are embeddings multiplied by √d<sub>model</sub>?</summary>

The paper doesn't say. The common explanation: the positional encodings added next lie in
[−1, 1], while freshly initialised embeddings are small, so the position signal would drown
out the token. Scaling keeps both at a similar size. (It also matters because the same matrix
is shared with the output layer.) See `step01_embeddings/embeddings.py`.
</details>

<details>
<summary>Why does a Transformer need positional encoding at all, when an RNN doesn't?</summary>

Self-attention is **permutation-equivariant**: shuffle the input tokens and the outputs are
shuffled the same way, with identical values. Nothing in attention knows order. An RNN reads
tokens one after another, so order is built into its computation. `vit/step02_vit_model/test_vit.py::test_position_embeddings_make_patch_order_matter`
shows this directly: with position embeddings zeroed, shuffling the patches doesn't change the output.
</details>

<details>
<summary>Sinusoidal vs learned position embeddings: what's the trade-off?</summary>

The paper found they perform about the same (Table 3, row E). Sinusoids need no parameters
and give values for any position, so they can in principle extend past the training length.
Learned embeddings (GPT-2, ViT) are a table with a hard maximum, `block_size`. RoPE (Bonus 17)
encodes position as a rotation, so scores depend only on relative distance.
</details>

### Attention (Steps 03–05)

<details>
<summary>Why divide by √d<sub>k</sub>?</summary>

If q and k have independent components with variance 1, their dot product has variance
d<sub>k</sub>. Large scores push softmax towards a one-hot output, where gradients are almost
zero. Dividing by √d<sub>k</sub> brings the variance back to 1. Try it in the
[Playground](playground/index.html) or notebook 1.
</details>

<details>
<summary>What is the computational cost of self-attention, and why does it matter?</summary>

The score matrix QK<sup>⊤</sup> is n × n, so time and memory grow as O(n² · d) in sequence
length n. Doubling the context quadruples the attention cost. This is why long-context models
need tricks such as FlashAttention, sparse attention, or sliding windows, and why ViT uses
16×16 patches rather than single pixels.
</details>

<details>
<summary>Why use multiple heads instead of one big attention?</summary>

One attention computes one weighted average per token, so it can only express one pattern of
"who to look at". With h heads of size d/h, the total cost is about the same, but each head
can attend to something different (previous token, matching subject, punctuation…).
`step05_multi_head_attention/demo.py` shows different heads producing different maps.
</details>

<details>
<summary>What's the difference between the padding mask and the causal mask?</summary>

The padding mask hides `<pad>` filler tokens (shape `(B, 1, 1, T)`, depends on the data). The
causal mask stops position i from seeing positions after i (shape `(1, 1, T, T)`, lower
triangular). The decoder's self-attention uses both, combined with AND. See `step04_masking/masks.py`.
</details>

<details>
<summary>In cross-attention, where do Q, K and V come from?</summary>

Q from the decoder; K and V from the encoder output (the "memory"). That's why the query and
key lengths can differ: 5 target tokens can attend over 9 source tokens.
`step12_inference/attention_map.py` plots it for a trained model.
</details>

### The layer (Steps 06–09)

<details>
<summary>What do attention and the feed-forward network each contribute?</summary>

Attention mixes information **between** positions (communication). The feed-forward network
processes each position **independently** (computation), and holds about two thirds of each
layer's parameters. `step06_feed_forward/test_feed_forward.py::test_each_position_processed_independently`.
</details>

<details>
<summary>Post-norm vs pre-norm?</summary>

The paper uses post-norm: `LayerNorm(x + Sublayer(x))`. GPT-2 and nearly all later models use
pre-norm: `x + Sublayer(LayerNorm(x))`. Pre-norm keeps a clean residual path from input to output,
so deep stacks train stably without careful warmup; post-norm can work slightly better when it
trains, but is more fragile. `step07_add_and_norm/add_and_norm.py` supports both.
</details>

<details>
<summary>LayerNorm vs BatchNorm: why LayerNorm here?</summary>

BatchNorm normalises each feature across the batch, so it depends on batch size and on padding
in variable-length sequences, and behaves differently at inference. LayerNorm normalises each
token's vector on its own, so it works the same for any batch size or sequence length.
</details>

### Training and inference (Steps 10–12)

<details>
<summary>What is teacher forcing, and what's the "shift right"?</summary>

During training the decoder receives the correct previous tokens (`<bos> y1 y2 …`) and predicts
the next ones (`y1 y2 … <eos>`) all at once. Input and target are the same sequence offset by one
position. The causal mask keeps each prediction from seeing its own answer. `step11_training/train_reverse.py`.
</details>

<details>
<summary>Why warmup?</summary>

Early in training, Adam's moment estimates are noisy and post-norm Transformers' gradients
can be large; a big learning rate then can blow training up. Warmup increases the rate
gradually. The paper's schedule: linear warmup for 4000 steps, then inverse-square-root decay.
</details>

<details>
<summary>What does label smoothing do, and what's the catch?</summary>

The target becomes 90% the right token and 10% spread over the vocabulary, which stops the
model from becoming over-confident. The paper notes it makes perplexity worse but improves
accuracy and BLEU. `step11_training/test_training.py` checks it against PyTorch's implementation.
</details>

<details>
<summary>Greedy vs beam search: is beam search always better?</summary>

No. Beam search finds sequences with higher total probability, but that isn't always the
correct or best output, and without a length penalty it prefers short outputs. In this repo's
reverse task, greedy scored 99% and beam search 98%. For open-ended generation (Bonus 13),
sampling usually reads better than either.
</details>

## From the Transformer to LLMs (Bonus 13–17)

<details>
<summary>Why is GPT "decoder-only", and what was removed?</summary>

There's no source sentence, so the encoder goes, and with it the cross-attention sub-layer.
What's left is masked self-attention + feed-forward, trained to predict the next token.
See the highlighted diagram in `bonus/step13_gpt_decoder_only`.
</details>

<details>
<summary>Why byte-level BPE instead of words or characters?</summary>

Words need a huge vocabulary and still fail on unseen words; characters make sequences long, and
attention costs O(n²). BPE learns frequent chunks, so common words become one token while rare
words split into pieces. Starting from bytes means any input can be encoded and no `<unk>`
token is needed.
</details>

<details>
<summary>Why can you cache K and V but not Q?</summary>

With a causal mask, an old token's keys and values never change when new tokens arrive. The
query is only needed for the token being generated now; old queries are never used again.
`bonus/step15_kv_cache`. Measured speedup on CPU: about 10× at 448 tokens.
</details>

<details>
<summary>Why does LoRA initialise B to zero?</summary>

So that B·A = 0 at the start: the adapted model is exactly the pretrained model, and
fine-tuning starts from a working point. If both were random, the first step would corrupt
the pretrained function. `bonus/step16_lora/test_lora.py::test_starts_identical_to_base`.
</details>

<details>
<summary>What does RoPE give you that learned position embeddings don't?</summary>

Scores that depend only on relative distance (m − n), and no fixed-size position table.
`bonus/step17_llama_style/test_llama.py::test_rope_depends_only_on_relative_position`.
</details>

## Paper 2 · Vision Transformer

<details>
<summary>Why is the patch embedding a Conv2d?</summary>

A convolution with kernel size = stride = P applies the same linear map to every
non-overlapping P×P patch, which is exactly "flatten each patch and multiply by E".
`vit/step01_patch_embedding/test_patch_embedding.py::test_conv_equals_patchify_plus_linear`.
</details>

<details>
<summary>Why did ViT need so much data to beat CNNs?</summary>

CNNs have built-in assumptions (inductive biases): locality and translation equivariance. ViT
has almost none: every patch attends to every patch from layer 1, and it must learn spatial
structure from data. With ImageNet alone it underperformed ResNets; with 14M–300M images it
overtook them (paper, Section 4.3).
</details>

<details>
<summary>What is the [CLS] token for?</summary>

It's an extra learned token with no image content. Through attention it collects information
from every patch, and its final vector is fed to the classifier. Averaging all patch outputs
works about as well (paper, Appendix D.3).
</details>
