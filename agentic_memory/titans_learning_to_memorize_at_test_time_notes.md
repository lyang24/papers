# Titans: Learning to Memorize at Test Time

**Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)

**Paper:** arXiv:2501.00663v1 (Dec 31, 2024)

---

## The Core Problem

The field has long faced a tension between two memory paradigms for sequence modeling:

1. **Attention (Transformers)** keeps *all* tokens in the context window without compression and models pairwise dependencies exactly — but at **quadratic cost** in time and memory, capping the usable context length. From a memory lens, attention is a **short-term memory**: precise but bounded.
2. **Recurrent / linear models** (linear Transformers, Mamba, DeltaNet, RetNet, GLA) compress history into a **fixed-size** vector- or matrix-valued hidden state — cheap and scalable, but a very long context cannot be faithfully squeezed into a small fixed state, so performance lags Transformers.

The authors frame the deeper limitation as a *memory-system* deficiency. Drawing on neuropsychology, they argue effective learning needs **distinct yet interconnected memory modules** (short-term, long-term, meta/persistent), and that existing architectures are missing one or more, plus the ability to *actively learn from data and memorize the abstraction of past history*. They distill this into five questions: (Q1) what is a good memory **structure**, (Q2) a good memory **update** rule, (Q3) a good **retrieval** process, (Q4) how to **interconnect** multiple memory modules, and (Q5) is a **deep** memory module needed to store long history.

The central observation that prior compressive-memory models share: they update memory **additively** (causing overflow on long context), rely only on **momentary surprise** (gradient w.r.t. the current token, missing the token *flow*), and most **lack a forgetting gate** — leading to poor memory management.

---

## The Big Idea: a Neural Long-Term Memory that Learns at Test Time

Titans introduces a **neural long-term memory module (LMM)** — a small MLP whose **weights are updated during inference** (test time), treating memorization as an *online learning* problem. The memory is a **meta in-context learner**: the outer loop (normal pretraining) learns *how to memorize*; the inner loop (running at test time) actually *writes* the incoming sequence into the memory's parameters. This makes memory **parametric and learned**, not an external text store.

Three ingredients make the update rule work:

### 1. Surprise as the write signal (momentary + past surprise)

Inspired by human memory ("an event that violates expectations is more memorable"), **surprise is measured as the gradient of an associative-memory loss w.r.t. the input**. A naive version updates memory by gradient descent on that loss:

```
M_t = M_{t-1} − θ_t · ∇ℓ(M_{t-1}; x_t)        ← momentary surprise only
```

But pure momentary surprise misses information arriving *after* a big surprise (the gradient flattens out / gets stuck in a local minimum). So surprise is split into two parts, combined like **gradient descent with momentum**:

```
M_t = M_{t-1} + S_t
S_t = η_t · S_{t-1}  −  θ_t · ∇ℓ(M_{t-1}; x_t)
       └ past surprise ┘   └ momentary surprise ┘
```

- `S_t` (the momentum term) acts as **a memory of surprise across the sequence** — a recent surprise keeps biasing what gets memorized.
- `η_t` is a **data-dependent surprise decay**: η→0 ignores past surprise (context just changed), η→1 fully carries it forward (token highly relevant to recent past).
- `θ_t` is a **data-dependent learning rate** gating how much momentary surprise enters memory.

### 2. Adaptive forgetting (weight decay)

To manage finite capacity over millions of tokens, an **adaptive forgetting gate** `α_t ∈ [0,1]` is added as a weight-decay term:

```
M_t = (1 − α_t) · M_{t-1} + S_t
```

α→0 keeps the past intact; α→1 wipes memory. The paper shows this **weight-decay mechanism generalizes the forgetting/gating used in modern RNNs** (Mamba2, LRU, etc.). Net result: the whole rule is exactly **mini-batch gradient descent with momentum and weight decay**, which is also what lets it be parallelized.

### 3. Associative-memory objective + retrieval

The inner-loop loss is associative (key→value), with keys/values projected from the token:

```
ℓ(M_{t-1}; x_t) = ‖ M_{t-1}(k_t) − v_t ‖²₂      k_t = x_t W_K ,  v_t = x_t W_V
```

`W_K, W_V` (and `W_Q`) are *outer-loop* hyperparameters; only `M` is updated in the inner loop. **Retrieval is just a forward pass with no weight update:** `y_t = M*(q_t)` where `q_t = x_t W_Q`.

### Deep memory

Unlike vector/matrix memories (which implicitly solve *linear* regression on history), Titans uses an MLP with `L_M ≥ 1` layers; `L_M ≥ 2` is strictly more expressive and empirically helps (see ablation §Deep Memory). This is the answer to Q5.

### Parallelizable training

Although the inner-loop update is O(N) sequential in theory, the paper tensorizes it: chunk the sequence (size b), express the chunked gradient-descent-with-momentum-and-weight-decay update via **matmuls + a parallel associative scan** for the momentum recurrence. This gives **parallelizable training with fast inference**, scaling **linearly** with context length.

---

## Architecture: Three Hyper-Heads, Three Variants

Titans is built from three "hyper-heads":
1. **Core** — short-term memory; the main data flow. Implemented as **attention with a limited (sliding) window**.
2. **Long-term memory** — the neural LMM above (still learning at test time).
3. **Persistent memory** — `N_p` **learnable but data-independent** parameters prepended to the sequence, encoding task knowledge (a.k.a. meta-memory). Motivated three ways: a task-knowledge store; an FFN-as-attention view (`FFN(x)=W_V·Softmax(W_K x)` behaves like data-independent attention weights); and a technical fix for attention's bias toward initial tokens.

The open question — *how to wire the LMM into a deep model* — yields **three variants**:

| Variant | How memory is used | Mechanism |
|---|---|---|
| **MAC** — Memory as Context | Memory output is **concatenated into the attention context** | Sequence is chunked into segments. For segment S^(t), the query retrieves history `h_t = M*_{t-1}(q_t)`; the attention input becomes `[persistent ‖ h_t ‖ S^(t)]`. Attention then **decides what is worth writing** into the LMM, which is updated with the attention output. |
| **MAG** — Memory as Gate | Memory and short-term attention run as **two parallel branches, fused by a gate** | One branch = **sliding-window attention** (with persistent prefix); other branch = the LMM over the full stream. Outputs combined by a non-linear gate `o = y ⊗ M*(x̃)`. Reads like a multi-head design with heterogeneous heads. SWA = precise short-term memory; LMM = fading long-term memory. |
| **MAL** — Memory as Layer | Memory is a **layer stacked before attention** | `y = M(x̃)` then `o = SW-Attn(y)` — the conventional "stack a recurrent layer under attention" hybrid (H3-style). Limited because it can't exploit attention and memory *complementarily*; capped by the weaker of the two layers. |

A fourth configuration, **LMM (a.k.a. Titans LMM)**, uses the neural memory **alone, no attention** — to test whether the long-term memory is a strong sequence model on its own.

**Key architectural finding:** MAC and MAG both beat MAL, even though all three share the same modules — so the *wiring matters*. Most existing hybrids in the literature are MAL-style, which the paper argues is suboptimal. MAC is best for long-range dependencies; MAG is close and trains faster.

**Details:** SiLU activations, ℓ₂-normalized queries/keys, 1D depthwise-separable convolution after Q/K/V projections, residual connections, gated output projection. **Theorem 4.1:** Titans can solve problems **beyond TC⁰**, making them strictly more expressive than Transformers, diagonal linear RNNs, and DeltaNet (all confined to TC⁰) on state-tracking tasks.

---

## Experimental Results

**Setup:** 4 scales (170M / 340M / 400M / 760M params). The 170M–400M models train on 15B tokens of FineWeb-Edu; 760M on 30B tokens. LLaMA-2 tokenizer (32K vocab), 4K training length, AdamW (lr 4e-4, cosine), 0.5M-token batches.

### Language Modeling & Common-Sense Reasoning (Table 1)

Perplexity (Wiki, LMBADA) and accuracy across PIQA, HellaSwag, WinoGrande, ARC-e/c, SIQA, BoolQ. Hybrid models marked `*`. Selected numbers:

**340M params / 15B tokens**

| Model | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ |
|---|---|---|---|
| Transformer++ | 31.52 | 41.08 | 42.92 |
| Mamba | 30.83 | 40.21 | 43.59 |
| TTT | 27.44 | 34.19 | 44.51 |
| Gated DeltaNet | 27.01 | 30.94 | 45.42 |
| **Titans (LMM)** | 26.18 | 29.97 | 46.17 |
| **Titans (MAL)\*** | 24.69 | 28.80 | 46.55 |
| **Titans (MAC)\*** | 25.43 | 28.13 | 47.36 |
| **Titans (MAG)\*** | 25.07 | 28.72 | **47.54** |

**400M params / 15B tokens**

| Model | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ |
|---|---|---|---|
| Transformer++ | 30.63 | 37.37 | 45.64 |
| Mamba2 | 26.34 | 33.19 | 46.91 |
| Gated DeltaNet | 25.47 | 29.24 | 47.26 |
| Gated DeltaNet-H2\* | 24.19 | 28.09 | 47.69 |
| **Titans (LMM)** | 25.03 | 28.99 | 47.83 |
| **Titans (MAG)\*** | 23.59 | 27.81 | 48.60 |
| **Titans (MAC)\*** | 25.61 | 27.73 | **48.65** |

**760M params / 30B tokens**

| Model | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ |
|---|---|---|---|
| Transformer++ | 25.21 | 27.64 | 48.69 |
| Mamba2 | 22.94 | 28.37 | 48.34 |
| Gated DeltaNet | 21.18 | 22.09 | 49.69 |
| Samba\* | 20.63 | 22.71 | 51.08 |
| Gated DeltaNet-H2\* | 19.88 | 20.83 | 51.49 |
| **Titans (LMM)** | 20.04 | 21.96 | 51.56 |
| **Titans (MAL)\*** | 19.07 | 20.33 | 50.97 |
| **Titans (MAG)\*** | 18.61 | 19.86 | 52.50 |
| **Titans (MAC)\*** | 19.93 | 20.12 | **52.51** |

Findings: the **LMM alone is the best non-hybrid model** (beats Transformer++, Mamba2, TTT, Gated DeltaNet) — isolating the value of momentum + weight decay over TTT (which is also gradient-based but lacks both). All three Titans hybrids beat Samba and Gated DeltaNet-H2.

### Needle-in-a-Haystack (S-NIAH, RULER benchmark — Table 2)

Effective context length at sequence lengths 2K–16K. Accuracy (%):

| Model | PK-2K | PK-8K | PK-16K | N-2K | N-16K | W-2K | W-16K |
|---|---|---|---|---|---|---|---|
| TTT | 98.4 | 98.0 | 88.4 | 60.2 | 4.4 | 78.8 | 0.0 |
| Mamba2 | 98.6 | 31.0 | 5.4 | 98.4 | 0.0 | 42.2 | 0.0 |
| DeltaNet | 96.8 | 98.6 | 71.4 | 47.2 | 5.4 | 46.2 | 0.0 |
| **Titans (LMM)** | 99.8 | 98.2 | 96.2 | 100.0 | 80.2 | 90.4 | 80.6 |
| **Titans (MAG)** | 99.4 | 97.4 | 97.4 | 99.2 | 98.6 | 98.0 | 88.2 |
| **Titans (MAL)** | 98.8 | 98.8 | 97.8 | 99.8 | 96.4 | 98.0 | 90.4 |
| **Titans (MAC)** | 99.2 | 99.0 | **98.4** | 99.6 | 97.4 | 98.2 | **95.2** |

Baselines collapse as length grows (Mamba2 → near 0 at 16K, can't *erase* memory); Titans stay near-flat, with **MAC best**. Attributed to momentum + forgetting + deep non-linear memory.

### BABILong (reasoning across facts in very long documents — Figure 6)

**Titans (MAC) outperforms every baseline, including far larger models** — Mamba-2.8B, RWKV-6-7B, RecurrentGemma-9B, Gemma-9B, Llama3.1-8B, **GPT-4, GPT-4o-mini** — in both the **few-shot** setup (using much smaller Titans) and the **fine-tuned** setup (vs. RMT, vs. Llama3.1-8B+RAG, vs. GPT-4 / Qwen2.5-72B / Llama3.1-70B). Notably, **Llama3.1-8B + RAG performs worse than Titans with ~70× fewer parameters**, and Titans scales **beyond a 2M-token context window** with higher accuracy than baselines — the headline long-context claim.

### Time-Series Forecasting (Table 3)

Dropping the LMM into the Simba framework (replacing its Mamba block). On ETT/ECL/Traffic/Weather, the **Neural Memory module outperforms all baselines** (Mamba-based Simba, Transformer-based PatchTST/iTransformer/Crossformer, linear RLinear/TiDE/DLinear, TimesNet). E.g. ETTm1 MSE/MAE = **0.358 / 0.387** (best); ETTm2 = **0.261 / 0.309**; ECL = **0.162 / 0.261**.

### DNA Modeling (GenomicsBenchmarks — Table 4, top-1 accuracy %)

| Model | Enhancer Cohn | Enhancer Ens | Human Reg. | Non-TATA Prom. | Human OCR Ens. |
|---|---|---|---|---|---|
| HyenaDNA | 74.2 | 89.2 | 93.8 | 96.6 | 80.9 |
| Transformer++ | 73.4 | 89.5 | 89.9 | 94.4 | 79.5 |
| Mamba | 73.0 | 89.5 | 89.5 | 96.6 | 79.0 |
| **Neural Memory Module** | 75.2 | 89.6 | 89.3 | 96.6 | 79.9 |

The LMM is **competitive with state-of-the-art genomics architectures**, showing the memory generalizes beyond language.

---

## Ablation Study (Table 5)

Base = neural memory module (LMM); one component changed at a time. Long-context column = BABILong accuracy.

| Model | LM ppl ↓ | Reasoning acc ↑ | Long-Context acc ↑ |
|---|---|---|---|
| **LMM (base)** | 27.01 | 47.83 | 92.68 |
| + Attn (MAC) | 26.67 | 48.65 | **97.95** |
| + Attn (MAG) | 25.70 | 48.60 | 96.70 |
| + Attn (MAL) | 25.91 | 47.87 | 96.91 |
| Linear Memory (vs deep) | 28.49 | 46.97 | 85.34 |
| w/o Convolution | 28.73 | 45.82 | 90.28 |
| w/o Momentum | 28.98 | 45.49 | 87.12 |
| **w/o Weight Decay** | 29.04 | 45.11 | 85.60 |
| w/o Persistent Memory | 27.63 | 46.35 | 92.49 |

**Every component helps.** Ranked by contribution: **weight decay (forgetting) > momentum > convolution > persistent memory**. Removing weight decay or using **linear (non-deep) memory** causes the biggest long-context drops (92.68 → 85.60 / 85.34) — confirming both the forgetting mechanism and **deep memory** are essential. Architecturally, **MAC and MAG ≫ MAL** in long-context, restating that the wiring (not just the modules) drives long-context gains. A separate study shows **deeper memory (L_M = 1→4) lowers perplexity at all sequence lengths** and is more robust to length, but **trains slower** — an effectiveness/efficiency trade-off.

---

## Key Takeaways

1. **Memory can be a learned process, not a stored object.** Titans reframes "remembering" as **test-time online learning**: the memory module's weights are updated by gradient descent *during inference*, so the model learns to compress arbitrary-length history into a small set of parameters as it reads.

2. **Surprise + momentum + adaptive forgetting is the write rule.** Gradient-as-surprise (momentary) plus a momentum carry (past surprise, decayed by η) plus weight-decay forgetting (α) is **exactly mini-batch gradient descent with momentum and weight decay** — which is also why it parallelizes. This generalizes the gating of modern RNNs while adding cross-token surprise tracking they lack.

3. **Deep, non-linear memory beats linear/matrix memory.** A vector/matrix memory implicitly assumes linear dependencies in history; an MLP memory (L_M ≥ 2) is strictly more expressive and empirically better at long context.

4. **How you wire memory into the model matters as much as the memory itself.** MAC/MAG (context / gated branch) consistently beat the conventional MAL (stacked layer) hybrid, even with identical components.

5. **Scales past 2M tokens with strong long-context accuracy.** A sub-billion-parameter Titans beats GPT-4 and RAG-augmented Llama on BABILong, and stays near-flat on needle-in-haystack where linear-RNN baselines collapse.

---

## Limitations

1. **v1 / preliminary scale.** The authors explicitly note this is a first version focused on *insights*; results for larger models were still being finalized. Largest reported model is 760M params on ≤30B tokens — small by LLM standards.
2. **Throughput cost of the memory.** The neural memory is **slightly slower than Mamba2 / Gated DeltaNet** due to deep memory and a more expressive update (and less-optimized kernels). Deeper memory trades accuracy for training speed.
3. **Simple MLP memory by design.** Memory is restricted to plain MLPs to isolate the *learning rule*; richer memory architectures (memory-layers, evolved transformers) are left as future work.
4. **Design-space breadth.** Choices like making η, θ, α functions of *chunks* rather than *tokens* (cheaper but less expressive), and larger-scale validation, are flagged as open.

---

## Where it sits (v1/v2)

This paper occupies the **parametric / test-time-learned / latent-memory axis** that the rest of the collection is thin on. Where almost every other system here stores memory as an **external, human-readable text artifact** that is *written then retrieved by similarity* — A-Mem's Zettelkasten notes, MemGPT's paged context store, MAGMA's multi-graph nodes, Zep's temporal KG — **Titans bakes memory into the weights of a learned neural module** and *retrieves it with a forward pass*. There is no text store, no embedding index, no retriever: remembering **is** an online gradient update, and recall **is** inference.

This maps directly onto the survey *Memory in the Age of AI Agents*' Forms taxonomy — its three branches are **token-level** (explicit, editable text), **parametric** (in the weights), and **latent** (in activations/KV/hidden states). The survey in fact lists **Titans explicitly under Latent Memory → Generate**, alongside Gist tokens, AutoCompressor, MemoRAG, and MemoryLLM: auxiliary mechanisms that *synthesize* machine-native memory rather than retrieving stored text. Titans is the clearest instance in this collection of memory as a **learned compressor of history into parameters at test time**.

The natural sibling is **LatentMem**, the collection's other learned-memory paper: both reject token-level external stores in favor of **compressed, machine-native representations injected into the model's internal state**, and both *learn* the memory policy rather than hand-coding it (LatentMem via RL/LMPO over role-aware latent embeddings; Titans via outer-loop meta-learning of an inner-loop test-time update). They differ on substrate — LatentMem injects latent *vectors* into hidden states and is built for multi-agent role-specialization, while Titans rewrites *weights* of a long-term memory module for single-stream long-context modeling — but they sit on the same v2 shift the OVERVIEW names: from **"store text and query it with similarity, using rules a human wrote"** to **"structure, learn, or generate memory with policies the system learns."** Contrasted with the token-level external-memory lineage (A-Mem, MemGPT), Titans is the **architectural / training-time** end of the spectrum: it changes *what memory is made of* rather than *how a text store is organized and retrieved.*
