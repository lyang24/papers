# LightMem: Lightweight and Efficient Memory-Augmented Generation

**Authors:** Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, Ningyu Zhang (Zhejiang University; National University of Singapore; State Key Lab. for Novel Software Technology, Nanjing University)

**Paper:** arXiv:2510.18866v4 (ICLR 2026)

**GitHub:** https://github.com/zjunlp/LightMem

---

## The Core Problem

Memory systems let LLMs move beyond stateless interaction by persisting, retrieving, and updating historical information — but contemporary systems pay a steep **efficiency tax**. LightMem identifies three concrete sources of overhead in mainstream memory pipelines:

1. **Redundant sensory memory.** In long dialogues, both user inputs and model responses contain substantial redundant, low-value content. Existing systems feed this raw data directly into `f_sum()` / topic-granularity extraction (typically executed by calling a *stronger* LLM), inflating token consumption with no proportional gain — and redundancy can even *weaken* in-context learning.

2. **Effectiveness/efficiency imbalance in short-term memory (STM).** Memory construction treats each turn in isolation or uses rigid fixed context-window boundaries. Too-fine granularity raises latency and underutilizes the STM buffer; too-coarse granularity without semantic grouping produces *entangled topics/semantics*, causing the backbone LLM to emit inaccurate or incomplete memory entries and lose fine-grained detail.

3. **Inefficient long-term memory (LTM) updating.** Updates and forgetting are usually done *during* inference. This tight coupling (i) injects long test-time latency in long-horizon tasks, and (ii) forces sequential updates due to read-after-write / write-after-read ordering constraints, rather than triggering them dynamically.

The research question: *Can we design an LLM memory that is both efficient and lightweight, inspired by human memory mechanisms?*

---

## The Big Idea: A Three-Stage Human-Memory Pipeline

LightMem is structured around the **Atkinson–Shiffrin model** of human memory, mapping each cognitive stage to a "light" module that filters, organizes, then consolidates information:

| Module | Cognitive Analogue | What It Does | Key Mechanism |
|---|---|---|---|
| **Light1 — Sensory Memory** | Sensory register / pre-attention | Rapidly filters redundant tokens from raw input and groups remainder by topic | Lightweight pre-compression (LLMLingua-2) + hybrid topic segmentation |
| **Light2 — Short-Term Memory** | Working memory buffer | Consolidates topic-based groups into structured, summarized memory units | Topic-aware buffering; summarize only when token buffer fills |
| **Light3 — Long-Term Memory** | Consolidation during sleep | Offline, decoupled reorganization / de-duplication / abstraction of entries | "Soft" test-time inserts + **sleep-time** parallel offline updates |

The unifying thesis: instead of running a heavy LLM call per memory unit on the critical path, LightMem **pushes cheap filtering forward** (token-level pre-compression) and **pushes expensive consolidation backward** (offline sleep-time update), so online inference stays fast and cheap.

---

## Architecture

### Light1 — Cognition-Inspired Sensory Memory (§3.1)

**Pre-Compressing Submodule.** A compression model θ removes redundant tokens before any LLM ever sees them. Following TokenSkip (Xia et al. 2025), LightMem uses **LLMLingua-2** (a lightweight BERT-style bidirectional encoder, <2GB GPU memory). Compression is framed as binary token classification ("retain" vs "discard"):

```
x̂ = { x_i ∈ x | P(retain x_i | x; θ) > τ },   τ = Percentile({x_j}, r)
P(retain x_i | x; θ) = softmax(ℓ_i)_1
```

where `r` is the compression rate (retention ratio) and τ is the r-th percentile of retention scores. LightMem can alternatively use a generative LLM with a cross-entropy / conditional-entropy criterion (high-entropy = more informationally unique = retained). Filtered content is buffered in a **sensory memory buffer** (size 512 tokens in experiments).

**Topic Segmentation Submodule.** When the sensory buffer fills, a **hybrid attention + similarity** segmentation fires. Using θ's attention matrix `M ∈ R^{n×n}` (turn-level) and an embedding model:

```
B1 = { k | M_{k,k−1} > M_{k−1,k−2}  and  M_{k,k−1} > M_{k+1,k}, 1<k<n }   (attention local maxima)
B2 = { k | sim(s_{k−1}, s_k) < τ, 1≤k<n }                                 (low semantic similarity)
B  = B1 ∩ B2
```

The intersection of attention-based (B1) and similarity-based (B2) boundaries mitigates attention sinks/dilution, yielding topic boundaries that beat either signal alone (>80% absolute segmentation accuracy).

### Light2 — Topic-Aware Short-Term Memory (§3.2)

Topic segments form an index `{topic, message turns}` where `message_turns = {user_i, model_i}`, placed into the STM buffer. **Summarization is triggered only when the buffer's token count reaches threshold `th`** (not per turn):

```
sum_i   = f_sum(S_i),   S_i ⊆ {user_i, model_i}, S_i ≠ ∅
Entry_i = { topic, e_i := embedding(sum_i), user_i, model_i }
```

Topic-constrained granularity is the sweet spot: feeding multiple raw sessions cuts API calls but mixes topics (degrading accuracy); per-turn input is accurate but expensive. Topic grouping minimizes API calls while preserving summarization accuracy.

### Light3 — Long-Term Memory with Sleep-Time Update (§3.3)

**Soft Updating at Test Time.** When entries arrive online, LightMem just **inserts** them into LTM with a timestamp — no LLM-driven merge/delete on the critical path. This decouples update from online inference, slashing interaction latency.

**Offline Parallel ("Sleep-Time") Update.** After all entries are inserted (or on an update trigger), each entry computes an independent update queue:

```
Q(e_i) = Top_k{ (e_j, sim(v_i, v_j)) | t_j ≥ t_i, j ≠ i }_{:n}
```

Only **later-timestamped** entries (`t_j ≥ t_i`) may update earlier ones, matching real temporal dynamics. Because queues are independent across entries, the `f_update` operations run **in parallel** offline — unlike baselines' strictly sequential updates whose latency accumulates. Crucially, LightMem does not merely *shift* update latency offline; the parallelism *reduces total* update latency.

**Why soft updates work.** LLMs are unreliable at real-time updates — given two related-but-noncontradictory facts they may wrongly flag a conflict and delete the older entry, causing irreversible loss. Soft, additive inserts preserve global information; consolidation happens later, reflectively. (Case study: "Tokyo trip" + "Kyoto inquiry" — a hard overwrite loses Tokyo context; LightMem's soft update keeps both.)

### Complexity (§4)

For a dialogue of N turns averaging T tokens each, with compression rate r, x compression iterations, and STM threshold th:

| | API Calls | Runtime | Summary Tokens |
|---|---|---|---|
| Baselines | N | O(N) | N·(L_sum-in + T + L_sum-out) |
| **LightMem** | N·(r^x·T / th) | O(N·r^x·T / th) | N·(r^x·T/th)·(L_sum-in + th + L_sum-out) |

Summarization fires only when the buffer fills, so call count drops by the factor `r^x·T/th`, and stricter retrieval (similarity + timestamp filtering, fraction R2 < R1) further cuts update calls.

---

## Experimental Results

**Setup.** Realistic **Incremental Dialogue Turn Feeding** — the full history is processed one turn at a time, as in real interaction. Pre-compressor: LLMLingua-2 throughout. Sensory buffer = 512 tokens. Backbones: **GPT-4o-mini**, **Qwen3-30B-A3B-Instruct-2507**, **GLM-4.6**. Datasets: **LongMemEval-S** and **LoCoMo**. Baselines: Full Text, Naive RAG, LangMem, A-MEM, MemoryOS, Mem0. Accuracy judged by GPT-4o-mini. Efficiency tracked over the *memory-bank-construction* stage (Summary + Update LLM calls); retrieval/usage held identical across methods for fairness.

### LongMemEval-S (GPT-4o-mini)

| Method | ACC (%) | Total Tokens (k) | Calls | Runtime (s) |
|---|---|---|---|---|
| FullText | 56.80 | 105.07 | — | — |
| NaiveRAG | 61.00 | — | — | — |
| LangMem | 37.20 | 1,102.16 | 867.38 | — |
| A-MEM | 62.60 | 1,605.81 | 5,132.06 | — |
| MemoryOS | 44.80 | 2,991.75 | 8,030.04 | — |
| Mem0 | 53.61 | 1,152.62 | 4,248.49 | — |
| **LightMem** (r=0.5, th=256) | **64.29** | 30.81 (online) / 47.02 (+offline) | 25.67 / 70.23 | 302.69 |
| **LightMem** (r=0.6, th=256) | **64.69** | 35.11 / 57.16 | 30.47 / 85.07 | 342.63 |
| **LightMem** (r=0.7, th=512) | **67.78** | 28.25 / 83.44 | 18.43 / 125.47 | 329.61 |

On LongMemEval, LightMem beats the strongest baseline (A-MEM) by **+2.09% to +6.40% ACC** with GPT, and up to **+7.67% with Qwen**.

**Efficiency (combined online + offline):**
- **GPT:** 10×–38× fewer total tokens; 3.6×–30× fewer API calls; 2.9×–12.4× faster runtime.
- **Qwen:** 6.9×–21.8× fewer tokens; 3.3×–17.1× fewer calls; 1.6×–6.3× faster.

**Efficiency (online test-time cost only) — the headline numbers:**
- **GPT:** 31.4×–**105.9×** fewer tokens; 17.1×–**159.4×** fewer API calls.
- **Qwen:** 30.1×–**117.1×** fewer tokens; 24.8×–**309.9×** fewer API calls.

### LoCoMo (GPT-4o-mini)

| Method | ACC (%) | Total Tokens (k) | Calls | Runtime (s) |
|---|---|---|---|---|
| FullText | 71.83 | — | — | — |
| NaiveRAG | 63.64 | — | — | — |
| LangMem | 57.20 | 1010.22 | 2229.37 | 2268.57 |
| A-MEM | 64.16 | 1149.43 | 6060.73 | 5543.90 |
| MemoryOS (regular) | 54.87 | 553.45 | 3332.59 | 1982.20 |
| Mem0 | 61.69 | 1693.39 | 4432.87 | 4540.70 |
| **LightMem** (0.7, 512) | **71.95** | 99.76 | 848.49 | 815.70 |
| **LightMem** (0.7, 768) | 70.26 | 41.65 | 737.80 | — |
| **LightMem** (0.8, 768) | **72.99** | 80.48 | 815.32 | 1079.40 |

On LoCoMo:
- **GPT:** +6.10% to +18.12% ACC; 2.87×–**20.92×** token efficiency; 13.29×–39.78× fewer API calls; 2.63×–8.21× faster.
- **Qwen:** +4.41% to **+29.29%** ACC; 3.33×–18.02× fewer tokens; 12.96×–**55.48×** fewer API calls; 1.18×–5.57× faster.

**Bottom line:** LightMem wins on nearly all metrics across both datasets and both backbones, *simultaneously* improving accuracy and cutting cost by 1–2 orders of magnitude.

---

## Ablation & Analysis

### Topic Segmentation (Light1) — Figure 3(c)

Removing topic segmentation slightly improves efficiency but **significantly hurts accuracy**: −6.3% ACC for GPT (68.6% → 64.3%), −5.4% for Qwen (73.2% → 69.2%), confirming the module lets the model perceive semantic units before constructing memory entries.

| | ACC | input (k) | output (k) | total (k) | calls |
|---|---|---|---|---|---|
| GPT, with topic seg | 68.6% | 18.9 | 9.4 | 28.2 | 18.4 |
| GPT, without | 64.3% (−6.3%) | 18.1 | 9.3 | 27.4 | 17.3 |
| Qwen, with topic seg | 73.2% | 13.2 | 19.2 | 32.4 | 20.0 |
| Qwen, without | 69.2% (−5.4%) | 13.5 | 18.3 | 31.9 | 19.2 |

The hybrid **attention + similarity** segmenter beats attention-only and similarity-only across all compression ratios, with >80% absolute segmentation accuracy.

### Compression Rate `r` and STM Threshold `th`

- **Pre-compression robustness:** at r ∈ [50%, 80%], compressed-vs-uncompressed QA accuracy is comparable — LLMs read compressed content fine. The submodule uses <2GB GPU and has negligible runtime impact.
- **Optimal `r` depends on `th`:** small buffers (th ∈ {0, 256}) prefer r = 0.6; large buffers (th ∈ {512, 1024}) prefer the higher-retention r = 0.7 (richer, less-compressed text leverages long-context ability and mitigates "lost in the middle"). Average optimal r ≈ 0.6. Lower r = fewer buffer triggers = fewer calls = lower cost.
- **`th` trade-off:** larger `th` consistently improves efficiency, but ACC is **non-monotonic** in `th` — a bigger buffer is not always better; the accuracy-optimal setting needs tuning per model/r.

### Sleep-Time Update (§5.6)

Soft + offline consolidation enhances long-term memory reliability and mitigates information loss vs. real-time hard updates (which risk deleting non-conflicting older memories). Only one extra model (LLMLingua-2) is introduced over baselines, and its latency is fully counted in the reported Runtime.

---

## Key Takeaways

1. **Filter early, consolidate late.** The central efficiency lever is moving cheap *token-level* filtering (LLMLingua-2, <2GB, no LLM call) to the front and pushing expensive LLM consolidation to an offline sleep phase — keeping the online path lightweight.

2. **Topic-aware buffering beats fixed windows.** Summarizing only when a topic-grouped buffer fills minimizes API calls *and* avoids the entangled-semantics problem that fixed-window/whole-session chunking creates.

3. **Soft + parallel offline updates dominate sequential real-time updates.** Decoupling update from inference removes test-time latency, prevents LLM mis-judged deletions, and lets independent update queues run in parallel — reducing *total* (not just online) update latency.

4. **Efficiency and accuracy are not a trade-off here.** LightMem improves ACC by up to +7.7% (LongMemEval) / +29.3% (LoCoMo) *while* cutting tokens up to 38× and API calls up to 30× (overall) — and up to 106× tokens / 159× calls online-only.

5. **Cheap, general components.** LLMLingua-2 is a drop-in BERT-scale compressor; the design needs only one extra model beyond baselines, making the efficiency gains broadly reproducible.

---

## Limitations & Future Work (per authors)

1. **Reliance on an external compressor.** Quality hinges on LLMLingua-2's compression fidelity; very aggressive `r` could drop critical tokens (accuracy degrades sharply below ~r=0.5 in Figure 3a).
2. **Tuning burden.** Optimal `(r, th)` is model- and dataset-dependent and non-monotonic in `th`, so it requires careful per-setting tuning.
3. **No explicit structured/multi-hop memory yet.** Current LTM is a flat, similarity-indexed entry store; the authors plan to add a lightweight **knowledge-graph memory** for explicit multi-hop reasoning.
4. **Update runtime still notable.** They plan to accelerate the update phase with offline pre-computed **KV caches**.
5. **Text-only.** A **multimodal** extension (visual/auditory/textual) is planned for embodied/real-world use.

---

## Where it sits (v1/v2)

LightMem is a **v2 (2026 frontier)** memory system whose defining contribution is *efficiency* rather than a new retrieval substrate — a direct response to the cost explosion of the v1 generation.

- **Sleep-time / offline consolidation is the shared v2 frontier.** LightMem's "soft insert online, consolidate offline in parallel" mirrors **MAGMA**'s dual-stream write path (fast non-blocking synaptic ingestion + slow asynchronous LLM consolidation), and instantiates the survey's "offline consolidation (sleep)" direction. Both decouple expensive relational/reflective processing from the latency-critical online path; LightMem's distinctive twist is the *parallel independent update queues* with timestamp-ordered soft updates.

- **Lightweight pre-compression vs. heavy LLM-per-memory pipelines.** This is LightMem's sharpest contrast with v1 foundational systems. **A-MEM** (Zettelkasten note generation + LLM linking per memory) and **MemoryOS** (multi-tier OS-style management) run a strong LLM on essentially raw data for every unit — exactly the "redundant sensory memory" overhead LightMem targets. In the experiments these baselines consume **millions of tokens and thousands of API calls** (e.g., MemoryOS at ~8M tokens / 8K calls on LongMemEval), while LightMem matches or beats their accuracy at tens of thousands of tokens. The insight: a BERT-scale token filter in front of the pipeline removes most of the cost that v1 systems paid an LLM to wade through.

- **Complementary, not competing, with structured-memory v2 work.** LightMem deliberately keeps LTM flat and similarity-indexed (its limitation), whereas MAGMA invests in disentangled temporal/causal/semantic/entity graphs for reasoning. The natural synthesis — which LightMem's own roadmap (lightweight KG memory) points toward — is a v2 system that pairs *cheap front-end filtering + sleep-time consolidation* with *structured multi-relational retrieval*.
