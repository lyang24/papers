# Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval

**Authors:** Yingyi Zhang, Junyi Li (equal contribution), Wenlin Zhang, Penyue Jia, Xianneng Li, Yichao Wang, Derong Xu, Yi Wen, Huifeng Guo, Yong Liu, Xiangyu Zhao (Dalian University of Technology, City University of Hong Kong, Huawei Technologies, University of Science and Technology of China)

**Paper:** arXiv:2603.09250v1 (cs.IR, 10 Mar 2026) — ICLR 2026 submission

**GitHub:** https://github.com/Applied-Machine-Learning-Lab/ICLR2026_RF-Mem

**Method name:** **RF-Mem** (Recollection–Familiarity Memory Retrieval)

---

## The Core Problem

Personalized LLMs depend on **memory retrieval** to inject a specific user's histories, preferences, and contextualized interactions into generation. The paper argues existing systems sit at two unsatisfying extremes:

1. **Full-context dumping** — feed the user's entire past memory into the prompt. Costly, unscalable, and falls off an "out-of-context cliff" as memory grows (at 1M tokens it literally exceeds the LLM window).
2. **One-shot similarity search** — a single dense top-K lookup. Fast but captures only surface matches; it misses deeper contextual cues (under-retrieval) or pulls in semantically-near-but-irrelevant noise.

The deeper diagnosis: current retrieval strategies are dominated by embedding-based **one-shot top-K** search, which corresponds to only *one half* of how human memory actually works. Two specific gaps:

- They **lack a Recollection path** — no mechanism to retrieve *evidence chains* for ambiguous queries, long-tail knowledge, or personalized reasoning.
- They **lack a path-switching mechanism** — every query is treated as homogeneous, regardless of confidence or task complexity, so the system either always under-retrieves or always pays the cost of deep retrieval.

---

## The Big Idea: Dual-Process Memory (Recollection vs. Familiarity)

RF-Mem grounds retrieval design in the **Recollection–Familiarity Dual-Process Theory** from cognitive science (Henson et al. 1999; Yonelinas et al. 2002; Rugg & Curran 2007; Yonelinas 2024). The theory posits human recognition is driven by two complementary mechanisms:

| Process | Cognitive character | Retrieval analogue |
|---|---|---|
| **Familiarity** | Fast, coarse, low-effort "feeling of knowing" — instant recognition without deliberate reasoning | One-shot dense top-K similarity search |
| **Recollection** | Slow, deliberate, effortful — chain-like reconstruction that recovers time/place/source-specific episodic detail | Multi-round, iterative evidence expansion |

The crucial insight the paper borrows: **humans regulate which process to use via a familiarity signal.** High confidence sustains reliance on Familiarity; decreasing familiarity and rising uncertainty *prompt the shift* to Recollection. RF-Mem therefore casts retrieval not as a one-shot operation but as a **dual-process controller** that adaptively alternates between fast recognition and slow reconstruction, gated by the system's own sense of familiarity.

This is the "fast–slow thinking" theme applied specifically to *personalization* memory retrieval, and the switch is **uncertainty-guided**.

---

## Architecture

RF-Mem is a five-stage pipeline: (1) user query input → (2) **RF-Mem retrieval module** (familiarity-uncertainty-guided selection between Familiarity and Recollection) → (3) extract memory text → (4) LLM answer generation. The whole method lives **entirely in embedding space** — only vector search and small-scale KMeans clustering, no extra LLM calls inside the retriever.

### 1. Familiarity Uncertainty-Driven Retrieval Selection (the gate)

A **probe retrieval** runs first. Given query embedding `x_t = φ(q)` and memory fragments `m_i` with embeddings `z_i = φ(m_i)`, it computes cosine similarities `s_i = ⟨x_t, z_i⟩` and takes a top-K candidate set. From these probe scores the gate derives **two signals**:

- **Mean similarity** `s̄ = (1/K) Σ s_i` — the familiarity strength.
- **Entropy** `H(p) = −Σ p_i log p_i`, where `p_i = softmax(λ(s_i − max_j s_j))` is a temperature-sharpened (λ) normalization of the scores — the *uncertainty* of where the evidence concentrates.

The switching policy uses thresholds `θ_high`, `θ_low` and an entropy gate `τ`:

```
Strategy(q) = Familiarity,   if s̄ ≥ θ_high
              Recollection,  if s̄ ≤ θ_low
              (in between θ_low < s̄ < θ_high):
                  Familiarity   if H(p) ≤ τ   (concentrated evidence → confident)
                  Recollection  if H(p) > τ   (diffuse evidence → uncertain)
```

So mean score handles the confident extremes; **entropy disambiguates the gray zone**. This mirrors the cognitive story: strong signal → fast recognition, weak signal → deliberate reconstruction, ambiguous → use uncertainty as the tie-breaker.

### 2. Familiarity Retrieval (fast path)

When the gate fires Familiarity, RF-Mem simply returns the top-K memory fragments by raw cosine similarity in a single pass — `C_t = Top-K({(m_i, ⟨x_t, z_i⟩)})`. Minimal latency, no expansion. This *is* the standard dense-retrieval baseline, used as the default.

### 3. Recollection Retrieval (slow path) — Retrieve-Cluster-Mix loop

When the gate fires Recollection, RF-Mem runs **multi-round, tree-like evidence expansion**, bounded by beam width `B`, fanout `F`, and max rounds `R`. Each round `r`:

1. **Candidate Memory Retrieval** — retrieve top-N for the current query, where `N = (B + r) × F < K`. N grows with the round so reformulated queries don't keep re-retrieving the same memories; duplicates from earlier rounds are excluded.
2. **Relevant Memory Clustering** — group the N candidate embeddings into **B clusters via KMeans**. Each cluster centroid `g_b` is the mean of its members and becomes a *branch* in the retrieval tree — an anchor that captures essential cues while reducing redundancy.
3. **Recollect-Query Generation via α-mix** — each centroid is blended with the running query, with a **residual** of the original query `x_t` preserved:

   ```
   x_b^(r+1) = norm( α·x^(r) + (1−α)·g_b + x_t )
   ```

   α weights the current query vs. the centroid; adding `x_t` back keeps the reconstruction grounded in the original intent (prevents drift).
4. **Loop** — the new recollect-queries drive the next round of retrieval. At most B active branches are kept per round; recursion is capped at R.

**Stop & generation:** terminates when round limit R is hit or a target item count is gathered. Final evidence = `Top-K(∪_{r=0}^{R} C^(r))` — a truncated union over all rounds. The metaphor: cue-driven deliberate reconstruction that progressively surfaces latent, temporally-dispersed context, trading a little extra latency for more diagnostic evidence.

### Default hyperparameters (from Appendix B)

- PersonaMem (turn-level corpus): `λ = 20, B = 3, F = 2, θ_high = 0.6, θ_low = 0.3`; retriever `multi-qa-MiniLM-L6-cos-v1`, single A100 GPU.
- PersonaBench / LongMemEval: `θ_high = 0.6, θ_low = 0.0` (purely entropy-driven below high); `B = 3` with `F = 1` for Recall@5 and `F = 2` for Recall@10; `λ = 30` (PersonaBench) / `λ = 20` (LongMemEval).
- Sensitivity analyses find moderate settings win: **small beam and low fanout (B = 2–3, F = 1–2)** are best; large fanout dilutes/over-expands. The α-mix is best in the **α ≈ 0.3–0.5** range — extreme α = 0 or α = 1 both reduce recall (need both centroid evidence and original-query grounding).

---

## Experimental Results

Three personalized-memory benchmarks. All comparisons are **retrieval-only** (every method operates over the same memory vectors) for fairness. Baselines: **Zero Memory** (no memory), **Full Context** (entire history in prompt), **Dense Retrieval** (= the Familiarity path / one-shot top-K), **Recollection** (RF-Mem's slow path always on).

### PersonaMem — Personalized Generation Accuracy (across corpus scales)

PersonaMem (Jiang et al. 2025): simulated user–LLM histories over 7 real-world tasks, evolving personas, memory lengths of 32K / 128K / 1M tokens. Metric = answer Accuracy.

| Method | 32K Overall | 128K Overall | 1M Overall | Retri. Time (32K) | Avg. Tokens (32K) |
|---|---|---|---|---|---|
| Zero Memory | 0.3854 | 0.3124 | 0.2730 | NA | 464.6 |
| Full Context | 0.6129 | 0.3231 | **OOC** | NA | 24657.8 |
| Dense Retrieval (Familiarity) | 0.5908 | 0.5259 | 0.4518 | 3.14ms | 3515.9 |
| Recollection (ours) | 0.6214 | 0.5288 | 0.4544 | 7.09ms | 3711.1 |
| **RF-Mem (ours)** | **0.6350\*** | **0.5394\*** | **0.4589\*** | 5.09ms | 3566.6 |

(\* = statistically significant over best baseline, two-sided t-test p < 0.05.)

Key findings:
- **Best overall accuracy at every scale**, with compact inputs. At 32K it beats Full Context by **+0.0221** using only ~3.6K tokens vs. 24.7K.
- **Full Context collapses as memory grows** — 0.6129 → 0.3231 (128K) → out-of-context (1M) — while RF-Mem stays stable and leads Dense Retrieval by **+0.0135 (128K)** and **+0.0071 (1M)**.
- **Efficiency–accuracy sweet spot:** RF-Mem is *both* more accurate and faster than always-on Recollection — 5.09ms vs 7.09ms (32K), 4.27ms vs 7.86ms (128K), 6.28ms vs 8.12ms (1M) — because it only pays for recollection when the question is actually unfamiliar.
- Per-task wins at 32K: Aligned Recommendations (0.7818), New Scenarios (0.6140), Shared Facts (0.5659).

### PersonaBench — Personalized Retrieval (Recall@K, multi-backbone)

PersonaBench (Tan et al. 2025a): synthetic private user documents/queries probing personal info; tests retrieval *before* generation. Metric = Recall@K. Shown: MiniLM backbone (also evaluated on MPNet and BGE).

| Method | R@5 Overall | R@10 Overall | R@10 Pref-Hard | Time (R@10) |
|---|---|---|---|---|
| Familiarity | 0.4484 | 0.5964 | 0.5561 | 13.68ms |
| Recollection | 0.4491 | 0.6062 | **0.6267** | 17.29ms |
| **RF-Mem** | **0.4701** | **0.6071** | 0.6267 | 15.22ms |

- RF-Mem matches or beats the best single-mode strategy on overall Recall while keeping latency near Familiarity (9–15ms) rather than near Recollection (15–20ms).
- The Familiarity vs. Recollection split confirms **complementary strengths**: Familiarity wins fact-centric queries (Basic Info, Preference-Easy); **Recollection wins context-heavy queries** (Preference-Hard 0.6267 vs 0.5561, Social). Neither mode alone is robust across all task types — which is exactly the motivation for adaptive switching.

### LongMemEval — Long-term Personalized Retrieval (Recall@K)

LongMemEval (Wu et al. 2025a): long-term factual retrieval, small (S) and medium (M) settings. BGE backbone shown.

| Method | LME-S R@5 | LME-S R@10 | LME-S R@50 | LME-M R@5 | Time (S) |
|---|---|---|---|---|---|
| Familiarity | 0.7924 | 0.8926 | 1.0000 | 0.4964 | 29.65ms |
| Recollection | 0.8162 | 0.9165 | 1.0000 | 0.5131 | 43.65ms |
| **RF-Mem** | **0.8186** | **0.9189** | 1.0000 | **0.5155** | 37.34ms |

- RF-Mem is consistently strongest on Recall@5/@10 across MiniLM, MPNet, and BGE backbones, while sitting between the two modes on latency (37–50ms vs Familiarity 25–31ms and Recollection 40–62ms).
- Recollection lifts Recall@5 by **>0.02** over Familiarity across retrievers (e.g., 0.7351 vs 0.7136 on MiniLM) — evidence that iterative expansion uncovers temporally-dispersed cues that one-shot search misses.

---

## Adaptive / Modularity Studies (a major contribution)

RF-Mem is positioned as a *retrieval layer* that **complements rather than replaces** existing memory machinery. Three integration studies:

| Integration | Setup | Result |
|---|---|---|
| **Index building (MemoryBank)** | Offline MemoryBank summary index + RF-Mem as online retriever (PersonaMem 32K) | RF-Mem gets the **highest overall accuracy** under both turn-level (0.4419) and summary index, and *narrows the drop* under summarization vs. single-path baselines |
| **Query expansion (HyDE)** | Nearline HyDE pseudo-doc expansion → RF-Mem online (PersonaBench) | RF-Mem matches/surpasses Familiarity across all categories (e.g., R@10 Overall 0.5194 vs 0.5120), so the dual-process gate survives upstream query reformulation |
| **Iterative RAG (Search-o1)** | Multi-turn reasoning loop with RF-Mem as the retrieval layer (PersonaMem 32K) | RF-Mem keeps the **highest overall** (0.5349 vs Familiarity 0.5271), retaining effectiveness inside iterative RAG |

Takeaway: RF-Mem is an orthogonal, modular **online retrieval controller** that layers on top of heterogeneous indexing (text summaries, graphs), query reformulation, and iterative pipelines.

---

## Ablation / Sensitivity Highlights

- **Each mode alone is insufficient** — the headline ablation *is* the Familiarity vs. Recollection vs. RF-Mem comparison throughout Tables 1–3: Familiarity is fast but shallow, Recollection is deep but ~2× slower and not always better, and only the adaptive combination is robust *and* efficient.
- **Entropy gating matters** — moderate entropy gating outperforms defaulting prematurely to Familiarity (extreme thresholds degrade results).
- **α-mix residual** — best around α = 0.3–0.5; both α = 0 (centroid only) and α = 1 (query only) hurt, validating the residual-grounded blend.
- **Beam/fanout** — small B (2–3) and low F (1–2) are optimal; high fanout over-expands and adds redundant noise.
- **Stable operating range** — across 32K→1M the entropy stays tightly banded (~0.17–0.18 median) and mean score median (~0.50–0.55) is largely scale-invariant, which is why fixed thresholds transfer across corpus sizes.
- **Recollection variant (Appendix)** — a graph-guided BFS recollection (instead of KMeans clustering) is weaker by default but becomes competitive under full-exploration + global re-ranking (Graph+bfs Full R@10 0.5986), suggesting structure-aware reconstruction is a viable alternative formulation.

---

## Key Takeaways

1. **Retrieval as a dual-process controller, not a one-shot lookup.** The core conceptual move is reframing personalized memory retrieval through Recollection–Familiarity dual-process theory and making the *mode itself* a runtime decision.

2. **The gate is uncertainty-driven, and cheap.** Mean similarity handles confident extremes; entropy over the probe-score softmax disambiguates the middle. The whole controller is embedding-space-only (vector search + small KMeans) — no extra LLM calls — so it adds negligible latency.

3. **Recollection = clustering + α-mix + residual.** Chain-like evidence reconstruction is achieved purely by iteratively retrieving, KMeans-clustering candidates into branch centroids, and α-mixing those centroids back with the original query — bounded by beam/fanout/rounds for tractability.

4. **Best of both worlds empirically.** RF-Mem beats both one-shot dense retrieval and full-context across generation (PersonaMem) and retrieval (PersonaBench, LongMemEval) benchmarks, scales to 1M-token corpora where Full Context goes out-of-context, and lands between the two modes on latency while leading on accuracy.

5. **Modular by design.** It's an online retrieval layer that composes with MemoryBank-style indexing, HyDE query expansion, and Search-o1 iterative RAG — complementing existing memory systems rather than competing with them.

---

## Limitations

1. **Threshold tuning is corpus/retriever-dependent.** θ_high, θ_low, τ, λ, B, F, α all need setting; though the paper shows the operating range is stable across scales for a fixed dataset, thresholds differ across benchmarks (e.g., θ_low = 0.3 for PersonaMem vs 0.0 for PersonaBench/LongMemEval) and backbones, so deployment requires calibration.
2. **Recollection still trades latency for coverage.** When the gate routes to Recollection, retrieval cost roughly doubles vs. Familiarity; the savings depend on a query distribution where many questions are "familiar."
3. **Retrieval-only evaluation.** By design the experiments isolate the retrieval component (same vectors for all methods); end-to-end gains depend on the generator and indexing stack, which the paper deliberately holds fixed rather than co-optimizes.
4. **Mostly simulated/synthetic data.** All three benchmarks are simulated or synthetic (no real PII). The authors flag ethical risks — over-amplification of user traits, behavioral-bias reinforcement, and profiling — for any real-world personalization deployment.
5. **KMeans cluster count = beam width.** Recollection branching is fixed to B KMeans clusters per round; the alternative graph-guided recollection is weaker by default and only catches up with a larger exploration budget.

---

## Where it sits (v1/v2)

RF-Mem is a **v2 (2026 frontier)** entry, and specifically a *personalization-focused* one. Whereas most of the v2 frontier (MAGMA, the long-context memory systems) targets **agentic long-term conversational memory** — temporal/causal reasoning over a single evolving session history — RF-Mem targets the **personalization** slice of memory: surfacing a *specific user's* preferences and persona to tailor generation.

Its central idea — a **cognitive dual-process (Recollection vs. Familiarity) grounding** with an uncertainty-gated switch — is a clean instance of the field's emerging **"fast–slow thinking" retrieval theme** that the survey (*A Survey of Personalization: From RAG to Agent*, by overlapping authors) frames. It directly parallels **MAGMA's intent-aware routing**: both reject one-size-fits-all similarity search and instead *route* retrieval based on a query-level signal. The difference in mechanism is instructive:
- **MAGMA** routes on *query intent* (WHY/WHEN/ENTITY) through orthogonal relational graph layers — a *structural* router.
- **RF-Mem** routes on *familiarity/uncertainty* (mean score + entropy) between a fast and a slow retrieval *process* — a *confidence* router living purely in embedding space.

Against earlier **v1 personalization** approaches, the contrast is about *representation vs. process*:
- **MemoryBank** builds static **user portraits / personality summaries** offline and queries them with a standard dense retriever — a *fixed, pre-computed* persona index.
- **MemoryOS** maintains a **90-dimensional persona** vector / structured user profile as the personalization substrate.

Both of those are **index-construction** advances: they decide *what to store* about the user, then retrieve it with one-shot similarity. RF-Mem is explicitly **complementary** — it leaves the index alone (and even demonstrates layering directly on top of MemoryBank's summary index) and instead innovates on the **retrieval strategy**: *how* to retrieve, adaptively deepening into deliberate recollection when the user's memory doesn't yield a confident match. In the v1→v2 arc, this marks the shift from *better user representations* (portraits, persona dimensions) to *better, cognitively-grounded, adaptive retrieval dynamics* over whatever representation exists.
