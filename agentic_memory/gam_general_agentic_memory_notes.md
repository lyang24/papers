# GAM: General Agentic Memory Via Deep Research

**Authors:** B.Y. Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, Zheng Liu (Beijing Academy of Artificial Intelligence; Renmin University of China; Peking University; Hong Kong Polytechnic University)

**Paper:** arXiv:2511.18423v1 (Nov 2025)

**GitHub:** https://github.com/VectorSpaceLab/general-agentic-memory

---

## The Core Problem

Most existing memory systems follow the principle of **Ahead-of-Time (AOT) Compilation**: substantial computation is performed *offline* to compress raw history into a lightweight static memory, and incoming requests are then served primarily from this pre-constructed memory (MemoryBank, A-Mem, Mem0, MemoryOS, LightMem all fit this mold). The paper argues this AOT-style paradigm has three critical, structural limitations:

1. **Memorization is data compression, so it is inevitably subject to information loss.** The precomputed memory is a compressed representation of raw data; it cannot satisfy the fine-grained information needs that clients request at runtime. Once a detail is summarized away, it is gone.
2. **Static structure cannot adapt to ad-hoc requests.** A memory built in advance assumes a fixed organization, which prevents it from flexibly handling unforeseen requests that demand nuanced interpretation and integration.
3. **Reliance on domain expertise / handcrafted heuristics.** AOT systems often depend on hand-designed rules for how memory is constructed and organized, which constrains generalization across domains and tasks.

The paper's framing inverts the usual relationship between search and memory:

> "Search is made as the core of memory, while memorization is conducted to enable effective search."

The claim is that **lossless memory can only be realized by searching over a database that holds the complete history** — the pre-computed memory exists only to *support* that search, not to replace it. The motivating epigraph is Einstein's: *"Intelligence is not the ability to store information, but to know where to find it."*

The paper also connects the under-performance of brute-force long-context LLMs to the recently-discussed **"context rot"** phenomenon: even with a 128K window large enough to cover the whole input, distracting/irrelevant text in long contexts severely degrades LLM performance. Simply extending the context window is not enough.

---

## The Big Idea: Just-in-Time (JIT) Compilation

GAM follows the principle of **Just-in-Time (JIT) Compilation** instead of AOT:

- **Offline stage** keeps only a *simple but useful* light memory, plus a **complete, lossless copy of the raw history** in a page-store. Memorization here is deliberately lightweight.
- **Online stage (runtime)** performs *intensive* computation — **deep research** — to construct a customized, high-utility context for each specific request, using the pre-built light memory as a guide into the page-store.

The formal objective (Definition 2.1) frames memory as a **min–max optimization**: produce the optimized context `c*` of *minimum size* that *maximizes* downstream task-completion performance:

```
c* = argmin_C* |c|,  where  C* = argmax_C Agent(task, context)
```

The history is defined as a sequence of temporally ordered **sessions**: `hist : s1, ..., sT`.

This design gives GAM three claimed advantages:
1. **High-fidelity & task-adaptability** — concise yet highly informative context tailored to the downstream task.
2. **Domain generalizability** — works across general scenarios without domain-specific heuristics.
3. **Optimizability** — harnesses frontier LLMs' agentic capability and test-time scalability, and can be further improved end-to-end via reinforcement learning.

---

## Architecture: A Dual-Agent (Memorizer + Researcher) Framework

GAM is realized as a **duo-design** of two LLM-based agents, each with customized prompts, that cooperate to produce the optimized context.

### 1. Memorizer (offline)

The memorizer processes the agent's streaming trajectory one session at a time. Each new session `s_i` triggers two operations:

**Memorizing** — produces a **memo** `µ_i`, a concise, well-structured snapshot of the new session, generated from *both* the new session and the existing memory `m_i` (so it highlights what's crucial relative to the whole trajectory). The memory is then incrementally updated:
```
Memorizer.memorize(s_i, m_i) → µ_i ;   m_i + {µ_i} → m_{i+1}
```

**Paging** — preserves the *complete* information of the trajectory. It generates a **header** `h_i` containing crucial contextual information from the preceding trajectory, decorates the session content with that header to form a **page** `p_i`, and appends it to the **page-store**:
```
Memorizer.page(s_i, m_i) → h_i ;   {header: h_i, content: s_i} → p_i ;   p.append(p_i)
```
This contextual-header design follows the same principle as **BGE landmark retrieval** and **Anthropic contextual retrieval** — preserving page-level semantic consistency so pages can be accurately retrieved later. Input is segmented into **2,048-token pages** for stream processing.

So the offline stage yields two artifacts: a **light memory** (the running set of memos) and a **lossless page-store** (the full history, contextualized).

### 2. Researcher (online / deep research)

The researcher answers each client request `r` by iteratively retrieving and integrating information from the page-store. The loop has three operations:

**Planning** — chain-of-thought reasoning over the current memory to analyze the underlying information need, then concrete search plans over the available toolkit `T`:
```
Researcher.plan(r, m_i, T) → {tool: t; parameter: ρ_t}_{t∈T}
```
The implementation offers **three search tools**:
- an **embedding model** for vector (semantic) search (BGE-M3 as default dense retriever),
- a **BM25 retriever** for keyword-based search,
- an **ID-based retriever** for direct page exploration (`page_index`).

**Searching** — executes each search action *in parallel*, retrieving relevant pages, then integrates the union of retrieved pages together with the previous integration result `I`:
```
For each t: t(ρ_t) → p_t ;   Researcher.integrate( ∪_{t∈T} p_t, I, r) → I
```

**Reflection** — judges (binary indicator `y`) whether `I` now fully satisfies `r`. If not, it analyzes what's missing and forms a new request `r'` to drive another round of deep research; if yes, it returns `I`:
```
Researcher.reflect(I, r) → y, r' ;   if y=No: Researcher(r', I) ;   if y=Yes: return I
```

The final output to the client is the integrated result, optionally accompanied by the original source pages. The prompts (in the appendix) define distinct sub-agents: **MemoryAgent** (memorizing), **PlanningAgent**, **IntegrateAgent**, **InfoCheckAgent** (the reflection "enough?" judge), and **FollowUpRequestAgent** (generates the next-round requests).

### 3. End-to-End Optimization (RL)

GAM supports a unified end-to-end optimization. Given training data `D = {(task, hist)}`, the memorizer builds `M, P`, the researcher generates candidate context `c`, the client samples an answer, and a reward `Γ(·)` scores it. The expected reward is:
```
R = E[ Γ(ans) ]   over task,hist ∼ D; M,P ∼ Memorizer; c ∼ Researcher; ans ∼ Client
```
The memorizer and researcher are trained via **reinforcement (policy gradient)** while the **client is excluded** from learning, with separate baselines `Γ̄_m`, `Γ̄_r` for each module.

---

## Experimental Setup

- **Benchmarks:** LoCoMo (conversational memory: single-hop, multi-hop, temporal-reasoning, open-domain), **HotpotQA** (multi-hop QA, MemAgent's curated set at three context lengths **56K / 224K / 448K** tokens by varying distractors), **RULER** at **128K** (retrieval, multi-hop tracing, aggregation, QA), and **NarrativeQA** (full book/script as context, ~87K avg tokens, 300-question subset).
- **Backbones:** GPT-4o-mini and Qwen2.5-14B-Instruct (both 128K context). Default dense retriever: BGE-M3.
- **GAM config:** max reflection depth = 3, max retrieved pages = 5, page size = 2,048 tokens.
- **Baselines:** memory-free (long-LLM brute force; RAG with 2,048-token segments, top-5) and memory-based (A-Mem, Mem0, MemoryOS, LightMem).

---

## Main Results

### LoCoMo (F1 / BLEU-1), GPT-4o-mini backbone

| Method | Single-Hop F1 | Multi-Hop F1 | Temporal F1 | Open-Domain F1 |
|---|---|---|---|---|
| Long-LLM | 46.05 | 26.38 | 28.96 | 14.89 |
| RAG | 47.87 | 30.78 | 38.19 | 14.16 |
| A-Mem | 33.75 | 22.09 | 32.24 | 13.49 |
| Mem0 | 42.58 | 27.19 | 25.45 | 15.03 |
| MemoryOS | 46.33 | 31.73 | 32.03 | 20.27 |
| LightMem | 34.92 | 28.96 | 42.96 | 15.81 |
| **GAM** | **58.93** | **38.19** | **51.52** | **30.63** |

(With the Qwen2.5-14B backbone GAM likewise leads: e.g. Single-Hop F1 **57.75**, Multi-Hop **48.93**, Temporal **59.45**, Open-Domain **33.30**.)

### HotpotQA (F1) / RULER-128K (Acc.) / NarrativeQA (Acc. & F1), GPT-4o-mini backbone

| Method | Hotpot-56K | Hotpot-224K | Hotpot-448K | RULER Retri. | RULER MT | RULER AGG. | NarrativeQA Acc. | NarrativeQA F1 |
|---|---|---|---|---|---|---|---|---|
| Long-LLM | 49.75 | 46.82 | 43.17 | 70.85 | 80.00 | 15.40 | 45.60 | 29.69 |
| RAG | 51.81 | 46.72 | 48.36 | 92.78 | 0.00 | 24.70 | 47.80 | 18.29 |
| A-Mem | 27.04 | 25.65 | 22.92 | 39.73 | 0.00 | 25.80 | 40.20 | 25.18 |
| Mem0 | 30.12 | 32.44 | 26.55 | 43.03 | 41.20 | 31.50 | 46.10 | 27.80 |
| MemoryOS | 24.58 | 30.25 | 23.13 | 54.58 | 3.00 | 5.20 | 34.60 | 23.45 |
| LightMem | 37.30 | 27.72 | 28.25 | 27.53 | 17.40 | 25.60 | 53.00 | 16.57 |
| **GAM** | **64.07** | **55.99** | **57.87** | **93.43** | **90.20** | **36.10** | **74.50** | **34.77** |

Key observations from the authors:
- **GAM wins on every benchmark**, against both memory-free and memory-based baselines, and the gap is *largest* on tasks needing multi-step retrieval/reasoning over dispersed information (HotpotQA, RULER).
- On **RULER multi-hop tracing (MT)** — tracking variable values across chained assignments — GAM exceeds **90% accuracy** while most baselines collapse (RAG, A-Mem, MemoryOS near 0).
- GAM stays **stable across context lengths** (56K → 448K HotpotQA), whereas long-LLM degrades. This is attributed to avoiding **context rot**.
- **RAG is high-variance**: good when relevant info is explicit (LoCoMo single-hop, RULER retrieval), poor when it's not (HotpotQA, RULER MT/AGG). Memory-based baselines have lower variance but lose crucial detail. GAM uses memory to *guide* search, getting the best of both.

---

## Analysis

### Model's Impact (which module needs the bigger LLM?)

Scaling the backbone from Qwen2.5-0.5B → 32B (and GPT-4o-mini) for memorizer vs. researcher independently (HotpotQA + NarrativeQA avg F1):

| Backbone | Memorizer-scaled (Avg F1) | Researcher-scaled (Avg F1) |
|---|---|---|
| Qwen2.5-0.5B | 48.83 | 9.08 |
| Qwen2.5-3B | 50.54 | 33.48 |
| Qwen2.5-7B | 51.53 | 43.85 |
| Qwen2.5-14B | 53.18 | 53.18 |
| Qwen2.5-32B | 53.50 | 54.50 |
| GPT-4o-mini | 54.05 | 55.45 |

**Key finding:** the **researcher is far more sensitive to model scale** than the memorizer. GAM stays strong even with a tiny 0.5B *memorizer* (48.83), but **collapses** when the *researcher* is shrunk to 7B or below (43.85 at 7B, 9.08 at 0.5B). The memorizer just extracts salient info (easy); the researcher must plan, search, and reflect iteratively (hard) — so capacity matters most there.

### Increasing Test-Time Computation

Two knobs, both yielding consistent gains (GAM benefits from test-time scaling, an advantage fixed-workflow baselines lack):
- **Reflection depth** swept 1→5 (default 3). GAM autonomously decides how many reflections to actually run; more depth collects more info but with **diminishing marginal gains** (many tasks don't need deep multi-step reasoning).
- **Retrieved pages** swept 3→20 (default 5). More pages per step → broader coverage → consistent improvement.

---

## Ablation Study (Detailed Factors)

### Search tools (HotpotQA F1 + NarrativeQA F1, Avg)

| Configuration | Hotpot-56K | Hotpot-224K | Hotpot-448K | NarrativeQA | Avg |
|---|---|---|---|---|---|
| Only Page-ID | 44.86 | 21.65 | 19.02 | 30.30 | 28.96 |
| Only Embedding | 39.59 | 32.71 | 26.67 | 30.25 | 32.31 |
| Only BM25 | 59.24 | 52.29 | 51.52 | 31.50 | 48.64 |
| Embedding + Page-ID | 47.25 | 34.78 | 28.43 | 33.41 | 35.97 |
| Embedding + BM25 | 61.37 | 55.00 | 54.90 | 33.20 | 51.12 |
| BM25 + Page-ID | 63.57 | 55.38 | 55.62 | 32.05 | 51.66 |
| **GAM (all 3 tools)** | **64.07** | **55.99** | **57.87** | **34.77** | **53.18** |

Any pair beats any single tool, and all three together is best — multiple tools give broader page-store coverage.

### Modules in isolation (the central ablation)

| Configuration | Hotpot-56K | Hotpot-224K | Hotpot-448K | NarrativeQA | Avg |
|---|---|---|---|---|---|
| Research **without** memory | 57.40 | 49.72 | 53.98 | 31.97 | 48.27 |
| Memory **without** research | 42.67 | 19.75 | 17.38 | 30.18 | 27.50 |
| **GAM (full)** | **64.07** | **55.99** | **57.87** | **34.77** | **53.18** |

- **Memory-without-research (27.50)** is by far the worst — confirming the paper's thesis that **pre-computed static memory is prone to severe information loss**. This is essentially the AOT paradigm, and it loses ~25 points vs. full GAM.
- **Research-without-memory (48.27)** also drops substantially — memory is crucial to *guide* effective search of the page-store.
- The two modules are complementary; neither alone reaches the full system.

### Output format (quality vs. token cost)

| Output Format | Hotpot avg F1 | NarrativeQA F1 | Avg F1 | Avg Tokens |
|---|---|---|---|---|
| Integration only (default) | ~59.3 | 34.77 | 53.18 | **105.90** |
| Integration + full Page | ~62.6 | 34.99 | **55.71** | 2379.82 |
| Integration + Extraction | ~61.0 | 34.82 | 54.47 | 230.76 |

The integration result alone is already highly competitive at only ~106 tokens; attaching source pages mitigates fine-grained detail loss for the best F1 (55.71) but at ~22× the token cost. "Integration + Extraction" is a good middle ground.

---

## Efficiency (HotpotQA, offline build + online serve, seconds)

| Method | 56K Total (s) | 56K F1 | 224K Total (s) | 224K F1 | 448K Total (s) | 448K F1 |
|---|---|---|---|---|---|---|
| A-Mem | 210.26 | 27.04 | 905.46 | 25.65 | 1797.29 | 22.92 |
| Mem0 | 37.57 | 30.12 | 165.47 | 32.44 | 275.05 | 26.55 |
| MemoryOS | 80.80 | 24.58 | 326.25 | 30.25 | 703.18 | 23.13 |
| LightMem | **5.13** | 37.30 | **16.86** | 27.72 | **40.78** | 28.25 |
| **GAM** | 69.32 | **64.07** | 269.37 | **55.99** | 575.65 | **57.87** |

GAM's online-serve time is notably higher than baselines (12–18s, since deep research happens at runtime), but total cost is comparable to Mem0/MemoryOS and far below A-Mem — while delivering **dramatically higher F1**. The authors frame this as the **best cost-effectiveness**: GAM deliberately moves heavy compute to query time (JIT), and that runtime cost buys large accuracy gains. Offline build grows roughly linearly with context length; online serve stays relatively flat.

---

## Key Takeaways

1. **Invert search and memory.** GAM's central thesis: memory should exist to *support search over a lossless history*, not to *replace* it. Keep the full history (page-store) and a light index (memos); do the hard work at query time.

2. **JIT beats AOT.** The module ablation makes the case empirically — "memory without research" (the AOT regime) scores 27.50 vs. full GAM's 53.18. Static, pre-compressed memory loses too much.

3. **The researcher is where intelligence lives.** Scaling experiments show the iterative plan→search→reflect researcher needs a capable LLM (collapses below 7B), while the memorizer can be tiny (0.5B still works). Memory construction is easy; runtime deep research is hard.

4. **Test-time scaling is a first-class lever.** More reflection depth and more retrieved pages both monotonically help — a property fixed-workflow memory systems structurally cannot offer.

5. **Multi-tool retrieval matters.** BM25 + embedding + page-ID together beat any subset; heterogeneous search modes cover heterogeneous information needs.

6. **End-to-end RL is built in.** Both agents are differentiable through a policy-gradient reward signal, so GAM can be continually optimized for downstream task reward.

---

## Limitations

1. **Higher online latency.** Deep research runs at query time, so online serving is much slower than static-memory baselines (12–18s vs. sub-second for Mem0/MemoryOS). The win is accuracy, not speed-per-query.
2. **Researcher capacity dependence.** Performance degrades sharply if the researcher's backbone is small (≤7B), so GAM effectively requires a strong frontier LLM at runtime.
3. **Page-store storage cost.** Keeping the complete, contextualized history losslessly trades storage and offline build time for fidelity; offline build grows linearly with history length.
4. **Evaluation scope.** Validated on QA-style memory and long-context benchmarks (LoCoMo, HotpotQA, RULER, NarrativeQA); generalization to multimodal or tool-heavy agentic trajectories is left as future work.

---

## Where it sits (v1/v2)

GAM is squarely a **v2 (2026-frontier) paper** and one of the clearest embodiments of the survey's **"retrieval → generation"** and **"just-in-time / runtime context construction"** shift. The defining move of v2 memory is that **memory is constructed on demand rather than precomputed** — and GAM makes that the explicit organizing principle (JIT vs. AOT).

- **Contrast with static-memory v1 approaches.** MemoryBank, A-Mem (Zettelkasten notes), Mem0, and MemoryOS all build a structured memory *upfront* and then serve queries from that compressed artifact — the AOT paradigm. GAM's ablation directly indicts this regime: "memory without research" (essentially the AOT setup) scores 27.50 vs. 53.18 for full GAM, quantifying the information loss that upfront compression incurs. GAM keeps memory deliberately *thin* and instead preserves a lossless page-store, doing the heavy lifting at query time.

- **MAGMA as a v2 sibling, but different axis.** MAGMA (also 2026) is v2 in that it makes retrieval *intent-aware* and reasoning-structured (four orthogonal graphs, adaptive traversal). But MAGMA still **builds rich structure ahead of time** (its slow-path consolidation densifies the graph offline). GAM takes the opposite bet: minimize offline structure, maximize *runtime* agentic computation. They represent two v2 strategies — "smarter precomputed structure" (MAGMA) vs. "lossless store + runtime deep research" (GAM).

- **ReadAgent lineage: "LLM reasons, then looks up."** GAM is a direct descendant of the ReadAgent pattern where the model first reasons over a compressed gist and then **looks up** the raw pages it needs. GAM's memorizer-produces-memos / researcher-plans-then-retrieves-pages loop is exactly this lineage scaled into a full agentic, reflection-driven, RL-optimizable system: the light memory is the "gist" used to decide *where to look*, and the page-store is the raw text it looks up — "knowing where to find it" rather than storing it.

In short, GAM operationalizes the v2 frontier thesis: **don't pre-bake the answer into memory; keep the evidence intact and let a capable agent do deep research to assemble the minimal optimal context at runtime.**
