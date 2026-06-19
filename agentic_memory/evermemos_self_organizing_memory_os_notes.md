# EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning

**Authors:** Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, Yafeng Deng (EverMind, Shanda Group)

**Paper:** arXiv:2601.02163v2 (Jan 2026)

**GitHub:** https://github.com/EverMind-AI/EverMemOS

---

## The Core Problem

LLMs are increasingly deployed as long-term interactive agents that must sustain coherent personas and user models over days, months, or years. But fixed context windows degrade on ultra-long inputs (the "Lost-in-the-Middle" phenomenon) and incur prohibitive cost. Existing memory systems try to externalize state, but **most treat memory as a flat collection of isolated records** and retrieve fragments by semantic similarity, recency, or heuristic scoring.

The authors argue that many failures stem **not from missing information but from poor integration**: fragmented episodic experiences are never consolidated into higher-level semantic structures. Without consolidation and abstraction, an agent may retrieve the relevant facts yet still fail to:

1. Detect conflicts (e.g., a stale preference vs. a newly introduced constraint)
2. Maintain a stable, evolving user model across interactions
3. Reason consistently over a long timeline

The motivating example (Figure 2): a user says they love IPAs (last month) but later mentions a dentist prescribed antibiotics for two weeks (last week). A fragment-based system recalls the IPA preference and recommends a craft beer. EverMemOS consolidates these episodes into a coherent state and recommends a non-alcoholic mocktail. The key limitation being addressed: **the absence of an explicit mechanism to transform fragmented episodic experience into coherent, stable knowledge structures that support long-horizon reasoning.**

---

## The Big Idea: An Engram-Inspired Memory Lifecycle

EverMemOS does **not** attempt to simulate biological memory at the neural level. Instead it borrows three *organizing principles* from cognitive neuroscience and translates them into a computational pipeline. It is explicitly grounded in three lines of memory research:

- **The engram lifecycle** (Josselyn et al., 2015) — discrete, stable memory traces are formed from experience → motivates **Episodic Trace Formation**.
- **Systems consolidation** (McGaugh, 2000) — transient episodes are gradually reorganized into stable long-term semantic structures → motivates **Semantic Consolidation**.
- **Reconstructive memory** (Schacter, 2008) — recall is an active *reconstruction* of context, not a static lookup → motivates **Reconstructive Recollection**.

The result is a three-phase lifecycle that shifts memory from *passive storage of records* to *structured organization of experience*.

---

## Memory Primitives

### MemCell — the atomic unit bridging raw data and semantics

A MemCell `c` is a tuple `c = (E, F, P, M)`:

| Field | Name | What it holds |
|---|---|---|
| **E** | Episode | A concise third-person narrative of the event, with resolved coreferences — the **semantic anchor** |
| **F** = {f₁..fₙ} | Atomic Facts | Discrete, verifiable statements derived from E, used for high-precision matching |
| **P** | Foresight | Forward-looking inferences / prospections (plans, temporary states) annotated with **validity intervals [t_start, t_end]** |
| **M** | Metadata | Contextual grounding — timestamps, source pointers |

This turns memory from a static record `(E, F)` into a **temporally grounded** representation that also encodes *Foresight* `(P)` — e.g., "[3 days] needs to prepare ID documents and clothing" vs. "[Long term] improving photography skills."

### MemScene — thematic cluster of MemCells

A scene-level structure that groups related MemCells into a coherent thematic unit (e.g., "Urban Travel Planning"). MemScenes are the unit of consolidation and the entry point for retrieval. The collection of all MemScenes / MemCells is the **MemBase**.

---

## Architecture: The Three Phases

### Phase I — Episodic Trace Formation

Transforms the unbounded interaction stream `D = {d₁..d_T}` into discrete, stable MemCells via a three-step pipeline:

1. **Contextual Segmentation** — A *Semantic Boundary Detector* processes interactions through a sliding window (via LLM prompting). On detecting a topic shift, accumulated turns are encapsulated as a raw episode history. (Boundary detection isn't perfect but proves robust downstream — see Table 3.)
2. **Narrative Synthesis** — The raw episode history is rewritten into a concise, third-person **Episode (E)** with resolved coreferences, establishing a stable semantic anchor and removing dialogue redundancy/ambiguity.
3. **Structural Derivation** — From E, the LLM extracts **Atomic Facts (F)** for precise matching and generates **Foresight (P)** signals with inferred validity intervals — distinguishing a temporary "flu" from a permanent "graduation." Bundled with metadata M → the final MemCell.

### Phase II — Semantic Consolidation

An **online** mechanism (no batch reprocessing) that organizes MemCells into higher-order structures:

- **Incremental Semantic Clustering** — When a new MemCell `c` arrives, the system embeds it and retrieves the nearest MemScene centroid. If similarity exceeds threshold τ, `c` is assimilated and the scene representation is incrementally updated; otherwise a new MemScene is instantiated. A **max-time-gap** constraint prevents clustering temporally distant MemCells together (e.g., if the closest-in-time MemCell already in a scene is farther than the gap, the new cell is not assimilated). Dataset-specific hyperparameters: τ = 0.70 / max gap 7 days on LoCoMo; τ = 0.50 / 30 days on LongMemEval.
- **Scene-Driven Profile Evolution** — When a MemCell is assimilated, EverMemOS refreshes a compact **User Profile** by prompting over *scene summaries* (not individual turns). The profile holds **explicit facts** (verifiable attributes incl. time-varying measurements) and **implicit traits** (preferences/habits), updated online with **recency-aware updates** for time-varying fields and **conflict tracking** when evidence is inconsistent. Prompting over scene summaries helps separate stable traits from temporary states.

### Phase III — Reconstructive Recollection

Retrieval is modeled as active reconstruction under a principle of **necessity and sufficiency** — composing only the grounded context required for a query rather than dumping all potentially relevant records.

1. **MemScene Selection** — Relevance between the query and all MemCells is computed by fusing dense + BM25 retrieval over their **Atomic Facts (F)** via Reciprocal Rank Fusion (RRF). Each MemScene is scored by the *maximum* relevance of its constituent MemCells; the top-N scenes are selected (default N=10).
2. **Episode + Foresight Filtering** — Within selected scenes, Episodes are pooled and re-ranked to a compact set (default K=10). **Foresight Filtering** then retains only time-valid Foresight whose interval satisfies `t_now ∈ [t_start, t_end]`, discarding expired ones.
3. **Agentic Verification & Query Rewriting** — An LLM-based verifier checks whether the retrieved context is *sufficient*. If not, a **query rewriting** step supplements retrieval (re-entering the loop); otherwise context is passed downstream. On LoCoMo (GPT-4.1-mini) this second round triggers for **31.0%** of questions.

**Two task modes** share this retrieval pipeline:
- **Memory-Augmented Reasoning** (default for quantitative benchmarks): uses retrieved Episodes as context.
- **Memory-Augmented Chat**: additionally injects the User Profile and time-valid Foresight (filtered by current time) — capabilities not covered by existing benchmarks, so shown via qualitative case studies.

**Implementation:** GPT-4.1-mini (or GPT-4o-mini where specified) for all reasoning/memory ops; Qwen3-Embedding-4B (dense) + BM25 fused via RRF; Qwen3-Reranker-4B for episode re-ranking.

---

## Experimental Results

Benchmarks: **LoCoMo** (1,540 questions over 10 ultra-long ~9K-token dialogues), **LongMemEval** (S-setting, ~115K tokens/conversation, 500 questions), plus a profile study on **PersonaMem-v2**. Evaluation uses an LLM-as-judge protocol following MemOS (GPT-4o-mini + two auxiliary judges, averaged, blind), validated against human annotation with **Cohen's κ > 0.89** (κ=0.891 LoCoMo, 0.978 LongMemEval; accuracy >98%).

### LoCoMo — GPT-4.1-mini backbone (accuracy %)

| Method | Avg Tokens | Single-Hop | Multi-Hop | Temporal | Open-Domain | **Overall** |
|---|---|---|---|---|---|---|
| MemoryOS | 5.5k | 67.30 | 59.34 | 42.26 | 59.03 | 60.11 |
| Mem0 | 1.0k | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 |
| MemU | 4.0k | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 |
| MemOS | 2.5k | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 |
| Zep | 1.4k | 90.84 | 81.91 | 77.26 | **75.00** | 85.22 |
| **EverMemOS** | 2.3k | **96.67** (↑6.4%) | **91.84** (↑12.1%) | **89.72** (↑16.1%) | **76.04** (↑1.4%) | **93.05** (↑9.2%) |

### LoCoMo — GPT-4o-mini backbone (accuracy %)

| Method | Avg Tokens | Single-Hop | Multi-Hop | Temporal | Open-Domain | **Overall** |
|---|---|---|---|---|---|---|
| MemoryOS | 5.2k | 62.43 | 56.50 | 37.18 | 40.28 | 54.70 |
| Mem0 | 1.0k | 66.71 | 58.16 | 55.45 | 40.62 | 61.00 |
| MemU | 4.0k | 72.77 | 62.41 | 33.96 | 46.88 | 61.15 |
| MemOS | 2.5k | 81.45 | 69.15 | 72.27 | 60.42 | 75.87 |
| Zep | 1.4k | 88.11 | 71.99 | 74.45 | **66.67** | 81.06 |
| **EverMemOS** | 2.5k | **91.08** (↑3.4%) | **86.17** (↑19.7%) | **81.93** (↑10.0%) | **66.67** (↑0.0%) | **86.76** (↑7.0%) |

(Parentheses = relative gain vs. the strongest baseline under the same backbone.)

### LongMemEval (accuracy %, GPT-4.1-mini; baselines from official MemOS leaderboard)

| Method | Token | SS-User | SS-Asst | SS-Pref | Multi-S | Know.Upd | Temp.Reas | **Overall** |
|---|---|---|---|---|---|---|---|---|
| MemU | 0.5k | 67.14 | 19.64 | 76.67 | 42.10 | 41.02 | 17.29 | 38.40 |
| Zep | 1.6k | 92.90 | 75.00 | 53.30 | 47.40 | 74.40 | 54.10 | 63.80 |
| Mem0 | 1.1k | 82.86 | 26.78 | **90.00** | 63.15 | 66.67 | 72.18 | 66.40 |
| MemOS | 1.4k | 95.71 | 67.86 | **96.67** | 70.67 | 74.26 | **77.44** | 77.80 |
| **EverMemOS** | 2.8k | **97.14** (↑1.5%) | **85.71** (↑14.3%) | 93.33 (↓3.5%) | **73.68** (↑4.3%) | **89.74** (↑20.6%) | 77.44 (↑0.0%) | **83.00** (↑6.7%) |

**Headline findings:**
- EverMemOS beats the strongest baseline overall on every benchmark — **+9.2% on LoCoMo** (vs. Zep, GPT-4.1-mini) and **+6.7% on LongMemEval** (vs. MemOS).
- The largest gains are on tasks that require **integrating dispersed evidence**: LoCoMo multi-hop (**+19.7%** on GPT-4o-mini), LoCoMo temporal (**+10.0% / +16.1%**), and LongMemEval knowledge-update (**+20.6%**) — directly validating MemScene-level consolidation.
- Gains are smaller where a flat store already suffices: open-domain (↑0.0–1.4%) and single-session preference (where it slightly *trails* MemOS/Mem0, ↓3.5%).
- Favorable accuracy–efficiency frontier (Figure 6): high accuracy at moderate retrieval budgets (2.3–2.8k tokens), because the agentic sufficiency check composes *necessary and sufficient* context instead of fixed-budget noise accumulation.

### PersonaMem-v2 profile study (32k; 9 scenarios; accuracy %)

| | Zep | Mem0 | MemU | MemoryOS | MemOS | **EverMemOS** |
|---|---|---|---|---|---|---|
| Has profile component | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| **Overall** | 43.40 | 43.85 | 38.70 | 40.05 | 50.72 | **53.25** |

EverMemOS wins overall, **+2.53 points over MemOS**. The internal ablation (Table 4) shows the consolidated User Profile contributes a large complementary signal: Ep.+Prof. **53.25** vs. Prof-only 48.30 vs. Ep-only 43.93 — **+9.32 points** of profile over episodes-only.

---

## Ablation Study

Degraded variants on LoCoMo / LongMemEval (overall accuracy %, Figure 4):

| Variant | LoCoMo | LongMemEval |
|---|---|---|
| **EverMemOS (full)** | **89.16** | **83.00** |
| w/o MemScene (flat retrieval over MemCells) | 81.82 | 79.60 |
| w/o MemCell (retrieval over raw dialogue) | 0.52* | 71.20 |
| w/o EverMemOS (no external memory, context only) | — | 5.00 |

*(The w/o-MemCell / w/o-external-memory rows collapse — long-horizon queries cannot be answered reliably within the context window alone.)*

Performance degrades **stepwise** as structure is removed:
1. Removing **MemScenes** eliminates scene-level organization → weakens cross-turn aggregation over related episodes.
2. Removing **MemCells** further drops the stable semantic units (episodes/facts) → forces retrieval onto raw dialogue matching.
3. Removing **external memory** entirely collapses long-horizon performance.

### Segmentation ablation (Table 3, w/o MemScene to isolate boundary quality)

| Segmentation | GPT-4.1-mini | Qwen3-4B |
|---|---|---|
| Fixed-Message-10 | 88.05 | 80.95 |
| Fixed-Token-512 | 87.55 | 80.67 |
| Fixed-Token-1024 | 84.52 | 75.19 |
| Session (Oracle) | 87.66 | 80.63 |
| **Semantic (EverMemOS)** | **89.16 / 89.78** | **83.07 / 82.73** |

Three findings: (i) semantic segmentation beats fixed heuristics (especially coarse token chunking); (ii) it even **beats ground-truth session boundaries** — sessions are not always the optimal retrieval unit; (iii) results are robust across segmentation backbones (≤0.7-point swing).

**MemBase scale (Table 5):** LoCoMo — 702 MemCells / 286 MemScenes (≈2.45 cells/scene); LongMemEval — 54,755 MemCells / 40,138 MemScenes.

---

## Case Studies (Memory-Augmented Chat, Figure 7)

- **Episode recall:** Asked "How did I get injury last time?", EverMemOS reconstructs the concrete past episode ("Grade-II sprain on your right ankle, during a badminton session, confirmed by medical diagnosis") rather than a generic explanation about overuse/warm-up.
- **Longitudinal profile:** Tracks waist 104→96 cm with stable weight across months and sets a trajectory-consistent goal ("focus on gradual continuation"), instead of a generic "aim for a healthy BMI."
- **Experience-grounded Foresight:** Recalls a frustrating Beijing trip (overcrowding, no advance ticket for the Forbidden City) and proactively advises advance reservations / off-peak visits for an upcoming Europe trip — anticipating a constraint the user never restated.

---

## Key Takeaways

1. **Memory as a lifecycle, not a store.** The central reframing is from passive record storage to an explicit form→consolidate→reconstruct lifecycle. The biggest wins are exactly on tasks needing *integration* of dispersed evidence (multi-hop, temporal, knowledge-update), which is what consolidation buys.
2. **Consolidation before retrieval is the differentiator.** By clustering related episodes into thematic MemScenes *before* a query arrives, the solver receives a complete narrative context, letting the LLM bridge dispersed evidence and resolve state conflicts that confuse fragment-based systems.
3. **Time-bounded Foresight is a genuinely novel primitive.** Forward-looking inferences with explicit validity intervals, filtered at recall time by `t_now`, let the system distinguish active plans/temporary states from expired ones — supporting proactive, safety-aware behavior most benchmarks don't even measure.
4. **Reconstruction with a sufficiency check beats fixed-budget retrieval.** The agentic verify-and-rewrite loop (triggered 31% of the time on LoCoMo) composes *necessary and sufficient* context, yielding the favorable accuracy–efficiency frontier rather than dumping context.
5. **Semantic segmentation > sessions.** Learned topic-shift boundaries outperform even ground-truth session partitions, suggesting "sessions" are a poor proxy for semantically coherent memory units.

---

## Limitations (Acknowledged by Authors)

1. **Text-only evaluation.** Though the MemCell/MemScene abstraction is claimed modality-agnostic, multimodal/embodied settings are out of scope and untested.
2. **LLM-mediated cost & latency.** Memory construction and retrieval introduce extra LLM calls (Phase I ≈9.4M tokens, Phase III ≈10.3M tokens over 1,540 LoCoMo questions), increasing latency vs. single-pass baselines; better end-to-end efficiency (caching/batching/async) is future work.
3. **Benchmark gaps.** Current benchmarks measure answer-level accuracy/recall and don't test conflict detection, profile stability, or ultra-long timelines — so Foresight/profile capabilities are only shown qualitatively, and the design's strongest claims aren't fully isolated by existing metrics.

---

## Where it sits (v1/v2)

EverMemOS is a **v2 "self-organizing memory OS"**: it is explicitly positioned as a unified, product-ready *Memory Operating System*, but its defining move is reorganizing memory into an emergent thematic structure (incremental MemScene clustering + online profile evolution) rather than a fixed storage hierarchy. It is best understood against the three other "OS"/self-organizing systems in this collection:

- **vs. MemoryOS (STM / MTM / LPM)** — MemoryOS borrows an OS *memory-management* metaphor: a fixed three-tier hierarchy (short-term → mid-term → long-term personal) with explicit paging/eviction between tiers. EverMemOS keeps the "OS" framing but **discards the fixed tier hierarchy** in favor of an organically growing scene graph; consolidation is driven by semantic similarity + time-gap clustering, not by capacity-based promotion/eviction. In the LoCoMo table MemoryOS is in fact the *weakest* baseline (60.11 GPT-4.1-mini), underscoring that flat hierarchical tiers without semantic consolidation lag behind scene-level abstraction.

- **vs. MemOS (MemCube: parametric / activation / plaintext)** — MemOS is the most "OS-like" of the three: it schedules across *heterogeneous memory substrates* (parametric weights, KV activations, plaintext) via a unified MemCube abstraction — an infrastructure-layer scheduler. EverMemOS operates **entirely in plaintext/semantic space** (no parametric or activation memory) and instead invests in the *cognitive lifecycle* of that plaintext memory. Notably MemOS is the strongest baseline on LongMemEval (77.80), and EverMemOS's MemScene consolidation pushes past it to 83.00 — gaining most on knowledge-update (+20.6%), i.e. exactly where conflict-resolving consolidation helps.

- **vs. Nemori (self-organizing, prediction-driven)** — Nemori is the closest sibling: also self-organizing and also cognitively inspired, but its consolidation is **prediction-driven** (memory updates triggered by prediction error / surprise). EverMemOS is instead **engram/consolidation-driven**: it forms discrete traces, then clusters and abstracts them into scenes. Both reject flat stores; the contrast is *what triggers reorganization* — surprise (Nemori) vs. thematic similarity + systems-consolidation (EverMemOS).

**Cognitive grounding is the through-line.** Where MAGMA leans on Complementary Learning Systems (fast/slow dual-stream writes) and Nemori on predictive coding, EverMemOS is grounded in the **engram lifecycle, systems consolidation, and reconstructive memory** triad. Its most distinctive contribution is taking *reconstructive recollection* seriously as an engineering principle — recall as active, sufficiency-checked reconstruction of "necessary and sufficient" context, rather than a static top-k lookup — paired with **time-bounded Foresight** as a first-class, expirable memory primitive. That combination is what lets it consolidate evolving experience and resolve conflicts where flat or purely-hierarchical OS designs retrieve the right fact yet still answer wrong.
