# Nemori: Adaptive Memory Distillation for LLM Agents

**Authors:** Wenquan Ma, Jiayan Nan, Wenlong Wu, Yize Chen (Fudan University, Shanda Group, Beihang University, Shanghai University of Finance and Economics)

**Paper:** arXiv:2508.03341v4 (Apr 2026). Current title: *"What Deserves Memory: Adaptive Memory Distillation for LLM Agents."* The system/codebase is named **NEMORI** throughout the paper.

**GitHub:** https://github.com/nemori-ai/nemori

---

## The Core Problem

Memory systems for LLM agents must decide **what information deserves retention** before storing it. The paper splits memory construction into two stages: **distillation** (what to retain / the entry form of an experience) and **management** (how to organize and maintain it). Existing distillation-time methods pre-position information by **encoding designer intuition** — importance scores (Generative Agents), emotional tags (Emotional RAG), or factual templates (Mem0). These heuristics have two failure modes:

1. **Subjective bias → irreversible information distortion.** A wrong importance/emotion/template judgment at distillation time permanently discards useful experience.
2. **Systemic bloat → retrieval noise.** To avoid distortion, systems over-store, amplifying noise at retrieval time.

The alternative — management-time methods (MemoryOS, Zep, A-MEM) — treat entries as opaque containers and infer utility post-hoc from structural metadata (access frequency, temporal decay, explicit relationships), never inspecting the nuanced content itself.

The paper's claim: utility should be **assessed from the interaction experience itself**, in a data-driven way, rather than from hand-written heuristics. The slogan — *"being agentic need not imply being heuristic."*

---

## The Big Idea: Predictability ⇒ Redundancy (Distillation via Prediction Error)

Nemori is a **training-free** framework inspired by **Predictive Coding Theory** (Rao & Ballard 1999; Friston's Free Energy Principle 2010; Clark 2013) — the idea that the brain sends predictions downward and propagates only the residual **prediction error** upward, i.e. brains are prediction machines. Nemori adapts this directly:

> **Prediction error signals information worth retaining; what is predictable is therefore redundant.**

Rather than scoring experiences with a heuristic, Nemori synthesizes an **anticipatory schema** of what an incoming episode probably contains (given existing knowledge) and distills as memory **only what the agent failed to predict**. This casts "future utility" as a matter of predictability — a data-driven signal instead of designer intuition.

The framework is built on **three parsimonious priors** (inductive biases over interaction sequences):

| Prior | Statement | Design requirement |
|---|---|---|
| **Structure Prior** | *Integrity of Episode* — interactions naturally group; messages mutually contextualize each other | Define episodes respecting latent integrity, not heuristic fixed-size chunking |
| **Representation Prior** | *Asymmetry of Perspective* — raw episodes are egocentric/noisy; recall is allocentric reasoning | Transform raw episodes into **narrative** representations that surface logical structure |
| **Distillation Prior** | *Predictability Implies Redundancy* — interaction sequences are highly redundant | Distill via the **semantic differential** between actual interactions and their anticipatory schema |

A second cognitive grounding is **Complementary Learning Systems** (McClelland et al. 1995): the two cascading modules echo the hippocampus (fast episodic) / neocortex (slow semantic) division.

---

## Architecture

Nemori is **two cascading modules** plus an (optional, swappable) management layer. The headline design choice: it centers on **distillation** and is deliberately **management-agnostic** — it defines generic `Evoke(·)` and `Consolidate(·)` interfaces so it can run on its own native management, on flat summarization, on naive RAG, or as a distillation kernel injected into third-party systems (A-MEM, MemoryOS).

### Module 1 — Episodic Memory Integration (§3.2)

Transforms raw interactions into coherent narrative episodes. Three submodules:

**1. Local Message Partitioning** (Structure Prior). Messages accumulate in a buffer `B`; once it reaches an **observation window length `w`** (default 20), an LLM call partitions the window into pairwise-disjoint coherent episodes (segmenting on topic shifts / intent transitions) rather than imposing fixed-size chunks. Buffer resets after partitioning.

**2. Narrative Episode Generation** (Representation Prior). Each raw episode `Pj` is rewritten by an LLM into a **narrative** `Nj` plus a short **episodic cue** `cj`, then embedded (`vj = femb(cj ∥ Nj)`). Each episodic memory stores `Mj = (cj, Nj, Pj, vj)`. Two consequences:
- **Dual-mode retrieval:** return the narrative `N` for efficiency, or the raw `P` for precision-critical answering.
- The **episode** (not the message) becomes the primary processing unit downstream — avoiding the message-wise processing trap that makes many baselines expensive.

**3. Associative Memory Integration.** Episodes split across window boundaries are re-stitched. For a new episode, retrieve top-`Ke` similar episodes; an LLM picks the one sharing **episodic continuity** (or `-1` for none). If a match is found, the two are **merged** into one superseding entry; otherwise the episode is inserted as distinct.

### Module 2 — Semantic Knowledge Distillation (§3.3)

The prediction-error engine. Three submodules, all routed through the abstract management interface `M`:

**1. Anticipatory Schema Synthesis.** `Evoke(Min, M)` pulls the context `Sin` the system already knows about the incoming episode (native impl: threshold-filtered similarity search, `sim > τ`). An LLM then **predicts what the episode contains** from the cue `cin` + evoked context `Sin` alone → anticipatory schema `P̂in`. This is the system's "guess."

**2. Prediction Error Distillation.** An LLM compares the **actual raw episode `Pin`** against the **predicted schema `P̂in`** and extracts only the **deviations/extensions** as semantic insights `Kin`. This is the "what deserves memory" decision, operationalized — predictable content is dropped as redundant.

**3. Agnostic Knowledge Consolidation.** `Consolidate(Kin, M)` writes insights into management. Native impl retrieves associative knowledge and issues a directive `δ ∈ {new, merge, conflict}`: **new** inserts a distinct entry; **merge** unifies with complementary existing knowledge; **conflict** purges outdated entries and replaces them (knowledge update).

### Response Generation (§3.4)

Inference is orthogonal to construction. For a query `Q`, Nemori retrieves **in parallel** top-`k` episodic entries and top-`m` semantic entries (`m = 2k`; default `k=10`), then concatenates narrative episodes + the raw text of the top-`r=2` episodes + semantic knowledge as context for the answer LLM. This **episodic + semantic complementarity** is central (see ablation).

---

## Experimental Results

**Setup.** Two benchmarks: **LoCoMo** (10 dialogues, ~24K avg tokens, 1,540 questions, 4 reasoning categories) and **LongMemEval_S** (500 conversations, ~105K avg tokens — an order of magnitude longer). Backbones: **gpt-4o-mini** and **gpt-4.1-mini** as both internal and answer models; embeddings `text-embedding-3-small`. Metric is primarily **LLM-judge score** (0–100), plus F1 and BLEU-1. Baselines: Full Context, RAG-4096, LangMem, Zep, Mem0, A-MEM, MemoryOS.

### LoCoMo — overall LLM-judge (average across categories)

| Method | gpt-4o-mini (Avg) | gpt-4.1-mini (Avg) |
|---|---|---|
| Full Context | 72.3 | 80.6 |
| RAG-4096 | 30.2 | 32.9 |
| LangMem | 51.3 | 73.4 |
| Zep | 58.5 | 61.6 |
| Mem0 | 61.3 | 66.3 |
| A-MEM | 52.5 | 61.4 |
| MemoryOS | 54.5 | 60.6 |
| **NEMORI** | **73.0** (+19.1% over Mem0) | **80.8** (+10.1% over LangMem) |

Nemori achieves the strongest average among memory systems and slightly **exceeds Full Context** on both backbones (73.0 vs 72.3; 80.8 vs 80.6) — evidence that its distillation captures the useful signal without context dilution.

### LoCoMo — per-category LLM-judge (gpt-4o-mini)

| Method | Temporal | Open-Domain | Multi-Hop | Single-Hop |
|---|---|---|---|---|
| Full Context | 56.2 | 48.6 | 66.8 | 83.0 |
| Mem0 | 50.4 | 40.6 | 60.3 | 68.1 |
| A-MEM | 54.2 | 22.9 | 43.6 | 58.2 |
| MemoryOS | 38.0 | 45.8 | 52.5 | 62.5 |
| **NEMORI** | **67.6** (+14.8%) | 45.8 | 61.7 (+2.3%) | **81.9** (+20.3%) |

**Exceptional temporal reasoning** is the standout: 67.6 (gpt-4o-mini) and 77.3 (gpt-4.1-mini), the largest gains over baselines. The episode-centric design front-loads reasoning from response time into memory formation — e.g. distilling "yesterday" into an explicit dated fact at write time, turning a hard temporal-reasoning query into simple fact retrieval. **Open-Domain is the one weak spot** (slightly below best baseline) — those questions need the backbone's *prior world knowledge* (e.g. answer "UNO" when the dialogue never names it), not just memory quality.

### LongMemEval_S (105K-token stress test) — scalability (RQ6)

| Question Type | gpt-4o-mini Full-ctx (101K) | gpt-4o-mini NEMORI (3.7–4.8K) | gpt-4.1-mini Full-ctx | gpt-4.1-mini NEMORI |
|---|---|---|---|---|
| Single-session Preference | 6.7 | 46.7 | 16.7 | 86.7 |
| Single-session Assistant | 89.3 | 83.9 | 98.2 | 92.9 |
| Temporal Reasoning | 42.1 | 61.7 | 60.2 | 72.2 |
| Multi-session | 38.3 | 51.1 | 51.1 | 55.6 |
| Knowledge Update | 78.2 | 61.5 | 76.9 | 79.5 |
| Single-session User | 78.6 | 88.6 | 85.7 | 90.0 |
| **Average** | **55.0** | **64.2** (+16.7%) | **65.6** | **74.6** (+13.7%) |

The Full-Context gap **widens** dramatically at 105K vs 9K context (+16.7%/+13.7% here vs only +1.0%/+0.2% on LoCoMo) — distillation becomes more valuable as context grows and attention dilutes, while Nemori uses **95–96% fewer tokens**.

### Efficiency

**Memory construction** (LoCoMo, gpt-4o-mini) — Nemori reduces **LLM calls by 59.5%** and **total tokens by 38.7%** vs the cheapest baselines, despite a multi-prompt pipeline, because it processes **episodes not messages**:

| Method | LLM Score | Calls | Total Tokens (k) |
|---|---|---|---|
| LangMem | 51.3 | 920.6 | 1010.2 |
| Mem0 | 61.3 | 1602.2 | 1693.4 |
| A-MEM | 52.5 | 1175.5 | 1149.4 |
| MemoryOS | 54.5 | 1016.1 | 526.5 |
| **NEMORI** | **73.0** | **373.2** (↓59.5%) | **322.9** (↓38.7%) |

**Response generation** (LoCoMo, gpt-4o-mini) — 2,745 tokens/query (**88% less than Full Context's 23,653**), accuracy 73.0 vs 72.3, and **47% lower end-to-end latency** (3,053ms vs 5,806ms).

---

## Ablation Study (LoCoMo, Overall LLM-judge)

| Variant | gpt-4o-mini | gpt-4.1-mini |
|---|---|---|
| w/o NEMORI (no memory) | 0.6 | 1.2 |
| **Nemori-s** (direct distillation, no prediction-error) | 52.0 | 65.5 |
| **w/o e** (no episodic retrieval; prediction-error semantic only) | 65.0 | 74.9 |
| **w/o s** (no semantic retrieval) | 54.7 | 76.9 |
| **w/o p** (fixed 20-msg chunks, no adaptive partitioning) | 68.0 | 75.7 |
| **NEMORI (Full)** | **73.0** | **80.8** |

Key findings:

1. **Prediction-error distillation is the core contribution.** `w/o e` (which still uses prediction-error-derived semantic memory) beats `Nemori-s` (direct distillation over raw episodes) by **+25.0%** (65.0 vs 52.0) on gpt-4o-mini and +14.4% on gpt-4.1-mini. The gap is most extreme on Temporal Reasoning (+73.9% on gpt-4o-mini), confirming prediction error excels at flagging time-sensitive surprises.
2. **Episodic + semantic are complementary and both indispensable.** Removing semantic retrieval is the most damaging on gpt-4o-mini (73.0 → 54.7, −25.1%); removing episodic retrieval costs −11.0%.
3. **Adaptive partitioning helps** (w/o p: 73.0 → 68.0) but performance is **robust to the window length `w`** (stable ±1% across w = 5–40).
4. **Native management is near-neutral on these benchmarks** (toggling it: 64.6 vs 65.0) — because LoCoMo rarely requires knowledge updates; retained for real-world deployment.
5. **Narrative-index beats raw-index** (Representation Prior validated): embedding narratives outperforms embedding raw episodes (76.9 vs 76.4). Top-K saturates by k≈10 (97% of peak).

### Third-Party Integration (RQ5) — Nemori as a distillation kernel

Feeding Nemori's **distilled semantic knowledge `K`** (instead of raw messages `P`) into A-MEM and MemoryOS **reduces their storage by 45–64%** while maintaining average performance (±4%) and **improving Core scores +1.9% to +6.1%** (e.g. A-MEM gpt-4o-mini: 397K → 142K tokens, ↓64.3%, Core +6.1%). This substantiates the management-agnostic claim — distilled memory is a compact, information-rich substrate for downstream systems.

---

## Key Takeaways

1. **"What deserves memory" = prediction error.** Operationalizing predictive-coding theory replaces hand-written importance/emotion/fact heuristics with a data-driven signal: store what existing knowledge fails to predict, drop the predictable (redundant) rest. This is the paper's central, transferable idea.

2. **Distillation vs management is a useful decomposition.** By formally separating *what to retain* (distillation) from *how to organize* (management), Nemori can be **management-agnostic** — a pluggable distillation layer that boosts/compresses A-MEM and MemoryOS rather than competing with them.

3. **Episode-centric processing is both better and cheaper.** Treating the episode (not the message) as the unit front-loads reasoning into memory formation (strong temporal results) while cutting LLM calls ~60% and construction tokens ~39%.

4. **Two cascading modules echo Complementary Learning Systems:** fast episodic integration (hippocampus) → slow semantic distillation (neocortex), bridged by the anticipatory-schema prediction step.

5. **Value scales with context length.** Marginal over Full Context at 9K (LoCoMo) but +13–17% at 105K (LongMemEval) with 95–96% fewer tokens — distillation pays off precisely where long-context attention dilutes.

---

## Limitations (Acknowledged by Authors)

1. **Naive management & retrieval.** Nemori focuses on distillation and adopts simple management/retrieval strategies; reported numbers reflect this design scope, not the ceiling of a complete system. More sophisticated reasoning-over-memory tasks may expose this as a bottleneck.
2. **Conceptual interfaces.** The `Evoke`/`Consolidate` interfaces are currently conceptual; lacking standardized protocols, concrete third-party integration still needs case-by-case implementation.
3. **(Implicit) heavy LLM dependence at write time** — partitioning, narration, integration, schema synthesis, and distillation are all LLM calls; the prediction-error signal is only as good as the backbone's predictions (narration 38.3% and distillation 30.3% dominate construction cost).

---

## Where it sits (v1/v2)

Nemori is a **v2 self-organizing episodic-memory** system: structure emerges from the data (latent-integrity partitioning + associative re-stitching) rather than a top-down taxonomy, and "what to remember" is decided by an intrinsic prediction-error signal rather than designer heuristics — squarely in v2's *"decide what/how/when to remember with policies the system learns or routes,"* and a clean operationalization of v2's **cognitive-grounding** theme (Predictive Coding + Complementary Learning Systems, alongside MAGMA's CLS dual-stream, LightMem's Atkinson-Shiffrin + sleep, and RF-Mem's dual-process).

**Group it with** the other self-organizing / emergent-structure systems in this collection:
- **EverMemOS** (`evermemos_self_organizing_memory_os.pdf`) — a self-organizing memory *OS*; Nemori is a self-organizing *distillation* layer that is management-agnostic and could in principle sit atop such an OS.
- **A-MEM** (`a_mem_agentic_memory_notes.md`) — **emergent organization** from note linking + memory evolution; Nemori takes emergence further upstream, into *what to write* rather than *how to link* what's written. (Notably, Nemori shows it can **compress A-MEM's store by 64%** as a distillation kernel — RQ5.)

**Cross-reference — Nemori is a baseline in this collection's MAGMA notes.** MAGMA (`magma_multi_graph_agentic_memory_notes.md`) treats Nemori as the **strongest prior baseline** on LoCoMo, reporting it at **~0.590 overall** (vs MAGMA's 0.700, a claimed +18.6%), and as the most efficient at build time (0.29h). Two reconciliation notes:
- MAGMA's LoCoMo number (0.590) is reported on its own scale/run and is *lower* than Nemori's self-reported 0.730 (gpt-4o-mini) / 0.808 (gpt-4.1-mini) here — different evaluation harnesses/judging, so the absolute numbers are not directly comparable across the two papers; the *relative ordering within each paper* is what's load-bearing.
- On **LongMemEval**, MAGMA's notes cite Nemori at 56.2 avg (3.7–4.8K tokens); Nemori's own paper reports 64.2 (gpt-4o-mini) / 74.6 (gpt-4.1-mini) at 3.7–4.8K — again, different runs/backbones. Both agree Nemori's defining trait is **95–96% token reduction with competitive-or-better accuracy**.
