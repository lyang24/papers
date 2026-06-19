# Agent Memory: v1 → v2 Evolution Overview

A cross-generation map of the 18 papers in this collection. **v1** = foundational era (2023–2025); **v2** = the 2026 frontier (plus a few late-2025 papers that already point at it). The dividing survey, *Memory in the Age of AI Agents* (Dec 2025), sits on the boundary and predicts most of what v2 delivers.

---

## 1. The roster

### v1 — foundational (2023–2025)
| Paper | Year | One-line |
|---|---|---|
| MemGPT | 2023.10 | OS virtual-memory metaphor; LLM self-pages context ↔ external DB |
| MemoryBank | 2023.05 | Ebbinghaus forgetting curve + user portraits; plug-and-play |
| ReadAgent | 2024 ICML | Fuzzy-trace gist memory; LLM-as-retriever beats embedding RAG |
| LOCOMO | 2024.02 | Benchmark; "observations > raw dialog > summaries" for retrieval |
| A-Mem | 2025.02 | Zettelkasten atomic notes; link generation + memory **evolution** |
| MemoryOS | 2025.05 | 3-tier STM/MTM/LPM; segment-page; heat-based eviction; 90-dim persona |
| *Survey* | 2025.12 | Forms / Functions / Dynamics taxonomy; boundary document |

### v2 — frontier (2025–2026)
| Paper | Method | Year | One-line |
|---|---|---|---|
| MAGMA | MAGMA | 2026.01 | Disentangled multi-graph (temporal/causal/semantic/entity) + intent-routed retrieval + dual-stream write |
| LatentMem | LatentMem | 2026.03 | Learnable **latent** memory (not text), role-aware, RL-trained composer (LMPO), multi-agent |
| Zep | Zep/Graphiti | 2025.01 | Bi-temporal knowledge graph; production framework |
| Multi-Granularity | **MemGAS** | 2025.05 | 4-level granularity + GMM association + entropy granularity router + PPR (training-free) |
| Cognee | Cognee | 2025.05 | Modular ECL (Extract-Cognify-Load) KG framework; TPE hyperparameter optimization |
| LightMem | LightMem | 2026 ICLR | Atkinson-Shiffrin 3-stage; sleep-time offline consolidation; extreme efficiency |
| GAM | GAM | 2025.11 | Just-in-Time compilation; dual-agent Memorizer + Researcher; runtime deep research |
| Evoking User Memory | **RF-Mem** | 2026.03 | Cognitive dual-process retrieval (fast Familiarity / slow Recollection); personalization |
| Learning How/What | **MemCoE** | 2026.05 | Cognition-inspired 2-stage: learn guideline + RL memory policy (multi-turn GRPO) |
| RTBF audit | WikiMem | 2025.07 | GDPR right-to-be-forgotten; quantify parametric memorization before unlearning |
| LLMs Get Lost | — | 2025.05 | **Diagnostic** (not a system): multi-turn drop ~39%; motivates memory/context mgmt |

> Two papers are not memory *systems*: **LLMs Get Lost** (diagnostic, like LOCOMO) and **RTBF audit** (privacy/forgetting of parametric memory). They are kept here for the broader picture but flagged as off-axis.

---

## 2. Cross-generation comparison by dimension

| Dimension | v1 (foundational) | v2 (frontier) | Exemplar shift |
|---|---|---|---|
| **Control policy** | Hand-crafted rules & heuristics (FIFO, forgetting curve, heat scores, fixed thresholds) | **Learned / RL-driven** memory policies | A-Mem fixed link+evolve rules → MemCoE (GRPO), LatentMem (LMPO) |
| **Memory representation** | Token-level explicit text (notes, summaries, observations) | Disentangled **graphs** or compressed **latent** vectors | Flat notes → MAGMA multi-graph / Zep temporal-KG / LatentMem latent |
| **Retrieval** | Cosine similarity (± LLM reasoning over text) | **Intent / causal-routed** or **on-demand generated** | top-k embeddings → MAGMA intent routing, RF-Mem dual-process, GAM JIT deep research |
| **When memory is built** | Eagerly, upfront (write-time summarization/structuring) | **Just-in-time** at query time (keep raw, compile context on demand) | MemoryBank/A-Mem build-upfront → GAM JIT "deep research" |
| **Structure depth** | Flat → planar notes → 3-tier hierarchy | Multi-graph, multi-granularity, learned latent space | MemoryOS 3-tier → MemGAS 4-granularity + router, MAGMA 4 graphs |
| **Agent scope** | Single agent, mostly conversational | **Multi-agent**, role-aware shared memory | single-agent → LatentMem role-conditioned multi-agent memory |
| **Efficiency posture** | Heavy: LLM call per memory op (A-Mem, MemoryOS) | Explicit **efficiency engineering** (pre-compress, sleep-time, fewer calls) | A-Mem/MemoryOS heavy → LightMem (up to 38× fewer tokens, 30× fewer API calls) |
| **Cognitive grounding** | Loose analogies (OS, forgetting) | Specific dual-process / consolidation theories operationalized | Ebbinghaus decay → CLS fast-slow (MAGMA), recollection/familiarity (RF-Mem), Atkinson-Shiffrin + sleep (LightMem) |
| **Consolidation** | Eviction + recursive summary (synchronous) | **Async / offline "sleep-time"** consolidation off the critical path | MemGPT recursive summary → MAGMA slow-path, LightMem sleep-time |
| **Forgetting** | Utility-driven decay of external memory (MemoryBank) | + **Privacy/legal-driven** unlearning of parametric memory (RTBF) | Ebbinghaus prune → GDPR right-to-be-forgotten audit |
| **Trust / privacy** | Out of scope | Emerging axis (auditing, unlearning) | — → RTBF / WikiMem |
| **Memory paradigm** | Static repository to be **queried** | Self-evolving / **generated** memory that co-adapts | "store then retrieve" → "construct/learn memory" (GAM, LatentMem, MemCoE) |

**The one-sentence story:** v1 *stores text and queries it with similarity, using rules a human wrote*; v2 *structures, learns, or generates memory — and decides what/how/when to remember with policies the system learns or routes by intent.*

---

## 3. Mapping onto the survey's predicted frontiers

The Dec-2025 survey listed eight frontiers. v2 populates them:

| Survey frontier (Section 7) | v2 papers that realize it |
|---|---|
| Retrieval → **Generation** | GAM (JIT runtime context), LatentMem (generated latent memory) |
| **Automated** memory management | MemCoE, LatentMem (learned write/select policies) |
| **RL** meets memory | MemCoE (GRPO), LatentMem (LMPO) |
| Multimodal memory | *(still open — none here fully address it)* |
| **Shared** memory in multi-agent systems | LatentMem (role-aware) |
| Memory for world models | *(open)* |
| **Trustworthy** memory (privacy/forgetting) | RTBF audit / WikiMem |
| Human-cognitive connections (offline consolidation) | LightMem (sleep-time), MAGMA (CLS dual-stream), RF-Mem (dual-process) |

So v2 is essentially the survey's roadmap being executed — with **multimodal** and **world-model** memory still wide open.

---

## 4. The MemGAS → RF-Mem → MemCoE research line (USTC / CityU / Huawei)

Three of the v2 papers are one continuous program of work from the same core authors — **Derong Xu, Xiangyu Zhao, Tong Xu, Enhong Chen** (USTC + City University of Hong Kong + Huawei Noah's Ark Lab), with overlapping co-authors (Yi Wen, Pengyue Jia, Wenlin Zhang, Yingyi Zhang, Yichao Wang, Huifeng Guo). They form a tight progression on long-term conversational / personalized memory:

| Step | Paper | Date | What it nails | What it leaves open → next step |
|---|---|---|---|---|
| **1. Structure** | **MemGAS** (multi-granularity) | 2025.05 | *How to organize & select* memory across granularities: 4-level units, GMM association, entropy-based granularity routing, Personalized PageRank — **training-free** | Selection is heuristic/entropy-based; retrieval still one-shot. → make retrieval cognitively adaptive |
| **2. Retrieve** | **RF-Mem** (recollection-familiarity) | 2026.03 | *How to retrieve* adaptively: a dual-process controller routes each query between fast **Familiarity** (one-shot) and slow **Recollection** (iterative cluster-and-mix), gated by a familiarity-uncertainty signal | Still retrieving from a fixed/hand-built store; write policy not learned. → learn what & how to write |
| **3. Learn to write** | **MemCoE** (two-stage optimization) | 2026.05 | *What & how to memorize*, **learned**: Stage 1 induces an organization guideline via textual gradients (prefrontal analogy); Stage 2 trains the memory-evolution policy with multi-turn **GRPO** (hippocampus analogy) | Text memory only; single-agent. → latent / multi-agent (cf. LatentMem) |

**The arc:** the group moved from *organizing* memory (MemGAS, training-free) → *retrieving* it adaptively (RF-Mem, cognitive routing) → *learning* what and how to write it (MemCoE, RL). It mirrors the whole-field v1→v2 shift in miniature — heuristic structure giving way to learned, cognition-grounded policies — and each paper is built on benchmarks the prior one used (LOCOMO, LongMemEval, PersonaMem, PersonaBench, PrefEval). A natural next move for this line would be **latent or multi-agent** memory, which is exactly where LatentMem (a different group) already points.

**Adjacent but distinct cognitive-grounding cluster** (different authors, same spirit): LightMem (Atkinson-Shiffrin + sleep), MAGMA (Complementary Learning Systems fast/slow), RF-Mem (dual-process) — v2's recurring theme of operationalizing a *specific* human-memory theory rather than a loose analogy.

---

## 5. Quick "what to read for what"

- **Best single architectural idea:** MAGMA (disentangled multi-graph + intent routing)
- **Best efficiency play:** LightMem (sleep-time consolidation, huge token/call savings)
- **Best "memory as learned policy":** MemCoE (RL) and LatentMem (latent + RL)
- **Best "build memory at query time":** GAM (JIT deep research)
- **Production frameworks:** Zep/Graphiti, Cognee
- **The same-author throughline:** MemGAS → RF-Mem → MemCoE
- **Diagnostics / motivation:** LOCOMO (v1), LLMs Get Lost (v2)
- **Trust / privacy axis:** RTBF audit / WikiMem
