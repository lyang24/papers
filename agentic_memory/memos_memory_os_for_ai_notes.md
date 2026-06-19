# MemOS: A Memory OS for AI System

**Authors:** Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, et al. (large multi-institution team led by **MemTensor (Shanghai) Technology Co., Ltd.**; with Institute for Advanced Algorithms Research Shanghai, China Telecom, Tongji, Zhejiang U., USTC, Peking U., Renmin U., Beihang, SJTU). Corresponding: Wentao Zhang, Zhi-Qin John Xu, Siheng Chen, Feiyu Xiong.

**Paper:** arXiv:2507.03724v4 (Dec 2025)

**Project:** https://memos.openmem.net/ | **Code:** https://github.com/MemTensor/MemOS

> **Naming caution (read first):** This is **MemOS** (MemTensor). It is **distinct** from the collection's existing **MemoryOS** (BUPT/Tencent, arXiv:2506.06326). They share the "memory operating system" tagline but are different systems with different abstractions. A side-by-side contrast is given at the end of this note.

---

## The Core Problem

LLMs have no well-defined **memory management system**. Knowledge lives in two disconnected places, neither of which is a *manageable, schedulable system resource*:

1. **Parametric memory** (billions of frozen weights) — strong generalization, but high update cost, poor interpretability, catastrophic-forgetting risk on retraining. Cannot reflect evolving knowledge in a timely way.
2. **RAG / external plaintext** — dynamic at inference time, but a **stateless "on-the-fly retrieval and transient composition" pipeline**. It lacks lifecycle tracking, versioning, provenance, and permission-aware scheduling. It can cite outdated and new regulations simultaneously with no reconciliation; it cannot retire obsolete facts.

The authors frame this through a **memory-hierarchy argument** (borrowed from their prior work **Memory3**): without an *explicit, intermediate memory layer* between parametric storage and external retrieval, models are stuck at a bad point on the read/write-cost vs. storage-cost tradeoff. The gap surfaces in four recurring failure contexts:

- **Long-range dependency modeling** — context windows, quadratic attention, and instruction drift over long horizons (user code style / writing style forgotten).
- **Adapting to knowledge evolution** — static parameters can't reflect legal/scientific/news updates; RAG patches statelessly with no versioning or temporal awareness.
- **Personalization & multi-role support** — sessions reset to blank; ChatGPT/Claude-style memory has capacity limits, unstable access, opaque updates, no editability.
- **Cross-platform memory migration** — memory trapped in instances forms "memory islands" (ideas in ChatGPT don't carry to Cursor), forcing context rebuilding.

The shared root cause: **the absence of a system-level mechanism for organizing and operating over memory** — analogous to how a computer OS manages CPU, RAM, and I/O across their lifecycle.

---

## The Big Idea: Memory as a First-Class OS Resource

The thesis: stop treating memory as either a "cache" or an "external retrieval module," and instead **redefine the operational logic and resource management of memory from a systems-level perspective**. Just as a traditional OS abstracts and schedules CPU/RAM/disk/IO across their lifecycles, MemOS abstracts and schedules *memory* as a first-class, schedulable, evolvable resource.

Three core capabilities motivate the design:

- **Controllability** — full lifecycle management (creation, activation, fusion, disposal) + multi-level permission control + operation auditing.
- **Plasticity** — restructure / migrate memory across tasks and roles via slicing, tagging, hierarchical mapping, and context binding.
- **Evolvability** — dynamic transitions and unified scheduling **among the three memory types** (parameter ⇄ activation ⇄ plaintext).

This is positioned as the entry into **Stage 4: Systematic Memory Governance** in the paper's four-stage taxonomy of LLM memory (Stage 1 definition/exploration → Stage 2 human-like memory → Stage 3 tool-based CRUD memory → **Stage 4 governance**). The broader vision is the **"Mem-training" paradigm**: a proposed *next scaling law* after pre-training and post-training, where models evolve continuously at runtime via explicit, controllable memory units rather than sporadic parameter updates.

---

## The Three Unified Memory Types

MemOS's central technical claim is that it is the **first hierarchical architecture to model and unify three distinct memory substrates**, and — crucially — to let memory **flow between them**.

| Type | What it is | Stored as | OS analogy | Properties |
|---|---|---|---|---|
| **Plaintext Memory** | Explicit, dynamically retrieved knowledge modules (retrieved passages, KG nodes, prompt templates) injected into model input | Text / graph structures | **I/O Buffer** (external episodes) | Explicitly stored, structurally organized; editable, traceable, versioned. Organized in a task–concept–fact graph hierarchy. |
| **Activation Memory** | Intermediate inference state — chiefly the **KV-cache**, plus hidden states `hˡᵢ` and attention weights `αˡᵢⱼ`; also steering vectors / semantic templates | Tensors (KV pairs, injection-layer activations) | **Cache** (fast working state) | Inference-coupled, state-responsive; short-term, dynamic, implicitly activated. Supports lazy loading, selective freezing, "instant memory paths." |
| **Parameter Memory** | Knowledge baked into fixed weights (FFN matrices `WˡMLP`, attention `WˡK/WˡV`); extendable via **LoRA / adapters** as loadable "capability modules" | Weight deltas (e.g., LoRA patch) | **Registers / Microcode** (long-term ability) | Implicitly embedded, statically encoded; activated without retrieval. High update cost, low interpretability. |

### The transformation paths (the load-bearing innovation)

Memory is not siloed — MemOS defines **consolidation / cross-type conversion pathways** so memory migrates toward its optimal invocation form:

- **Plaintext ⇒ Activation**: frequently used plaintext is pre-transformed into KV/attention templates for faster decoding ("instant memory paths").
- **Plaintext / Activation ⇒ Parameter**: stable, cross-task knowledge is **distilled into parameter modules** (capability plugins).
- **Parameter ⇒ Plaintext**: cold/outdated parameters are **offloaded ("backpatched") into external plaintext** to restore flexibility.

This bidirectional flow — bridging retrieval with parameter-based learning — is what distinguishes MemOS from systems that handle only plaintext memory.

---

## The Core Abstraction: the MemCube

The **MemCube** is the universal encapsulation unit — the minimal schedulable/composable memory resource, analogous to a process control block or a page in an OS. Every memory type (plaintext, activation, parameter) is wrapped as a MemCube with a uniform interface. Each MemCube has two parts:

1. **Memory Payload** — the semantic content. Polymorphic by `type`:
   - `explicit` / text → plaintext content
   - `activation` / tensor → injection-layer activation state
   - `parametric` / `lora_patch` → low-rank weight delta targeting a named module (e.g., `mlp.6.down_proj`)
2. **Metadata Header** — three metadata groups that make memory governable:
   - **Descriptive Identifiers** — Timestamp, **Origin Signature** (inference-extracted / user input / external retrieval / parameter finetune), Semantic Type (task prompt / fact / user preference) — "semantic fingerprints."
   - **Governance Attributes** — Access Control (read/write/share scope), Lifespan Policy (TTL / decay), Priority Level, Compliance & Traceability (sensitivity tags, watermarks, logs). This is the **governance kernel**.
   - **Behavioral Usage Indicators** — runtime access patterns (frequency/recency → "hot"/"cold"), plus the signals that drive **Policy-Aware Scheduling**, the **Contextual Fingerprint** (fast retrieval/alignment), and the **Version Chain** (modification lineage → rollback / conflict resolution).

MemCubes can be **composed, migrated, fused, and version-controlled** over time — they are the carrier that flows across all three architectural layers.

---

## Architecture: Three Layers

MemOS adopts a modular three-layer architecture (mirroring the OS-component mapping in Table 2), with the MemCube flowing through all layers.

### 1. Interface Layer
The entry point. Exposes a unified, composable **Memory API** (provenance / update / log-query) and **Memory Pipelines** (declarative DSL chains like `retrieve → augment → update → archive` with transactional consistency + rollback).
- **MemReader** — semantic abstraction module. Parses NL prompts into a structured **MemoryCall** (caller ID, context scope, memory type, access intent, time window). E.g., "Summarize my meeting notes from last month" → extracts time range, memory type, output target. Handles coreference resolution and dialogue slot filling.

### 2. Operation Layer (the control center)
- **MemOperator** — organizes memory via three complementary mechanisms: (i) flexible **tagging**, (ii) **knowledge-graph** linking, (iii) **semantic layering** (private / shared / global). Provides **hybrid retrieval** (structured rule-based filtering + semantic vector search) and **task-aligned routing** that decomposes input into a **topic–concept–fact** schema, resolving *what / where / in-what-order* to search.
- **MemScheduler** — the central dispatcher. Goes beyond retrieval: it **type-aware transforms and loads** memory into runtime context based on task semantics, call frequency, and content stability. Stable+frequent → Activation (KV cache, low prefill latency); abstract reusable rules → Parameter (distillation/adapters); time-sensitive/session-specific → Plaintext. Performs **cross-type migration** (e.g., promote frequently-recalled plaintext to KV; demote underused KV to cold plaintext).
- **MemLifecycle** — models each memory as a **finite state machine**. States: **Generated → Activated → Merged → Archived** (plus an **Expired** terminal and a **Frozen** state for legal/compliance memories where edits are disabled but full history is retained). Offers a **"Time Machine"**: snapshot + historical rollback for forgetting detection, user retractions, and counterfactual "what-if" simulation.

### 3. Infrastructure Layer
- **MemGovernance** — access control (ternary model: user identity × memory object × calling context; private/shared/read-only), TTL enforcement, access-frequency GC, sensitive-content detection/redaction, provenance + semantic watermarking, audit interfaces.
- **MemVault** — versioned multi-repository store (user-private, expert knowledge, industry-shared, contextual pools, pipeline caches) over a unified **MemoryAdapter** abstraction (vector store / relational DB / blob).
- **MemLoader / MemDumper** — bidirectional memory migration across platforms (edge↔cloud, system handover) with auto-filled provenance.
- **MemStore** — open **publish/subscribe** interface for memory exchange between models/institutions; enables the **"paid memory as modular installables"** marketplace vision (a physician publishes diagnostic heuristics as an installable memory plugin).

---

## Evaluation

All baselines run on the **same GPT-4o-mini backbone** for architectural parity; experiments on an 80GB H800 GPU. The evaluated model is **MemOS-1031**. Baselines: **MIRIX** (six-component memory), **Mem0** (slot-based top-k), **Zep** (time-aware KG), **Memobase**, **Supermemory** (dynamic KG), **MemU** (multimodal summarized files).

### LoCoMo Benchmark (LLM-Judge Scores, Table 3)

| Method | Tokens | Single-hop ↑ | Multi-hop ↑ | Temporal ↑ | Open-domain ↑ | **Overall ↑** | F1 ↑ |
|---|---|---|---|---|---|---|---|
| MIRIX | — | 68.22 | 54.26 | 68.54 | 46.88 | 64.33 | 28.10 |
| Mem0 | 1172 | 73.33 | 58.75 | 52.34 | 45.83 | 64.57 | 43.46 |
| Zep | 2701 | 66.23 | 52.12 | 54.82 | 33.33 | 59.22 | 41.23 |
| Memobase | 2102 | 73.12 | 64.65 | — | 53.12 | 72.01 | — |
| MemU | 617 | 66.34 | 63.12 | — | 50.01 | 56.55 | — |
| Supermemory | 500 | 67.30 | 51.12 | — | 42.67 | 55.34 | — |
| **MemOS-1031** | 1589 | **81.09** | **67.49** | **55.90** | **75.80** | **81.20** | **50.18** |

> Note: some baseline cells are missing in the source table (printed as gaps); reproduced as `—` above. MemOS leads on every reported LoCoMo category. The standout gaps are **open-domain (75.80 vs ≤53.12)** and **overall (81.20)** — achieved at a modest ~1.6K-token context, so the high judge scores are *not* from retrieval token overflow.

### LongMemEval Benchmark (accuracy %, Table 4)

| Method | Tokens | Single-sess. pref ↑ | Single-sess. asst ↑ | Temporal ↑ | Multi-session ↑ | Knowledge update ↑ | Single-sess. user ↑ | **Overall ↑** |
|---|---|---|---|---|---|---|---|---|
| MIRIX | 1.6k | 53.3 | 63.6 | 25.6 | 30.1 | 52.6 | 72.9 | 43.49 |
| Zep | 1.1k | 53.3 | 75.0 | 54.1 | 47.4 | 74.4 | 92.9 | 63.8 |
| Mem0 | 1.5k | 90.0 | 26.8 | 72.2 | 63.2 | 66.7 | 82.9 | 66.4 |
| Memobase | 0.4k | 80.1 | 23.2 | 75.9 | 66.9 | **89.7** | 92.9 | 72.4 |
| MemU | 0.5k | 89.9 | 58.9 | 44.4 | 52.6 | 55.1 | 85.7 | 58.4 |
| Supermemory | 1.4k | 76.7 | 19.6 | 17.3 | 42.1 | 41.0 | 67.1 | 38.4 |
| **MemOS-1031** | — | **96.7** | 67.9 | **77.4** | **70.7** | 74.3 | **95.7** | **77.8** |

MemOS wins **overall (77.8 vs Memobase 72.4)** and places first or second in every category **except knowledge-update** (where Memobase's 89.7 leads).

### PreFEval — Personalization (Table 5, Personalized Response ↑)

| Method | 0 turns | +10 irrelevant turns |
|---|---|---|
| Bare LLM | 9.6 | 2.8 |
| Bare LLM (+rag-5) | 51.2 | 43.2 |
| Mem0 | 65.9 | 63.7 |
| Supermemory | 58.4 | 56.7 |
| MemU | 54.2 | 51.8 |
| **MemOS-1031** | **77.2** | **71.9** |

MemOS has the best Personalized Response in both settings **and** the lowest Preference-Unaware error — robust to 10 injected irrelevant turns.

### PersonaMem — Precision 1-in-4 (Table 6)

| Method | Precision ↑ | Tokens |
|---|---|---|
| MIRIX | 38.4 | — |
| Mem0 | 43.1 | 140 |
| Zep | 57.8 | 1657 |
| Memobase | 58.9 | 2092 |
| MemU | 56.8 | 496 |
| Supermemory | 53.9 | 204 |
| **MemOS-1031** | **61.2** | 1424 |

### Retrieval Robustness under QPS Pressure (Table 7)
MemOS sustains a **100% success rate** for both add and search even at **100 QPS**, with the lowest latency across nearly all metrics. Competing APIs degrade sharply (e.g., MemU drops to single-digit success %; Mem0 add-success falls to ~41.7% at 40 QPS).

### KV-Based Memory Acceleration (Table 8 — the token/latency-savings result)
MemOS converts hot, stable plaintext into **KV-form activation memory** pre-cached on GPU, avoiding repeated prompt encoding. Comparing **KV-cache injection vs. prompt-based injection** (TTFT), output sequences stay identical (semantic equivalence), and acceleration grows with model size and context length:

| Model | Best-case TTFT Speedup |
|---|---|
| Qwen3-8B (long ctx / short query) | **94.2%** |
| Qwen3-32B (long ctx / short query) | **88.8%** |
| Qwen2.5-72B (long ctx / short query) | **91.4%** |

> Speedups range ~18.6%–94.2% across the context/query grid. The headline takeaway: avoiding redundant prompt prefill via activation-memory caching cuts time-to-first-token by up to **~94%** on large models / long contexts. (Note: this is a **latency/compute savings via KV reuse** result; the v4 paper's quantified efficiency claim is this TTFT reduction, not a single fixed "% token savings" number — treat any "~35% token savings" figure as belonging to an earlier MemOS write-up rather than this v4 table.)

---

## Ablation / Configuration Studies

- **Chunk size & Top-K (Fig. 9):** performance rises **monotonically with memory capacity**, most strongly on **multi-hop** and **temporal** reasoning (the long-range-retrieval-bound tasks). F1/ROUGE-L/BLEU also improve; cosine similarity stays high → stable semantic alignment even at deep retrieval. No collapse at large K.
- **KV vs. prompt injection (§6.5):** isolates the activation-memory contribution — identical outputs, large TTFT reduction → validates the plaintext⇒activation transformation path as a real, free latency win.
- **QPS robustness (§6.4):** isolates the scheduling/organization layer — 100% success under load isolates MemScheduler + hybrid retrieval as the stability source.

---

## Key Takeaways

1. **"Memory as a first-class OS resource" is the thesis.** The contribution is less any single algorithm and more a *systems framing*: abstraction (MemCube) + scheduling (MemScheduler) + lifecycle FSM (MemLifecycle) + governance (MemGovernance), mapped 1:1 onto classic OS components.
2. **MemCube is the unifying abstraction.** One encapsulation — payload + three-group metadata — wraps plaintext, activation, and parameter memory identically, making all three schedulable, versionable, permissioned, and migratable.
3. **The three-way memory flow is the differentiator.** Plaintext⇄Activation⇄Parameter conversion lets knowledge move toward its cheapest/fastest form: distill stable facts into weights, cache hot facts as KV, offload cold weights to text.
4. **Strong empirical results at low token cost.** SOTA on LoCoMo (81.20 overall), LongMemEval (77.8), PreFEval, PersonaMem — all on GPT-4o-mini parity — plus 100% retrieval success at 100 QPS and up to ~94% TTFT reduction via KV reuse.
5. **A marketplace/governance vision.** MemStore + provenance + watermarking enable "paid memory as installable plugins" and cross-LLM sharing (proposed **Memory Interchange Protocol, MIP**), positioning memory as a tradeable, governed asset.

---

## Limitations & Open Questions

1. **Heavy systems engineering.** Three memory substrates + three architectural layers + a dozen named modules (MemReader, MemOperator, MemScheduler, MemLifecycle, MemGovernance, MemVault, MemLoader/Dumper, MemStore) is a large, complex stack relative to flat vector-memory baselines.
2. **Parameter-memory transformations are aspirational at scale.** Distilling plaintext into LoRA modules and "backpatching" cold parameters to text are described conceptually; the quantified evaluation is dominated by the **plaintext** and **KV/activation** paths — the full parameter⇄plaintext round-trip is under-quantified.
3. **Sparse/missing baseline cells.** Several LoCoMo cells in Table 3 are blank in the source, complicating exact head-to-head comparison on some sub-tasks.
4. **Evaluation scope.** Benchmarks are long-context conversational / personalization (LoCoMo, LongMemEval, PreFEval, PersonaMem); generalization to multimodal or embodied agents is future work.
5. **The "Mem-training scaling law" is a vision, not a result.** The claim that runtime memory evolution is the next scaling axis after pre-/post-training is motivational, not yet empirically demonstrated.

---

## MemOS (MemTensor) vs. MemoryOS (BUPT/Tencent) — Do Not Confuse

Both are in this collection and both call themselves a "memory operating system," but they are fundamentally different systems:

| Aspect | **MemOS** (this paper) | **MemoryOS** (`memoryos_memory_operating_system_notes.md`) |
|---|---|---|
| Authors / org | **MemTensor (Shanghai)** + multi-univ. consortium | **BUPT + Tencent AI Lab** (Kang, Ji, Zhao, Bai) |
| arXiv | 2507.03724 (Jul/Dec 2025) | 2506.06326 (May 2025) |
| Central abstraction | **MemCube** (universal memory unit; payload + governance metadata) | **Segment–Page** memory (OS segment/page management) |
| Memory taxonomy | **Parametric + Activation (KV) + Plaintext** — and *flow between them* | **STM → MTM → LPM** (Short / Mid / Long-term Personal), purely plaintext/dialogue |
| OS metaphor borrowed | Whole-OS resource governance (scheduler, file system, ACLs, drivers) mapped to modules | **Segment-page memory management + heat-based (LRU/working-set) eviction** |
| Scope of "memory" | **Unifies model internals** (weights, KV-cache) *and* external text | **External conversational memory only** — does not touch weights or KV-cache |
| Key differentiator | Cross-type transformation (plaintext⇄activation⇄parameter); MemStore marketplace; full governance kernel | Dialogue-chain segmentation; heat-based page eviction; user-portrait LPM |
| Lifecycle | FSM: Generated→Activated→Merged→Archived (+Frozen, Time Machine) | Page heat scoring → MTM→LPM promotion / eviction |

**One-line discriminator:** MemOS is the **breadth-maximalist** system that tries to unify *parametric + latent + plaintext* memory under one schedulable abstraction; MemoryOS is a **conversational-memory** system that borrows OS *segment-page paging* to organize a STM/MTM/LPM dialogue store.

---

## Where it sits (v1/v2)

- **Generation: v2 memory-OS.** MemOS is a **v2** system — it moves past v1 retrieval-augmented note stores toward a *governed, scheduled, lifecycle-managed* memory substrate. Among all "memory-OS" entries in this collection it is the **most ambitious in breadth**: the only one that explicitly tries to **unify parametric + latent (activation/KV) + plaintext** memory in a single resource model with conversions between them.
- **Relation to MemGPT's OS metaphor.** MemGPT (`memgpt_llms_as_operating_systems_notes.md`) introduced the OS analogy as **virtual-context paging** — function-style swaps between a fixed context window (main memory) and external storage (disk). MemOS generalizes that metaphor from *just context paging* to a **full OS resource-management discipline**: scheduler, file system (MemVault), ACL/governance, device drivers (MemLoader/Dumper), package manager (MemStore). Where MemGPT paged *plaintext context*, MemOS schedules *all three memory substrates* including model weights and KV-cache.
- **Relation to the survey's three Forms.** The memory survey (`memory_in_age_of_ai_agents_survey_notes.md`) classifies memory into three **Forms — token (plaintext), parametric, and latent (KV/hidden-state)**. MemOS is, in effect, a **concrete engineering instantiation of all three Forms at once**: Plaintext Memory = the token form, Parameter Memory = the parametric form, Activation Memory = the latent form — with MemCube as the common wrapper and MemScheduler converting between Forms at runtime. Most other systems in the collection (A-MEM, Mem0, Zep, MAGMA, MemoryBank, MemoryOS) live almost entirely in the **token Form**; MemOS is the standout attempt to operate across all three.
