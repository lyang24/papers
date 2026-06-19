# Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers

**Author:** Pengfei Du (Hong Kong Research Institute of Technology)

**Paper:** arXiv:2603.07670v1 (8 Mar 2026)

**Targets:** Advanced Intelligent Systems (manuscript stage)

---

## Why This Survey Exists

LLM agents increasingly run in settings where a single context window is far too small to capture what happened, what was learned, and what must not be repeated. Memory — the ability to persist, organize, and selectively recall information across interactions — is what turns a stateless text generator into a genuinely adaptive agent. The motivating image: a debugging assistant that, without memory, rediscovers the directory layout every Monday and retries the exact fix that crashed the build on Friday.

A memory-focused review already existed (Zhang et al. 2024), but the landscape shifted with a 2025–2026 wave — **Agentic Memory** (learned memory control), **MemBench**, **MemoryAgentBench**, **MemoryArena** — that introduced learned control and agentic benchmarks coupling memory with action. This survey covers work from 2022 through early 2026 and asks three research questions:

- **RQ1** — How should memory in LLM agents be **decomposed and formalized**?
- **RQ2** — What **mechanisms** exist, and what **trade-offs** do they impose?
- **RQ3** — How should memory be **evaluated** when the ultimate test is downstream agent performance?

---

## Problem Formulation: The Write–Manage–Read Loop (Section 2)

The survey's organizing primitive is a **write–manage–read loop tightly coupled with perception and action**, formalized inside a POMDP-style agent cycle.

At each step *t*, the agent receives input *x_t* and must produce action *a_t*, consulting accumulated memory in between:

- **Read:** `a_t = π_θ(x_t, R(M_t, x_t), g_t)` — policy reads from memory `R`, conditioned on goals `g_t`.
- **Write/Manage:** `M_{t+1} = U(M_t, x_t, a_t, o_t, r_t)` — update function `U` writes to and manages memory given action, observation `o_t`, and reward-like signal `r_t`.

Two emphasized points:
1. `U` is **not a simple append** — a well-designed `U` summarizes, deduplicates, scores priority, resolves contradictions, and deletes.
2. `π_θ` and `(R, U)` form a **feedback loop** — decisions shape what gets written, and what is written shapes future decisions. One bad write can pollute the store for many downstream steps (powerful but brittle).

### Connection to POMDPs (2.2)

Memory `M_t` plays the role of the agent's **belief state**: an internal summary of history standing in for the unobservable true world state. Classical POMDP solvers update beliefs via Bayesian filtering; LLM agents do something analogous but messier — through natural-language compression, vector indexing, or structured storage. The point: agent memory is **not a database-lookup problem** but maintaining a *sufficient statistic* of interaction history for good action selection under hard compute/storage budgets.

### Five Design Objectives and Their Tensions (2.3)

Memory mechanisms are pulled along five axes that **tug against each other**:

| Objective | Question |
|---|---|
| **Utility** | Does memory actually improve task outcomes? |
| **Efficiency** | Token, latency, storage cost per unit of utility gained? |
| **Adaptivity** | Can it update incrementally from feedback without full retrain? |
| **Faithfulness** | Is recall accurate and current? (Stale/hallucinated recall can be worse than none.) |
| **Governance** | Privacy respected, deletion supported, policy compliant? |

Maximizing utility tempts storing everything (bloats storage, governance headaches); aggressive compression buys efficiency but silently discards the rare fact that turns out critical weeks later. The "right" balance shifts with the application — a medical triage agent sits on a very different faithfulness–efficiency frontier than a recipe recommender.

### Memory as a Differentiator (2.4)

A four-layer reasoning stack aspiration (customer-support example): **procedural** says *how*, **semantic** says *what the policy is*, **episodic** says *what happened*, **working** holds the live reasoning context. Most current systems implement only two layers well and bridge them with crude heuristics; the **episode→semantic consolidation** step is the most underserved. Ablation evidence that memory rivals model scaling: Generative Agents degenerate within 48 simulated hours without reflection; Voyager loses 15.3× tech-tree speed without its skill library; MemoryArena task completion drops from >80% to ~45% when an active memory agent is swapped for a long-context-only baseline. **The "has memory" vs. "no memory" gap often exceeds the gap between LLM backbones.**

---

## The Three-Dimensional Taxonomy (Section 3)

Mirroring cognitive-science distinctions (Atkinson–Shiffrin, Tulving, Baddeley, Squire), the survey organizes the space along **three orthogonal dimensions**: temporal scope, representational substrate, and control policy.

### Dimension 1 — Temporal Scope (3.1)

| Scope | Description | Anchor Example |
|---|---|---|
| **Working memory** | Whatever fits the current context window. Maps to Baddeley's central executive (LLM) + buffer (window); shared capacity bottleneck. | — |
| **Episodic memory** | Records of concrete experiences — tool calls, turns, observations, with timestamp / importance / embedding. | Generative Agents observation stream |
| **Semantic memory** | Abstracted, de-contextualized knowledge consolidated from episodes ("user prefers DD/MM/YYYY"). Rarely automatic. | — |
| **Procedural memory** | Reusable skills and executable plans, indexed by NL descriptions. | Voyager skill library (runnable JS) |

The **hard open question** is the *transition policy*: when an episodic record graduates to semantic status, and when a semantic fact is instantiated back into working memory for a task.

### Dimension 2 — Representational Substrate (3.2)

How memory is physically stored constrains what the agent can efficiently do.

| Substrate | Strength | Limitation | Example |
|---|---|---|---|
| **Context-resident text** | Transparent, zero infrastructure | Ruthlessly capacity-limited | summaries, scratchpads, CoT |
| **Vector-indexed stores** | Scales to millions of records (ANN/FAISS) | Loses structured relationships ("what's similar?" not "what caused what?") | dense passage retrieval |
| **Structured stores** | Relational queries ("all API failures for service X in 7 days") | Upfront schema design | SQL (ChatDB), KV maps, knowledge graphs |
| **Executable repositories** | Invoke stored skills directly, sidestep regeneration | — | Voyager code library, tool defs |
| **Hybrid stores** | The production norm | Orchestration complexity | MemGPT (main / recall DB / archival vector) |

### Dimension 3 — Control Policy (3.3) — "the most consequential, least discussed dimension"

*Who decides what to store, retrieve, and discard?*

| Control | Description | Example |
|---|---|---|
| **Heuristic** | Hard-coded rules (top-k, summarize every n turns, expire after d days). Predictable, debuggable, context-blind. | — |
| **Prompted self-control** | Memory ops exposed as tool calls; LLM decides when to invoke. Quality hinges on instruction-following + API documentation. | MemGPT `core_memory_append`, `archival_memory_search` |
| **Learned control** | Memory ops as policy actions optimized end-to-end. Discovers non-obvious strategies (preemptive summarization) but costly to train. | Agentic Memory (store/retrieve/update/summarize/discard via 3-stage RL + step-wise GRPO) |

---

## Five Core Mechanism Families (Section 4)

The survey examines **five mechanism families** in depth, each with concrete systems and empirical trade-offs.

### 1. Context-Resident Memory and Compression (4.1)

Keep relevant info in the prompt; everything the LLM "sees" is working memory with perfect in-window recall. When history outgrows the window, compression strategies emerge: (i) **sliding windows**, (ii) **rolling summaries**, (iii) **hierarchical summaries** (turn/session/topic), (iv) **task-conditioned compression** (Self-Controlled Memory hands the decision to the agent).

- **Core pathology — summarization drift:** each compression pass silently discards low-frequency details; after enough passes the agent "remembers" a sanitized version. Worked example: a critical day-one instruction ("never call the production DB directly") survives the first pass but vanishes by the third.
- **Second pathology — attentional dilution** ("lost in the middle"): even within a large window, injecting more content degrades focus on any single piece. Bigger windows delay but don't eliminate the problem (and cost grows quadratically). Implication: supplement — not replace — context-resident memory with an external full-fidelity store.

### 2. Retrieval-Augmented Memory Stores (4.2)

RAG adapted to agents: the store holds **living interaction records** (tool logs, observations, corrections, partial plans, reflections), not encyclopedia articles.

- **Indexing granularity:** fine-grained gives precise recall but fragments reasoning; coarse preserves context but drowns signal. Sweet spot is **multi-granularity** with adaptive resolution. Default: dense passage retrieval + FAISS ANN, often augmented with BM25 and metadata filters.
- **Query formulation:** the immediate input *x_t* is often a poor query ("Why did that crash?" needs a log from two sessions ago). Strategies: LLM-reformulated queries, multi-query fan-out + fusion, subgoal-as-signal, Self-RAG's retrieve-or-not gate.
- **Scale:** RETRO and trillion-token datastores suggest retrieval scales to years of history without architectural change; the bottleneck shifts from **storage to relevance** (most *useful*, not most *similar*).
- **Read-write memory:** RET-LLM writes structured triplets at storage time, queried in NL at read time — schema at write, flexibility at read.

### 3. Reflective and Self-Improving Memory (4.3)

Reflexion: after failing, write a NL post-mortem and prepend it next attempt — no gradients, just self-critiques (91% pass@1 HumanEval vs. 80% GPT-4). Generative Agents add a richer pipeline (cluster observations → synthesize higher-order reflections; retrieval = recency × relevance × importance). ExpeL contrasts success/failure trajectories into "rules of thumb"; Think-in-Memory separates recall from a dedicated thinking step.

- **Central risk — self-reinforcing error:** a false belief ("API X always errors with param Y") is never re-tested. **Over-generalization** is its sibling. Severity *scales with agent lifetime* — most dangerous exactly where memory is most needed.
- **Mitigation — reflection grounding:** require each reflection to cite specific episodic evidence (three concrete failure instances), giving an auditable trail. Partial fix; cited evidence may itself be unrepresentative.

### 4. Hierarchical Memory and Virtual Context Management (4.5)

MemGPT borrows OS **virtual memory**: **main context (RAM)** = active window; **recall storage (disk)** = searchable DB of all past messages; **archival storage (cold)** = vector-indexed long-term store. The agent pages between tiers via memory functions, with an interrupt mechanism on each message/timer event. JARVIS-1 extends to multimodal (visual / textual plan / skill stores); CoALA proposes a generalized working/episodic/semantic/procedural blueprint around a central executive.

- **Achilles' heel — orchestration:** page the wrong things in and waste tokens; archive too aggressively and create **"memory blindness"** (agent doesn't know the fact exists in cold storage). Failures are **silent** — no exception, just a slightly worse response — and compound over time. Needs detailed operation logs + retrospective analysis.

### 5. Policy-Learned Memory Management (4.4)

Heuristics and prompted control aren't optimized for the end task. **Agentic Memory (AgeMem)** treats five ops — store / retrieve / update / summarize / discard — as callable tools, optimized end-to-end via three stages: **(1) supervised warm-up** on demonstrations, **(2) task-level RL** with outcome rewards, **(3) step-level GRPO** for dense credit assignment. Beats strong baselines on five long-horizon benchmarks; surfaces non-obvious tactics (proactive summarization before the context fills; discarding semantically redundant records).

- **Open concerns:** long-horizon RL is expensive; learned forgetting could delete safety-critical info; policies may fail to transfer; interpretability lags capability.

### (Bonus) Parametric Memory and Weight-Based Adaptation (4.6)

A separate family embeds memory *inside* the weights via fine-tuning/adapters (MemLLM fine-tunes an explicit read-write module; joint retrieval-generation training beats frozen-retriever baselines). Seamless integration ("the model just knows"), but hard to audit (where is the birthday stored?), hard to delete from (immature unlearning), and expensive to update — so most deployments favor **non-parametric, inspectable** stores.

---

## Evaluation: From Recall to Agentic Utility (Section 5)

### Why Classical Retrieval Metrics Fall Short (5.1)

Precision@k and nDCG tell you the right document was retrieved — not whether the agent *used it correctly* or whether retrieving it was *worth the latency*. Agent memory evaluation must jointly assess **memory quality and decision quality**, plus staleness, contradiction, forgetting quality, and governance — concerns classical IR ignores.

### The Four-Benchmark Landscape (5.2)

| Benchmark | Year | What It Probes | Headline Finding |
|---|---|---|---|
| **LoCoMo** | 2024 | Very long-term conversational memory (≤35 sessions, 300+ turns, 9k–16k tokens); factual QA / event summarization / dialogue gen | Even RAG-augmented LLMs lag humans, esp. on temporal & causal dynamics |
| **MemBench** | 2025 | Factual vs. reflective memory × participation vs. observation modes; effectiveness / efficiency / capacity | Capacity degrades as store grows |
| **MemoryAgentBench** | 2025 | Four cognitive competencies: accurate retrieval, test-time learning, long-range understanding, **selective forgetting** | No system masters all four; most fail on forgetting |
| **MemoryArena** | 2026 | Memory inside complete agentic tasks (web nav, preference-constrained planning, progressive search, sequential formal reasoning) where later subtasks depend on earlier learning | LoCoMo near-aces plummet to **40–60%** — deep gap between passive recall and active use |

Feature comparison (Table 2): only **LoCoMo & MemoryArena** are multi-session; **all four** are multi-turn; **only MemoryArena** has agentic tasks; **only MemoryAgentBench** tests forgetting; **only LoCoMo** is multimodal.

### Cross-Cutting Lessons (5.4)

1. **Long context is not memory** — long-context models underperform purpose-built memory systems on selective retrieval / active management. "Passive recall aces are poor memory agents."
2. **RAG helps, but the human gap is wide** — bottleneck is no longer storage but **retrieval quality**; agents surface plausible-but-stale records.
3. **Nobody evaluates forgetting well** — only MemoryAgentBench tests it; in long deployments the inability to discard outdated info poisons retrieval precision.
4. **Cross-session coherence is underexplored** — most benchmarks are within-session; consistency across hours/days is distinct and largely unsolved.
5. **The parametric–non-parametric gap is real** — different failure profiles (seamless integration vs. inspectability/governance); the optimal blend is open.
6. **Evaluation must include cost** — a +5% accuracy that triples latency/storage may not be an improvement; benchmarks should mandate token + latency reporting.

### A Practical Metric Stack (5.3)

A four-layer evaluation stack for deployment:

- **Layer 1 — Task effectiveness:** success rate, factual correctness, plan completion.
- **Layer 2 — Memory quality:** retrieved-record precision/recall, contradiction rate, staleness distribution, coverage of task-relevant facts.
- **Layer 3 — Efficiency:** latency per op, prompt tokens consumed by memory, retrieval calls per step, storage growth.
- **Layer 4 — Governance:** privacy leakage rate, deletion compliance, access-scope violations.

Ablations should isolate the **write policy**, **retrieval strategy**, and **compression module** separately.

---

## Where Memory Makes or Breaks the Agent (Section 6)

Memory is *not uniformly important* — a one-shot translator barely needs it; a month-long collaborator can't function without it. The survey's application-by-domain map:

| Domain | Dominant Memory Type | Distinctive Challenge | Systems |
|---|---|---|---|
| **Personal assistants** | Semantic (preferences/profile) | Personalization without overstepping privacy | MemoryBank (Ebbinghaus decay), MemGPT |
| **Software engineering** | Procedural (verified patterns, arch decisions) | **Structural scale** — index codebases of thousands of files, not just chats | ChatDev, MetaGPT (standardized docs) |
| **Open-world games** | Episodic + procedural | **Compositional skill reuse** — chain skills creatively | Voyager, JARVIS-1, Ghost in the Minecraft |
| **Tool use / API orchestration** | Procedural (tool catalog) | **Schema drift** — API updates invalidate stored usage; needs versioning | AgentBench, DERA |
| **Multi-agent collaboration** | Shared/coordination layer | Shared vs. private boundaries; concurrent-write consistency → role-based access control | AutoGen, CAMEL, ProAgent |
| **Scientific reasoning** | Semantic + **uncertainty tracking** | Memory as hypothesis ledger / evidence accumulator; maintain confidence levels | — |
| **Cross-domain transfer** (emerging) | Procedural | Which memories generalize vs. are hopelessly context-specific | Tree of Thoughts |

**Summary pattern (6.8):** different domains stress different memory types; no existing system supports all profiles simultaneously — suggesting the next leap is **modular, pluggable memory architectures** composed per deployment rather than monolithic designs.

---

## Engineering Realities (Section 7)

A practical playbook rarely discussed in research papers:

- **Write path (7.1):** storing everything verbatim is almost always wrong (noise degrades precision). Need filtering, canonicalization, deduplication, priority scoring, metadata tagging. The optimal filter threshold is **risk-driven and application-specific** (medical can't afford false negatives; chat can).
- **Staleness, contradictions, drift (7.2):** sending a card to an ex's old address is *harmful*, not just unhelpful. Need temporal versioning (prefer newest), source attribution (user statement > agent inference), contradiction detection, periodic consolidation.
- **Read path (7.3):** not every step needs retrieval — two-stage retrieval (BM25/metadata → cross-encoder rerank), retrieve-or-not gating, token budgeting, cache layers for high-frequency records.
- **Latency and cost (7.4):** pipelines add 200–500ms vs. sub-second expectations. Mitigations: async writes, progressive retrieval (generate while retrieving), dynamic routing. A modest context + targeted retrieval often beats both pure long-context and pure retrieval.
- **Privacy, compliance, deletion (7.5):** encryption, per-user scoping, PII redaction, retention policies, auditable deletion *across every tier* (including vector index + backups). When memories leaked into fine-tuned weights, only **machine unlearning** helps — far from production-ready. The intersection of memory governance and unlearning is an urgent open problem.
- **Three architecture patterns (7.6):**
  - **Pattern A — Monolithic context:** all memory in the prompt. Zero infra, transparent, capacity-capped, drift-prone. For short-lived agents / prototyping.
  - **Pattern B — Context + retrieval store:** working memory in window, long-term in external store. The **production workhorse** (coding assistants, support bots, copilots). Main challenge: retrieval quality.
  - **Pattern C — Tiered memory with learned control:** multiple tiers managed by a learned/prompted controller (MemGPT, AgeMem). Most headroom, most engineering/training. **Recommendation: start with B, instrument it, graduate to C only when data shows learned control helps.**
- **Observability & debugging (7.7):** memory systems are notoriously hard to debug (was it retrieval? write path? compression? reasoning?). Need full operation logging, replay tools, "memory diffs," regression tests for memory behavior. Absence of this infra is a primary reason demos fail to reach production.

---

## Open Challenges / Emerging Frontiers (Section 9)

Ten stated frontiers:

1. **Principled consolidation** — escape the hoarding-vs-amnesia oscillation via offline, sleep-like consolidation (hippocampal replay analogy). Concrete idea: **dual-buffer consolidation** — new memories sit in a "hot" buffer on probation, promoted to long-term only after re-verification / dedup / importance scoring (mirrors hippocampal-to-neocortical transfer).
2. **Causally grounded retrieval** — semantic similarity answers "what looks like this?" not "what caused this?" Blend similarity + temporal ordering + causal traversal + counterfactual relevance. Concrete starting point: a lightweight **causal metadata layer** annotating each record with an estimated causal parent at write time (LLM-generated, approximate).
3. **Multi-agent memory governance** — access control over shared stores, consensus for concurrent writes, knowledge transfer across specializations; distributed memory with merge semantics and hierarchical shared memory + per-agent caches remain open.
4. **Memory-efficient architectures** — sparse retrieval, compressed session vectors, memory-native architectures (Recurrent Memory Transformers), retrieval-free adapter injection — none has yet shown strong agent-level performance.
5. **Deeper neuroscience integration** — spreading activation for retrieval, reconsolidation theory for updates, Ebbinghaus curves + spaced repetition for reinforcement timing.
6. **Trustworthy reflection** — guard against entrenched confirmation bias via external validation, uncertainty quantification (confidence decay), adversarial probing of stored beliefs, expiration of unvalidated reflections.
7. **Foundation models for memory management** — a *task-agnostic* memory controller trained across diverse tasks to write/retrieve/summarize/forget/consolidate with general competence (as instruction-tuned LLMs gave general language). AgeMem is a first step; training data could be bootstrapped by having advanced LLMs retrospectively annotate which historical memory ops were helpful/harmful.
8. **Multimodal and embodied memory** — fuse text, vision, audio, proprioception, tool state; robotics adds spatial memory, real-time latency, cross-modal retrieval (find a visual memory via a textual query).
9. **Learning to forget** — forgetting is a feature, not a bug; learn selective forgetting policies maximizing long-term utility under safety/compliance constraints; tie to machine unlearning when memories influenced behavior.
10. **Standardized evaluation** — no community-standard harness exists; a **GLUE-style shared leaderboard** (conversational / agentic / multi-session tracks, standardized four-layer metrics) would accelerate progress.

**Conclusion theme:** the field traversed three generations — prompt-level compression → retrieval-augmented external stores → end-to-end learned policies. The closing call: *memory deserves the same engineering investment as the LLM itself* (model selection gets months of benchmarking; memory architecture often gets an afternoon) — treating memory as a first-class component may be the highest-leverage intervention for agent builders.

---

## Key Takeaways

1. **Single organizing primitive: the write–manage–read loop**, grounded in POMDPs, where memory is the agent's *belief state* (a sufficient statistic of history), not a database lookup.
2. **Three-axis taxonomy** = *temporal scope* (working / episodic / semantic / procedural) × *representational substrate* (context / vector / structured / executable / hybrid) × *control policy* (heuristic / prompted / learned).
3. **Five mechanism families:** context-resident compression, retrieval-augmented stores, reflective self-improvement, hierarchical virtual context, and policy-learned management (plus a parametric weight-based family).
4. **Evaluation has shifted** from static recall (Precision@k, nDCG) to multi-session agentic tests; a four-layer metric stack (effectiveness / memory quality / efficiency / governance) and the LoCoMo→MemBench→MemoryAgentBench→MemoryArena progression expose that *long context ≠ memory* and that forgetting is barely measured.
5. **Three generations, one thesis:** prompt compression → retrieval stores → learned policies, with the unifying argument that memory should be engineered as a first-class system component.

---

## Where it sits (v1/v2)

This is the **second, newer** survey in the collection (Mar 2026, single-author, Du) relative to **"Memory in the Age of AI Agents"** (Dec 2025 / arXiv Jan-2025 v2, 40+ authors, Hu/Liu et al.). They cover overlapping literature but organize it very differently.

**Taxonomy — how they differ:**

- **Dec-2025 survey** uses a **Forms / Functions / Dynamics** lattice: *Forms* = what carries memory (token-level / parametric / latent, sub-divided by topology and generation/reuse/transform); *Functions* = why (factual / experiential / working); *Dynamics* = how it operates (formation → evolution → retrieval). It is **representation-and-function-centric** and exhaustively catalogs system families per cell.
- **Mar-2026 survey** uses a **temporal-scope / representational-substrate / control-policy** lattice anchored in a **POMDP write–manage–read loop**, then commits to **five named mechanism families** examined in depth. It is **mechanism-and-engineering-centric**: heavier on design objectives/tensions, ablation evidence, a four-layer metric stack, and a production playbook (write/read paths, three architecture patterns, observability).

**Mapping & divergence:** the older survey's *Latent* and *Parametric* Forms are first-class top-level categories; the newer survey demotes parametric memory to a single subsection (4.6) and largely omits latent/KV-compression memory (which it brackets as LLM-internal, outside scope), reflecting its agent-systems rather than representation-learning framing. Conversely, the newer survey elevates **control policy** (heuristic → prompted → learned) to a full taxonomy axis and centers **AgeMem-style RL-learned management** as the third "generation," whereas the older survey treats RL-meets-memory as one frontier among eight. The newer survey's **POMDP/belief-state formalization**, **five-objective tension analysis**, **four-benchmark deep dive (LoCoMo/MemBench/MemoryAgentBench/MemoryArena)**, and **engineering-reality playbook** have no direct counterpart in the older one; the older survey's **open-source framework comparison** (MemGPT/Mem0/MemoryOS/MemOS/Zep/MIRIX) has no counterpart in the newer one.

**Where they agree on frontiers:** principled/continual **consolidation**, **trustworthy memory** (privacy, unlearning, robust/non-entrenching reflection), **learned forgetting**, **multimodal/embodied memory**, **multi-agent shared-memory governance**, and the move **from hand-crafted rules toward learned/foundation-model memory control**. Both also invoke **human-cognitive grounding** (Atkinson–Shiffrin, Tulving, Baddeley) and both argue **long context does not subsume memory**.

**Where they diverge on frontiers:** the older survey frames a distinct **"retrieval vs. generation"** axis (actively *synthesizing* memory on demand, e.g., MemGen/VisMem) and a **memory-for-world-models** direction — neither is a named frontier in the newer survey. The newer survey instead foregrounds **causally grounded retrieval**, a **foundation model for memory management**, and **standardized GLUE-style evaluation** as headline open problems that the older survey does not isolate. Net: the Dec-2025 survey is the broader, more comprehensive *map of representations and functions*; the Mar-2026 survey is the more focused, *systems-and-evaluation playbook* updated with the 2025–2026 learned-control and agentic-benchmark wave.
