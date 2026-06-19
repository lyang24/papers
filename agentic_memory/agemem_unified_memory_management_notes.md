# AgeMem: Learning Unified Long-Term and Short-Term Memory Management for LLM Agents

**Authors:** Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, Libing Wu (Wuhan University, Alibaba Group)

**Paper:** arXiv:2601.01885v2 (Apr 2026)

**GitHub:** Not provided in the paper (built on AgentScope + Trinity-RFT frameworks)

---

## The Core Problem

LLM agents are fundamentally bottlenecked by their finite context window. Memory splits into two complementary types:

- **Long-term memory (LTM):** persistent store of user/task knowledge — governs *what to store, update, or discard*.
- **Short-term memory (STM):** the information in the current input context — governs *what to retrieve, summarize, or remove from the active context*.

The problem: existing systems treat LTM and STM as **separate, loosely coupled modules**, each optimized independently and stitched together ad hoc. The paper frames prior architectures as two patterns (Figure 1):

- **(a) Static STM + trigger-based LTM** — LTM fires fixed memory operations at predefined moments (e.g., Mem0, Zep, MemoryBank); STM is a static context buffer optionally augmented by RAG.
- **(b) Static STM + agent-based LTM** — an auxiliary "memory manager" LLM decides what/how to store into LTM, but STM remains static.

Both depend on handcrafted rules or auxiliary expert models, leading to fragmented memory construction and suboptimal long-horizon reasoning. The authors identify three concrete challenges:

- **(C1) Functional heterogeneity coordination** — LTM and STM serve distinct but interdependent roles; need a unified mechanism that orchestrates their interplay.
- **(C2) Training paradigm mismatch** — LTM and STM are conventionally trained with different RL recipes, and standard RL assumes continuous trajectories with stable rewards, which conflicts with the *fragmented and discontinuous* experiences produced by memory operations.
- **(C3) Practical deployment constraints** — reliance on an auxiliary expert LLM for memory control inflates inference cost and training complexity.

---

## The Big Idea: Memory Operations as Tools in the Agent Policy, Trained End-to-End with RL

AgeMem's central move is to stop treating memory as an external pipeline and instead **fold memory management directly into the agent's own policy**. It does this by exposing **five (six) memory operations as tool-based actions** that the single LLM agent can invoke autonomously, then trains the whole thing end-to-end with reinforcement learning so that early storage decisions and later reasoning are optimized against the *same terminal task reward*.

Two design commitments distinguish it:

1. **Unified control over BOTH tiers.** A single policy `π_θ(a_t | s_t)` chooses from a hybrid action space mixing ordinary language generation with memory tool calls spanning persistent LTM *and* contextual STM. There is no separate memory-manager model (addresses C3).

2. **End-to-end RL with delayed credit assignment.** Storage, retrieval, filtering, and summarization are all learned jointly under a single delayed task reward via a step-wise GRPO variant (addresses C1, C2).

---

## Architecture

### The Six Memory Tools (Table 1)

The agent's action space is augmented with six structured tools. Three act on LTM, three on STM:

| Tool | Target | Function |
|---|---|---|
| **ADD** | LTM | Add new knowledge entry to the store `M_t` |
| **UPDATE** | LTM | Modify an existing entry (by `memory_id`) when new info supersedes old |
| **DELETE** | LTM | Remove a stale/incorrect entry from `M_t` |
| **RETRIEVE** | STM | Pull top-k semantically relevant memories from `M_t` into context `C_t` (k typically 3-5) |
| **SUMMARY** | STM | Compress a specified span of interaction history (LLM-based) to shrink context while preserving salient info |
| **FILTER** | STM | Remove context messages whose semantic similarity to a criterion exceeds threshold `θ_f`, suppressing distractors |

RETRIEVE uses cosine similarity between encoded query and memory embeddings. SUMMARY's `span` parameter can be "all" (all non-system messages) or "N" (last N messages). Together they give the agent expressive yet interpretable control over the full memory lifecycle.

### Unified RL Formulation

At each step `t`, the agent observes a state `s_t = (C_t, M_t, T)` composed of the short-term conversation context `C_t`, the long-term store `M_t`, and the task spec `T` (query `q`, contextual info `I_q`, and — training only — expected answer `A_q`). The policy selects `a_t` from the hybrid action space. The trajectory reward sums weighted task/memory-quality components plus a penalty term (Eq. 1), and the objective maximizes expected trajectory reward (Eq. 2).

### Three-Stage Trajectory / Curriculum

Each trajectory is split into three consecutive stages `τ = (τ^(1), τ^(2), τ^(3))`. **Crucially, `M_t` (LTM) persists across all three stages, while `C_t` (context) is reset before Stage 2** to prevent information leakage — forcing the agent to rely on genuine LTM retrieval rather than residual context.

- **Stage 1 — LTM construction.** Casual interaction with contextual info `I_q`; agent identifies salient information and stores it into LTM via ADD/UPDATE/DELETE.
- **Stage 2 — STM control under distractors.** Context is reset (LTM retained); agent is fed synthetically-generated distractor messages (via `DistractorGen`) and must FILTER/SUMMARY to suppress noise while preserving useful content.
- **Stage 3 — Integrated reasoning + coordination.** Agent receives a formal query, must RETRIEVE relevant LTM, manage `C_t`, and produce the final answer.

The authors stress the curriculum is **not tied to QA supervision** — it only needs a temporal separation between information exposure and task execution so the usefulness of memory decisions can be judged under delayed outcomes. Stage 2 distractors are generated synthetically and need no annotations.

### Step-wise GRPO (the credit-assignment trick)

Memory operations create **sparse and discontinuous rewards** (intermediate `r_t` is typically zero; reward arrives only at trajectory completion). To connect long-range task reward to earlier memory decisions, AgeMem adapts GRPO:

- For each task `q`, sample a group of `K` parallel rollouts; each yields a terminal reward `r_T = R(τ_k)`.
- Compute a **group-normalized advantage** at the terminal step (Eq. 5): `A_T = (r_T − μ_G)/(σ_G + ε)`.
- **Broadcast that terminal advantage to every preceding step of the same trajectory** (`A_t = A_T`), including Stage-1 storage and Stage-2 filtering actions.

This makes the final task outcome supervise *every* intermediate memory decision — long-range credit assignment across heterogeneous stages, without an explicit value function. The objective (Eq. 6) is the standard GRPO clipped/importance-weighted objective with a KL penalty to a reference policy.

The full training is a **three-stage progressive RL strategy**: the model first acquires LTM storage, then STM context management, then coordinates both under full task settings. (The paper's progression is: supervised/warm-up behaviors via the staged curriculum → task-level outcome reward → step-level GRPO dense credit assignment.)

### Composite Reward (Eq. 7)

`R(τ) = w^⊤ R + P_penalty`, with `w = [w_task, w_context, w_memory]`:

- **`R_task`** — primary signal; LLM-as-a-judge score `S_judge(A_pred, A_q) ∈ [0,1]`, dominant component.
- **`R_context`** (STM quality) — combines (i) compression efficiency (economical tokens), (ii) **preventive actions** (rewards *early* summarization/filtering to avoid overflow), (iii) information preservation (penalizes losing query-relevant content).
- **`R_memory`** (LTM quality) — combines (i) storage quality (fraction of stored entries judged high-quality/reusable), (ii) maintenance (rewards meaningful UPDATE/DELETE to fight staleness), (iii) semantic relevance of retrieved memories to the query.
- **`P_penalty`** — heavy penalties for context overflow or exceeding the interaction/turn limit.

---

## Experimental Results

### Setup

- **Benchmarks (5):** ALFWorld (SR), SciWorld (SR), PDDL (Progress Rate), BabyAI (SR), HotpotQA (LLM-as-judge). Covers embodied action, game reasoning, and knowledge-intensive QA.
- **Zero-shot transfer:** AgeMem is RL-fine-tuned **only on the HotpotQA training set** (which conveniently supplies Stage-1 supporting facts), then evaluated directly on all five benchmarks.
- **Backbones:** Qwen2.5-7B-Instruct, Qwen3-4B-Instruct.
- **Baselines:** No-Memory, LangMem, A-Mem, Mem0, Mem0g (graph variant), plus AgeMem-noRL (the tool framework without RL fine-tuning).

### Main Results (Table 2) — Qwen3-4B-Instruct

| Method | ALFWorld | SciWorld | PDDL | BabyAI | HotpotQA | **Average** |
|---|---|---|---|---|---|---|
| No-Memory | 38.51 | 47.89 | 30.14 | 55.83 | 47.48 | 43.97 |
| LangMem | 40.89 | 50.42 | 28.42 | 53.80 | 42.70 | 43.25 |
| A-Mem | 34.31 | 50.14 | 34.41 | 61.35 | 48.48 | 45.74 |
| Mem0 | 41.17 | 51.38 | 31.72 | 60.05 | 39.16 | 44.70 |
| Mem0g | 36.69 | 47.76 | 29.61 | 57.59 | 38.12 | 41.95 |
| AgeMem-noRL | 38.02 | 50.42 | 27.52 | 57.48 | 54.49 | 45.59 |
| **AgeMem (Ours)** | **48.97** | **59.48** | **35.07** | **72.56** | **55.49** | **54.31** |

AgeMem hits **54.31%** average, beating the best baseline A-Mem (45.74%) by **+8.57 pts**, and beating Mem0 (44.70%) by +9.61. RL adds **+8.72 pts** over AgeMem-noRL (45.59%), and the gain over No-Memory is +23.52% relative.

### Main Results (Table 2) — Qwen2.5-7B-Instruct

| Method | ALFWorld | SciWorld | PDDL | BabyAI | HotpotQA | **Average** |
|---|---|---|---|---|---|---|
| No-Memory | 27.16 | 13.80 | 10.15 | 50.80 | 38.36 | 28.05 |
| LangMem | 38.27 | 28.29 | 15.85 | 51.34 | 37.43 | 34.23 |
| A-Mem | 34.68 | 28.06 | 18.39 | 58.82 | 43.95 | 36.78 |
| Mem0 | 37.49 | 26.99 | 13.96 | 60.58 | 46.66 | 37.14 |
| Mem0g | 35.34 | 30.50 | 14.86 | 58.78 | 42.06 | 36.31 |
| AgeMem-noRL | 37.90 | 28.67 | 8.87 | 46.34 | 45.36 | 33.43 |
| **AgeMem (Ours)** | **41.07** | **35.55** | **17.31** | **61.42** | **54.44** | **41.96** |

AgeMem reaches **41.96%** average, beating the best baseline Mem0 (37.14%) by **+4.82 pts**; RL adds **+8.53 pts** over AgeMem-noRL (33.43%); +49.59% relative over No-Memory.

### Memory Quality (Figure 2)

Using HotpotQA ground-truth facts, an LLM evaluator scores relevance between stored memories and facts (MQ). AgeMem achieves the **highest MQ on both backbones: 0.533 (Qwen2.5-7B) and 0.605 (Qwen3-4B)** — i.e., the tool-based, RL-optimized policy stores more selective, reusable knowledge.

### STM / Context Efficiency (Figure 3)

Measured by average prompt token count on HotpotQA, vs. a "-RAG" variant that replaces STM tools with plain RAG:

- Qwen2.5-7B: AgeMem **2,117** tokens vs. AgeMem-RAG 2,186 — **3.1% reduction**.
- Qwen3-4B: AgeMem **2,191** tokens vs. AgeMem-RAG 2,310 — **5.1% reduction**.

Learned STM tools control context expansion better than static RAG while maintaining task performance.

### Tool Usage Analysis (Table 3) — Emergent Tactics

Average tool calls per episode, before (noRL) vs. after (GRPO) RL:

| Tool | Qwen2.5 noRL | Qwen2.5 GRPO | Qwen3 noRL | Qwen3 GRPO |
|---|---|---|---|---|
| ADD (LTM) | 0.92 | 1.64 | 2.49 | 2.64 |
| UPDATE (LTM) | 0.00 | 0.13 | 0.13 | 0.34 |
| DELETE (LTM) | 0.00 | 0.08 | 0.00 | 0.22 |
| RETRIEVE (STM) | 2.31 | 1.95 | 4.62 | 4.35 |
| SUMMARY (STM) | 1.08 | 0.82 | 0.11 | 0.96 |
| FILTER (STM) | 0.02 | 0.31 | 0.15 | 0.16 |
| **Total calls** | 4.33 | 4.92 | 7.50 | 8.67 |

The **emergent behaviors after RL**:

- **More proactive LTM construction & maintenance** — ADD rises (0.92→1.64 on Qwen2.5), and UPDATE/DELETE emerge from ~zero (the agent learns to *maintain* and prune the store rather than just append).
- **Proactive context control** — FILTER frequency jumps notably (0.02→0.31 on Qwen2.5), reflecting deliberate discarding of distracting/redundant records; SUMMARY is used preventively before context fills (rewarded via the "preventive actions" term in `R_context`).
- **RETRIEVE becomes *more selective, not less trained*** — retrieval frequency drops (Qwen2.5: 2.31→1.95; Qwen3: 4.62→4.35). The paper argues this is a **qualitative shift**: pre-RL the agent retrieves reactively/repeatedly to compensate for poor Stage-1 storage; post-RL, better ADD/UPDATE means stored knowledge is higher-quality, so retrieval becomes query-driven and used only when genuinely needed. The drop coincides with *higher* task performance and MQ — efficiency, not under-training.

---

## Ablation Studies

### LTM / STM / RL components (Figure 4, Qwen2.5-7B, three representative datasets)

Progressive build-up over the No-Memory base:

- **+LT (LTM tools only, no RL):** +10.6% / +14.2% / +7.4% over baseline.
- **+LT/RL (RL with LTM tools):** further gains, notably +6.3% on HotpotQA.
- **+LT/ST/RL (full AgeMem):** best across all benchmarks, overall +13.9% / +21.7% / +16.1%.
- Adding STM tools gives the biggest marginal boost on SciWorld (+3.1%) and HotpotQA (+2.4%), confirming learned context management beats static RAG.

### Reward function (Table 4 + Figure 5, Qwen2.5-7B on HotpotQA)

Full composite reward (**All-Returns**) vs. task-only (**Answer-Only**):

| Strategy | Judge (↑) | Tokens (↓) | MQ (↑) | Tool calls |
|---|---|---|---|---|
| Answer-Only | 0.509 | 2078 | 0.479 | 3.93 |
| **All-Returns** | **0.544** | 2117 | **0.533** | 4.92 |

The full reward converges faster and reaches higher final performance and memory quality; the small extra token cost buys meaningfully better reasoning. The multi-component reward (task + context + memory) is essential, not just the task signal.

### FILTER threshold `θ_f` sensitivity (Table 5, HotpotQA)

| `θ_f` | Judge (↑) | MQ (↑) | Avg. Tokens |
|---|---|---|---|
| 0.4 | 0.524 | 0.511 | 2089 |
| 0.5 | 0.551 | 0.550 | 2116 |
| 0.6 | 0.544 | 0.533 | 2117 |
| 0.7 | 0.530 | 0.526 | 2149 |
| 0.8 | 0.531 | 0.510 | 2134 |

Performance is **stable across `θ_f ∈ [0.4, 0.8]`** — AgeMem is not sensitive to precise threshold tuning. Too low → over-aggressive filtering discards useful context; too high → marginal context leaks through and slightly degrades MQ. Token counts stay similar, so the effect is selection *quality*, not length.

---

## Key Takeaways

1. **Memory-ops-as-tools, unified across both tiers.** AgeMem's defining contribution is exposing the full memory lifecycle — store/retrieve/update/summarize/discard (ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER) — as tools in *one* agent policy that controls LTM and STM jointly, rather than two separate modules glued together.

2. **End-to-end RL ties early storage to final reasoning.** The biggest lever is RL fine-tuning (+8.5-8.7 pts average over the same tool framework without RL). Step-wise GRPO broadcasts the terminal advantage to every prior memory decision, so Stage-1 ADD and Stage-2 FILTER are optimized against the actual downstream task outcome — solving the sparse/discontinuous reward problem memory ops create.

3. **The three-stage persistent-LTM / reset-STM curriculum** forces genuine memory use: by wiping context before the task but keeping LTM, the agent cannot cheat with residual context and must learn proper store-then-retrieve behavior. The curriculum depends only on information *timing*, not QA-style supervision, so it generalizes.

4. **RL induces sensible emergent tactics** — proactive SUMMARY/FILTER before context fills, discarding of semantically-redundant records (UPDATE/DELETE emerge from zero), and a shift from reactive over-retrieval to selective query-driven retrieval, all while task accuracy *and* memory quality go up.

5. **Better quality at lower cost** — highest memory quality (MQ 0.533/0.605) and reduced prompt tokens (3-5% below RAG) alongside the accuracy wins, with strong zero-shot cross-domain transfer (trained on HotpotQA only).

---

## Limitations (Acknowledged by Authors)

1. **Fixed tool set.** The six-tool abstraction is clean and effective but could be extended to finer-grained memory control.
2. **Controlled evaluation.** Five benchmarks demonstrate cross-domain transfer but remain relatively controlled vs. open-ended real-world deployment; persistent long-term dialogue and real-user interaction are not yet tested.
3. **Single training source.** Trajectories are derived solely from HotpotQA; richer interaction structures from other data sources would broaden applicability.

---

## Where it sits (v1/v2)

**This is a v2 "RL-memory" paper** — and arguably the **purest "memory-ops-as-tools + RL" instance** of the family. Whereas v1 memory systems are *engineered* pipelines (heuristics, schedules, knowledge-graph builders, auxiliary manager LLMs), v2 systems make memory management a *learned policy*. AgeMem pushes this furthest by putting the entire memory lifecycle for **both tiers** into a single agent's action space and optimizing it end-to-end with step-wise GRPO.

- **Group with:** Memory-R1 (Yan et al. 2025), Mem-α, MemCoE, LatentMem, and Memory-as-Action (Zhang et al. 2025a) — all RL-trained memory agents. AgeMem explicitly positions against Memory-R1 / Memory-as-Action, arguing those optimize *one aspect* of memory at a time (e.g., LTM ops) while leaving retrieval/summarization/STM as fixed heuristics or separately-tuned modules — so early storage and later reasoning stay loosely coupled. AgeMem's claim to novelty is the **single unified learnable control problem** under delayed supervision spanning persistent LTM ops *and* contextual STM ops.

- **Contrast with separate-tier / engineered designs:** MemoryOS and similar tiered architectures impose an explicit LTM/STM/manager hierarchy with hand-designed promotion/eviction policies; Mem0/Zep/A-Mem are trigger- or agent-based LTM builders with static STM (RAG). AgeMem's distinguishing bet is **UNIFIED LTM+STM control inside the agent policy, learned rather than scheduled** — no auxiliary expert model, no fixed memory schedule, with the two tiers co-optimized against the same terminal reward.
