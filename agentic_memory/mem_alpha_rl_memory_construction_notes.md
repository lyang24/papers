# Mem-α: Learning Memory Construction via Reinforcement Learning

**Authors:** Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu (Anuttacon, UC San Diego, Stanford)

**Paper:** arXiv:2509.25911v1 (Sep 2025)

**GitHub / Resources:** Datasets, Models, and Source Code released (links in paper header; project under Anuttacon)

---

## The Core Problem

Memory-augmented LLM agents are equipped with rich, multi-component memory systems (e.g., Mem0, MemGPT, MIRIX) that expose **memory update tools** — insert facts, update summaries, store events. But these systems assume the base model can use those tools effectively **out-of-the-box**, driven purely by system prompts. In practice, language models lack the inherent ability to decide:

1. **What** information to store (which details matter for future questions)
2. **How** to structure it (which memory component — core summary, semantic fact, episodic event)
3. **When** to update it (insert vs. update vs. delete; consolidate vs. append)

This problem gets worse as the memory architecture grows more complex. Long, hand-tuned system prompts can only partially mitigate it, and they cannot cover all scenarios. Crucially, **smaller models with weak instruction-following are completely overwhelmed** by complex tool sets, and even GPT-4o-class models update complex memory systems poorly. The result is **suboptimal memory construction and information loss**.

The natural fix — supervised fine-tuning — fails because there are **no ground-truth memory construction traces**: no existing model produces reliable enough memory write sequences to supervise against. This motivates reinforcement learning, where the agent discovers good write policies by **trial and error**, optimizing directly for downstream task success rather than imitating a (nonexistent) gold trajectory.

---

## The Big Idea: Learn the Memory WRITE Policy with RL

Mem-α frames **memory construction as a sequential decision-making problem** and trains the agent's **write policy** with reinforcement learning. The defining choices:

- **Only the write/construction policy is learnable.** Retrieval and answer generation are a fixed, decoupled RAG pipeline (BM25 retriever + frozen generator). All learning pressure goes into *how the agent builds memory*, not how it reads or answers.
- **The reward is the downstream QA accuracy over the full interaction history.** Memory is judged purely by whether the final memory state lets a frozen reader answer questions correctly — an outcome reward that directly optimizes for "store the right things, organized the right way."
- **A genuinely complex, multi-component memory** (core / episodic / semantic) with a real tool set (insert / update / delete) — not the single-paragraph or flat fact-list memories used by prior RL-memory work.
- **Length generalization is the headline result:** trained only on instances ≤30K tokens (averaging <20K), the learned policy generalizes to sequences **exceeding 400K tokens (up to 474K) — over 13× the training length** — evidence that RL teaches *general* memory-management principles rather than memorizing patterns.

The contrast with prior memory systems: instead of *providing tools and hoping the model uses them*, Mem-α *trains the model to use them*.

---

## Architecture

### 1. The Memory Instantiation (core / episodic / semantic)

The memory system Mem-α learns to operate has three complementary components, each with distinct update semantics:

| Component | What it stores | Form | Allowed operations |
|---|---|---|---|
| **Core Memory** | The most critical, always-in-context summary | Single persistent paragraph, **max 512 tokens** (follows MemGPT) | `update` only — requires holistic rewrite to stay coherent |
| **Semantic Memory** | Factual / declarative knowledge about the world and user | Expandable list of discrete atomic factual statements | `insert`, `update`, `delete` |
| **Episodic Memory** | Temporally-grounded events and experiences | Chronologically-organized list of **timestamped** events | `insert`, `update`, `delete` |

The design reflects different update patterns: semantic and episodic memories benefit from **incremental, fine-grained edits**, while core memory needs **holistic revision** to preserve summary quality. The memory architecture is **modular and decoupled from the RL framework** — researchers can swap in a simpler or more complex memory (e.g., MIRIX) without touching the training method.

### 2. The RL Task Setup (sequential memory construction)

The agent processes a sequence of conversation chunks `C = {c₁, ..., cₙ}` spanning diverse formats (casual discussion, storytelling, book sharing, classification examples). At each step *t*:

- It observes chunk `cₜ` and the current memory `M_{t−1}` (M₀ = empty).
- It issues an **action** = a sequence of write operations `aₜ = (aₜ⁽¹⁾, ..., aₜ⁽ᴷ⁾)`, where each operation is a structured function call from `A_write = {memory_insert, memory_update, memory_delete}` with arguments (record id, memory type, content).
- Operations are applied sequentially to transform `M_{t−1}` into `Mₜ`, then the agent advances to the next chunk.

After all chunks are processed, the **terminal memory `Mₙ`** and the full action sequence `A = {a₁, ..., aₙ}` are scored. Because each chunk triggers a distinct write action, instances yield **long action sequences** — the core of what makes this a long-horizon RL problem.

### 3. Decoupled RAG Evaluation (the reward channel)

To score memory quality, a **fixed, non-learnable** RAG pipeline answers evaluation questions from `Mₙ`:

1. **Retrieval** — a fixed **BM25** retriever selects the top-k entries from semantic and episodic memory pools for each question.
2. **Generation** — a **frozen generator** produces an answer from the question + retrieved support.
3. **Scoring** — predicted answers are compared to references, inducing the correctness reward.

Keeping retrieval/generation frozen ensures all gradient signal shapes the **write policy** alone.

### 4. The Composite Reward

The final per-action reward combines four components: `rₜ = r₁ + r₂,ₜ + β·r₃ + γ·r₄,ₜ`

| Reward | Name | What it measures | Granularity |
|---|---|---|---|
| **r₁** | Correctness | Downstream QA accuracy over Mₙ (dataset-specific metric, e.g. SubEM, EM, LLM-judge, keyword-hit) | Global (shared across all actions) |
| **r₂** | Tool Call Format | Fraction of function calls that are well-formed and execute successfully | Per-action |
| **r₃** | Compression | `1 − l_m / l_c` (memory length vs. total chunk length) — rewards compact memory | Global |
| **r₄** | Memory Content | Fraction of operations judged **semantically valid** by Qwen3-32B (does the op satisfy its definition) | Per-action |

`r₂` weight is fixed at 1.0 because function-call success is critical; `r₁` weight is also 1.0; only **β (compression)** and **γ (content quality)** are tuned. Default: **β = 0.05, γ = 0.1**.

### 5. Policy Optimization

Mem-α uses **GRPO (Group Relative Policy Optimization)**. The advantage is the group-normalized reward `Aₜ = (rₜ − μ_group) / (σ_group + ε)`, and the objective is the standard clipped GRPO surrogate maximized over all actions in the sequence. **The KL term is dropped to encourage exploration.**

**Training setup:** `verl` framework, **Qwen3-4B** backbone (Qwen3-8B tried but performed worse), 32× H100 GPUs, lr 1e-6, batch size 32, GRPO rollout n = 8, ~3 days, 205 steps, best checkpoint by validation.

### 6. Training Dataset

A purpose-built dataset of **4,139 instances** (stratified-sampled to a balanced **562-instance** subset for RL due to compute cost + class imbalance), drawn from 8 sources across three of MemoryAgentBench's four capability dimensions (Conflict Resolution excluded — no realistic benchmark):

| Category | Datasets | Targets |
|---|---|---|
| **Accurate Retrieval (AR)** | SQuAD, HotpotQA, PerLTQA, LongMemEval-Train | Store + precisely retrieve single/multi-hop facts |
| **Test-Time Learning (TTL)** | NLU, TREC-Coarse, PubMed-RCT | Learn new classification patterns from in-context examples (labels replaced with numeric 0–4) |
| **Long-Range Understanding (LRU)** | BookSum | Integrate info across many segments (summarization) |

LongMemEval-Train was built from 200 oracle questions, concatenating haystack dialogues into 10K–30K-token contexts (50 samples), **with no overlap** against the MemoryAgentBench eval set.

---

## Experimental Results

Two evaluation regimes: **validation** (matches training distribution) and **out-of-distribution test** = MemoryAgentBench. All methods use BM25 retrieval + Qwen3-32B as the reader. **"Mem."** = total memory size in thousands of tokens.

### Validation Datasets (in-distribution)

Baselines: **Long-Context** (Qwen3-32B, 32K window), **RAG-Top2** (BM25 top-2 + Qwen3-32B), **MemAgent**, **MEM1**. Mem-α uses Qwen3-4B.

| Method | SQuAD (AR) | HotpotQA (AR) | PerLTQA (AR) | TREC-C (TTL) | NLU (TTL) | PubMed (TTL) | BookSum (LRU) | **Avg.** |
|---|---|---|---|---|---|---|---|---|
| Long-Context | 0.742 | 0.852 | 0.605 | 0.623 | 0.708 | 0.533 | 0.052 | 0.588 |
| RAG-Top2 | 0.762 | 0.849 | 0.623 | 0.612 | 0.508 | 0.570 | 0.042 | 0.567 |
| MemAgent | 0.091 | 0.140 | 0.052 | 0.562 | 0.290 | 0.343 | 0.103 | 0.236 |
| MEM1 | 0.039 | 0.083 | 0.068 | 0.269 | 0.056 | 0.175 | 0.085 | 0.111 |
| **Mem-α (4B)** | **0.786** | **0.832** | **0.659** | **0.666** | **0.658** | **0.545** | **0.187** | **0.642** |

- Mem-α (a 4B model) **beats Long-Context (0.642 vs 0.588) and RAG-Top2 (0.567)**, both using a 32B reader over raw context.
- The **flat-memory RL baselines collapse** — MemAgent (0.236) and MEM1 (0.111) use single-paragraph memory, validating the structured core/episodic/semantic design.
- Memory footprint: Mem-α uses ~7.9K avg tokens vs ~10.8K (Long-Context) / 11.3K (RAG-Top2) — roughly **a ~50% reduction** in stored tokens versus raw-context baselines.

### MemoryAgentBench (out-of-distribution test) — Length Generalization

This is where length generalization shows up: test documents run up to **474K tokens** (Multi-Doc), vs. ≤30K seen in training.

| Method | Single-Doc (AR) | Multi-Doc (AR) | LME(S) (AR) | TREC-C (TTL) | NLU (TTL) | TREC-F (TTL) | Clinic (TTL) | Banking77 (TTL) | InfBench (LRU) | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|---|
| Long-Context | 0.280 | 0.270 | 0.292 | 0.640 | 0.740 | 0.340 | 0.860 | 0.770 | 0.125 | 0.461 |
| RAG-Top2 | 0.690 | 0.450 | 0.581 | 0.690 | 0.650 | 0.210 | 0.700 | 0.750 | 0.065 | 0.502 |
| MemAgent | 0.070 | 0.160 | 0.050 | 0.370 | 0.260 | 0.210 | 0.250 | 0.370 | 0.043 | 0.198 |
| MEM1 | 0.070 | 0.180 | 0.090 | 0.180 | 0.000 | 0.000 | 0.090 | 0.000 | 0.029 | 0.071 |
| **Mem-α (4B)** | **0.740** | **0.680** | **0.520** | 0.710 | 0.710 | 0.410 | 0.730 | 0.700 | **0.129** | **0.592** |

- Mem-α leads overall (**0.592 vs 0.502 RAG-Top2 / 0.461 Long-Context**), with the biggest gains on **Accurate Retrieval** and **Long-Range Understanding**.
- **Multi-Doc** holds at 0.680 even though instances reach **474K tokens — >13× the training length** — while Long-Context (0.270) and RAG-Top2 (0.450) degrade badly. This is the central length-generalization claim.
- Memory compression is dramatic on long inputs: e.g. Multi-Doc memory 134K (Mem-α) vs 474K (RAG-Top2's stored chunks); InfBench 19K vs 181K.

### RL is the source of the gains (not just the architecture)

Same memory framework, three policies — base Qwen3-4B (untrained), GPT-4.1-mini, and RL-trained Mem-α:

| Policy (same memory framework) | SQuAD | HotpotQA | PerLTQA | TREC-C | NLU | PubMed | BookSum | **Avg.** |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B (untrained) | 0.338 | 0.637 | 0.557 | 0.416 | 0.381 | 0.281 | 0.130 | 0.389 |
| gpt-4.1-mini | 0.426 | 0.749 | 0.492 | 0.637 | 0.519 | 0.544 | 0.246 | 0.517 |
| **Qwen3-4B w/ Mem-α (RL)** | **0.786** | **0.832** | **0.659** | **0.666** | **0.658** | **0.545** | **0.187** | **0.642** |

- The **untrained Qwen3-4B on the same memory scores only 0.389** — below RAG-Top2 and Long-Context. The architecture alone is not enough.
- RL lifts the *same* 4B model from **0.389 → 0.642**, **surpassing even gpt-4.1-mini (0.517)**. The gain is the RL-learned write policy, not the memory structure.

---

## Ablation: Reward Component Sensitivity (β, γ)

| β | γ | SQuAD | HotpotQA | PerLTQA | TREC-C | NLU | PubMed | BookSum | **Avg.** | BookSum Mem. |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.0 | 0.701 | 0.802 | 0.652 | 0.423 | 0.542 | 0.501 | 0.183 | 0.543 | 4.9K |
| 0.0 | 0.1 | 0.817 | 0.853 | 0.678 | 0.605 | 0.629 | 0.572 | 0.183 | 0.630 | 4.5K |
| **0.05** | **0.1** | 0.786 | 0.832 | 0.659 | 0.666 | 0.658 | 0.545 | 0.187 | **0.642** | **2.2K** |
| 0.2 | 0.1 | 0.822 | 0.838 | 0.615 | 0.558 | 0.176 | 0.401 | 0.193 | 0.525 | 3.0K |
| 0.4 | 0.1 | 0.691 | 0.810 | 0.533 | 0.475 | 0.405 | 0.455 | 0.201 | 0.509 | 1.5K |

Two key findings:

1. **The memory content reward (γ) is critical.** Setting **γ = 0 causes catastrophic degradation** — the model fails to learn meaningful write strategies, producing disorganized memory that can't support downstream tasks. (Note: the paper's text references this, while the β=0.05/γ=0.0 row above still scores 0.543; the γ-content signal is what disciplines *how* operations are formed.)
2. **The compression reward (β) is task-dependent.** Increasing β shrinks memory but eventually hurts performance (β=0.2 → 0.525, β=0.4 → 0.509). The chosen **β=0.05, γ=0.1** best balances efficiency and accuracy — e.g. it cuts BookSum memory to **2.2K vs 4.5K** (β=0) while holding accuracy roughly constant elsewhere.

---

## Case Study: How the Learned Policy Writes Better

On a "condo advice" conversation, comparing how three policies populate the same memory:

- **Untrained Qwen3-4B:** leaves **core memory empty**, stores only **one** generic semantic entry — severe information loss as distinct concepts collapse into one statement.
- **GPT-4.1-mini:** good semantic organization (3 distinct entries), but **inefficient episodic memory** — creates multiple events with **identical timestamps** that should be merged, and **only records user behavior, ignoring assistant responses**.
- **Mem-α (RL-trained):** maintains an informative core summary, organizes semantic facts into distinct entries, **consolidates same-timestamp episodic events into a single comprehensive entry**, and captures **both user and assistant behavior**. Better organization → more information retained per token.

This is the qualitative signature of a *learned* write policy: consolidation, completeness, and component-appropriate routing that prompting alone does not reliably produce.

---

## Key Takeaways

1. **Train the write policy, freeze the read path.** By making only memory construction learnable and routing reward through a fixed RAG reader, Mem-α isolates and directly optimizes the hardest, least-supervisable part of memory agents — *what to store and how to structure it*.

2. **Outcome reward over QA accuracy is enough to teach "what matters."** No gold memory traces are needed; downstream correctness (plus tool-format, compression, and content-validity shaping rewards) is sufficient signal for RL to discover good construction strategies.

3. **Structured memory + RL beats flat memory + RL decisively.** MemAgent/MEM1 (single-paragraph) collapse (0.111–0.236 avg); Mem-α's core/episodic/semantic design reaches 0.642 — but only *with* RL (untrained on the same structure: 0.389).

4. **A 4B model, once RL-trained, beats GPT-4.1-mini and 32B long-context/RAG baselines** on memory construction — learned policy > bigger model with prompted policy.

5. **RL induces transferable memory principles, not memorized patterns.** Trained on ≤30K tokens, the policy generalizes to **>400K tokens (up to 474K, >13×)**, the paper's strongest evidence that the agent learned *general* memory management.

---

## Limitations (Acknowledged by Authors)

1. **Conflict Resolution is out of scope** — the framework targets only three of four MemoryAgentBench dimensions; contradiction handling / overwriting stale knowledge was excluded due to a lack of realistic (non-synthetic) benchmarks.
2. **Simulated, decoupled setup** — retrieval and generation are frozen and memory lives in a simulated pipeline; moving to real databases and production systems introduces latency, scalability, and safety challenges not addressed here.
3. **Memory architecture could be richer** — the authors suggest integrating more sophisticated systems (e.g., MIRIX) for additional structural advantages on complex reasoning.
4. **RL cost** — training requires 32× H100 GPUs for ~3 days even on a stratified 562-instance subset; the full 4,139-instance set was too expensive to train on directly.
5. **Backbone sensitivity** — Qwen3-8B underperformed Qwen3-4B, hinting the recipe is not yet robust across model scales.

---

## Where it sits (v1/v2)

This is a **v2 "RL-meets-memory" paper**: rather than hand-designing a memory architecture and retrieval heuristic (v1, e.g. MAGMA's multi-graph + adaptive traversal), Mem-α **learns the memory policy with reinforcement learning** and an outcome reward.

Within the RL-memory frontier, the key distinction is **which part of the memory loop is learned**:

- **Mem-α — learns WHAT TO WRITE (memory construction).** It trains the *write/insert/update/delete* policy over a structured core/episodic/semantic memory; retrieval and answer generation stay fixed. The contribution is teaching the agent which information to store and how to organize it across components.
- **Memory-R1 — learns to MANAGE and USE memory.** It focuses on memory *management* (manage operations) and *utilization* (using retrieved memory to answer), on shorter LoCoMo-style settings with a simpler memory representation.
- **MemCoE — learns WHAT + HOW (construction and the structure/retrieval behavior jointly).** It spans both what to store and how the memory is shaped/used.

Grouped under the survey's **"RL meets memory"** frontier, these three are complementary slices of the same problem: Mem-α owns **construction (the write side)**, Memory-R1 owns **management + use (the read side)**, and MemCoE bridges **both**. Mem-α's standout contribution to this frontier is demonstrating that an RL-learned write policy generalizes far beyond its training length (>13×), suggesting learned construction captures general principles rather than dataset-specific patterns.
