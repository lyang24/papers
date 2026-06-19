# AtomMem: Learnable Dynamic Agentic Memory with Atomic Memory Operation

**Authors:** Yupeng Huo, Yaxi Lu, Zhong Zhang, Haotian Chen, Yankai Lin (Renmin University of China; Tsinghua University)

**Paper:** arXiv:2601.08323v3 (Mar 2026)

**GitHub:** https://github.com/RUCBM/AtomMem

---

## The Core Problem

Most existing LLM-agent memory mechanisms rely on **static, expert-crafted workflows** — memory operations are confined to predefined pipelines (e.g., "update your memory with new information every step", or exponential forgetting schedules) rather than decided autonomously by the model. The authors call this the implicit **"one-size-fits-all" assumption**: a fixed rule that works in generic scenarios but fails in complex ones.

Concretely:

1. A continuous memory-fusion or mandatory "update-at-every-step" routine ignores **information density** — it forces redundant updates even when new data is sparse.
2. Predefined forgetting schedules (e.g., exponential decay) may **prematurely discard early-but-critical cues** needed for long-horizon reasoning.
3. Workflows hand-tuned for one task family (e.g., QA) **don't transfer** to another (e.g., interactive web agents).

The illustrative failure (Figure 1): the *same* static workflow preserves the important "user's initial goal" in Task A but discards it in Task B, because the rule cannot adapt to what each task actually needs. The paper's framing: memory management should be a **decision-making problem the model learns**, not a fixed mechanism.

The authors group prior work into:
- **Static workflows** — *Imitation-based* (MemoryBank likens agent memory to human memory; MemGPT likens context to an OS) and *Prior-based* (hand-crafted expert pipelines like Mem0, A-Mem, MemoRAG). Common flaw: the workflow is hard-coded.
- **RL-in-memory** — *Summarization-based* (MemAgent, Mem1 use step-wise overwriting summaries, but are locked into a mandatory update-at-every-step routine) and *Heuristic-tool-based* (Memory-as-Action, AgentFold add tools like context pruning/folding, but the tool designs still embed manual priors).

AtomMem's contrast: give the model **only the most atomic memory operations** and let RL learn the policy, maximally exposing the memory-as-decision-making paradigm.

---

## The Big Idea: Memory as a Learnable CRUD Decision Process

AtomMem reframes memory management as a **dynamic decision-making problem**. It deconstructs high-level memory workflows into their **fundamental atoms — the standard CRUD primitives (Create, Read, Update, Delete)** — and then learns, via reinforcement learning (GRPO), an autonomous task-aligned policy that decides *when* to store, retrieve, revise, or remove information based on the current situation.

The slogan from Figure 1: replace many human-designed rules with **"only one rule: make your own decision based on the current situation."**

### Why CRUD as the atomic action space

The choice of CRUD rests on three argued properties:

| Property | Meaning |
|---|---|
| **Completeness** | CRUD is a universal operator set — any valid memory state is reachable from the current state via a sequence of CRUD ops. Under ideal optimization, memory can reach its maximal performance potential. |
| **Atomic Minimality** | Any higher-level memory *tool* (pruning, folding, fusion, summarization) can be expressed as a structured *combination* of CRUD ops; CRUD primitives themselves cannot be decomposed further. |
| **Task-agnosticness** | CRUD is not tied to any downstream task. Its task-completion ability rests entirely on the LLM's per-step decision capability — which is itself optimizable — making it a candidate foundation for general-purpose agent memory. |

The framing's key advantage: memory effectiveness is **no longer bounded by expert rules**, but by the model's own decision-making capacity, which RL can improve.

---

## Architecture

### Memory as a POMDP

Memory management is modeled as a Partially Observable Markov Decision Process (S, A, P, Ω, O, R, γ), where memory is an explicitly **controllable component of the environment**:

- **Global state** `s_t = (s_t^env, s_t^mem)` — external environment + internal memory state.
- **Action** `a_t = (a_t^env, a_t^mem)` — a *joint* decision: a task action (e.g., web search) plus a memory-management action from the atomic CRUD space.
- **Transition** `P(s_{t+1}|s_t, a_t)` — crucially, the internal memory state `s_{t+1}^mem` is directly modified by the agent's own actions.
- **Observation** `o_t = (o_t^env, o_t^mem)` — memory is **not fully observable**: `o_t^mem` is determined by previous memory actions (e.g., what a Read retrieved), making memory access an **explicit decision variable**.

Memory is treated as part of the environment and **reset at the start of each task** (`s_0^mem = ∅`) — this is *within-task* working memory, deliberately distinguished from cross-task experience accumulation (ExpEL/Memp style).

### Atomic memory operations (the evolution mechanism)

Memory at step *t* is a **dynamic set** `M_t = {m_i}_{i=1..N_t}`, where each `m_i` is a stored entry. The learnable action space is:

```
A_mem = {Create, Read, Update, Delete}
```

Each primitive (except Read) is a **state-transition operator over M_t**. The concrete realization (Appendix Table 5) — each op is a structured XML token in the model's vocabulary:

| Op | XML schema | Effect on the store |
|---|---|---|
| **Create** | `<create_memory>{content}</create_memory>` | Insert new content as a standalone entry into the vector DB |
| **Read** | `<read_memory>{query}</read_memory>` | Retrieve top-k relevant entries (does **not** alter state) |
| **Update** | `<update_memory>{memory id: content}</update_memory>` | Selectively modify an existing entry by its identifier |
| **Delete** | `<delete_memory>memory id</delete_memory>` | Permanently remove an entry by identifier |

**Compositional macro-actions.** At each decision step, conditioned on observation `o_t`, the policy emits a *sequence* of memory actions `A_t = {a_t^1, ..., a_t^K}` — a compositional macro-action within a single environment step (paired XML tags let the model, e.g., add several memories or update-then-delete in one response). Non-read operations are executed sequentially, yielding a **composed transition**:

```
M_{t+1} = a_t^K ∘ ... ∘ a_t^1 (M_t),   where each a_t^k ∈ {Create, Update, Delete}
```

This is the **fine-grained dynamic consolidation** mechanism: the store evolves entry-by-entry through atomic edits the policy chooses, rather than via a fixed densification/forgetting schedule.

### Hybrid (dual-path) memory retrieval

Read produces a memory observation (the content requested at step *t−1* becomes the observation at step *t*). Retrieval combines two paths, ranking information by importance:

- **Deterministic retrieval (Scratchpad).** A special entry `m_t^scr` is retrieved at **every** step. It captures global task state and pivotal step-wise information — functionally identical to other entries, differing only in its mandatory retrieval schedule. (Updated via `<update_scratchpad>`.) This is short-term/working memory.
- **Selective retrieval (Vector DB).** The agent emits a textual query `q_t`, and entries are retrieved by semantic similarity: `M̂_t = TopK({m_i ∈ M_t | sim(q_{t-1}, m_i)})`. This is long-term store.

Unified observation: `o_t = {o_t^env, m_t^scr, M̂_t}`. Critical information lives in the scratchpad; potentially useful detail lives in the vector DB — the paper finds this multi-path setup improves both effectiveness and robustness.

### Learning / optimization (GRPO)

Because memory operations are realized as **structured tokens in the vocabulary**, optimizing output-sequence likelihood *implicitly optimizes the memory policy* — no external memory-controller module is needed.

- **Algorithm:** Group Relative Policy Optimization (GRPO), Dr.GRPO variant (no advantage normalization). Fully **on-policy** (each rollout used for a single update).
- **Reward:** terminal-only (no intermediate rewards). For QA tasks → **exact match (EM)** vs ground truth; for web tasks → **LLM-as-a-judge**.
- **Advantage:** group-relative, `A_i = r_i − (1/|G|) Σ_{j∈G} r_j` over G repeated rollouts of the same task.
- **Credit assignment:** the task-level advantage is **uniformly distributed across all output tokens**, including memory-operation tokens. This jointly optimizes memory usage and task performance end-to-end.
- **Objective:** `J(θ) = E[(1/G) Σ_i ρ_θ^i A_i − β·D_KL(π_θ ‖ π_ref)]`.

Backbone: **Qwen3-8B** for all agents; **Qwen3-embedding-0.6B** for retrieval. Storage = FAISS. Documents chunked (default 4K tokens) and fed step-by-step; on a Read, 6 entries are retrieved. Trained on VeRL on NVIDIA A800 GPUs (Table 7: batch 16, rollout group 16, lr 1e-6, clip-high 0.28).

---

## Experimental Results

Tasks: **3 long-context multi-hop QA** (HotpotQA, 2WikiMultihopQA, MuSiQue) and **2 multi-turn web** benchmarks (GAIA, WebWalkerQA). All methods share the **Qwen3-8B** backbone.

QA difficulty is augmented two ways: (1) **NIAH-style long context** — relevant docs shuffled and interleaved with many irrelevant docs (train on 200 docs ≈ 28K tokens; test scaled to **800 docs ≈ 112K tokens**, a 4× extension); (2) **multi-question setting** — 1–10 questions presented simultaneously with their docs mixed, stressing maintenance of multiple semantically independent memories. Web tasks allow up to 40 web tool calls (Google search + Jina URL Reader); training data from Asearcher.

### Main results (Table 1)

Long-context QA reported at 200-doc and 800-doc scales; web at single scale. "Avg." is across all reported columns.

| Method | HotpotQA 200 | HotpotQA 800 | 2WikiMQA 200 | 2WikiMQA 800 | Musique 200 | Musique 800 | GAIA | WebWalker | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| **Training-free** | | | | | | | | | |
| Full Context | 63.5 | 62.0 | 55.7 | 49.2 | 42.8 | 41.9 | 23.3 | 29.5 | – |
| Vanilla RAG | 67.8 | 63.1 | 46.5 | 40.0 | 38.5 | 37.1 | 20.4 | 24.0 | – |
| Generative Agents | 38.8 | 10.0 | 12.3 | 2.0 | 19.8 | 8.4 | 22.3 | 29.5 | – |
| Mem0 | 38.2 | 33.9 | 24.2 | 18.3 | 14.0 | 11.2 | 25.2 | 28.3 | – |
| A-Mem | 73.5 | 70.4 | 62.7 | 57.1 | 47.1 | 41.6 | 30.1 | 29.0 | – |
| **Trained** | | | | | | | | | |
| MemAgent | 76.5 | 71.1 | 65.8 | 57.7 | 54.7 | 46.0 | 44.5 | 33.0 | – |
| AtomMem w/o RL | 65.9 | 60.1 | 52.8 | 55.0 | 47.8 | 50.0 | 40.0 | 35.2 | 45.6 |
| **AtomMem (ours)** | **77.8** | **72.9** | **67.5** | **62.5** | **55.1** | **51.4** | **48.5** | **37.4** | **48.7** |

Key findings:
- **AtomMem outperforms all trained and untrained baselines on average**, ~3–8 points over prior static-workflow methods under the same Qwen3-8B backbone.
- **Robust scaling:** at the 800-doc setting (4× the training context), AtomMem keeps a clear lead — evidence that it learned a *content-aware* policy that mitigates information overload as noise grows.
- **RL is decisive:** AtomMem improves by **~9 points on average** over its w/o-RL ablation (48.7 vs 45.6 overall; e.g., Musique 200 jumps 47.8 → 55.1, GAIA 40.0 → 48.5). Directly optimizing memory *decisions* with task-level feedback — not just having the CRUD tools — is what drives the gains.

### Efficiency (Table 6)

| Method | Wall Clock (s/task) | Avg. Tokens | Avg. LLM Calls | Avg. Retrieve Calls |
|---|---|---|---|---|
| **AtomMem (ours)** | **97.6** | 570.5 | **10.9** | 10.9 |
| MemAgent | **49.7** | 264.1 | 8.0 | 0.0 |
| Mem0 | 247.8 | 431.7 | 12.9 | 88.6 |
| Generative Agents | 416.0 | 494.6 | 375.9 | 418.0 |
| A-Mem | 662.4 | 237.7 | 400.0 | 402.0 |

AtomMem and MemAgent are far more efficient than the other workflows, which invoke the LLM many times per input (serialized, nearly unscalable: A-Mem/Generative Agents make ~400 LLM calls per task). AtomMem is slightly slower than MemAgent due to (1) longer prompts from integrating multiple tools and (2) the added vector-DB component whose retrieval adds latency.

---

## Training Dynamics (the learned policy)

A central empirical contribution: **what policy does RL discover?** (Figure 3, QA tasks)

- **Behavior shifts from under-managed to task-aligned.** Early in training the model **over-relies on Read** and neglects maintenance (lots of redundant retrieval). As training proceeds, **Read usage drops sharply** while **Create, Update, and Delete increase substantially** — the model learns to keep a *compact, task-relevant* store: preserve useful info, revise stale entries, remove redundancy.
- **Update is rare but pivotal.** Update frequency stays low relative to Create, but it is the "critical few" — the ablation (below) shows removing Update hurts badly, whereas removing Delete barely matters *on these accumulation-style, non-conflicting-fact tasks*.
- **The policy is task-conditioned, not fixed.** When the task condition changes, the operation frequencies follow **entirely different trends** — direct evidence that effective memory control benefits from *learned task-aligned patterns* rather than a single static strategy.

---

## Ablation Study

### Operations and components (Table 2; bracketed = relative drop)

| Variant | HotpotQA | 2WikiMQA | Musique |
|---|---|---|---|
| **AtomMem (full)** | **77.8** | **67.5** | **55.1** |
| w/o Update | 71.4 (−6.4) | 62.6 (−4.9) | 47.9 (−7.2) |
| w/o Delete | 76.5 (−1.3) | 67.3 (−0.2) | 54.2 (−0.9) |
| w/o Scratchpad | 71.8 (−6.0) | 56.3 (−11.2) | 46.0 (−9.1) |
| w/o Storage (vector DB) | 69.2 (−8.6) | 59.4 (−8.1) | 43.9 (−11.2) |
| w/o Both | 25.6 (−52.2) | 27.1 (−40.4) | 12.1 (−43.0) |

Findings:
1. **Update is critical, Delete is not (on these tasks).** Removing Update drops 5–7 points across the board (selectively revising entries keeps representations accurate and compact as new evidence arrives). Removing Delete is near-marginal because these are information-accumulation tasks with non-conflicting facts. *Caveat:* when memory capacity is capped (Appendix C), Update and Delete become much more important.
2. **Scratchpad + storage are a structural necessity.** Removing either drops 5–11 points; removing **both is catastrophic** (>40 points). The learned policy can lean on whichever component survives (robust to single-component failure), but the two hold fundamentally different information (global task state vs fine-grained reusable facts) and **cannot substitute for each other**. Trained-from-scratch *scratchpad-only* and *storage-only* variants both stay well below full AtomMem (Figure 4), and storage-only benefits only marginally from RL — the **synergy** raises the performance ceiling.

### Hyperparameters

- **Retrieval size K (Table 3):** must match task demand — K=3 clearly hurts, K=12 adds little over K=6 (benchmarks need only 2–4 hop reasoning, so ~6 retrieved docs suffice). **Robust to chunk size** (2048/4096/8192) thanks to Qwen3's long-context ability + RL extraction.
- **Embedding model (Table 4):** learned embeddings beat random selection by **+7.4 points avg** (similarity matching matters). Scaling the embedder helps modestly: 0.6B → 66.5, 4B → 67.2, 8B → 68.2 avg.

### Memory-capacity limit (Appendix C, Figure 6)

Capping the store at 20 entries and prompting the model to use Delete more: **Update and Delete frequencies rise significantly** (they don't add entries), while **Create and Read fall** (the model learns to create only within capacity; aggressive early discarding made Reads return little, so Read collapses then slightly recovers once the policy learns to retain only useful entries). Confirms the policy adapts its operation mix to environmental constraints.

---

## Case Study (HotpotQA, Figure 5)

The same agent applies **different `a_t^mem` strategies depending on `o_t^env`** — illustrating the dynamic nature:
- **Case 1 (unrelated docs):** uses the scratchpad to *log the absence* of relevant info; stores only potentially-related background entries.
- **Case 2 (partial info, e.g., one film's release date):** commits the found evidence to memory and proactively issues a `<read_memory>` request for the missing piece.
- **Case 3 (all info present):** synthesizes facts in the scratchpad to derive the answer, then uses `<update_memory>` to **overwrite useless entries with the conclusion**.

Together: a learned context-sensitive workflow that decides when to ignore, retrieve, update, or consolidate based on the informational sufficiency of the current observation.

---

## Key Takeaways

1. **Memory-as-decision-making.** Casting memory as a POMDP where the store is a controllable part of the environment, and exposing only atomic CRUD ops, turns memory management into something RL can optimize end-to-end — the ceiling is the model's decision capacity, not an expert's rule set.

2. **Atomic CRUD is a complete, minimal, task-agnostic operator set.** Higher-level memory tools are just compositions of CRUD; giving the model the primitives lets it *learn* the higher-level workflow per task.

3. **Compositional macro-actions evolve the store entry-by-entry.** `M_{t+1} = a_t^K ∘ … ∘ a_t^1(M_t)` is fine-grained dynamic consolidation — no fixed densification or forgetting schedule.

4. **RL is what unlocks it (~9 pts).** The CRUD tools alone (w/o RL) underperform; the gain comes from learning *when/how* to apply operations via task-level GRPO with terminal rewards uniformly credited across tokens.

5. **The learned policy is interpretable and task-aligned.** Read usage falls while Create/Update/Delete rise, and the mix re-shapes itself when task conditions (or capacity limits) change — discovering structured strategies rather than a fixed routine.

6. **Dual-path memory (scratchpad + vector DB) is structurally necessary.** Neither alone suffices; their synergy raises the ceiling, and removing both is catastrophic.

---

## Limitations (Acknowledged by Authors)

1. **RL is computationally intensive** — training to convergence takes ~2–3 days on an 8-GPU (A800) cluster, a bottleneck for scaling to even longer-horizon or noisier tasks.
2. **Coarse credit assignment** — task-level advantage is spread *uniformly* across all actions/tokens; ideally each memory entry's contribution to task success would be measured precisely, but per-entry value estimation is non-trivial and left to future work (they deliberately avoid inventing a bespoke RL algorithm for one downstream task).
3. **Delete's value is task-dependent** — on the studied accumulation-style, non-conflicting-fact QA tasks, explicit deletion is near-marginal; its importance only emerges under capacity limits, so the general utility of removal ops on conflict-heavy workloads is under-explored.

---

## Where it sits (v1/v2)

AtomMem is a **v2 "learnable-memory" system**: the memory *policy itself* is trained (GRPO), not hand-designed. Its distinguishing move is **fine-grained atomic-op evolution** — the store is evolved entry-by-entry through compositional CRUD macro-actions the policy chooses per step, with task-level RL credit flowing through the operation tokens.

Relative to neighbors:

- **vs A-Mem (note + evolve):** A-Mem is *training-free* — it writes Zettelkasten-style notes and uses a hand-crafted "generate links / evolve neighbors" heuristic on each insertion; the evolution rule is fixed. AtomMem replaces that hand-crafted evolution with a *learned* policy over atomic edits. (Notably, A-Mem is AtomMem's strongest training-free baseline in Table 1, but AtomMem beats it across all settings.)
- **vs MemCoE (learned policy):** both learn a memory-management policy rather than hard-coding it; AtomMem's specific contribution is grounding that policy in a *complete, minimal CRUD operator set* argued to be a general-purpose foundation, optimized purely with terminal task reward and uniform token-level credit (no external memory-controller module).
- **vs Memory-R1's discrete ops (ADD/UPDATE/DELETE):** closest sibling — Memory-R1 also uses RL to learn discrete memory operations. AtomMem differs by (a) including **Read as an explicit, learnable decision variable** (memory access is part of the POMDP's partial observability, not always-on retrieval), (b) emitting **compositional macro-actions** (multiple ops per environment step, composed as a single state transition), and (c) pairing a mandatory **scratchpad** (deterministic short-term retrieval) with the selective vector store — a dual-path design the ablation shows is structurally necessary.
