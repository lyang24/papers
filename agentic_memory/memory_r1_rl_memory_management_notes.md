# Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning

**Authors:** Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schuetze, Volker Tresp, Yunpu Ma (LMU Munich, Munich Center for ML, TU Munich, U. Cambridge, U. Hong Kong, TU Darmstadt, U. Edinburgh)

**Paper:** arXiv:2508.19828v5 (Aug 2025, last revised 14 Jan 2026)

**GitHub:** Not released at time of writing (no code link in the paper)

---

## The Core Problem

Memory-augmented LLM agents face two coupled but under-learned challenges, and existing systems handle both with **static, heuristic, in-context rules** that carry no learning signal tied to correctness:

1. **Memory management — what to remember, update, or discard.** CRUD-style systems (MemGPT, MemLLM) and the more common `{ADD, UPDATE, DELETE, NOOP}` operator set (Mem0) let a *vanilla* LLM pick operations from prompt instructions alone. Even trivial cases break: a user says "I adopted a dog named Buddy" and later "I adopted another dog named Scout." A vanilla manager reads the second statement as a **contradiction**, issues `DELETE`+`ADD`, and overwrites the original — fragmenting the memory. The correct move is a single `UPDATE` consolidating both dogs.

2. **Memory utilization — filtering retrieved noise before answering.** The RAG paradigm appends retrieved entries to the prompt with no filtering or prioritization. Too few entries omit crucial context; too many flood the model with distractors. The LLM is forced to reason over relevant and irrelevant content together and gets distracted (the "lost in the middle" failure). Humans, by contrast, retrieve broadly but then **filter**, integrating only the most useful pieces.

Supervised fine-tuning offers limited help because it is impractical to label every memory operation or retrieval decision. The authors argue **RL is the missing ingredient**: by optimizing an outcome-based reward (final answer correctness), the model can *learn* when to add/update/delete/retain and how to use retrieved memories — no per-operation labels required.

---

## The Big Idea: Two RL-Trained Agents With Outcome-Driven Rewards

Memory-R1 is the **first RL framework for memory-augmented LLMs**. It splits the problem into two specialized agents, both fine-tuned with **PPO or GRPO** using only the downstream answer's correctness as the reward signal:

| Agent | Role | Action space / mechanism | Reward |
|---|---|---|---|
| **Memory Manager** | Maintains and evolves the memory bank | Structured ops `{ADD, UPDATE, DELETE, NOOP}` + updated content `m'` | Answer correctness of the *frozen* Answer Agent after the op is applied (EM) |
| **Answer Agent** | Answers questions over retrieved memories | **Memory Distillation** policy — preselect relevant entries from 60 RAG candidates, then reason | Exact Match between generated and gold answer |

The defining move is that **neither agent is given labels for its own intermediate action.** The Memory Manager never sees a "correct operation" label; it only learns that an operation was good *if it led the Answer Agent to answer correctly*. This outcome-driven signal needs no manual annotation, is scalable, and — critically — works with as few as **152 training QA pairs**.

---

## Architecture

The pipeline runs in two stages over multi-session dialogues (a dialogue = multiple sessions at different times; each session = several turns; answering requires synthesizing across sessions).

### Stage 1 — Memory Bank Construction (Memory Manager)

At each dialogue turn, the LLM extracts and summarizes information worth remembering, retrieves related existing entries from the memory bank, and the Memory Manager chooses an operation.

Formally, the manager is a policy π_θ taking extracted info `x` and retrieved memories `M_old`, emitting an operation `o` with content `m'`:

```
(o, m') ~ π_θ(· | x, M_old)
```

**`{ADD, UPDATE, DELETE, NOOP}` semantics** (from the prompt, adapted from Mem0):
- **ADD** — new info not in memory; generate a fresh ID.
- **UPDATE** — info already present but materially different; keep the same ID, preserve `old_memory`, and *consolidate* (e.g. "Buddy" + "Scout" → one entry listing both dogs).
- **DELETE** — retrieved fact contradicts memory; return the same ID.
- **NOOP / NONE** — fact already present or irrelevant; no change.

The whole point of RL here is teaching the nuance between these — favoring **consolidation over fragmentation**. The case study shows the trained manager rewriting "Joanna is allergic to most reptiles..." into a single UPDATE that folds in "turtles and cockroaches" *and* preserves the emotional context ("finds turtles peaceful... but is allergic to them"), where the vanilla manager fired three DELETEs + one ADD and discarded the nuance.

**Reward (outcome-driven):** After applying `(o, m')`, the updated bank is passed to the *frozen* Answer Agent, and the reward is

```
R_answer = EM(y_pred, y_gold)
```

This exact-match signal needs no manual labels and is sufficient to teach effective memory operations.

### Stage 2 — Question Answering (Answer Agent + Memory Distillation)

For each question, **60 candidate memories** are retrieved via similarity-based RAG (top-30 per participant × 2 speakers). The Answer Agent is a policy mapping the question `q` and retrieved set `M_ret` to an answer:

```
y ~ π_θ(· | q, M_ret)
```

The agent first performs **Memory Distillation** — explicitly selecting the subset of retrieved memories it deems relevant (output *before* the answer) — then reasons only over those. In the case study, asked "Does John live close to a beach or the mountains?", the un-distilled base model consumed all 60 entries and answered "mountains" (misled by irrelevant mountaineering mentions); the distilled Answer Agent surfaced only beach-related entries and answered "beach" correctly. Distillation discards noise, focuses on true signal, and improves factual accuracy.

### RL Optimization (both agents)

- **PPO:** clipped objective `J(θ) = E[min(ρ_θ A, clip(ρ_θ, 1−ε, 1+ε) A)]` with importance ratio ρ_θ and answer-based advantage A; actor + critic trained jointly (lr 1e-6 / 1e-5).
- **GRPO:** samples a group of G candidate actions per state, standardizes rewards into group-relative advantages `A_i = (r_i − mean(r)) / std(r)`, with a KL term to the reference policy. **No value function needed** (only the actor is updated via grouped return normalization).
- Both use **EM against ground truth** as the reward, decoding temperature τ=1.0 during training (exploration), greedy τ=0 at test. Implemented in the **VERL** framework. The two agents are trained **separately** for stability under sparse rewards (acknowledged as a limitation).

**Data construction (data-efficient).** From LoCoMo, a 1:1:8 split gives **152 / 81 / 1307** train/val/test QA pairs (adversarial subset excluded, following Mem0). For the manager, GPT-4o-mini builds a temporal memory bank from the preceding turns; the current turn + bank + linked QA form one training tuple — *no operation labels*. Memory-SFT (a behavior-cloning ablation) instead clones GPT-5-generated trajectories.

---

## Experimental Results

### LoCoMo Benchmark (main results)

Long multi-session dialogues (~300–600 turns, ~9k–26k tokens, up to 35 sessions). Metrics: token-level **F1**, **BLEU-1 (B1)**, and **LLM-as-a-Judge (J)**, across Single-Hop, Multi-Hop, Open-Domain, and Temporal question types. All baselines re-implemented on the *same* backbones (temp 0, 2048 max tokens).

**Backbone: LLaMA-3.1-8B-Instruct** (Overall columns)

| Method | F1 ↑ | B1 ↑ | J ↑ |
|---|---|---|---|
| LoCoMo (RAG) | 8.97 | 7.27 | 12.17 |
| A-Mem | 26.08 | 21.78 | 40.78 |
| Mem0 | 30.61 | 23.55 | 53.30 |
| MemoryOS | 34.64 | 29.36 | 51.26 |
| Memory-SFT (GPT-5 clone) | 39.51 | 30.84 | 61.13 |
| **Memory-R1-PPO** | 41.72 | 33.70 | 59.53 |
| **Memory-R1-GRPO** | **43.14** | **36.44** | **61.51** |

**Backbone: Qwen-2.5-7B-Instruct** (Overall columns)

| Method | F1 ↑ | B1 ↑ | J ↑ |
|---|---|---|---|
| LoCoMo (RAG) | 11.41 | 8.71 | 13.62 |
| A-Mem | 29.20 | 24.40 | 44.76 |
| Mem0 | 30.41 | 22.22 | 45.68 |
| MemoryOS | 35.04 | 27.99 | 48.20 |
| Memory-SFT (GPT-5 clone) | 42.81 | 32.98 | 58.76 |
| **Memory-R1-PPO** | 41.05 | 32.91 | 57.54 |
| **Memory-R1-GRPO** | **45.02** | **37.51** | **62.74** |

Per-type highlights (Qwen-2.5-7B, GRPO): the largest relative jumps are on the hardest types — **Multi-Hop** (F1 35.65 vs Mem0's 18.59, J 53.01 vs 37.35) and **Temporal** (F1 49.86 vs 26.90).

**Headline gains** (GRPO over strongest baseline MemoryOS):
- LLaMA-3.1-8B: **+28.5% F1, +34.0% B1, +30.2% J** (relative). PPO: +17.2% / +17.6% / +19.4%.
- Qwen-2.5-7B: **+24.5% F1, +24.1% B1, +20.0% J** (relative).
- GRPO also **beats Memory-SFT** despite SFT being distilled from GPT-5 — outcome-driven RL > supervised imitation of a stronger teacher.

### Generalization (zero-shot transfer)

Models trained *only* on LoCoMo and evaluated zero-shot on **MSC** (Multi-Session Chat) and **LongMemEval**. Both PPO and GRPO variants kept consistent improvements across all three benchmarks and all metrics, never having seen MSC/LongMemEval — gains spanning single-hop, multi-hop, open-domain, and temporal types.

### Scalability (Qwen-2.5 3B / 7B / 14B)

Both PPO- and GRPO-tuned variants consistently outperform the base model at every scale on F1/B1/J, with gains *persisting* as the backbone grows — RL teaches memory management regardless of backbone capacity.

---

## Ablation Studies

All on LLaMA-3.1-8B. Grey baselines = pipeline without RL fine-tuning.

### Effect of each RL component (Overall F1 / B1 / J)

| Configuration | PPO | GRPO |
|---|---|---|
| **Full Memory-R1** | 41.0 / 32.9 / 57.5 | 45.0 / 37.5 / 62.7 |
| − RL Memory Manager (scripted ops) | 34.5 / 28.1 / 49.0 | 37.5 / 30.6 / 52.9 |
| − RL Answer Agent (static retrieval) | 32.5 / 24.6 / 59.4 | 33.0 / 24.9 / 59.9 |
| − Memory Distillation | 39.3 / 30.9 / 57.4 | 41.0 / 34.4 / 60.1 |

Findings:
1. **RL Memory Manager** matters — removing it drops GRPO F1 from 45.0 → 37.5. Outcome-driven RL produces better operations than scripted control.
2. **RL Answer Agent** is the largest single lever on lexical metrics (GRPO F1 45.0 → 33.0). Reward-driven tuning beats static retrieval.
3. **Memory Distillation** adds a consistent boost on top (GRPO 41.0 → 45.0 F1), confirming that filtering distractors reduces noise and improves reasoning.

### Compounding benefit (Answer Agent × Manager quality)

Pairing the RL Answer Agent with a **stronger** Memory Manager (GPT-4o-mini vs LLaMA-3.1-8B) yields *larger* gains: F1 +10.10 → +19.72, B1 +10.81 → +18.19, J +5.05 → +15.76. The two agents **compound** — a cleaner memory bank lets the Answer Agent extract more value.

### PPO vs GRPO

Trained on the Answer Agent with EM reward: **GRPO converges faster initially** (grouped return normalization gives stronger early guidance), but both reach **comparable final reward**.

### Reward design (Answer Agent)

| Reward | F1 ↑ | B1 ↑ | J ↑ |
|---|---|---|---|
| PPO (LLM-as-a-Judge reward) | 33.69 | 23.36 | **63.58** |
| PPO (Exact-Match reward) | **41.05** | **32.91** | 57.54 |

A J-based reward pushes the agent toward verbose, descriptive answers (e.g. "Yes, John and James studied together, as they were part of the same online programming group...") that score high on the judge but get penalized by string-overlap F1/B1 and break length-controlled comparison. The authors adopt **EM** for balanced gains across all three metrics.

### Distillation vs Reranking

Compared against a Base + Reranker pipeline: reranking gives modest accuracy gains but **substantial latency overhead**, whereas Memory-R1's learned distillation achieves **higher accuracy with lower median and tail latency** — a better accuracy/latency trade-off.

---

## Key Takeaways

1. **Memory management can be *learned*, not scripted.** Framing operation selection `{ADD, UPDATE, DELETE, NOOP}` as an RL problem with a downstream-correctness reward teaches the nuance (consolidate vs fragment) that in-context heuristics miss — the Buddy/Scout and Joanna-allergy cases are concrete failures of vanilla CRUD that RL fixes.

2. **Outcome-driven RL removes the labeling bottleneck.** Because the only signal is final-answer EM, there is no need to annotate the "correct" memory operation or the "correct" retrieved subset — which is exactly what makes labeling intractable in this space.

3. **Extreme data efficiency.** State-of-the-art LoCoMo results from **152 QA pairs**, generalizing zero-shot to MSC and LongMemEval and scaling across 3B–14B backbones. RL > SFT even when the SFT teacher is GPT-5.

4. **Retrieval should be filtered, not dumped.** Memory Distillation (retrieve broadly, then select) is a cheap, learnable alternative to rerankers that improves both accuracy and latency.

5. **The two agents compound.** A better Memory Manager amplifies the Answer Agent's gains — clean memory and good utilization are multiplicative, not additive.

---

## Limitations (Acknowledged by Authors)

1. **Dialogue-centric evaluation.** All benchmarks (LoCoMo, MSC, LongMemEval) are conversational; extending to **multimodal** data may surface challenges beyond this work's scope.
2. **Separate, non-end-to-end training.** The Memory Manager and Answer Agent are trained *separately* for stability under sparse rewards. This is necessary but makes the pipeline less straightforward; an **end-to-end multi-agent RL** approach could simplify training and enable richer coordination (left as future work).
3. **Sparse outcome reward.** Relying solely on final-answer EM is what enables label-free training, but the EM-vs-J reward analysis shows the reward choice meaningfully shapes behavior (verbosity, length control), so the signal is not neutral.

---

## Where it sits (v1/v2)

Memory-R1 belongs squarely to the **v2 "RL meets memory" subfield** of the agentic-memory landscape. The v1 generation — A-Mem (Zettelkasten note-linking), MemoryOS (OS-style read/write/management), MemGPT, Mem0 — all rely on **hand-crafted rules and in-context heuristics** to decide memory operations. They have *structure* but no *learned policy*: the LLM picks `{ADD, UPDATE, DELETE, NOOP}` from prompt instructions with no feedback tied to whether the resulting memory state actually helps downstream answering.

Memory-R1's contribution is to make the **memory-management policy itself learned** — the operation choice and the retrieval-filtering choice are both optimized via PPO/GRPO against outcome reward, rather than scripted. This is the same axis along which the survey groups the emerging **"RL meets memory" frontier**: Memory-R1 sits alongside **Mem-α**, **AgeMem**, **MemCoE**, and **LatentMem** as work that replaces hand-crafted memory rules (A-Mem / MemoryOS) with *trained* control. Where MAGMA-style v2 work pushes on **structured representation** (disentangled multi-graph memory, intent-aware traversal), Memory-R1 pushes on the orthogonal axis of **learned control over an otherwise simple flat store** — and the two directions are complementary (the ablation's "stronger manager → bigger answer gains" hints that better-structured memory and better-learned policies compound).
