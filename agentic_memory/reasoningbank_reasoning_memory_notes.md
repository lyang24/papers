# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

**Authors:** Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, Tomas Pfister (UIUC, Google Cloud AI Research, Yale)

**Paper:** arXiv:2509.25140v2 (Sep 2025, rev. Mar 2026)

**GitHub:** https://github.com/google-research/reasoning-bank

---

## The Core Problem

LLM agents deployed in persistent, long-running roles encounter a continuous stream of tasks, but they **largely fail to learn from accumulated interaction history**. Approaching each new task in isolation, they (i) repeat similar past errors, (ii) discard valuable insights from related problems, and (iii) lack self-evolving capabilities that would make the system more capable over time.

Existing agent-memory work mostly stores past interactions for reuse, but in two impoverished forms:

1. **Raw trajectories** (e.g., Synapse) — comprehensive and original, but too lengthy and noisy to apply directly to a new query.
2. **Successful routines / workflows** (e.g., AWM) — distilled procedures, but they only capture *what worked*.

Both share two fundamental drawbacks:

- They **cannot distill higher-level, transferable reasoning patterns** — they remain at the level of concrete actions or procedures.
- By **over-emphasizing successful experiences**, they leave the valuable lessons from an agent's *own failures* largely unexploited.

Consequently, existing memory designs remain **passive record-keeping** rather than actionable, generalizable guidance for future decisions. This is the crucial difference from MAGMA / Mem0 / A-Mem, which target the *form* and *structure* of factual/conversational memory; ReasoningBank targets *what content* should be stored and reused — strategy-level reasoning, not facts.

---

## The Big Idea: Distill Reasoning Strategies from Successes AND Failures, in a Closed Loop

ReasoningBank distills and organizes memory items from **both successful and failed experiences, judged by the agent itself without ground-truth labels**. It captures effective strategies from successes *and* preventative lessons (guardrails) from failures, abstracting both into a collection of actionable, transferable reasoning principles.

The process operates as a **closed loop** across a stream of tasks (test-time learning, no ground truth available):

1. **Retrieve** — facing a new task, retrieve relevant memory items to guide actions.
2. **Extract** — after the task, an LLM-as-a-Judge labels the trajectory success/failure, and a memory extractor distills new reasoning items.
3. **Consolidate** — integrate new items back into the bank, so the agent continually evolves.

On top of this experience learner, the paper introduces **MaTTS (Memory-aware Test-Time Scaling)**: allocate more inference compute per task to generate abundant, diverse experiences, which provide *contrastive signals* for synthesizing higher-quality memory — and that better memory in turn steers more effective scaling. This bidirectional synergy positions **memory-driven experience scaling as a new scaling dimension** (scaling experience through *depth* per task, not just *breadth* across more tasks).

---

## Architecture

### Problem Setup (Test-Time Learning)

An agent policy π_L(·|M, A) is parameterized by a backbone LLM L, conditioned on memory module M (= ReasoningBank, initialized empty) and action space A (web navigation ops for browsing; bash commands for SWE). A streaming sequence of queries Q = {q1, ..., qN} arrives one at a time; each must be completed without access to future queries and **without ground-truth feedback**. The agent must self-evolve using only its own past trajectories and self-verification. Two challenges: (i) how to extract and preserve useful memory; (ii) how to leverage it to avoid re-discovering known strategies or repeating mistakes.

### Memory Schema (what is stored)

Each memory item is a structured, human-interpretable + machine-usable knowledge unit that **abstracts away low-level execution detail while preserving transferable reasoning**:

| Field | Role |
|---|---|
| **Title** | Concise identifier summarizing the core strategy / reasoning pattern |
| **Description** | One-sentence summary of the item |
| **Content** | The distilled reasoning steps, decision rationales, or operational insights extracted from past experience |

Example item — Title: *"Prioritize user account sections for personal data"*; Content: *"Systematically look for and click on links ..."*. Crucially, extraction prompts forbid mentioning specific websites, queries, or string contents — the item must capture **generalizable insight**, not a replayable trace.

### The Three-Step Integration Loop

**(i) Memory Retrieval.** Query ReasoningBank with the current task context; retrieve the top-*k* relevant items via embedding-based similarity search. Retrieved items are injected into the agent's **system instruction**, grounding action prediction in past experience.

**(ii) Memory Extraction.** After task completion:
- **Proxy correctness signal:** an **LLM-as-a-Judge** labels the trajectory as success or failure given (query, trajectory), *without any ground-truth reference*.
- **Divergent extraction by outcome:** successful experiences contribute **validated strategies**; failed experiences supply **counterfactual signals and pitfalls** that sharpen guardrails. Up to 3 memory items are extracted per trajectory. (The thinking process of π_L is used as an approximation of the lengthy raw observations.)

**(iii) Memory Consolidation.** New items are incorporated into ReasoningBank via a simple **addition** operation, maintaining an evolving repository.

> Design note: the authors deliberately keep retrieval (plain embedding similarity) and consolidation (simple add) *simple* to isolate the contribution of the reasoning-oriented memory *content* itself. Both components are orthogonal and could be swapped for more sophisticated mechanisms (adaptive retrieval, hierarchical consolidation, episodic/graph structure as in MAGMA).

### MaTTS: Memory-aware Test-Time Scaling

A naive combination of memory + TTS (called **Vanilla TTS** / "MaTTS w/o aggregation") simply converts *more* independent trajectories into *more* independent memory items — it fails to exploit the contrastive signal that arises from redundant exploration on the *same* problem. MaTTS fixes this by deliberately learning from the abundant success/failure trajectories generated during scaling. Two instantiations (scaling factor *k*; *k*=1 means no scaling):

- **Parallel scaling — self-contrast.** Generate *k* trajectories for the same query under memory guidance, then **contrast across them** to identify consistent reasoning patterns and filter out spurious solutions → reliable memory curation.
- **Sequential scaling — self-refinement.** Iteratively refine reasoning within a single trajectory after initial completion; the **intermediate refinement notes** (attempts, corrections, insights not present in the final solution) are themselves harvested as memory signal.

The result: extra test-time compute translates into more transferable, higher-quality memory, and high-quality memory steers the scaled exploration toward promising paths — a positive feedback loop.

---

## Experimental Results

**Setup.** Web browsing: **WebArena** (general navigation, 5 subsets) and **Mind2Web** (cross-task / cross-website / cross-domain generalization). Software engineering: **SWE-Bench-Verified** (repo-level issue resolution). Backbones: Gemini-2.5-flash, Gemini-2.5-pro, Claude-3.7-sonnet. ReAct-style agents via BrowserGym (web) and bash-only (SWE). Baselines: **No Memory**, **Synapse** (trajectory memory), **AWM** (workflow memory). Metrics: success rate (SR ↑) and average steps / efficiency (Step ↓). MaTTS rows use parallel scaling, *k*=5, pass@1.

### WebArena — Overall Success Rate (SR) and Steps

| Backbone | Method | Overall SR | Overall Step |
|---|---|---|---|
| **Gemini-2.5-flash** | No Memory | 40.5 | 9.7 |
| | Synapse | 42.1 | 9.2 |
| | AWM | 44.1 | 9.0 |
| | **ReasoningBank** | **48.8** | **8.3** |
| | **+MaTTS** | **51.8** | **7.9** |
| **Gemini-2.5-pro** | No Memory | 46.7 | 8.8 |
| | Synapse | 47.7 | 8.5 |
| | AWM | 47.6 | 8.7 |
| | **ReasoningBank** | **53.9** | **7.4** |
| | **+MaTTS** | **56.3** | **7.1** |
| **Claude-3.7-sonnet** | No Memory | 41.7 | 8.0 |
| | Synapse | 42.6 | 7.9 |
| | AWM | 40.8 | 8.9 |
| | **ReasoningBank** | **46.3** | **7.3** |
| | **+MaTTS** | **48.8** | **7.2** |

Key findings:
- ReasoningBank improves overall WebArena SR by **+8.3 / +7.2 / +4.6** points over No Memory across the three backbones — and consistently beats both Synapse and AWM.
- It simultaneously **reduces interaction steps** (more effective *and* more efficient): up to 1.4 fewer steps vs No Memory and 1.6 fewer vs other memory baselines.
- AWM (workflow memory) sometimes *degrades* vs No Memory (e.g., Claude overall 40.8 < 41.7), exposing the brittleness of success-only routine memory.
- On the **Multi** subset (transfer memory across multiple websites), ReasoningBank gains +4.6 avg SR over the strongest baseline, while AWM fails to help or even degrades.

### Mind2Web — Generalization (Task-level SR ↑)

| Backbone | Method | Cross-Task SR | Cross-Website SR | Cross-Domain SR |
|---|---|---|---|---|
| **Gemini-2.5-flash** | No Memory | 3.3 | 1.7 | 1.0 |
| | Synapse | 3.5 | 1.9 | 1.1 |
| | AWM | 3.5 | 2.1 | 0.7 |
| | **ReasoningBank** | **4.8** | **2.3** | **1.6** |
| **Gemini-2.5-pro** | No Memory | 3.5 | 3.4 | 1.4 |
| | Synapse | 3.6 | 3.2 | 1.5 |
| | AWM | 3.7 | 2.3 | 1.2 |
| | **ReasoningBank** | **5.1** | **3.8** | **1.7** |

(The full table also reports element accuracy, action-F1, and step-success-rate, all favoring ReasoningBank.) Gains are **most pronounced in cross-domain**, the setting demanding the highest generalization — confirming that strategy-level memory transfers better than trajectories or workflows.

### SWE-Bench-Verified — Issue Resolution

| Backbone | Method | Resolve Rate ↑ | Avg Steps ↓ |
|---|---|---|---|
| **Gemini-2.5-flash** | No Memory | 34.2 | 30.3 |
| | Synapse | 35.4 | 30.7 |
| | **ReasoningBank** | **38.8** | **27.5** |
| **Gemini-2.5-pro** | No Memory | 54.0 | 21.1 |
| | Synapse | 53.4 | 21.0 |
| | **ReasoningBank** | **57.4** | **19.8** |

ReasoningBank improves resolve rate (+3.4 to +4.6) **while cutting ~2.8 steps** vs No Memory (and ~1.3 vs Synapse) — efficiency and effectiveness together, in a domain very different from web browsing.

### MaTTS Scaling (WebArena-Shopping, Gemini-2.5-flash)

Effect of scaling factor *k* (SR), comparing memory mechanisms:

| Setting | k=1 (no scaling) | k=5 |
|---|---|---|
| MaTTS w/o memory — parallel | 39.0 | 42.2 (fluctuates 39.0–42.2) |
| MaTTS w/o memory — sequential | ~37.4 | ~40.6 (fluctuates) |
| Vanilla TTS (MaTTS w/o aggregation) — parallel | 49.7 | 52.4 |
| Vanilla TTS — sequential | 49.7 | 51.9 |
| **MaTTS (full) — parallel** | 49.7 | **55.1** |
| **MaTTS (full) — sequential** | 49.7 | **54.5** |

- Both parallel and sequential scaling boost SR, but **only when paired with ReasoningBank** do gains become large and stable (w/o memory they fluctuate and saturate).
- **MaTTS consistently beats Vanilla TTS** (55.1 vs 52.4 parallel; 54.5 vs 51.9 sequential at k=5) — memory-aware aggregation of contrastive signal matters.
- **Sequential** wins at small *k* (early refinement adds insight) but **saturates** once the model succeeds/fails decisively; **parallel dominates at larger k** (diverse rollouts keep providing contrastive signal).

### The Synergy: Better Memory ↔ Stronger Scaling (WebArena-Shopping, parallel k=5)

| Memory | No Scaling | Pass@1 (avg quality after curation) | Best-of-5 (BoN) |
|---|---|---|---|
| No Memory | 39.0 | 39.0 | 42.2 |
| Synapse | 40.6 | 41.2 | 44.4 |
| AWM | 44.4 | 45.5 | 47.6 |
| **ReasoningBank** | 49.7 | **53.0** | **55.1** |

Two directions of the synergy:
1. **Better memory → stronger scaling.** BoN gain from scaling grows with memory quality: No Memory 39.0→42.2 (tiny), Synapse →44.4, AWM →47.6, ReasoningBank 49.7→**55.1** (largest).
2. **Scaling → better memory.** Pass@1 (avg trajectory quality after curation) *drops or barely moves* for weak memory (Synapse 40.6→41.2, AWM 44.4→45.5) — extra rollouts are wasted — but **rises 49.7→53.0 for ReasoningBank**. Only strong memory can harness scaling's diversity into constructive contrastive signal, closing the virtuous cycle.

---

## Ablation / Analysis

### Learning from Failures Is the Key Differentiator (WebArena-Shopping, Gemini-2.5-flash)

| Method | Success-only memory | + Failure trajectories |
|---|---|---|
| (No Memory baseline) | 39.0 | 39.0 |
| Synapse | 40.6 | 41.7 |
| AWM | 44.4 | **42.2 (drops!)** |
| **ReasoningBank** | 46.5 | **49.7** |

Synapse and AWM build memory *only* from successes; adding failures gives them little (Synapse +1.1) or actively **hurts** (AWM −2.2, failures become noise). ReasoningBank is *designed* to distill reasoning from both, so failures become constructive (+3.2). This is the core empirical evidence that the success-AND-failure distillation — not just "more data" — drives the gains.

### Targeted Efficiency: Fewer Steps on the *Right* Track (WebArena, steps)

Step reduction is decomposed by outcome (a good system should shorten *successful* paths, not merely truncate doomed ones):

| Domain | No Mem (Success) | ReasoningBank (Success) | No Mem (Failed) | ReasoningBank (Failed) |
|---|---|---|---|---|
| Shopping | 6.8 | **4.7 (↓2.1)** | 8.7 | 7.3 (↓1.4) |
| Admin | 8.4 | **7.0 (↓1.4)** | 10.4 | 9.5 (↓0.9) |
| Gitlab | 8.6 | **7.6 (↓1.0)** | 15.7 | 15.5 (↓0.2) |
| Reddit | 6.1 | **5.0 (↓1.1)** | 7.6 | 6.8 (↓0.8) |

Reductions are **larger on successful instances** (up to 2.1 steps, a 26.9% relative cut) than on failed ones — memory guides the agent down effective reasoning paths rather than just cutting failures short.

### Robustness to LLM-as-a-Judge Noise

The judge's measured accuracy vs ground truth is **72.7%** on WebArena-Shopping. Simulating verifier accuracy from 100% down to 50% (random guess), ReasoningBank's SR stays roughly stable within the 70%–90% accuracy band — the framework is **robust to imperfect self-judgment**, which is what makes ground-truth-free test-time learning viable.

### Emergent, Evolving Strategies

A human case study traces a single memory item ("User-Specific Information Navigation") evolving over the test-time timeline: from **procedural/execution** rules ("find navigation links") → **atomic self-reflection** (re-verify element identifiers) → **adaptive checks** (leverage search/filters to ensure completeness) → **compositional strategy** (cross-reference task requirements, reassess options). Strategies are not flat/monolithic; they mature from low-level actions to high-level reasoning, resembling RL learning dynamics.

---

## Key Takeaways

1. **What you store beats how much you store.** Distilling *generalizable reasoning strategies* (title/description/content) outperforms storing raw trajectories (Synapse) or success-only workflows (AWM) — the gains come from memory *content quality*, deliberately isolated by keeping retrieval/consolidation trivially simple.

2. **Failures are a first-class signal.** Self-judged failed trajectories yield counterfactual guardrails. ReasoningBank turns failures into +3.2 SR; success-only methods can't use them and sometimes degrade when fed failures.

3. **Memory and test-time scaling are synergistic, not additive.** MaTTS shows a bidirectional loop: better memory steers scaling toward promising rollouts; diverse rollouts (self-contrast for parallel, self-refinement for sequential) forge better memory. Vanilla TTS without memory-aware aggregation leaves most of the gain on the table.

4. **Effectiveness and efficiency improve together.** Up to ~20% relative SR improvement *and* up to ~16% fewer interaction steps — and the step savings concentrate on successful tasks, evidence of purposeful guidance rather than early truncation.

5. **Ground-truth-free and robust.** LLM-as-a-Judge (only ~73% accurate) is sufficient because the system tolerates verification noise — essential for real streaming deployment where labels never arrive.

---

## Limitations (Acknowledged by Authors)

1. **Focus on memory content, not structure** — the work studies *what* to store/reuse and does not extensively compare against episodic or hierarchical memory *architectures* (orthogonal concerns about memory form/structure). Combining ReasoningBank's content with such structures is future work.
2. **Intentionally simple retrieval and consolidation** — plain embedding retrieval + additive consolidation isolate content quality, but adaptive retrieval / hierarchical consolidation could likely push performance further.
3. **Dependence on LLM-as-a-Judge for correctness signals** — automatic labeling enables scale without ground truth but can inject noise on ambiguous tasks or judge errors; stronger verifiers, human-in-the-loop, or ensemble judging could improve reliability of memory induction.

---

## Where it sits (v1/v2)

This collection has been thin on **experiential / skill / strategy memory** — the survey's **"Experiential Memory" Function** (learning *how to act* from past task execution, distinct from remembering *facts* or *conversation*). ReasoningBank fills that gap directly.

- **Contrast with factual / conversational memory (Mem0, A-Mem, MAGMA).** Those systems answer *"what was said / what is true?"* — they index user facts, dialogue history, and entities, optimizing retrieval structure (graphs, Zettelkasten notes, multi-graph disentanglement) over a knowledge store. ReasoningBank answers a different question: *"what strategy should I use to act, and what pitfalls must I avoid?"* Its content is **procedural/strategic reasoning**, not declarative facts; its benchmark is **task success rate and steps** on agentic tasks (WebArena, Mind2Web, SWE-Bench), not QA accuracy on conversation logs (LoCoMo, LongMemEval). The two lines are complementary: a deployed agent plausibly wants MAGMA-style factual recall *and* ReasoningBank-style strategy memory.

- **Experiential lineage it extends.** ReasoningBank sits in the line of **ExpeL** (LLM agents as experiential learners that extract insights from successes and failures), **AWM / Agent Workflow Memory** (induce reusable workflows — but success-only, procedural, and a direct baseline here), and **Voyager** (a growing skill library for embodied lifelong learning). Its advances over this lineage: (a) distilling *strategy-level reasoning units* rather than replayable trajectories or rigid workflows, (b) treating self-judged *failures* as first-class memory, ground-truth-free, in a streaming closed loop, and (c) introducing **MaTTS** — coupling experiential memory with test-time scaling so the two reinforce each other, a dimension absent from prior experiential-memory work.

In short, for this collection ReasoningBank is the **canonical "experiential / skill memory" entry**: where Mem0/A-Mem/MAGMA make agents *remember better*, ReasoningBank makes agents *act better over time* by remembering *how*.
