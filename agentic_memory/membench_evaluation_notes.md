# MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents

**Authors:** Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen (Gaoling School of AI, Renmin University of China); Quanyu Dai, Zhenhua Dong (Huawei Noah's Ark Lab)

**Paper:** arXiv:2506.21605 (20 Jun 2025) | https://github.com/import-myself/Membench

---

## The Problem

Prior memory benchmarks (LOCOMO, LongMemEval, PerLTQA, PersonaChat-LT) share three blind spots:

1. **Only factual memory.** They test recall of explicitly-stated facts and neglect *reflective* memory — the higher-level preferences/emotions a user never states directly but that can be inferred from many low-level expressions. (E.g., "I love prosciutto-and-melon / salted-maple ice cream" → *taste preference: sweet-and-salty*.)
2. **Only the participation scenario.** The agent is always a first-person chat participant. They ignore the **observation scenario**, where the agent is a passive third-person observer recording a stream of user messages (no agent replies, no action module interference).
3. **Only effectiveness (accuracy).** They evaluate whether the answer is right, but not the **efficiency** (read/write latency) or **capacity** (how much memory before accuracy collapses) of the memory *mechanism* — which matter in real deployments. They also typically dump everything into a long context rather than running the agent's actual store/retrieve memory loop.

MemBench is positioned (Table 1 in the paper) as the first dataset to add **both** observation scenario (OS) **and** reflective memory (RM) on top of profiles + participation + factual memory:

| Dataset | Profiles | Scenarios | Levels |
|---|---|---|---|
| PerLTQA | yes | PS | FM |
| LoCoMo | no | PS | FM |
| LongMemEval | no | PS | FM |
| **MemBench** | **yes** | **PS & OS** | **FM & RM** |

---

## What They Built

### 1. The Dataset (built on MemSim)

The construction pipeline extends **MemSim** (a Bayesian simulator for personal-assistant memory) along the two new axes (reflective memory, participation scenario).

**a) User relation graph sampling.** Each user is a graph of a profile plus related entities — individuals (relatives, colleagues), events, places, items (see Figure 2). MemSim handles factual attribute sampling; MemBench adds sampling of **high-level (reflective) attributes**. To make reflective attributes realistic, they mine three recommendation datasets — **MovieLens, Food (recipes), Goodreads** — taking each user's most-frequent positively-rated item category as their high-level preference (or summarizing via GPT-4o-mini when no category exists). They then build three **one-to-many mappings** from each high-level preference to its low-level factual items (e.g., taste "sweet" → {candy, honey, apple pie, ...}).

**b) Memory dataset construction.** For the observation scenario MemSim's flow is reused directly; for the **participation scenario** they use a *self-dialogue* method — pick several low-level attributes under a high-level preference, generate evidence dialogues for them, and weave the key turns into a fuller multi-turn conversation so it reads naturally (e.g., a Star-Wars mention buried inside chit-chat about the movie). Agent replies are **pre-defined** so other agent modules don't pollute the memory measurement.

**c) Time-based sessions.** Turns inside a session get continuous timestamps (gaps ~1 min); across sessions order is preserved but gaps are larger (~1 day). Figure 1 shows the full flow: extract an event ("Build Start 2024", time "next week Mon 7:00 PM") → generate evidence dialogue + question → merge with other-attribute dialogue → answer from a resolved absolute label ("2024-10-07 Monday 19:00").

### 2. Two interaction scenarios

- **Participation Memory Scenario** — user↔agent dialogue. The agent must remember **both** the user's messages **and** its own (pre-defined) responses (e.g., what it recommended). Data unit = multi-turn sessions.
- **Observation Memory Scenario** — a one-way stream of user messages; the agent only records, never acts. Data unit = message lists.

The point: in PS, other agent modules (reasoning) interact with memory; in OS they don't — so the two cannot be collapsed into one setting.

### 3. Two memory levels and their task taxonomy

**Factual memory** — specific attributes of users/entities, surfaced in dialogue (8 question types, Table 6):

| Type | What it tests |
|---|---|
| Single-hop | answer from one message |
| Multi-hop | combine multiple messages |
| Comparative | compare two entities on a shared attribute |
| Aggregative | aggregate >2 entities on a common attribute |
| Post-processing | extra reasoning steps (e.g. resolve "next Monday" → date) |
| Knowledge-update | the answer changes over time as the user corrects/updates it |
| Single-session-assistant | answer from a single *assistant* message (e.g. "what did you recommend?") |
| Multi-session-assistant | answer from multiple assistant messages across sessions |

(The last two — recalling the *agent's own* outputs — exist only in the participation scenario.)

**Reflective memory** — extract/summarize a high-level trait from many low-level expressions (2 types, Table 7):

| Type | What it tests |
|---|---|
| Preference | infer a low-level→high-level preference from multiple messages |
| Emotion | infer the user's emotional state from consecutive messages in a time window |

### 4. Four metrics (effectiveness + efficiency + capacity)

- **Memory Accuracy** — all questions are **multiple-choice** (avoids judging free-form phrasing); accuracy = choice vs. gold choice.
- **Memory Recall** — for retrieval-based mechanisms, did retrieval surface the pre-tagged **key evidence** turns? (Reported as Recall@10.)
- **Memory Capacity** — the token threshold at which accuracy sharply collapses (may not exist for pure retrieval mechanisms).
- **Memory Efficiency** — **read time (RT)** and **write time (WT)** per operation, in seconds.

### 5. Dataset statistics

500 user relation graphs, plus dialogues/message-lists/questions. Totals (Table 2; TPT = avg tokens per trajectory):

| Subset | # Session | # Question | # Trajectory | TPT |
|---|---|---|---|---|
| PS · Reflective | 3.5k | 3.5k | 3.5k | 2,195 |
| PS · Factual | 51k | 39k | 8k | 10,285 |
| OS · Reflective | 2k | 2k | 2k | 745 |
| OS · Factual | 8.5k | 8.5k | 8.5k | 617 |

(~53k questions total — the figure the survey cites.) Key-evidence turns are spread roughly evenly across rounds in a session to mimic real answer-location distribution (Figure 4).

---

## Experimental Setup

**Noise injection for difficulty.** They generate irrelevant dialogues/messages from a **News (twitter-news)** dataset (verified not to conflict with the real facts) and interleave them between sessions. Tuning the noise ratio controls difficulty; at high noise each individual test averages **>100k tokens**.

**Two sized sub-datasets** (uniformly sampled):
- **Sub-dataset 1 (ordinary):** PS ≈ 360 factual + 120 reflective (≈10k tokens/session); OS ≈ 280 factual + 60 reflective (≈1k tokens/list).
- **Sub-dataset 2 (100k):** PS ≈ 90 factual + 30 reflective (≈100k tokens/session); OS ≈ 84 factual + 15 reflective (≈10k tokens/list).

**Time-aware memory loop.** Rather than long-context dumping, content is fed turn-by-turn: at round *t* only the round-*t* message is input; rounds < *t* must come back through the memory mechanism. This matches how a deployed agent's memory actually works.

**Seven memory mechanisms** (implemented on **MemEngine**, agent base model **Qwen2.5-7B-Instruct**, retriever **multilingual-e5-small**):
FullMemory, RecentMemory, RetrievalMemory, GenerativeAgent (Park 2023), MemoryBank (Zhong 2024), MemGPT (Packer 2023), Self-Controlled Memory / SCMemory (Wang 2023). Action modules are left untouched to isolate the memory effect.

---

## Key Experimental Results

### Factual memory (Table 3, accuracy)

| Mechanism | PS-Acc 10k | PS-Acc 100k | OS-Acc 1k | OS-Acc 100k |
|---|---|---|---|---|
| FullMemory | 0.647 | 0.489 | 0.786 | 0.631 |
| RecentMemory | 0.639 | 0.422 | 0.800 | 0.512 |
| **RetrievalMemory** | **0.692** | **0.833** | **0.883** | **0.933** |
| GenerativeAgent | 0.478 | 0.455 | 0.779 | 0.476 |
| MemoryBank | 0.442 | 0.456 | 0.721 | 0.488 |
| MemGPT | 0.455 | 0.411 | 0.789 | 0.488 |
| SCMemory | 0.355 | 0.444 | 0.529 | 0.429 |

- On the small set, **FullMemory / RetrievalMemory / RecentMemory lead** — the more elaborate mechanisms (GenerativeAgent, MemoryBank, MemGPT, SCMemory) don't justify their complexity.
- At **100k**, FullMemory and especially RecentMemory **drop** (target falls outside the window); RecentMemory drops most (smallest window). **RetrievalMemory is the only mechanism that improves with scale** (0.692→0.833 PS, 0.883→0.933 OS) — its retrieval isolates the right evidence regardless of total length.
- RetrievalMemory **Recall@10**: PS 0.776 (10k) / 0.749 (100k); OS 0.847 (10k) / 0.769 (100k).

### Efficiency (Table 3, seconds per op)

FullMemory / RecentMemory are essentially free (RT/WT < 0.001–0.001s). The elaborate mechanisms are costly:
- **MemGPT** has the worst **read** time — RT **4.549s** (PS, 10k).
- **MemoryBank** has the worst **write** time — WT **8.047s** (PS, 100k) and **18.243s** (OS, 100k).
- GenerativeAgent WT ≈ 6.1s (PS) / 6.2s (OS) at 100k. SCMemory RT 1.531s (PS, 10k).

So accuracy and latency point opposite directions: the simple retrieval/full mechanisms are both **more accurate and far cheaper** than the cognitively-elaborate ones, on this benchmark.

### Reflective memory (Table 4, accuracy)

| Mechanism | PS-Acc 10k | PS-Acc 100k | OS-Acc 1k | OS-Acc 100k |
|---|---|---|---|---|
| FullMemory | 0.733 | 0.533 | 0.883 | 0.333 |
| RecentMemory | 0.700 | 0.333 | 0.867 | 0.400 |
| **RetrievalMemory** | 0.692 | **0.833** | 0.883 | **0.933** |
| GenerativeAgent | 0.742 | 0.333 | 0.883 | 0.200 |
| MemoryBank | 0.692 | 0.400 | 0.900 | 0.333 |
| MemGPT | 0.733 | 0.367 | 0.883 | 0.200 |
| SCMemory | 0.542 | 0.267 | 0.783 | 0.333 |

- On the small set GenerativeAgent / MemGPT / MemoryBank actually do **well** on reflective memory (0.73–0.90) — evidence that well-designed mechanisms *can* capture inferred high-level traits.
- But at **100k they collapse** (0.20–0.40) — limited context windows + built-in **forgetting** mechanisms drop the dispersed low-level cues needed to infer a preference. Again only **RetrievalMemory holds up** (0.833 PS / 0.933 OS).
- Takeaway: capturing reflective memory is feasible, but **sustaining it over long interactions is an open problem.**

### Capacity (Section 4.4, Figure 5)

On the 100k observation set, accuracy is tracked round-by-round after the key-evidence turn as tokens grow to ~140k. **MemGPT and SCMemory show a sharp accuracy cliff** — an upper limit on how much these mechanisms can retain under Qwen2.5-7B-Instruct. (GenerativeAgent and RecentMemory curves also shown.)

### Base-model comparison (Table 5, Sub-dataset 1)

Re-running FullMemory / RecentMemory / RetrievalMemory / GenerativeAgent under **Qwen2.5-7B-Instruct, GPT-4o-mini, Llama-3.1-8B-Instruct, GLM-4-9B-chat**:
- Base model strongly affects results under the same window; **GPT-4o-mini is usually best** (e.g. factual-PS FullMemory 0.736 vs Qwen 0.647).
- **Llama-3.1-8B is weak on factual memory but still decent on reflective memory** — the two abilities don't move together.
- Quirk: GenerativeAgent + GPT-4o-mini is markedly **slower** than other models; otherwise inter-model latency differences are small.

---

## Key Takeaways

1. **Simple beats elaborate, here.** Plain **RetrievalMemory** dominates — highest accuracy on both factual and reflective tasks, the only mechanism that *improves* as memory scales to 100k, and cheap. The cognitively-motivated systems (GenerativeAgent, MemoryBank, MemGPT, SCMemory) underperform and are 1–4 orders of magnitude slower on read or write.
2. **Reflective memory is real but fragile.** Mechanisms can infer high-level preferences/emotions at small scale but lose them at 100k — forgetting + small windows wipe out the scattered cues. This is the headline new capability MemBench measures that prior benchmarks couldn't.
3. **Effectiveness ≠ efficiency ≠ capacity.** Accuracy alone hides that MemoryBank's writes cost ~18s and MemGPT/SCMemory accuracy falls off a capacity cliff. Reporting RT/WT/capacity surfaces real deployment trade-offs.
4. **Observation ≠ participation.** Splitting the two scenarios (and forcing the agent to also remember its *own* recommendations in PS) is a dimension LOCOMO/LongMemEval don't have.
5. **It's a benchmark paper, not a system.** Like LOCOMO, it measures and exposes gaps (forgetting kills long-horizon reflective memory; complexity rarely pays off) rather than proposing a new memory mechanism.

**Limitations (stated):** evaluation is over *structured* synthetic data (graph-derived profiles), so it tests structured-memory ability rather than fully open dialogue; reflective memory beyond preference/emotion (e.g. richer emotional memory) is left for future work.

---

## Where it sits (v1/v2)

MemBench is a **v2 benchmark** whose contribution is **broadening what "memory evaluation" measures**, not raising conversational difficulty. The v1 diagnostic line (LOCOMO) and the early-v2 LongMemEval both evaluate **factual recall over participation-scenario conversational QA** (single-/multi-hop, temporal, etc., scored by F1). MemBench keeps that and adds three orthogonal axes none of them have:

- **Reflective / experiential memory** — inferring un-stated high-level preferences and emotions from many low-level expressions. LOCOMO and LongMemEval contain *no* reflective-memory tasks; this is the headline capability MemBench measures that they don't.
- **Observation scenario** — a passive third-person message stream, in addition to first-person participation.
- **Mechanism-level efficiency + capacity** — read/write latency and the token threshold where accuracy collapses, evaluated through the agent's *actual* turn-by-turn store/retrieve loop (via MemEngine/MemSim) rather than long-context dumping.

**Relation to MemoryAgentBench** (also in this collection, *incremental multi-turn*): the two are complementary v2 evaluation efforts pushing past LOCOMO from different sides. MemoryAgentBench stresses the **incremental, multi-turn ingestion** process — how memory behaves as information arrives turn-by-turn and must be updated/competed over time. MemBench's time-aware loop shares that "feed it incrementally, recall only via the mechanism" philosophy, but its distinctive axes are the **factual-vs-reflective level split** and the **participation-vs-observation scenario split**, plus explicit **efficiency/capacity** metrics. Where MemoryAgentBench asks *"can the memory keep up with a stream of updates?"*, MemBench asks *"can it also infer what was never said, as an observer, cheaply, at 100k tokens?"*

**Relation to the survey's benchmark table** (*Memory in the Age of AI Agents*, Dec 2025): the survey lists MemBench at **~53,000 samples / "interactive scenarios"**, alongside LoCoMo (~300 samples, conversational) and LongMemEval (5 tasks / ~500 samples, interactive). MemBench is by far the **largest-scale** of the three and the only one tagged for *interactive scenarios* in the dual (participation + observation) sense. Its finding — that a plain retrieval store outperforms the elaborate cognitively-grounded mechanisms (MemoryBank, MemGPT, GenerativeAgent) once memory is long, and that those mechanisms' forgetting collapses reflective memory — is a useful empirical counterweight to the v2 trend of ever-more-elaborate memory architectures: it argues that **efficiency and long-horizon retention, not mechanism complexity, are where current systems actually fall down.**
