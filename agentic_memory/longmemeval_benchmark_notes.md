# LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory

**Authors:** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu (UCLA, Tencent AI Lab Seattle, UC San Diego)

**Paper:** ICLR 2025 | arXiv:2410.10813 | https://github.com/xiaowu0162/LongMemEval

---

## The Problem

Commercial chat assistants (ChatGPT, Coze) and open-source systems (MemoryBank, etc.) now bolt on "memory" to track user-AI chat histories. But nobody had **holistically** evaluated whether that memory actually works over *sustained* interaction. Existing long-term memory benchmarks have two structural flaws:

1. **They don't reflect real user-AI interaction.** Most evaluate human-human conversation (MSC, LOCOMO, DialSim) or omit task-oriented dialogue. Their histories are also a fixed few-thousand tokens — too short to be hard as systems improve.
2. **Their question coverage is narrow.** MemoryBank and PerLTQA barely test cross-session synthesis or temporal reasoning. *Even LOCOMO* never tests recall of information the **assistant** provided, nor reasoning over **updated** user information.

The question: **do current memory systems and long-context LLMs actually remember across long, realistic, task-oriented histories?** Answer: they drop 30%+.

---

## What They Built

### LongMemEval Dataset

500 manually-created questions, each requiring recall of information hidden inside one or more task-oriented user-assistant dialogues. The history length is **freely configurable** (needle-in-a-haystack style), with two standard settings:

- **LongMemEval_S** — ~115k tokens/question (~50 sessions). The main benchmark.
- **LongMemEval_M** — 500 sessions/question (~1.5M tokens). The stress setting.

#### Comparison to prior long-term memory benchmarks

| Benchmark | Domain | #Sess | #Q | Context Depth | IE | MR | KU | TR | ABS |
|---|---|---|---|---|---|---|---|---|---|
| MSC | Open-Domain | 5k | – | 1k | ✗ | ✗ | ✗ | ✗ | ✗ |
| DuLeMon | Open-Domain | 30k | – | 1k | ✗ | ✗ | ✗ | ✗ | ✗ |
| MemoryBank | Personal | 300 | 194 | 5k | ✓ | ✗ | ✗ | ✓ | ✗ |
| PerLTQA | Personal | 4k | 8593 | 1M* | ✓ | ✗ | ✗ | ✗ | ✓ |
| LoCoMo | Personal | 1k | 7512 | 10k | ✓ | ✓ | ✗ | ✓ | ✓ |
| DialSim | TV Shows | 1k–2k | 1M | 350k | ✓ | ✓** | ✗ | ✓ | ✓ |
| **LongMemEval** | **Personal** | **50k** | **500** | **115k, 1.5M** | **✓** | **✓** | **✓** | **✓** | **✓** |

(*approximated; **at most 2 sessions.) LongMemEval is the only one covering **all five** abilities, and the only one combining a long, freely-extensible, *task-oriented* (user-AI) history with full ability coverage. Note: fewer questions than LOCOMO (500 vs 7,512), but each is harder and embedded in a far longer haystack.

---

## The 5 Core Memory Abilities

The benchmark is organized around five abilities a real personalized assistant needs:

1. **Information Extraction (IE)** — recall specific facts from extensive history, including details mentioned by either the **user or the assistant** (LOCOMO never tests assistant-side recall).
2. **Multi-Session Reasoning (MR)** — synthesize/aggregate/compare information across multiple sessions.
3. **Knowledge Updates (KU)** — recognize when the user's personal info *changes* and update memory accordingly (new in this benchmark vs LOCOMO).
4. **Temporal Reasoning (TR)** — reason over both explicit time mentions and timestamp metadata.
5. **Abstention (ABS)** — recognize unanswerable / false-premise questions and say "I don't know."

### Seven question types

These five abilities map onto seven concrete question types:

- **single-session-user** — recall info the user stated within one session (IE)
- **single-session-assistant** — recall info the **assistant** stated within one session (IE)
- **single-session-preference** — use user info to produce a personalized response
- **multi-session** — aggregate across ≥2 sessions (MR)
- **knowledge-update** — track a changed life state (KU)
- **temporal-reasoning** — reason with timestamps + explicit time references (TR)
- **abstention** — 30 questions drawn from the above types, rewritten as "false premise" questions (ABS)

### Curation pipeline (human-heavy)

- Ontology of **164 user attributes** in five categories (lifestyle, belongings, life events, situational context, demographics).
- An LLM (Llama 3 70B Instruct) drafts seed (question, answer) pairs from attribute-focused background paragraphs; **human experts filter and rewrite every question** for difficulty.
- Answers are manually decomposed into one or more **evidence statements** (with optional timestamps). Most questions need evidence from multiple sessions (**up to six**).
- Each evidence statement is embedded into a task-oriented **evidence session** via self-chat, where the user LLM conveys it **indirectly** (e.g., asks about car insurance to reveal "bought a new car") — making memorization harder.
- All evidence sessions are manually screened: verify inclusion, distribute evidence across positions, rephrase into natural colloquial language (esp. time mentions), annotate evidence positions.
- **History compilation:** sample unrelated user-AI sessions (from self-chat on non-conflicting attributes + ShareGPT + UltraChat), randomly insert evidence sessions, assign plausible timestamps.
- ~400 human-hours on construction, ~150 on the commercial-system study; 3 expert NLP annotators.

### Evaluation metrics

- **QA:** LLM-as-judge (GPT-4o, `gpt-4o-2024-08-06`) — >97% agreement with human experts. Exact match is too brittle for flexible answers.
- **Memory recall:** because evidence locations are human-annotated, **Recall@k and NDCG@k** can be computed for any system that exposes its retrieval results.

---

## Headline Finding: LongMemEval Is Hard

### Commercial memory-augmented assistants collapse vs. offline reading

Because ChatGPT/Coze only expose memory through web UIs, annotators hand-fed **97 questions** in a *much easier* 3–6 session history (~10× shorter than LongMemEval_S), then asked the question in a fresh session. Compared to "Offline Reading" (same GPT-4o reading the full history as plain context):

| System | LLM | Accuracy | Offline Reading (GPT-4o) |
|---|---|---|---|
| ChatGPT | GPT-4o | 0.5773 | 0.9184 |
| ChatGPT | GPT-4o-mini | 0.7113 | – |
| Coze | GPT-4o | 0.3299 | 0.9184 |
| Coze | GPT-3.5-turbo | 0.2474 | – |

GPT-4o-backed ChatGPT drops **~37%** and Coze **~64%** relative to just reading the context. ChatGPT tends to **overwrite** crucial info as chat continues; Coze fails to record **indirectly** provided info. Recalling isolated facts ≠ genuine memory.

### Long-context LLMs drop 30–60% reading the full 115k-token history

Is the benchmark trivial if you just stuff all 115k tokens into a long-context model? No. Compared to the **Oracle** setting (answering with only the evidence sessions as context), reading the *full* LongMemEval_S history costs 30–60%:

**Without Chain-of-Note:**

| Model | Size | Oracle | Full (S) | % Drop |
|---|---|---|---|---|
| GPT-4o | – | 0.870 | 0.606 | 30.3% |
| Llama 3.1 Instruct | 70B | 0.744 | 0.334 | 55.1% |
| Llama 3.1 Instruct | 8B | 0.710 | 0.454 | 36.1% |
| Phi-3 128k Instruct | 14B | 0.702 | 0.380 | 45.9% |
| Phi-3.5 Mini Instruct | 4B | 0.660 | 0.342 | 48.1% |

**With Chain-of-Note:**

| Model | Size | Oracle | Full (S) | % Drop |
|---|---|---|---|---|
| GPT-4o | – | 0.924 | 0.640 | 30.7% |
| Llama 3.1 Instruct | 70B | 0.848 | 0.286 | 66.3% |
| Llama 3.1 Instruct | 8B | 0.710 | 0.420 | 40.8% |
| Phi-3 128k Instruct | 14B | 0.722 | 0.344 | 52.4% |
| Phi-3.5 Mini Instruct | 4B | 0.652 | 0.324 | 50.3% |

Even the best long-context model (GPT-4o) loses ~30 points, and ~50 sessions is *still short* — degradation is expected to worsen as histories grow. Chain-of-Note doesn't rescue the full-history setting (it raises the Oracle ceiling but the drop persists). The "lost-in-the-middle" failure is real.

---

## Unified Memory Framework: 3 Stages, 4 Control Points

The paper's second contribution: a key-value datastore view of memory-augmented assistants with **three stages** — **indexing → retrieval → reading** — and **four control points (CP)**:

- **CP1: Value** — format/granularity of each stored session (whole session? round? extracted facts?)
- **CP2: Key** — what you index on (the value itself? extracted summaries/keyphrases/facts?)
- **CP3: Query** — how the retrieval query is formed (raw question? time-expanded?)
- **CP4: Reading strategy** — how the LLM reads retrieved items (direct? Chain-of-Note? JSON?)

Nine existing systems (In-context RAG, MemoryBank, LD-Agent, CoN, ChatGPT, Coze, RAPTOR, MemWalker, HippoRAG) are recast as instantiations of this framework. Setup for their own experiments: Stella V5 1.5B dense retriever; GPT-4o / Llama 3.1 70B / Llama 3.1 8B readers; Llama 3.1 8B for indexing-time extraction; retrieved items always sorted by timestamp.

### Decomposition results (CP1: Value)

On LongMemEval_M, budget-aware comparison of value granularity:

- **Round > Session.** Decomposing sessions into individual rounds (one user msg + one assistant response) **improves** reading, especially with GPT-4o; roughly neutral with Llama 3.1 8B.
- **Further compressing into summaries/facts hurts overall** QA due to information loss — **except** for **multi-session reasoning**, where **fact decomposition consistently helps** (uniform, simplified facts across sessions aid retrieval/reading).
- Optimal token budget is reader-dependent: Llama 3.1 8B drops sharply beyond ~3k retrieved tokens; GPT-4o keeps improving past 20k.

### Indexing / key-expansion results (CP2: Key)

On LongMemEval_M — using compressed forms (summary/keyphrase/fact) **alone** as keys does *not* beat using the value itself (the retriever already handles long-text semantics). The win comes from **document expansion**: concatenate extracted **facts** with the original value to form the key (multi-pathway retrieval).

| Value = Round | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | GPT-4o Top-5 | GPT-4o Top-10 |
|---|---|---|---|---|---|---|
| K = V | 0.582 | 0.481 | 0.692 | 0.512 | 0.615 | 0.670 |
| K = fact | 0.530 | 0.411 | 0.654 | 0.449 | 0.588 | 0.664 |
| K = keyphrase | 0.282 | 0.159 | 0.392 | 0.303 | 0.425 | 0.489 |
| **K = V + fact** | **0.644** | **0.498** | **0.784** | **0.536** | **0.657** | **0.720** |
| K = V + keyphrase | 0.478 | 0.359 | 0.636 | 0.410 | 0.541 | 0.652 |

| Value = Session | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | GPT-4o Top-5 |
|---|---|---|---|---|---|
| K = V | 0.706 | 0.617 | 0.783 | 0.638 | 0.670 |
| K = summary | 0.572 | 0.448 | 0.648 | 0.468 | 0.554 |
| K = fact | 0.642 | 0.524 | 0.814 | 0.571 | 0.644 |
| **K = V + fact** | **0.732** | **0.620** | **0.862** | **0.652** | **0.714** |
| K = V + keyphrase | 0.710 | 0.587 | 0.768 | 0.602 | 0.665 |

**Fact-augmented key expansion → +9.4% Recall@k and +5.4% final accuracy on average across all models.**

### Time-aware retrieval results (CP3: Query)

Naive time-agnostic retrieval is bad on temporal questions. Fix: index values by the **dates of events they contain**, and at retrieval have an LLM (M_T) extract a **time range** from the query to filter irrelevant values. On the TR subset of LongMemEval_M:

| Key Setting | Round Recall@5 | Round Recall@10 |
|---|---|---|
| K = V | 0.421 | 0.499 |
| K = V w/ Query Expansion (M_T = GPT-4o) | 0.565 | 0.722 |
| K = V w/ Query Expansion (M_T = Llama 3.1 8B) | 0.448 | 0.570 |
| K = V + fact | 0.489 | 0.550 |
| K = V + fact w/ Query Expansion (M_T = GPT-4o) | 0.526 | 0.722 |

**Time-aware query expansion → +11.3% recall (rounds) / +6.8% recall (sessions)** on average. But it **requires a strong LLM** for M_T — Llama 3.1 8B hallucinates time ranges and the gain shrinks.

### Reading results (CP4: Reading Strategy)

Even with **oracle retrieval** (only evidence sessions given), a bad reading strategy costs up to **10 absolute points** for GPT-4o. The combination that wins is **Chain-of-Note + JSON-structured items**:

- Without CoN, JSON format doesn't consistently beat natural language.
- **With CoN, JSON consistently helps readers of all sizes.** CoN decomposes long-context reading into "copy important details" then "reason over concise notes."

---

## Key Takeaways

1. **Recalling isolated facts ≠ memory.** Commercial assistants that "remember" you still drop ~37% (ChatGPT) to ~64% (Coze) vs. just reading the context — they overwrite or miss indirectly-stated info.
2. **Long context is not a substitute for memory.** Even GPT-4o loses ~30 points reading a 115k-token history vs. the oracle, and ~50 sessions is still on the short side.
3. **Round is the sweet-spot value granularity.** Whole sessions are too coarse; over-compressing to facts loses info — except facts help **multi-session reasoning**.
4. **Don't replace the value with a compressed key — augment it.** K = V + fact (document expansion) gives multi-pathway retrieval: +9.4% recall, +5.4% accuracy.
5. **Temporal reasoning needs explicit time handling.** Time-aware indexing + LLM query expansion adds 6.8–11.3% recall, but only with a capable LLM.
6. **Reading is its own bottleneck.** Perfect retrieval still needs Chain-of-Note + structured (JSON) formatting — up to 10 points on the table.
7. **It's a benchmark + design-study paper.** It both measures the gap *and* extracts actionable memory-design guidance across indexing/retrieval/reading.

---

## Where it sits (v1/v2)

LongMemEval is, alongside **LOCOMO**, one of the **two canonical long-term-memory benchmarks** in this collection. Where LOCOMO is the v1 "observations > raw dialog > summaries" diagnostic, LongMemEval became the **default scorecard for v2 systems**: nearly every v2 paper here — **MAGMA, MemGAS, LightMem, EverMemOS**, and others — reports LongMemEval numbers (usually LongMemEval_S) as its headline result. The *Memory in the Age of AI Agents* survey (the v1→v2 boundary) and the unified-pipeline lineage (MemGAS → RF-Mem → MemCoE) both treat LOCOMO + LongMemEval as the paired baselines a new memory system must clear.

**Why it's harder/longer than LOCOMO and why v2 adopted it:**

- **Much longer haystack.** LongMemEval_S is ~115k tokens/question (~50 sessions) and LongMemEval_M is ~1.5M tokens (500 sessions), versus LOCOMO's ~10k-token (~19-session) conversations. It is **freely extensible**, so it doesn't saturate as systems improve — exactly what a v2 benchmark needs.
- **Real user-AI, task-oriented dialogue.** LOCOMO is human-human persona chat; LongMemEval is user↔assistant task-oriented interaction, which is what deployed memory systems actually face (long-context inputs, long-form responses, assistant-side facts).
- **Two abilities LOCOMO can't test.** **Assistant-side information extraction** and **knowledge updates** (tracking changed user state over time) are unique to LongMemEval — precisely the dimensions v2 systems claim to handle via structured/temporal memory (Zep, MAGMA, EverMemOS).
- **Indirect evidence + abstention.** Evidence is conveyed *incidentally* in self-chat, and false-premise questions force genuine abstention — stressing precision, not just recall.
- **Built-in retrieval diagnostics.** Human-annotated evidence locations give Recall@k / NDCG@k, letting v2 systems report *where* in the indexing/retrieval/reading stack they win — the same decomposition (value/key/query/reading) that v2 papers iterate on.

In short: LongMemEval is the **longer, more realistic, more ability-complete** successor benchmark. LOCOMO motivates the problem; LongMemEval is the bar v2 memory systems compete on.
