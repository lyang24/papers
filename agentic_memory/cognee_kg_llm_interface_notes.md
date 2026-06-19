# Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning

**Authors:** Vasilije Marković, Lazar Obradović, Laszlo Hajdu, Jovan Pavlović (Cognee Inc.; Innorenew CoE; University of Primorska, FAMNIT)

**Paper:** arXiv:2505.24478v1 (May 2025) — "This is a preliminary version. A revised and expanded version is in preparation."

**GitHub:** https://github.com/cognee-ai/cognee

---

## The Core Problem

Integrating LLMs with Knowledge Graphs (KGs) — the "GraphRAG" family of systems — produces complex, modular pipelines with **many interacting hyperparameters** (chunk size, retriever type, top-k, prompt templates, graph-construction prompts) that directly affect downstream performance. While hyperparameter optimization (HPO) has been studied for *classical* RAG, its role in **graph-enhanced** pipelines is underexamined.

The paper does not propose a new architecture. Instead it asks an empirical question: **how much performance can you recover from a KG+LLM memory system purely by tuning configuration, without touching the architecture?** And, just as importantly, it interrogates whether standard QA metrics (EM, F1) are even adequate to measure that progress.

The study is conducted inside **Cognee**, an open-source modular framework for end-to-end KG construction and retrieval, whose clean component separation makes it well-suited to controlled optimization experiments — the whole pipeline can be treated as a single objective function.

---

## The Big Idea: Treat the Whole Pipeline as a Tunable Objective Function

Cognee is built around a modular **Extract–Cognify–Load (ECL)** pipeline. The term *cognify* (coined by Kevin Kelly) describes "adding intelligence to already digitized systems"; in Cognee it means transforming unstructured input into structured, semantically grounded graph representations.

| Stage | What it does |
|---|---|
| **Extract** | Ingests heterogeneous inputs (text, PDFs, images, audio transcripts, source code), normalizes and deduplicates them, records metadata in a relational DB |
| **Cognify** | Chunks documents, then applies schema-based transformations using **Pydantic models** to extract entities, relations, attributes, and summaries via an LLM, assembling graph fragments linked to their source |
| **Load** | Writes outputs to three backends: a **graph DB** (entity/relation queries), a **relational store** (metadata), and a **vector index** (similarity search) |

Because every stage is independently configurable and replaceable, the authors expose the entire ECL pipeline plus retrieval and generation as a **parameterized process**, then drive it with a standard HPO algorithm. They name this optimization layer **Dreamify**.

The conceptual claim: in modular memory systems, **"cognification" comes not from design alone but from how systems are tuned, measured, and adapted over time.**

---

## Architecture

### Default Cognee Pipeline (ECL stages)

| Stage | Description |
|---|---|
| **Ingestion** | Load and normalize raw inputs into file or object storage |
| **Tagging** | Classify by media type, merge metadata, deduplicate (content hashes), organize into datasets |
| **Chunking** | Segment documents into token-limited chunks via a configurable strategy |
| **Graph Construction** | LLM fills structured schema objects (entities, relations, summaries); converts to graph fragments |
| **Indexing** | Write outputs to graph, relational, and vector stores |

An orchestration layer manages input validation, scheduling, and endpoint checks. The system ships as a Python package with containerized deployment and a browser UI.

### Retrieval Strategies (single config switch)

Cognee offers several retrievers selectable by one parameter, ranging from pure vector retrieval to graph-structure-aware completion:

| Retriever | Description |
|---|---|
| **Summary-Based** | Retrieves chunk-level summaries via semantic similarity |
| **Chunk-Level** | Retrieves original text chunks by embedding similarity |
| **Graph Neighborhood** | Retrieves nodes adjacent to a matched graph entity |
| **RAG** | Passes retrieved text chunks to an LLM for answer generation (`cognee_completion`) |
| **Graph Completion** | Retrieves graph triples + LLM generation (`cognee_graph_completion`) |
| **Graph-Summary Completion** | Summarizes a subgraph with an LLM before generating |

The two strategies actually compared in the study are **`cognee_completion`** (vector-retrieved text chunks → LLM) and **`cognee_graph_completion`** (vector similarity + graph structure → triplets formatted as structured text → LLM, emphasizing relational context for multi-hop reasoning).

### Built-in Evaluation Framework

A four-stage pipeline driven by one declarative config file: **corpus construction → question answering → answer evaluation → metric aggregation**. Before each run, all memory layers are cleared and documents are reprocessed. Supports structured LLM-based grading (e.g., **GEval / DeepEval**) on correctness, EM, F1, and contextual coverage, plus direct LLM comparison. Results are reported with **bootstrap mean estimates and confidence intervals** plus an interactive dashboard.

---

## The Optimization Setup (Dreamify)

**Six tunable parameters** were optimized:

| Parameter | Description | Range / Options |
|---|---|---|
| **Chunk size** | Tokens per document segment used during graph extraction | 200–2000 tokens |
| **Retriever type** (`search_type`) | Text chunks (`cognee_completion`) vs. graph triplets (`cognee_graph_completion`) | 2 strategies |
| **Top-k** | Number of retrieved items passed to the LLM | 1–20 |
| **QA prompt** (`qa_system_prompt`) | Answer-generation instruction template | 3 variants (tone/verbosity) |
| **Graph prompt** (`graph_prompt`) | Template guiding entity/relation extraction | 3 variants (single-step vs. incremental) |
| **Task getter** (`task_getter_type`) | Dataset preprocessing / whether document summaries are generated for the retriever | summary vs. no-summary |

**Optimizer:** Tree-structured Parzen Estimator (TPE) — chosen for mixed categorical + ordered-integer search spaces. Grid search was impractical at this scale; random search underperformed in early tests.

**Experimental protocol:**
- **9 experiments** = 3 datasets (HotPotQA, TwoWikiMultiHop, MuSiQue) × 3 metrics (EM, F1, DeepEval correctness)
- **50 trials per experiment**; each trial is a full pipeline run (ingestion → graph construction → retrieval → generation), ~30 min/trial, run sequentially
- Per dataset, a **manually filtered** subset of **24 training + 12 test** instances (excluding ungrammatical, ambiguous, mislabeled, or unsupported examples — filtering done once, before tuning, to avoid cherry-picking)
- A single merged KG was built per trial from all training-set passages
- EM/F1 computed deterministically; correctness via DeepEval (LLM-as-judge against gold reference)
- Final results report **best-config performance on the held-out test set**, with **non-parametric bootstrap** confidence intervals

---

## Results

### Training-Set Performance (Table 2)

Baseline = default (heuristically chosen) config; Optimized = best of 50 TPE trials. Relative gain is the percentage increase from baseline (undefined where baseline = 0).

| Benchmark | Metric | Baseline | Optimized | Relative Gain (%) |
|---|---|---|---|---|
| MuSiQue | Correctness | 0.414 | 0.674 | 62.8 |
| MuSiQue | EM | 0.000 | 0.500 | — (baseline 0) |
| MuSiQue | F1 | 0.145 | 0.654 | 351.0 |
| TwoWikiMultiHop | Correctness | 0.348 | 0.582 | 67.2 |
| TwoWikiMultiHop | EM | 0.000 | 0.458 | — (baseline 0) |
| TwoWikiMultiHop | F1 | 0.148 | 0.625 | 321.6 |
| HotPotQA | Correctness | 0.476 | 0.815 | 71.2 |
| HotPotQA | EM | 0.042 | 0.667 | 1496.0 |
| HotPotQA | F1 | 0.169 | 0.840 | 396.7 |

Optimization improved **every** dataset/metric pair. The eye-popping EM/F1 gains are partly an artifact: the default config was tuned for **conversational** output, while these benchmarks reward **short, dry answers**, so strict EM penalized factually-correct-but-verbose baselines (several baselines were 0). The improvement is therefore as much about *answer-style alignment via prompt tuning* as about retrieval quality.

### Hold-Out (Generalization) Performance (Table 3)

Best config from each experiment, evaluated on the unseen 12-instance test set:

| Benchmark | Metric | Train Set | Hold-Out Set |
|---|---|---|---|
| HotPotQA | EM | 0.667 | 0.583 |
| HotPotQA | Correctness | 0.815 | 0.715 |
| HotPotQA | F1 | 0.840 | 0.819 |
| MuSiQue | EM | 0.500 | 0.375 |
| MuSiQue | Correctness | 0.674 | 0.596 |
| MuSiQue | F1 | 0.654 | 0.581 |
| TwoWikiMultiHop | EM | 0.458 | 0.417 |
| TwoWikiMultiHop | Correctness | 0.582 | 0.482 |
| TwoWikiMultiHop | F1 | 0.625 | 0.704 |

Gains **persisted on held-out data** but were generally smaller than in training — most metrics degraded moderately (expected, given tiny hold-out sets and no early stopping / regularization). In one case (**F1 on TwoWikiMultiHop**) test (0.704) actually *exceeded* train (0.625). Takeaway: task-specific tuning generalizes reasonably well within a benchmark.

---

## Key Findings & Discussion

1. **Systematic tuning yields consistent, generalizable gains** without any architectural change — configuration-level changes alone meaningfully move downstream performance.

2. **Gains are non-uniform and task-specific.** No single configuration was best across all benchmarks; effects were largely nonlinear. "Generalization across tasks requires adaptation, not just reuse." High-performing configs *did* tend to share **chunk size** and **retrieval method** settings.

3. **Standard metrics are part of the problem.** EM and F1 routinely penalize semantically correct but differently phrased answers. The LLM-based correctness grader (DeepEval) is more tolerant of lexical variation but **introduces its own noise** — several near-verbatim answers got less than full credit due to format sensitivity and implicit assumptions. This variability "highlights both the value of tuning and the limitations of standard evaluation measures."

4. **Prompt design is a major lever**, especially for EM/F1: constrained, direct prompts aligned outputs with expected answer format. Graph-construction prompts (single-step vs. incremental extraction) changed graph granularity and consistency.

5. **TPE was effective but trial-level performance was volatile** — more stable/expressive optimizers are flagged as future work.

---

## Key Takeaways

1. **A modular ECL memory pipeline can be treated as a black-box objective function** and tuned with off-the-shelf HPO (TPE/Dreamify). Modularity is what makes the optimization tractable.

2. **Configuration, not just architecture, is a first-class lever** for KG+LLM reasoning systems. The paper's thesis is that future progress depends "not only on architectural advances but also on clearer frameworks for optimization and evaluation."

3. **Graph completion vs. plain RAG is one tunable knob among six** — the study frames structured graph retrieval as a configuration choice rather than an always-on win, and lets the optimizer decide per task.

4. **Evaluation is unsolved for graph-based memory.** QA metrics (EM/F1) under-credit correct-but-rephrased answers; LLM-judges (DeepEval/GEval) trade lexical rigidity for grader noise. The authors call for leaderboards / shared benchmark infrastructure for graph-augmented RAG.

5. **"Cognification" is a process, not a design** — intelligence in these systems emerges through how they are tuned, measured, and adapted over time.

---

## Limitations (Acknowledged by Authors)

1. **Tiny evaluation sets** — 24 train / 12 test instances per dataset. The authors explicitly attribute hold-out variability to small set sizes and uneven benchmark QA quality.

2. **No early stopping or regularization** — a deliberately simple training setup, which likely explains part of the train→test degradation; more robust tuning regimes are left to future work.

3. **Metric noise both ways** — strict EM/F1 under-credit correct answers; the LLM-as-judge (DeepEval) introduces format-sensitivity inconsistencies. QA-based metrics "do not fully capture the complexity of graph-based systems."

4. **Single optimizer** — only TPE tested; trial-level volatility suggests other search strategies (Bayesian, RL-based, multi-objective) may do better.

5. **Narrow evaluation scope** — only three well-known multi-hop QA benchmarks; custom/domain-specific tasks and broader parameter spaces are flagged as needed to probe true generalization.

6. **Preliminary version** — the paper itself notes a revised/expanded version is in preparation.

---

## Where it sits (v1/v2)

Cognee belongs to the **v1 foundational wave (2023–2025) of open-source, production-oriented memory frameworks** that turn unstructured data into queryable knowledge graphs for LLM retrieval — the same cohort as **Zep/Graphiti**, **Mem0**, **GraphRAG (Microsoft)**, and **HippoRAG**. Like those systems it adopts the now-standard "build a KG, then do hybrid graph+vector retrieval" recipe, and like Zep/Graphiti it's a real, deployable Python framework (containerized, with a UI) rather than a research prototype.

What distinguishes Cognee within this group is its emphasis on a **tunable, modular ECL pipeline treated as an optimization target**. Where **MAGMA** (v2, 2026) advances the *architecture* — disentangling temporal/causal/semantic/entity relations into orthogonal graph layers with intent-aware traversal — Cognee's contribution is methodological: it argues that a large fraction of achievable performance in any KG+LLM memory system is locked behind **configuration choices** (chunking, retriever, prompts, graph-construction templates), and provides the Dreamify HPO layer to unlock it. It also surfaces a problem that the more architecture-focused v2 papers must still contend with: **the evaluation metrics themselves (EM/F1, LLM-as-judge) are too noisy to cleanly measure progress** in modular graph-memory systems.

In short: among KG-based memory systems, Cognee is the "infrastructure + optimization + evaluation harness" entry — closer in spirit to a production framework (Zep/Graphiti, Mem0) than to a single architectural novelty (MAGMA, GraphRAG's community-summarization), and notable for making *tuning and measurement*, rather than a new model, the central object of study.
