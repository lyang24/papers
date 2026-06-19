# MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents

**Authors:** Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li (University of Texas at Dallas, University of Florida)

**Paper:** arXiv:2601.03236v1 (Jan 2026)

**GitHub:** https://github.com/FredJiang0324/MAMGA

---

## The Core Problem

Existing Memory-Augmented Generation (MAG) systems store past interactions in **monolithic repositories or minimally structured memory buffers**, relying primarily on semantic similarity, recency, or heuristic scoring to retrieve relevant content. This design **entangles** temporal, causal, and entity information into a single undifferentiated store, which means:

1. The system can retrieve *what* occurred but struggles to reason about *why* (no explicit causal structure)
2. Temporal expressions like "last Friday" are not grounded to absolute timestamps
3. Entity permanence is lost — the same person mentioned across disjoint timeline segments may not be linked
4. Retrieval is driven purely by associative proximity (semantic similarity) rather than mechanistic dependency

For example, A-MEM organizes memories into Zettelkasten-like linked notes, but retrieval still relies on semantic embedding similarity, missing temporal and causal relationships. GraphRAG and Zep use knowledge graphs but still organize memory around associative proximity rather than explicitly modeling distinct relational dimensions.

---

## The Big Idea: Disentangled Multi-Graph Memory

MAGMA represents each memory item (event node) across **four orthogonal relational graphs** simultaneously, yielding a disentangled representation of how events, concepts, and participants are related:

| Graph | Edge Type | What It Captures | Example |
|---|---|---|---|
| **Temporal** | Strictly ordered pairs (τ_i < τ_j) | Chronological ordering | Event A happened before Event B |
| **Causal** | Directed logical entailment | Why something happened | Event A *caused* Event B |
| **Semantic** | Undirected, cosine sim > θ | Conceptual similarity | Events A and B are about the same topic |
| **Entity** | Events linked to abstract entity nodes | Object permanence across timeline | "James" in session 1 = "James" in session 5 |

The key insight: **these four relational dimensions are orthogonal** — knowing two events are semantically similar tells you nothing about their causal relationship or temporal ordering. By maintaining them as separate graph layers over a shared node set, MAGMA can route retrieval through the right relational dimension for each query type.

---

## Architecture

MAGMA is organized into three logical layers:

### 1. Data Structure Layer

The unified storage substrate combining:

- **A Vector Database** — for dense semantic search over event embeddings
- **Four Relation Graphs** — sharing a common node set N but with separate edge sets (E_temporal, E_causal, E_semantic, E_entity)

Each **Event-Node** n_i is defined as:
```
n_i = ⟨c_i, τ_i, v_i, A_i⟩
```
where c_i is the event content, τ_i is a discrete timestamp, v_i ∈ R^d is a dense embedding, and A_i is a structured attribute set (entity references, temporal cues, contextual descriptors).

### 2. Query Process (Retrieval Pipeline)

A four-stage adaptive hierarchical retrieval system:

**Stage 1 — Query Analysis & Decomposition:**
- **Intent Classification:** A lightweight classifier maps the query to an intent type T_q ∈ {WHY, WHEN, ENTITY}, which acts as a "steering wheel" — WHY queries prioritize causal edges, WHEN queries prioritize temporal edges, etc.
- **Temporal Parsing:** Resolves relative time expressions ("last Friday") to absolute timestamps for hard time-window filtering.
- **Representation Extraction:** Generates both a dense embedding (for semantic search) and sparse keywords (for exact lexical matching).

**Stage 2 — Multi-Signal Anchor Identification:**
Rather than starting retrieval from the entire graph, MAGMA first identifies a small set of **anchor nodes** as entry points. These are found by fusing three signals using **Reciprocal Rank Fusion (RRF)**:
```
S_anchor = Top_K( Σ_{m ∈ {vec, key, time}} 1/(k + r_m(n)) )
```
This ensures robust starting points regardless of query modality (a purely temporal query won't fail just because the semantic embedding doesn't match well).

**Stage 3 — Adaptive Traversal Policy (Heuristic Beam Search):**
Starting from anchor nodes, MAGMA expands context via a heuristic beam search. At each step, it calculates a **dynamic transition score** for moving from node n_i to neighbor n_j:
```
S(n_j | n_i, q) = exp(λ₁ · φ(type(e_ij), T_q) + λ₂ · sim(n_j, q))
                       ↑ Structural Alignment    ↑ Semantic Affinity
```
The structural alignment function φ **dynamically rewards edge types** based on the detected query intent — e.g., for WHY queries, causal edges get high weight; for WHEN queries, temporal edges dominate. This is the core mechanism that makes retrieval intent-aware rather than purely similarity-driven.

The beam search retains the top-k nodes at each step, bounded by a max depth (5 hops) and max nodes (200) budget.

**Stage 4 — Narrative Synthesis via Graph Linearization:**
The retrieved subgraph G_sub is transformed into a coherent narrative context through:
1. **Topological Ordering** — WHEN queries sort by timestamp; WHY queries do a topological sort on causal edges (causes precede effects)
2. **Context Scaffolding with Provenance** — Each node is serialized as a structured block with timestamp, content, and explicit reference ID, forcing the LLM to act as an interpreter of evidence rather than a creative writer
3. **Salience-Based Token Budgeting** — High-relevance nodes retain full semantic detail; low-relevance nodes are summarized into brevity codes ("...3 intermediate events...")

### 3. Write/Update Process (Memory Evolution)

A **dual-stream pipeline** inspired by Complementary Learning Systems theory (Kumaran et al., 2016):

**Fast Path (Synaptic Ingestion):**
Operates on the critical path with strict latency requirements. Performs only non-blocking operations:
1. Event segmentation (extract structured metadata from raw interaction)
2. Dense/sparse embedding and vector indexing
3. Update the immutable temporal backbone (n_{t-1} → n_t)
4. Enqueue the new node for slow-path processing

No LLM reasoning occurs here — the agent remains responsive regardless of memory size.

**Slow Path (Asynchronous Consolidation):**
A background worker that dequeues events and densifies the graph structure:
1. Retrieve the local neighborhood N(n_t) within 2 hops
2. Use an LLM (Φ) to infer latent connections:
   ```
   E_new = Φ_reason(N(n_t), H_history)
   ```
3. Add high-value causal (E_causal) and entity (E_entity) links to the graph

This effectively trades compute time for relational depth — the causal and entity structure deepens over time without blocking the agent's responsiveness.

---

## Experimental Results

### LoCoMo Benchmark (Long-Context Conversational Memory)

Ultra-long conversations (~9K tokens avg), testing long-range temporal and causal retrieval. All methods use gpt-4o-mini.

| Method | Multi-Hop | Temporal | Open-Domain | Single-Hop | Adversarial | **Overall** |
|---|---|---|---|---|---|---|
| Full Context | 0.468 | 0.562 | 0.486 | 0.630 | 0.205 | 0.481 |
| A-MEM | 0.495 | 0.474 | 0.385 | 0.653 | 0.616 | 0.580 |
| MemoryOS | 0.552 | 0.422 | 0.504 | 0.674 | 0.428 | 0.553 |
| Nemori | 0.569 | 0.649 | 0.485 | 0.764 | 0.325 | 0.590 |
| **MAGMA** | 0.528 | **0.650** | **0.517** | **0.776** | **0.742** | **0.700** |

Key findings:
- MAGMA outperforms the best baseline (Nemori) by **18.6%** overall (0.700 vs 0.590)
- The biggest gap is on **adversarial queries** (0.742 vs 0.325-0.616) — the Adaptive Traversal Policy avoids semantically similar but structurally irrelevant distractors
- Strong temporal reasoning (0.650) validates the Temporal Inference Engine's ability to ground relative time expressions

### LongMemEval Benchmark (100K+ token stress test)

| Question Type | Full-context (101K tokens) | Nemori (3.7-4.8K) | **MAGMA (0.7-4.2K)** |
|---|---|---|---|
| single-session-preference | 6.7% | 62.7% | **73.3%** |
| single-session-assistant | **89.3%** | 73.2% | 83.9% |
| temporal-reasoning | 42.1% | 43.0% | **45.1%** |
| multi-session | 38.3% | **51.4%** | 50.4% |
| knowledge-update | 78.2% | 52.6% | **66.7%** |
| single-session-user | **78.6%** | 77.7% | 72.9% |
| **Average** | 55.0% | 56.2% | **61.2%** |

MAGMA achieves highest average accuracy (61.2%) while using only **0.7-4.2K tokens per query** — a **>95% reduction** compared to the full-context baseline's 101K tokens.

### System Efficiency

| Method | Build Time (h) | Tokens/Query (k) | Latency (s) |
|---|---|---|---|
| Full Context | N/A | 8.53 | 1.74 |
| A-MEM | 1.01 | **2.62** | 2.26 |
| MemoryOS | 0.91 | 4.76 | 32.68 |
| Nemori | **0.29** | 3.46 | 2.59 |
| **MAGMA** | 0.39 | 3.37 | **1.47** |

MAGMA achieves the **lowest query latency** (1.47s) — ~40% faster than the next best retrieval baseline (A-MEM at 2.26s) — thanks to the Adaptive Traversal Policy pruning irrelevant subgraphs early, and the dual-stream architecture offloading complex indexing to the background.

---

## Ablation Study

| MAGMA Variant | Judge Score | F1 | BLEU-1 |
|---|---|---|---|
| w/o Adaptive Policy | 0.637 | 0.413 | 0.357 |
| w/o Causal Links | 0.644 | 0.439 | 0.354 |
| w/o Temporal Backbone | 0.647 | 0.438 | 0.349 |
| w/o Entity Links | 0.666 | 0.451 | 0.363 |
| **MAGMA (Full)** | **0.700** | **0.467** | **0.378** |

Three key findings:
1. **Adaptive Traversal Policy** is the most critical component (largest drop: 0.700 → 0.637) — without it, retrieval degenerates into a static graph walk that introduces structurally irrelevant information
2. **Causal Links** and **Temporal Backbone** are complementary, non-substitutable axes of reasoning (comparable drops when either is removed)
3. **Entity Links** contribute a smaller but consistent improvement — they maintain entity permanence and reduce hallucinations in entity-centric queries

---

## Case Studies (from LoCoMo)

### Case 1: Fact Retrieval — "What instruments does Melanie play?"
- **A-MEM:** Failed — summarization abstracted away "violin" from early sessions
- **MemoryOS:** Partial — only retrieved "clarinet" via surface-level semantic matching
- **MAGMA:** Correct ("Clarinet and Violin") — traversed the entity-centric subgraph around [Entity: Melanie], aggregating diverse natural language predicates ("playing", "enjoy") across disjoint sessions

### Case 2: Multi-Hop Logic — "How many children does Melanie have?"
- **Baselines:** Extracted "two children" from a photo caption, missing the "son" mentioned separately in an accident context
- **MAGMA:** Correctly deduced "at least three" by resolving entity links between Node A (Photo: "two kids"), Node B (Accident: "son"), and Node C (Dialogue: "brother"), synthesizing across multiple entity nodes

### Case 3: Temporal Grounding — "When did she hike after the roadtrip?"
- **A-MEM:** Copied the session timestamp ("20 October 2023")
- **MemoryOS:** Hallucinated a future date ("29 December 2025")
- **MAGMA:** Correct ("19 October 2023") — the Temporal Parser identified "yesterday" in the dialogue, computed T_session(Oct20) - 1 day = Oct 19, and anchored this resolved date to the Event Node

---

## Key Takeaways

1. **Disentangling relational dimensions is powerful.** Separating temporal, causal, semantic, and entity relations into orthogonal graph layers, rather than collapsing everything into a single similarity-based store, enables fundamentally different types of reasoning (chronological, causal, factual).

2. **Intent-aware retrieval routing** (the Adaptive Traversal Policy) is the single most impactful component — it ensures the retrieval path follows the relational structure that matches the query's logical needs, rather than always defaulting to semantic similarity.

3. **The dual-stream write architecture** (fast synaptic ingestion + slow asynchronous consolidation) is an elegant solution for balancing responsiveness with structural depth, inspired by Complementary Learning Systems theory from neuroscience.

4. **Graph linearization matters.** The way retrieved subgraphs are serialized into the LLM prompt — topological ordering, provenance scaffolding, salience-based token budgeting — significantly impacts reasoning quality by forcing the LLM to be an evidence interpreter rather than a creative writer.

5. **>95% token reduction** with competitive or better accuracy shows that structured multi-graph retrieval can be dramatically more efficient than dumping full context into the LLM.

---

## Limitations (Acknowledged by Authors)

1. **LLM dependency for consolidation** — causal and entity link inference depends on the underlying LLM's reasoning fidelity; erroneous or missing relations may propagate to downstream retrieval
2. **Engineering complexity** — maintaining four graph layers + dual-stream processing is more complex than flat vector-only memory systems, potentially limiting applicability in resource-constrained environments
3. **Evaluation scope** — primarily evaluated on long-context conversational benchmarks (LoCoMo, LongMemEval); generalization to multimodal agents or heterogeneous observation streams remains future work
