# Zep: A Temporal Knowledge Graph Architecture for Agent Memory

**Authors:** Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef (Zep AI)

**Paper:** arXiv:2501.13956v1 (Jan 2025)

**GitHub:** https://github.com/getzep/graphiti (Graphiti engine) — Zep is the commercial memory layer at https://www.getzep.com

---

## The Core Problem

Retrieval-Augmented Generation (RAG) was designed for **broad, largely static corpora** — document contents that seldom change. But for agents to become persistent assistants, they need a "memory" over a **continuously evolving** stream of data: ongoing user conversations plus related business and world data. The authors argue current RAG is fundamentally unsuitable for this:

1. Entire conversation histories and business datasets **do not fit inside LLM context windows**, so simply stuffing context does not scale.
2. Static document retrieval has **no mechanism for updating facts** when the world changes — a user who *used to* live in Boston and now lives in Seattle leaves two contradictory facts in a flat store.
3. There is **no notion of when a fact was true** versus when the system learned it, so temporal reasoning ("what was her job *before* she moved?") is impossible.
4. Existing graph-RAG systems (GraphRAG, AriGraph) capture entity structure but **lack a principled temporal model** for fact validity and invalidation.

MemGPT [Packer et al.] introduced the idea of giving agents memory, but it treats memory as a paging problem over a flat archive and does not model evolving, time-stamped relationships.

---

## The Big Idea: A Bi-Temporal, Hierarchical Knowledge Graph

Zep's memory is a temporally-aware dynamic knowledge graph **G = (N, E, φ)** built and maintained by the **Graphiti** engine. Two ideas distinguish it from prior graph-RAG memory:

**1. Bi-temporal modeling.** Every fact carries two independent timelines:
- **T** — the *chronological* timeline of when events actually happened / facts were true in the world (`t_valid`, `t_invalid`).
- **T′** — the *transactional* timeline of when Zep ingested or modified the data (`t_created`, `t_expired`), used for database-style auditing.

This separation is what lets Zep answer "when was X true?" distinctly from "when did we learn X?", and is described by the authors as a novel advance over previous graph-based RAG proposals.

**2. A three-tier hierarchy of subgraphs**, mirroring psychological models that distinguish *episodic* memory (distinct events) from *semantic* memory (concept associations):

| Subgraph | Node type | What it stores | Edges |
|---|---|---|---|
| **Episode** `G_e` | Episodic nodes (raw messages, text, JSON) | Non-lossy raw input — the source of truth | Episodic edges link episodes → the entities they mention |
| **Semantic Entity** `G_s` | Entity nodes (extracted, resolved entities) | Entities + facts (semantic edges) extracted from episodes | Semantic edges = relationships between entity pairs |
| **Community** `G_c` | Community nodes (clusters of entities) | High-level map-reduce summaries of strongly-connected entity clusters | Community edges link communities → member entities |

The hierarchy — episodes → facts → entities → communities — extends hierarchical RAG strategies, draws episodic/semantic separation from **AriGraph**, and borrows community summarization from **GraphRAG**.

---

## Architecture

### Knowledge Graph Construction

**Episodes.** Ingestion begins with raw data units (`message`, `text`, or `JSON`; the paper focuses on `message`). Each message carries a reference timestamp `t_ref`, which lets Zep resolve relative/partial dates in the content ("next Thursday," "in two weeks," "last summer") to absolute datetimes. Episodes are kept verbatim as a **non-lossy store**, with bidirectional indices so semantic artifacts can be traced back to source episodes for citation.

**Entity extraction & resolution.**
- Processes the current message plus the last **n = 4** messages (two full turns) for NER context; the speaker is always extracted as an entity.
- Uses a **reflexion-inspired reflection** step to reduce hallucinations and improve coverage.
- Embeds each entity name into a **1024-dim** vector space; finds candidate duplicates via **cosine similarity + full-text search**, then resolves duplicates through an LLM entity-resolution prompt that emits an updated name/summary.
- Writes to the graph using **predefined Cypher queries** (not LLM-generated queries) to guarantee schema consistency and reduce hallucination.

**Facts (semantic edges).** Facts are extracted as predicates between entity pairs; the same fact can be extracted between multiple entities, letting Graphiti model **multi-entity facts via hyper-edges**. Edge deduplication is constrained to edges between the *same entity pair*, which both prevents bad merges and shrinks the search space.

**Temporal extraction & edge invalidation** — the key differentiator. For each fact, Graphiti tracks four timestamps: `t_created`, `t_expired` (on T′) and `t_valid`, `t_invalid` (on T). When a new edge is added, an LLM compares it against semantically related existing edges to detect **contradictions**. On a temporally-overlapping contradiction, the system **invalidates** the stale edge by setting its `t_invalid` to the new edge's `t_valid` — preserving the historical record while prioritizing newer information along T′. This is how memory updates non-destructively as conversations evolve.

**Communities.** Built via **label propagation** (rather than GraphRAG's Leiden algorithm) specifically because label propagation has a cheap **dynamic single-step extension**: when a new entity arrives, it joins the community held by the plurality of its neighbors, and the summary is updated incrementally. This delays the need for full community refreshes, cutting latency and LLM cost (periodic full refreshes are still needed as communities drift).

### Memory Retrieval

The search API implements **f(α) → β**: a text query α in, a formatted context string β out. It composes three stages, **f(α) = χ(ρ(φ(α)))**:

**1. Search (φ)** — high recall. Three complementary methods over the three textual object types (fact field of semantic edges, entity names, community names):

| Method | Captures | Implementation |
|---|---|---|
| Cosine semantic similarity `φ_cos` | Semantic similarity | Neo4j / Lucene |
| Okapi BM25 full-text `φ_bm25` | Word/lexical similarity | Neo4j / Lucene |
| Breadth-first search `φ_bfs` | Contextual similarity (nodes near in the graph) | n-hop graph traversal, can be seeded with recent episodes |

**2. Reranker (ρ)** — high precision. Supports **Reciprocal Rank Fusion (RRF)** and **Maximal Marginal Relevance (MMR)**, plus graph-native rerankers: an **episode-mentions** reranker (frequently referenced facts surface more easily), a **node-distance** reranker (localizes context around a centroid node), and the most powerful/expensive option, **cross-encoder LLM** relevance scoring.

**3. Constructor (χ)** — serializes results to text: for each semantic edge, the fact + `t_valid`/`t_invalid` range; for each entity, name + summary; for each community, its summary. The output prompt explicitly labels facts with valid date ranges so the LLM can reason temporally.

---

## Experimental Results

**Models:** BGE-m3 (BAAI) for embedding/reranking; `gpt-4o-mini-2024-07-18` for graph construction; `gpt-4o-mini` and `gpt-4o-2024-11-20` for the chat agent; `gpt-4-turbo-2024-04-09` added for direct DMR comparability with MemGPT. For each query, the top relevant edges/nodes are retrieved and formatted into a context string.

### Deep Memory Retrieval (DMR)

DMR (from the MemGPT paper) is a 500-conversation subset of Multi-Session Chat: each conversation has 5 sessions, up to 12 messages/session (~60 messages total), with one Q/A pair. An LLM judge scores responses against golden answers.

| Memory | Model | Score |
|---|---|---|
| Recursive Summarization | gpt-4-turbo | 35.3% |
| Conversation Summaries | gpt-4-turbo | 78.6% |
| MemGPT† | gpt-4-turbo | 93.4% |
| Full-conversation | gpt-4-turbo | 94.4% |
| **Zep** | gpt-4-turbo | **94.8%** |
| Conversation Summaries | gpt-4o-mini | 88.0% |
| Full-conversation | gpt-4o-mini | 98.0% |
| **Zep** | gpt-4o-mini | **98.2%** |

† Reported in the MemGPT paper.

Zep beats MemGPT (94.8% vs 93.4%) and slightly edges the full-conversation baseline on both models. **But the authors openly critique DMR**: only ~60 messages per conversation (fits easily in modern context windows), single-turn fact-retrieval questions, ambiguous phrasing ("favorite drink to relax with"), and poor representation of real enterprise use — the high full-context scores show the benchmark is too easy to discriminate memory systems.

### LongMemEval (LME)

The harder benchmark: conversations average **~115,000 tokens** (the authors quote both 115k and an average of ~115k). Six question types. Answers scored by GPT-4o with LongMemEval's question-specific prompts. Tests run from a Boston laptop against Zep hosted on AWS us-west-2 (so Zep's latency *includes* extra network latency the baselines did not incur). Attempts to evaluate MemGPT on LME failed because its framework couldn't ingest the message histories.

| Memory | Model | Score | Latency | Latency IQR | Avg Context Tokens |
|---|---|---|---|---|---|
| Full-context | gpt-4o-mini | 55.4% | 31.3 s | 8.76 s | 115k |
| **Zep** | gpt-4o-mini | **63.8%** | **3.20 s** | 1.31 s | **1.6k** |
| Full-context | gpt-4o | 60.2% | 28.9 s | 6.01 s | 115k |
| **Zep** | gpt-4o | **71.2%** | **2.58 s** | 0.684 s | **1.6k** |

Headline gains: **+15.2%** accuracy with gpt-4o-mini, **+18.5%** with gpt-4o, while shrinking context from ~115k to ~1.6k tokens (~98% reduction) and cutting latency by roughly **90%**.

### LongMemEval — per-question-type breakdown

| Question Type | Model | Full-context | Zep | Delta |
|---|---|---|---|---|
| single-session-preference | gpt-4o-mini | 30.0% | 53.3% | 77.7% ↑ |
| single-session-assistant | gpt-4o-mini | 81.8% | 75.0% | 9.06% ↓ |
| temporal-reasoning | gpt-4o-mini | 36.5% | 54.1% | 48.2% ↑ |
| multi-session | gpt-4o-mini | 40.6% | 47.4% | 16.7% ↑ |
| knowledge-update | gpt-4o-mini | 76.9% | 74.4% | 3.36% ↓ |
| single-session-user | gpt-4o-mini | 81.4% | 92.9% | 14.1% ↑ |
| single-session-preference | gpt-4o | 20.0% | 56.7% | 184% ↑ |
| single-session-assistant | gpt-4o | 94.6% | 80.4% | 17.7% ↓ |
| temporal-reasoning | gpt-4o | 45.1% | 62.4% | 38.4% ↑ |
| multi-session | gpt-4o | 44.3% | 57.9% | 30.7% ↑ |
| knowledge-update | gpt-4o | 78.2% | 83.3% | 6.52% ↑ |
| single-session-user | gpt-4o | 81.4% | 92.9% | 14.1% ↑ |

Key findings:
- The **largest gains are on the hardest, cross-session question types** — single-session-preference, multi-session, and temporal-reasoning — exactly where the bi-temporal graph should help.
- With the more capable gpt-4o, Zep also improves on **knowledge-update**, validating that better models exploit Zep's temporal/invalidation data more effectively.
- The one consistent **regression is single-session-assistant** (−17.7% gpt-4o, −9.06% gpt-4o-mini) — questions answerable from a single recent assistant turn, where retrieval can drop the exact wording that full-context keeps. The authors flag this as needing further work.

---

## Ablation / Component Notes

The paper does not present a formal component-by-component ablation table. Instead its evaluation deliberately exercises only a **subset** of Graphiti's search functionality (top edges + entity summaries), reserving breadth-first seeding, community retrieval, and the full reranker suite for future work. The reported numbers therefore represent a *floor* rather than the system's full capability. Notable design choices justified in the text rather than ablated:
- **Label propagation over Leiden** for communities — chosen for cheap dynamic extension (lower latency / LLM cost), at the cost of gradual drift requiring periodic refresh.
- **Predefined Cypher over LLM-generated queries** — chosen for schema consistency and fewer hallucinations.
- **Reflexion-style reflection** in entity extraction — to raise recall and cut hallucinations.

---

## Key Takeaways

1. **Bi-temporal modeling is the core contribution.** Separating "when a fact was true" (T) from "when we learned it" (T′), with explicit `t_valid`/`t_invalid` edge invalidation, lets memory update non-destructively and supports genuine temporal reasoning — the area where Zep's largest accuracy gains appear.

2. **Episodic + semantic + community hierarchy** gives both lossless provenance (raw episodes you can cite) and high-level abstraction (community summaries), echoing human episodic/semantic memory.

3. **Edge invalidation, not deletion**, is how Zep handles a changing world — contradictory facts coexist with disjoint validity ranges instead of overwriting each other.

4. **Massive efficiency win.** ~115k → ~1.6k context tokens and ~90% latency reduction while *improving* accuracy shows structured graph retrieval beats brute-force full-context, and matters for production cost/latency.

5. **Honest benchmark critique.** The authors argue DMR is too easy (near-saturated by full-context) and call for harder, enterprise-style memory benchmarks — a useful framing for the whole subfield.

---

## Limitations (Acknowledged by Authors)

1. **single-session-assistant regression** — Zep underperforms full-context on questions answerable from one recent turn; retrieval can drop load-bearing exact wording.
2. **Weaker models underuse temporal data** — gpt-4o-mini gains less than gpt-4o on temporal/knowledge-update, suggesting Zep's temporal context needs more capable models (or further engineering) to be fully exploited.
3. **Partial evaluation** — experiments use only a subset of Graphiti's retrieval (no breadth-first seeding, community retrieval, or cross-encoder reranking evaluated), and the structured-business-data + traditional-RAG capabilities are unmeasured for lack of suitable benchmarks.
4. **No head-to-head vs MemGPT on LME** — MemGPT couldn't ingest LongMemEval histories, so the strongest comparison is against a full-context baseline, not the prior SOTA memory system.
5. **Benchmark scarcity** — the authors note no existing benchmark assesses Zep's ability to synthesize conversation history *with* structured business data, its actual production use case.

---

## Where it sits (v1/v2)

Zep is a **v1 (foundational, 2023–2025)** agent-memory system — and a notably *production-grade* one: the Graphiti engine and Zep service are a deployed commercial framework, not just a research prototype, with the paper foregrounding latency, scalability, and cost.

- **vs MemGPT (v1):** MemGPT framed memory as OS-style paging over a flat archive; Zep replaces the flat store with a **temporal knowledge graph** and directly beats it on DMR (94.8% vs 93.4%).
- **vs A-Mem (v1):** A-Mem links memories as Zettelkasten-style notes retrieved by semantic similarity; Zep instead imposes explicit **entity/fact/community structure with bi-temporal edge validity**, enabling fact invalidation and temporal queries that note-similarity cannot.
- **vs MAGMA (v2):** MAGMA's 2026 multi-graph design (separate temporal / causal / semantic / entity layers with intent-aware traversal) can be read as a *successor* to Zep's single hierarchical graph — and MAGMA's own related work explicitly critiques Zep for organizing memory around associative proximity rather than fully disentangling distinct relational dimensions (notably, Zep has **no explicit causal graph**).

In short: Zep is the **temporal-knowledge-graph** point in the design space — its signature ideas are bi-temporality and non-destructive edge invalidation — sitting between MemGPT's flat archives and the later multi-graph, causally-aware frontier systems.
