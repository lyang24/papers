# HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models

**Authors:** Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su (The Ohio State University; Stanford University)

**Paper:** arXiv:2405.14831v3 (May 2024, rev. Jan 2025) — NeurIPS 2024

**GitHub:** https://github.com/OSU-NLP-Group/HippoRAG

---

## The Core Problem

Standard retrieval-augmented generation (RAG) encodes **each passage in isolation**. This makes RAG fundamentally unable to support tasks that require **integrating knowledge across passage boundaries** — scientific literature review, legal case briefing, medical diagnosis, and, as a tractable proxy, multi-hop QA.

The authors crystallize the failure with **path-finding multi-hop questions**, exemplified by: *"Which Stanford professor works on the neuroscience of Alzheimer's?"* Given a corpus describing thousands of Stanford professors and thousands of Alzheimer's researchers, no single passage mentions both characteristics together. To answer, the system must *associate disparate information* about Prof. Thomas Südhof across passages.

Two existing strategies both fall short:

1. **Single-step RAG** (BM25, Contriever, GTR, ColBERTv2) — encodes passages independently, so it can never join information that does not co-occur in one passage.
2. **Iterative / multi-step RAG** (e.g., IRCoT) — interleaves retrieval and LLM generation across multiple rounds. This can follow a *known* path, but (a) it is expensive and slow (many LLM calls over retrieved documents), and (b) it still struggles with **path-finding** questions where there are too many candidate paths to explore rather than one path to follow.

A separate class (RAPTOR, GraphRAG, MemWalker) integrates information offline by **summarizing** knowledge — but summaries must be **recomputed whenever new data is added**, so they are poorly suited to a continuously updating long-term memory.

---

## The Big Idea: Hippocampal Indexing for LLMs

HippoRAG is inspired by the **hippocampal memory indexing theory** (Teyler & Discenna, 1986). In this theory, human long-term memory is implemented by three components serving two functions:

- **Pattern separation** — ensuring distinct experiences get distinct, non-interfering representations (done during encoding).
- **Pattern completion** — retrieving a complete memory from a partial cue (done during retrieval).

The biological components map directly onto HippoRAG's design:

| Brain component | Role in memory | HippoRAG analogue |
|---|---|---|
| **Neocortex** | Processes perceptual input into high-level features; stores actual memory content | **LLM** doing OpenIE (extract triples) + the indexed passages themselves |
| **Parahippocampal regions (PHR)** | Route signals between neocortex and hippocampus; detect synonymy | **Retrieval encoders** (Contriever / ColBERTv2) adding synonymy edges and linking query entities to KG nodes |
| **Hippocampus** | Holds a sparse *index* of interconnected pointers + associations; does context-based recall via densely-connected CA3 neurons | **Open knowledge graph** + the **Personalized PageRank** algorithm run over it |

The crucial conceptual move: the hippocampal index is a **set of interconnected pointers with associations**, not a store of the content itself. HippoRAG's KG plays exactly this role — it is a schemaless web of associations that points back to passages, and **new knowledge is integrated by simply adding edges**, never by re-summarizing (unlike RAPTOR/GraphRAG).

---

## Architecture

HippoRAG has two phases that mirror memory encoding and memory retrieval.

### 1. Offline Indexing (analogous to memory encoding / pattern separation)

Build the artificial hippocampal index — an **open KG** — over the whole corpus, passage-by-passage:

1. **OpenIE via LLM.** A strong instruction-tuned LLM `L` (default GPT-3.5-turbo-1106) processes each passage via **1-shot open information extraction**, producing a schemaless set of noun-phrase **nodes** `N` and relation **edges** `E`. The prompt is two-step: first extract named entities, then add them to the prompt to extract the final triples (which also include general-concept noun phrases). This two-step design balances generality against named-entity bias.
   - Extracting salient signals as **discrete noun phrases** (rather than dense vectors) is what provides fine-grained **pattern separation**.
2. **Synonymy edges via PHR encoders.** A retrieval encoder `M` adds extra edges `E'` between *similar-but-not-identical* noun phrases when their cosine similarity exceeds a threshold `τ` (= 0.8). These augment the index for downstream **pattern completion**.
3. **Passage matrix `P`.** A `|N| × |P|` matrix recording how many times each KG node (noun phrase) appears in each original passage — the link from index back to memory content.

### 2. Online Retrieval (analogous to memory retrieval / pattern completion)

Given a query `q`:

1. **Named-entity extraction (neocortex).** The LLM `L` extracts **query named entities** `Cq = {c1, …, cn}` from `q` via a 1-shot prompt (e.g., *Stanford*, *Alzheimer's*).
2. **Entity linking (PHR).** Encode each `ci` with `M` and pick the KG node with highest cosine similarity → the **query nodes** `Rq`. These are the **partial cues** for pattern completion.
3. **Personalized PageRank (the synthetic hippocampus).** Run **PPR** over the KG using a personalized distribution `n` that places all probability mass on the query nodes (zero elsewhere). PPR distributes probability outward *only* through the neighborhood of the seed nodes — biasing the walk toward subgraphs jointly relevant to the query entities (e.g., flowing through to *Prof. Thomas*). This is the heart of the method: **PPR performs multi-hop reasoning in a single retrieval step**, mirroring how the densely-connected CA3 region completes a full memory from partial cues.
4. **Passage ranking.** Multiply the resulting node distribution `n'` by the passage matrix `P` to get a score `p` for every passage; rank and retrieve.

### Node Specificity (a neurobiologically-plausible IDF)

A global IDF signal would require, biologically, an aggregator neuron connected to *all* nodes — computationally prohibitive in a brain. Instead, HippoRAG defines **node specificity** `s_i = |P_i|^{-1}` (inverse of the number of passages node `i` was extracted from) — a purely *local* signal already available at each node. Each query node's seed probability is multiplied by `s_i` before PPR, so rarer (more specific) entities steer the walk more strongly. (In Figure 2, *Stanford* gets a larger weight than *Alzheimer's* because it appears in fewer documents.)

**Key hyperparameters:** synonymy threshold `τ = 0.8`, PPR damping factor `= 0.5` (restart probability), both tuned on 100 MuSiQue training examples; performance is reported as robust to them.

---

## Experimental Setup

- **Datasets (1,000 questions each from the dev set):** **MuSiQue** (answerable) and **2WikiMultiHopQA** — the two primary, genuinely multi-hop benchmarks — plus **HotpotQA**, included for completeness despite being a known-weaker test of multi-hop reasoning (many spurious shortcuts). Following IRCoT, all supporting + distractor passages are pooled into a single retrieval corpus per dataset (e.g., 11,656 passages / 91,729 nodes for MuSiQue).
- **Baselines:** BM25, Contriever, GTR, ColBERTv2 (single-step); Propositionizer and RAPTOR (LLM-augmented single-step); **IRCoT** (multi-step / iterative).
- **Metrics:** Recall@2 / Recall@5 for retrieval; Exact Match (EM) / F1 for QA.
- **Defaults:** LLM = GPT-3.5-turbo-1106 (temp 0); retriever = Contriever or ColBERTv2.

---

## Experimental Results

### Single-Step Retrieval (Recall@2 / Recall@5)

| Method | MuSiQue R@2 | R@5 | 2Wiki R@2 | R@5 | HotpotQA R@2 | R@5 | Avg R@2 | Avg R@5 |
|---|---|---|---|---|---|---|---|---|
| BM25 | 32.3 | 41.2 | 51.8 | 61.9 | 55.4 | 72.2 | 46.5 | 58.4 |
| Contriever | 34.8 | 46.6 | 46.6 | 57.5 | 57.2 | 75.5 | 46.2 | 59.9 |
| GTR | 37.4 | 49.1 | 60.2 | 67.9 | 59.4 | 73.3 | 52.3 | 63.4 |
| ColBERTv2 | 37.9 | 49.2 | 59.2 | 68.2 | 64.7 | 79.3 | 53.9 | 65.6 |
| RAPTOR | 35.7 | 45.3 | 46.3 | 53.8 | 58.1 | 71.2 | 46.7 | 56.8 |
| Proposition | 37.6 | 49.3 | 56.4 | 63.1 | 58.7 | 71.1 | 50.9 | 61.2 |
| **HippoRAG (Contriever)** | 41.0 | 52.1 | 71.5 | **89.5** | 59.0 | 76.2 | 57.2 | 72.6 |
| **HippoRAG (ColBERTv2)** | **40.9** | **51.9** | **70.7** | 89.1 | 60.5 | 77.7 | **57.4** | **72.9** |

The headline gain is on **2WikiMultiHopQA**: roughly **+11 R@2 and +20 R@5** over the best baseline (its entity-centric design suits HippoRAG perfectly). MuSiQue gains ~3 points. HotpotQA is competitive but not dominant — its weaker knowledge-integration demand plus a concept-context tradeoff.

### Multi-Step Retrieval — IRCoT + HippoRAG are complementary

| Method | MuSiQue R@2 | R@5 | 2Wiki R@2 | R@5 | HotpotQA R@2 | R@5 |
|---|---|---|---|---|---|---|
| IRCoT + ColBERTv2 | 41.7 | 53.7 | 64.1 | 74.4 | 67.9 | 82.0 |
| **IRCoT + HippoRAG (ColBERTv2)** | **45.3** | **57.6** | **75.8** | **93.9** | **67.0** | **83.0** |

Using HippoRAG as IRCoT's retriever adds ~4% (MuSiQue), ~18% (2Wiki), ~1% (HotpotQA) on R@5.

### QA Performance (EM / F1, ColBERTv2 reader)

| Retriever | MuSiQue EM | F1 | 2Wiki EM | F1 | HotpotQA EM | F1 | Avg EM | Avg F1 |
|---|---|---|---|---|---|---|---|---|
| None | 12.5 | 24.1 | 31.0 | 39.6 | 30.4 | 42.8 | 24.6 | 35.5 |
| ColBERTv2 | 15.5 | 26.4 | 33.4 | 43.3 | 43.4 | 57.7 | 30.8 | 42.5 |
| **HippoRAG (ColBERTv2)** | **19.2** | **29.8** | **46.6** | **59.5** | 41.8 | 55.0 | **35.9** | **48.1** |
| IRCoT (ColBERTv2) | 19.1 | 30.5 | 35.4 | 45.1 | 45.5 | 58.4 | 33.3 | 44.7 |
| **IRCoT + HippoRAG** | **21.9** | **33.3** | **47.7** | **62.7** | **45.7** | **59.2** | **38.4** | **51.7** |

QA gains track retrieval gains: single-step HippoRAG lifts F1 by ~3 / 17 / -3 points on MuSiQue / 2Wiki / HotpotQA, and the strongest config (IRCoT + HippoRAG) reaches **62.7 F1 on 2Wiki** vs 45.1 for IRCoT alone.

### Efficiency — single-step beats iterative at a fraction of the cost

Online retrieval over 1,000 queries (GPT-3.5 Turbo), Appendix G:

| | ColBERTv2 | IRCoT | HippoRAG |
|---|---|---|---|
| API Cost ($) | 0 | 1–3 | 0.1 |
| Time (minutes) | 1 | 20–40 | 3 |

HippoRAG matches or beats IRCoT's QA performance while being **10–30× cheaper** and **6–13× faster** online — because it only extracts named entities from the *query* once, rather than running the LLM over all retrieved documents across multiple iterative rounds.

### All-Recall (all supporting passages retrieved — true multi-hop success)

| | MuSiQue AR@2 | AR@5 | 2Wiki AR@2 | AR@5 | HotpotQA AR@2 | AR@5 |
|---|---|---|---|---|---|---|
| ColBERTv2 | 6.8 | 16.1 | 25.1 | 37.1 | 33.3 | 59.0 |
| **HippoRAG** | **10.2** | **22.4** | **45.4** | **75.7** | 33.8 | 57.9 |

On the stricter all-recall metric, the 2Wiki gap *widens* (AR@5 75.7 vs 37.1, ~+38 points) — confirming the gains come from retrieving **all** supporting documents (genuine single-step multi-hop reasoning), not partial wins on more queries.

---

## Ablation / "What Makes HippoRAG Work?" (Avg R@2 / R@5)

| Variant | Avg R@2 | Avg R@5 |
|---|---|---|
| **HippoRAG (full)** | **57.4** | **72.9** |
| OpenIE → REBEL (closed end-to-end model) | 46.2 | 58.4 |
| OpenIE → Llama-3.1-8B-Instruct | 54.4 | 67.8 |
| OpenIE → Llama-3.1-70B-Instruct | 57.1 | 72.5 |
| PPR → `Rq` nodes only (no graph walk) | 50.7 | 56.2 |
| PPR → `Rq` nodes + direct neighbors | 42.2 | 59.2 |
| w/o Node Specificity | 54.7 | 70.9 |
| w/o Synonymy Edges | 56.2 | 70.5 |

Findings:

1. **PPR is essential.** Replacing PPR with just the seed nodes, or seeds + direct neighbors, drops average R@2 by 7–15 points. Strikingly, *naively* adding the 1-hop neighborhood **without** PPR is *worse* than using query nodes alone — the graph walk's probability-weighting, not mere graph expansion, is what works. This is the core ablation justifying the hippocampal-CA3 analogy.
2. **LLM-based OpenIE matters.** The end-to-end REBEL model collapses performance (GPT-3.5 extracts ~2× as many triples). Open-weight **Llama-3.1-70B is competitive with / better than GPT-3.5** on two of three datasets — a cheaper indexing path for large corpora.
3. **Node specificity** helps most on MuSiQue/HotpotQA (term-weighting matters), negligible on 2Wiki (entity-dominated). **Synonymy edges** help most on 2Wiki (entity standardization), confirming the two enhancements address complementary regimes.

---

## Case Studies (path-following vs path-finding)

- **Path-following — "In which district was Alhandra born?"** Even though *Alhandra* and *Vila de Xira* never co-occur with the district in one passage, HippoRAG follows the KG edge (Alhandra → born-in → Vila de Xira → Lisbon District) in a single step. IRCoT can also solve this — but at 10–30× the cost.
- **Path-finding — "Which Stanford professor works on the neuroscience of Alzheimer's?"** Both ColBERTv2 and IRCoT **fail** (they retrieve unrelated Stanford neuroscientists), because there are too many paths to follow iteratively. HippoRAG's PPR over its web of associations surfaces **Thomas Südhof** correctly — the scenario that motivated the whole paper.

---

## Key Takeaways

1. **A schemaless KG built by LLM OpenIE is a strong "hippocampal index."** Storing salient signals as discrete noun phrases (pattern separation) and linking them with relation + synonymy edges gives a continuously-updatable associative memory where new knowledge = new edges (no re-summarization).
2. **Personalized PageRank is the engine of single-step multi-hop retrieval.** Seeding PPR with query entities and reading off passage scores lets the system "complete the pattern" — joining information across passage boundaries in one retrieval step, where iterative RAG needs many.
3. **Single-step can beat iterative at an order-of-magnitude lower cost.** HippoRAG matches/exceeds IRCoT's QA while being 10–30× cheaper and 6–13× faster online — the practically important axis for serving users.
4. **The neuroscience mapping is load-bearing, not decorative.** Each component (LLM=neocortex, encoders=PHR, KG+PPR=hippocampus, node specificity=local IDF) corresponds to a real function, and the PPR ablation shows the associative graph walk is what delivers the gains.
5. **Path-finding multi-hop QA** is a genuinely new capability — questions trivial for an informed human but out of reach for prior retrievers — that HippoRAG can begin to tackle.

---

## Limitations (Acknowledged by Authors)

1. **All components are off-the-shelf, untrained.** Error analysis (Appendix F) attributes most errors to NER and OpenIE — direct fine-tuning of these modules is unexplored headroom.
2. **PPR is relation-agnostic.** Graph-search errors suggest improvements beyond plain PPR, e.g., letting *relations* guide traversal directly rather than walking an undirected association graph.
3. **OpenIE consistency degrades on longer documents** — extraction quality is uneven across passage lengths.
4. **Scalability unproven.** Results are on ~1,000-question / ~6K–12K-passage corpora. The efficiency and efficacy of the synthetic hippocampal index *as it grows far beyond these benchmarks* remain to be validated empirically.

---

## Where it sits (v1/v2)

HippoRAG is the **NEUROBIOLOGICAL KG-MEMORY lineage SOURCE.** It is the paper that established the template now central to graph-based agentic memory: *build an open knowledge graph from a corpus offline (LLM OpenIE), then retrieve by seeding a graph-propagation algorithm with query entities and reading scores back out over passages.*

- It is **foundational and v1-era (2024)** — predating the agentic-memory wave, framed as RAG for static-corpus multi-hop QA rather than evolving conversational memory, and using plain undirected PPR with off-the-shelf components.
- But it is **pivotal for v2 graph memory.** Its **Personalized-PageRank-over-KG retrieval** is the direct ancestor of:
  - **MemGAS's** Personalized PageRank routing over its memory graph,
  - the **graph-routing / entity-seeded traversal** in systems like **MAGMA** (multi-graph anchor identification + adaptive traversal) and **Zep/Graphiti** (temporal KG with graph-based retrieval).

Where HippoRAG used a single undifferentiated association graph and one PPR pass, the v2 lineage **disentangles** the graph into typed relational layers (temporal / causal / semantic / entity) and makes traversal **intent-aware** — but the underlying idea, *entity-seeded propagation over an LLM-built KG as a synthetic hippocampus*, originates here.
