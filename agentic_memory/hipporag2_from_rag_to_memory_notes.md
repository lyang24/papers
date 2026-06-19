# From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)

**Authors:** Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su (The Ohio State University; University of Illinois Urbana-Champaign)

**Paper:** arXiv:2502.14802v2 (ICML 2025, PMLR 267)

**GitHub:** https://github.com/OSU-NLP-Group/HippoRAG

---

## The Core Problem

Continual learning — continuously absorbing, integrating, and leveraging new knowledge — is a hallmark of human intelligence that LLMs struggle to approximate. The two parametric routes both fail: **continual fine-tuning** suffers catastrophic forgetting and is too expensive for frequent updates, and **model editing** produces highly localized changes that don't ripple to related facts. **RAG** sidesteps both by injecting knowledge non-parametrically at inference time, and has become the de facto production solution.

But standard vector RAG only emulates one slice of human long-term memory. It cannot capture two vital interconnected properties:

1. **Sense-making** (Klein et al., 2006) — interpreting larger, more complex, or uncertain contexts (e.g. a full novel), which requires integrating disparate passages.
2. **Associativity** (Suzuki, 2005) — drawing multi-hop connections between disparate facts, which independent vector retrieval cannot emulate.

Structure-augmented RAG methods tried to fix these gaps but introduced a damaging **regression**: each gains on the dimension it targets while dropping below standard RAG on the others. Concretely (Figure 1):
- **HippoRAG** (the predecessor) added associativity via a KG + Personalized PageRank, but its entity-centric NER-based design loses query context and **collapses on large-scale discourse / sense-making**.
- **RAPTOR / GraphRAG / LightRAG** add summarization or KG structure for sense-making, but the LLM-generated summary noise **deteriorates simple and multi-hop QA**.

The crux: no prior structure-augmented method beats the strongest *embedding-based* RAG across **all three** of factual, sense-making, and associative memory simultaneously. HippoRAG 2 sets out to be robust across the board — "from RAG to MEMORY."

---

## The Big Idea: Deeper Passage Integration + Online LLM Use

HippoRAG 2 keeps HippoRAG's neurobiological framing (LLM = neocortex, KG + PPR = hippocampal auto-associative index, retrieval encoder = parahippocampal linker) and its two-stage offline-indexing / online-retrieval skeleton, but adds three refinements that close the regression:

1. **Dense-Sparse Integration (§3.2)** — passages are added back into the KG as first-class **passage nodes**, so the graph carries both *concept* (sparse phrase nodes) and *context* (dense passage nodes).
2. **Deeper Contextualization (§3.3)** — query linking moves from entity-level **NER-to-node** to **query-to-triple**, aligning the full query semantics with contextual KG triples instead of isolated entities.
3. **Recognition Memory (§3.4)** — an online LLM filter prunes irrelevant retrieved triples before they become PPR seed nodes, mirroring recall-vs-recognition in human memory.

The result: deeper passage integration into the PPR graph + more effective *online* (inference-time) use of the LLM, pushing RAG closer to human long-term memory.

---

## Architecture

Two stages, both run with Llama-3.3-70B-Instruct (extraction + filtering) and NV-Embed-v2 (retriever) by default.

### Offline Indexing (build the open KG)

1. **OpenIE by LLM** — extract schema-less open KG triples (subject phrase → relation edge → object phrase) from every passage. Same as HippoRAG.
2. **Synonym detection by embedding** — add **synonym edges** between phrase nodes whose embedding similarity exceeds a threshold, linking synonyms across passages (this is what lets old and new knowledge interconnect during continual learning).
3. **Dense-sparse integration** — add each passage as a **passage node**, connected to all phrases it produced via a **context edge** labeled "contains."

The final open KG therefore has three edge types — **relation edges** (triples), **synonym edges** (phrase↔phrase), and **context edges** (passage→phrase) — and two node types: phrase nodes (sparse/conceptual) and passage nodes (dense/contextual). This is the key structural upgrade over HippoRAG, whose graph was phrase-only and merely *ensembled* passage scores after the fact.

### Online Retrieval & QA

1. **Retrieve passages and triples** — the embedding model scores both passages and triples against the query. HippoRAG 2 uses **query-to-triple** linking by default (vs. HippoRAG's NER-to-node), matching the whole query to triples to capture contextual relationships.
2. **Recognition memory (triple filtering)** — retrieve top-k triples (k=5 by default), then an LLM filters them down to a relevant subset T' ⊆ T. Prompt tuned with DSPy MIPROv2.
3. **Assign seed node weights** — phrase nodes from the *filtered* triples become seed nodes, weighted by their average ranking scores; **all passage nodes** are also seeded (broad activation aids multi-hop), weighted by embedding similarity scaled by a **reset-probability weight factor** (0.05 by default) that balances phrase vs. passage influence. If no triples survive filtering, it falls back to plain top-ranked passage retrieval.
4. **PPR graph search** — Personalized PageRank propagates probability mass from seed nodes across the KG; passages are ranked by resulting PageRank score.
5. **QA reading** — top-5 passages are fed to a reader LLM (GPT-4o-mini or Llama-3.3-70B-Instruct) for the final answer.

### The "recall vs. recognition" framing

Recognition memory (step 2) is the cognitive-science hook: **recall** retrieves without external cues, while **recognition** identifies relevant information *given* an external stimulus. Modeling query-to-triple as retrieve-then-LLM-filter mirrors recognition and removes noisy seed nodes that would otherwise mislead PPR.

---

## Experimental Results

Evaluation spans the three memory dimensions: **Simple QA** (factual — NQ, PopQA), **Multi-Hop QA** (associative — MuSiQue, 2Wiki, HotpotQA, LV-Eval), **Discourse Understanding** (sense-making — NarrativeQA). 1,000 queries each for the QA datasets (124 for LV-Eval, 293 for NarrativeQA). Metric: token-based F1 for QA, passage recall@5 for retrieval.

### Table 2 — QA Performance (F1, Llama-3.3-70B-Instruct reader)

All structure-augmented baselines use the same Llama-3.3-70B extractor + NV-Embed-v2 retriever for fairness. † = HippoRAG 2 significantly beats the best NV-Embed-v2 baseline (p<0.05).

(The paper's Table 2 includes an Avg column; the per-dataset values below are as reported, and HippoRAG 2 has the highest average. I omit per-row averages I cannot verify cleanly from the source extraction.)

| Method | NQ | PopQA | MuSiQue | 2Wiki | HotpotQA | LV-Eval | NarrativeQA |
|---|---|---|---|---|---|---|---|
| *None (no retrieval)* | 54.9 | 32.5 | 26.1 | 42.8 | 47.3 | 6.0 | 12.9 |
| Contriever | 58.9 | 53.1 | 31.3 | 41.9 | 62.3 | 8.1 | 19.7 |
| BM25 | 59.0 | 49.9 | 28.8 | 51.2 | 63.4 | 5.9 | 18.3 |
| GTR (T5-base) | 59.9 | 56.2 | 34.6 | 52.8 | 62.8 | 7.1 | 19.9 |
| GTE-Qwen2-7B | 62.0 | 56.3 | 40.9 | 60.0 | 71.0 | 7.1 | 21.3 |
| GritLM-7B | 61.3 | 55.8 | 44.8 | 60.6 | 73.3 | 9.8 | 23.9 |
| **NV-Embed-v2 (7B)** | 61.9 | 55.7 | 45.7 | 61.5 | 75.3 | 9.8 | 25.7 |
| RAPTOR | 50.7 | 56.2 | 28.9 | 52.1 | 69.5 | 5.0 | 21.4 |
| GraphRAG | 46.9 | 48.1 | 38.5 | 58.6 | 68.6 | 11.2 | 23.0 |
| LightRAG | 16.6 | 2.4 | 1.6 | 11.6 | 2.4 | 1.0 | 3.7 |
| HippoRAG | 55.3 | 55.9 | 35.1 | 71.8 | 63.5 | 8.4 | 16.3 |
| **HippoRAG 2** | **63.3†** | **56.2** | **48.6†** | **75.5** | **75.3** | **12.9†** | **25.9** |

Highlights:
- HippoRAG 2 has the **highest average F1** and the best/second-best slot on every column.
- Beats strongest dense retriever NV-Embed-v2 by **+9.5 F1 on 2Wiki** and **+3.1 on LV-Eval** (paper-stated deltas), plus a gain on MuSiQue (48.6 vs 45.7), while *not regressing* on factual (NQ/PopQA) or sense-making (NarrativeQA).
- Beats predecessor **HippoRAG** dramatically where HippoRAG had collapsed: NarrativeQA **25.9 vs 16.3**, MuSiQue **48.6 vs 35.1** — confirming the deeper passage integration fixes HippoRAG's sense-making/context loss.
- **LightRAG craters** on this QA-reader setup (avg ~5.6), the clearest example of summarization-noise hurting QA.

### Table 3 — Retrieval Performance (passage recall@5)

GraphRAG/LightRAG omitted (they don't directly produce passage retrievals).

| Retriever | NQ | PopQA | MuSiQue | 2Wiki | HotpotQA | **Avg** |
|---|---|---|---|---|---|---|
| BM25 | 56.1 | 35.7 | 43.5 | 65.3 | 74.8 | 55.1 |
| Contriever | 54.6 | 43.2 | 46.6 | 57.5 | 75.3 | 55.4 |
| GTR | 63.4 | 49.4 | 49.1 | 67.9 | 73.9 | 60.7 |
| GTE-Qwen2-7B | 74.3 | 50.6 | 63.6 | 74.8 | 89.1 | 70.5 |
| GritLM-7B | 76.6 | 50.1 | 65.9 | 75.3 | 92.4 | 72.2 |
| **NV-Embed-v2 (7B)** | 75.4 | 51.0 | 69.7 | 73.9 | 94.5 | 73.4 |
| RAPTOR | 68.3 | 48.7 | 57.8 | 66.2 | 86.9 | 65.6 |
| HippoRAG (reproduced) | 44.4 | 53.8 | 53.2 | 90.4 | 77.3 | 63.8 |
| **HippoRAG 2** | **78.0** | **51.7** | **74.7** | **90.4** | **96.3** | **78.2** |

Highlights:
- Highest recall@5 on nearly every dataset; **+5.0 (MuSiQue)** and **+13.9 (2Wiki)** over NV-Embed-v2 — the associativity payoff.
- HippoRAG wins PopQA (entity-centric) but lags everywhere else; reproducing it with the stronger LLM+retriever only nudged it **+1.3 F1** over its original paper, underscoring that the gains here come from architecture, not the backbone swap.

### Robustness experiments

- **Different dense retrievers (Table 7, MuSiQue recall@5):** HippoRAG 2 beats raw dense retrieval regardless of backbone — GTE-Qwen2 63.6→**68.8**, GritLM-7B 66.0→**71.6**, NV-Embed-v2 69.7→**74.7**.
- **Corpus expansion / continual learning (Figure 3):** NQ and MuSiQue each split into 4 segments; HippoRAG 2's margin over NV-Embed-v2 stays **remarkably consistent** as segments are incrementally added. Both methods hold up on simple QA as the corpus grows, but both degrade similarly on the harder associative task — a noted open challenge for continual-learning benchmarks.
- Works with both open-source (Llama-3.3-70B) and proprietary (GPT-4o-mini) readers; GPT-4o-mini follows the same trend.

---

## Ablation Study (Table 4 — multi-hop recall@5)

| Variant | MuSiQue | 2Wiki | HotpotQA | **Avg** |
|---|---|---|---|---|
| **HippoRAG 2 (full)** | **74.7** | **90.4** | **96.3** | **87.1** |
| w/ NER-to-node (HippoRAG's linking) | 53.8 | 91.2 | 78.8 | 74.6 |
| w/ Query-to-node | 44.9 | 65.5 | 68.3 | 59.6 |
| w/o Passage Node (no dense-sparse integ.) | 63.7 | 90.3 | 88.9 | 81.0 |
| w/o Filter (no recognition memory) | 73.0 | 90.7 | 95.4 | 86.4 |

Findings:
1. **Query-to-triple linking is the single biggest lever** — replacing it with HippoRAG's NER-to-node drops avg recall **87.1 → 74.6** (query-to-triple gives **+12.5%** over NER-to-node on average). Query-to-node is even worse (59.6), because queries and KG nodes live at different granularities while NER results and KG phrase nodes are both phrase-level.
2. **Passage nodes (dense-sparse integration) matter** — removing them drops **87.1 → 81.0**, biggest hit on MuSiQue (74.7 → 63.7).
3. **Recognition memory (triple filter)** gives a smaller but consistent lift (86.4 → 87.1).
4. **Reset-probability weight factor (Table 5):** the phrase-vs-passage balance is crucial; 0.05 chosen as the default sweet spot across NQ and MuSiQue dev sets.

### Qualitative example (Table 6)

For multi-hop "What county is Erik Hort's birthplace a part of?", NV-Embed-v2 finds only the "Erik Hort" passage (insufficient for 2-hop). HippoRAG 2's query-to-triple step surfaces the triple `(Erik Hort, born in, Montebello)`, then PPR ranks the "Montebello, New York" passage to the top — bridging the second hop that pure dense retrieval misses.

---

## Key Takeaways

1. **Putting passages back into the graph fixes the regression.** HippoRAG's phrase-only, entity-centric KG was the root cause of its sense-making collapse; adding passage nodes + context edges (dense-sparse integration) is what lets HippoRAG 2 be robust on *all three* memory types instead of trading one for another.

2. **Query-to-triple linking is the dominant contributor.** Matching the whole query against contextual triples — rather than extracting entities (NER) — is the largest single ablation swing (+12.5% recall). Context, not just concepts, drives good seed-node selection.

3. **Online LLM use (recognition memory) is cheap insurance.** Filtering retrieved triples before PPR removes noisy seeds and is the inference-time analog of human recognition; small but consistent gains.

4. **Comprehensiveness beats specialization.** Every prior structure-augmented method (HippoRAG, RAPTOR, GraphRAG, LightRAG) wins its home task but loses elsewhere; HippoRAG 2's headline claim is being the first to beat the strongest embedding RAG (NV-Embed-v2) on factual, sense-making, and associative simultaneously — ~+7 points avg on associativity with no factual/sense-making regression.

5. **KG-for-retrieval, not KG-as-corpus.** Unlike GraphRAG/LightRAG, HippoRAG 2 uses the KG to *guide retrieval* over the original passages rather than to *replace* the corpus with LLM summaries, which is why it avoids the summarization noise that hurts those methods on QA.

---

## Limitations

1. **Continual-learning degradation on hard tasks remains unsolved** — Figure 3 shows associative-task performance still decays as the corpus grows (for both HippoRAG 2 and the dense baseline); the method preserves its *margin* but not absolute robustness.
2. **Heavy LLM dependence in indexing and retrieval** — OpenIE extraction, synonym detection, and triple filtering all rely on a strong (70B) LLM; quality and compute cost scale with it. Token/time/memory analysis is relegated to appendices.
3. **Episodic / conversational memory is explicitly future work** — the authors note long-conversation episodic memory (the regime MAGMA/Zep target) is not yet addressed; evaluation is QA/discourse over document corpora, not multi-session dialogue.
4. **Benchmark scope** — sense-making is tested only via NarrativeQA (293 queries, 10 documents), a relatively narrow probe of the discourse-understanding dimension.

---

## Where it sits (v1/v2)

**HippoRAG 2 is the v2-bridging upgrade that takes "RAG" toward "MEMORY."** The paper's whole thesis is a reframing: standard RAG is only one corner of human long-term memory, and the path to *non-parametric continual learning* is a system that simultaneously serves factual, sense-making, and associative memory. The title — *From RAG to Memory* — is the explicit statement of that move.

- **vs. HippoRAG** (its direct predecessor, also in this collection as `hipporag_neurobiological_memory`): HippoRAG v1 introduced the neurobiological framing (LLM neocortex + KG/PPR hippocampus) and won on associative multi-hop QA, but its **phrase-only, NER-driven, entity-centric** design lost passage context and collapsed on discourse (NarrativeQA 16.3) and underperformed dense retrievers on simple QA. HippoRAG 2 keeps the PPR core but **(a)** re-injects passages as nodes (dense-sparse integration), **(b)** links via query-to-triple instead of NER, and **(c)** adds online LLM triple-filtering (recognition memory). Net effect: NarrativeQA 16.3 → 25.9, MuSiQue F1 35.1 → 48.6, retrieval avg 63.8 → 78.2 — turning a specialist into a generalist.

- **vs. Zep / MAGMA KG-memory:** all three replace flat vector stores with graph-structured memory, but they target different regimes. **Zep and MAGMA are agentic conversational memory** — they index multi-session dialogue, ground temporal expressions, and (MAGMA) disentangle temporal/causal/semantic/entity relations into separate graph layers with intent-aware traversal, optimized for *episodic recall over an agent's interaction history*. **HippoRAG 2 is document-corpus memory** for continual knowledge ingestion: a single open KG (relation + synonym + context edges) over a passage corpus, retrieved by Personalized PageRank rather than intent-routed beam search. HippoRAG 2 has no notion of timestamps, causal edges, or sessions — exactly the episodic/temporal axis that MAGMA and Zep specialize in, and which HippoRAG 2 lists as future work. They are complementary: HippoRAG 2 deepens *what you know*; MAGMA/Zep organize *what happened*.
