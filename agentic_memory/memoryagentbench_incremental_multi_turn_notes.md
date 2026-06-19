# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench)

**Authors:** Yuanzhe Hu*, Yu Wang*, Julian McAuley (UC San Diego)

**Paper:** ICLR 2026 | arXiv:2507.05257 | Benchmark nickname: **MemoryAgentBench**

---

## The Problem

Agent benchmarks (GAIA, SWE-Bench, OpenHands, etc.) overwhelmingly measure **reasoning** — planning, tool use, code synthesis — and largely ignore **memory**: how agents memorize, update, and retrieve long-term information. The paper calls agents equipped with memory mechanisms **memory agents** (memory can live in parameters, vectors, textual histories, or external databases; the paper focuses on textual histories + external databases since those dominate real deployments).

Existing evaluations don't fit memory agents:

- **Long-context benchmarks** (LongBench ~20k, LooGLE ~24k, NovelQA ~200k, NOCHA ~127k, Loong ~100k, ∞-Bench ~150k) feed the **entire context in a single block** and test static reading comprehension. They don't reflect the **incremental, multi-turn** nature of memory.
- **Conversational-memory benchmarks** (LOCOMO ~9k, LongMemEval) are either too short to challenge modern models, or use synthetic conversations with **limited topical diversity** and less realistic interaction patterns.
- **No prior benchmark covers all four memory competencies.**

The key conceptual claim: **memory ≠ long context.** Memory is a *compressed, distilled* representation of the past — it selectively extracts salient details, discards irrelevant info, and adds new inferences over time. A memory agent processes input **incrementally** (piece by piece, abstracting and consolidating), so a dataset that dumps the whole context at once cannot properly evaluate it.

---

## The Four Core Competencies

Grounded in classic memory / cognitive science (James 1890; McClelland et al. 1995; Anderson & Neely 1996; Wimber et al. 2015), the paper defines four complementary competencies a memory agent should have:

1. **Accurate Retrieval (AR)** — extract the correct snippet in response to a query (single-hop or multi-hop, as long as the answer is reachable with a single query).
2. **Test-Time Learning (TTL)** — incorporate new behaviors or acquire new skills *during deployment*, without additional training (e.g., learning a classification scheme from in-context labeled examples).
3. **Long-Range Understanding (LRU)** — integrate information distributed across extended contexts (≥100k tokens) and answer questions needing a *global* understanding of the whole sequence.
4. **Selective Forgetting (SF)** — revise, overwrite, or remove previously stored info when faced with contradictory evidence (akin to model editing / knowledge unlearning).

(Note: the task framing's fourth pillar is sometimes phrased as "conflict resolution" — in this paper it is operationalized as **Selective Forgetting**: prioritizing the newest fact when facts conflict.)

---

## The Incremental Multi-Turn Evaluation Protocol

This is the defining design choice. Memory must be **built turn-by-turn**, not handed over all at once.

**Dataset format.** Every dataset is standardized into chunks `c1, c2, …, cn`, questions `q1, …, qm`, answers `a1, …, am`. The chunks `c1…cn` constitute a single "conversation."

**Interaction protocol.** Unlike standard long-context eval (raw text in one block), each chunk `ci` is wrapped inside a simulated **User–Assistant dialogue** and prefixed with an explicit **memorization instruction** (e.g., *"Please memorize it and I will ask some questions…"*) to trigger the agent's memory mechanism.

**Incremental ingestion.** All agents must take chunks **one by one**, absorb each into memory, and **incrementally update** memory. Only **after all chunks are seen** are questions asked. This simulates incremental information processing — exactly what static long-context benchmarks skip.

**Multiple questions per context.** To amortize the cost of rebuilding memory over a huge input, datasets like EventQA, FactConsolidation, and LongMemEval (S*) pair **one long context with many questions** (e.g., LME(S*) = 5 contexts × 300 questions). Injecting 1M tokens for a single question is wasteful.

**SF guardrails.** For Selective Forgetting, the prompt explicitly tells agents that facts are indexed by serial numbers and **"newer facts have larger serial numbers,"** mandating that conflicts be resolved by choosing the newest fact.

---

## Datasets (per competency)

Datasets the authors **built themselves** are marked (ours). Avg lengths use GPT-4o-mini tokenizer.

| Competency | Dataset | Metric | Avg Len | Notes |
|---|---|---|---|---|
| **AR** | SH-Doc QA | Accuracy | 197K | Single-hop gold-passage retrieval QA (NIAH-style) |
| | MH-Doc QA | Accuracy | 421K | Multi-hop gold-passage retrieval QA |
| | LongMemEval (S*) | Accuracy | 355K | Reformulated chat history → 5 dialogues, 300 Qs |
| | **EventQA** (ours) | Accuracy | 534K | Read a novel, pick the correct next event from candidates after ≤5 prior events; fully automated pipeline |
| **TTL** | BANKING77 / CLINC150 / NLU / TREC-Coarse / TREC-Fine | Accuracy | 103K | Multi-class intent/question classification (77/151/68/6/50 labels) from in-context labeled examples |
| | Movie Recommendation | Recall@5 | 1.44M | Recommend 20 movies from thousands of dialogue turns |
| **LRU** | ∞Bench-Sum | F1 | 172K | Novel summarization (1000–1200 words) with entity replacement |
| | Detective QA | Accuracy | 124K | Long-range reasoning over detective novels (10 novels, 71 hard Qs) |
| **SF** | **FactConsolidation-SH** (ours) | Accuracy | up to 262K | Single-hop fact judgment over counterfactual edit pairs (built from MQUAKE) |
| | **FactConsolidation-MH** (ours) | Accuracy | up to 262K | Multi-hop fact judgment; contexts at 6K/32K/64K/262K |

Two **new datasets**: **EventQA** (stresses AR via temporal-sequence recall) and **FactConsolidation** (stresses SF via ordered true→contradictory fact pairs, newer fact appears later to simulate updates).

---

## Systems Evaluated (three families)

1. **Long-Context Agents** — maintain a context buffer of most-recent tokens; once the window (128K–1M) overflows, evict earliest chunks **FIFO**. Relies purely on positional recency. Backbones tested: GPT-4o, GPT-4o-mini, GPT-4.1-mini, Gemini-2.0-Flash, Claude-3.7-Sonnet.
2. **RAG Agents** — store past info in an external pool and retrieve as needed. Three sub-types:
   - *Simple RAG*: raw text + string matching — **BM25**
   - *Embedding RAG*: dense vectors + cosine sim — **Contriever, Text-Embedding-3-Small/Large, Qwen3-Embedding-4B**
   - *Structure-Augmented RAG*: build a graph/tree/timeline first — **RAPTOR, GraphRAG, MemoRAG, HippoRAG-v2, Mem0, Cognee, Zep**
3. **Agentic Memory Agents** — iterative agentic loops (reformulate query, look up memory, reflect, update working memory) — **Self-RAG, MemGPT, MIRIX**.

Unless noted, all RAG and commercial agents use **GPT-4o-mini** as backbone. Default chunk size 4096 (512 for SH/MH-Doc QA, LME(S*), and all SF tasks); retrieval top-k = 10.

---

## Key Results (Table 3 — Overall, GPT-4o-mini backbone unless noted)

Columns: AR Avg / TTL Avg / LRU Avg / SF Avg / Overall.

### Long-Context Agents
| Backbone | AR | TTL | LRU | SF | Overall |
|---|---|---|---|---|---|
| GPT-4o | 58.1 | 50.0 | 54.9 | 32.5 | 48.8 |
| GPT-4o-mini | 49.2 | 48.6 | 46.2 | 25.0 | 42.2 |
| GPT-4.1-mini | **71.8** | 46.2 | **49.1** | 20.5 | 46.9 |
| Gemini-2.0-Flash | 65.1 | 46.4 | 41.6 | 16.5 | 42.4 |
| Claude-3.7-Sonnet | 59.7 | **53.9** | **62.2** | 22.5 | **49.6** |

### RAG Agents (GPT-4o-mini backbone)
| Agent | AR | TTL | LRU | SF | Overall |
|---|---|---|---|---|---|
| GPT-4o-mini (ref) | 49.2 | 48.6 | 46.2 | 25.0 | 42.3 |
| BM25 (Simple) | **60.5** | 44.5 | 35.6 | 25.5 | **41.5** |
| Contriever | 33.9 | 42.9 | 29.8 | 12.5 | 29.8 |
| Text-Embed-3-Small | 53.8 | 42.7 | 36.3 | 15.5 | 37.1 |
| Text-Embed-3-Large | 54.6 | 44.3 | 37.3 | 16.0 | 38.0 |
| Qwen3-Embedding-4B | 54.7 | 45.1 | 37.0 | 16.0 | 38.2 |
| RAPTOR | 36.8 | 35.9 | 29.8 | 7.5 | 27.0 |
| GraphRAG | 40.9 | 24.8 | 19.9 | 8.0 | 23.4 |
| MemoRAG | 34.5 | 45.1 | 30.0 | 14.0 | 30.9 |
| HippoRAG-v2 | **65.1** | 35.8 | 36.2 | 29.5 | **41.6** |
| Mem0 | 32.6 | 21.2 | 20.7 | 10.0 | 21.1 |
| Cognee | 28.3 | 22.8 | 16.0 | 15.5 | 20.6 |
| Zep | 37.5 | 37.5 | 16.2 | 5.0 | 24.0 |

### Agentic Memory Agents
| Agent | AR | TTL | LRU | SF | Overall |
|---|---|---|---|---|---|
| Self-RAG | 33.6 | 12.8 | 14.0 | 11.0 | 18.7 |
| MemGPT | 34.3 | 40.8 | 14.0 | 15.5 | 28.3 |
| MIRIX (4o-mini) | 47.5 | 38.4 | 21.0 | 8.0 | 26.2 |
| MIRIX (4.1-mini) | **62.0** | 40.5 | **54.0** | 11.5 | **37.7** |

### Headline findings
1. **RAG wins on Accurate Retrieval.** Most RAG agents beat the GPT-4o-mini backbone on AR (BM25 60.5, HippoRAG-v2 65.1 vs 49.2) — RAG excels at pulling a small crucial snippet.
2. **Long-context wins on TTL and LRU.** RAG and commercial agents only retrieve partial context, so they can't form a holistic understanding or learn across the whole input. Claude-3.7 LC hits LRU 62.2; RAG tops out ~37.
3. **Everyone fails Selective Forgetting.** **All methods fail multi-hop SF — at most 7% accuracy (FC-MH).** Only long-context agents do reasonably on single-hop. Forgetting out-of-date memory is the hardest competency across the board.
4. **No single method masters all four competencies** — the central empirical conclusion.

### Backbone & reasoning-model ablations
- **RAG saturates on backbone quality:** once the backbone is strong enough, upgrading GPT-4o-mini → GPT-4.1-mini gives only marginal RAG gains. But **Agentic Memory benefits a lot** — MIRIX jumps +9.7 avg (AR +23.2, LRU +9.0) going to GPT-4.1-mini. Stronger backbones may unlock agentic memory.
- **Reasoning models validate FactConsolidation but don't solve it** (Table 5). o4-mini: FC-SH 100.0 @6K → 61.0 @32K; FC-MH 80.0 @6K → **14.0 @32K**. The task is solvable short-context, but **long-range reasoning collapses as context grows** — confirming the dataset is fair and that SF remains genuinely hard at scale.
- **Chunk-size / top-k:** smaller chunks + more retrieval calls help AR (finer-grained relevance, esp. embeddings); but varying chunk size *hurts* LRU (RAG is structurally ill-suited to integrating a large coherent context). More retrieved chunks generally helps (capped at 10 ≈ 40k tokens for cost).

---

## Key Takeaways

1. **Memory is a distinct axis from reasoning and from long context.** Memory is compressed/distilled and built incrementally; you cannot evaluate it by dumping the whole history at once.
2. **The incremental multi-turn protocol is the core contribution** — chunks injected one at a time, memory updated online, questions only after full ingestion.
3. **Different families have complementary, non-overlapping strengths** (RAG→retrieval, long-context→TTL+LRU), and **none covers all four**.
4. **Selective Forgetting / conflict resolution is the unsolved frontier** — multi-hop SF is ≤7% for every system, and even reasoning models fall apart at 32K.
5. This is a **benchmark paper**, not a solution paper: it defines the four competencies, supplies a unified protocol + two new datasets (EventQA, FactConsolidation), and quantifies how far current memory agents fall short.

---

## Where it sits (v1/v2)

**MemoryAgentBench is a "v2"-style memory benchmark** whose distinguishing move is **incremental / online memory construction** plus **explicit coverage of all four core competencies** (AR, TTL, LRU, SF) in one unified protocol.

- **vs. LOCOMO / LongMemEval (static or session-level history):** LOCOMO (~9k tokens) and other earlier QA benchmarks essentially present a fixed conversation history and ask questions over it — they primarily probe **Accurate Retrieval** and are short enough that modern models aren't stressed (per the paper's Table 1, LOCOMO covers only AR; LongMemEval covers AR + LRU). LongMemEval does inject sessions gradually but is limited by **low topical diversity** and **less realistic interaction patterns**. MemoryAgentBench instead **reconstructs long-context datasets into multi-turn chunked dialogues fed incrementally**, scales contexts to 100k–1.44M tokens, and is the **only benchmark in its comparison table that covers all four competencies** (AR + TTL + LRU + SF) *and* all three agent categories (Long-Context, RAG, Agentic Memory).
- **vs. long-context benchmarks (∞-Bench, NovelQA, NOCHA, Loong):** those provide the entire context in a single block and measure static reading comprehension — they cannot evaluate the abstraction/consolidation/updating that defines memory.
- **vs. MemBench / other memory-agent benchmarks (StoryBench, RealTalk, MemoryBank, PerLTQA):** these advance realism or scale on subsets of the competencies but, per the paper, none provides the combination of **incremental injection + full four-competency coverage + cross-family agent comparison**. MemoryAgentBench's specific additions are **Selective Forgetting as a first-class, stress-tested competency** (via the new FactConsolidation dataset) and **Test-Time Learning** (classification + recommendation), the two axes most often missing elsewhere.

In short: where v1 conversational-memory benchmarks asked "can the agent recall a fact from a long but static history?", MemoryAgentBench (v2) asks "as memory is **built turn-by-turn online**, can the agent **retrieve, learn new skills, understand globally, and forget stale facts** — all four?" — and finds today's agents can do some but never all.
