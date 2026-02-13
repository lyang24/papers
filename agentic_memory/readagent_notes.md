# ReadAgent: A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts

**Authors:** Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, Ian Fischer (Google DeepMind)

**Paper:** ICML 2024 | arXiv:2402.09727 | read-agent.github.io

---

## The Problem

LLMs have two limitations with long text:

1. An explicit **context length limit** (e.g., 8K tokens for PaLM 2-L)
2. Performance **degrades** even within the context window — "Lost in the Middle" (Liu et al., 2023) shows LLMs attend poorly to information in the middle of long inputs, and irrelevant/distracting information in context actually hurts performance (Shi et al., 2023)

Humans don't work this way. We read a book, retain a **fuzzy gist** of each section, and when we need a specific detail, we **flip back** to the relevant page.

---

## The Core Idea: Fuzzy-Trace Theory Meets LLMs

Inspired by **Fuzzy-Trace Theory** (Reyna & Brainerd, 1995) from psychology:

- Humans form two types of memory: **verbatim** (exact details) and **gist** (substance/meaning irrespective of exact words)
- People prefer to reason with gists rather than verbatim memories
- When details are needed, humans look them up in the original source
- Gist memories last much longer than verbatim memories

ReadAgent replicates this in three steps: **Episode Pagination** -> **Memory Gisting** -> **Interactive Look-Up**.

---

## Architecture: Three Steps

### Step 1: Episode Pagination

The LLM reads through the document and decides **where to pause** — creating natural "pages" (episodes). Unlike fixed-size chunking, the LLM picks semantically meaningful break points:

- Scene transitions
- End of dialogues
- Narrative transitions
- End of arguments

**Implementation:** Numbered tags (e.g., `<5>`, `<6>`, `<7>`) are inserted between paragraphs after a `min_words` threshold. The LLM is given text up to `max_words` and picks which tag is a natural pause point. Content between consecutive pause points becomes a "page."

**Hyperparameters:**

| Dataset | max_words | min_words |
|---|---|---|
| QuALITY | 600 | 280 |
| QMSum | 600 | 280 |
| NarrativeQA Gutenberg | 3000 | 500 |
| NarrativeQA Movie Scripts | 1000 | 600 |

### Step 2: Memory Gisting (Compression)

Each page is independently compressed into a short **gist** by prompting the LLM to "shorten" the passage.

Key design choices:
- Uses the word **"shorten"** not "summarize" — "shorten" preserves narrative flow better for concatenation, while "summarize" tends to restructure the content
- Each gist is tagged with its page number (e.g., `<Page 2> {GIST CONTENT}`) for contextualization
- All gists are concatenated in order to form the **gist memory**

**Compression rates achieved:**
- QuALITY: 85.53% compression (4,122 words -> ~650 words avg.)
- NarrativeQA Gutenberg: 96.80% compression (70,619 words -> ~2,217 words avg.)
- NarrativeQA Movies: 91.98% compression
- QMSum: 83.13% compression

The larger the page size, the more compression is possible (more redundancy between neighboring paragraphs to remove), but also more details are lost.

### Step 3: Interactive Look-Up and Response

Given a task (e.g., a question), the LLM reads all gists + the question, then decides which original page(s) to look up. Two strategies:

**ReadAgent-P (Parallel):**
- Selects multiple pages at once in a single prompt
- Instructed to select 1-N pages but use as few as possible to avoid distraction
- Lower computational cost

**ReadAgent-S (Sequential):**
- Selects one page at a time
- Sees previously expanded pages before choosing the next
- More informed decisions but higher cost (up to N LLM calls for N pages)
- Better for less structured documents (e.g., meeting transcripts)

**After look-up:** The selected raw pages **replace** their corresponding gists in the gist memory (preserving overall narrative flow). The LLM then answers the question from this hybrid context of gists + expanded pages.

---

## Why This Beats Standard RAG

Standard RAG uses embedding similarity to retrieve relevant chunks. ReadAgent uses **the LLM's own reasoning over the gist memory** to decide what to look up. Two key advantages:

### 1. Global Context

The gist memory provides a bird's-eye view of the entire document. The LLM understands *where* information fits in the overall narrative before choosing what to look up. Standard RAG has no such global awareness.

### 2. Better Retrieval Decisions

The LLM reasons about what pages are actually relevant, rather than relying on embedding cosine similarity which can surface semantically-similar-but-distracting pages.

### Case Study: "Off Course" (Short Story)

Question: "What was Dameri's purpose in landing on earth?"
- Correct answer: (D) He arrived on accident
- **Neural retrieval chose (C)** — "interesting animal specimens" — because it retrieved pages prominently mentioning animals, which were distracting
- **ReadAgent chose (D)** — it had the gist of the full story providing global context, and looked up only the 2 most relevant pages

In all three examples where ReadAgent beat retrieval on this story, ReadAgent chose to look up only 2 pages (out of a maximum of 4 allowed), avoiding unnecessary distraction. This flexibility is itself an advantage over standard top-k retrieval.

---

## Experimental Results

### QuALITY (Multiple-Choice QA, ~4K words avg.)

| Method | Compression Rate | Accuracy |
|---|---|---|
| BM25 Top-4 | 58.57% | 84.42% |
| Neural Retrieval Top-4 | 60.68% | 84.88% |
| Full Raw Content | 0% | 85.83% |
| Gist Memory only | 85.53% | 77.52% |
| **ReadAgent-P (1-2 pages)** | 72.17% | **86.16%** |
| **ReadAgent-S (1-6 pages)** | 58.53% | **87.17%** |

ReadAgent **outperforms using the full original text** while only consuming ~28% of the words (**3.5x effective context extension**). Less (but better-targeted) context beats more context.

### NarrativeQA Gutenberg Test (Books, ~71K words avg., max 344K)

| Method | LR-1 | LR-2 | ROUGE-L |
|---|---|---|---|
| BM25 Top-4 | 53.60% | 66.16% | 0.197 |
| Neural Retrieval Top-4 | 50.62% | 62.05% | 0.191 |
| Gist Memory only | 55.79% | 71.19% | 0.217 |
| **ReadAgent-P (1 page)** | **59.98%** | **73.23%** | **0.226** |
| **ReadAgent-S (1-3 pages)** | **60.55%** | **72.79%** | 0.219 |

At ~97% compression (**~20x effective context length**), ReadAgent improves LR-1 by ~13% and ROUGE-L by ~32% over the best retrieval baseline. An 8K-token LLM can effectively handle 344K-word books.

### QMSum (Meeting Transcripts, ~10K words avg.)

| Method | LR-1 | LR-2 |
|---|---|---|
| BM25 Top-5 | 39.09% | 84.44% |
| Neural Retrieval Top-6 | 40.81% | 87.01% |
| Truncated (first 6K words) | 14.71% | 52.45% |
| Gist Memory only | 40.20% | 89.83% |
| ReadAgent-P (1-4 pages) | 39.95% | 90.56% |
| **ReadAgent-S (1-6 pages)** | **46.57%** | **91.54%** |

ReadAgent-S significantly outperforms ReadAgent-P here. Meeting transcripts are less structured than stories — sequential look-up lets the model actively search through the gist to locate relevant information, which matters more for unstructured documents.

---

## Key Ablations

### LLM Pagination vs. Fixed-Size Chunking

| Pagination Method | Accuracy (QuALITY) |
|---|---|
| LLM-decided pause points | **86.83%** |
| Uniform-length pages | 85.71% |

Semantically meaningful boundaries help comprehension.

### ReadAgent Retrieval vs. Neural Retrieval (both look up 1 page)

| Method | Accuracy (QuALITY) |
|---|---|
| Gist Memory + Neural Retrieval Top-1 | 82.65% |
| **ReadAgent-P (1 page)** | **84.13%** |

The LLM's reasoning over gist context makes better retrieval decisions than embedding similarity.

### Compression Trade-Off

| max_words | GistMem Accuracy | ReadAgent (1-5 pgs) Accuracy |
|---|---|---|
| 400 (81.8% CR) | 78.91% | **86.82%** |
| 600 (85.5% CR) | 77.52% | 86.83% |
| 800 (88.1% CR) | 76.22% | 86.34% |
| 1200 (91.4% CR) | 73.97% | 85.67% |

Higher compression hurts gist-only performance, but ReadAgent with look-ups is robust across compression levels. At very high compression, look-up accuracy starts to suffer.

---

## Computational Cost Analysis

- **Pagination**: upper bound is `max_words / min_words` passes of the document
- **Gisting**: one additional pass of the raw input
- **Look-ups**: operate on gists (much shorter), bounded by max look-ups allowed
- **Response**: similar to look-up cost

Gisting is a **one-time cost** that can be amortized across many questions about the same document. On QuALITY (230 articles, 2086 questions):
- Full text: 8,708,434 words consumed
- ReadAgent (1-page look-up): 6,499,856 words (**25.4% savings**)
- ReadAgent (2-page look-up): 6,933,357 words (**20.4% savings**)

---

## Bonus: Web Navigation (Mind2Web)

ReadAgent was adapted to **web page navigation**:
- **Pagination**: HTML decomposed by DOM tree depth (depth 5-7) instead of LLM-decided
- **Gisting**: HTML snippets summarized into gists
- **Look-up**: Agent selects relevant snippets to decide what to click/type/select

ReadAgent outperforms raw HTML input (+11-15% element accuracy) and even beats MindAct with a supervisedly-trained retriever, despite being fully zero-shot. 97.4% of gisted web inputs fit in 8K context vs. only 51.5% of raw HTML.

---

## Key Takeaways

1. **Gist + look-up > full context.** Counter-intuitively, compressed gists plus targeted page look-ups outperform giving the LLM the entire document. Less distraction, more focus.
2. **LLM-as-retriever beats embedding-based retrieval.** When the LLM can see a global summary and reason about what to look up, it makes better retrieval decisions than cosine similarity.
3. **No training needed.** ReadAgent is pure prompting — no fine-tuning, no trained retriever, works with any instruction-tuned LLM.
4. **Scales to 20x effective context.** A 344K-word book can be handled by an 8K-token context LLM.
5. **The human reading analogy works.** Episode pagination (deciding what to "page" together), gisting (fuzzy memory), and look-up (flipping back) directly mirror how humans read long documents.
6. **Semantic pagination matters.** LLM-decided break points outperform uniform chunking.

---

## Connections to Other Papers

| Concept | MemGPT | MemoryBank | LOCOMO | ReadAgent |
|---|---|---|---|---|
| Core metaphor | OS virtual memory | Human forgetting | Benchmark for memory | Human reading process |
| Compression | Recursive summarization | Hierarchical event summary | Session summaries + observations | **Gist memory** (shortened pages) |
| Retrieval trigger | LLM self-directs via function calls | Automatic on every message | Varies by setup | **LLM reasons over gists** |
| Retrieval method | Cosine similarity (pgvector) | Cosine similarity (FAISS) | Cosine similarity (DRAGON) | **LLM prompt-based reasoning** |
| Key insight | Let LLM manage its own memory | Forgetting makes memory more natural | Observations > raw dialog > summaries | **Gist + targeted look-up > full context** |

ReadAgent's finding that **LLM-based retrieval over compressed gists beats embedding-based retrieval** is complementary to LOCOMO's finding that **structured observations beat raw dialog retrieval**. Both suggest that the standard RAG pipeline (embed -> cosine similarity -> top-k) leaves significant performance on the table.
