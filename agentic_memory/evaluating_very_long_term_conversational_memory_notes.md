# Evaluating Very Long-Term Conversational Memory of LLM Agents (LOCOMO)

**Authors:** Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang (UNC Chapel Hill, USC, Snap Inc.)

**Paper:** arXiv:2402.17753 | https://snap-research.github.io/locomo

---

## The Problem

Previous long-term dialogue research (including the MSC dataset used by MemGPT and MemoryBank) only evaluates over **~5 sessions and ~1K tokens**. That's not really "long-term." Nobody had tested whether LLMs, long-context models, or RAG actually work for *truly* long conversations — the kind spanning weeks or months with hundreds of turns.

This paper asks: **how well do current approaches actually handle very long-term conversational memory?** Spoiler: not well.

---

## What They Built

### 1. LOCOMO Dataset

A dataset of 50 very long-term conversations, dramatically larger than anything before:

| Dataset | Avg. Turns | Avg. Sessions | Avg. Tokens |
|---|---|---|---|
| MSC (used by MemGPT) | 53 | 4 | 1,226 |
| Conversation Chronicles | 59 | 5 | 1,055 |
| **LOCOMO** | **305** | **19** | **9,209** |

That's **9x longer** than MSC, across **4x more sessions**, spanning months of simulated time.

### 2. Data Generation Pipeline (Human-Machine)

Each conversation involves two virtual agents, each with:

**a) Rich Persona** — expanded from MSC persona seeds using GPT-3.5 into full personality descriptions (name, age, job, hobbies, relationships, etc.)

**b) Temporal Event Graph** — a causally-linked timeline of up to 25 life events over 6-12 months. E.g., "aspires to be hotel manager" -> "enrolls in hotel management course" -> "posts excitement about course on social media." Events have dates and causal edges (`caused_by` field). Generated iteratively in batches of k=3 using `text-davinci-003`.

**c) Memory-Augmented Agent Architecture** (based on Generative Agents / Park et al. 2023):

- **Short-term memory**: Recursive session summaries — summary of session k is conditioned on summary of session k-1 plus the raw conversation of session k.
- **Long-term memory**: A database of "observations" — assertive factual statements extracted from each conversation turn. E.g., "Nate won his first video game tournament" or "Joanna enjoys writing, reading, watching movies." Each observation is linked to source turn IDs for traceability.
- **Reflect & Respond**: Agent conditions its response on:
  - Latest session summary
  - Retrieved relevant observations from long-term memory
  - Current session conversation history
  - Persona statement
  - Events from temporal event graph that occurred between the last and current session
- **Image sharing & reaction**: Agents can share images (via web search from LLM-generated captions/keywords) and react to received images (via BLIP-2 captioning + LLM response generation).

**d) Human Verification & Editing** — annotators edited ~15% of dialog turns and removed/substituted ~19% of images to fix long-range inconsistencies and ensure grounding to event graphs. Specific edit types:
- Remove irrelevant images
- Fix information inconsistent with earlier/later turns
- Ensure conversation details match event graph
- Remove events from graph if not discussed in conversation

---

## The Evaluation Benchmark (3 Tasks)

### Task 1: Question Answering (Testing Memory Recall)

7,512 total questions across five reasoning categories:

1. **Single-hop** (36%) — answer from a single session
2. **Multi-hop** (15%) — synthesize information across multiple sessions
3. **Temporal reasoning** (21%) — requires understanding time-related cues in the conversation
4. **Open-domain knowledge** (4%) — needs external/commonsense knowledge combined with conversation info
5. **Adversarial** (25%) — designed to trick the model; correct answer is "unanswerable"

Metric: F1 score for exact matches with normalized answers. Each QA sample annotated with source turn IDs.

### Task 2: Event Summarization (Testing Causal/Temporal Understanding)

Given the full conversation, summarize the life events for each speaker. Compared against the ground-truth temporal event graphs using **FactScore** — decomposes both reference and prediction into atomic facts, then measures:
- **Precision**: how many predicted atomic facts match the ground truth
- **Recall**: how comprehensively the ground truth facts are covered
- **F1**: harmonic mean of precision and recall

### Task 3: Multi-Modal Dialogue Generation

Generate the next response (text + image) given conversation history. Evaluated with BLEU, ROUGE, and MM-Relevance.

---

## Experimental Setup

### Three Approaches Tested

1. **Base LLMs** — constrained context, earlier dialogues truncated. Models: Mistral-7B, Llama-2-70B-chat, GPT-3.5-turbo (4K), GPT-4-turbo (4K)
2. **Long-context LLMs** — extended context window. Model: GPT-3.5-turbo-16K (tested at 4K, 8K, 12K, 16K)
3. **RAG** — retrieve relevant context from a database, then use GPT-3.5-turbo-16K as reader. Retriever: DRAGON. Three retrieval unit types:
   - **Raw dialog turns** — retrieve top-k most similar dialog turns
   - **Observations** — retrieve top-k assertions/facts about speakers
   - **Session summaries** — retrieve top-k session-level summaries

---

## Key Experimental Results

### QA Task

#### Base Models (truncated context)

| Model | Overall F1 |
|---|---|
| Mistral-7B (8K) | 13.9 |
| Llama-2-70B (4K) | 17.9 |
| GPT-3.5-turbo (4K) | 22.4 |
| GPT-4-turbo (4K) | 32.1 |
| **Human** | **87.9** |

All models massively lag behind humans (56% gap for the best model).

#### Long-Context (GPT-3.5-turbo-16K)

| Context | Single-hop | Multi-hop | Temporal | Open-domain | Adversarial | Overall |
|---|---|---|---|---|---|---|
| 4K | 31.7 | 25.4 | 16.8 | 27.6 | 13.1 | 24.1 |
| 8K | 38.8 | 31.2 | 21.0 | 35.0 | 8.4 | 25.2 |
| 12K | 51.1 | 40.4 | 25.0 | 36.5 | 6.4 | 33.5 |
| 16K | 56.4 | 42.0 | 20.3 | 37.2 | **2.1** | 37.8 |

More context improves factual recall BUT **adversarial accuracy collapses from 13.1% to 2.1%**. The model hallucinates more with more context — gets easily tricked into answering unanswerable questions.

#### RAG (Best Configurations)

| Retrieval Unit | top-k | Overall F1 |
|---|---|---|
| None (base GPT-3.5) | - | 22.4 |
| Dialog | 5 | 31.7 |
| Dialog | 50 | 34.8 |
| **Observation** | **5** | **41.4** |
| Observation | 50 | 37.8 |
| Summary | 5 | 32.5 |
| Summary | 10 | 31.5 |

Key insight: **Observations (structured factual assertions) as retrieval units with just top-5 results achieves the best overall F1 (41.4)**. More retrieved results hurts performance — signal-to-noise ratio matters. Summaries perform worst despite high recall accuracy, likely due to information loss during summarization.

### Event Summarization Task

| Model | FactScore F1 |
|---|---|
| Mistral-7B (8K) | 23.0 |
| Llama-2-70B (4K) | 28.3 |
| GPT-3.5-turbo (4K) | **45.9** |
| GPT-4-turbo (4K) | 45.1 |
| GPT-3.5-turbo-16K | 39.9 |

The long-context model actually **performs worse** than the base model (39.9 vs 45.9 F1). Having more context doesn't help — the model fails to utilize it properly for understanding causal and temporal dynamics.

#### Five Categories of Summarization Errors

1. **Missing information** — fails to make temporal/causal connections over long conversation
2. **Hallucination** — pads events with non-existent details or details from different events
3. **Misunderstanding dialog cues** — confuses humor/sarcasm with serious statements
4. **Wrong speaker attribution** — assigns events to the wrong person
5. **Saliency errors** — treats trivial exchanges as significant life events

### Multi-Modal Dialog Generation

- Observation-augmented MiniGPT-5 outperforms base and summary-augmented versions
- MM-Relevance drops as conversation length increases
- RAG with observations partially mitigates the degradation

---

## Key Takeaways

1. **Current approaches are far from human-level** on very long-term conversations. Even the best setup (RAG with observations) achieves 41.4 F1 vs human's 87.9 — a 53% gap.

2. **Long-context models are a double-edged sword** — they improve factual recall but dramatically worsen on adversarial questions (hallucination increases from 13% to 98% error rate) and actually hurt event summarization. More context ≠ better understanding.

3. **RAG with observations > RAG with raw dialog > RAG with summaries.** Converting conversations into structured factual assertions (observations) before indexing is the best retrieval strategy. Don't just store raw chat logs — extract and structure the knowledge first.

4. **Temporal reasoning is the hardest problem** — 73% below human performance. Models fundamentally struggle with time-related reasoning in conversations.

5. **Signal-to-noise ratio matters** — retrieving fewer, more relevant items (top-5) outperforms retrieving more items (top-50). Quality over quantity in retrieved context.

6. **This is a benchmark paper, not a solution paper** — it identifies the problem and measures how bad current approaches are, setting the stage for future memory systems to be evaluated properly.

---

## Connections to MemGPT and MemoryBank

| Concept | MemGPT | MemoryBank | LOCOMO |
|---|---|---|---|
| Working memory / structured facts | Working context (key facts) | User portraits | Observations (assertive statements) |
| Raw history storage | Recall storage | In-depth memory | Dialog history database |
| Summarization | Recursive summary in FIFO queue | Hierarchical event summary | Session-level summaries |
| Evaluation scale | ~5 sessions, ~1K tokens (MSC) | 10 days, 15 users | **19 sessions, 9K tokens, months** |
| Key finding | Self-directed retrieval works | Forgetting curve improves naturalness | Structured observations > raw dialog for retrieval |

The LOCOMO findings validate and extend ideas from both papers:
- MemGPT's working context (structured facts) aligns with LOCOMO's finding that observations outperform raw dialog
- MemoryBank's hierarchical summarization is shown by LOCOMO to be the weakest retrieval strategy (information loss)
- Both MemGPT and MemoryBank were evaluated on relatively short contexts; LOCOMO shows the problem is much harder at real scale
