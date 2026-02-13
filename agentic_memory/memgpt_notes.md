# MemGPT: Towards LLMs as Operating Systems

**Authors:** Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)

**Paper:** https://research.memgpt.ai | arXiv:2310.08560

---

## The Core Problem

LLMs have **fixed-size context windows** (e.g., 4k-128k tokens). This creates two major bottlenecks:

1. **Long conversations** - after a few dozen messages, earlier messages fall out of context and the model "forgets"
2. **Long documents** - legal filings, annual reports, etc. can easily exceed millions of tokens

Simply scaling context windows has **diminishing returns**: the quadratic cost of self-attention grows fast, and research ("Lost in the Middle," Liu et al. 2023) shows models struggle to use information in the middle of large contexts anyway.

---

## The Big Idea: Treat the Context Window Like RAM

MemGPT borrows the concept of **virtual memory** from operating systems:

| OS Concept | MemGPT Analogy |
|---|---|
| Physical RAM | LLM's context window (prompt tokens) |
| Disk storage | External databases (archival + recall storage) |
| Virtual memory / paging | LLM autonomously moves data in/out of context |
| Page faults | LLM searches external storage to retrieve needed info |
| Memory pressure warnings | System alerts when context is ~70% full |

The key insight: **let the LLM itself manage what's in its context** by giving it function calls to read/write/search external storage.

---

## Architecture

### Main Context (3 Components in the Prompt Tokens)

The LLM's prompt tokens are split into three contiguous sections:

1. **System Instructions** (read-only, static) - describes the memory hierarchy and available functions, the MemGPT control flow, and instructions on how to use memory functions.
2. **Working Context** (read-write, fixed-size) - a scratchpad for key facts, user preferences, persona info. Modified only via function calls like `working_context.append()` or `working_context.replace()`.
3. **FIFO Queue** (read-write) - rolling message history including user messages, system messages, and function call I/O. Old messages get evicted and summarized recursively. The first slot always holds a recursive summary of everything evicted so far.

### External Storage (2 Databases)

- **Recall Storage** - full message history database. Searchable via `recall_storage.search(query)`. Every message ever exchanged is saved here by the queue manager.
- **Archival Storage** - long-term knowledge store (e.g., uploaded documents, accumulated facts). Searchable via `archival_storage.search(query)` with pagination. Backed by PostgreSQL with pgvector for vector search.

### Queue Manager

Manages messages between recall storage and the FIFO queue:

- Appends incoming messages to the FIFO queue
- Writes both incoming and generated messages to recall storage
- Handles **context overflow** via a queue eviction policy:
  - At ~70% capacity: inserts a "memory pressure" warning so the LLM can proactively save important info
  - At 100% capacity: flushes ~50% of messages, generates a new recursive summary from the existing summary + evicted messages

### Function Executor

Parses the LLM's output as function calls, validates arguments, executes them, and feeds results (including runtime errors) back to the LLM. This feedback loop lets the system learn from its actions and adjust.

---

## Control Flow and Function Chaining

Events trigger LLM inference. Events can be:

- User messages
- System messages (e.g., memory pressure warnings)
- User interactions (e.g., document upload alerts)
- Timed/scheduled events (allowing unprompted operation)

MemGPT supports **chaining function calls** before responding to the user. A special `request_heartbeat=true` flag tells the system: "run inference again immediately after this function completes, don't wait for the user." This enables multi-step retrieval - e.g., searching for a key, finding a value that is itself a key, searching again, etc.

If the flag is absent (a "yield"), MemGPT pauses until the next external event.

---

## Worked Example: Conversational Memory

Here is the flow with a concrete chatbot scenario:

1. User says: "my bf James baked me a birthday cake"
2. MemGPT calls `working_context.append("Boyfriend named James")` and `working_context.append("Birthday is February 7")`
3. Many messages later, context fills up -> **System Alert: Memory Pressure**
4. MemGPT proactively saves important info before eviction
5. Later, user says: "actually James and I broke up"
6. MemGPT calls `working_context.replace("Boyfriend named James", "Ex-boyfriend named James")`
7. Even later, user mentions Six Flags -> MemGPT calls `recall_storage.search("six flags")` and retrieves old messages including "James and I first met at Six Flags"

The agent **autonomously decides** when to store, update, and retrieve information - no user intervention required.

---

## Experimental Results

### 1. Conversational Agents (Multi-Session Chat)

Based on the Multi-Session Chat (MSC) dataset (Xu et al., 2021) with 5 chat sessions per conversation.

#### Deep Memory Retrieval (DMR) - Testing Consistency

The agent is asked a question that explicitly refers back to a prior conversation session.

| Model | Accuracy | ROUGE-L (R) |
|---|---|---|
| GPT-3.5 Turbo (baseline) | 38.7% | 0.394 |
| GPT-3.5 Turbo + MemGPT | 66.9% | 0.629 |
| GPT-4 (baseline) | 32.1% | 0.296 |
| GPT-4 + MemGPT | **92.5%** | **0.814** |
| GPT-4 Turbo (baseline) | 35.3% | 0.359 |
| GPT-4 Turbo + MemGPT | **93.4%** | **0.827** |

Baselines only see a lossy summary of past sessions. MemGPT has full access to conversation history via paginated search, nearly tripling accuracy.

#### Conversation Opener Task - Testing Engagement

MemGPT-powered agents craft openers that reference past conversations, matching or exceeding human-written openers in similarity scores (SIM-1, SIM-3, SIM-H).

### 2. Document Analysis

#### Multi-Document Question Answering

- MemGPT's accuracy is **independent of document count** - it pages through archival storage as needed
- Fixed-context baselines degrade as documents are truncated to fit the context window
- MemGPT can iteratively page through retriever results, while baselines are limited to whatever fits in context

#### Nested Key-Value Retrieval

A synthetic task where values can themselves be keys, requiring multi-hop lookups (0-4 nesting levels, 140 UUID pairs).

| Model | Nesting Level 0 | Nesting Level 1 | Nesting Level 2 | Nesting Level 3 |
|---|---|---|---|---|
| GPT-3.5 (baseline) | ~good | 0% | 0% | 0% |
| GPT-4 (baseline) | ~good | ~good | drops | 0% |
| GPT-4 + MemGPT | ~perfect | ~perfect | ~perfect | **~perfect** |

MemGPT with GPT-4 maintains near-perfect accuracy at all nesting levels by chaining search calls. Fixed-context models hit 0% by level 3.

---

## Key Takeaways

1. **The OS metaphor is powerful** - treating context as RAM and using paging/eviction is a clean abstraction for memory management in LLM agents.
2. **Self-directed memory management works** - the LLM can learn to manage its own memory via function calls described in the system prompt, no fine-tuning needed.
3. **Recursive summarization + searchable history** is better than just stuffing everything into a bigger context window.
4. **Function chaining** enables multi-step reasoning that fixed-context models cannot do.
5. **Memory pressure warnings** are key - they give the LLM a chance to proactively save important information before eviction happens.

---

## Retrieval Mechanism Details

Both archival storage and recall storage use **dense vector similarity search** under the hood:

- **Embedding model:** OpenAI `text-embedding-ada-002`
- **Database:** PostgreSQL with the **pgvector** extension
- **Index:** HNSW (Hierarchical Navigable Small World) for approximate, sub-second query times
- **Similarity metric:** Cosine similarity
- **Pagination:** Results are returned 10 per page to avoid overflowing context

### MemGPT vs MemoryBank: Retrieval Comparison

Both systems use cosine similarity over dense embeddings. The fundamental difference is **who controls retrieval**:

| | MemGPT | MemoryBank |
|---|---|---|
| Retrieval mechanism | Cosine similarity (pgvector + ada-002) | Cosine similarity (FAISS + MiniLM/Text2vec) |
| **Who triggers retrieval** | **The LLM itself** via function calls | **The system automatically** on every user message |
| Query formulation | **LLM decides** what to search for | Current conversation context is the query |
| Multi-step retrieval | Yes - LLM can chain searches, paginate, refine queries | No - single retrieval pass |

- **MemoryBank:** Every time the user sends a message, the system automatically encodes it and retrieves the top-k most similar memories. The LLM has no say in what gets retrieved.
- **MemGPT:** The LLM **actively decides** when to search, what query to use, and whether to page through more results. It can also choose *not* to search at all. This is what makes the nested KV task possible - the LLM chains multiple searches with different queries.

Both use dense vector similarity under the hood, but MemGPT gives the LLM **agency over its own retrieval**, while MemoryBank treats retrieval as an automatic preprocessing step.

---

## Future Directions (from the paper)

- Applying MemGPT to other domains with massive or unbounded contexts
- Integrating different memory tier technologies (databases, caches)
- Improving control flow and memory management policies
- Bridging more OS architecture concepts into AI systems
