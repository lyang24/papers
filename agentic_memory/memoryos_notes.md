# MemoryOS: Memory Operating System of AI Agent

**Authors:** Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai (Beijing University of Posts and Telecommunications, Tencent AI Lab)

**Paper:** arXiv:2506.06326 (May 2025) | https://github.com/BAI-LAB/MemoryOS

---

## The Problem

Prior memory systems each tackle a **single dimension** in isolation:

- **MemGPT** (architecture-driven): OS-style paging with read/write calls, but its flat FIFO queue causes **topic mixing** as dialogue length grows
- **MemoryBank** (retrieval-oriented): forgetting curve + user portraits, but simple memory decay alone is **insufficient** for managing complex conversational memory
- **A-Mem** (knowledge-organization): graph-based interconnected notes, but **heavy multi-step link generation** inflates latency and accumulates errors
- **TiM / Think-in-Memory** (knowledge-organization): stores chains-of-thought, but single-stage hash retrieval **cannot preserve cross-topic dependencies**

No unified operating system had been proposed to systematically integrate storage, updating, retrieval, and generation into a single coherent framework.

---

## The Core Idea: Segment-Page Memory from Real OS Design

MemoryOS borrows **segment-page memory management** from real operating systems (Multics, many-core processors):

- **Segments** = logical groupings by topic (like memory segments in OS)
- **Pages** = individual dialogue turns within a segment (like memory pages within segments)
- **Heat-based eviction** = prioritize frequently-accessed, recently-used content (like LRU/working-set models in OS)

This is organized into a **three-tier hierarchy**: Short-Term Memory (STM) -> Mid-Term Memory (MTM) -> Long-term Personal Memory (LPM), managed by four modules: Storage, Updating, Retrieval, and Generation.

---

## Architecture: Four Modules

### Module 1: Memory Storage (Three Tiers)

#### Short-Term Memory (STM)

- Fixed-length queue (size 7) of dialogue pages
- Each page = `{Query Q, Response R, Timestamp T, meta_chain}`
- **Dialogue chain mechanism**: LLM evaluates whether a new page is contextually related to prior pages
  - If related: linked into the existing chain, chain summary updated
  - If semantically discontinuous: new chain starts from current page
  - Chain metadata = LLM-generated summary of all pages in the chain

#### Mid-Term Memory (MTM)

- **Segmented-paging architecture**: dialogue pages grouped by topic into segments
- A page is assigned to an existing segment if similarity exceeds threshold:

```
F_score(page, segment) = cos(e_s, e_p) + Jaccard(K_s, K_p) > θ
```

Where:
- `e_s`, `e_p` = embedding vectors of segment summary and page
- `K_s`, `K_p` = keyword sets (extracted by LLM) for segment and page
- `Jaccard = |K_s ∩ K_p| / |K_s ∪ K_p|`
- θ = 0.6

- If no segment matches, a new segment is created
- Segment content is summarized by LLM from its constituent pages
- Maximum capacity: 200 segments

#### Long-term Personal Memory (LPM)

**User Persona:**
- **User Profile** (static): fixed attributes — gender, name, birth year
- **User KB** (dynamic): factual information extracted from interactions, stored as FIFO queue (max 100 entries), incrementally updated
- **User Traits** (dynamic): 90 dimensions across three categories:
  1. Basic needs and personality
  2. AI alignment dimensions
  3. Content platform interest tags

**Agent Persona:**
- **Agent Profile** (static): role, character traits, consistent self-description
- **Agent Traits** (dynamic): attributes developed through interactions (e.g., recommended items, interaction history), FIFO queue (max 100 entries)

---

### Module 2: Memory Updating

#### STM -> MTM Update (FIFO)

- New dialogue pages are appended to STM queue
- When queue reaches capacity (7 pages), the oldest page is transferred to MTM
- Page is matched to an existing MTM segment by F_score, or creates a new segment

#### MTM -> LPM Update (Heat-Based)

Each segment has a **Heat score**:

```
Heat = α · N_visit + β · L_interaction + γ · R_recency
```

Where (α = β = γ = 1):
- `N_visit` = number of times the segment has been retrieved
- `L_interaction` = total number of dialogue pages in the segment
- `R_recency = exp(-Δt / μ)` — exponential time decay since last access (μ = 1e+7 seconds)

**Two operations based on Heat:**

1. **Segment-to-LPM transfer**: Segments with Heat > τ (threshold = 5) are transferred to LPM:
   - User Traits updated from segment content (90-dimension extraction by LLM)
   - User KB updated with factual information about the user
   - Agent Traits updated with factual information about the agent
   - After transfer, `L_interaction` resets to zero -> heat declines -> prevents redundancy

2. **Segment eviction**: When MTM exceeds maximum capacity, segments with lowest heat are deleted

#### LPM Update

User KB and Agent Traits maintain FIFO queues (max 100 entries each). User Traits are continuously evolved by LLM extraction from incoming segment content.

---

### Module 3: Memory Retrieval

Three-source retrieval for every user query:

**STM Retrieval:** All dialogue pages in STM are returned (full recent context — it's small enough to include entirely).

**MTM Retrieval (Two-Stage):**
1. Select **top-m segments** (m=5) by F_score matching against the query
2. Within selected segments, select **top-k dialogue pages** (k=5 or 10) by semantic similarity
3. After retrieval, segment's `N_visit` and `R_recency` are updated (increasing heat)

**LPM Retrieval:**
- User KB and Agent Traits: top-10 entries by semantic relevance to query
- User Profile, Agent Profile, and User Traits: **always fully included** (these are compact enough)

---

### Module 4: Response Generation

Final prompt construction:

```
Prompt = [STM context] + [MTM retrieved pages] + [LPM persona info] + [User Query]
```

The LLM generates a response that:
- Is **contextually coherent** with recent interactions (STM)
- Draws on **historical dialogue details** and topic summaries (MTM)
- Is **personalized** to user and agent identities (LPM)

---

## Experimental Results

### GVD Dataset (15 virtual users, 10 days of conversations)

| Model (GPT-4o-mini) | Accuracy | Correctness | Coherence |
|---|---|---|---|
| MemoryBank | 78.4 | 73.3 | 91.2 |
| TiM | 84.5 | 78.8 | 90.8 |
| MemGPT | 87.9 | 83.2 | 89.6 |
| A-Mem | 90.4 | 86.5 | 91.4 |
| **MemoryOS** | **93.3** | **91.2** | **92.3** |

Improvement over best baseline (A-Mem): +3.2% accuracy, +5.4% correctness, +1.0% coherence.

### LoCoMo Benchmark (GPT-4o-mini)

| Method | Single-hop F1 | Multi-hop F1 | Temporal F1 | Open-domain F1 |
|---|---|---|---|---|
| MemoryBank | 5.00 | 9.68 | 5.56 | 6.61 |
| TiM | 16.25 | 18.43 | 8.35 | 23.74 |
| MemGPT | 26.65 | 25.52 | 9.15 | 41.04 |
| A-Mem* | 22.61 | 33.23 | 8.04 | 34.13 |
| **MemoryOS** | **35.27** | **41.15** | **20.02** | **48.62** |

**Average improvement: 49.11% on F1, 46.18% on BLEU-1** over baselines.

Notable: **Temporal reasoning sees 119% improvement** — the heat-based time decay and timestamp-aware segment organization help enormously with time-related questions.

### Efficiency Comparison

| Method | Tokens (retrieval) | Avg. LLM Calls | Avg. F1 |
|---|---|---|---|
| MemoryBank | 432 | 3.0 | 6.84 |
| TiM | 1,274 | 2.6 | 18.01 |
| MemGPT | 16,977 | 4.3 | 29.13 |
| A-Mem* | 2,712 | 13.0 | 26.55 |
| **MemoryOS** | **3,874** | **4.9** | **36.23** |

MemoryOS uses **4.4x fewer tokens than MemGPT** and **2.7x fewer LLM calls than A-Mem** while achieving the best F1.

### Ablation Study

Impact of removing components (biggest drop -> smallest):

1. **-MemoryOS** (entire system): drastic performance collapse
2. **-MTM** (mid-term memory): most significant individual component
3. **-LPM** (persona module): second most impactful
4. **-Chain** (dialogue chain): least impact but still contributes

### Hyperparameter Analysis (top-k retrieved pages)

- Performance improves as k increases from 5 to 10-20
- Beyond k=20, improvements diminish and noise increases
- Optimal setting: k=10 (best performance/overhead trade-off)

---

## Key Takeaways

1. **Three tiers > two tiers.** Adding mid-term memory between short and long-term creates a buffer that organizes conversations by topic before committing to long-term persona storage.
2. **Heat-based eviction is better than pure FIFO or forgetting curve.** Combining retrieval frequency (`N_visit`), interaction depth (`L_interaction`), and recency (`R_recency`) into a single heat score keeps the most valuable content accessible.
3. **Segment-page organization solves topic mixing.** MemGPT's flat FIFO queue loses topic coherence as conversations grow; grouping pages into topical segments preserves it.
4. **The persona module matters.** 90-dimension user trait extraction + dynamic knowledge base enables genuinely personalized responses.
5. **Mid-term memory is the most critical component** — ablation shows MTM contributes more than LPM or the dialogue chain.
6. **Efficient by design** — fewer tokens and LLM calls than MemGPT and A-Mem while outperforming both.

---

## Cross-Paper Comparison

| Aspect | MemGPT | MemoryBank | LOCOMO | ReadAgent | MemoryOS |
|---|---|---|---|---|---|
| Memory tiers | 2 (main + external) | 1 (flat storage) | N/A (benchmark) | 2 (gist + raw) | **3 (STM/MTM/LPM)** |
| Topic organization | None (flat FIFO) | None | N/A | Episode pages | **Segment-page** |
| Eviction strategy | FIFO + recursive summary | Ebbinghaus decay | N/A | N/A | **Heat-based (freq + depth + recency)** |
| Persona modeling | Not a focus | User portraits | N/A | N/A | **90-dim user traits + dynamic KB** |
| Retrieval | LLM self-directed | Auto dense retrieval | Various tested | LLM over gists | **Two-stage (segment then page)** |
| LoCoMo F1 (GPT-4o-mini) | 29.13 avg | 6.84 avg | Benchmark paper | N/A | **36.23 avg** |
| Key weakness | Topic mixing in long convos | Decay alone insufficient | N/A | No persistent memory | More complex system to configure |

MemoryOS can be seen as the synthesis of ideas from all prior papers:
- From **MemGPT**: the OS metaphor and hierarchical storage
- From **MemoryBank**: the time-decay concept (refined into heat-based eviction)
- From **LOCOMO**: the evaluation benchmark and the finding that structured knowledge > raw dialog
- From **ReadAgent**: the idea that semantic chunking (segment-page) outperforms uniform chunking
