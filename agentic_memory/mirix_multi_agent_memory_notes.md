# MIRIX: Multi-Agent Memory System for LLM-Based Agents

**Authors:** Yu Wang, Xi Chen (MIRIX AI)

**Paper:** arXiv:2507.07957v1 (Jul 2025)

**GitHub:** https://github.com/Mirix-AI/MIRIX (project site: https://mirix.io/)

---

## The Core Problem

Most LLM-based personal assistants are **stateless beyond their prompt window** — they retain nothing unless context is re-provided. Existing memory-augmented systems try to fix this but remain fundamentally limited along three axes:

1. **Lack of compositional memory structure.** Systems like Mem0, Letta/MemGPT, and ChatGPT's memory store all history in a single flat store (distilled facts or text chunks) without routing into specialized memory types (procedural, episodic, semantic). This makes retrieval inefficient and inaccurate.
2. **Poor multi-modal support.** Text-centric memory mechanisms break down when the majority of input is non-verbal — images, interface layouts, maps, screenshots. Knowledge-graph systems (Zep, Cognee) model entity relationships well but cannot represent sequential events, emotional states, full documents, or images.
3. **Scalability and abstraction.** Storing raw inputs — especially high-resolution images — leads to prohibitive storage with no abstraction layer to summarize and retain only salient information. A long-context baseline like Gemini can only fit ~500 full-resolution screenshots (or ~3,600 resized to 256×256) into context.

MIRIX's thesis: the two key capabilities a memory-augmented agent must possess are **Routing** (sending each piece of information to the right specialized memory) and **Retrieving** (pulling the right memory back at query time). It builds a system designed explicitly around both — and crucially, one that **transcends text to handle visual/multimodal experience**.

---

## The Big Idea: Six Specialized Memory Components + a Multi-Agent Coordinator

MIRIX replaces the flat store with **six distinct, hierarchically-structured memory components**, each managed by its own dedicated agent, all coordinated by a central **Meta Memory Manager**. The system uses **eight agents total**: six Memory Managers (one per component) + the Meta Memory Manager + a Chat Agent.

### The Six Memory Components

| Component | What It Stores | Key Fields | Example |
|---|---|---|---|
| **Core Memory** | High-priority, always-visible persistent info (inspired by MemGPT) | `persona` block + `human` block | "User's name is David"; "User enjoys Japanese cuisine" |
| **Episodic Memory** | Time-stamped events / temporally-grounded interactions (acts like a structured log or calendar) | `event_type`, `summary`, `details`, `actor`, `timestamp` | "2025-03-05 10:15 — user asked to schedule a meeting" |
| **Semantic Memory** | Abstract, time-independent knowledge: concepts, named entities, the user's social graph | `name`, `summary`, `details`, `source` | "John is a friend of the user who enjoys jogging and lives in SF" |
| **Procedural Memory** | Goal-directed how-to processes: workflows, guides, scripts | `entry_type` (workflow/guide/script), `description`, `steps` | "Steps to file a travel reimbursement form" |
| **Resource Memory** | Full or partial documents, transcripts, multi-modal files the user is engaged with | `title`, `summary`, `resource_type` (doc/markdown/pdf_text/image/voice_transcript), `content` | A friend's picnic plan; a project proposal PDF |
| **Knowledge Vault** | Verbatim, sensitive info that must be preserved exactly | `entry_type` (credential/bookmark/contact_info/api_key), `source`, `sensitivity` (low/med/high), `secret_value` | Addresses, phone numbers, API keys, passwords |

The key design move: **Core Memory** rewrites itself when it exceeds 90% capacity (to stay compact); **Episodic** entries are time-indexed and track change over time; **Semantic** entries persist unless conceptually overwritten; **Knowledge Vault** high-sensitivity entries are access-controlled and excluded from casual retrieval to prevent leakage. This is the human cognitive-science taxonomy (episodic vs. semantic vs. procedural) made operational, rather than a single "long-term" catch-all.

---

## Architecture

### Multi-Agent Workflow

A single agent cannot manage this much heterogeneous, structured memory, so MIRIX is built as a **modular multi-agent system**:

- **Meta Memory Manager** — the central router. Analyzes incoming content and decides which memory components are relevant, dispatching the input to the corresponding Memory Managers.
- **Six Memory Managers** — one per component. Each maintains its own memory type, updates in **parallel**, and avoids redundancy within its type.
- **Chat Agent** — a demonstration agent (the memory system is pluggable into any agent) that talks to the user with full access to accumulated memories.

**Memory Update Workflow:** New input → system auto-searches the existing memory base → retrieved info + user input passed to Meta Memory Manager → Meta routes to relevant Memory Managers → managers update in parallel (deduplicating) → managers report back → Meta sends acknowledgment. On ScreenshotVQA, each update step makes one call to the Meta Memory Manager and between zero and six calls to the specialized managers.

**Conversational Retrieval Workflow:** On a user query, the Chat Agent first does a **coarse retrieval** across all six components returning high-level summaries; then it analyzes the query, decides which components warrant **targeted searches**, picks retrieval methods, consolidates, and synthesizes the response. If the query implies a memory update (a new fact or correction), the Chat Agent talks directly to the relevant Memory Managers to apply precise edits.

### Active Retrieval

In Mem0/MemGPT, retrieval must be **explicitly triggered**, or the LLM falls back on stale parametric knowledge (e.g., answering "Elon Musk" for "Who is the CEO of Twitter?" even after memory was told it's Linda Yaccarino). MIRIX's **Active Retrieval** removes the need for explicit prompting:

1. The agent first **generates a topic** from the input context (e.g., infers topic "CEO of Twitter" from the question).
2. That topic retrieves the **top-10 most relevant entries from each of the six components**.
3. Retrieved content is injected into the system prompt, **tagged by source** (`<episodic_memory>...</episodic_memory>`) so the model knows both content and origin.

Beyond Active Retrieval, MIRIX exposes multiple retrieval functions — `embedding_match`, `bm25_match`, `string_match` — and the agent selects the appropriate one per situation.

### The Multimodal Pipeline (the differentiator)

MIRIX's packaged app (React-Electron frontend, Uvicorn backend) captures a screenshot **every 1.5 seconds**, discards visually-similar duplicates, and triggers a memory update once **20 unique screenshots** accumulate (~every 60s). Key engineering: a **streaming upload strategy** — each screenshot is uploaded immediately rather than batched — leveraging the **Gemini API's** ability to load images via Google Cloud URLs. This cuts end-to-end latency from ~50s (GPT-4 direct image upload) to **under 5s** (Gemini). Critically, MIRIX **never stores raw images** — only the compact extracted information in a SQLite DB, which is the source of its enormous storage savings.

---

## Experimental Results

Both benchmarks use **LLM-as-a-Judge** (GPT-4.1 as judge) on accuracy. MIRIX and Full-Context are run 3× and averaged; baselines run once.

### ScreenshotVQA (the multimodal benchmark)

A **new benchmark** introduced by the paper: nearly **5,000–20,000 high-resolution (2K–4K) screenshots** spanning up to a month of real computer usage from **3 PhD students** (5,886 images / 1 day; 18,178 / 20 days; 5,349 / 1 month), with manually-authored, verified questions (11 / 21 / 55 respectively). No existing memory system (Letta, Mem0, etc.) can even process this multimodal input, so baselines are:
- **Gemini** (long-context): ingests screenshots resized to 256×256 (~3,600 most-recent fit in context).
- **SigLIP@50** (RAG): retrieves top-50 images per query (kept at native 2K–4K res), then Gemini answers.

Backbone: `gemini-2.5-flash-preview-04-17`. Storage = resized image size (Gemini) / full retrieved-image size (SigLIP) / `sqlite.db` size (MIRIX).

| Method | S1 Acc | S1 Store | S2 Acc | S2 Store | S3 Acc | S3 Store | **Overall Acc** | **Overall Store** |
|---|---|---|---|---|---|---|---|---|
| Gemini (long-ctx) | 0.0000 | 142.10 MB | 0.0952 | 438.86 MB | 0.2545 | 129.14 MB | 0.1166 | 236.70 MB |
| SigLIP@50 (RAG) | 0.3636 | 22.55 GB | 0.4138 | 19.88 GB | 0.5455 | 2.82 GB | 0.4410 | 15.07 GB |
| **MIRIX** | **0.5455** | **20.57 MB** | **0.5667** | **19.83 MB** | **0.6727** | **7.28 MB** | **0.5950** | **15.89 MB** |

Headline numbers (from the paper):
- vs **RAG (SigLIP)**: **+35%** accuracy (0.595 vs 0.441) while **reducing storage by 99.9%** (15.89 MB vs 15.07 GB).
- vs **long-context (Gemini)**: **+410%** accuracy (0.595 vs 0.117) with a **93.3% storage reduction** (15.89 MB vs 236.70 MB).

This is the headline contribution: a memory system that works on visual streams at all, and does so far more accurately *and* ~1000× more compactly than the only baselines that can run.

### LOCOMO (long-form text conversation)

10 conversations, ~600 dialogues / ~26K tokens each, ~200 questions per conversation. The Chat Agent answers using **only retrieved memories** (no transcript access). Adversarial category excluded (following Mem0) for fair comparison. Backbone `gpt-4.1-mini` (chosen over `gpt-4o-mini` for stronger function-calling: 29.75 vs 22.12 multi-turn acc on the Berkeley Function Calling Leaderboard). Top block reports baselines re-run with `gpt-4.1-mini`; lower block shows Mem0's published `gpt-4o-mini` numbers for reference.

| Backbone | Method | Single-Hop | Multi-Hop | Open-Domain | Temporal | **Overall** |
|---|---|---|---|---|---|---|
| gpt-4o-mini | A-Mem | 39.79 | 18.85 | 54.05 | 49.91 | 48.38 |
| gpt-4o-mini | LangMem | 62.23 | 47.92 | 71.12 | 23.43 | 58.10 |
| gpt-4o-mini | OpenAI | 63.79 | 42.92 | 62.29 | 21.71 | 52.90 |
| gpt-4o-mini | Mem0 | 67.13 | 51.15 | 72.93 | 55.51 | 66.88 |
| gpt-4o-mini | Mem0g | 65.71 | 47.19 | 75.71 | 58.13 | 68.44 |
| gpt-4o-mini | Memobase | 63.83 | 52.08 | 71.82 | 80.37 | 70.91 |
| gpt-4o-mini | Zep | 74.11 | 66.04 | 67.71 | 79.76 | 75.14 |
| gpt-4.1-mini | LangMem | 74.47 | 61.06 | 67.71 | 86.92 | 78.05 |
| gpt-4.1-mini | RAG-500 | 37.94 | 37.69 | 48.96 | 61.83 | 51.62 |
| gpt-4.1-mini | Zep | 79.43 | 69.16 | 73.96 | 83.33 | 79.09 |
| gpt-4.1-mini | Mem0 | 62.41 | 57.32 | 44.79 | 66.47 | 62.47 |
| gpt-4.1-mini | **MIRIX** | **85.11** | **83.70** | 65.62 | **88.39** | **85.38** |
| gpt-4.1-mini | Full-Context (upper bound) | 88.53 | 77.70 | 71.88 | 92.70 | 87.52 |

Key findings:
- **State-of-the-art overall: 85.38%**, beating the strongest open-source competitor (LangMem) by **over 8 points** and approaching the Full-Context upper bound (87.52%). (Note: at ~9K effective tokens, LOCOMO's Full-Context is essentially the ceiling, so nearly recovering it is the strong result.)
- **Multi-Hop is MIRIX's biggest win: 83.70**, outperforming all baselines by **24+ points** — and even *beating* Full-Context (77.70). MIRIX pre-consolidates dispersed facts into single events at write time (e.g., "Caroline moved from her hometown, Sweden, 4 years ago"), so the non-reasoning backbone doesn't have to stitch evidence at query time.
- **Single-Hop and Temporal** are strong (85.11 / 88.39), validating the hierarchical storage. The small Single-Hop gap vs Full-Context comes from ambiguous questions where MIRIX prioritizes the *confirmed consolidated event* over an earlier *plan*.
- **Open-Domain (65.62)** is MIRIX's weakest category and below Full-Context — "what-if" questions need global understanding, and MIRIX still relies on RAG-style retrieval, which lacks a global view.

(Appendix per-run: MIRIX overall = 83.98 / 87.34 / 84.82 across three runs — consistently SOTA despite variance. Note Zep with `gpt-4.1-mini` only hit 49.09 under the mem0 implementation, so the authors flagged a likely implementation error and reported the cleaner number in the main table.)

---

## Ablation / Component Analysis

MIRIX does not include a formal leave-one-component-out ablation table. Its component analysis is instead **qualitative and per-category** on LOCOMO:

- **Hierarchical, consolidated storage** drives the Multi-Hop and Single-Hop gains — writing pre-stitched consolidated events removes query-time reasoning burden.
- **Episodic + temporal indexing** drives the Temporal category (88.39).
- The **RAG-dependence** of retrieval is identified as the bottleneck on Open-Domain — the one category where the structured approach does not help and even trails Full-Context.

The ScreenshotVQA storage comparison effectively ablates the **abstraction layer**: MIRIX storing extracted info instead of raw pixels is what yields the 99.9% / 93.3% storage reductions while *improving* accuracy.

---

## Beyond the Benchmarks: Applications

The paper invests heavily in the productized vision:
- **Packaged personal assistant** — real-time screen monitoring, personalized memory base, local SQLite storage for privacy, memory visualization (Semantic Memory as a tree; Procedural Memory as a list).
- **Wearable devices** — AI glasses / pins continuously ingesting audio + visual + queries; hybrid on-device (Knowledge Vault local) / cloud (Resource Memory offloaded) memory management.
- **Agent Memory Marketplace** — a (speculative) decentralized ecosystem where structured personal memory becomes a tradeable digital asset, with encryption, fine-grained sharing permissions, and decentralized storage. This is forward-looking vision rather than evaluated work.

---

## Key Takeaways

1. **Specialized, routed memory beats a flat store.** Splitting memory into six cognitively-grounded components (Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault) with a Meta Memory Manager routing each input to the right place gives both better accuracy and cleaner retrieval than monolithic distilled-fact stores.

2. **Multi-agent coordination makes heterogeneous memory tractable.** One agent cannot maintain six structurally-different memory types; dedicating a Manager per type (updating in parallel) plus a central router is the organizing principle.

3. **Multimodal memory is the headline.** MIRIX is the first of these systems to operate on visual streams (screenshots) at scale, beating RAG by 35% and long-context by 410% — *and* doing so with ~1000× less storage by abstracting raw pixels into compact structured records rather than storing images.

4. **Write-time consolidation pays off at read time.** Pre-stitching dispersed facts into single consolidated events is what lets a non-reasoning backbone (`gpt-4.1-mini`) win Multi-Hop by 24+ points — even beating Full-Context.

5. **Active Retrieval removes the trigger problem.** Auto-generating a topic and injecting top-k source-tagged memories per component prevents the model from silently falling back on stale parametric knowledge.

---

## Limitations

1. **RAG bottleneck on global-understanding queries.** Open-Domain ("what-if") questions still trail Full-Context because retrieval remains fundamentally RAG-style and lacks a global view of the whole history.
2. **No formal component ablation.** The contribution of each of the six components (and the Meta Manager's routing) is argued qualitatively and per-category rather than isolated quantitatively.
3. **Backbone / function-calling dependence.** The multi-agent update path requires many function calls per step; performance is sensitive to the backbone's function-calling ability (the explicit reason for choosing `gpt-4.1-mini` over `gpt-4o-mini`), and the Gemini-specific streaming-upload optimization ties the multimodal pipeline to one provider's cloud-URL image loading.
4. **Small, narrow multimodal benchmark.** ScreenshotVQA is built from only 3 users and 87 total questions; the authors note building larger, more challenging real-world benchmarks as future work.
5. **Marketplace vision is unvalidated.** The Agent Memory Marketplace (privacy, decentralized storage, tokenized memory trading) is aspirational and not experimentally evaluated.

---

## Where it sits (v1/v2)

MIRIX fills the collection's biggest gap: **multimodal + multi-agent shared memory**. The survey frontiers list "multimodal memory" and "shared memory in multi-agent systems" as open problems, and until now this collection only had **single-agent, text-only** systems.

- **vs. A-Mem and MemoryOS** (single-agent, text-only): A-Mem builds Zettelkasten-style note graphs and MemoryOS adds OS-style short/mid/long-term tiers, but both operate over a single agent and consume only text. MIRIX is the first here to (a) run a genuine *multi-agent* memory system (eight coordinated agents, one router + six managers + chat) and (b) build memory from *visual* input (high-res screenshots) rather than text alone.

- **vs. MAGMA** (single-agent, multi-graph, text): MAGMA disentangles relational structure into four orthogonal *graph layers* and routes retrieval by query intent — sophisticated structure, but still single-agent and text-only. MIRIX instead disentangles by *memory type* across *parallel agents*, and adds the multimodal axis MAGMA explicitly lists as future work.

- **vs. LatentMem** (multi-agent, but latent/text): LatentMem is also multi-agent but shares memory in a *latent/learned* representation and remains text-centric. MIRIX is multi-agent with *explicit, human-readable, hierarchically-structured* memory components and is the one system here that crosses into the *visual* modality.

In short: within this collection, MIRIX is the **multimodal, multi-agent, explicitly-typed-memory** corner — complementary to MAGMA's graph-disentangled retrieval and LatentMem's latent shared memory, and a direct step beyond the single-agent text systems (A-Mem, MemoryOS). Its trade-off is that retrieval is still RAG-grounded (weak on global "what-if" reasoning) and its strongest evidence is on a small, newly-introduced visual benchmark.
