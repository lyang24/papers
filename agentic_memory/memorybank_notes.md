# MemoryBank: Enhancing Large Language Models with Long-Term Memory

**Authors:** Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, Yanlin Wang (Sun Yat-Sen University, HIT, KTH)

**Paper:** arXiv:2305.10250 | https://github.com/zhongwanjun/MemoryBank-SiliconFriend

---

## The Problem

LLMs have no persistent memory across conversations. This is particularly painful for scenarios requiring **sustained interaction over days/weeks/months**:

- Personal AI companions
- Psychological counseling
- Secretarial assistants

Without long-term memory, the AI can't recall what you talked about yesterday, can't track your emotional state over time, and can't learn your personality.

---

## The Big Idea: A Memory System Inspired by Human Forgetting

MemoryBank is a **plug-and-play memory module** for any LLM (closed-source like ChatGPT or open-source like ChatGLM/BELLE). It has three pillars:

1. **Memory Storage** - the warehouse
2. **Memory Retrieval** - finding relevant memories
3. **Memory Updating** - forgetting and reinforcing, inspired by the Ebbinghaus Forgetting Curve

The key differentiator from MemGPT: where MemGPT focuses on the OS/paging metaphor to handle context overflow, MemoryBank focuses on **psychologically-inspired forgetting** and **user personality modeling**.

---

## Architecture: Three Pillars

### 1. Memory Storage (The Warehouse)

Three layers of information are stored:

- **In-Depth Memory:** Every conversation turn is stored with timestamps in chronological order - the raw, full-resolution record.
- **Hierarchical Event Summary:** Daily conversations are summarized into daily event summaries, which are further synthesized into a global summary. This mirrors how humans remember the gist of their week, not every word. Prompt used: "Summarize the events and key information in the content [dialog/events]".
- **Dynamic Personality Understanding:** The system continuously builds a **user portrait** - daily personality insights aggregated into a global personality profile. E.g., "Linda is introverted, determined, ambitious, and values personal growth." Prompt used: "Based on the following dialogue, please summarize the user's personality traits and emotions. [dialog]".

### 2. Memory Retrieval (Dense Retrieval)

Uses a **dual-tower dense retrieval model** (similar to Dense Passage Retrieval / DPR):

- Every conversation turn and event summary is a "memory piece" encoded into a vector using an encoder model `E(·)`
- Vectors are indexed with **FAISS** for fast similarity search
- Current conversation context is encoded as a query, and the most similar memories are retrieved
- The encoder is swappable (MiniLM for English, Text2vec for Chinese)
- Implemented via LangChain

### 3. Memory Updating (The Forgetting Curve)

This is the most novel component. Inspired by **Ebbinghaus' Forgetting Curve**:

```
R = e^(-t/S)
```

Where:

- `R` = memory retention (probability of remembering)
- `t` = time elapsed since the memory was formed
- `S` = memory strength (starts at 1, increments each time the memory is recalled)
- `e` ≈ 2.71828

Three principles from Ebbinghaus:

1. **Rate of Forgetting** - memory retention decreases over time
2. **Time and Memory Decay** - steep initial forgetting, then leveling off
3. **Spacing Effect** - recalling a memory resets the curve and makes it harder to forget

When a memory is recalled in conversation, **S increases by 1 and t resets to 0** - the memory becomes more durable. Unrecalled memories naturally decay over time and can be pruned.

Note: The authors acknowledge this is a simplified model. Real memory processes are more complex and vary by person and information type.

---

## SiliconFriend: The Application

To demonstrate MemoryBank, they built **SiliconFriend**, a long-term AI companion chatbot, in two stages:

### Stage 1: Psychological Fine-Tuning (open-source models only)

- Fine-tuned with **38k psychological dialogues** parsed from online sources
- Uses **LoRA** (Low-Rank Adaptation) for parameter-efficient tuning (rank 16, 3 epochs on A100)
- Gives the model empathy, emotional awareness, and counseling ability
- Applied to ChatGLM (6.2B params) and BELLE (7B LLaMA-based); ChatGPT used as-is

### Stage 2: MemoryBank Integration

- Conversations are logged to memory storage with timestamps
- Memory updating runs using Ebbinghaus Forgetting Curve principles
- During conversation, the user's message becomes the retrieval query
- The prompt is augmented with: **relevant memories + global user portrait + global event summary**
- SiliconFriend generates responses that reference past memories and are tailored to the user's portrait

### Three LLM Backends

1. **ChatGPT** (closed-source) - proprietary, strongest overall abilities
2. **ChatGLM** (open-source, 6.2B params) - bilingual, optimized for Chinese
3. **BELLE** (open-source, 7B LLaMA-based) - bilingual, uses ChatGPT-synthesized instruction data

---

## How It Works in Practice

### Memory Recall Example

1. Day 1: User asks "I want to learn Python." SiliconFriend recommends the book *Automate the Boring Stuff with Python*
2. Day 2: User asks "Please write a quicksort program." SiliconFriend writes it
3. Day 7: User asks "You once recommended a book to me, what's its name?" -> SiliconFriend correctly recalls the book
4. User asks "Did we write the heap sort algorithm together?" -> SiliconFriend correctly says **no**, they didn't

### Personality-Aware Responses

- For "Linda" (introverted, values personal growth): recommends cooking classes, museums, art exhibitions
- For "Emily" (open-minded, curious): recommends hiking, bike rides, learning instruments
- Same question ("suggestions for weekend activities?"), different answers tailored to personality profiles

### Empathetic Companionship

When a user says "I recently broke up with my girlfriend", SiliconFriend (vs baseline ChatGLM):
- Provides more empathetic, emotionally supportive responses
- Offers constructive suggestions while acknowledging emotions
- Maintains a positive, encouraging tone throughout the conversation

---

## Experimental Results

### Setup

- 15 virtual users with diverse personalities (generated by ChatGPT)
- 10 days of simulated conversations covering diverse topics
- 194 probing questions (97 English, 97 Chinese) to test memory recall
- Human annotators score retrieval accuracy, response correctness, coherence, and ranking

### Results

| Model | Retrieval Acc. | Correctness | Coherence | Ranking |
|---|---|---|---|---|
| SiliconFriend ChatGLM (EN) | 0.809 | 0.438 | 0.680 | 0.498 |
| SiliconFriend BELLE (EN) | 0.814 | 0.479 | 0.582 | 0.517 |
| SiliconFriend ChatGPT (EN) | 0.763 | **0.716** | **0.912** | **0.818** |
| SiliconFriend ChatGLM (CN) | 0.840 | 0.418 | 0.428 | 0.510 |
| SiliconFriend BELLE (CN) | 0.856 | 0.603 | 0.562 | 0.565 |
| SiliconFriend ChatGPT (CN) | 0.711 | **0.655** | **0.675** | **0.758** |

### Key Findings

1. **Retrieval accuracy is high across all models** (~76-86%) - MemoryBank's retrieval works regardless of the base LLM
2. **ChatGPT dominates in correctness and coherence** - the base model's capability matters for generating good answers from retrieved memories
3. **Language matters** - ChatGLM and ChatGPT are better in English; BELLE is better in Chinese
4. **Open-source models can achieve comparable retrieval** but lag in response quality due to weaker base model capabilities

---

## Key Takeaways

1. **The forgetting curve is a clever addition** - rather than storing everything forever (which gets noisy), the Ebbinghaus-inspired decay naturally prunes unimportant memories while reinforcing frequently-recalled ones.
2. **User portraits are powerful** - continuously building and updating a personality profile lets the AI tailor responses over time.
3. **It's model-agnostic** - works with both open and closed-source LLMs as a plug-in module.
4. **Hierarchical summarization** (raw conversations -> daily summaries -> global summary) gives the AI both fine-grained and bird's-eye views of history.
5. **Psychological fine-tuning matters** - LoRA tuning on 38k psychological dialogues significantly improves empathy and emotional support quality.

---

## Comparison with MemGPT

| Aspect | MemGPT | MemoryBank |
|---|---|---|
| Core metaphor | OS virtual memory / paging | Human cognitive memory |
| Primary goal | Overcome context window limits | Long-term retention + personality modeling |
| Memory management | LLM self-directs via function calls | External module with forgetting curve |
| Forgetting mechanism | Eviction + recursive summarization | Ebbinghaus exponential decay |
| User modeling | Not a focus | Core feature (user portraits) |
| Retrieval | Vector search in archival storage | Dense retrieval with FAISS |
| LLM integration | Requires function calling support | Plug-and-play with any LLM |

The two approaches are **complementary**: MemGPT handles the "how to fit more into context" problem, while MemoryBank handles the "what to remember long-term and how to forget gracefully" problem.
