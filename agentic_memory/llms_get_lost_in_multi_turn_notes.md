# LLMs Get Lost in Multi-Turn Conversation

**Authors:** Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville (Microsoft Research, Salesforce Research)

**Paper:** arXiv:2505.06120v1 (May 2025)

**Code/Data:** github.com/Microsoft/lost_in_conversation, datasets/Microsoft/lost_in_conversation

---

## The Problem

LLMs are sold as conversational interfaces, yet they are almost always *evaluated* in a single-turn, fully-specified setting: the user hands over one complete, well-formed instruction and the model answers it in one shot. Real usage looks nothing like this. Analysis of public chat logs shows that **underspecification is pervasive** — users start vague and clarify their needs gradually over multiple turns ("the principle of least effort"). One cited study found users reveal the *entire* instruction up front only 34% of the time.

A growing body of multi-turn benchmarks exists, but the authors argue almost all of them treat conversation as **episodic**: each turn is effectively an independent subtask (refinement, follow-up, expansion) that can be evaluated in isolation. Episodic tasks test multi-turn *context tracking* but never force the model to *fuse* dispersed information to resolve an underspecified request. The authors show episodic framing **overestimates** real multi-turn ability.

The central, unanswered question: **what happens to LLM performance when the exact same task is delivered piecemeal across an underspecified, multi-turn conversation instead of all at once?**

---

## Core Idea: Sharded Instructions and Sharded Simulation

The contribution is a controlled simulation methodology that takes existing single-turn benchmark instructions and **shards** them — splitting one fully-specified instruction into a set of smaller "shards," each carrying a single atomic piece of information. The shards jointly contain exactly the same information as the original; they are just revealed one at a time.

**Example (GSM8K):**

> **Fully-specified (original):** "Jay is making snowballs to prepare for a snowball fight with his sister. He can build 20 snowballs in an hour, but 2 melt every 15 minutes. How long will it take before he has 60 snowballs?"

> **Sharded:**
> - Shard 1: How long before Jay's ready for the snowball fight?  *(high-level intent — always first)*
> - Shard 2: He's preparing for a snowball fight with his sister.
> - Shard 3: He can make 20 snowballs per hour.
> - Shard 4: He's trying to get to 60 total.
> - Shard 5: The problem is that 2 melt every 15 minutes.

A valid sharded instruction must satisfy five properties (Appendix B): **P1 Information Preservation**, **P2 Clear Initial Intent** (shard 1 = the high-level objective), **P3 Order Insensitive** (shards 2..k are decontextualized, revealable in any order), **P4 Maximal Sharding** (maximize the number of single-fact shards), **P5 Minimal Transformation** (stay close to the original wording). Shards are built with a semi-automatic pipeline: Segmentation → Rephrasing → Verification (all LLM-driven via GPT-4o), followed by manual Inspection & Edit by an author. ~100 sharded instructions per task took 1-4 hours of manual work.

### The Sharded Simulation Environment

Three parties interact:
- **Assistant** — the LLM under test. Critically, it is *not* told it is in a multi-turn underspecified conversation and is *not* nudged toward any conversational strategy. The goal is to observe **default** behavior.
- **User simulator** — a low-cost LLM (GPT-4o-mini) that holds the full sharded instruction and decides which shard to reveal next, rephrasing it to fit naturally into the dialogue (without changing its content).
- **System** — classifies each assistant reply into one of seven response strategies (clarification, refusal, hedging, interrogation, discussion, missing, or **answer attempt**, following Herlihy et al.), extracts the answer span when an answer attempt is made, and scores it with the task evaluator.

Each turn reveals **at most one shard**. The conversation ends when (1) the evaluator marks an answer attempt correct, or (2) the simulator runs out of shards. Manual annotation of several hundred conversations found simulation errors in <5% of cases, and errors that *disfavored* the assistant in <2% — so observed degradations are not artifacts of a buggy simulator.

### Five Simulation Types

Built from the same sharded instructions, varying the *pace of information disclosure*:

| Type | Turns | Specification | Role |
|---|---|---|---|
| **FULL** | Single | Fully-specified (original instruction) | Baseline / aptitude ceiling |
| **CONCAT** | Single | Fully-specified (shards concatenated as bullets) | Control — isolates rephrasing loss from multi-turn loss |
| **SHARDED** | Multi | Underspecified, ≤1 shard/turn | **Primary multi-turn condition** |
| **RECAP** | Multi | SHARDED + final turn restating all shards | Mitigation: agent-style one-shot recap |
| **SNOWBALL** | Multi | Each turn restates all prior shards + 1 new | Mitigation: turn-level recap |

CONCAT is the key control: if a model does well on FULL and CONCAT but poorly on SHARDED, the loss is due specifically to underspecification + multi-turn structure, *not* information lost during sharding.

### Metrics: Decomposing Performance into Aptitude vs Unreliability

Each instruction is simulated N=10 times (default temperature T=1.0), producing scores S = {S₁..Sₙ} on a 0-100 scale. Three metrics:

- **Averaged Performance** `P = mean(S)` — unbiased mean score.
- **Aptitude** `A⁹⁰ = percentile₉₀(S)` — best-case (top 10%) performance.
- **Unreliability** `U⁹⁰₁₀ = percentile₉₀(S) − percentile₁₀(S)` — the spread between best- and worst-case runs.

On a box plot: the upper whisker is aptitude; the whisker-to-whisker distance is unreliability. This decomposition is the analytical heart of the paper — a drop in P can come from losing aptitude (the model got dumber) or from rising unreliability (the model's quality became a coin flip).

---

## Experimental Scale

- **600 sharded instructions** (90-120 per task), **6 generation tasks**, **15 LLMs**, **3 main simulation types** (FULL, CONCAT, SHARDED), **N=10** each → **200,000+ simulated conversations**, ~$5,000 total cost.

**Tasks (programming + natural language):**

| Task | Type | Source benchmark(s) | Metric |
|---|---|---|---|
| Code | Programming | HumanEval, LiveCodeBench | Functional accuracy |
| Database (text-to-SQL) | Programming | Spider | Functional accuracy |
| Actions (API calls) | Programming | Berkeley Function Calling Leaderboard | Exact match |
| Math | NL | GSM8K | Exact match |
| Data-to-Text | NL | ToTTo | BLEU |
| Summary | NL (long-context) | Summary of a Haystack | Joint Score (coverage + citation) |

**Models (8 families):** GPT-4o-mini, GPT-4o, o3, GPT-4.1 (OpenAI); Claude 3 Haiku, Claude 3.7 Sonnet (Anthropic); Gemini 2.5 Flash, Gemini 2.5 Pro (Google); Llama3.1-8B-Instruct, Llama3.3-70B-Instruct, Llama 4 Scout (Meta); OLMo-2-13B (AI2); Phi-4 (Microsoft); Deepseek-R1; Command-A (Cohere). Includes two reasoning models (o3, R1).

---

## Headline Results

### Universal degradation: every model, every task

Averaged across the six tasks, **every LLM degrades on every task** from FULL to SHARDED, with an **average performance drop of ~39%**. Models that score 90%+ in the lab-like single-turn setting collapse to ~65% on the *identical* tasks once the instruction is dispersed across turns. Single-turn average ~90% (FULL) drops to ~65% (SHARDED) — a 25-point absolute drop.

| Setting | Avg performance | Interpretation |
|---|---|---|
| FULL | ~90% | Single-turn, fully-specified ceiling |
| CONCAT | ~95.1% of FULL | Rephrasing alone costs little |
| SHARDED | ~65% (≈ -39% vs FULL) | Multi-turn underspecification is the killer |

CONCAT staying at ~95% of FULL is the crucial control: the loss is **not** from information destroyed during sharding. (Smaller models — Llama3.1-8B, OLMo-2-13B, Claude 3 Haiku — show somewhat larger CONCAT drops of 8-14%, indicating weaker robustness to benign paraphrasing.)

Selected per-model FULL → SHARDED degradation (last-column aggregate across 6 tasks):

| Model | Approx. FULL→SHARDED degradation |
|---|---|
| Llama3.1-8B-Instruct | ~32% |
| GPT-4o-mini | ~40% |
| GPT-4o | ~42% |
| o3 | ~35% |
| Claude 3.7 Sonnet | ~35% |
| Deepseek-R1 | ~32% |
| GPT-4.1 | ~46% |
| Gemini 2.5 Pro | ~44% |

The striking pattern: **stronger models get just as lost as weaker ones** (degradations cluster around 30-40%). High single-turn aptitude buys no protection against getting lost.

### Reasoning models don't escape

Extra test-time compute does not help. The two reasoning models (o3, Deepseek-R1) degrade like everyone else. The analysis points to a mechanism: reasoning models produce responses ~33% longer on average, and longer responses pack in more **assumptions** — which then derail the conversation by confusing the model about which requirements came from the user vs. its own earlier turns.

### The core insight: it's unreliability, not aptitude

Decomposing the FULL → SHARDED drop with A⁹⁰ and U⁹⁰₁₀:

| Component | Change (FULL → SHARDED) |
|---|---|
| **Aptitude (A)** | **−16%** (minor) |
| **Unreliability (U)** | **+112%** (more than doubles) |

In single-turn settings, more able models are also more *reliable* (GPT-4.1 and Gemini 2.5 Pro have the lowest unreliability; Llama3.1-8B and OLMo-2-13B the highest). In the SHARDED setting this coupling breaks: **all** models — regardless of aptitude — exhibit very high unreliability, with performance swinging ~50 percentage points on average between the best and worst run of the *same* instruction.

This is the paper's defining statement of the **"lost in conversation"** phenomenon: the big drop in average performance is overwhelmingly driven by an **explosion in unreliability**, not a loss of raw capability. **When an LLM takes a wrong turn early in a conversation, it gets lost and does not recover** — it makes premature assumptions, commits to a flawed solution attempt, and then anchors to that attempt rather than course-correcting as new information arrives.

### Four root causes (Appendix F)

1. **Premature answer attempts** — models propose a full solution before they have enough information, baking in incorrect assumptions about underspecified details.
2. **Over-reliance on previous (incorrect) answers** — answers get progressively "bloated" as the model patches its earlier flawed attempt instead of rethinking.
3. **Loss-of-middle-turns** — models over-weight the first and last turns and under-use middle turns (an in-conversation echo of the "lost in the middle" long-context effect).
4. **Verbosity** — overly long responses introduce more assumptions and distract from what the user actually said.

### Gradual Sharding: even 2 turns is enough

To check whether the maximal (adversarial-feeling) one-shard-per-turn sharding drives the effect, the authors ran a **gradual sharding** experiment: 31 instructions, each expanded into shard-sets of size 1 through 8, holding task complexity fixed and varying *only* granularity (tested on GPT-4o and GPT-4o-mini). Result: models get lost at **two shards and beyond** — the minor aptitude loss + large unreliability spike appears as soon as the conversation spans ≥2 turns. **Providing all information at once (1 shard) is the only setting that preserves reliability.** Granularity beyond that barely matters.

---

## Attempted Mitigations (and why they fall short)

### Agent-style recapitulation (RECAP / SNOWBALL)

Can multi-turn handling just be offloaded to an agent framework that re-states user info? Tested on 4 tasks (Code, Database, Math, Actions) with GPT-4o and GPT-4o-mini:

| Model | SHARDED | RECAP | SNOWBALL |
|---|---|---|---|
| GPT-4o-mini | 50.4 | 86.8 | 66.5 |
| GPT-4o | 59.1 | 93.0 | 76.6 |

Both help, but neither reaches FULL/CONCAT. **RECAP** recovers most but is unrealistic — it recaps on the *final* turn, which isn't known a priori in a live conversation. **SNOWBALL** (realistic turn-by-turn recap) only mitigates the FULL→SHARDED drop by ~15-20%. Conclusion: offloading to an agent wrapper is a partial patch; **LLMs should natively support multi-turn interaction.**

### Lowering temperature

Does T=0 fix the unreliability? The authors swept assistant temperature (AT) and user-simulator temperature (UT) over {1.0, 0.5, 0.0}, measuring unreliability U⁹⁰₁₀:

| Setting | GPT-4o-mini AT=1.0 → 0.0 | GPT-4o AT=1.0 → 0.0 |
|---|---|---|
| FULL | 16.0 → 6.8 | 17.8 → 2.8 |
| CONCAT | 20.2 → 9.5 | 20.2 → 5.8 |
| SHARDED (UT=0.0) | 38.5 → 30.5 | 35.8 → 29.7 |

In single-turn (FULL/CONCAT), dropping temperature cuts unreliability by 50-80%. In SHARDED, it **barely helps** — even with *both* user and assistant at T=0.0, unreliability stays ~30%. A one-token difference early in a multi-turn conversation cascades into divergent trajectories. **Lowering temperature is ineffective for multi-turn reliability.**

### What "fixes" it (negative control): episodic tasks

A 7th task, **document-level Translation** (WMT 2019, reveal 2 sentences/turn), showed **no SHARDED degradation** (BLEU within 10% across all settings). Because translation is *episodic* — decomposable into independent per-sentence subtasks — the model never has to fuse dispersed information. This confirms the effect is specifically tied to **non-decomposable, underspecified** generation, and identifies the three task properties that induce getting lost: (1) generative (not extractive/classification), (2) sufficiently complex to yield many shards, (3) non-decomposable solution (each new shard reshapes the whole answer).

---

## Key Takeaways

1. **The diagnosis is unreliability, not incapability.** Average performance drops ~39% multi-turn, but that's a small (−16%) aptitude loss plus a massive (+112%) unreliability spike. Models *can* still do the task; they just become wildly inconsistent at doing it.

2. **Getting lost is universal and frontier-proof.** All 15 models — small open-weight to SOTA closed-weight, including reasoning models — degrade 30-40%. Aptitude does not buy multi-turn robustness.

3. **Wrong early turns are unrecoverable.** Models make premature assumptions, commit to a flawed answer attempt, and then over-anchor to it. They cannot course-correct as missing requirements trickle in.

4. **Known remedies don't transfer.** Lowering temperature (effective single-turn) and agent-style concatenation (RECAP/SNOWBALL) both fail to close the gap in genuine multi-turn settings.

5. **Practical user advice:** if a conversation goes sideways, *start over* and re-consolidate everything into one message rather than persisting — empirically more effective than continuing to fight a lost conversation.

6. **Sharding is a reusable evaluation primitive.** The semi-automatic sharding pipeline lets any single-turn benchmark be converted into a multi-turn underspecified one — a methodological contribution beyond the specific findings.

---

## Limitations (Acknowledged by Authors)

1. **Idealized, LLM-driven simulation** — the user simulator reveals exactly one clean shard per turn and guarantees the conversation ends with full information. Real human-AI dialogue is messier (terminology confusion, frustration, derailing, infeasible goals). The authors argue this makes their numbers a **lower bound** — real-world degradation is likely worse.
2. **Analytical tasks only** — all tasks have a verifiable analytical solution. Open-ended/creative tasks (where evaluation is itself unsolved) are untested.
3. **Text-only, English-only** — multilingual and multimodal multi-turn behavior is left to future work.

---

## Where it sits (v1/v2)

This is **not a memory-system paper** — it builds no architecture, store, or retrieval mechanism. It is a **diagnostic / motivation paper**, and an important anchor for the rest of this collection.

- **It quantifies the disease the memory systems treat.** The headline result — LLMs lose track of information dispersed across turns, make premature assumptions, and can't recover — is precisely the failure mode that agentic-memory architectures (A-MEM, MemoryOS, Nemori, MAGMA, etc.) exist to mitigate. Where those systems propose *solutions* (structured stores, consolidation, intent-aware retrieval, context management), this paper supplies the rigorous *problem statement*: native multi-turn context handling is unreliable, and the gap is driven by unreliability rather than aptitude. Its finding that naive agent-style recap (RECAP/SNOWBALL) only recovers ~15-20% is, in effect, a challenge to memory systems to do better than trivial re-injection of past turns.

- **Same role as LOCOMO in the collection.** Like LoCoMo, this is a **benchmark/diagnostic** paper rather than a system paper — both define controlled settings that expose where vanilla LLMs fail at conversational state. The two are complementary along the time axis: LoCoMo stresses *long-range* memory across many sessions (does the model remember a fact from session 1 in session 20?), whereas Lost-in-Conversation stresses *within-conversation* information fusion across underspecified turns (can the model integrate requirements that arrive piecemeal *now*?). LoCoMo motivates long-term episodic memory; Lost-in-Conversation motivates robust short-horizon context tracking and consolidation. Together they bracket why agents need explicit memory/context management at both timescales.

- **v1 foundational (2023-2025).** As a May 2025 Microsoft/Salesforce paper, it belongs to the foundational wave that *framed the problem* the 2026-frontier (v2) memory systems are now engineering against. Its concrete "lost in conversation" decomposition — and its call for LLM builders to optimize *reliability* jointly with aptitude — is the empirical justification for treating memory and context management as a first-class concern rather than an afterthought.
