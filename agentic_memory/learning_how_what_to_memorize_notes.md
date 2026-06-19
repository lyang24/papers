# Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory (MemCoE)

**Authors:** Derong Xu, Shuochen Liu, Pengfei Luo, Pengyue Jia, Yingyi Zhang, Yi Wen, Yimin Deng, Wenlin Zhang, Enhong Chen, Xiangyu Zhao\*, Tong Xu\* (University of Science and Technology of China / State Key Lab of Cognitive Intelligence, City University of Hong Kong, Dalian University of Technology, Xi'an Jiaotong University)

**Paper:** arXiv:2605.00702v1 (May 2026), ACL 2026

**GitHub:** https://github.com/Applied-Machine-Learning-Lab/ACL2026_MemCoE

**Method name:** **MemCoE** (Cognition-inspired two-stage optimization for evolving memory)

---

## The Core Problem

LLM agents need long-term, evolving user memory for consistent personalization, but the context window prevents retaining the full dialogue history, and naively storing/retrieving snippets fails to capture *dynamic, shifting* preferences. Existing memory systems fall into two camps, each with a weakness:

1. **Static hand-crafted pipelines** (Mem0, A-Mem, MemoryBank-style decay) convert raw dialogue into memory banks using predefined extraction/update rules. These are brittle under non-stationary user behavior and cannot learn from interaction feedback.

2. **RL-based memory agents** (MemAgent, Memory-R1, Mem-α) treat memory operations as learnable actions and train an end-to-end update policy. But they optimize against **sparse, delayed outcome-level rewards** (final answer correctness). With only simple instructions and a huge free-form action space ("what to write/forget"), the policy is weakly constrained, exploration is hard, and long-horizon optimization is unstable and data-hungry.

The gap: there is no **process-level** signal telling the policy *how* memory should be organized, only *whether* the final answer was right.

---

## The Big Idea: Decouple "How" from "What" via a Cognitive Analogy

MemCoE draws on **memory schema theory** (Alba & Hasher, 1983) and the functional division of labor between two brain systems:

| Brain System | Function | MemCoE Component |
|---|---|---|
| **Prefrontal regions** | Dynamically select/configure an appropriate **schema** (organizing prior, attentional priorities) | **Memory Guideline** (how to organize) |
| **Hippocampus regions** | **Instantiate** the schema by encoding concrete **episodic details** | **Agent Policy** (what to update) |

The biological insight is that maintaining a **stable schema-level organizing prior** (slow-changing) while letting a separate system **flexibly encode context-specific episodic details** within that scaffold is more robust than one monolithic process. MemCoE maps this to a clean separation:

- **How to memorize** = the memory organization pattern (learned once as a textual guideline)
- **What to memorize** = the update content (learned by RL, *constrained* by the guideline)

Crucially, Stage 1 induces a guideline that defines a **stable set of memory operations**, which **shrinks the action space** the Stage-2 policy must explore. This directly fixes the "huge free-form action space + sparse reward" pathology of prior RL-memory work.

---

## Problem Setup

A user interacts with an assistant over time. Dialogue snippets `h_t` accumulate into history `H = {h_1, ..., h_t}`. The system maintains an evolving textual **user memory bank** `M_t`, updated by an evolution operator `T`:

```
M_{t+1} = T(M_t, h_t; S, φ)
```

where `S` is a **learnable memory-update prompt** (the natural-language parameter that regulates *how* memory updates), and `φ` are the LLM parameters. The agent answers a query with `y_t = A(x, M_t)`. The challenge is designing `T` so `M_t` evolves coherently with `H`.

---

## Architecture: Two-Stage Optimization

### Stage 1 — Memory Guideline Induction (MGI): learning *how* (♣)

Treats the instruction prompt `S` as a **global natural-language parameter** and optimizes it from data (prompt-as-parameter, in the TextGrad / OPRO / Reflexion lineage). Goal: induce an optimized, domain-agnostic guideline `S*` that teaches the agent the correct memory-evolution procedure.

**1. Contrastive feedback as textual gradient.** At step `k`, with current guideline `S^(k)`, run the memory-evolution operator over a history `H` and do multiple forward passes to answer query `x`, producing trajectories `{τ_i}` (each containing query, intermediate memory states, candidate response). Using task supervision, select at least one **correct** trajectory `τ⁺` and treat suboptimal ones as **contrastive negatives** `{τ_j⁻}`. A feedback instruction `P_g` compares `τ⁺` against `{τ_j⁻}`, highlighting desired properties and typical errors. The resulting natural-language reflection is a **textual gradient**:

```
g^(k) = Grad(τ⁺, {τ_j⁻}; P_g)
```

**2. Batch-level gradient aggregation.** To get a stable, general signal, aggregate textual gradients across a mini-batch `B`, synthesizing instance-level critiques into one abstract update direction (identifying common failure patterns):

```
G^(k) = Aggr({g^(k)}_(H,x); P_a)
```

**3. Optimization step.** Apply the merged gradient via natural-language editing:

```
S^(k+1) = Optim(S^(k), G^(k); P_o)
```

Conceptually this performs gradient-like steps on a contrastive objective; the induced `S*` approximately maximizes expected reward `R` (correctness) over `(τ⁺, {τ_j⁻})`. The output is a transferable, schema-consistent guideline encoding effective memory-operation principles.

### Stage 2 — Guideline-Aligned Memory Policy Optimization (GMPO): learning *what* (♠)

**Fix** `S*`. Treat the parameters `φ` of the evolution operator `T` and agent `A` as a **unified policy** over memory-augmented trajectories. For each `(H, x)`, roll out under `S*` to produce a trajectory interleaving memory updates `M_{t+1} = T(M_t, h_t; S*, φ)` and intermediate responses, ending in a final answer.

**Dual reward signal:**

- **Guideline-aware reward** `R_S(τ; S*) ∈ [0,1]` — a *dense, process-level* signal. For each memory-update segment, an LLM scores whether the update strictly follows the prescribed output format (required fields, tags, structure). This encourages **guideline-aligned, well-structured** edits rather than arbitrary free-form text. **This is the key novelty vs. prior RL-memory** — it provides supervision at every update step, not just the end.
- **Answer reward** `R_ans(τ) ∈ {0,1}` — task correctness (exact / judged match of final response vs. reference).

Combined:
```
R(τ) = (1 − λ) · R_S(τ; S*) + λ · R_ans(τ)
```
where `λ` balances guideline fidelity vs. answer accuracy.

**Policy optimization with multi-turn GRPO.** Optimize `φ` using **Group Relative Policy Optimization** over groups of trajectories on multi-conversation memory evolution. GRPO samples a group, computes group-normalized advantages from `R(τ)`, and applies a clipped policy-gradient update. In the multi-conversation setting each trajectory is decomposed into `n_i` conversations, and the group-relative advantage `Â_i = (R_i − mean) / std` is assigned to **all token-level actions** (both memory-update tokens and answer tokens) across conversations, with a KL penalty to a frozen reference policy. (Follows MemAgent's multi-conv RL formulation.)

```
φ* = arg max_φ  E_{(H,x)~D, τ~π_φ(·|H,x;S*)} [ R(τ) ]
```

The result: a memory-evolution policy that **follows the induced guideline** while **selectively storing** the information most beneficial for downstream interaction.

---

## Experimental Results

**Backbone:** Qwen2.5-7B-Instruct for all methods (Mem-α uses Qwen3-4B). Retriever: all-MiniLM-L6-v2, Top-10. Training: 300 examples sampled from PersonaMem. Inference feeds full dialogue history; each evolve round inputs a 4K-token chunk. Hardware: 4× A6000.

**Benchmarks:** PersonaMem (in-domain, preference evolution over long multi-session histories; 32K/128K context), PrefEval (out-of-domain, explicit vs. implicit preference multi-choice, 1,000 each), PersonaBench (out-of-domain, noisy heterogeneous user corpora; F1). Metric is accuracy except PersonaBench (F1).

### Overall comparison (Table 1)

| Method | PersonaMem 32K | PersonaMem 128K | PrefEval Explicit | PrefEval Implicit | PersonaBench (no noise) | PB 0.3 | PB 0.5 | PB 0.7 | **Overall** |
|---|---|---|---|---|---|---|---|---|---|
| Long Context | 34.36 | 25.05 | 31.70 | 30.80 | 29.00 | 19.10 | 17.83 | 13.00 | 26.90 |
| RAG | 48.67 | 38.90 | 47.80 | 32.40 | 29.09 | 28.16 | 24.31 | 23.00 | 36.68 |
| Mem0 | 48.53 | 39.67 | 57.60 | 46.40 | 17.60 | 19.75 | 19.22 | 17.80 | 38.23 |
| A-Mem | 48.26 | 38.22 | 62.30 | 52.80 | 30.32 | 28.56 | 25.19 | 24.45 | 42.64 |
| LightMem | 50.72 | 39.93 | 64.20 | 54.80 | 19.08 | 18.74 | 19.65 | 17.80 | 41.21 |
| MemAgent (RL) | 53.58 | 43.59 | 72.30 | 63.60 | 20.05 | 19.36 | 16.51 | 17.92 | 45.00 |
| Mem-α (RL) | 53.37 | 42.86 | 71.90 | 62.50 | 19.92 | 17.02 | 16.43 | 15.59 | 44.19 |
| **MemCoE (Ours)** | **57.06** | **47.24** | **81.30** | **69.90** | **32.27** | **29.89** | **25.99** | **25.09** | **52.02** |

Findings:
- MemCoE wins **all 8 settings** and the overall score (**52.02** vs. best baseline MemAgent 45.00, a ~7-point / **+15.6%** relative gain).
- Largest absolute jumps over the best RL baseline on **PrefEval Explicit** (81.30 vs. 72.30, +9.0) and **PersonaBench no-noise F1** (32.27 vs. ~20).
- RL baselines (MemAgent, Mem-α) are competitive but lag — consistent with the thesis that **sparse outcome rewards alone are insufficient**; the guideline + process reward are what close the gap.
- Long Context degrades sharply under noisy histories (PersonaBench 0.7 = 13.00), while MemCoE stays robust (25.09) by filtering irrelevant content during evolution.

### Cross-LLM transferability of MGI guidelines, no RL (Table 3)

The induced guideline is **portable** across backbones — optimize once, deploy anywhere.

| Method | Qwen2.5-7B-Instruct | gpt-4o-mini | gemini-2.5-flash | GPT-5 |
|---|---|---|---|---|
| RAG | 48.67 | 47.44 | 61.15 | 63.80 |
| A-Mem | 48.26 | 48.47 | 62.37 | 64.42 |
| MemCoE (guideline opt. w/ Qwen2.5-7B) | 53.37 | 52.56 | 64.62 | **66.67** |
| MemCoE (guideline opt. w/ gpt-4o-mini) | 52.56 | 54.19 | **67.28** | 64.83 |

Both MGI variants beat the baselines across **all four** LLMs (including GPT-5 and gemini-2.5-flash), showing the guideline captures **model-agnostic** memory-update principles rather than overfitting one LLM. Optimizing with gpt-4o-mini generalizes best across backbones.

### Effect of guideline quality (Figure 6)

Guideline quality directly drives downstream accuracy. Manual prompt → LLM rewrite → MGI:

| Guideline | PersonaMem 32K | PersonaMem 128K |
|---|---|---|
| Manual Prompt | 48.25 | 39.30 |
| LLM Rewrite | 50.46 | 41.50 |
| **MGI (Ours)** | **53.28** (+10.4%) | **43.76** (+11.3%) |

MGI gives +10.4% / +11.3% relative over the manual prompt, stable across 3 seeds.

### Other analyses
- **Efficiency (Fig. 3):** MemCoE sits on a favorable accuracy-vs-time frontier — best performance while among the faster methods. It *internalizes* extraction/update/forgetting into one evolution pass, avoiding the repeated extract-then-merge LLM calls of A-Mem/Mem0. MemAgent/Mem-α are fast but underperform.
- **Top-K retrieval (Fig. 4):** "Ours (RAG)" peaks around K=20 (~55.83) and can even beat full-history; vanilla RAG *degrades* as K grows (more distractors), dropping below empty memory. Retrieval helps only when paired with MemCoE to transform evidence into coherent memory.
- **Per-round token budget (Fig. 5):** A clear trade-off — too small (1K–2K) forces many rounds and compounds errors / uncontrolled forgetting; too large (8K–32K) makes each evolution step too complex. A **moderate budget (≈4K)** is best.

---

## Ablation Study (Table 2)

| Setting | PersonaMem 32K | PersonaMem 128K | PrefEval Explicit | PrefEval Implicit |
|---|---|---|---|---|
| **MemCoE (full)** | **57.06** | **47.24** | **81.30** | **69.90** |
| w/o CF (contrastive feedback) | 56.44 | 46.33 | 78.30 | 68.10 |
| w/o GR (guideline reward) | 56.24 | 46.06 | 79.50 | 68.30 |
| w/o MGI (Stage 1) | 54.81 | 44.50 | 73.20 | 63.60 |
| w/o GMPO (Stage 2) | 53.37 | 43.97 | 77.40 | 66.20 |
| w/o ALL | 48.47 | 39.09 | 71.70 | 60.60 |

Findings:
1. Removing the **component-level** signals (CF or GR) causes consistent but *smaller* drops — both contrastive textual feedback and the guideline-aligned reward improve update reliability.
2. Removing an **entire stage** hurts more: dropping **MGI** most damages preference retention (PrefEval Explicit/Implicit 81.30/69.90 → 73.20/63.60); dropping **GMPO** most damages long-horizon tracking on PersonaMem (57.06/47.24 → 53.37/43.97). The two stages are complementary — guideline governs PrefEval-style preference following, RL policy governs PersonaMem-style long-history tracking.
3. **w/o ALL** collapses everything (PersonaMem 32K → 48.47), confirming the learned guideline **and** guideline-aligned policy optimization are both critical.

---

## Key Takeaways

1. **Decoupling "how" from "what" is the core contribution.** Separating a slow, schema-level **organizing guideline** (Stage 1) from a fast, episodic **content policy** (Stage 2) mirrors the prefrontal→hippocampus division and stabilizes long-horizon memory learning that pure end-to-end RL struggles with.

2. **A learned guideline shrinks the RL action space.** Prior RL-memory methods drown in a huge free-form edit space under sparse rewards. By first inducing a fixed set of memory operations, MemCoE constrains exploration so Stage-2 RL can focus on *content selection* under dense process rewards.

3. **Process-level reward > outcome-only reward.** The guideline-aware reward `R_S` scores every update segment for format/structure adherence, giving dense supervision that fixes the unstable-training problem of outcome-only RL-memory.

4. **The "how" half is model-agnostic and reusable.** MGI guidelines transfer across Qwen, gpt-4o-mini, gemini-2.5-flash, and GPT-5 — optimize once, deploy on any backbone.

5. **Prompt-as-parameter meets RL.** MemCoE bridges textual-gradient prompt optimization (TextGrad/OPRO/Reflexion lineage) with GRPO policy learning, using the optimized prompt to *define* the RL reward — a tighter coupling than either technique alone.

---

## Limitations (Acknowledged by Authors)

1. **Scorer dependence.** Stage-2 process rewards come from an **LLM-based scorer**; performance is sensitive to that scorer's reliability.
2. **Budget/round tuning.** Requires careful tuning of per-round token budget and number of evolution rounds; splitting long histories into many rounds lets small update errors **compound** into unintended forgetting or over-generalized entries.
3. **Single-objective policy.** Memory evolution is treated as a single-objective policy under a fixed guideline; explicitly balancing competing objectives (stability vs. plasticity, informativeness vs. brevity) is left as non-trivial future work.

---

## Where it sits (v1/v2)

MemCoE is a clear **v2 (2026 frontier)** paper, and a textbook example of the survey's headline shift: **hand-crafted memory rules → learnable / RL-driven memory**. It directly attacks the v1 design philosophy on both axes — *how* and *what* to memorize.

**Contrast with v1 hand-crafted write policies** — every v1 system bakes its update logic into fixed heuristics:
- **A-Mem:** fixed Zettelkasten **link-generation + memory-evolution rules** (new note triggers neighbor updates via a static prompt template).
- **MemoryOS:** **heat-based thresholds** govern promotion/eviction between short/mid/long-term tiers (hand-tuned scoring + capacity rules).
- **MemoryBank:** **Ebbinghaus-style decay** — a predefined forgetting curve deterministically ages memory strength.
- **Mem0 / LightMem:** predefined extract-then-merge pipelines.

In all of these, *how* memory is organized is a frozen designer choice and *what* is stored follows fixed rules. MemCoE replaces both: the **"how" is learned** as a textual guideline via contrastive textual-gradient optimization (MGI), and the **"what" is learned** via guideline-aligned GRPO (GMPO). In the paper's own Table 1, A-Mem (42.64 overall) and LightMem (41.21) are exactly the hand-crafted baselines MemCoE (52.02) surpasses.

**Connection to LatentMem (also in this collection):** LatentMem likewise abandons hand-crafted write rules and **learns memory via RL** — both are v2 "learned-memory" papers. The instructive difference is *where* the learning happens: LatentMem learns memory in a **latent/parametric** space, whereas MemCoE keeps memory as **human-readable text** and learns the **policy + guideline** that govern it. MemCoE's distinct twist over generic RL-memory (LatentMem, MemAgent, Memory-R1, Mem-α) is the **two-stage decoupling with a dense guideline-aware process reward**, its answer to the "sparse-outcome-reward → unstable RL" problem that the agentic-RL-memory literature repeatedly hits.

**Author lineage:** This is part of the **Derong Xu / Xiangyu Zhao / Tong Xu (USTC / CityU / Huawei)** line of agentic-memory work that recurs across this collection. The same group authored the multi-granularity long-term memory association/selection paper (Xu et al., ICLR 2026), "Personalize before Retrieve" query expansion (Zhang et al., AAAI 2026), recollection-familiarity adaptive retrieval (Zhang et al., ICLR 2026), and task-oriented adversarial memory adaptation (Deng et al., 2026). MemCoE is the group's RL-meets-cognitive-schema entry in that line — and the acknowledgements (Huawei Innovation Research Program, Tencent, Bytedance, Kuaishou, Didi) reflect the same industrial-academic cluster.
