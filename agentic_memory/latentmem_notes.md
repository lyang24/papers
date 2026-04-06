# LatentMem: Customizing Latent Memory for Multi-Agent Systems

**Authors:** Muxin Fu, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, Yang Yang (Tongji University, Shanghai AI Lab, CUHK, Nanjing University, Shanghai Jiao Tong University)

**Paper:** arXiv:2602.03036v2 (Mar 2026)

**GitHub:** https://github.com/KANABOON1/LatentMem

**Model:** https://huggingface.co/Kana-s/LatentMem-Qwen3-4B

---

## The Core Problem

Multi-agent systems (MAS) use memory to accumulate experience and coordinate across tasks, but existing memory designs suffer from two fundamental bottlenecks:

1. **Memory homogenization** — Most systems adopt a one-size-fits-all memory strategy, ignoring the functional heterogeneity of agents. A "Code Agent" and a "Test Agent" receive the same memory representation, even though they need very different information from past trajectories. This undermines role adherence and amplifies correlated errors.

2. **Information overload** — MAS inherently involves long interaction contexts, and multi-granularity memory designs (trajectories, insights, skills) amplify this burden by introducing large volumes of discrete textual entries. This overwhelms agents and obscures critical decision signals.

The central question: *Can we design a learnable memory that is both role-aware and token-efficient, without extensive manual engineering?*

---

## The Big Idea: Learnable, Role-Aware Latent Memory

Instead of storing memory as explicit text tokens (summaries, insights, code snippets) that consume context window space, LatentMem **compresses past trajectories into compact latent embeddings** that are customized per-agent based on their role profiles. These latent memories are injected directly into the model's hidden state space — not as text tokens, but as additional latent vectors prepended to the hidden states.

The key paradigm shift (compared to existing MAS memory):

| Existing MAS Memory | LatentMem |
|---|---|
| Handcrafted memory patterns (summaries, insights, skills) | Learned latent representations |
| Token space — consumes context window | Latent space — fixed-length embeddings |
| Same memory for all agents | Role-aware: conditioned on agent profiles |
| Manual engineering of memory structure | End-to-end optimized via RL (LMPO) |

---

## Architecture

LatentMem consists of two core components:

### 1. Experience Bank (B)

A lightweight storage of **raw MAS trajectories** — no summarization, no insight extraction, no human priors. Following the "bitter lesson" (Sutton, 2019), scalable systems should rely on general learning mechanisms rather than hand-crafted knowledge.

Each trajectory τ records, at each step: the active agent index α_j, its input prompt p_j, and its output o_j.

**Retrieval:** Similarity-based retrieval using MiniLM embeddings:
```
T_q = top-K over B (sim(v(q), v(τ_i)))
```

**Update:** After a task completes, the new trajectory is simply appended:
```
B ← B ∪ {τ_new}
```

No trajectory condensation or insight extraction — the memory composer handles all compression into latent space.

### 2. Memory Composer (C)

A learned deep network σ_φ that transforms raw trajectories into compact, role-aware latent memories. At each reasoning step j, it takes:
- The retrieved trajectories T_q
- The active agent's role profile γ_{a_j}

And produces a fixed-length latent memory matrix:
```
m_j = σ_φ(γ_{a_j}, T_q) ∈ R^{L' × D}
```
where L' = 8 (fixed latent sequence length) and D is the hidden dimension of the backbone LLM.

**Memory injection:** The latent memory m_j is **concatenated to the agent's hidden states** h_j (not to the text tokens):
```
π̃(p_j, m_j) = π(concat(h_j, m_j))
```
This is transparent to the MAS architecture — no modifications to the agent framework are needed. The latent memories act as additional conditioning vectors in the model's internal representation space.

**Implementation:** The memory composer is a lightweight transformer initialized from the backbone LLM and trained using LoRA. It maps variable-length trajectory + role tokens into a fixed-length latent output.

---

## Latent Memory Policy Optimization (LMPO)

The key innovation: LatentMem is **end-to-end trainable via reinforcement learning** through the latent memory. Since the memory composer's output m_j is a differentiable function of φ, and the agent's policy is conditioned on m_j, task-level rewards can be backpropagated through the latent memories to optimize the composer.

### Gradient Flow

The generation of a trajectory τ is factorized sequentially:
```
P(τ | q, T_q; φ, {θ_k}) = Π_j P(o_j | p_j, m_j; θ_{a_j})
```
where m_j = σ_φ(T_q, γ_{a_j}). Since π̃ is conditioned on m_j, the gradient of any task-level objective can backpropagate through the agent's forward pass to refine φ.

Crucially, the **agent backbone parameters {θ_k} are frozen** — only the memory composer φ is trained. This is far cheaper than fine-tuning the backbone LLM.

### Policy Optimization

LMPO is a variant of GRPO (Group Relative Policy Optimization). Given a query q and retrieved trajectories T_q:

1. Sample G trajectories: {τ̂_i} ~ P(· | q, T_q; φ, {θ_k})
2. Evaluate each with reward R(τ̂_i) and compute group-based advantage:
   ```
   Â_i = (R(τ̂_i) - mean) / (std + ε)
   ```
3. **Token-level surrogate objective** (not trajectory-level): Standard RL for MAS uses trajectory-level objectives, but this causes tokens in longer interactions to contribute disproportionately less to the gradient. LMPO instead uses a token-level objective:
   ```
   J_LMPO(φ) = (1/|total_tokens|) Σ_{i,j,t} L_{i,j,t}(φ)
   ```
   where each token's loss is weighted by the importance sampling ratio r_{i,j,t}(φ) measuring how the updated memory modulates the agent's policy at that specific token.

This token-level formulation ensures the memory composer captures critical coordination patterns even in long multi-agent interactions.

---

## Experimental Results

### Main Results (Qwen3-4B-Instruct backbone)

Evaluated across **6 benchmarks** (TriviaQA, KodCode, StrategyQA, PopQA — in-domain; BigCodeBench, PDDL — out-of-domain) and **4 MAS frameworks** (AutoGen, MacNet, CAMEL, DyLAN).

Key numbers (average across all benchmarks):

| MAS Framework | No-memory | Best Token-level Baseline | **LatentMem** |
|---|---|---|---|
| AutoGen | 53.61 | 56.09 (G-Memory) | **62.75** |
| MacNet | 50.73 | 57.21 (G-Memory) | **60.02** |
| CAMEL (held-out) | 53.22 | 55.27 (G-Memory) | **61.12** |
| DyLAN (held-out) | 51.51 | 56.15 (G-Memory) | **60.72** |

Highlights:
- **Up to 16.20% improvement** over vanilla (no-memory) settings on TriviaQA with AutoGen
- **Up to 18.45% improvement** on code generation tasks (KodCode)
- Outperforms single-agent memory baselines (Voyager, Generative) by **7.86% on average**
- Outperforms multi-agent memory baselines (MetaGPT, ChatDev, OAgents, G-Memory) by **6.66% on average**

### Generalization

- **Out-of-domain benchmarks** (BigCodeBench, PDDL): LatentMem improves AutoGen on PDDL by **7.10%**, while MetaGPT and Voyager drop by up to 4.44% and 2.77%
- **Unseen MAS frameworks** (CAMEL, DyLAN — never seen during training): LatentMem boosts CAMEL on KodCode by **7.05%**, whereas nearly all baselines decline
- This demonstrates the role-aware latent representation generalizes across domains, agent roles, and collaboration patterns

### Comparison with Multi-Agent Fine-Tuning (MARTI)

Under the same computational budget (same backbone, same training data, same GRPO algorithm):

| MAS | Method | KodCode | TriviaQA |
|---|---|---|---|
| AutoGen | MARTI (backbone fine-tuning) | 74.20 | 64.78 |
| AutoGen | **LatentMem** | **76.80** | **76.51** |
| MacNet | MARTI | 73.10 | 62.31 |
| MacNet | **LatentMem** | **78.90** | **65.98** |

LatentMem consistently outperforms direct backbone fine-tuning, suggesting it better exploits the structural advantages of complex MAS by optimizing the memory interface rather than the agent weights.

### Cost Analysis

LatentMem achieves the **best performance-to-cost ratio** among memory-based approaches:
- Uses **50% fewer tokens** than text-based memory baselines (latent memories don't consume context window)
- Reduces inference time to **~2/3** compared to mainstream memory designs
- On DyLAN + TriviaQA: achieves +11.68% over No-memory with substantially lower time overhead, and cuts inference time by 2.16× relative to OAgents

### Scaling with Task Horizon

As more experiences accumulate (tracked via cumulative accuracy over question indices):
- LatentMem **steadily improves** and surpasses all baselines, including multi-granularity systems like G-Memory
- While G-Memory degrades when K > 3 retrieved trajectories (information overload), LatentMem **consistently improves** — its latent compression effectively distills useful information from redundant trajectories

---

## Framework Analysis

### Role-Aware Memory Visualization

t-SNE visualizations of latent memories show **clear role-specific clustering**:
- On AutoGen + KodCode (in-domain): user-proxy and assistant agent memories form well-separated clusters
- On CAMEL + BigCodeBench (out-of-domain, unseen MAS): critic, user-proxy, summarizer, and actor agent memories remain well separated

This confirms LatentMem avoids memory homogenization even on entirely novel task domains and collaboration patterns.

### Ablation Study

| Variant | KodCode (AutoGen) | PDDL (AutoGen) | KodCode (MacNet) | PDDL (MacNet) |
|---|---|---|---|---|
| w/o role profiles | -2.30% | -3.60% | -6.45% | -7.63% |
| w/o experience bank | -3.60% | -7.63% | -1.77% | -3.48% |
| **Full LatentMem** | baseline | baseline | baseline | baseline |

- **Role profiles** matter more for complex MAS (MacNet with 5 agents: -6.45%) than simple MAS (AutoGen with 2 agents: -2.30%)
- **Experience bank** (real-time trajectory updates) matters more for complex, out-of-distribution tasks (PDDL: -7.63%)

### Sensitivity: Latent Memory Length L'

Performance improves with larger L' but with diminishing returns. L' = 8 balances accuracy and computational cost. Even L' = 2 already provides meaningful improvement over no-memory baselines.

---

## Case Study

On a PDDL ball-sorting task with MacNet (5 agents):

- **Vanilla MacNet:** Suffers from step repetition — agents repeat the same pick-move-drop cycle
- **MacNet + OAgents (text memory):** Retrieved trajectories mislead agents, causing them to disobey task specifications (moving balls to wrong rooms)
- **MacNet + LatentMem:** Agents generate correct initial actions; when an invalid action occurs, the agent self-corrects by validating and refining its action, then produces a correct follow-up. The role-aware latent memory reinforces role compliance and enables coordination

---

## Key Takeaways

1. **Latent memory is a viable alternative to text-based memory for MAS.** By compressing trajectories into fixed-length latent embeddings rather than discrete text entries, LatentMem avoids information overload while preserving (and improving) the utility of past experience.

2. **Role-aware conditioning is essential for multi-agent memory.** Different agents need different information from the same trajectory. Conditioning the memory composer on agent role profiles produces naturally differentiated memories that reinforce role compliance.

3. **End-to-end RL optimization of memory (LMPO) works.** By keeping agent backbones frozen and only training the memory composer, LatentMem achieves better results than both hand-crafted memory designs and direct backbone fine-tuning — at lower cost.

4. **Token-level (not trajectory-level) objectives matter for MAS RL.** The token-level surrogate in LMPO ensures that critical coordination patterns in long multi-agent interactions contribute meaningfully to the gradient, unlike trajectory-level objectives where they get diluted.

5. **The "bitter lesson" applies to agent memory.** Storing raw trajectories and learning to compress them (rather than hand-engineering summaries, insights, or skill schemas) leads to better generalization across domains, agent roles, and MAS frameworks.

6. **Practical efficiency:** 50% fewer tokens, ~2/3 inference time, no modifications to existing MAS architectures — LatentMem is a drop-in memory module.

---

## Limitations and Future Directions

- Currently trained on Qwen3-4B and Llama-3.1-8B; scaling to larger backbones and more diverse MAS topologies remains to be explored
- The experience bank uses simple similarity retrieval — more sophisticated retrieval strategies could further improve performance
- The latent memories are opaque (not human-interpretable), which may limit debugging and trust in high-stakes applications
- Cross-MAS transfer (training on AutoGen/MacNet, deploying on CAMEL/DyLAN) works well, but truly open-ended continual adaptation across arbitrarily different MAS frameworks is an open question
