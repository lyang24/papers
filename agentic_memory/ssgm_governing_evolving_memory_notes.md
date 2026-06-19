# SSGM: Governing Evolving Memory in LLM Agents — Risks, Mechanisms, and the Stability- and Safety-Governed Memory Framework

**Authors:** Chingkwun Lam, Jiaxin Li, Lingfei Zhang, Kuo Zhao (corresponding) — College of Intelligent Science and Engineering, Jinan University

**Paper:** arXiv:2603.11768v2 (19 May 2026)

> Type note: This is a **position/survey + conceptual-framework** paper, not a system with experiments. It contributes (1) a taxonomy of evolving-memory systems, (2) a four-dimensional failure taxonomy, (3) the **SSGM** governance architecture with a formal read/write/reconcile lifecycle and a bounded-drift theorem, and (4) testable research hypotheses + evaluation protocols. No empirical numbers are reported (see "Experiments" below).

---

## The Core Problem

Agent memory is shifting from **static retrieval databases** (RAG, fixed summarization schedules, relevance-and-recency heuristics) to **dynamic, self-refining memory** where the agent autonomously decides when to `ADD`, `UPDATE`, `DELETE`, or `RETRIEVE` (e.g., Memory-R1's RL policy, Mem0/AtomMem atomic consolidation, A-MEM's Zettelkasten link evolution). Memory becomes a **mutable asset that evolves alongside the agent** rather than an immutable log.

Granting agents the autonomy to rewrite their own memory imports the **stability–plasticity dilemma** into artificial systems. The key danger that distinguishes evolving memory from static RAG:

> In static RAG, an error is **isolated to a single retrieval step**. In evolving memory, errors are **cumulative and persistent** — they feed back into storage and compound over time.

The paper frames this as a **compounding failure loop across three lifecycle interfaces** (Figure 1):

1. **Input ingestion → Memory Poisoning** (malicious/hallucinated content written as truth)
2. **Consolidation → Semantic Drift** (knowledge degrades through iterative summarization)
3. **Retrieval → Conflict / Hallucination** (stale or fabricated facts retrieved and acted on)

The critique of the field: recent surveys obsess over **retrieval accuracy/efficiency** and treat *how* memory updates as well-studied, but the **protocols for long-term correctness and safety of those updates remain underexplored**. SSGM's thesis: **memory evolution must be decoupled from memory governance.**

---

## Part 1 — What Evolves: Content, Structure, Policy

The paper first taxonomizes *what* changes in evolving memory along three facets (Table 1 in the paper surveys ~30 systems across these):

| Facet | What evolves | Representative systems |
|---|---|---|
| **Content** (what is stored) | Add/update/summarize/delete units; abstraction; procedural + multimodal knowledge | Generative Agents (summarization), VideoARM/WorldMM/TeleMem (multimodal state), LEGOMem/AWM (procedural workflow rules) |
| **Structure** (how organized) | From flat lists → self-organizing graphs / KGs / hierarchies | A-MEM (Zettelkasten), HippoRAG (spreading activation), MemoRAG (global graph), HiMem (hierarchical reconsolidation), ChatDB (SQL schema) |
| **Policy** (how managed) | From fixed heuristics → learned/RL or evolutionary policies | Memory-R1 (PPO), MemAgent (GRPO), AtomMem (atomic ops), DarwinMem (training-free "survival of the fittest") |

## Part 2 — How Memory Evolves: Mechanisms

Three mechanism families effect the changes:

1. **Reflection / self-supervised updates** — agent critiques its own outcomes and stores lessons (Reflexion: "if the solution was wrong, explain the error"; Self-Refine; MemR3 reflects *during* retrieval).
2. **Reinforcement learning / outcome-driven optimization** — memory management framed as a **POMDP**. At time *t* the agent has observation *o_t*, memory state *M_t*, latent state *s_t*; policy `π_θ(a_t | o_t, M_t)` chooses from a memory-action subset:
   `A_mem = {ADD(c), UPDATE(idx, v), DELETE(idx), RETRIEVE(q)}`
   optimized to maximize discounted return `J(θ) = E[Σ γ^t r(s_t, a_t)]`. Memory-R1 uses PPO; MemAgent uses GRPO for sparse long-horizon rewards.
3. **Consolidation & forgetting** — biologically inspired. Unmanaged, `|M|` grows linearly → latency. Two governance ideas surfaced here that SSGM reuses:
   - **Budgeted / "forgetting-by-design"** (Alqithami's Forgetful-but-Faithful, FiFA benchmark): bounded Priority-Decay forgetting *preserves* coherence and privacy without losing functionality.
   - **Temporal decay (Weibull)** (Huang et al. / LiCoMemory): relevance `w(Δτ) = exp(−(Δτ/η)^κ)`, where Δτ = time since last successful retrieval, η = scale, κ = shape (curvature of forgetting). More expressive than plain exponential decay. SSGM prunes/archives items below a freshness threshold θ_fresh.

---

## Part 3 — Why Memory Fails: The Four-Dimensional Failure Taxonomy (the RISKS)

This is the paper's risk catalog (Table 2). Four categories, each with failure modes, mechanism, and the SSGM mitigation:

| Category | Failure mode | Mechanism / manifestation | SSGM mitigation |
|---|---|---|---|
| **Stability** | **Semantic Drift** | Iterative/lossy summarization gradually strips nuance → distorts ground truth | Ground-Truth Anchoring (reconciliation R) |
| | **Procedural Drift** | Reinforcement of suboptimal/outdated workflows (e.g., rigidifying a convoluted API workaround) | Rule Verification |
| | **Goal / Role Drift** | Alignment shift from accumulated interaction bias (esp. long-term role-play) | Role Partitioning |
| **Validity** | **Memory Hallucination** | Fabricated/non-existent facts stored as truth | Consistency Verifier (Truth Maintenance System) |
| | **Temporal Obsolescence** | Stored fact is *correct but stale*; conflicts with new state (distinct from hallucination) | Weibull decay function |
| **Efficiency** | **Retrieval Latency** | Search scales linearly/quadratically with history | Hierarchical indexing |
| | **Index Bloat** | Accumulation of redundant/noisy episodic logs | Active forgetting / pruning |
| **Safety** | **Memory Poisoning** | Injection of malicious instructions into storage (indirect prompt injection) | Write Filtering (firewall) |
| | **Privacy Leakage** | Unauthorized cross-session / cross-user retrieval; "topology-induced" leakage | Provenance + ACLs |

**Drift, formalized.** Semantic drift at horizon T is the embedding divergence between current memory and a ground-truth ledger:

`δ(M_T, K_true) = 1 − sim(E(M_T), E(K_true))`

where `E(·)` is a fixed embedding model and sim is cosine similarity. `K_true` is approximated in practice by an immutable ledger of raw observations (which preserves detail lost to summarization). Illustrative drift example (Figure 2): "I like mild spicy food" → rewritten to "likes spicy food" → "loves very spicy food" → preference *violation* (system recommends ghost-pepper wings). This is **preference-intensity drift under repeated lossy summarization**.

**Safety specifics worth noting:**
- **Topology-induced leakage** (Liu et al., "Topology Matters"): the *network structure* of multi-agent systems intrinsically governs leakage — **fully connected memory graphs maximize vulnerability** (a "Leakage Plateau" is the measurable signal).
- **Poisoning** (Greshake et al.): indirect prompt injection during ingestion lets adversarial instructions solidify into long-term storage.

---

## Part 4 — The SSGM Framework

**Central principle:** structurally **decouple the agent's generative cognitive policy from the memory substrate** via an active, intercepting **Governance Middleware**. In unrestricted systems the agent is both sole generator and sole validator of its knowledge base — that unchecked autonomy is the root cause of drift, forgetting, and poisoning. SSGM is a **conceptual/theoretical architecture + design principles**, not a software release.

### Three-layer architecture (Figure 4)

- **Cognition Layer** — LLM cognitive policy (reasoning + tool use); proposes memory deltas but never writes directly.
- **Governance Layer (Middleware)** — two gates intercept all memory traffic:
  - **Write Validation Gate** (upper / consolidation path) — consistency check before any write.
  - **Read Filtering Gate** (lower / retrieval path) — provenance + temporal relevance + access control before any read reaches the context window.
- **Memory Layer — Dual Memory Substrate:**
  - **Mutable Active Graph** — fast, editable, for semantic reasoning.
  - **Immutable Episodic Log** (append-only `K_ledger`) — the operational source of truth / evidence trace, enabling replay and rollback.

### Four design principles

| # | Principle | Mechanism | Biological analogy |
|---|---|---|---|
| **P1** | **Pre-Consolidation Validation** | Write Validation Gate = a **Truth Maintenance System (TMS)**. Retrieve core facts `M_core`; if `ΔM ∧ M_core ⊨ ⊥` (logical contradiction via NLI check), **reject** the update. Prevents hallucination cascades from corrupting the graph. | Hippocampus filters/reality-monitors before committing to neocortex |
| **P2** | **Temporal & Provenance Grounding** | Read Filtering Gate evaluates candidates on two axes: (a) **cryptographic provenance** `σ(µ)` (was this written by a trusted source vs. an injected prompt?), and (b) **Weibull cognitive decay** — prune items below a dynamic freshness threshold. | Forgetting / freshness |
| **P3** | **Access-Scoped Retrieval** | Retrieval must **not** rely on semantic similarity alone. Inject identity constraints (**ABAC / ACL**) into the query-execution layer to isolate memory sub-graphs → blocks topology-induced cross-tenant / cross-persona leakage. | — |
| **P4** | **Reversible Reconciliation** | Dual-track storage (Mutable Active Graph + Immutable Episodic Log) enables **asynchronous reconciliation**: periodically "replay" and correct drifted concepts against the raw trace; provides a **rollback** path on severe behavioral degradation. | Episodic vs. semantic memory division |

### Formal read–write–reconcile lifecycle

Let `M_{t−1}` = mutable state, `M_core` = protected core facts, `K_ledger` = append-only raw-observation ledger, `K_true` = idealized semantic target.

- **Read (Constrained Retrieval), Eq. 5** — semantic Top-K *then* governance filter:
  `C_t = { µ ∈ Top-K(q_t, M_{t−1}) | ACL(µ, u_id) ∧ (w(Δτ_µ) ≥ θ_fresh) }`
  i.e., semantic retrieval proposes candidates, governance removes anything the user can't access or that's stale.

- **Write (Gated Transition), Eq. 6** — vs. naive `M_t = M_{t−1} ∪ Agent(C_t)`:
  `M_t = M_{t−1} ∪ G_write(Agent(C_t), M_core)`, where `G_write(ΔM, M_core) = ΔM` if `ΔM ∧ M_core ⊭ ⊥`, else `∅`. Only non-contradictory updates are admitted.

- **Reconciliation (Drift Bounding), Eq. 7** — periodic re-alignment to the ledger:
  `M_clean ← argmin_M E[ δ(R(M, K_ledger), K_true) ]`
  The three reference objects are kept distinct: `M_core` protects facts at write-time, `K_ledger` is the raw trace for correction, `K_true` is the ideal evaluation target.

### Theorem 1 — Bounded Semantic Drift

**Setup:** each consolidation step adds ≤ `ε_step` of semantic error before reconciliation; R restores residual error to a constant independent of horizon.

- **Naive system:** expected drift at T scales **O(T · ε_step)** — linear, unbounded as T → ∞.
- **SSGM with reconciliation every N steps:** expected drift **O(N · ε_step)** — bounded by the *window size N*, not the horizon T.

*Proof sketch:* without Eq. 7, per-step errors accumulate additively to `T · ε_step`. With reconciliation every N steps, error accrues for at most one window (r < N steps) before correction, so the dominant term is N, not T. This is SSGM's core theoretical payoff: **stability even when T ≫ N.**

---

## Experiments / Numbers

**None.** This is a conceptual framework paper — no benchmark results, ablations, or measured metrics are reported. Instead it offers **three testable research hypotheses + evaluation protocols** for the community:

| Hypothesis | Claim | Proposed evaluation |
|---|---|---|
| **H1** — Governance gates bound drift | Agents with `G_write` + R show an **asymptotic upper limit** on drift `δ(M_T, K_true)` over long horizons (T > 100 turns); baselines drift ~linearly | LongMemEval + LLM-as-Judge + BERTScore vs. ground-truth text over extended timesteps |
| **H2** — Access-scoped retrieval cuts leakage | Enforcing Eq. 5 sharply lowers cross-tenant adversarial **injection success rate** without hurting task success | Multi-user role-play; measure the **"Leakage Plateau"** when adversarial data is injected into neighboring graph nodes |
| **H3** — Latency vs. coherence trade-off | Inline contradiction checks (Eq. 6) measurably **raise write latency**; asynchronous governance (run R during idle) can recover coherence without hurting fluidity | Quantify the latency/coherence Pareto trade-off |

The conclusion also calls for standardized **safety benchmarks (e.g., MemoryBench)** that stress-test stability under adversarial drift, and **machine-unlearning protocols** to surgically remove toxic memories.

---

## Key Takeaways

1. **The defining risk of evolving memory is *compounding*.** Unlike static RAG where errors are one-shot, self-updating memory creates a feedback loop where poisoning (ingest), drift (consolidate), and hallucination (retrieve) accumulate and persist. This is the paper's central argument and the reason governance is needed at all.

2. **Decouple evolution from governance.** The single architectural commitment: the agent proposes memory deltas but a **Governance Middleware** with a Write Validation Gate and a Read Filtering Gate sits between cognition and storage. The agent is no longer its own sole validator.

3. **A dual substrate makes drift *reversible*.** Pairing a Mutable Active Graph with an append-only Immutable Episodic Log (the raw ledger) is what enables asynchronous reconciliation and rollback — and what makes Theorem 1's bound possible.

4. **Drift can be bounded, not just observed.** Reconciling every N steps turns drift growth from O(T) into O(N) — a clean, horizon-independent stability guarantee that reframes the goal from "retrieval accuracy" to **memory integrity**.

5. **Safety is partly a *topology* problem.** Fully connected multi-agent memory graphs maximize leakage; the fix is access-scoped retrieval (ABAC/ACL injected into query execution), not better embeddings.

6. **Three fundamental trade-offs are unavoidable** (see Limitations) — latency vs. safety, stability vs. plasticity, and graph scalability — and the paper frames them as the research agenda rather than claiming to solve them.

---

## Limitations (Acknowledged by Authors)

1. **Latency–Safety trade-off.** The governance layer is a "System 2" verification step; validating consistency + provenance on *every* update incurs latency that can make the agent unresponsive in real time. Mitigation direction: **asynchronous governance** (optimistically update, sanitize in background).
2. **Stability–Plasticity conflict.** Aggressive consistency filtering risks **knowledge ossification** — if the gate rejects everything conflicting with established memory, the agent can't adapt to *legitimate* change (e.g., a user moving house). Distinguishing "drift" from "valid update" is an open conflict-resolution problem.
3. **Scalability of graph structures.** Graph memories (Zep, MAGMA) reason well but are hard to keep consistent at scale; growing history degrades traversal and entity resolution, demanding better pruning/compression.
4. **No empirical validation.** SSGM is a blueprint; the bounded-drift theorem rests on assumptions (per-step ε bound, constant-residual reconciliation) and H1–H3 are proposed, not tested.

---

## Where it sits (v1/v2)

SSGM is a **v2 paper on a new axis: GOVERNANCE / safety.** Where most of this collection asks *how to make memory better* (organize, retrieve, learn, compress), SSGM asks *how to keep evolving memory **stable and safe*** — it is the explicit counterpart to the collection's **self-evolving memory systems**:

- **A-Mem** (Zettelkasten link evolution), **MemCoE** (learned RL write policy via GRPO), **Memory-R1 / MemAgent** (RL `ADD/UPDATE/DELETE/RETRIEVE`), **AtomMem** (atomic consolidation) all *grant the agent autonomy to rewrite its own memory*. SSGM is the paper that says: that autonomy is precisely the catalyst for semantic drift, poisoning, and goal drift — so it must be wrapped in governance. It cites A-MEM, Memory-R1, MemAgent, AtomMem, Zep, and MAGMA directly, treating them as the systems whose evolution it governs (and naming Zep/MAGMA in its scalability limitation).

- It operationalizes several **cognitive-grounding** ideas this collection tracks (hippocampal filtering, episodic-vs-semantic split, Weibull/cognitive decay shared with **LightMem** and LiCoMemory) but turns them toward *integrity* rather than efficiency.

**Relation to the LTM Security survey** (`security_long_term_memory_survey.pdf`): SSGM is the *constructive* complement to that survey's *descriptive* threat catalog. The security survey enumerates attack surfaces on long-term memory (poisoning, leakage, extraction); SSGM proposes the **defensive architecture** — Write Filtering (firewall) against poisoning, provenance + ABAC/ACLs against leakage — and adds the *intrinsic* (non-adversarial) failure modes (drift, obsolescence) that a pure-security framing misses. Its Safety row in Table 2 (Memory Poisoning, Privacy Leakage) is exactly the survey's territory.

**Relation to the RTBF privacy paper** (`what_should_llms_forget_rtbf.pdf` / WikiMem): both sit on v2's **trust / privacy / forgetting** axis, but at different memory layers. RTBF/WikiMem addresses **parametric** memory — *which individual–fact associations a model has memorized in its weights*, the forget-set-identification step before unlearning. SSGM addresses **external/agentic** memory — governing what gets *written to and retrieved from* the evolving store, with provenance/ACLs and active forgetting. They are complementary halves of "trustworthy memory": SSGM keeps the external memory safe and clean going forward; RTBF/WikiMem locates baked-in personal data for erasure. SSGM's own conclusion explicitly calls for "machine unlearning protocols to surgically remove toxic memories," which is the bridge to the RTBF line.

**One-line placement:** the **governance/safety axis** of v2 — the natural answer to "if memory now self-evolves (A-Mem, Memory-R1, MemCoE), who validates it?" — pairing a descriptive failure taxonomy with a decoupled governance middleware and a bounded-drift guarantee.
