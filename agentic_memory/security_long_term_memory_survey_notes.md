# A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle

**Authors:** Zehao Lin, Xixuan Hao, Renyu Fu, Shaobo Cui, Kai Chen, Chunyu Li, Zhiyu Li, Feiyu Xiong (MemTensor; Shanghai Jiao Tong University)

**Paper:** arXiv:2604.16548v2 (Jun 2026) — [cs.CR]

---

## Why This Survey Exists

Every other survey in this collection asks how to *build* better agent memory — what to store, how to structure it, when to retrieve, how to forget for utility. This one asks the adversarial question the others left open: **once memory is writable, retrievable, and persistent across sessions, it becomes an attack surface.** The shift from stateless chatbots to agents with long-term memory (LTM) qualitatively changes the security landscape — attackers no longer aim to corrupt a single response, but to manipulate *how the system remembers, retrieves, plans, and acts across all future interactions*.

The authors argue that conventional defenses — prompt-injection filters, RAG-corruption mitigations — are **fundamentally insufficient** for LTM because they operate within a single session or retrieval episode. A larger context window does *not* subsume LTM security: a big window improves within-session recall but does not create content that is automatically retained, shared, and retrieved across future sessions. Three properties make LTM a genuinely new security substrate:

1. **Persistence** — a poisoned entry can be recalled across an indefinite number of future sessions, long after the originating context has closed.
2. **Statefulness** — the unit of analysis shifts from isolated inputs to the agent's *evolving memory state*. An agent that accumulates a cluster of subtly biased episodic memories may drift behaviorally long before any single entry trips a safety classifier.
3. **Propagation** — in multi-agent / shared-state systems, contamination spreads through inter-agent messages, shared stores, and tool arguments, cascading across session, role, and user boundaries.

The central question: *when an LLM agent acquires writable, retrievable, cross-session persistent memory, what qualitative change occurs in the security landscape, and how should the field organize its response?*

**Positioning vs. prior surveys (their Table 1):** existing agent/RAG-security surveys (Zhang 2025d, Wu 2025b, He 2024, Tang 2026, Mu 2026, Bodea 2026, Luo 2026) address at most *one* of {lifecycle organization, benign risk, unifying framework} partially, and none covers all three. This survey's contribution is the unified, **lifecycle-oriented** treatment plus a formal governance framework.

---

## The Organizing Idea: A Memory Lifecycle Framework

Both offensive and defensive papers tend to treat each work as an isolated phenomenon. This survey instead organizes everything along **two axes**:

- **Six lifecycle phases:** WRITE → STORE → RETRIEVE → EXECUTE → SHARE & PROPAGATE → FORGET & ROLLBACK
- **Four security objectives:** Integrity, Confidentiality, Availability, Governance

The payoff is that attacks become *cross-phase chains*: poison **seeded** at WRITE lies **dormant** in STORE, is **reactivated** at RETRIEVE, **steers behavior** at EXECUTE, **spreads** via SHARE, and **survives** incomplete FORGET. The most effective defensive intervention often lies in a *different* phase than where the attack is seeded — which single-turn, input-centric frameworks cannot see.

### The six phases (Section 2)

| # | Phase | What happens | Core security question |
|---|---|---|---|
| ❶ | **WRITE** | Commit content to LTM via explicit user input, dialogue summarization, environmental observation, or cross-agent sharing | Are memory writes subject to *source authentication & authorization*, or can untrusted external content enter as if user-endorsed? |
| ❷ | **STORE** | Index, compress, merge, decay, evict | Can poisoned memory persist, stay traceable, or get *amplified*? (A poisoned entry distilled into a "lesson" gains retrieval priority and apparent authority.) |
| ❸ | **RETRIEVE** | Bring entries back via embedding / keyword / graph / hybrid search | Retrieval is not neutral lookup — it *shapes downstream reasoning* by selecting which memories enter context |
| ❹ | **EXECUTE** | Retrieved memories drive planning, reasoning, tool use | A sufficiently salient memory can *override explicit user instructions* — shifting poisoning from a data-integrity issue to a **control-flow** problem |
| ❺ | **SHARE & PROPAGATE** | Memory spreads across agents (laterally), users → org (vertically), or sessions (temporally) | Can a poisoned memory escape its original context and be reused/inherited/trusted at broader scope? |
| ❻ | **FORGET & ROLLBACK** | Delete, trace, roll back, verify removal | Can the system actually *recover* after a successful attack — or do residues remain? |

**The thesis (stated up front and repeated):** robust LTM security **cannot be retrofitted at retrieval or execution time alone**. It must be anchored in **storage-time provenance, versioning, and policy-aware retention from the outset.**

---

## Attack Patterns Across the Lifecycle (Section 3)

### WRITE — Seeding poisoned memory

The WRITE phase is where memory poisoning first enters. The survey's key insight: ordering write-path attacks by **decreasing attacker privilege** reveals that as required access shrinks, *persistence and scope expand while effectiveness stays high*.

| Attack | Attacker access | Mechanism / evidence |
|---|---|---|
| **AgentPoison** (Chen 2024) | Corpus write | Embedding-space backdoor; **<0.1% poison rate, ≥80% ASR** |
| **InjecMEM** (Tian 2026) | Single interaction | Retriever-agnostic anchoring + Multi-GCG; **76.6% ASR on MemoryOS** |
| **MINJA** (Dong 2025) | **Query-only** | Self-generated poison — bridging steps + indication prompts make the *agent itself* write the malicious memory |
| **MemoryGraft** (Srivastava & He 2025) | Ingestion artifact | Procedural-memory grafting; imitates semantic *procedures* rather than injecting facts; trigger-free |
| **eTAMP** (Zou 2026) | **Environment only** | Web-observation poisoning — adversarial web content absorbed during normal browsing; 8× "frustration" amplification; cross-session *and* cross-site |

The progression is the headline: AgentPoison needs corpus access → MINJA shows query-only interaction *permanently* alters memory state → eTAMP closes the privilege gap entirely (attacker only manipulates a web page the agent happens to visit). **Defensive implication:** the write-stage attack surface includes not just explicit memory updates but *any observable context that can influence what the agent decides to store*.

Also catalogued here: corpus-level poisoning (**BadRAG**, **Phantom**) and environment-injected **SpAIware** (persistent-memory exfiltration in LLM apps).

### STORE — Provenance, retention, and audit

A critical control point, often overlooked. Three mechanisms make STORE adversarially relevant:
- **Indexing** decides which future queries can reactivate an entry → retrieval keys, semantic labels, and storage-priority become *adversarial amplifiers*.
- **Retention / decay / eviction** regulate both availability and confidentiality. Frequency- or recency-based retention can inadvertently keep adversarial entries alive while evicting legitimate ones; without expiration or user-accessible deletion, sensitive content accumulates → growing privacy exposure.
- **Versioning / audit** determine whether post-breach forensics, rollback, and verified forgetting are even *possible*. Provenance, write logs, snapshots, diff-auditable history are prerequisites.

Named attack flavors: **Compression Amplification** (poison distilled into a high-priority "lesson" — A-MEM, MemGPT as the *amplifying substrate*), **Provenance Stripping** (entries lose their lineage — MemOS context).

### RETRIEVE & EXECUTE — From selection to action steering

RETRIEVE is the chokepoint reconnecting past writes with present use. Two attacks define the escalation:
- **MCFA / Memory Control-Flow Attack** (Xu 2026) — a salient retrieved memory **overrides the user's explicit instruction**, dictating *which tools* the agent invokes, in *what order*, with *what arguments*. This is the "storage → steering" leap: poisoning becomes a control-flow problem.
- **MemoryGraft** (Srivastava & He 2025) — poisoned *procedural* memories, retrieved as prior "successful experiences," are reused as task-solving strategies that silently replace the agent's behavior *without a separate trigger*.

Retrieval-poisoning lineage: **PoisonedRAG**, **GraphRAG-under-fire** (Liang 2025a), **KEPo** (knowledge-evolution poison on graph RAG). Backdoor/trigger: **AgentPoison**, **Phantom**.

**The end-to-end chain (Section 3.4):** a manipulated web observation during routine browsing → summarized into a memory card → persisted to LTM → days/weeks later, in a *different session and task*, retrieved and silently steers a tool call. The poisoning window closes *before* execution begins, so **single-turn detection cannot observe both ends of the chain.** This is the survey's strongest argument for lifecycle thinking.

### SHARE & PROPAGATE — Internal channels dominate leakage

Contamination spreads via inter-agent messages, shared stores, and tool-mediated data flows:
- **Agent Smith** (Gu 2024) — a single adversarial image propagates *exponentially* through pairwise exchanges, reaching ~1 million multimodal agents.
- **Troublemaker / contagious jailbreak** (Men 2025) — analogous propagation through dialogue.
- **ComPromptMized / AI Worm** (Cohen 2025) — zero-click *self-replicating* prompts traversing GenAI email assistants; each recipient's agent absorbs and re-emits the payload to its own contacts.
- **Cross-user contamination** (Yang 2026 "No attacker needed" — *unintentional* leakage in shared-state agents; Wang 2025 privacy risks in agent memory).

### FORGET & ROLLBACK — Residual state and failed recovery

Distinct from other phases: the risk is **not** injecting/reactivating a payload, but **failing to fully remove** poisoned or sensitive state after detection. A single memory item leaves **derivatives** across raw dialogue logs, summarized cards, vector indexes, reflected lessons, shared stores, and audit records. Deleting only the visible entry leaves retrievable residue → contamination *reappears after apparent cleanup*. Cataloged: **Ghost of the Past** (residual privacy leakage), **PersistBench** (when should memories be forgotten), reappearance-after-deletion (Wang 2024b), **Agentic Unlearning** (Wang 2026a), failed traceback (**RAGForensics**).

---

## Defenses: Prevention, Containment, Recovery (Section 4)

Defenses mirror the attack lifecycle — no single intervention breaks the full chain, so a **layered, distributed** architecture is required.

| Phase | Defense direction | Representative work |
|---|---|---|
| **WRITE** (prevention) | Human-verified freezing; provenance tagging at write time | **VerificAgent** (audit candidate memories before freezing into a safety contract); **MemCube/MemOS** metadata (source, version, sensitivity, temporal); **PROV-AGENT** (provenance) |
| **STORE** | *Comparatively underdeveloped.* Versioned snapshots, write logs, content-addressable records, compression audits | W3C **PROV** as a serialization layer; **MemOS** snapshots |
| **RETRIEVE** (most mature) | Certifiable aggregation; trust scoring; activation-based detection; memory-native consensus | **RobustRAG** (certifiable robustness via isolate-then-aggregate); **TrustRAG** / **SeCon-RAG** (clustering, trust scoring, contradiction-aware filtering); **RevPRAG** (poisoning via activation differences); **A-MemGuard** (compares independent reasoning paths, treats disagreement as anomaly, distills attacks into lesson memories) |
| **EXECUTE** (containment) | Information-flow control (separate data/control planes); fine-grained tool privilege; execution isolation | **CaMeL**, **FIDES**, **PCAS** (info-flow control); **Progent**, **IsolateGPT**, typed privilege separation (Jacob 2025) |
| **SHARE** (governance) | Principal-aware access modeling; graph-level contagion detection | **Collaborative Memory** (time-evolving principal–resource graph); **BlindGuard** (unsupervised graph anomaly detection for *unknown* contagion) |
| **FORGET & ROLLBACK** (recovery) | Forensic traceback; snapshot/rollback; machine unlearning | **RAGForensics** (post-hoc traceback for *static* RAG); **MemOS** snapshots; **TOFU**, **RAG-unlearn**, machine unlearning (Bourtoule 2021) |

**The asymmetry the map reveals:** defenses cluster at RETRIEVE (well-studied) and are **sparse at STORE, SHARE, and FORGET** — precisely the phases that govern persistence, propagation, and recovery. End-to-end evaluation of forget/rollback for *dynamically updated* agent memory (verifying deletion, reliable rollback, forensic traceback) remains largely missing.

---

## The Governance Dimension: Verifiable Memory Governance (VMG) (Section 5)

The survey's formal contribution. After surveying attacks and defenses, it asks: *what verifiable mechanisms must an LTM system provide to maintain auditable, recoverable control over its own memory state?* It answers with **five primitives**, each given a predicate definition over a memory store `M_t` (where each entry `m` carries metadata `src(m), scope(m), prov(m), ver(m)`):

| Primitive | Predicate (informal) | Withstands | Proposed metric |
|---|---|---|---|
| **Write Authorization (WA)** | Every entry has an authenticated source `src(m)≠⊥` and passes a write-time authorization check | Query-induced injection, environment poisoning (MINJA, InjecMEM, eTAMP) | Adaptive injection-survival rate at bounded FPR |
| **Provenance Visibility (PV)** | Every entry is queryable + lineage-complete back to its originating write, through summarization/merging | Source-monitoring failure, MemoryGraft | Fraction of entries traceable to source event |
| **Principal-Scoped Retrieval (PS)** | Retrieval returns only entries whose authorized scope includes the querying principal | Cross-user contamination, black-box extraction (Yang 2026, Wang 2025, AgentLeak) | Cross-principal leakage rate under scripted query suites |
| **Rollbackability (RB)** | Versioned snapshots + write logs suffice to restore `M_t` to an exact prior state `M_{t0}` | Compression-amplified toxins, behavioral drift | Time-to-remediation; fraction of toxic entries localized |
| **Verified Forgetting (VF_ε)** | After deletion `F_X`, no probing query recovers target content `X` (bounded by ε) — across raw logs, summaries, vector indices, *and* propagated copies | Residual derivatives, reappearance in summaries | Post-deletion membership / reappearance tests |

ε-VMG holds when all five hold simultaneously: `MS_ε(M_t) := WA ∧ PV ∧ PS ∧ RB ∧ VF_ε`.

### The dependency tower (the key diagnostic)

VMG is **not** a layer you bolt on at the end. The primitives form a **design pre-order**:

```
VF_ε  ⪯  RB  ⪯  PV  ⪯  WA        (PS is orthogonal but required for any confidentiality claim)
```

`X ⪯ Y` means implementing X *requires* Y as an architectural precondition. PV depends on WA because lineage-complete provenance needs an authenticated `src(m)` at every write; without write-time authorization, provenance records exist but can't be anchored to a verified origin. Rendered as a tower (their Figure 3): **WA is the widest, most mature foundation; each successive layer narrows and is less deployed** — RB and VF are largely absent in published architectures, and PV is rare.

**The actionable conclusion:** near-term progress should build **provenance infrastructure (PV) first**, before pursuing higher-level primitives whose architectural preconditions are not yet in place. This is the operational restatement of the survey's thesis — anchor security at STORE-time, not RETRIEVE/EXECUTE-time.

---

## Resources, Benchmarks, and the Gap

The survey does *not* contribute a benchmark — its repeated finding is that **no existing benchmark covers the full memory lifecycle.** Existing evaluations are scattered:
- **Attack/injection benchmarks:** ASB (Agent Security Bench), AgentDojo, InjecAgent, poisoning-attack benchmarks (Zhang 2025b).
- **Privacy / contextual-integrity:** **CIMemories** (compositional contextual integrity of persistent memory), **AgentLeak** (full-stack multi-agent privacy leakage).
- **Memory/persistence benchmarks:** **PersistBench** (when should memories be forgotten), **BenchPreS** (context-aware personalized preference selectivity), LongMemEval, LoCoMo, MemBench (these last three measure *capability*, not security).

The stated future direction: design benchmarks that assess LTM security across the **entire** lifecycle — cross-session contamination, share-time propagation, forget-time recovery — which current benchmarks only partially cover.

### Human-memory grounding (Appendix B)

The lifecycle decomposition is grounded in a deliberate analogy to cognitive neuroscience: human memory is **reconstructive and reconsolidatable**, not a fixed archive. Human misinformation operates through *source-monitoring failure, confidence inflation, reconsolidation after recall, and social contagion* (Loftus; Roediger; Nader) — which map cleanly onto agent failures: provenance failure, read-time rewriting via summarization/reflection, and cross-agent contamination. This is the same cognitive-grounding move the v2 systems papers make (CLS, dual-process, Atkinson-Shiffrin) — but turned toward *security* rather than retrieval quality.

---

## Key Takeaways

1. **Persistent memory is a new security substrate, not bigger context.** Its defining properties — persistence, statefulness, propagation — are exactly what single-session prompt-injection / RAG defenses cannot handle.
2. **Attacks are cross-phase chains.** Poison seeded at WRITE (increasingly with *zero* attacker privilege, just a poisoned web page) reactivates at RETRIEVE and steers tool use at EXECUTE — with the seeding window closed before execution, defeating single-turn detection.
3. **Memory poisoning has escalated into a control-flow problem** (MCFA): a salient memory can override explicit user instructions.
4. **The defense map is lopsided** — mature at RETRIEVE, sparse at STORE / SHARE / FORGET, exactly the phases that govern persistence, propagation, and recovery.
5. **Governance must be built in from STORE-time.** The five VMG primitives form a dependency tower (WA → PV → RB → VF, PS orthogonal); since PV underpins recovery and is rare in practice, **provenance infrastructure is the near-term priority.**
6. **No benchmark covers the full lifecycle** — the field's most concrete open gap.

---

## Where it sits (v1/v2)

This paper opens the **security / safety axis that the rest of the collection completely lacked.** Every other paper here — MemGPT, MemoryOS, Mem0, MAGMA, LatentMem, GAM, the EverMemOS / AtomMem / SSGM cluster — is about making memory *better*: richer structure, learned policies, JIT construction, cheaper consolidation. None treats memory as something an adversary attacks. This survey's premise is the direct consequence of theirs: **as agents adopt persistent, writable, cross-session memory (Mem0, MemoryOS, MemOS, Zep, EverMemOS), that memory becomes the attack surface.** The very mechanisms the systems papers celebrate are re-read here as vulnerabilities — MemGPT/A-MEM's compression *amplifies* poison into authoritative lessons; MemoryOS is the concrete target of InjecMEM (76.6% ASR); shared multi-agent memory (LatentMem's selling point) is the propagation channel; experience reuse (ReasoningBank, MemoryGraft's target) is the procedural-grafting vector.

**Two collection papers it connects to most directly:**

- **RTBF / WikiMem** (`what_should_llms_forget_rtbf`) — the collection's only prior privacy/forgetting paper, but *off-axis* relative to this one. WikiMem audits **parametric** memory: which `(h, p, v)` triples a model has memorized in its *weights*, producing the forget-set that unlearning presupposes. This survey covers the **external/agentic** memory lifecycle. They are complementary halves of the forgetting problem: WikiMem = "what's baked into the weights and how do we even find it" (the GDPR-erasure side); this survey's **Verified Forgetting (VF_ε)** primitive = "after we delete from the *memory store*, prove it can't reappear across logs, summaries, indices, and propagated copies." Both invoke **TOFU / machine unlearning** (Bourtoule, Maini); WikiMem feeds the forget set, VF_ε verifies the removal. Together they show forgetting must be solved at *both* the parametric and the agentic-store layer.

- **SSGM** (`ssgm_governing_evolving_memory`) — the collection's other governance paper, and a near-sibling published weeks earlier (arXiv 2603.11768, May 2026 vs. this 2604.16548, Jun 2026). SSGM ("Stability and Safety-Governed Memory") also identifies a memory **risk lifecycle** — input ingestion (poisoning), consolidation (semantic/procedural drift), retrieval (hallucination/conflict) — and proposes a governance architecture that decouples memory *evolution* from *execution* via consistency verification, temporal-decay modeling, and dynamic access control *before* consolidation. The two papers are strikingly convergent and mutually reinforcing: **SSGM emphasizes the *benign* failure mode (drift, the stability-plasticity dilemma) and a defensive architecture; this survey emphasizes the *adversarial* mode (poisoning, injection, contagion) and a formal verifiable-governance specification (VMG's five predicates).** SSGM's "govern before consolidation" is essentially this survey's thesis — anchor control at STORE/WRITE-time, never retrofit at RETRIEVE — stated as an architecture rather than a predicate set. Read together, SSGM + this survey form the collection's **governance pillar**, and both explicitly cite the *Memory in the Age of AI Agents* survey as the build-side counterpart whose systems they are securing.

**Net:** if v1 stored text and queried it, and v2 learned/structured/generated memory, then this paper marks the moment the field admits that **evolving, persistent memory is also an attack surface that must be authorized, traced, scoped, rolled back, and verifiably forgotten** — the trust/privacy frontier the *Memory in the Age of AI Agents* survey flagged (Section 7.7) but did not flesh out, now given a concrete attack taxonomy, defense map, and formal governance framework.
