# What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests

**Author:** Dimitri Staufer (TU Berlin)

**Paper:** arXiv:2507.11128v1 (15 Jul 2025) — accepted at the 7th Workshop on eXplainable Knowledge Discovery in Data Mining (XKDD 2025), ECML PKDD 2025, Porto

**Dataset/Code:** WikiMem (released CC-BY); evaluation code at the anonymized `WikiMem-Eval` repo

---

## The Core Problem

Machine unlearning for LLMs is built on a hidden assumption: that you already **know the forget set** `D_f` — the precise data you want to remove. Every gradient-based, distillation-based, or local-edit unlearning method takes `D_f` as input. But the actual hard problem for the GDPR's **Right to Be Forgotten (RTBF, Art. 17 Right to Erasure)** is the step *before* unlearning:

> *Which individual–fact associations has the model actually memorized about a given person?*

This is fundamentally different from search-engine RTBF. A search engine is an information-retrieval system with an inverted index mapping terms to documents; de-listing means editing or deleting index entries that link to a person. An LLM instead encodes information **implicitly across its parameters**, with no direct link back to training documents. Personal data is baked into distributed weights, making it hard to even *locate*, let alone erase.

Existing privacy-auditing work doesn't fill this gap because it:

1. Operates at the **population level** ("how often does the model leak *any* email address?") rather than the **entity level** ("what does the model know about person *h*?")
2. Targets a **small set of identifier types** (e.g., ProPILE covers only email, phone, address, family relation, affiliation; RWKU covers only high-profile people and a narrow set of verifiable attributes)
3. Relies on **resource-intensive per-target probing** (e.g., Zhou et al. need per-entity soft-prompt training, which doesn't scale and isn't model-agnostic)
4. Ignores **prompt-formulation variability** — even minor stylistic changes flip whether a fact is "leaked"

So the contribution is a way to *quantify, at the individual level, which facts a model has memorized* — producing the forget set that unlearning methods presuppose.

---

## Framing: Personal Data as a Recoverable Triple

The paper formalizes personal data in a way that matches how LLMs internally store knowledge. Rather than documents or URLs, an LLM encodes statistical associations between a **subject `h`**, a **property `p`**, and a **value `v`** — a triple `(h, p, v)` (e.g., `(Simone de Beauvoir, place of birth, Paris)`).

> Personal data is defined as any factual association `(h, p, v)` the model has learned **with sufficient fidelity to be recoverable through inference**.

Under the GDPR this counts as personal data if it pertains to an identifiable individual and is accessible "via reasonably likely means." This definition is what makes the metric legally meaningful: it ties memorization measurement to the GDPR's accessibility threshold.

---

## Contribution 1: The WikiMem Dataset

A large-scale, open dataset of natural-language **"canaries"** built from Wikidata (a knowledge base with >1.3 billion uniquely-identified entities; `Q5` = "human").

| Property | Value |
|---|---|
| Human-related properties covered | **243** |
| Total natural-language canary templates | **5,650** |
| Counterfactuals per association | **100** randomly sampled human–value pairs |
| Source | Wikidata triples `(h, p, v)` |
| License | CC-BY |

**Property filtering pipeline** (multi-stage):
1. Restrict to human entities via `P31 = Q5` ("instance of: human")
2. Keep only properties whose Wikidata datatype is suitable for text prompting: `wikibase-item`, `string`, `quantity`, `time`
3. Measure usage frequency by iterating the **entire knowledge-base dump (~130 hours)**; discard any property associated with fewer than **100 distinct humans**

This yields 243 diverse properties including `occupation` (P106), `spouse`, `blood type`, `hair color`, and `convicted of`. Each property carries its label plus aliases (e.g., for `occupation`: "profession", "job", "work", "career").

**Counterfactual generation:** stream the compressed Wikidata dump; for each property collect `(h, v)` pairs (`h` = English label of a human, `v` = object Q-ID), deduplicate by name, shuffle, and fetch English labels for 100 pairs via the `wbgetentities` API. These become the distractors (e.g., for `occupation`: "nurse", "carpenter").

**Canary construction** — three types, to test robustness to phrasing:

- **Declarative (baseline):** triples → declarative English via regex + SpaCy parsing, in **copular** ("h is employed by v"), **possessive** ("h's mother-in-law is v"), or **transitive** ("h holds a diplomatic passport of v") form. One baseline template per `(h, p, v)`.
- **Paraphrased variants:** a **FLAN-T5-XL** fine-tuned on the `chatgpt-paraphrases` dataset (Quora QP + SQuAD2.0 + CNN/DailyMail + GPT-3.5 paraphrases) generates **50 rewrites** per baseline; **o4-mini-high** ranks them by semantic similarity *and* lexical/syntactic diversity, keeping the **top 10**.
- **Contextualized canaries:** prepend up to **4 auxiliary true facts** about `h` to narrow the anonymity set (following Nakka et al.'s PII-Compass finding). E.g., for 50 Cent's number of children: *"50 Cent's country of citizenship is USA. Their place of birth is South Jamaica. Their occupation is rapper. Their number of children is v."*

---

## Contribution 2: A Model-Agnostic Memorization Metric

A method to quantify (1) **Association** — whether the model links subject `h` to value `v` under property `p` — and (2) **Strength** — how robust that link is, independent of phrasing. Three stages: calibrated NLL ranking → memorization decision → strength score.

### Calibrated NLL Scoring

The base signal is **Negative Log-Likelihood** of a candidate completion, which directly reflects the model's confidence:

```
NLL(h, v) = − Σ_{t=1}^{T} log p(w_t | w_<t)
```

(Perplexity is just `exp(NLL)` normalized by token count `T`.) It works on black-box models too, since logprobs can be reconstructed via logit-bias queries and top-k outputs.

For each candidate value `v_i` the score subtracts two confounds:

```
s(h, v_i) = [NLL(h₀, v_i) − NLL(h, v_i)]  −  α · E_{h̃∈S(h)}[ NLL(h₀, v_i) − NLL(h̃, v_i) ]
              └─── subject calibration ───┘     └────── similar-name adjustment ──────┘
```

- **`h₀`** = a generic subject ("This person's…"), estimating what the model predicts for an *average* individual — neutralizes the model's prior over values.
- **`S(h)`** = similar-looking name variants (e.g., "Enaj Doe", "Jane Eod" for "Jane Doe") — neutralizes phonetic/cultural name biases. Set `α = 1`.

### Memorization Decision

Sort candidates by descending `s(h, v_i)`, giving each a rank `r_i`. A subject may have multiple ground-truth values under a property (e.g., "biologist" *and* "zookeeper" under occupation). **Memorization is declared if any ground truth ranks #1**:

```
∃ v_gt ∈ V_gt  s.t.  r_gt = 1
```

(Relaxed when ground truth and counterfactual are near-synonyms — e.g., "English" vs "British English" for native language — treated as equal if sentence-embedding cosine similarity > 0.75.)

### Memorization Strength (z*)

If the top ground truth `v*` ranks #1, measure the **lead margin** (the confidence "jump") over the best counterfactual `s̄`:

```
Δ* = s(h, v*) − s̄
```

Then z-normalize it against the distribution of "stand-out margins" `Δ_i` across all candidates (mean `µ`, std `σ`):

```
z* = (Δ* − µ) / σ
```

A larger `z*` means the model's confidence in `v*` exceeds the typical margin by more standard deviations — a stronger, more certain association. Crucially, `z*` needs **only likelihood queries (no white-box access)** and applies to any causal LM — this is the model-agnostic property prior methods lacked.

---

## Experimental Setup

- **Subjects:** 200 Wikidata humans — **100 high web-presence** ("well-known") and **100 low** ("lesser-known") — split by a composite score over Wikipedia page views, article length, and number of language editions.
- **Properties tested:** the 5 most frequent — occupation (P106), language (P1412), place of birth (P19), sex/gender (P21), citizenship (P27).
- **Per subject–property pair:** 1 base template + 10 paraphrased variants (**11 canaries**), each scored against **100 counterfactuals**.
- **Models:** 15 causal LMs, **410M–70B params**, across 4 families — **LLaMA 3.1, Mistral, Pythia, Qwen3** — base and instruction-tuned variants.
- **Precision:** loaded via Hugging Face, evaluated in **4-bit NF4 with double quantization**. Negligible difference vs FP/16-bit; but **2-bit and 1-bit sharply degraded** memorization recall — extreme quantization itself prevents accurate fact retrieval.
- **Hardware:** cluster of 4× A100 + an A6000.

The reported metric uses a **strict** interpretation: a subject–property pair counts as memorized only if **all 11 template variants** yield a rank-1 ground-truth prediction. `M(%)` is the percentage of paraphrased templates per pair surfacing the ground truth at rank 1. (A **lenient** "any paraphrase succeeds" definition gives **>80%** across all properties for the well-known cohort on LLaMA 3.1-8B — and the paper notes the GDPR's "reasonably foreseeable means" standard may make the lenient threshold the legally relevant one.)

---

## Results

### Memorization by web presence and scale (Table 1)

Mean memorization rate `M(%)`, strength `z*`, and count of subjects `H_M=0` with **zero** memorized properties, per cohort:

| Model | Well-known M(%) | Well-known z* | Well-known H_M=0 | Lesser-known M(%) | Lesser-known z* | Lesser-known H_M=0 |
|---|---|---|---|---|---|---|
| Pythia-410M | 10.22 | 2.73 | 6/100 | 4.38 | 2.51 | 31/100 |
| Pythia-6.9B | 26.34 | 3.09 | 0/100 | 11.73 | 2.92 | 7/100 |
| Pythia-12B | 24.22 | 3.11 | 0/100 | 9.51 | 2.97 | 17/100 |
| Qwen3-8B | 30.08 | 3.09 | 0/100 | 11.67 | 3.03 | 12/100 |
| Qwen3-30B-A3B | 37.41 | 3.72 | 0/100 | 18.31 | 3.07 | 9/100 |
| LLaMA 3.1-8B | **38.84** | 3.38 | 1/100 | **22.43** | 3.08 | 3/100 |
| LLaMA 3.1-70B | 38.33 | **3.77** | 0/100 | 18.66 | **3.37** | 5/100 |
| Mistral-7B v0.3 | 28.98 | 3.16 | 3/100 | 13.50 | 2.66 | 23/100 |
| Mistral-Small-24B-2501 | 19.43 | 2.55 | 13/100 | 6.92 | 2.27 | 46/100 |

**Key findings:**

1. **Web presence dominates.** Every model memorizes well-known subjects far more than lesser-known ones (e.g., LLaMA 3.1-8B: 38.8% vs 22.4%). Memorization correlates with a subject's online footprint.
2. **More facts with scale, but it plateaus.** Memorization *rate* rises from tiny models (Pythia-410M at 10.22%) up the scale, but plateaus — Pythia-12B and LLaMA 3.1-70B don't beat their smaller siblings (12B < 6.9B; 70B ≈ 8B on rate).
3. **Strength keeps climbing with scale.** `z*` rises monotonically with size — **2.73 (Pythia-410M) → 3.77 (LLaMA 70B)**, especially for well-known subjects. Interpretation: *larger models don't necessarily memorize more facts, but encode the ones they do with greater certainty.*
4. **Mistral-Small-24B is the outlier** — unexpectedly low memorization for its scale (19.43% / 13 subjects with zero memorized properties in the well-known cohort). Manual cross-checks in Open WebUI (with Wikipedia hints) surfaced only broad generalities, suggesting it was trained on filtered/deduplicated data.

### Instruction tuning (Figure 1, LLaMA 3.1-8B base vs instruct)

- Instruction tuning **raises** per-subject memorization across all properties — **except "Sex or Gender,"** where memorization is already low and **drops near zero** for the instruct variant. The author reads this as a deliberate **safety constraint** added for that attribute.

### Contextualized canaries

- Adding auxiliary true facts about the subject **consistently lowered** memorization rates and strengths across all models. Hypothesis: the extra context spreads probability mass over multiple plausible continuations, lowering the relative likelihood of the single ground-truth value. (Notably the *opposite* of prior PII-extraction findings, where context helped.)

---

## Key Takeaways

1. **Identifying the forget set is the missing prerequisite for RTBF.** Unlearning research assumes `D_f` is given; this paper provides the auditing layer that *constructs* `D_f` at the individual level, per `(h, p, v)` triple — the step that actually makes RTBF actionable against parametric memory.

2. **A model-agnostic, black-box-compatible metric.** Calibrated NLL ranking + the `z*` strength score need only likelihood queries — no soft-prompt training, no white-box access, no per-target optimization. This is what lets it scale to 200 subjects × 15 models where prior methods (ProPILE, Zhou et al.) could not.

3. **Calibration matters.** Subtracting a generic-subject baseline and a similar-name correction is what separates *genuine subject-fact memorization* from the model's generic prior over values and its name-based biases.

4. **Memorization tracks web presence and scale — but scale buys certainty, not coverage.** Rate plateaus while strength (`z*`) keeps rising. Bigger models are more *confident* about the facts they hold, which is arguably the more privacy-relevant signal.

5. **The strict/lenient threshold is a legal choice, not just a technical one.** Strict (all 11 paraphrases) vs lenient (any paraphrase) shifts measured memorization from ~20–40% to >80%. The GDPR's "reasonably foreseeable means" standard may make the lenient (higher) number the one regulators care about.

6. **Quantization is a confound for auditing.** 4-bit NF4 preserves memorization, but 2-bit/1-bit destroy it — an auditor must probe at a precision that doesn't itself suppress recall.

---

## Limitations (acknowledged by the author)

1. **Strict-threshold assumption.** Treating a pair as memorized only if all 11 variants succeed is a strong assumption that linguistic diversity suffices to show memorization with high confidence; it may undercount.
2. **No completeness/temporal checks for multi-valued properties.** For properties with multiple values (e.g., "worked for"), any valid value is accepted without verifying completeness or temporal correctness (worked-for vs works-for).
3. **Wikidata phrasing bias.** Reliance on Wikidata labels/aliases — which don't always match natural usage ("Jane keeps a pet cat") — risks conflating *factual recall* with the model's ability to parse/produce canonical Wikidata phrasing, or to favor counterfactuals merely for grammaticality.
4. **Counterfactual difficulty is uncontrolled.** A fixed set of type-consistent counterfactuals isn't guaranteed to be uniformly hard or disambiguating; a model may hold a semantically-correct fact phrased differently or mapped to a different Q-ID (e.g., "mathematician" vs ground-truth "logician").
5. **Scope.** English-only prompts; subjects drawn from Wikidata (cultural/systemic bias). Future work: black-box evaluation of proprietary models, and user studies with individuals *outside* Wikidata.

---

## Where it sits (v1 / v2)

This is the **trustworthy / privacy / forgetting** dimension of agent memory — the corner the agentic-memory survey flags as a frontier ("trustworthy memory" and "forgetting"). Most papers in this collection (MAGMA, A-MEM, MemoryBank, the survey) treat memory as **external, retrievable storage**: how to write, organize, retrieve, and consolidate an agent's experiences. This paper instead addresses **parametric memory** — facts baked into the *weights* themselves — and the legal-ethical question of the **right to delete** them. It is the auditing front-end to machine unlearning: you cannot forget what you cannot first locate, and in an LLM personal data has no index entry to delete.

**v1 vs v2:** This is a **foundational (v1, 2023–2025)** contribution to the privacy/forgetting axis. It is a *measurement / auditing* paper — it builds the dataset (WikiMem) and the metric (`z*`) that quantify the problem, deliberately stopping short of doing the unlearning. The 2026-frontier (v2) follow-through would be coupling this auditing layer to actual unlearning, extending it to proprietary black-box and multilingual models, and handling the multi-valued/temporal facts it currently sidesteps.

**Contrast — two very different "forgetting":**

| | **This paper (RTBF unlearning)** | **MemoryBank (Ebbinghaus decay)** |
|---|---|---|
| What is forgotten | Parametric facts in model weights | Entries in an external memory store |
| Why forget | **Privacy / legal compliance** (GDPR Art. 17) | **Utility** — keep memory relevant, prune stale low-value items |
| Mechanism | Identify `(h, p, v)` → build forget set → (downstream) machine unlearning | Time-based exponential decay of retention strength, refreshed on recall |
| Goal of forgetting | *Erase* specific information so it can't be recovered | *Manage* information so the useful stuff stays salient |
| Reversibility | Should be irreversible (truly deleted) | Reversible — recall resets the decay curve |

In short: MemoryBank forgets to **remember better**; RTBF unlearning forgets because someone has the **right to be deleted**. One is curation of external memory for utility; the other is erasure of parametric memory for privacy — and this paper supplies the missing first step (what to forget) for the latter.
