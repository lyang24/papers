# RaBitQ: Quantizing High-Dimensional Vectors with Theoretical Error Bound

**Paper**: "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search" (SIGMOD 2024)
**Authors**: Jianyang Gao, Cheng Long (Nanyang Technological University)

---

## Executive Summary

RaBitQ is a **randomized quantization method** that quantizes D-dimensional vectors into **D-bit strings** while providing **rigorous theoretical error bounds**. It addresses the critical limitation of Product Quantization (PQ) and its variants: **lack of theoretical guarantees**, which leads to unpredictable and sometimes catastrophic failures on real-world datasets.

**Key Innovation**: RaBitQ is the first quantization method for ANN search that achieves:
1. **Unbiased distance estimation** with sharp probabilistic error bounds
2. **Asymptotically optimal** error bound: O(1/√D) with D-bit codes
3. **Better empirical accuracy** than PQ even with **half the code length** (D bits vs 2D bits)
4. **Stable performance** across all datasets without parameter tuning

---

## Core Methodology

### 1. Index Phase: Codebook Construction

**Normalization**:
```
o = (o_r - c) / ||o_r - c||
```
- Normalizes vectors to unit sphere using cluster centroids
- Reduces problem to estimating inner products ⟨o, q⟩

**Codebook Construction**:
```
C = {±1/√D}^D  (deterministic bi-valued vectors)
C_rand = {P·x | x ∈ C}  (randomly rotated)
```
- Starts with 2^D bi-valued vectors (vertices of hypercube on unit sphere)
- Applies random orthogonal transformation P (Johnson-Lindenstrauss Transform)
- Codebook stored implicitly (only P is materialized)

**Quantization**:
- For data vector o, find nearest vector in C_rand
- Store as D-bit string x̄_b where each bit indicates sign of coordinate
- Pre-compute ⟨ō, o⟩ for distance estimation

### 2. Query Phase: Distance Estimation

**Unbiased Estimator**:
```
⟨o, q⟩ ≈ ⟨ō, q⟩ / ⟨ō, o⟩
```
where:
- ō = quantized data vector (D-bit string)
- q = query vector (not quantized in asymmetric version)
- ⟨ō, o⟩ = pre-computed during indexing

**Theoretical Guarantee** (Theorem 3.2):
```
E[⟨ō, q⟩ / ⟨ō, o⟩] = ⟨o, q⟩  (unbiased)

P[|⟨ō, q⟩/⟨ō, o⟩ - ⟨o, q⟩| > ε₀/√(D-1)] ≤ 2e^(-c₀ε₀²)

|⟨ō, q⟩/⟨ō, o⟩ - ⟨o, q⟩| = O(1/√D) with high probability
```

**Distance Computation**:
```
||o_r - q_r||² = ||o_r - c||² + ||q_r - c||² - 2·||o_r - c||·||q_r - c||·⟨ō, q⟩/⟨ō, o⟩
```

### 3. Efficient Implementations

**Single Vector (Bitwise Operations)**:
- Quantize query to 4-bit unsigned integers: q̄_u
- Compute ⟨x̄_b, q̄_u⟩ using bitwise AND and popcount
- **3x faster** than original PQ implementation

**Batch (SIMD-based)**:
- Same fast SIMD implementation as PQ Fast Scan
- Split into 4-bit sub-segments, use SIMD shuffle instructions
- Comparable efficiency to PQ when using similar code lengths

---

## RaBitQ vs Product Quantization (PQ): Comprehensive Comparison

### Architecture Comparison

| Aspect | **RaBitQ** | **PQ and Variants** |
|--------|-----------|-------------------|
| **Codebook** | Randomly rotated bi-valued vectors | Cartesian product of sub-codebooks learned via K-means |
| **Codebook Size** | 2^D (implicit) | (2^k)^M where M=D/2, k=4 or 8 |
| **Codebook Storage** | D×D floats (matrix P) | M × 2^k × (D/M) floats |
| **Code Length** | D bits (default) | M×k bits (typically 2D bits for M=D/2, k=4) |
| **Code Format** | Bit string | Sequence of k-bit unsigned integers |

### Theoretical Guarantees

| Property | **RaBitQ** | **PQ** |
|----------|-----------|--------|
| **Theoretical Error Bound** | ✅ **O(1/√D)** with high probability | ❌ **None** |
| **Unbiased Estimator** | ✅ **Yes**: E[estimate] = true distance | ❌ **No**: Biased by ~0.8 |
| **Optimality** | ✅ **Asymptotically optimal** (matches lower bound from [Alon & Klartag 2017]) | ❌ N/A |
| **Failure Probability** | ✅ **Bounded**: 2exp(-c₀ε₀²) | ❌ **Unbounded** |
| **Stability Guarantee** | ✅ **Works on all data distributions** | ❌ **Can fail catastrophically** |

### Distance Estimation

**RaBitQ**:
- **Asymmetric**: Query not quantized, only data vectors
- **Estimator**: ⟨ō, q⟩ / ⟨ō, o⟩ (carefully designed based on geometric analysis)
- **Error**: Bounded, unbiased, concentrates around true value

**PQ**:
- **Asymmetric**: Query not quantized (in ADC variant)
- **Estimator**: Simply treats quantized vector as original: ⟨ō, q⟩
- **Error**: No bound, biased, can be arbitrarily large

### Empirical Performance (from Experiments)

**Accuracy (64-bit codes)**:

| Dataset | RaBitQ Avg Error | PQ Avg Error | RaBitQ Max Error | PQ Max Error |
|---------|-----------------|-------------|-----------------|-------------|
| MSong | ~5% | **>50%** 🔴 | ~20% | **>100%** 🔴 |
| SIFT | ~3% | ~7% | ~20% | ~60% |
| GIST | ~2% | ~5% | ~15% | ~80% |
| DEEP | ~5% | ~12% | ~25% | ~60% |
| Word2Vec | ~8% | ~15% | ~40% | **>200%** 🔴 |

🔴 = Catastrophic failure

**Key Insight**: RaBitQ achieves better accuracy than PQ **even when using half the code length** (D bits vs 2D bits)!

**ANN Recall (K=100)**:

| Dataset | RaBitQ @ 90% Recall | PQ @ 90% Recall | Speedup |
|---------|-------------------|----------------|---------|
| SIFT | ~5000 QPS | ~3000 QPS | 1.7x |
| GIST | ~800 QPS | ~500 QPS | 1.6x |
| MSong | ~4000 QPS | **Cannot reach 60%** 🔴 | N/A |

**Efficiency**:
- **Single vector**: RaBitQ is **3x faster** than original PQ (bitwise ops vs RAM lookup)
- **Batch**: RaBitQ comparable to PQ Fast Scan (both use SIMD)
- **Memory**: RaBitQ uses ~50% less memory for codes (D bits vs 2D bits)

### Why PQ Fails: The Root Causes

**1. Heuristic Codebook Construction**:
- K-means on sub-segments is a **heuristic** with no guarantees
- Local optimum trap: Cannot guarantee quantization error bounds
- **Dataset-dependent**: Works well on some datasets, fails on others

**2. Biased Distance Estimator**:
- Treating quantized vector as original is **intuitive but wrong**
- Systematic underestimation by ~20% (empirically observed)
- No theoretical analysis of error accumulation

**3. No Robustness Guarantee**:
- Cannot predict when/why it will fail
- Example: MSong dataset - **>50% average error, <60% recall**
- Requires exhaustive testing on every new dataset

### Parameter Tuning

| Aspect | **RaBitQ** | **PQ** |
|--------|-----------|--------|
| **Parameters** | ε₀, B_q | M (sub-segments), k (bits per segment), re-ranking count |
| **Tuning Required** | ❌ **No** - Theory provides explicit guidance | ✅ **Yes** - Empirical trial and error |
| **Default Settings** | ε₀=1.9, B_q=4 **(works across all datasets)** | M=D/2, k=4, re-rank=??? **(varies per dataset)** |
| **Theory-Guided** | ✅ ε₀=Θ(√log(1/δ)), B_q=Θ(log log D) | ❌ No theoretical guidance |

**Critical Finding**: For PQ, **no single re-ranking parameter works well across datasets**:
- SIFT/DEEP/GIST: 1,000 candidates sufficient
- Image/Word2Vec: Need 2,500+ candidates
- MSong: **Even 2,500 candidates insufficient**

RaBitQ uses **error-bound-based re-ranking** (no parameter tuning needed).

---

## Key Technical Innovations

### 1. Geometric Relationship Analysis (Lemma 3.1)

RaBitQ explicitly derives the relationship between ⟨o, q⟩ and ⟨ō, q⟩:

```
⟨ō, q⟩ = ⟨ō, o⟩·⟨o, q⟩ + ⟨ō, e₁⟩·√(1 - ⟨o, q⟩²)
```

where e₁ is the component of q orthogonal to o.

**PQ's approach**: Ignores this relationship, simply uses ⟨ō, q⟩ directly ❌

### 2. Distribution Analysis (Lemma B.3)

RaBitQ proves that due to random rotation:
- ⟨ō, o⟩ concentrates around **0.8** (verified empirically)
- ⟨ō, e₁⟩ has expectation **0** and is independent of ⟨ō, o⟩
- This enables construction of unbiased estimator

**PQ's approach**: No distributional analysis ❌

### 3. Randomized Uniform Scalar Quantization (Theorem 3.3)

For query quantization, RaBitQ proves:
- B_q = Θ(log log D) bits sufficient
- Error from scalar quantization: O(1/√D) (negligible)
- **Only 4 bits needed** in practice (vs 32 bits for float)

**Innovation**: Randomization makes quantization unbiased!

### 4. Error-Bound-Based Re-Ranking

Using the confidence interval from Theorem 3.2:
```
[⟨ō, q⟩/⟨ō, o⟩ - ε₀/√D,  ⟨ō, q⟩/⟨ō, o⟩ + ε₀/√D]
```

RaBitQ can **automatically decide** which vectors to re-rank:
- If lower bound of distance > current best exact distance → **skip**
- Otherwise → **re-rank**

**PQ's approach**: Fixed number of top-k re-ranking (requires tuning) ❌

---

## When Each Method Excels

### Use RaBitQ When:

✅ **Robustness is critical** - cannot afford unpredictable failures
✅ **Working with new/unknown datasets** - no time for extensive tuning
✅ **Memory is constrained** - half the code length of PQ
✅ **Single-vector queries** - 3x faster than original PQ
✅ **Theoretical guarantees needed** - provable error bounds
✅ **Error-bound-based re-ranking** - automatic, no tuning

### Use PQ When:

⚠️ **Dataset is known to work well** (e.g., SIFT with standard settings)
⚠️ **Extensive tuning resources available**
⚠️ **Only batch queries** (Fast Scan implementation)
⚠️ **No theoretical guarantees needed**

**Reality Check**: Given RaBitQ's superior performance, there are **few scenarios where PQ is preferable**.

---

## Performance Deep Dive: Why RaBitQ Wins

### 1. Memory Efficiency

**Storage per vector**:
```
RaBitQ:  D bits                    (e.g., 128 bits for SIFT)
PQ:      M × k bits                (e.g., 64 × 4 = 256 bits for SIFT)
Ratio:   ~2x memory savings
```

**Total memory (1M SIFT vectors)**:
```
Raw vectors:  512 MB (128 × 4 bytes)
RaBitQ codes: 16 MB  (128 bits)    → 32x compression
PQ codes:     32 MB  (256 bits)    → 16x compression
```

### 2. Computational Efficiency

**Distance estimation complexity**:
```
RaBitQ (single): B_q bitwise ops  ≈ 4 AND + 4 popcount = ~8 ops
PQ (single):     M RAM lookups    = 64 cache misses    = ~640 cycles
RaBitQ (batch):  SIMD shuffle     ≈ Same as PQ Fast Scan
PQ (batch):      SIMD shuffle     ≈ Same as RaBitQ
```

**Measured latency (GIST, single vector)**:
```
RaBitQ:  5-10 ns/vector
PQ:      15-30 ns/vector
Speedup: 3x
```

### 3. Accuracy Stability

**Standard deviation of relative error** (across queries):
```
Dataset   RaBitQ    PQ       Stability Improvement
SIFT      1.2%      3.8%     3.2x more stable
GIST      0.8%      2.9%     3.6x more stable
MSong     2.1%      45.3%    21.6x more stable 🔥
```

### 4. Recall Consistency

**Minimum recall achieved** (across different parameter settings):
```
Dataset   RaBitQ    PQ
SIFT      88%       72%
GIST      92%       78%
MSong     85%       <60% 🔴  Catastrophic failure
```

RaBitQ **never fails catastrophically** across all tested datasets.

---

## Real-World Impact: The MSong Disaster

**Dataset**: MSong (420-dimensional audio features, 992K vectors)

### PQ's Performance:
- **Average error**: >50%
- **Maximum error**: >100% (estimated distance is negative!)
- **Recall@100**: <60% even with 2,500 re-ranked
- **Root cause**: K-means clustering on 420-dim subspaces produces poor codebook

### RaBitQ's Performance:
- **Average error**: ~5%
- **Maximum error**: ~20%
- **Recall@100**: ~90% with automatic re-ranking
- **Why it works**: Theoretical guarantees hold regardless of data distribution

**Lesson**: PQ's lack of theoretical bounds means it can **fail silently and catastrophically** on production data.

---

## Integration with ANN Systems

### With Inverted File (IVF)

Both methods combine with IVF:
```
1. Coarse quantizer: Cluster data into k' buckets (e.g., 4,096)
2. Fine quantizer: Quantize residuals within buckets
```

**RaBitQ advantage**:
- Normalizes with cluster centroids → better residual distribution
- Error-bound-based re-ranking → no parameter tuning

**PQ limitation**:
- Fixed k top vectors re-ranked → requires tuning per dataset

### With Graph-Based Methods

**Current status**: Both can combine with graph methods (e.g., HNSW, NGT-QG)

**RaBitQ challenges**:
- Graph search is sequential → hard to batch for SIMD
- Future work: Efficient single-vector graph integration

**PQ challenges**:
- Same batching issue for graph methods
- Additional: No theoretical guarantees compound with graph approximation

---

## Limitations and Future Work

### Current Limitations

1. **Normalization assumption** (for multiplicative bound):
   - Additive bound O(1/√D) holds always
   - Multiplicative bound requires well-normalized data
   - Future: Study optimal normalization strategies

2. **Graph-based integration**:
   - Current work focuses on IVF
   - Graph methods require different batching strategies
   - Future: Adapt RaBitQ for efficient graph search

3. **Non-Euclidean distances**:
   - RaBitQ designed for Euclidean distance
   - Future: Extend to cosine similarity, inner product search

### Theoretical Extensions

**Already proven**:
- ✅ Unbiased for cosine similarity (same as inner product of unit vectors)
- ✅ Applicable to maximum inner product search (MIPS)
- ✅ Can be used for neural network quantization

**Future opportunities**:
- Better normalization with provable guarantees
- Adaptive B_q based on query importance
- Extensions to other metric spaces

---

## Key Takeaways

### For Practitioners

1. **RaBitQ provides superior accuracy** with half the memory of PQ
2. **No parameter tuning required** - theory provides explicit guidance
3. **Robust across all datasets** - never fails catastrophically like PQ
4. **Drop-in replacement** for PQ in existing systems (IVF, SIMD implementations)
5. **Use error-bound-based re-ranking** - automatically balances accuracy/efficiency

### For Researchers

1. **First quantization method with theoretical guarantees** for ANN search
2. **Achieves asymptotic optimality** - O(1/√D) with D bits
3. **Demonstrates value of theory** - principled design outperforms heuristics
4. **Opens new research directions**:
   - Optimal normalization for vector quantization
   - Theoretical analysis of other quantization methods
   - Integration of guarantees across ANN system components

### The Bottom Line

**RaBitQ fundamentally changes the quantization landscape for ANN search**:
- **Before RaBitQ**: Use heuristic methods (PQ, OPQ), hope they work, tune extensively
- **After RaBitQ**: Use principled method with guarantees, deploy confidently

**The PQ Era is Over**: With better accuracy, lower memory, theoretical guarantees, and no tuning required, RaBitQ makes PQ obsolete for serious applications.

---

## Implementation Availability

- **Code**: https://github.com/gaoj0017/RaBitQ
- **Integration**: Compatible with Faiss, easy to integrate
- **Production-ready**: Extensively tested on 6 real-world datasets, 2B+ vectors

---

## References

**RaBitQ**: Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024

**PQ**: Jégou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

**Theoretical Foundation**: Alon & Klartag, "Optimal Compression of Approximate Inner Products and Dimension Reduction", FOCS 2017
