# SAQ: Major Improvements Over RaBitQ

**Paper**: "SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation" (SIGMOD 2025)

---

## Quick Summary

SAQ is a new vector quantization method that **dramatically improves upon RaBitQ** while keeping its theoretical guarantees. Think of it as RaBitQ 2.0 - it solves all of RaBitQ's major pain points:

- **80x faster** at building the index (quantizing vectors)
- **80% lower error** at the same compression rate
- **12.5x faster** at searching while maintaining the same accuracy

---

## The 3 Big Problems with RaBitQ (That SAQ Fixes)

### Problem 1: RaBitQ is PAINFULLY Slow to Build Indexes

**What's the issue?**
- To quantize 1 billion vectors with 3,072 dimensions using 9 bits per dimension
- RaBitQ takes **over 3,600 CPU hours** (that's 150 days on a single CPU!)
- Why? It uses an exponentially complex algorithm: O(2^B · D log D)
- For every vector, it has to search through 2^B combinations to find the best codeword

**Real-world impact:**
Imagine you work at a company with billions of embeddings (like OpenAI, Google, etc.). With RaBitQ, building your search index could take **weeks** even on powerful servers. This makes it impractical for production use.

**How SAQ fixes it:**
- SAQ uses **Code Adjustment Quantization (CAQ)** with O(D) complexity
- Same dataset: **~26 seconds** instead of 3,600 hours
- **80x speedup** at 9 bits per dimension
- The key insight: RaBitQ's unit-norm constraint is unnecessary! The norm cancels out in the distance formula.

**Simple analogy:**
- **RaBitQ**: Like trying every combination of a 512-digit lock to find the right code
- **SAQ**: Start with a good guess, then tweak each digit one by one to improve it

---

### Problem 2: RaBitQ Wastes Bits on Unimportant Dimensions

**What's the issue?**
- RaBitQ gives **the same number of bits to every dimension** (e.g., 5 bits per dimension for all 1,024 dimensions)
- But after PCA projection, dimensions have **very different importance**:
  - First 100 dimensions: High variance (contain most of the signal)
  - Last 500 dimensions: Near-zero variance (mostly noise)
- Giving the same bits to all dimensions = wasting bits!

**Simple analogy:**
Imagine storing a photo where:
- Top half has lots of detail (faces, text)
- Bottom half is just a plain blue sky

RaBitQ uses the same resolution (bits) for both halves. That's wasteful!

**How SAQ fixes it:**
- **Dimension Segmentation**: Split dimensions into segments and give each segment different bit budgets
- Example allocation:
  ```
  Dimensions 1-64:    8 bits each (important stuff)
  Dimensions 65-256:  5 bits each (medium importance)
  Dimensions 257-512: 3 bits each (less important)
  Dimensions 513+:    1 bit each (mostly noise)
  ```
- Uses **dynamic programming** to find optimal allocation in under 1 second

**Results:**
- At the same bit budget, SAQ achieves **2-5x lower quantization error**
- Or equivalently: Get RaBitQ's accuracy using **37.5% fewer bits**

---

### Problem 3: RaBitQ Can't Do Fine-Grained Progressive Search

**What's the issue?**
- RaBitQ can only do progressive approximation with **1 bit at a time**
- If you have 8-bit codes, you can only use: 1 bit, 2 bits, 3 bits... progressively
- You **cannot** take an arbitrary subset like "use the 4 most significant bits from my 8-bit code"
- Why? The unit-norm constraint means partial codes don't form valid quantized vectors

**Why this matters:**
Progressive search is crucial for speed. You want to:
1. Use 1-2 bits to quickly eliminate obviously bad candidates
2. Use 4 bits to refine your top candidates
3. Use full 8 bits only for final ranking

With RaBitQ, you're stuck with rigid 1-bit increments.

**How SAQ fixes it:**
- Because SAQ removes the unit-norm constraint, you can **sample any number of bits** from a B-bit code
- Take the first 4 bits from an 8-bit code → it's still a valid quantized vector!
- Enables **flexible multi-stage search** strategies

**Example workflow:**
```
Stage 1: Use 2 bits → Prune 90% of candidates (super fast)
Stage 2: Use 4 bits → Refine remaining 10% (still fast)
Stage 3: Use 8 bits → Final ranking of top 1% (precise)
```

---

## Bonus Innovation: Multi-Stage Distance Estimation

SAQ adds another clever trick on top of dimension segmentation:

**The idea:**
Process dimension segments in order of **decreasing importance** (highest variance first):

```
Stage 1: Look at Segment 0 (highest variance, 8 bits)
         → Compute partial distance
         → If it's already worse than current best: STOP, prune this candidate!

Stage 2: Add Segment 1 (5 bits)
         → Refine distance estimate
         → If still worse: STOP!

Stage 3: Add remaining segments
         → Final distance
```

**Why it works:**
- High-variance dimensions give you the **most signal about distance**
- If a candidate looks bad after checking the important dimensions, no need to check the rest!
- On average, access **2.6-3.8x fewer bits** per comparison while maintaining accuracy

**Simple analogy:**
When judging if two houses are similar:
- First check: Neighborhood, size, price (most important)
- If already very different → Don't bother checking carpet color and doorknob style!

---

## Performance Comparison: The Numbers

### Indexing Speed (How Fast Can You Build the Index?)

| Dataset | Dimensions | Bits | RaBitQ Time | SAQ Time | Speedup |
|---------|------------|------|-------------|----------|---------|
| DEEP | 256 | 4 | 3.9s | 0.7s | **5.9x** |
| DEEP | 256 | 8 | 21.5s | 0.7s | **32x** |
| GIST | 960 | 9 | 165.9s | 2.0s | **85x** |
| MSMARCO | 1024 | 9 | 1773.6s | 26.1s | **68x** |

**Key insight:** As you increase bits (B), RaBitQ becomes exponentially slower, while SAQ stays fast!

---

### Search Speed (Queries Per Second at 95% Recall)

| Dataset | Bits | RaBitQ QPS | SAQ QPS | Improvement |
|---------|------|------------|---------|-------------|
| MSMARCO | 3 | 235 | 3,427 | **14.6x faster** |
| MSMARCO | 5 | 1,958 | 3,726 | **1.9x faster** |
| OpenAI-1536 | 3 | 1,338 | 2,812 | **2.1x faster** |
| OpenAI-1536 | 5 | 2,329 | 3,112 | **1.3x faster** |

**When does SAQ shine?**
- **High-dimensional datasets** (MSMARCO: 1024-D, OpenAI: 1536-D): SAQ dominates!
- **Low-dimensional datasets** (DEEP: 256-D): Similar performance (SAQ's overhead not worth it)

**Why?** Dimension segmentation has more room to optimize when there are more dimensions to segment!

---

### Quantization Accuracy (Lower is Better)

At the same bit budget, SAQ achieves much lower error:

| Dataset | RaBitQ Error | SAQ Error | Improvement |
|---------|--------------|-----------|-------------|
| DEEP (B=4) | 0.51% | 0.27% | **1.9x better** |
| GIST (B=4) | 0.28% | 0.05% | **5.6x better** |
| MSMARCO (B=4) | 0.28% | 0.10% | **2.8x better** |
| OpenAI-1536 (B=4) | 0.23% | 0.09% | **2.6x better** |

**Or flip it around:** SAQ with **5 bits** achieves the same accuracy as RaBitQ with **8 bits**
- That's **37.5% fewer bits** for same accuracy!
- Means **37.5% less storage** for your billion-vector database

---

## Memory Efficiency

**Example:** 1 billion OpenAI embeddings (1536 dimensions)

| Method | Bits/dim | Total Storage | Compression |
|--------|----------|---------------|-------------|
| Full precision (float32) | 32 | ~6 TB | 1x |
| RaBitQ | 8 | 1.5 TB | 4x |
| RaBitQ | 5 | 937 GB | 6.4x |
| SAQ | 5 | 937 GB | 6.4x (same storage, **better accuracy**) |
| SAQ (optimized) | 3-8 (variable) | ~800 GB | 7.5x (better than uniform 5!) |

SAQ's dimension segmentation means it can achieve **higher compression with better accuracy** through smart bit allocation.

---

## Summary: Why SAQ is a Game-Changer

### What SAQ Keeps from RaBitQ:
✅ Theoretical guarantees (error bounds)
✅ High accuracy compared to other methods
✅ Support for arbitrary bit budgets

### What SAQ Fixes:
✅ **80x faster indexing** - practical for billion-scale datasets
✅ **2-5x lower error** - or use 37% fewer bits for same accuracy
✅ **Flexible bit allocation** - spend bits where they matter most
✅ **Progressive approximation** - arbitrary bit sampling for multi-stage search
✅ **Multi-stage pruning** - process important dimensions first, stop early

### Bottom Line:
RaBitQ was theoretically excellent but practically limited by:
- Exponential indexing time (impractical for B≥9)
- Rigid bit allocation (wastes bits on unimportant dimensions)
- Limited progressive search (only 1-bit steps)

**SAQ solves all three problems while maintaining RaBitQ's theoretical guarantees.**

For production systems with billions of high-dimensional vectors, SAQ is what makes RaBitQ's approach actually usable.

---

## When to Use What?

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Small dataset (<1M vectors) | Any method works | Speed differences negligible |
| Low-dimensional (D<300) | RaBitQ or SAQ | Similar performance |
| High-dimensional (D>1000) | **SAQ** | Dimension segmentation shines |
| Need fast indexing | **SAQ** | 80x faster than RaBitQ |
| Tight bit budget | **SAQ** | Better accuracy per bit |
| Production billion-scale | **SAQ** | Only practical option |
| Research/comparison | RaBitQ | Baseline with theoretical guarantees |

---

## Key Takeaway

If you're already using RaBitQ: **Switch to SAQ**. It's strictly better in almost every way.

If you're choosing a quantization method: SAQ offers the best combination of:
- Theoretical soundness (inherited from RaBitQ)
- Practical efficiency (80x faster indexing)
- Accuracy (2-5x lower error)
- Flexibility (progressive search, adaptive bit allocation)
