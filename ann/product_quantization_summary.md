# Product Quantization vs No Quantization: Summary

Based on "Product quantization for nearest neighbor search" by Jégou et al., INRIA

## Overview

Product quantization is a compact coding technique for approximate nearest neighbor (ANN) search in high-dimensional spaces. This document compares it against traditional exhaustive search without quantization.

---

## Product Quantization Approach

### Key Concept
- **Decomposes** D-dimensional vectors into m subspaces of dimension D* = D/m
- **Quantizes** each subspace separately using distinct quantizers
- **Represents** vectors as short codes (e.g., 64 bits) composed of subspace quantization indices
- **Estimates** Euclidean distances from codes using lookup tables

### Memory Usage
For 128-dimensional SIFT descriptors:
- **With quantization (64-bit codes)**: 8 bytes per vector
- **Storage reduction**: ~16x compared to full vectors (128 × 4 bytes = 512 bytes for float32)
- **Codebook size**: m × k* × D* floats (e.g., 8 × 256 × 16 = 32,768 floats ≈ 128KB)

### Distance Computation
- **Asymmetric Distance Computation (ADC)**: Query not quantized, database vectors quantized
  - Pre-compute distances from query subvectors to all subquantizer centroids
  - For each database vector: m lookup operations + m additions
  - **Complexity per comparison**: O(m) vs O(D) for exact

- **Symmetric Distance Computation (SDC)**: Both query and database quantized
  - Even faster but less accurate
  - Uses pre-computed distance tables between centroids

### Search Quality
- **Approximate search**: Some accuracy loss due to quantization error
- **SIFT dataset (64-bit codes, m=8, k*=256)**:
  - ADC achieves ~65% recall@100
  - Can reach >90% recall with longer codes (128 bits)
- **Mean Squared Distance Error (MSDE)**: Bounded by quantization error MSE(q)

### Scalability
- **Tested on**: 2 billion SIFT descriptors
- **Combined with inverted file (IVFADC)**:
  - Coarse quantizer (k' centroids) for non-exhaustive search
  - Encodes residual vectors with product quantizer
  - Search only w nearest clusters (w << k')
  - **Search time**: ~3-10ms per query on 1M vectors (vs 17ms exhaustive ADC)

### Advantages
1. **Memory efficient**: Enables indexing billions of vectors
2. **Fast distance estimation**: Lookup table operations
3. **Flexible trade-offs**: Adjust m and k* for memory/accuracy balance
4. **Scalable**: Non-exhaustive search with inverted files
5. **Better than binary codes**: More distinct distance values than Hamming embedding

---

## No Quantization (Exhaustive Exact Search)

### Approach
- Store full D-dimensional vectors (typically float32 or float64)
- Compute exact Euclidean distances between query and all database vectors
- Return k nearest neighbors with perfect accuracy

### Memory Usage
For 128-dimensional SIFT descriptors:
- **Per vector**: 128 × 4 bytes = 512 bytes (float32)
- **1M vectors**: ~512 MB
- **1B vectors**: ~512 GB (requires distributed storage or external memory)

### Distance Computation
- **Complexity**: O(n×D) for n database vectors of dimension D
- **SIFT (D=128)**: 128 multiply-add operations per vector comparison
- **No pre-computation**: Each distance calculated from scratch

### Search Quality
- **Perfect accuracy**: Always finds true nearest neighbors (recall@1 = 100%)
- **No approximation error**: Exact Euclidean distances

### Scalability Limitations
- **Memory bottleneck**: Cannot fit large datasets in RAM
  - 1M SIFT vectors: 512 MB (feasible)
  - 100M SIFT vectors: 51.2 GB (challenging)
  - 1B+ vectors: Requires external storage with severe I/O overhead

- **Computational cost**: Linear in dataset size
  - 1M vectors × 128 dims = 128M operations per query
  - Impractical for real-time search on large datasets

### Limitations of Traditional Indexing
- **KD-trees, branch-and-bound**: Degrade to exhaustive search in high dimensions (curse of dimensionality)
- **Memory overhead**: Indexing structures may exceed vector storage
- **Re-ranking requirement**: Methods like LSH and FLANN need full vectors in RAM for final distance verification

---

## Direct Comparison

| Aspect | Product Quantization | No Quantization |
|--------|---------------------|-----------------|
| **Memory per vector** | 8 bytes (64-bit) | 512 bytes (128D float32) |
| **Memory reduction** | **16x** | 1x (baseline) |
| **Distance computation** | m lookups (~8 ops) | D mult-adds (128 ops) |
| **Search complexity** | O(n×m) or O((n/k')×m) with IVF | O(n×D) |
| **Accuracy** | ~65-90% recall (tunable) | 100% (exact) |
| **Max practical scale** | **Billions** of vectors | Millions of vectors |
| **Search time (1M SIFT)** | 8.8-17ms (ADC/IVFADC) | Not reported (~100ms+ estimated) |
| **RAM requirement (1B vectors)** | ~8 GB codes + codebooks | ~512 GB (infeasible) |

---

## Performance Results (From Paper)

### SIFT Dataset (1M vectors, 10K queries)
- **Product Quantization (ADC, 64-bit)**:
  - Recall@100: 65.2%
  - Search time: 17.2 ms/query
  - Memory: 8 MB for codes

- **Product Quantization (IVFADC, 64-bit, k'=1024, w=8)**:
  - Recall@100: 68.2%
  - Search time: 8.8 ms/query
  - Compares only ~2.8% of database

- **Exhaustive exact search** (estimated):
  - Recall@100: 100%
  - Search time: >100 ms/query
  - Memory: 512 MB for vectors

### GIST Dataset (1M vectors, 960 dims, 500 queries)
- **ADC (64-bit)**:
  - Recall@100: 65.2%
  - Search time: 17.2 ms
  - Memory: 8 MB codes

- **Exhaustive exact** (estimated):
  - Memory: 960 × 4 × 1M = 3.84 GB
  - Significantly slower (7.5x more dimensions than SIFT)

### Comparison with State-of-the-Art
Product quantization outperforms:
- **Spectral Hashing** (binary codes): PQ achieves same recall@100 with 10x fewer returned vectors
- **FLANN** (hierarchical indexing): Better quality/time trade-offs at lower memory
- **Hamming Embedding**: Significantly better accuracy for same 64-bit code length

---

## When to Use Each Approach

### Use Product Quantization When:
- Dataset is **very large** (>10M vectors)
- **Memory is constrained**
- **Speed is critical** for real-time applications
- **Approximate results are acceptable** (e.g., image search, recommendation)
- Need to index **billions of vectors** (e.g., web-scale search)

### Use No Quantization (Exact Search) When:
- Dataset is **small** (<1M vectors)
- **Perfect accuracy is required**
- **Sufficient RAM** available
- **Latency is not critical**
- Use cases: scientific computing, medical imaging, high-stakes applications

### Hybrid Approaches:
- **Re-ranking**: Use product quantization for initial retrieval (short-list), then exact distances for top-k refinement
- **Combines**: Speed of quantization with accuracy of exact search for final results
- Used in FLANN, LSH, and demonstrated in paper's IVFADC evaluation

---

## Key Insights

1. **Memory vs Accuracy Trade-off**: Product quantization achieves 16x memory reduction with ~65% recall, enabling billion-scale search impossible with exact methods

2. **Speed Gains**: 8-16x fewer operations per distance + non-exhaustive search (98% reduction in comparisons) = orders of magnitude speedup

3. **Scalability**: The only evaluated method that successfully indexed and searched 2 billion vectors

4. **Asymmetric is Better**: ADC (query not quantized) significantly outperforms SDC (both quantized) for same computational cost

5. **Parameter Selection**: Better to use fewer subquantizers (m) with more centroids (k*) than many subquantizers with few centroids for given bit budget

6. **Practical Impact**: Enables real-time search in applications previously requiring exhaustive search or distributed systems

---

## Conclusion

Product quantization represents a fundamental shift in nearest neighbor search, sacrificing exact accuracy for dramatic improvements in memory efficiency and speed. For modern large-scale applications (billions of vectors), it's not just an optimization—**it's the only feasible approach**. Traditional exact search remains superior only for small datasets where perfect accuracy justifies the memory and computational costs.

The paper's results demonstrate that product quantization achieves the best trade-off between search quality, memory usage, and computational efficiency compared to all evaluated alternatives, making it the state-of-the-art for large-scale approximate nearest neighbor search as of publication (2011).
