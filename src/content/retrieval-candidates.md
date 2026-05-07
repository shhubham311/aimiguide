# Retrieval & Candidate Set Generation

## Why Retrieval Matters

A production recommendation system at Netflix serves 200M+ users from a catalog of 17,000+ titles. At TikTok, the catalog exceeds billions of videos. **Retrieving candidates from millions of items in under 50ms** is one of the hardest engineering challenges in recommendation systems. The retrieval stage narrows the full catalog (millions) down to a manageable candidate set (hundreds-thousands) that the ranking stage can then score precisely.

```
┌──────────────────────────────────────────────────────────────────┐
│              The Recommendation Funnel                            │
│                                                                   │
│  Full Catalog          Candidates           Ranked               │
│  (10M items)      →   (~1000 items)    →   (Top 10)             │
│                                                                   │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐             │
│  │  RETRIEVAL  │    │  RANKING   │    │  RE-RANK   │             │
│  │            │    │            │    │            │             │
│  │  Fast &    │    │  Precise   │    │  Business  │             │
│  │  Scalable  │───►│  Scoring   │───►│  Rules     │             │
│  │  ~50ms     │    │  ~100ms    │    │  ~10ms     │             │
│  └────────────┘    └────────────┘    └────────────┘             │
│                                                                   │
│  Goal: Recall      Goal: Precision    Goal: Diversity,          │
│  Don't miss        Score accurately   freshness,                │
│  good items                          business goals             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Strategies

### 1. Embedding-Based Retrieval (Two-Tower + ANN)

```
┌─────────────────────────────────────────────────────────┐
│  Embedding-Based Retrieval Pipeline                      │
│                                                          │
│  Offline (periodic):                                     │
│  ┌──────────────┐                                       │
│  │ Train Two-   │                                       │
│  │ Tower Model  │                                       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │ Compute ALL  │────►│ Build FAISS  │                  │
│  │ Item Embeds  │     │ Index        │                  │
│  └──────────────┘     └──────────────┘                  │
│                                                          │
│  Online (per request):                                   │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │ Compute User │────►│ FAISS Search │──► Top-500       │
│  │ Embedding    │     │ ~1-5ms       │    candidates   │
│  └──────────────┘     └──────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

### 2. Multi-Channel Retrieval

Real systems use **multiple retrieval channels** and merge results:

```
┌─────────────────────────────────────────────┐
│  Multi-Channel Retrieval                     │
│                                               │
│  Channel 1: Two-Tower (ANN)       ─┐         │
│  Channel 2: Graph-based           ─┤         │
│  Channel 3: Content-based (TF-IDF)─┤──► Merge │
│  Channel 4: Trending/Popular      ─┤   &     │
│  Channel 5: Recently published    ─┤ Dedup   │
│  Channel 6: User's follow graph   ─┘         │
│                                               │
│  → Combined candidate set (~1000-5000)       │
│  → Pass to ranking stage                     │
└─────────────────────────────────────────────┘
```

### 3. Graph-Based Retrieval (Item-to-Item)

```
User watches "The Matrix"
       │
       ▼
Item-Item Graph (co-occurrence):
  "The Matrix" ──► "Inception" (weight: 0.8)
  "The Matrix" ──► "Interstellar" (weight: 0.7)
  "The Matrix" ──► "Blade Runner" (weight: 0.6)
  "The Matrix" ──► "Ex Machina" (weight: 0.4)
       │
       ▼
  Candidates: [Inception, Interstellar, Blade Runner, Ex Machina]
```

---

## Candidate Generation with FAISS

```python
import numpy as np
import faiss
import time

class CandidateRetriever:
    def __init__(self, embedding_dim=128, index_type="HNSW"):
        self.dim = embedding_dim

        # Choose index type based on catalog size
        if index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(embedding_dim, M=32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
        elif index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlists = 4096
            nsubquantizers = 64
            nbits = 8
            self.index = faiss.IndexIVFPQ(quantizer, embedding_dim,
                                           nlists, nsubquantizers, nbits)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def build_index(self, item_embeddings, item_ids=None):
        """Build the FAISS index from item embeddings."""
        # Ensure L2-normalized (for cosine similarity via inner product)
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        normalized = item_embeddings / (norms + 1e-10)

        if isinstance(self.index, faiss.IndexIVFPQ):
            # Need to train IVF index first
            self.index.train(normalized.astype('float32'))

        self.index.add(normalized.astype('float32'))

        if item_ids is not None:
            self.item_ids = item_ids
        else:
            self.item_ids = np.arange(len(item_embeddings))

        print(f"Index built with {self.index.ntotal} items")

    def retrieve(self, user_embedding, top_k=500):
        """Retrieve top-K candidates for a user."""
        # Normalize user embedding
        user_emb = user_embedding / (np.linalg.norm(user_embedding) + 1e-10)
        user_emb = user_emb.reshape(1, -1).astype('float32')

        start = time.time()
        scores, indices = self.index.search(user_emb, top_k)
        latency = (time.time() - start) * 1000

        # Map back to item IDs
        candidate_ids = self.item_ids[indices[0]]
        candidate_scores = scores[0]

        return list(zip(candidate_ids, candidate_scores)), latency

    def batch_retrieve(self, user_embeddings, top_k=500):
        """Retrieve for multiple users simultaneously."""
        norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        normalized = user_embeddings / (norms + 1e-10)

        scores, indices = self.index.search(normalized.astype('float32'), top_k)
        return indices, scores


# Usage
retriever = CandidateRetriever(embedding_dim=128, index_type="HNSW")

# Simulate 10M item catalog
item_embeddings = np.random.randn(10_000_000, 128).astype('float32')
retriever.build_index(item_embeddings)

# Retrieve for a single user
user_emb = np.random.randn(128).astype('float32')
candidates, latency = retriever.retrieve(user_emb, top_k=500)
print(f"Retrieved {len(candidates)} candidates in {latency:.2f}ms")
```

---

## Approximate Nearest Neighbor (ANN) Deep Dive

### HNSW (Hierarchical Navigable Small World)

```
┌────────────────────────────────────────────────────────────┐
│  HNSW Index Structure                                       │
│                                                              │
│  Layer 2 (sparse):     A ──────────────────────── E        │
│                        │                                    │
│  Layer 1 (medium):     A ── B ── C ── D ── E              │
│                        │    │              │                │
│  Layer 0 (dense):     A ── B ── C ── D ── E              │
│                     / │ \ / │ \ / │ \ / │ \                │
│                    F  G  H  I  J  K  L  M  N                │
│                                                              │
│  Search starts at top layer, greedily moves toward target, │
│  then drops to lower layer to refine.                      │
│                                                              │
│  M = max connections per node (32 typical)                  │
│  efSearch = beam width during search (64-256)               │
└────────────────────────────────────────────────────────────┘
```

### Recall vs Speed Tradeoff

```
Recall@100
    │
1.0├───────────── HNSW (efSearch=256)
   │          ╱
0.95├───────╱─── HNSW (efSearch=128)
   │      ╱     IVF (nprobe=256)
0.90├────╱──────── IVF (nprobe=128)
   │   ╱       ╱
0.85├──╱──────╱─── IVF (nprobe=32)
   │ ╱      ╱    IVF+PQ (nprobe=32)
0.80├╱──────╱─────
   │      ╱
0.75├─────╱────────── IVF+PQ (nprobe=8)
   │    ╱
   └───┴────┬────┬────┬────┬────► Query time (ms)
           0.5   1    2    5

    Tradeoff: Higher recall = slower queries
    Target: > 95% recall at < 5ms for production
```

---

## Real-World Retrieval Systems

### YouTube's Candidate Generation (2016)

```
┌─────────────────────────────────────────────────┐
│  YouTube Candidate Generation                    │
│                                                   │
│  Search query (or user context)                  │
│       │                                           │
│       ▼                                           │
│  ┌─────────────────────────────────┐             │
│  │  Retriever (Softmax over        │             │
│  │  hundreds of thousands)         │             │
│  │                                  │             │
│  │  Features:                       │             │
│  │  - Search query embedding        │             │
│  │  - User watch history            │             │
│  │  - Geographic region             │             │
│  │  - Age, gender                   │             │
│  └─────────────┬───────────────────┘             │
│                │                                  │
│                ▼                                  │
│  Top ~500 candidates                             │
│                │                                  │
│                ▼                                  │
│  ┌─────────────────────────────────┐             │
│  │  Ranking Model                  │             │
│  │  (More features, deeper model)  │             │
│  │  Score → Top 10                 │             │
│  └─────────────────────────────────┘             │
└─────────────────────────────────────────────────┘
```

---

## Exercises

### Exercise 1: Design a Retrieval Strategy
Design the retrieval stage for a food delivery app with 50K restaurants and 2M users.

**Solution:**
```
Multi-channel retrieval:

Channel 1: Two-Tower ANN
  - User features: location, cuisine preferences, price range, past orders
  - Restaurant features: cuisine, price range, rating, location
  - FAISS HNSW index over 50K restaurants

Channel 2: Location-based
  - Geo-distance filter (restaurants within delivery radius)
  - Pre-filter candidates before scoring

Channel 3: Popularity + Recency
  - Recently trending restaurants
  - Currently open restaurants (hours filter)

Channel 4: User's past favorites
  - Restaurants ordered 2+ times
  - Similar cuisine to favorites

Merge channels → Remove duplicates → Pass top 500 to ranking
```

### Exercise 2: Estimate FAISS Memory
You have 10M items with 128-dim float32 embeddings using HNSW. How much memory?

**Solution:**
```
Raw embeddings: 10M × 128 × 4 bytes = 5.12 GB

HNSW overhead: ~2× for graph connections (M=32)
Total: ~10-15 GB

With PQ compression (64 subquantizers, 8 bits):
10M × 64 × 1 byte = 0.64 GB
Plus quantizer centroids: 64 × 128 × 4 = 0.03 MB
Total: ~0.64 GB (8× reduction!)
```

### Exercise 3: Recall Computation
A retrieval system returns 500 candidates. The user's relevant items in the full catalog are [A, B, C, D, E]. The system returns [A, B, F, G, H, ...]. What's the recall@500?

**Solution:**
```
Recall@500 = |Relevant items retrieved| / |Total relevant items|
            = |{A, B}| / |{A, B, C, D, E}|
            = 2 / 5 = 0.4 (40%)

Good retrieval systems target recall@500 > 0.90 (90%)
```

---

## Key Takeaways

1. **Retrieval** narrows millions of items to hundreds using fast, approximate methods (~5-50ms)
2. **Multi-channel retrieval** combines embedding search, content matching, popularity, and user history
3. **FAISS** is the standard tool for ANN search—HNSW for accuracy, IVF+PQ for memory efficiency
4. **The recall-speed tradeoff** is the fundamental challenge: higher recall requires more computation
5. **Pre-filtering** (geography, availability) reduces the effective catalog size before retrieval
