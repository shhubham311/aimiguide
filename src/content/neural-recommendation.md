# Neural Network Based Recommendation: The Two-Tower Approach

## Why Neural Recommendations Matter

Traditional collaborative filtering using matrix factorization works well for explicit feedback (ratings) but struggles with implicit signals (clicks, views, dwell time). At companies like YouTube, Pinterest, and TikTok, billions of interactions must be processed to serve personalized recommendations in under 200ms. **Neural recommendation models**—especially the **Two-Tower architecture**—solve this by learning dense embeddings from rich feature inputs and enabling efficient retrieval over millions of items.

---

## The Two-Tower Model Architecture

### Core Concept: Separate User and Item Towers

```
┌──────────────────────────────────────────────────────────────┐
│                    Two-Tower Architecture                      │
│                                                               │
│  User Tower                        Item Tower                 │
│  ──────────                        ─────────                  │
│                                                               │
│  ┌──────────────┐                 ┌──────────────┐           │
│  │ User ID      │                 │ Item ID      │           │
│  │ (embedding)  │                 │ (embedding)  │           │
│  └──────┬───────┘                 └──────┬───────┘           │
│         │                                │                   │
│  ┌──────┴───────┐                 ┌──────┴───────┐           │
│  │ Age, Gender  │                 │ Category     │           │
│  │ (embedding)  │                 │ (embedding)  │           │
│  └──────┬───────┘                 └──────┬───────┘           │
│         │                                │                   │
│  ┌──────┴───────┐                 ┌──────┴───────┐           │
│  │ Past clicks  │                 │ Title text   │           │
│  │ (avg embed)  │                 │ (BERT embed) │           │
│  └──────┬───────┘                 └──────┬───────┘           │
│         │                                │                   │
│  ┌──────┴───────┐                 ┌──────┴───────┐           │
│  │ Concatenate   │                 │ Concatenate   │           │
│  │ + Dense       │                 │ + Dense       │           │
│  │ + ReLU        │                 │ + ReLU        │           │
│  └──────┬───────┘                 └──────┬───────┘           │
│         │                                │                   │
│  ┌──────┴───────┐                 ┌──────┴───────┐           │
│  │ Dense(128)   │                 │ Dense(128)   │           │
│  │ + ReLU       │                 │ + ReLU       │           │
│  └──────┬───────┘                 └──────┬───────┘           │
│         │                                │                   │
│  ┌──────┴───────┐                 ┌──────┴───────┐           │
│  │ L2 Normalize │                 │ L2 Normalize │           │
│  │ (128-dim)    │                 │ (128-dim)    │           │
│  └──────┬───────┘                 └──────┬───────┘           │
│         │                                │                   │
│         │    User Embedding u    Item Embedding v            │
│         │         │                    │                    │
│         └─────────┼────────────────────┘                    │
│                   │                                          │
│                   ▼                                          │
│          ┌────────────────┐                                 │
│          │  Dot Product   │                                 │
│          │  score = u · v │                                 │
│          └────────────────┘                                 │
│                                                               │
│  Training: In-batch softmax (softmax over dot products)      │
│  Serving:  Approximate Nearest Neighbor (ANN) search         │
└──────────────────────────────────────────────────────────────┘
```

### Why L2 Normalize?

After L2 normalization, dot product equals cosine similarity:

```
u_norm = u / ||u||,  v_norm = v / ||v||

u_norm · v_norm = (u · v) / (||u|| × ||v||) = cosine_similarity(u, v)

This bounds scores in [-1, 1] and makes embeddings comparable across users.
```

---

## Training with In-Batch Softmax

### The Loss Function

```
L = -log(exp(u · v_pos) / Σ_j exp(u · v_j))

Where:
  u       = User embedding for the target user
  v_pos   = Item embedding for the positive (clicked) item
  v_j     = Item embeddings for ALL items in the batch
  Σ_j     = Sum over all items in the batch (including negative samples)
```

```
Batch of 4 training examples:
┌─────────────────────────────────────────────────┐
│  (u₁, v₁+)  u₁ liked item v₁                  │
│  (u₂, v₂+)  u₂ liked item v₂                  │
│  (u₃, v₃+)  u₃ liked item v₃                  │
│  (u₄, v₄+)  u₄ liked item v₄                  │
│                                                   │
│  For user u₁:                                    │
│  Positive: u₁ · v₁ (clicked)                    │
│  Negatives: u₁ · v₂, u₁ · v₃, u₁ · v₄ (not     │
│  clicked by u₁—served as in-batch negatives)    │
│                                                   │
│  Loss = -log(softmax([u₁·v₁, u₁·v₂, u₁·v₃,    │
│                        u₁·v₄])₀)               │
└─────────────────────────────────────────────────┘
```

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, n_categories,
                 embed_dim=128, dense_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        # ── User Tower ──
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.age_embed = nn.Embedding(100, embed_dim)       # Bucketized age
        self.user_dense = nn.Sequential(
            nn.Linear(embed_dim * 2, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim),
        )

        # ── Item Tower ──
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.category_embed = nn.Embedding(n_categories, embed_dim)
        self.item_dense = nn.Sequential(
            nn.Linear(embed_dim * 2, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim),
        )

    def user_tower(self, user_id, age):
        u = self.user_embed(user_id)
        a = self.age_embed(age)
        u = torch.cat([u, a], dim=-1)
        u = self.user_dense(u)
        u = F.normalize(u, p=2, dim=-1)  # L2 normalize
        return u

    def item_tower(self, item_id, category):
        v = self.item_embed(item_id)
        c = self.category_embed(category)
        v = torch.cat([v, c], dim=-1)
        v = self.item_dense(v)
        v = F.normalize(v, p=2, dim=-1)  # L2 normalize
        return v

    def forward(self, user_ids, ages, item_ids, categories):
        user_embeddings = self.user_tower(user_ids, ages)
        item_embeddings = self.item_tower(item_ids, categories)
        scores = torch.sum(user_embeddings * item_embeddings, dim=-1)
        return scores

    def compute_loss(self, user_ids, ages, item_ids, categories):
        """In-batch softmax loss."""
        batch_size = user_ids.size(0)

        # Compute all user and item embeddings
        user_emb = self.user_tower(user_ids, ages)       # (B, D)
        item_emb = self.item_tower(item_ids, categories) # (B, D)

        # Compute full score matrix: (B, B)
        # Each user against all items in batch
        score_matrix = user_emb @ item_emb.T             # (B, B)

        # Labels: diagonal = positive pairs
        labels = torch.arange(batch_size, device=user_ids.device)

        # Cross-entropy with in-batch negatives
        loss = F.cross_entropy(score_matrix, labels)

        # Temperature scaling (optional)
        temperature = 0.05
        loss_scaled = F.cross_entropy(score_matrix / temperature, labels)

        return loss_scaled


# Training
model = TwoTowerModel(n_users=10000, n_items=50000, n_categories=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy batch
user_ids = torch.randint(0, 10000, (256,))
ages = torch.randint(18, 80, (256,))
item_ids = torch.randint(0, 50000, (256,))
categories = torch.randint(0, 100, (256,))

# Training step
optimizer.zero_grad()
loss = model.compute_loss(user_ids, ages, item_ids, categories)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```

---

## Serving with Approximate Nearest Neighbor (ANN)

### The Bottleneck Problem

```
Naive approach: Compute dot product with ALL items
  - 10M items × 128-dim × each request = TOO SLOW

ANN approach: Pre-compute item embeddings, use index for fast search
  - Build FAISS index of all item embeddings
  - Query with user embedding → top-K in ~1ms
```

### FAISS Integration

```python
import faiss
import numpy as np

# Pre-compute ALL item embeddings (offline)
# In production, do this after training and periodically recompute
all_item_ids = torch.arange(50000)
all_categories = torch.randint(0, 100, (50000,))

with torch.no_grad():
    item_embeddings = model.item_tower(all_item_ids, all_categories).numpy()
    # Shape: (50000, 128)

# Build FAISS index
dimension = item_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)       # Inner Product (= cosine after L2 norm)
index.add(item_embeddings.astype('float32'))

# Query: recommend for a user
user_id = torch.tensor([42])
age = torch.tensor([30])
with torch.no_grad():
    user_embedding = model.user_tower(user_id, age).numpy()

# Search top-K
k = 10
scores, indices = index.search(user_embedding.astype('float32'), k)

print(f"Top-{k} recommended items: {indices[0]}")
print(f"Scores: {scores[0]}")
```

### FAISS Index Types

```
┌────────────────────────────────────────────────────────────┐
│  FAISS Index Selection                                       │
│                                                              │
│  Index Type         Speed    Accuracy   Memory    Best For  │
│  ────────────────────────────────────────────────────────   │
│  FlatL2/FlatIP     Slow     Exact      High      < 1M items│
│  IVFFlat           Fast     ~95%       Medium    1-10M     │
│  IVFPQ             Faster   ~90%       Low       10M-100M  │
│  HNSW              Fast     ~99%       High      < 10M     │
│  HNSWPQ            Faster   ~95%       Medium    > 10M     │
│                                                              │
│  Rule of thumb:                                               │
│  - < 1M items: HNSW (best accuracy-speed tradeoff)          │
│  - 1M-100M: IVF + PQ (good compression)                     │
│  - > 100M: IVFPQ or HNSWPQ + GPU                           │
└────────────────────────────────────────────────────────────┘
```

---

## Advanced: Multi-Task Two-Tower

Real systems often predict multiple objectives simultaneously:

```python
class MultiTaskTwoTower(TwoTowerModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional task heads
        self.click_head = nn.Linear(self.embed_dim, 1)    # Click prediction
        self.like_head = nn.Linear(self.embed_dim, 1)     # Like prediction
        self.share_head = nn.Linear(self.embed_dim, 1)    # Share prediction

    def forward_multi_task(self, user_ids, ages, item_ids, categories):
        user_emb = self.user_tower(user_ids, ages)
        item_emb = self.item_tower(item_ids, categories)

        # Combine user and item embeddings
        combined = user_emb * item_emb  # Element-wise (Hadamard product)

        click_prob = torch.sigmoid(self.click_head(combined))
        like_prob = torch.sigmoid(self.like_head(combined))
        share_prob = torch.sigmoid(self.share_head(combined))

        return click_prob, like_prob, share_prob
```

---

## Exercises

### Exercise 1: Compute Two-Tower Score
Given user embedding u = [0.6, 0.8] and item embedding v = [0.3, 0.4], compute the score after L2 normalization.

**Solution:**
```
||u|| = √(0.36 + 0.64) = √1.0 = 1.0
||v|| = √(0.09 + 0.16) = √0.25 = 0.5

u_norm = [0.6, 0.8] (already normalized)
v_norm = [0.6, 0.8] (0.3/0.5, 0.4/0.5)

score = u_norm · v_norm = 0.6×0.6 + 0.8×0.8 = 0.36 + 0.64 = 1.0

Perfect match! v is a scaled version of u.
```

### Exercise 2: In-Batch Softmax Loss
Batch of 3: scores for user u₁ are [0.8, 0.2, 0.1] (v₁ is positive). Compute loss.

**Solution:**
```
Temperature τ = 0.05

Scaled scores = [0.8/0.05, 0.2/0.05, 0.1/0.05] = [16, 4, 2]

Softmax:
exp(16) + exp(4) + exp(2) ≈ 8886111 + 54.6 + 7.4 ≈ 8886173

P(v₁|u₁) = exp(16) / 8886173 ≈ 0.999993

Loss = -log(0.999993) ≈ 0.000007

Very low loss → model correctly identifies the positive item with high confidence.
```

### Exercise 3: FAISS Throughput
You have 10M items with 128-dim embeddings. How long does a HNSW search take vs brute force?

**Solution:**
```
Brute force (FlatIP): 10M × 128 × 2 FLOPs ≈ 2.56 GFLOP ≈ ~50ms on CPU

HNSW (M=32, ef=128): Visits ~500-2000 of 10M items ≈ ~1-2ms on CPU

HNSW is ~25-50× faster with > 99% recall for this scale.
```

---

## Key Takeaways

1. **Two-tower models** learn separate user and item embeddings that can be pre-computed for fast serving
2. **L2 normalization** converts dot product to cosine similarity, bounding scores in [-1, 1]
3. **In-batch softmax** provides efficient negative sampling during training
4. **FAISS** enables sub-millisecond retrieval over millions of items using approximate nearest neighbor search
5. **Multi-task learning** extends two-tower to predict clicks, likes, shares, and dwell time simultaneously
