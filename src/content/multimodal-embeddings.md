# Multi-modal Representations using Embeddings

## Why Multi-modal Matters

Real-world AI systems don't work with just text OR just images — they work with **everything at once**. A search engine understands text queries and matches them to images. A recommendation system uses text descriptions, images, and user behavior simultaneously.

## What Are Embeddings?

Embeddings are **dense vector representations** of data in a continuous space where **semantic similarity** corresponds to **spatial proximity**.

```
Text: "king"    → [0.50, 0.68, 0.12, ...]  (768 dims)
Text: "queen"   → [0.49, 0.71, 0.15, ...]  (close to king!)
Text: "apple"   → [-0.23, 0.01, 0.89, ...]  (far from king)

Image: 🐕       → [0.82, 0.34, 0.56, ...]  (512 dims)
Image: 🐩       → [0.80, 0.36, 0.52, ...]  (close to 🐕!)
```

## Joint Embedding Spaces

The key insight of multi-modal AI: **map different modalities into the SAME vector space**.

```
┌──────────────────────────────────────────────────┐
│           JOINT EMBEDDING SPACE                  │
│                                                   │
│   Text "a dog playing"  ·                       │
│                        ·  ·                      │
│   Image [dog photo]  ·     ·                     │
│                    ·        ·                     │
│              ·             ·                     │
│   Text "a cat" ·               ·                 │
│                ·  Image [cat photo]              │
│                                                   │
│  Similar concepts → Close vectors                 │
│  Different concepts → Far vectors                │
└──────────────────────────────────────────────────┘
```

## Contrastive Learning

The most popular approach to learn joint embeddings. The idea: **pull matching pairs together, push non-matching pairs apart**.

```
CLIP (Contrastive Language-Image Pre-training):
  - Positive pair: (image, its caption) → similar embeddings
  - Negative pair: (image, random caption) → dissimilar embeddings

Loss: InfoNCE (Normalized Temperature-Scaled Cross-Entropy)

  L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

Where:
  z_i = image embedding, z_j = text embedding
  sim(a,b) = a·b / (||a|| × ||b||)  (cosine similarity)
  τ = temperature parameter (typically 0.07)
```

## CLIP: Connecting Vision and Language

```python
# Using sentence-transformers for text embeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Text embeddings
texts = [
    "A dog playing fetch in the park",
    "A cat sleeping on a couch",
    "Machine learning algorithms",
    "A puppy catching a frisbee",
]

embeddings = model.encode(texts)

# Compute similarity matrix
sim_matrix = util.cos_sim(embeddings, embeddings)
print("Similarity Matrix:")
for i, text in enumerate(texts):
    print(f"  {text[:35]:35s} | {' '.join(f'{s:.2f}' for s in sim_matrix[i][:4])}")
```

## Building a Multi-modal Search System

```python
class MultiModalSearch:
    def __init__(self):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts = []
        self.embeddings = None
    
    def add_documents(self, texts):
        self.texts.extend(texts)
        self.embeddings = self.text_model.encode(self.texts)
    
    def search(self, query, top_k=5):
        query_emb = self.text_model.encode([query])
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)[0]
        return [(self.texts[hit['corpus_id']], hit['score']) for hit in hits]

search = MultiModalSearch()
search.add_documents([
    "Linear regression with gradient descent implementation",
    "Convolutional neural networks for image classification",
    "Natural language processing with transformers",
    "Reinforcement learning for game playing agents",
    "Generative adversarial networks for image synthesis",
    "Recommendation systems using collaborative filtering",
    "Time series forecasting with ARIMA models",
    "Transfer learning with pre-trained models",
])

results = search.search("deep learning for computer vision", top_k=3)
for text, score in results:
    print(f"  [{score:.4f}] {text}")
```

## Cross-Modal Retrieval

```python
# Text-to-Image search (conceptual implementation)
def cross_modal_search(query_text, image_captions, model):
    """
    query_text: user's text query
    image_captions: list of (image_id, caption) tuples
    model: sentence transformer
    """
    query_emb = model.encode([query_text])
    caption_embs = model.encode([cap for _, cap in image_captions])
    
    similarities = util.cos_sim(query_emb, caption_embs)[0]
    ranked = sorted(zip(image_captions, similarities), key=lambda x: -x[1])
    
    return [(img_id, score) for (img_id, _), score in ranked]

captions = [
    ("img_001", "a beautiful sunset over the ocean"),
    ("img_002", "a cat sitting on a windowsill"),
    ("img_003", "golden hour lighting over calm waters"),
    ("img_004", "a dog running through a meadow"),
]

results = cross_modal_search("peaceful water at dusk", captions, model)
for img_id, score in results[:2]:
    print(f"  {img_id}: {score:.4f}")
```

## Embedding Space Properties

```
Good embedding space should have:
  1. Semantic similarity → Close vectors
  2. Linear relationships: king - man + woman ≈ queen
  3. Robustness: typos, paraphrases → similar embeddings
  4. Generalization: unseen concepts mapped reasonably

Evaluation metrics:
  - Recall@K: Do relevant items appear in top K?
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)
```

## 🧠 Exercises

### Exercise 1: Create a text similarity system that finds the most similar sentence pairs.

<details>
<summary>Solution</summary>

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "I love machine learning",
    "Artificial intelligence is fascinating",
    "The weather is nice today",
]

embs = model.encode(sentences)
pairs = []
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = util.cos_sim(embs[i], embs[j]).item()
        pairs.append((sentences[i], sentences[j], sim))

pairs.sort(key=lambda x: -x[2])
for s1, s2, sim in pairs[:3]:
    print(f"  [{sim:.4f}] '{s1}' ↔ '{s2}'")
```

</details>

### Exercise 2: Build a simple "zero-shot classifier" using embeddings: compute similarity between a new text and class descriptions.

<details>
<summary>Solution</summary>

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

classes = {
    "technology": "Software, hardware, programming, computers, AI",
    "sports": "Football, basketball, tennis, athletics, Olympics",
    "science": "Physics, chemistry, biology, research, experiments",
    "politics": "Government, elections, policy, democracy, laws",
}

class_embs = model.encode(list(classes.values()))

def classify(text):
    text_emb = model.encode([text])
    sims = util.cos_sim(text_emb, class_embs)[0]
    best_idx = sims.argmax().item()
    return list(classes.keys())[best_idx], sims[best_idx].item()

print(classify("The new iPhone has amazing AI features"))   # technology
print(classify("Champions League final was incredible"))    # sports
print(classify("New quantum computing breakthrough"))        # science
```

</details>
