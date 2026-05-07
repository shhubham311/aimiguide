# Advanced NLP & Embedding Concepts (Word2Vec)

## Why This Topic Matters

Natural Language Processing (NLP) is one of the most impactful applications of deep learning, powering technologies like Google Translate, Siri, ChatGPT, and sentiment analysis systems. The foundational breakthrough that enabled modern NLP was the concept of **word embeddings** — dense vector representations that capture semantic meaning.

Word2Vec (Mikolov et al., 2013) was the seminal model that proved words could be represented as mathematical vectors where **semantic relationships are preserved as geometric relationships**. The famous example: `king - man + woman ≈ queen`. This principle extends to all modern NLP: BERT, GPT, and T5 all build upon the embedding concept.

---

## 1. The Problem: Representing Text for Neural Networks

### Why Not One-Hot Encoding?

One-hot encoding represents each word as a sparse binary vector:

```
Vocabulary: ["king", "queen", "man", "woman", "apple"]

king  → [1, 0, 0, 0, 0]
queen → [0, 1, 0, 0, 0]
man   → [0, 0, 1, 0, 0]
woman → [0, 0, 0, 1, 0]
apple → [0, 0, 0, 0, 1]

Problems:
  1. Dimension = vocabulary size (often 50,000–500,000)
  2. All vectors are orthogonal (cosine similarity = 0 for all pairs)
  3. No notion of similarity: king is equally dissimilar to queen and apple
  4. Extremely sparse — wastes memory and compute
```

### The Solution: Dense Embeddings

```
Embedding vectors (simplified 3D):

king  → [0.95,  0.80,  0.12]
queen → [0.93,  0.82,  0.15]
man   → [0.91,  0.05,  0.10]
woman → [0.89,  0.07,  0.13]
apple → [0.02,  0.01,  0.95]

Properties:
  • Dimension is small (100–300 typically)
  • Similar words have similar vectors (cosine similarity ≈ 1)
  • king ≈ queen (both royalty)
  • man ≈ woman (both human)
  • apple is far from all human/royalty words
  • Geometric relationships encode meaning!
```

### The Distributional Hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth (1957)

Words appearing in similar contexts have similar meanings. This is the theoretical foundation of all embedding methods.

---

## 2. Word2Vec: The Architecture

Word2Vec has two architectures:
1. **CBOW (Continuous Bag of Words)**: Predict the center word from surrounding context
2. **Skip-Gram**: Predict surrounding context words from the center word

### CBOW vs Skip-Gram

```
CBOW (Context → Target):

  "The cat sat on the mat"
       ↓    ↓         ↓
  Context: [cat, on] → Model → Predict: "sat"

  ┌─────┐   ┌─────┐   ┌─────┐
  │ cat │──→│     │   │     │
  │     │   │ AVG │──→│     │──→ P(sat)
  │ on  │──→│     │   │     │
  └─────┘   └─────┘   └─────┘
  Context    Hidden     Output
  Words      Layer      (Softmax)


Skip-Gram (Target → Context):

  "The cat sat on the mat"
              ↓
  Target: "sat" → Model → Predict: [cat, on, the, mat]

  ┌─────┐   ┌─────┐   ┌─────┐
  │     │   │     │   │ cat │
  │ sat │──→│     │──→│ on  │
  │     │   │     │   │ the │
  └─────┘   └─────┘   │ mat │
  Target     Hidden    └─────┘
  Word       Layer     Context Predictions
```

### Skip-Gram: Mathematical Formulation

**Objective:** Maximize the probability of observing context words given the center word:

$$\mathcal{L} = \sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)$$

Where $c$ is the context window size and $T$ is the sequence length.

**Conditional probability with softmax:**

$$P(w_{\text{context}} \mid w_{\text{center}}) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w \in V} \exp(\mathbf{v}'_{w} \cdot \mathbf{v}_{w_I})}$$

Where:
- $\mathbf{v}_{w_I} \in \mathbb{R}^d$ is the input (center) word embedding
- $\mathbf{v}'_{w_O} \in \mathbb{R}^d$ is the output (context) word embedding
- $V$ is the vocabulary size

**Problem:** The denominator requires summing over the entire vocabulary ($|V|$ softmax), which is computationally prohibitive for large vocabularies.

---

## 3. Efficient Training: Negative Sampling

### The Solution

Instead of computing the full softmax, approximate it by distinguishing the **true context word** from a few **random noise (negative) words**:

$$\log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I})]$$

Where:
- $\sigma$ is the sigmoid function
- $k$ is the number of negative samples (typically 5–20)
- $P_n(w)$ is the noise distribution (typically $P(w) \propto \text{freq}(w)^{3/4}$)

```
Positive sample:   "sat" → "cat"  ✓ (maximize similarity)
Negative samples:  "sat" → "table" ✗ (minimize similarity)
                   "sat" → "frog"  ✗ (minimize similarity)
                   "sat" → "car"   ✗ (minimize similarity)
                   "sat" → "phone" ✗ (minimize similarity)

The model learns to push positive pairs together and negative pairs apart.
```

### Word2Vec Architecture Diagram

```
Skip-Gram with Negative Sampling:

                    Vocabulary
                    Size: V
Input Word: "sat"
     │
     ▼
┌──────────┐     ┌──────────────────┐     ┌──────────┐
│ Embedding │     │   Output         │     │ Sigmoid  │
│ Lookup    │     │   Embeddings     │     │ + Score  │
│ (d dims)  │     │   (V × d)        │     │          │
└────┬─────┘     └────┬─────────────┘     └────┬─────┘
     │                │                      │
     │    ┌──── context word (positive)
     │    │    ┌──── negative word 1
     ├────┼────┼──── negative word 2
     │    │    └──── ... negative word k
     │    │
     ▼    ▼
  [v_sat] → dot product → [log σ(v_sat · v_cat)]
                               + [log σ(-v_sat · v_neg1)]
                               + [log σ(-v_sat · v_neg2)]
                               + ...
                               = Loss

Two embedding matrices:
  W_in:  [V × d]  — input embeddings (these are Word2Vec vectors)
  W_out: [V × d]  — output embeddings (usually discarded after training)
```

---

## 4. Embedding Geometry: Analogies

### The Parallelogram Law

The most celebrated property of Word2Vec embeddings is that **semantic relationships are captured as vector offsets**:

$$\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$$

```
2D Embedding Space (illustrative):

           Royalty
              ↑
              │      king ●
              │          ╲
              │           ╲   (gender vector)
              │            ╲
              │         ●   ╲
              │        queen  │
              │                │
              │    ●───●──────●
              │   man   woman
              │        (gender)
              └────────────────→ Human/Male

Key insight: The vector from "man" to "woman" is approximately
the same as the vector from "king" to "queen".
This means: gender_direction ≈ v_woman - v_man ≈ v_queen - v_king

Other analogies that work:
  Paris - France + Italy ≈ Rome        (capital relationship)
  walk - walked + swim ≈ swam          (tense relationship)
  Japan - sushi + Germany ≈ bratwurst  (cultural relationship)
```

### Cosine Similarity

The standard measure of word similarity:

$$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum a_i^2} \sqrt{\sum b_i^2}}$$

Range: $[-1, 1]$ where $1$ = identical direction, $0$ = orthogonal, $-1$ = opposite.

---

## 5. Beyond Word2Vec: GloVe and FastText

### GloVe (Global Vectors)

Combines Word2Vec's local context windows with **global matrix factorization**:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

Where:
- $X_{ij}$ is the co-occurrence count of words $i$ and $j$ in the corpus
- $f$ is a weighting function that caps very frequent co-occurrences
- $\mathbf{w}_i, \tilde{\mathbf{w}}_j$ are word vectors

### FastText (Subword Embeddings)

FastText extends Word2Vec by representing each word as a sum of **character n-gram** vectors:

```
Word: "where"
Character n-grams (3-6 chars):
  <wh, whe, her, ere, re>, <whe, wher, here, ere>, ...
  
Vector representation:
  v("where") = v(<wh) + v(whe) + v(her) + v(ere) + v(re>) + ...

Advantages:
  • Can represent OOV (out-of-vocabulary) words
  • Handles morphologically similar words well
  • Works better for languages with rich morphology (Finnish, Turkish)
```

---

## 6. PyTorch Implementation

### 6.1 Training Word2Vec from Scratch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

# ════════════════════════════════════════════════
# Step 1: Preprocess Text
# ════════════════════════════════════════════════
corpus = """
the cat sat on the mat the dog lay on the rug
the cat chased the dog the dog barked at the cat
the king and the queen ruled the kingdom
the man and the woman walked in the park
the boy and the girl played in the garden
"""

# Tokenize
tokens = corpus.lower().split()
vocab = set(tokens)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# ════════════════════════════════════════════════
# Step 2: Generate Skip-Gram Training Pairs
# ════════════════════════════════════════════════
def generate_skipgram_pairs(tokens, window_size=2):
    """Generate (center_word, context_word) pairs."""
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                pairs.append((word2idx[center], word2idx[tokens[j]]))
    return pairs

skipgram_pairs = generate_skipgram_pairs(tokens, window_size=2)
print(f"Training pairs: {len(skipgram_pairs)}")

# ════════════════════════════════════════════════
# Step 3: Skip-Gram Model with Negative Sampling
# ════════════════════════════════════════════════
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Input (center) embeddings
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Output (context) embeddings
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)
        
        # Noise distribution (unigram^0.75)
        word_freq = Counter(tokens)
        total = sum(word_freq.values())
        self.noise_dist = torch.tensor(
            [word_freq[idx2word[i]] ** 0.75 for i in range(vocab_size)]
        )
        self.noise_dist = self.noise_dist / self.noise_dist.sum()
    
    def forward(self, center, context, neg_samples):
        """
        center:     [batch_size] center word indices
        context:    [batch_size] context word indices  
        neg_samples: [batch_size, num_neg] negative sample indices
        """
        # Get embeddings
        center_embeds = self.in_embeddings(center)        # [B, d]
        context_embeds = self.out_embeddings(context)      # [B, d]
        neg_embeds = self.out_embeddings(neg_samples)      # [B, K, d]
        
        # Positive score
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [B]
        pos_loss = torch.nn.functional.logsigmoid(pos_score)          # [B]
        
        # Negative score
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)           # [B]
        
        # Total loss (negative because we want to maximize)
        return -(pos_loss + neg_loss).mean()
    
    def get_embeddings(self):
        """Return the learned word embeddings."""
        return self.in_embeddings.weight.data.numpy()

# ════════════════════════════════════════════════
# Step 4: Training
# ════════════════════════════════════════════════
embed_dim = 50
num_neg_samples = 5
model = SkipGramNegSampling(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert pairs to tensors
centers = torch.tensor([p[0] for p in skipgram_pairs])
contexts = torch.tensor([p[1] for p in skipgram_pairs])

def get_neg_samples(batch_size, num_neg):
    """Sample negative examples from noise distribution."""
    return torch.multinomial(model.noise_dist, batch_size * num_neg, 
                             replacement=True).view(batch_size, num_neg)

# Training loop
for epoch in range(200):
    # Shuffle
    perm = torch.randperm(len(centers))
    centers_shuf = centers[perm]
    contexts_shuf = contexts[perm]
    
    total_loss = 0
    for i in range(0, len(centers_shuf), 32):
        batch_center = centers_shuf[i:i+32]
        batch_context = contexts_shuf[i:i+32]
        batch_neg = get_neg_samples(len(batch_center), num_neg_samples)
        
        optimizer.zero_grad()
        loss = model(batch_center, batch_context, batch_neg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")

# ════════════════════════════════════════════════
# Step 5: Evaluate Embeddings
# ════════════════════════════════════════════════
import numpy as np

embeddings = model.get_embeddings()  # [vocab_size, embed_dim]

def cosine_similarity(word1, word2):
    a = embeddings[word2idx[word1]]
    b = embeddings[word2idx[word2]]
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def most_similar(word, top_k=5):
    vec = embeddings[word2idx[word]]
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    embeds_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = embeds_norm @ vec_norm
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip the word itself
    return [(idx2word[i], similarities[i]) for i in top_indices]

print("\nSimilarity Tests:")
print(f"king vs queen: {cosine_similarity('king', 'queen'):.4f}")
print(f"king vs dog:   {cosine_similarity('king', 'dog'):.4f}")
print(f"cat vs dog:    {cosine_similarity('cat', 'dog'):.4f}")

print("\nMost similar to 'king':")
for word, sim in most_similar('king'):
    print(f"  {word:10s}: {sim:.4f}")
```

### 6.2 Using Pre-trained Embeddings

```python
import torch
import torch.nn as nn

# ════════════════════════════════════════════════
# Using nn.Embedding for NLP models
# ════════════════════════════════════════════════

# Random initialization
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)
token_ids = torch.tensor([1, 5, 100, 42])
embedded = embedding(token_ids)  # Shape: [4, 300]

# ════════════════════════════════════════════════
# Text Classification with Embeddings
# ════════════════════════════════════════════════
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: [batch_size, seq_len] — integer token IDs
        embedded = self.embedding(x)              # [B, S, E]
        pooled = embedded.mean(dim=1)             # [B, E] — average pooling
        return self.fc(pooled)                    # [B, num_classes]

# Model with GloVe-like initialization (conceptual)
vocab_size = 50000
embed_dim = 300
model = TextClassifier(vocab_size, embed_dim, num_classes=5)

# Load pre-trained embeddings (e.g., GloVe)
# pretrained_embeddings = load_glove_vectors()
# model.embedding.weight.data.copy_(pretrained_embeddings)
# model.embedding.weight.requires_grad = False  # Freeze embeddings
```

### 6.3 From Word Embeddings to Contextual Embeddings

```python
# Word2Vec/GloVe: Same word → Same embedding (context-independent)
# Contextual (BERT/GPT): Same word → Different embedding (context-dependent)

# Example with BERT (using transformers library)
# from transformers import AutoModel, AutoTokenizer
# 
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased')
# 
# # "bank" has different meanings:
# text1 = "I deposited money in the bank"
# text2 = "I sat by the river bank"
# 
# # BERT produces DIFFERENT embeddings for "bank" in each context
# # Word2Vec produces the SAME embedding for "bank" in both cases
```

---

## 7. Real-World ML Connections

| Application | Embedding Technique | Why |
|-------------|-------------------|-----|
| Sentiment Analysis | GloVe/FastText + CNN | Word embeddings capture sentiment-bearing words |
| Machine Translation | Transformer (learned embeddings) | Contextual embeddings handle word ambiguity |
| Named Entity Recognition | BERT embeddings | Context-aware representations |
| Search Engines | Word2Vec/GloVe | Semantic similarity for document ranking |
| Recommendation Systems | Item embeddings | Users and items in same embedding space |
| Question Answering | BERT/GPT embeddings | Deep contextual understanding |

---

## 8. Exercises

### Exercise 1: Compute Cosine Similarity
Given vectors $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [4, 5, 6]$, compute cosine similarity by hand.

**Solution:**
$$\mathbf{a} \cdot \mathbf{b} = 1(4) + 2(5) + 3(6) = 32$$
$$\|\mathbf{a}\| = \sqrt{1+4+9} = \sqrt{14} = 3.742$$
$$\|\mathbf{b}\| = \sqrt{16+25+36} = \sqrt{77} = 8.775$$
$$\cos(\mathbf{a}, \mathbf{b}) = \frac{32}{3.742 \times 8.775} = \frac{32}{32.834} = 0.9746$$

### Exercise 2: Skip-Gram Negative Sampling Loss
For the pair (center="cat", context="sat"), with embeddings $\mathbf{v}_{\text{cat}} = [0.5, -0.3]$ and $\mathbf{v}'_{\text{sat}} = [0.2, 0.8]$, and negative samples with $\mathbf{v}'_{\text{dog}} = [-0.1, 0.4]$, compute the loss.

**Solution:**
Positive score: $\mathbf{v}_{\text{cat}} \cdot \mathbf{v}'_{\text{sat}} = 0.5(0.2) + (-0.3)(0.8) = 0.1 - 0.24 = -0.14$
Negative score: $\mathbf{v}_{\text{cat}} \cdot \mathbf{v}'_{\text{dog}} = 0.5(-0.1) + (-0.3)(0.4) = -0.05 - 0.12 = -0.17$

$$\text{Loss} = -\left[\log\sigma(-0.14) + \log\sigma(0.17)\right]$$
$$= -\left[\log(0.465) + \log(0.542)\right]$$
$$= -[-0.765 + (-0.612)] = 1.377$$

### Exercise 3: Embedding Visualization
Train Word2Vec on a larger corpus and visualize the embeddings using PCA or t-SNE.

**Solution:**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assume embeddings is a numpy array [vocab_size, embed_dim]
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

words_to_plot = ['king', 'queen', 'man', 'woman', 'cat', 'dog', 'boy', 'girl']
plt.figure(figsize=(8, 6))
for word in words_to_plot:
    x, y = embeddings_2d[word2idx[word]]
    plt.scatter(x, y)
    plt.annotate(word, (x, y), fontsize=12)
plt.title("Word Embeddings (PCA)")
plt.grid(True, alpha=0.3)
plt.show()
```

### Exercise 4: Build a Sentence Encoder
Implement average pooling over Word2Vec embeddings to create sentence vectors, then use cosine similarity for sentence similarity.

**Solution:**
```python
def sentence_vector(sentence, embeddings, word2idx, embed_dim):
    """Average word embeddings to get sentence vector."""
    words = sentence.lower().split()
    vecs = []
    for w in words:
        if w in word2idx:
            vecs.append(embeddings[word2idx[w]])
    if len(vecs) == 0:
        return np.zeros(embed_dim)
    return np.mean(vecs, axis=0)

s1 = "the cat sat on the mat"
s2 = "the dog lay on the rug"
s3 = "the king ruled the kingdom"

v1 = sentence_vector(s1, embeddings, word2idx, embed_dim)
v2 = sentence_vector(s2, embeddings, word2idx, embed_dim)
v3 = sentence_vector(s3, embeddings, word2idx, embed_dim)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print(f"s1 vs s2 (both about animals/furniture): {cosine(v1, v2):.4f}")
print(f"s1 vs s3 (animals vs royalty):           {cosine(v1, v3):.4f}")
```

---

## 9. Summary

| Concept | Key Idea |
|---------|----------|
| Word Embedding | Dense vector representation capturing word meaning |
| Word2Vec (Skip-Gram) | Predict context from center word |
| Word2Vec (CBOW) | Predict center word from context |
| Negative Sampling | Efficient approximation of full softmax |
| GloVe | Global co-occurrence statistics + local context |
| FastText | Subword (n-gram) embeddings for OOV handling |
| Cosine Similarity | Measure semantic similarity between embeddings |
| Embedding Analogies | Vector arithmetic captures relationships |

The evolution from Word2Vec → GloVe → FastText → ELMo → BERT → GPT represents a progression from **static, context-independent** embeddings to **dynamic, context-aware** representations. Understanding Word2Vec is the essential foundation for understanding all of modern NLP.
