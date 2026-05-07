# Search, Retrieval & Recommendations

## Why This Matters

Search and recommendation are the **most deployed ML systems** in industry. Google, YouTube, Netflix, Amazon, Spotify — all built on search and retrieval fundamentals.

## The Full Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    SEARCH & RECOMMENDATION                     │
│                                                               │
│  Query/Input                                                  │
│      │                                                        │
│      ▼                                                        │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Candidate  │───▶│   Ranking    │───▶│  Re-ranking    │   │
│  │ Generation  │    │   Model      │    │  + Business    │   │
│  │ (Retrieval) │    │ (Scoring)    │    │    Rules       │   │
│  └─────────────┘    └──────────────┘    └────────────────┘   │
│   From millions     Score & sort         Diversity, freshness │
│   → thousands       candidates           brand safety         │
│                                                               │
│  Methods:            Methods:            Methods:              │
│  • ANN search        • Pointwise         • Rules              │
│  • BM25              • Pairwise          • Blending           │
│  • Embedding sim     • Listwise          • MMR diversity      │
│  • Collaborative     • Two-tower NN      • Context boost      │
└──────────────────────────────────────────────────────────────┘
```

## 1. Text Search: BM25

The classic information retrieval algorithm.

```
BM25 Score for document D and query Q:

  BM25(D, Q) = Σ IDF(qᵢ) × (f(qᵢ, D) × (k₁ + 1)) / (f(qᵢ, D) + k₁ × (1 - b + b × |D|/avgdl))

Where:
  f(qᵢ, D) = frequency of term qᵢ in document D
  |D| = document length
  avgdl = average document length
  k₁ = term frequency saturation (typically 1.2-2.0)
  b = length normalization (typically 0.75)
  IDF(qᵢ) = log((N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) + 1)
  N = total documents, n(qᵢ) = documents containing qᵢ
```

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.N = 0
        self.df = Counter()  # document frequency
        self.tf = []         # term frequency per doc
    
    def fit(self, documents):
        self.N = len(documents)
        self.tf = [Counter(doc.lower().split()) for doc in documents]
        self.doc_lengths = [len(self.tf[i]) for i in range(self.N)]
        self.avgdl = sum(self.doc_lengths) / self.N
        for term_freq in self.tf:
            for term in set(term_freq):
                self.df[term] += 1
    
    def _idf(self, term):
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)
    
    def search(self, query, top_k=5):
        query_terms = query.lower().split()
        scores = []
        for i in range(self.N):
            score = 0
            dl = self.doc_lengths[i]
            for term in query_terms:
                if term not in self.tf[i]:
                    continue
                f = self.tf[i][term]
                idf = self._idf(term)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

# Usage
docs = [
    "machine learning is the study of computer algorithms",
    "deep learning is a subset of machine learning",
    "natural language processing uses neural networks",
    "computer vision enables machines to see and understand images",
    "reinforcement learning trains agents through rewards",
]

bm25 = BM25()
bm25.fit(docs)
results = bm25.search("machine learning algorithms")
for doc_idx, score in results:
    print(f"  [{score:.4f}] {docs[doc_idx]}")
```

## 2. Vector Search (Semantic)

```python
from sentence_transformers import SentenceTransformer, util

class VectorSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
    
    def index(self, documents):
        self.documents = documents
        self.embeddings = self.model.encode(documents)
    
    def search(self, query, top_k=5):
        query_emb = self.model.encode([query])
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)[0]
        return [(self.documents[h['corpus_id']], h['score']) for h in hits]

vs = VectorSearch()
vs.index([
    "Linear regression predicts continuous values",
    "Logistic regression classifies binary outcomes",
    "Random forest is an ensemble of decision trees",
    "Neural networks learn complex non-linear patterns",
    "K-means clusters data into K groups",
])

results = vs.search("how to predict numbers")
for doc, score in results[:3]:
    print(f"  [{score:.4f}] {doc}")
```

## 3. Recommendation Architecture

```
┌─────────────────────────────────────────────────┐
│          RECOMMENDATION SYSTEM ARCHITECTURE       │
│                                                   │
│  User Context                                     │
│  ┌──────────┐                                     │
│  │ User ID  │                                     │
│  │ History  │───┐                                 │
│  │ Context  │   │                                 │
│  └──────────┘   │                                 │
│                 ▼                                 │
│  ┌──────────────────────┐                        │
│  │ Candidate Generation │  Millions → Thousands  │
│  │ • Collaborative filt │                        │
│  │ • Content-based      │                        │
│  │ • Popularity         │                        │
│  └──────────┬───────────┘                        │
│             ▼                                     │
│  ┌──────────────────────┐                        │
│  │    Ranking Model     │  Score each candidate   │
│  │ • Two-tower NN       │                        │
│  │ • Pointwise model    │                        │
│  │ • Feature crosses    │                        │
│  └──────────┬───────────┘                        │
│             ▼                                     │
│  ┌──────────────────────┐                        │
│  │    Re-ranking        │  Final 10-20 items     │
│  │ • Diversity (MMR)    │                        │
│  │ • Freshness boost    │                        │
│  │ • Business rules     │                        │
│  └──────────────────────┘                        │
└─────────────────────────────────────────────────┘
```

## 4. Query Understanding

```python
import re

class QueryUnderstanding:
    def __init__(self):
        self.synonyms = {
            "phone": ["mobile", "smartphone", "cellphone"],
            "laptop": ["notebook", "macbook", "computer"],
        }
    
    def normalize(self, query):
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query)
        return query
    
    def expand(self, query):
        words = query.split()
        expanded = set(words)
        for word in words:
            for syn_group in self.synonyms.values():
                if word in syn_group:
                    expanded.update(syn_group)
        return list(expanded)
    
    def classify_intent(self, query):
        query = query.lower()
        if any(w in query for w in ["buy", "price", "cheap", "deal"]):
            return "transactional"
        elif any(w in query for w in ["what", "how", "why", "guide"]):
            return "informational"
        else:
            return "navigational"

qu = QueryUnderstanding()
print(qu.normalize("  Buy cheap phones  "))
print(qu.expand("laptop deals"))
print(qu.classify_intent("how to build a neural network"))
```

## 🧠 Exercises

### Exercise 1: Build a search system that combines BM25 (keyword matching) with vector search (semantic similarity).

<details>
<summary>Solution approach</summary>

```python
class HybridSearch:
    def __init__(self, alpha=0.5):
        self.bm25 = BM25()
        self.vector_search = VectorSearch()
        self.alpha = alpha  # weight for BM25, (1-alpha) for vector
    
    def index(self, documents):
        self.bm25.fit(documents)
        self.vector_search.index(documents)
    
    def search(self, query, top_k=10):
        bm25_results = {idx: score for idx, score in self.bm25.search(query, top_k=50)}
        vector_results = {idx: score for idx, (doc, score) in enumerate(self.vector_search.search(query, top_k=50))}
        
        # Normalize and combine
        all_docs = set(bm25_results.keys()) | set(vector_results.keys())
        combined = []
        for doc_idx in all_docs:
            bm25_score = bm25_results.get(doc_idx, 0)
            max_bm25 = max(bm25_results.values()) if bm25_results else 1
            vec_score = vector_results.get(doc_idx, 0)
            
            norm_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            final = self.alpha * norm_bm25 + (1 - self.alpha) * vec_score
            combined.append((doc_idx, final))
        
        combined.sort(key=lambda x: -x[1])
        return combined[:top_k]
```

</details>
