# Learning to Rank & Loss Functions — Complete Guide

## Table of Contents
1. [Introduction to Learning to Rank](#introduction)
2. [Ranking Problem Formulation](#formulation)
3. [Pointwise Approach](#pointwise)
4. [Pairwise Approach](#pairwise)
5. [Listwise Approach](#listwise)
6. [Ranking Architectures](#architecture)
7. [Python Implementations](#python)
8. [ML Connections & Industry Applications](#ml-connections)
9. [Exercises](#exercises)

---

<a name="introduction"></a>
## 1. Introduction to Learning to Rank

### 1.1 What is Learning to Rank?

```
Learning to Rank (LTR) is the application of machine learning to construct
ranking models for information retrieval systems.

Traditional search: hand-crafted scoring functions
  score(query, doc) = tf · idf · BM25 · PageRank · ...

Learning to Rank: learn the scoring function from data
  score(q, d) = f(q, d; θ)  where θ is learned from relevance judgments

The goal: given a query and a set of documents, return the most relevant
documents in the optimal order.

┌─────────────────────────────────────────────────────────────┐
│                  RANKING IN THE REAL WORLD                    │
│                                                             │
│  Web Search:    query → ranked list of web pages            │
│  E-commerce:    query → ranked product listings             │
│  Recommender:   user → ranked list of items                 │
│  Ads:           query → ranked advertisements               │
│  Email:         inbox → ranked by importance                 │
│  News Feed:     user → ranked stories                       │
│  Q&A:           question → ranked answers                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Types of Relevance Labels

```
Graded relevance (typical 5-point scale):
  0 = Not relevant        (completely unrelated)
  1 = Poorly relevant     (marginally related)
  2 = Fairly relevant     (partially useful)
  3 = Highly relevant     (useful answer)
  4 = Perfectly relevant  (ideal result, needs no improvement)

Binary relevance:
  0 = Not relevant
  1 = Relevant

Ordinal categories:
  N = Non-relevant
  R = Relevant
  HR = Highly Relevant

Example query: "machine learning tutorials"
  ┌─────────────────────────────┬─────────┐
  │ Document                    │ Grade   │
  ├─────────────────────────────┼─────────┤
  │ Andrew Ng's ML Course       │   4     │
  │ Stanford CS229 Lecture Notes│   4     │
  │ scikit-learn Documentation  │   3     │
  │ "ML for Beginners" YouTube  │   2     │
  │ Wikipedia: Machine Learning │   2     │
  │ Random blog about AI        │   1     │
  │ Cooking recipe website      │   0     │
  └─────────────────────────────┴─────────┘
```

### 1.3 Three Paradigms of Learning to Rank

```
┌────────────────────────────────────────────────────────────────────┐
│                    THREE APPROACHES TO LTR                          │
│                                                                    │
│  POINTWISE              PAIRWISE               LISTWISE            │
│  ─────────              ─────────              ─────────           │
│  Score each             Compare pairs          Rank entire          │
│  document               of documents           list at once         │
│  independently          within a query         for a query         │
│                                                                    │
│  Loss on individual     Loss on relative       Loss on             │
│  (query, doc) pair      ordering of pairs      permutation         │
│                                                                    │
│  Reg/Cls on single      Classify pair order    Directly optimize   │
│  document               as "correct"/"wrong"   ranking metrics     │
│                                                                    │
│  Examples:              Examples:              Examples:            │
│  • PRank                • RankNet              • ListNet            │
│  • MCRank               • RankBoost            • ListMLE            │
│  • SVMrank              • LambdaRank           • LambdaMART         │
│                         • GBRank               • Softmax            │
│                                                                    │
│  Simple but ignores     Captures relative      Best performance    │
│  position and inter-    position but not       but most complex    │
│  document relationships list-level metrics     to implement        │
└────────────────────────────────────────────────────────────────────┘
```

---

<a name="formulation"></a>
## 2. Ranking Problem Formulation

### 2.1 Formal Definition

```
Input:
  • Query q with feature vectors for n documents: {(q, d₁), (q, d₂), ..., (q, dₙ)}
  • Feature vector φ(q, d) ∈ ℝᵈ for each query-document pair
  • Relevance labels y = (y₁, y₂, ..., yₙ) where yᵢ ∈ {0, 1, 2, 3, 4}

Output:
  • Ranking function f(q, d) = wᵀφ(q, d) that assigns a score to each document
  • Sorted list: d_{π(1)} ≽ d_{π(2)} ≽ ... ≽ d_{π(n)} where π is the permutation

Learning:
  • Minimize loss: L(π, y) over all queries
  • Or maximize ranking metric: NDCG@k, MAP@k

Feature vector φ(q, d) typically includes:
  ┌──────────────────────────────────────────────┐
  │ Query-dependent features:                    │
  │  • BM25 score                                │
  │  • TF-IDF similarity                        │
  │  • Query term coverage                       │
  │  • Edit distance                             │
  │                                              │
  │ Document-dependent features:                 │
  │  • PageRank / authority score                │
  │  • Document length                           │
  │  • Freshness / last updated                  │
  │  • URL depth                                 │
  │                                              │
  │ Cross features:                              │
  │  • Query-document similarity                 │
  │  • Click-through rate                        │
  │  • Dwell time                                │
  │  • Bounce rate                               │
  └──────────────────────────────────────────────┘
```

### 2.2 Feature Engineering for Ranking

```
Typical features used in web search ranking (Microsoft LETOR dataset style):

Query-Level Features (same for all docs in a query):
  QF1: Number of query terms
  QF2: Sum of IDF values
  QF3: Query length

Term-Level Features (per term, aggregated):
  TF1: Sum of TF of query terms in doc body
  TF2: Minimum TF of query terms
  TF3: Maximum TF of query terms
  IDF1: Sum of IDF values of matched terms

Document-Level Features:
  DL1: Document length (bytes)
  DL2: Document length (terms)
  DL3: Number of inbound links

Query-Document Features:
  QD1: BM25 score
  QD2: Cosine similarity of TF-IDF vectors
  QD3: Coverage ratio (matched terms / query terms)
  QD4: Jaccard similarity

Behavioral Features:
  B1: Click-through rate for (q, d)
  B2: Dwell time on document
  B3: Bounce rate

Example feature vector for (query="python tutorial", doc="python.org"):
  φ = [3, 4.2, 2, 15, 3, 8, 5.6, 12.3, 45000, 2100, 500, 8.7, 0.82, 0.67, 0.023]
       ↑                                                                                          ↑
       QF1                                                                                      B1
```

---

<a name="pointwise"></a>
## 3. Pointwise Approach

### 3.1 Overview

```
The pointwise approach treats ranking as a regression or classification problem
on individual query-document pairs.

Key idea: Learn a function f(q, d) ≈ y where y is the relevance grade.

  For each (query, document) pair independently:
    f(qᵢ, dⱼ; θ) → ŷᵢⱼ ≈ yᵢⱼ

  Ranking is produced by sorting documents by f(q, d) scores.

                    Query q
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      (q,d₁)       (q,d₂)       (q,d₃)
      score: 4      score: 2      score: 0
        │             │             │
        ▼             ▼             ▼
    f(q,d₁)=3.8   f(q,d₂)=2.1   f(q,d₃)=0.3
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
            Sort by score → d₁ ≻ d₂ ≻ d₃
```

### 3.2 Pointwise Methods

```
┌──────────────────────────────────────────────────────────────┐
│ Pointwise Method          │ Loss Function                    │
├───────────────────────────┼──────────────────────────────────┤
│ Regression (MSE)          │ L = (f(q,d) - y)²                │
│ Ordinal Regression        │ L = Σⱼ<ᵢ σ(f(q,dⱼ)-f(q,dᵢ))   │
│ Classification            │ L = CrossEntropy(f(q,d), y)      │
│ PRank (Perceptron Rank)   │ L = Hinge loss on thresholds     │
└───────────────────────────┴──────────────────────────────────┘
```

### 3.3 Cross-Entropy Loss for Ranking

```
When treating relevance grades as classes:

  L = -Σᵢ Σⱼ yᵢⱼ log(pᵢⱼ)

  where pᵢⱼ = softmax(f(q, dᵢ))_j = exp(f(q,dᵢ)) / Σₖ exp(f(q,dₖ))

For binary relevance (relevant vs. not):
  L = -[y·log(σ(f(q,d))) + (1-y)·log(1-σ(f(q,d)))]

  where σ(z) = 1/(1+exp(-z)) is the sigmoid function

Multi-class (grades 0-4):
  L = -Σ(c=0 to 4) I(y=c) · log(p_c)

  where p_c = softmax(f(q,d))_c

Implementation considerations:
  • Treat each grade as a separate class
  • Can also use ordinal regression (preserve ordering)
  • Does NOT directly optimize ranking metrics
```

### 3.4 Advantages and Disadvantages

```
Advantages:
  ✓ Simple to implement (standard ML algorithms)
  ✓ Scales well to large datasets
  ✓ Works with any regression/classification model
  ✓ Well-understood theory

Disadvantages:
  ✗ Ignores relative position of documents
  ✗ Treats each (q,d) pair independently
  ✗ Doesn't optimize ranking-specific metrics (NDCG, MAP)
  ✗ Position bias not captured
  ✗ For regression: predicting grade 3 vs 4 treated as equal error
    to predicting 0 vs 1 (but ranking-wise, 3 vs 4 matters less)
```

---

<a name="pairwise"></a>
## 4. Pairwise Approach

### 4.1 Overview

```
The pairwise approach models the relative ordering between pairs of documents.

Key idea: Learn to determine which document is more relevant than another.

  For each pair (dᵢ, dⱼ) where yᵢ ≠ yⱼ:
    P(dᵢ ≻ dⱼ | q) should be high when yᵢ > yⱼ

  Instead of predicting absolute relevance, predict relative preference.

          Query q
            │
    ┌───────┴───────┐
    ▼               ▼
  (q, d₁)        (q, d₂)
  grade: 4       grade: 2
    │               │
    └───────┬───────┘
            ▼
    Is d₁ ≻ d₂?  →  YES (predict 1)
    
    ┌───────┴───────┐
    ▼               ▼
  (q, d₂)        (q, d₃)
  grade: 2       grade: 2
    │               │
    └───────┬───────┘
            ▼
    Is d₂ ≻ d₃?  →  TIE (skip or predict 0.5)
```

### 4.2 RankNet

```
RankNet (Burges et al., 2005) uses a neural network to learn pairwise
preferences with cross-entropy loss.

Score difference: sᵢⱼ = f(q, dᵢ) - f(q, dⱼ)

Probability that dᵢ is ranked above dⱼ:
  Pᵢⱼ = σ(sᵢⱼ) = 1 / (1 + exp(-sᵢⱼ))

  σ(z) = 1/(1+exp(-z))  is the sigmoid function

  When sᵢⱼ >> 0:  Pᵢⱼ → 1  (dᵢ confidently ranked above dⱼ)
  When sᵢⱼ = 0:   Pᵢⱼ = 0.5 (tie)
  When sᵢⱼ << 0:  Pᵢⱼ → 0  (dⱼ ranked above dᵢ)

Target probability based on true labels:
  P̄ᵢⱼ = 1/2 (1 + S(yᵢ, yⱼ))

  where S(yᵢ, yⱼ) = { +1 if yᵢ > yⱼ
                        0 if yᵢ = yⱼ
                       -1 if yᵢ < yⱼ }

Pairwise cross-entropy loss for one pair:
  Lᵢⱼ = -P̄ᵢⱼ · log(Pᵢⱼ) - (1 - P̄ᵢⱼ) · log(1 - Pᵢⱼ)

Total loss:
  L = Σ_q Σ_{i,j: yᵢ>yⱼ} Lᵢⱼ

Gradient with respect to sᵢⱼ:
  ∂L/∂sᵢⱼ = -σ(sᵢⱼ) + P̄ᵢⱼ

  For yᵢ > yⱼ (P̄ᵢⱼ = 1):
    ∂L/∂sᵢⱼ = -σ(sᵢⱼ) + 1

  This pushes sᵢⱼ → ∞ (making dᵢ ranked above dⱼ)
```

### 4.3 RankNet Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RankNet Architecture                      │
│                                                             │
│    Feature Vector φ(q, dᵢ)      Feature Vector φ(q, dⱼ)    │
│          │                            │                     │
│          ▼                            ▼                     │
│  ┌───────────────┐          ┌───────────────┐              │
│  │  Hidden Layer 1│          │  Hidden Layer 1│              │
│  │  (shared weights)│         │  (shared weights)│            │
│  └───────┬───────┘          └───────┬───────┘              │
│          │                            │                     │
│          ▼                            ▼                     │
│  ┌───────────────┐          ┌───────────────┐              │
│  │  Hidden Layer 2│          │  Hidden Layer 2│              │
│  │  (shared weights)│         │  (shared weights)│            │
│  └───────┬───────┘          └───────┬───────┘              │
│          │                            │                     │
│          ▼                            ▼                     │
│       sᵢ = f(q,dᵢ)              sⱼ = f(q,dⱼ)             │
│          │                            │                     │
│          └──────────┬─────────────────┘                     │
│                     ▼                                       │
│              sᵢⱼ = sᵢ - sⱼ                                │
│                     │                                       │
│                     ▼                                       │
│              Pᵢⱼ = σ(sᵢⱼ)                                 │
│                     │                                       │
│                     ▼                                       │
│           Cross-Entropy Loss                                │
│                                                             │
│  Key: Both sub-networks share weights (siamese architecture)│
└─────────────────────────────────────────────────────────────┘
```

### 4.4 LambdaRank

```
Problem with RankNet: Optimizes pairwise error rate, which doesn't directly
correlate with ranking metrics like NDCG.

LambdaRank solution: Weight the gradient by the change in the ranking metric.

Key insight: Instead of just using ∂L/∂sᵢⱼ, multiply by the "importance"
of swapping dᵢ and dⱼ in the ranked list.

  λᵢⱼ = -σ(sᵢⱼ) × ΔNDCG

  where ΔNDCG = |NDCG(swap dᵢ, dⱼ) - NDCG(current)|

  ΔNDCG is large when:
  • Both documents are near the top of the ranking
  • The relevance grades are very different
  • Swapping them would significantly change NDCG

  λᵢⱼ is small when:
  • Documents are low in the ranking (swapping doesn't matter much)
  • Both have similar relevance grades

Intuition:
  ┌──────────────────────────────────────────────────┐
  │ RankNet gradient:  "Get the order right"         │
  │ LambdaRank gradient: "Get the order right, and  │
  │   it matters MORE to get the top positions right" │
  └──────────────────────────────────────────────────┘

For a pair (dᵢ, dⱼ) with yᵢ > yⱼ:

  RankNet:     λ = σ'(sᵢⱼ)      (same for all positions)
  LambdaRank:  λ = σ'(sᵢⱼ) × |ΔNDCGᵢⱼ|  (position-dependent)
```

### 4.5 Pairwise Advantages and Disadvantages

```
Advantages:
  ✓ Captures relative document ordering
  ✓ LambdaRank directly optimizes NDCG (empirically)
  ✓ LambdaMART is one of the best performing LTR methods
  ✓ More natural for ranking than pointwise

Disadvantages:
  ✗ Number of pairs grows as O(n²) per query
  ✗ All pairs treated equally (in RankNet, not LambdaRank)
  ✗ Doesn't consider the entire list at once
  ✗ Can overfit to easy pairs (clearly relevant vs. irrelevant)
  ✗ Inconsistent with top-k metrics (cares about all positions)
```

---

<a name="listwise"></a>
## 5. Listwise Approach

### 5.1 Overview

```
The listwise approach optimizes the ranking of the entire document list
for a query at once.

Key idea: Define loss directly on the permutation (or probability
distribution over permutations) of the ranked list.

  Input:  {(q, d₁, y₁), (q, d₂, y₂), ..., (q, dₙ, yₙ)}
  Output: Permutation π that optimizes L(π, y)

          Query q
            │
  ┌─────────┼─────────┐
  ▼         ▼         ▼
d₁(4)    d₂(2)    d₃(0)
d₄(3)    d₅(1)    d₆(2)
  │         │         │
  └────┬────┘         │
       │              │
       └──────┬───────┘
              │
              ▼
  Consider ALL 6! = 720 permutations
  Find π that maximizes NDCG(π, y)
```

### 5.2 ListNet

```
ListNet (Cao et al., 2007) defines a probability distribution over
permutations using the Plackett-Luce model.

Score-based probability:
  P(π) = Π(i=1 to n) [exp(f(q,d_{π(i)})) / Σ(j=i to n) exp(f(q,d_{π(j)}))]

Top-1 probability (simplification):
  P(dᵢ is first) = exp(sᵢ) / Σⱼ exp(sⱼ)

  where sᵢ = f(q, dᵢ)

True probability (from relevance labels):
  P*(dᵢ is first) = exp(yᵢ) / Σⱼ exp(yⱼ)

Cross-entropy loss (list-level):
  L = -Σᵢ P*(dᵢ is first) × log(P(dᵢ is first))

This can be extended to top-k:
  P(top-k = {d₁,...,dₖ}) = Π(i=1 to k) [exp(sᵢ) / Σ(j≥i) exp(sⱼ)]

Advantages:
  • Differentiable (can use neural networks)
  • Considers all documents simultaneously
  • Simple and elegant formulation
```

### 5.3 ListMLE

```
ListMLE maximizes the likelihood of the correct permutation.

Given the ideal ranking π* (sorted by relevance):
  L = -log P(π*)

  P(π*) = Π(i=1 to n) [exp(s_{π*(i)}) / Σ(j=i to n) exp(s_{π*(j)})]

Negative log-likelihood:
  L = -Σ(i=1 to n) [s_{π*(i)} - log(Σ(j=i to n) exp(s_{π*(j)}))]

  = -Σ(i=1 to n) s_{π*(i)} + Σ(i=1 to n) log(Σ(j=i to n) exp(s_{π*(j)}))

Relationship to ListNet:
  ListMLE is a special case of ListNet using the true permutation
  as the target distribution.

  ListNet:  L = -Σᵢ P*(top-1 = dᵢ) log(P(top-1 = dᵢ))
  ListMLE:  L = -log P(ideal permutation)
```

### 5.4 LambdaMART

```
LambdaMART (Burges, 2010) combines LambdaRank gradients with MART
(Multiple Additive Regression Trees = Gradient Boosted Trees).

Algorithm:
  ┌──────────────────────────────────────────────────────┐
  │ LambdaMART Training                                  │
  │                                                      │
  │ For round t = 1, 2, ..., T:                          │
  │   1. Compute λᵢ for each document:                   │
  │      λᵢ = Σⱼ λᵢⱼ                                   │
  │                                                      │
  │      where λᵢⱼ = -σ(sᵢ - sⱼ) × |ΔNDCGᵢⱼ|          │
  │      (sum over all pairs involving document dᵢ)      │
  │                                                      │
  │   2. Fit a regression tree h_t to predict λᵢ       │
  │      (using Newton step: value = λᵢ / |λᵢ| = sign)  │
  │                                                      │
  │   3. Update: sᵢ ← sᵢ + η × h_t(φ(q, dᵢ))          │
  │                                                      │
  │   4. Update leaf values using Newton-Raphson step    │
  └──────────────────────────────────────────────────────┘

Lambda computation for document dᵢ:

  λᵢ = Σ_{j: yⱼ > yᵢ} (-∂L/∂sᵢⱼ) × |ΔNDCGᵢⱼ|
      + Σ_{j: yⱼ < yᵢ} (∂L/∂sᵢⱼ) × |ΔNDCGᵢⱼ|

  = Σ_{j ≠ i} -σ(-(yᵢ - yⱼ)(sᵢ - sⱼ)) × |ΔNDCGᵢⱼ| / Z

  where Z is a normalization constant.

Why LambdaMART works so well:
  ┌─────────────────────────────────────────────────────┐
  │ 1. Directly optimizes NDCG (via lambda gradients)  │
  │ 2. MART captures complex feature interactions       │
  │ 3. Non-parametric (no need to specify feature       │
  │    interactions)                                    │
  │ 4. Robust to overfitting with proper regularization│
  │ 5. Consistently top performer in LTR competitions   │
  └─────────────────────────────────────────────────────┘
```

### 5.5 Comparison of All Approaches

```
┌─────────────────┬───────────┬───────────┬───────────┐
│ Property        │ Pointwise │ Pairwise  │ Listwise  │
├─────────────────┼───────────┼───────────┼───────────┤
│ Input           │ Single    │ Document  │ Entire    │
│                 │ (q,d)     │ pair      │ list      │
├─────────────────┼───────────┼───────────┼───────────┤
│ Output          │ Relevance │ Preference│ Ranking   │
│                 │ score     │ order     │ order     │
├─────────────────┼───────────┼───────────┼───────────┤
│ Loss            │ MSE / CE  │ Cross-    │ List-level│
│                 │           │ entropy   │ NLL / λ   │
├─────────────────┼───────────┼───────────┼───────────┤
│ Optimizes       │ Indiv.    │ Pairwise  │ List      │
│ ranking metric? │ No        │ Partially │ Yes       │
│                 │           │ (Lambda)  │           │
├─────────────────┼───────────┼───────────┼───────────┤
│ Complexity      │ O(n)      │ O(n²)     │ O(n log n)│
│ per query       │           │           │ or O(n²)  │
├─────────────────┼───────────┼───────────┼───────────┤
│ Position        │ No        │ No        │ Yes       │
│ awareness       │           │           │ (Lambda)  │
├─────────────────┼───────────┼───────────┼───────────┤
│ Implementation  │ Easy      │ Medium    │ Hard      │
├─────────────────┼───────────┼───────────┼───────────┤
│ Best method     │ Baseline  │ RankNet   │ LambdaMART│
│ representative  │           │ LambdaRank│ ListMLE   │
├─────────────────┼───────────┼───────────┼───────────┤
│ Typical         │ 0.70-0.80 │ 0.75-0.85 │ 0.80-0.90 │
│ NDCG@10         │           │           │           │
└─────────────────┴───────────┴───────────┴───────────┘
```

---

<a name="architecture"></a>
## 6. Ranking Architectures

### 6.1 Complete Ranking System

```
┌─────────────────────────────────────────────────────────────────────┐
│                PRODUCTION RANKING SYSTEM ARCHITECTURE                │
│                                                                     │
│  Query                                                             │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │  Query      │    │  Document   │    │  Candidate  │            │
│  │  Under-     │───►│  Retrieval  │───►│  Generation │            │
│  │  standing   │    │  (BM25,     │    │  (re-rank)  │            │
│  │             │    │   ANN)      │    │             │            │
│  └─────────────┘    └─────────────┘    └─────────────┘            │
│  • Spell check       • inverted index      • LTR model            │
│  • Query expansion   • vector search       • neural ranker        │
│  • Intent classif.   • semantic search     • cross-encoder        │
│  • Entity recogn.    • 1000+ candidates    • top 100 docs         │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │  Feature    │    │  Scoring    │    │  Business   │            │
│  │  Extraction │───►│  & Ranking  │───►│  Logic      │            │
│  │             │    │             │    │             │            │
│  └─────────────┘    └─────────────┘    └─────────────┘            │
│  • Compute φ(q,d)   • Apply f(q,d)      • Diversity            │
│  • Cached features  • Sort by score     • Freshness boost       │
│  • Real-time feats  • Score normalize   • Personalization      │
│                       10-100 docs        • Blending             │
│                                                                     │
│  Final Ranked List                                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1. [Ad Slot - Sponsored]  Relevant Product A    ★★★★★      │  │
│  │ 2. [Ad Slot - Sponsored]  Relevant Product B    ★★★★       │  │
│  │ ────────────────────────────────────────────────────────     │  │
│  │ 3. Organic Result C       Very Relevant         ★★★★       │  │
│  │ 4. Organic Result D       Relevant             ★★★        │  │
│  │ 5. Organic Result E       Somewhat Relevant    ★★          │  │
│  │ ...                                                        │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Modern Neural Ranking Approaches

```
┌──────────────────────────────────────────────────────────────┐
│  Two-Stage Neural Ranking                                     │
│                                                              │
│  Stage 1: Bi-Encoder (Retrieval)                             │
│  ┌────────────┐    ┌────────────┐                            │
│  │  Query     │    │  Document  │                            │
│  │  Encoder   │    │  Encoder   │                            │
│  │  (BERT)    │    │  (BERT)    │                            │
│  └─────┬──────┘    └─────┬──────┘                            │
│        │                 │                                   │
│        ▼                 ▼                                   │
│     q_vec             d_vec    (embeddings in same space)    │
│        │                 │                                   │
│        └────► cos(q,d) ◄─┘   (dot product / cosine sim)     │
│                                                              │
│  Fast: O(1) lookup with ANN index (FAISS, ScaNN)            │
│  But: Limited interaction between query and document         │
│                                                              │
│  Stage 2: Cross-Encoder (Re-ranking)                         │
│  ┌──────────────────────────────────┐                        │
│  │  [CLS] Query [SEP] Document [SEP]│                        │
│  │         ↓                        │                        │
│  │  BERT / RoBERTa / XLM-R         │                        │
│  │         ↓                        │                        │
│  │  Relevance Score (single value)  │                        │
│  └──────────────────────────────────┘                        │
│                                                              │
│  Slow: O(n) for n candidates (can't pre-compute)            │
│  But: Full cross-attention between query and document       │
│                                                              │
│  Typical pipeline:                                           │
│  BM25 → 1000 docs → Bi-encoder → 100 docs → Cross-encoder  │
│  → 10 docs → Final ranking                                  │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 Ranking Loss Functions Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RANKING LOSS FUNCTIONS                             │
├──────────────┬──────────────────────────────────────────────────────┤
│ Loss         │ Formula                                              │
├──────────────┼──────────────────────────────────────────────────────┤
│ Pointwise    │                                                      │
│  MSE         │ L = Σᵢ (f(q,dᵢ) - yᵢ)²                            │
│  CrossEntropy│ L = -Σᵢ yᵢ log(softmax(f(q,dᵢ)))                   │
├──────────────┼──────────────────────────────────────────────────────┤
│ Pairwise    │                                                      │
│  RankNet    │ L = Σᵢⱼ [-P̄ᵢⱼ log σ(sᵢ-sⱼ) - (1-P̄ᵢⱼ) log(1-σ)] │
│  Hinge      │ L = Σᵢⱼ max(0, 1 - (sᵢ-sⱼ) · S(yᵢ,yⱼ))          │
│  MarginRank │ L = Σᵢⱼ max(0, 1 - (sᵢ-sⱼ)) · I(yᵢ>yⱼ)          │
├──────────────┼──────────────────────────────────────────────────────┤
│ Listwise    │                                                      │
│  ListNet    │ L = -Σᵢ P*(dᵢ first) log P(dᵢ first)              │
│  ListMLE    │ L = -Σᵢ [sᵢ - log Σⱼ≥ᵢ exp(sⱼ)]                   │
│  LambdaRank │ λᵢⱼ = -σ(sᵢ-sⱼ) × |ΔNDCGᵢⱼ|                       │
│  Softmax    │ L = -Σᵢ yᵢ log(softmax(sᵢ))                        │
│  ApproxNDCG │ L = -Σᵢ G(yᵢ) log(softmax(sᵢ)/Σⱼ exp(sⱼ))        │
└──────────────┴──────────────────────────────────────────────────────┘
```

---

<a name="python"></a>
## 7. Python Implementations

### 7.1 Pointwise Ranking

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn

# ============================================================
# 7.1.1 Pointwise Regression
# ============================================================
def pointwise_rank_regression(X_train, y_train, X_test, query_ids_test):
    """
    Pointwise ranking using regression.
    
    Parameters:
        X_train: feature matrix (n_train, d)
        y_train: relevance grades (n_train,)
        X_test: feature matrix (n_test, d)
        query_ids_test: query group ids for test set
    """
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict scores
    scores = model.predict(X_test)
    
    # Rank within each query group
    ranked_results = {}
    for qid in np.unique(query_ids_test):
        mask = query_ids_test == qid
        q_scores = scores[mask]
        q_indices = np.argsort(-q_scores)  # descending
        ranked_results[qid] = q_indices
    
    return model, ranked_results


# ============================================================
# 7.1.2 Pointwise Classification
# ============================================================
def pointwise_rank_classification(X_train, y_train, X_test):
    """
    Pointwise ranking using ordinal classification.
    """
    model = LogisticRegression(
        multi_class='multinomial', max_iter=1000, random_state=42
    )
    model.fit(X_train, y_train)
    
    # Use probability of high relevance classes as score
    # Score = expected relevance = Σ(c * P(c))
    probs = model.predict_proba(X_test)
    classes = model.classes_
    scores = probs @ classes  # expected relevance grade
    
    return model, scores


# ============================================================
# 7.1.3 Neural Pointwise Ranker
# ============================================================
class PointwiseRankNet(nn.Module):
    """Neural network for pointwise ranking."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def predict_scores(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)


# ============================================================
# 7.1.4 Synthesize Ranking Data
# ============================================================
def generate_ranking_data(n_queries=500, docs_per_query=50, n_features=20,
                           n_relevant=10):
    """Generate synthetic ranking data for experimentation."""
    np.random.seed(42)
    
    X_list = []
    y_list = []
    query_ids = []
    
    for q in range(n_queries):
        # Generate features for documents
        X_q = np.random.randn(docs_per_query, n_features)
        
        # Create a "true" scoring function with some structure
        weights = np.random.randn(n_features)
        true_scores = X_q @ weights + np.random.randn(docs_per_query) * 0.5
        
        # Convert to relevance grades (0-4)
        # Top docs get higher grades
        sorted_idx = np.argsort(-true_scores)
        y_q = np.zeros(docs_per_query, dtype=int)
        y_q[sorted_idx[:max(1, n_relevant//5)]] = 4
        y_q[sorted_idx[n_relevant//5:n_relevant//2]] = 3
        y_q[sorted_idx[n_relevant//2:n_relevant]] = 2
        y_q[sorted_idx[n_relevant:2*n_relevant]] = 1
        # Rest are 0 (not relevant)
        
        # Shuffle to remove the sorted order
        shuffle_idx = np.random.permutation(docs_per_query)
        X_list.append(X_q[shuffle_idx])
        y_list.append(y_q[shuffle_idx])
        query_ids.extend([q] * docs_per_query)
    
    return np.vstack(X_list), np.concatenate(y_list), np.array(query_ids)
```

### 7.2 Pairwise Ranking (RankNet)

```python
# ============================================================
# 7.2.1 RankNet Implementation
# ============================================================
class RankNet(nn.Module):
    """
    RankNet: Pairwise neural ranking model.
    
    Uses a siamese architecture with shared weights.
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x_i, x_j):
        """Score a pair of documents."""
        s_i = self.fc(x_i).squeeze(-1)
        s_j = self.fc(x_j).squeeze(-1)
        return s_i, s_j
    
    def pairwise_loss(self, s_i, s_j, label):
        """
        Pairwise cross-entropy loss.
        
        label: +1 if dᵢ ≻ dⱼ, -1 if dⱼ ≻ dᵢ
        """
        s_diff = s_i - s_j
        target_prob = 0.5 * (1 + label)  # 1 if label=+1, 0 if label=-1
        
        # P_ij = sigmoid(s_i - s_j)
        log_prob = -nn.functional.logsigmoid(label * s_diff)
        loss = target_prob * log_prob + (1 - target_prob) * (
            -nn.functional.logsigmoid(-label * s_diff)
        )
        return loss.mean()
    
    def score(self, x):
        """Score a single document."""
        return self.fc(x).squeeze(-1)


def generate_pairs(X, y, query_ids):
    """
    Generate pairwise training data from ranking data.
    
    Returns:
        X_i, X_j: feature vectors for each pair
        labels: +1 if i should rank above j, -1 otherwise
    """
    X_i_list, X_j_list, label_list = [], [], []
    
    for qid in np.unique(query_ids):
        mask = query_ids == qid
        q_X = X[mask]
        q_y = y[mask]
        
        # Generate pairs with different relevance
        for i in range(len(q_y)):
            for j in range(len(q_y)):
                if q_y[i] > q_y[j]:
                    X_i_list.append(q_X[i])
                    X_j_list.append(q_X[j])
                    label_list.append(1)
                elif q_y[i] < q_y[j]:
                    X_i_list.append(q_X[i])
                    X_j_list.append(q_X[j])
                    label_list.append(-1)
    
    return (np.array(X_i_list), np.array(X_j_list),
            np.array(label_list))


def train_ranknet(X, y, query_ids, input_dim, epochs=10, batch_size=256,
                  lr=0.001):
    """Train a RankNet model."""
    # Generate pairwise data
    X_i, X_j, labels = generate_pairs(X, y, query_ids)
    
    # Convert to tensors
    X_i_t = torch.FloatTensor(X_i)
    X_j_t = torch.FloatTensor(X_j)
    labels_t = torch.FloatTensor(labels)
    
    # Model
    model = RankNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    n_samples = len(labels)
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start+batch_size]
            
            s_i, s_j = model(X_i_t[batch_idx], X_j_t[batch_idx])
            loss = model.pairwise_loss(s_i, s_j, labels_t[batch_idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_idx)
        
        avg_loss = total_loss / n_samples
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# ============================================================
# 7.2.2 LambdaRank Implementation
# ============================================================
def compute_lambda_gradients(scores, labels, rank_metric='ndcg'):
    """
    Compute LambdaRank gradients for a single query.
    
    Parameters:
        scores: predicted scores for documents (n,)
        labels: true relevance grades (n,)
        rank_metric: 'ndcg' or 'map'
    
    Returns:
        lambdas: gradient for each document (n,)
    """
    n = len(scores)
    lambdas = np.zeros(n)
    
    # Sort documents by predicted scores (descending)
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    
    # Compute DCG gains
    gains = (2 ** sorted_labels - 1)
    discounts = np.log2(np.arange(2, n + 2))
    dcg = np.sum(gains / discounts)
    
    # Ideal DCG
    ideal_order = np.argsort(-labels)
    ideal_gains = (2 ** labels[ideal_order] - 1)
    idcg = np.sum(ideal_gains / discounts)
    
    if idcg == 0:
        return lambdas
    
    # Compute pairwise lambdas
    for i in range(n):
        for j in range(i + 1, n):
            # Only compute for pairs with different labels
            if sorted_labels[i] == sorted_labels[j]:
                continue
            
            # Position in the ranking
            pos_i = i  # already sorted by scores
            
            # Score difference
            s_diff = scores[order[i]] - scores[order[j]]
            
            # Sigmoid derivative
            rho = 1.0 / (1.0 + np.exp(s_diff))
            
            # Compute |ΔNDCG| if we swap i and j
            # Current contribution
            gain_i = (2 ** sorted_labels[i] - 1) / np.log2(i + 2)
            gain_j = (2 ** sorted_labels[j] - 1) / np.log2(j + 2)
            
            # After swap
            gain_i_swapped = (2 ** sorted_labels[i] - 1) / np.log2(j + 2)
            gain_j_swapped = (2 ** sorted_labels[j] - 1) / np.log2(i + 2)
            
            delta_ndcg = abs(
                (gain_i_swapped + gain_j_swapped - gain_i - gain_j) / idcg
            )
            
            # Lambda gradient
            lambda_ij = rho * delta_ndcg
            
            # Gradient direction
            if sorted_labels[i] > sorted_labels[j]:
                # i should be above j
                lambdas[order[i]] += lambda_ij
                lambdas[order[j]] -= lambda_ij
            else:
                lambdas[order[i]] -= lambda_ij
                lambdas[order[j]] += lambda_ij
    
    return lambdas
```

### 7.3 Listwise Ranking

```python
# ============================================================
# 7.3.1 ListNet Implementation
# ============================================================
class ListNet(nn.Module):
    """
    ListNet: Listwise neural ranking using top-1 probability.
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """Score all documents for a query."""
        return self.fc(x).squeeze(-1)
    
    def listwise_loss(self, scores, labels):
        """
        Listwise cross-entropy loss.
        
        Uses Plackett-Luce model (top-1 probability).
        """
        # Predicted probability distribution
        log_pred = nn.functional.log_softmax(scores, dim=0)
        
        # True probability distribution (from labels)
        log_true = nn.functional.log_softmax(
            labels.float(), dim=0
        )
        
        # Cross-entropy: KL divergence
        loss = -torch.sum(torch.exp(log_true) * (log_true - log_pred))
        return loss


def train_listnet(X, y, query_ids, input_dim, epochs=10, lr=0.001):
    """Train a ListNet model."""
    model = ListNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_queries = 0
        
        for qid in np.unique(query_ids):
            mask = query_ids == qid
            q_X = torch.FloatTensor(X[mask])
            q_y = torch.LongTensor(y[mask])
            
            scores = model(q_X)
            loss = model.listwise_loss(scores, q_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_queries += 1
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Avg Loss: {total_loss/n_queries:.4f}")
    
    return model


# ============================================================
# 7.3.2 ListMLE Implementation
# ============================================================
class ListMLE(nn.Module):
    """ListMLE: Maximum Likelihood Estimation for listwise ranking."""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)
    
    def listmle_loss(self, scores, labels):
        """
        ListMLE loss: negative log-likelihood of the ideal permutation.
        """
        # Sort by relevance (descending) to get ideal permutation
        sorted_indices = torch.argsort(labels, descending=True)
        sorted_scores = scores[sorted_indices]
        
        # NLL using Plackett-Luce model
        n = len(sorted_scores)
        loss = 0
        
        for i in range(n):
            # s_{π*(i)} - log(sum of remaining scores)
            remaining_scores = sorted_scores[i:]
            loss += -sorted_scores[i] + torch.logsumexp(remaining_scores, dim=0)
        
        return loss / n
```

### 7.4 Evaluation Utilities

```python
# ============================================================
# 7.4.1 NDCG@K Implementation
# ============================================================
def dcg_at_k(relevances, k):
    """Compute Discounted Cumulative Gain at position k."""
    relevances = np.array(relevances)[:k]
    gains = 2 ** relevances - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(relevances, k):
    """Compute Normalized DCG at position k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================
# 7.4.2 Complete Evaluation Pipeline
# ============================================================
def evaluate_ranking(model, X_test, y_test, query_ids_test, k=10):
    """
    Evaluate a ranking model using NDCG@k and MRR@k.
    """
    ndcg_scores = []
    mrr_scores = []
    
    for qid in np.unique(query_ids_test):
        mask = query_ids_test == qid
        q_X = X_test[mask]
        q_y = y_test[mask]
        
        # Predict scores
        if hasattr(model, 'score'):
            scores = model.score(torch.FloatTensor(q_X)).detach().numpy()
        elif hasattr(model, 'predict'):
            scores = model.predict(q_X)
        else:
            scores = model(torch.FloatTensor(q_X)).detach().numpy()
        
        # Rank documents
        ranked_indices = np.argsort(-scores)
        ranked_relevances = q_y[ranked_indices]
        
        # Compute metrics
        ndcg_scores.append(ndcg_at_k(ranked_relevances, k))
        
        # MRR
        for rank, rel in enumerate(ranked_relevances, 1):
            if rel >= 3:  # Consider grades 3+ as "relevant"
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)
    
    return {
        'NDCG@{}'.format(k): np.mean(ndcg_scores),
        'MRR@{}'.format(k): np.mean(mrr_scores)
    }


# ============================================================
# 7.4.3 Compare All Approaches
# ============================================================
def compare_ranking_approaches():
    """Compare pointwise, pairwise, and listwise approaches."""
    # Generate data
    X, y, query_ids = generate_ranking_data(
        n_queries=200, docs_per_query=30, n_features=15
    )
    
    # Train/test split (by query)
    n_queries = len(np.unique(query_ids))
    train_queries = np.random.choice(
        np.unique(query_ids), size=int(0.8 * n_queries), replace=False
    )
    
    train_mask = np.isin(query_ids, train_queries)
    test_mask = ~train_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    q_train, q_test = query_ids[train_mask], query_ids[test_mask]
    
    input_dim = X.shape[1]
    
    # 1. Pointwise (Gradient Boosting)
    print("=" * 50)
    print("POINTWISE (Gradient Boosting)")
    print("=" * 50)
    pw_model, _ = pointwise_rank_regression(
        X_train, y_train, X_test, q_test
    )
    pw_results = evaluate_ranking(pw_model, X_test, y_test, q_test, k=10)
    print(f"Results: {pw_results}")
    
    # 2. Pairwise (RankNet)
    print("\n" + "=" * 50)
    print("PAIRWISE (RankNet)")
    print("=" * 50)
    rw_model = train_ranknet(X_train, y_train, q_train, input_dim, epochs=5)
    rw_results = evaluate_ranking(rw_model, X_test, y_test, q_test, k=10)
    print(f"Results: {rw_results}")
    
    # 3. Listwise (ListNet)
    print("\n" + "=" * 50)
    print("LISTWISE (ListNet)")
    print("=" * 50)
    ln_model = train_listnet(X_train, y_train, q_train, input_dim, epochs=5)
    ln_results = evaluate_ranking(ln_model, X_test, y_test, q_test, k=10)
    print(f"Results: {ln_results}")
```

---

<a name="ml-connections"></a>
## 8. ML Connections & Industry Applications

### 8.1 LTR at Top Companies

```
┌────────────────────────────────────────────────────────────────┐
│  Company    │ Approach          │ Key Features                  │
├────────────┼───────────────────┼───────────────────────────────┤
│ Google     │ LambdaMART + DNN  │ Billions of queries/day       │
│            │ + BERT re-ranker  │ Freshness, personalization    │
├────────────┼───────────────────┼───────────────────────────────┤
│ Microsoft  │ LambdaMART ( Bing │ LETOR benchmark creator       │
│ (Bing)     │ was key paper)    │ Neural ranking + tree models  │
├────────────┼───────────────────┼───────────────────────────────┤
│ Amazon     │ Gradient Boosted  │ Personalization, conversion   │
│ (Product   │ Trees + Neural    │ rating prediction, diversity  │
│  Search)   │ Networks          │                               │
├────────────┼───────────────────┼───────────────────────────────┤
│ Meta       │ DSSM + Cross-     │ Social search, news feed      │
│ (Facebook) │ Encoder           │ ranking, ad ranking           │
├────────────┼───────────────────┼───────────────────────────────┤
│ Netflix    │ Matrix Factoriz.  │ Row-wise (personalization)    │
│            │ + Deep Learning   │ + context-aware ranking       │
├────────────┼───────────────────┼───────────────────────────────┤
│ LinkedIn   │ Gradient Boosted  │ Feed ranking, job             │
│            │ Trees             │ recommendations, search       │
├────────────┼───────────────────┼───────────────────────────────┤
│ Uber       │ XGBoost + DNN     │ Driver matching, ETA          │
│            │                   │ prediction, dispatch          │
└────────────┴───────────────────┴───────────────────────────────┘
```

### 8.2 Connections to Other ML Areas

```
┌──────────────────────────────────────────────────────────┐
│  LTR ↔ Recommendation Systems                            │
│  • Ranking items for a user ≈ ranking docs for a query  │
│  • Same loss functions apply (pairwise, listwise)       │
│  • Collaborative filtering as feature in LTR            │
│                                                          │
│  LTR ↔ NLP                                               │
│  • BERT/reranking models are LTR at core                │
│  • Query understanding via NLP                          │
│  • Semantic search = neural ranking                     │
│                                                          │
│  LTR ↔ Computer Vision                                  │
│  • Visual search: rank images by visual similarity      │
│  • Multi-modal ranking (text + image features)          │
│  • Feature extraction from images as ranking features   │
│                                                          │
│  LTR ↔ Reinforcement Learning                           │
│  • Position-based models use RL                        │
│  • Learning from clicks (bandits)                      │
│  • Counterfactual learning                             │
│                                                          │
│  LTR ↔ Causal Inference                                 │
│  • Position bias in click data                         │
│  • Unbiased learning-to-rank (ULTR)                    │
│  • Propensity scoring for click models                 │
└──────────────────────────────────────────────────────────┘
```

---

<a name="exercises"></a>
## 9. Exercises

### Exercise 1: Pointwise Ranking (Beginner)
```python
# Use sklearn to build a pointwise ranking model:
# 1. Generate synthetic ranking data (100 queries, 20 docs each)
# 2. Train a RandomForestRegressor on relevance grades
# 3. Evaluate using NDCG@5 and NDCG@10
# 4. Compare with a LinearRegression baseline
```

### Exercise 2: Pairwise Loss from Scratch (Intermediate)
```python
# Implement pairwise hinge loss:
# L = Σ max(0, 1 - (sᵢ - sⱼ)) for all pairs where yᵢ > yⱼ
# 1. Implement the loss function
# 2. Train a linear model using gradient descent
# 3. Evaluate on ranking data
# 4. Compare training time with pointwise approach
```

### Exercise 3: RankNet with PyTorch (Intermediate)
```python
# Build a complete RankNet pipeline:
# 1. Implement RankNet with 2 hidden layers
# 2. Implement efficient pair generation (skip equal labels)
# 3. Train on MSLR-WEB10K or synthetic data
# 4. Plot training loss over epochs
# 5. Evaluate NDCG@k at each epoch
```

### Exercise 4: LambdaMART with LightGBM (Advanced)
```python
# LambdaMART is available in LightGBM:
# 1. Use lightgbm.LGBMRanker with lambdarank objective
# 2. Prepare data with proper group structure
# 3. Tune hyperparameters (n_estimators, learning_rate, num_leaves)
# 4. Compare with pointwise GBDT
# 5. Analyze feature importance
```

### Exercise 5: Neural Cross-Encoder Re-ranker (Advanced)
```python
# Build a neural re-ranking system:
# 1. Use sentence-transformers to encode queries and docs
# 2. Implement a bi-encoder for initial retrieval
# 3. Implement a cross-encoder for re-ranking (BERT-based)
# 4. Build a two-stage pipeline
# 5. Measure latency trade-off between stages
```

### Exercise 6: Listwise Loss Comparison (Advanced)
```python
# Compare different listwise losses:
# 1. Implement ListNet (cross-entropy on permutation)
# 2. Implement ListMLE (NLL on ideal permutation)
# 3. Implement ApproxNDCG loss
# 4. Train all on same data
# 5. Plot NDCG@10 vs. training epochs for each
# 6. Analyze convergence speed and final performance
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    KEY TAKEAWAYS                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Pointwise: Score individual documents                           │
│    • Simple but ignores inter-document relationships             │
│    • MSE / Cross-Entropy loss                                   │
│                                                                  │
│  Pairwise: Compare document pairs                                │
│    • RankNet: sigmoid + cross-entropy on pairs                  │
│    • LambdaRank: weighted gradients by ΔNDCG                    │
│    • Better than pointwise but O(n²) complexity                 │
│                                                                  │
│  Listwise: Optimize entire list                                  │
│    • ListNet: probability distribution over permutations         │
│    • ListMLE: NLL of ideal ranking                               │
│    • LambdaMART: best overall performance                        │
│                                                                  │
│  Production: Two-stage architecture is standard                  │
│    • Stage 1: Fast retrieval (BM25, bi-encoder, ANN)            │
│    • Stage 2: Careful re-ranking (cross-encoder, LambdaMART)    │
│                                                                  │
│  Key formula: P(dᵢ ≻ dⱼ) = σ(f(q,dᵢ) - f(q,dⱼ))              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
