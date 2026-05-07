# Evaluation Metrics: Offline (MRR@K, MAP@K, NDCG@K) & Online (CTR, Session Metrics) — Complete Guide

## Table of Contents
1. [Introduction to Recommendation Evaluation](#introduction)
2. [Offline Evaluation Metrics](#offline)
3. [NDCG@K — Normalized Discounted Cumulative Gain](#ndcg)
4. [MAP@K — Mean Average Precision](#map)
5. [MRR@K — Mean Reciprocal Rank](#mrr)
6. [Other Offline Metrics](#other-offline)
7. [Online Evaluation Metrics](#online)
8. [Metric Selection Guide](#selection)
9. [Python Implementations](#python)
10. [Exercises](#exercises)

---

<a name="introduction"></a>
## 1. Introduction to Recommendation Evaluation

### 1.1 Why Evaluation Matters

```
┌────────────────────────────────────────────────────────────────┐
│           THE EVALUATION HIERARCHY                             │
│                                                                │
│  Level 1: Offline Metrics (this lesson's focus)               │
│  ┌────────────────────────────────────────────┐               │
│  │ NDCG@K, MAP@K, MRR@K, Recall@K, HR@K     │               │
│  │ Hit Rate, Coverage, Diversity, Novelty    │               │
│  │                                            │               │
│  │ + Fast, cheap, reproducible                │               │
│  │ - May not reflect real user behavior       │               │
│  └────────────────────────────────────────────┘               │
│                        │                                        │
│                        ▼                                        │
│  Level 2: Online Metrics (A/B Testing)                        │
│  ┌────────────────────────────────────────────┐               │
│  │ CTR, Conversion Rate, Revenue, Session     │               │
│  │ length, Bounce Rate, Engagement            │               │
│  │                                            │               │
│  │ + Reflects real user behavior              │               │
│  │ - Expensive, slow, requires traffic        │               │
│  └────────────────────────────────────────────┘               │
│                        │                                        │
│                        ▼                                        │
│  Level 3: Business Metrics                                     │
│  ┌────────────────────────────────────────────┐               │
│  │ Revenue, Retention, LTV, NPS               │               │
│  │                                            │               │
│  │ + Ultimate measure of success              │               │
│  │ - Slowest feedback, influenced by many     │               │
│  │   factors beyond the recommender           │               │
│  └────────────────────────────────────────────┘               │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Types of Recommendation Tasks

```
┌────────────────────────────────────────────────────────────────┐
│  Task              │ Description              │ Best Metrics   │
├────────────────────┼──────────────────────────┼────────────────┤
│ Point-wise         │ Predict relevance score  │ RMSE, MAE,    │
│ prediction         │ for a single item        │ LogLoss       │
├────────────────────┼──────────────────────────┼────────────────┤
│ Top-K              │ Rank and return K best   │ NDCG@K, MAP@K │
│ recommendation     │ items for a user/query   │ HR@K, Recall@K│
├────────────────────┼──────────────────────────┼────────────────┤
│ Classification     │ Is item relevant or not? │ AUC, F1,      │
│ (binary)           │                           │ Precision     │
├────────────────────┼──────────────────────────┼────────────────┤
│ Next-item          │ Predict the next item    │ MRR@K,        │
│ prediction         │ a user will interact     │ NDCG@K        │
├────────────────────┼──────────────────────────┼────────────────┤
│ Sequential         │ Recommend a sequence     │ SR@K, NDCG@K  │
│ recommendation     │ of items                 │ Session metrics│
├────────────────────┼──────────────────────────┼────────────────┤
│ Similarity search  │ Find similar items       │ Recall@K,     │
│                    │ to a given item/query    │ NDCG@K        │
└────────────────────┴──────────────────────────┴────────────────┘
```

---

<a name="offline"></a>
## 2. Offline Evaluation Metrics

### 2.1 Relevance Labels

```
Most ranking metrics depend on relevance labels:

Binary relevance:
  0 = Not relevant, 1 = Relevant

Graded relevance:
  0 = Not relevant
  1 = Weakly relevant (marginally useful)
  2 = Fairly relevant (partially answers the query)
  3 = Highly relevant (useful answer)
  4 = Perfectly relevant (ideal result)

Example: Search for "python machine learning"
  ┌─────────────────────────────────┬─────────┬────────┐
  │ Document                        │ Binary  │ Graded │
  ├─────────────────────────────────┼─────────┼────────┤
  │ scikit-learn docs               │    1    │   4    │
  │ Andrew Ng's ML course           │    1    │   4    │
  │ Hands-On ML book (Géron)        │    1    │   3    │
  │ Python tutorials (W3Schools)    │    1    │   2    │
  │ Wikipedia: Python               │    0    │   1    │
  │ Random cooking blog             │    0    │   0    │
  └─────────────────────────────────┴─────────┴────────┘
```

---

<a name="ndcg"></a>
## 3. NDCG@K — Normalized Discounted Cumulative Gain

### 3.1 Definition

```
NDCG measures the quality of a ranked list, accounting for:
  1. Relevance of each item (graded relevance)
  2. Position of each item (higher positions matter more)

Step 1: Compute DCG (Discounted Cumulative Gain)

  DCG@K = Σ(i=1 to K) gain(i) / discount(i)

  where:
    gain(i) = 2^rel_i - 1    (exponential gain from relevance grade)
    discount(i) = log₂(i + 1) (logarithmic position discount)

  Or (alternative formulation used by some systems):
    DCG@K = Σ(i=1 to K) rel_i / log₂(i + 1)    (linear gain)

Step 2: Compute Ideal DCG (IDCG)

  IDCG@K = DCG@K of the ideally ranked list
  (documents sorted by relevance, highest first)

Step 3: Normalize

  NDCG@K = DCG@K / IDCG@K

  NDCG@K ∈ [0, 1]
  1.0 = perfect ranking (all relevant items at top in correct order)
  0.0 = no relevant items in top K
```

### 3.2 Worked Example

```
Query: "best laptops 2024"
K = 5

Predicted ranking (our system):
  Position 1: Dell XPS 15      (relevance: 3)
  Position 2: MacBook Pro      (relevance: 2)
  Position 3: ThinkPad X1      (relevance: 3)
  Position 4: HP Spectre       (relevance: 0)
  Position 5: Acer Swift       (relevance: 1)

Step 1: DCG@5
  DCG = (2³-1)/log₂(2) + (2²-1)/log₂(3) + (2³-1)/log₂(4) + 
        (2⁰-1)/log₂(5) + (2¹-1)/log₂(6)
      = 7/1 + 3/1.585 + 7/2 + 0/2.322 + 1/2.585
      = 7 + 1.893 + 3.5 + 0 + 0.387
      = 12.78

Step 2: IDCG@5 (ideal order: 3, 3, 2, 1, 0)
  IDCG = (2³-1)/log₂(2) + (2³-1)/log₂(3) + (2²-1)/log₂(4) +
         (2¹-1)/log₂(5) + (2⁰-1)/log₂(6)
       = 7/1 + 7/1.585 + 3/2 + 1/2.322 + 0/2.585
       = 7 + 4.416 + 1.5 + 0.431 + 0
       = 13.347

Step 3: NDCG@5
  NDCG = 12.78 / 13.347 = 0.9575

ASCII visualization:
  Relevance profile of our ranking:
  Position:  1     2     3     4     5
  Relevance: ███   ██    ███   ░     █
             3     2     3     0     1

  Ideal relevance profile:
  Position:  1     2     3     4     5
  Relevance: ███   ███   ██    █     ░
             3     3     2     1     0

  The penalty comes from having relevance 3 at position 3
  instead of position 2.
```

### 3.3 Discount Function Visualization

```
  Gain & Discount Analysis for K=10:

  Relevance grade:    4     3     2     1     0
  Exponential gain:   15    7     3     1     0

  Position discount:  1.00  1.59  2.00  2.32  2.59  2.81  3.00  3.17  3.32  3.46
  (log₂(pos+1))

  Effective value of relevance 4 at each position:
  Pos 1:  15.00  ████████████████  (100%)
  Pos 2:   9.44  ██████████        (63%)
  Pos 3:   7.50  ████████          (50%)
  Pos 4:   6.47  ███████           (43%)
  Pos 5:   5.79  ██████            (39%)
  Pos 10:  4.33  █████             (29%)

  Key insight: A highly relevant item at position 1 is worth
  3x more than the same item at position 10!
```

### 3.4 Mean NDCG@K

```
For evaluation over multiple queries/users:

  Mean NDCG@K = (1/Q) Σ(q=1 to Q) NDCG@K(q)

  where Q is the number of queries/users.

Example:
  Query 1: NDCG@5 = 0.95
  Query 2: NDCG@5 = 0.72
  Query 3: NDCG@5 = 0.88
  Query 4: NDCG@5 = 0.45
  Query 5: NDCG@5 = 0.91

  Mean NDCG@5 = (0.95 + 0.72 + 0.88 + 0.45 + 0.91) / 5 = 0.782
```

---

<a name="map"></a>
## 4. MAP@K — Mean Average Precision

### 4.1 Definition

```
MAP@K measures the average precision across all relevant items,
considering their positions in the ranked list.

Step 1: Precision@K
  Precision@K = (# of relevant items in top K) / K

Step 2: Average Precision@K
  AP@K = (1/min(R, K)) × Σ(k=1 to K) P(k) × rel(k)

  where:
    R = total number of relevant items
    P(k) = precision at position k
    rel(k) = 1 if item at position k is relevant, 0 otherwise

Step 3: Mean Average Precision
  MAP@K = (1/Q) Σ(q=1 to Q) AP@K(q)
```

### 4.2 Worked Example

```
Query: "machine learning books"
K = 6
Total relevant items: R = 3

Predicted ranking:
  Position 1: "Deep Learning" (Goodfellow)     → RELEVANT
  Position 2: "ML for Dummies"                  → NOT RELEVANT
  Position 3: "Pattern Recognition" (Bishop)    → RELEVANT
  Position 4: "Cooking Italian"                 → NOT RELEVANT
  Position 5: "Hands-On ML" (Géron)             → RELEVANT
  Position 6: "Gardening 101"                   → NOT RELEVANT

Step-by-step calculation:
  ┌───────────┬────────┬───────────┬──────────┬──────────────┐
  │ Position k│ rel(k) │ # relevant│ Prec@k   │ rel(k)*Prec@k│
  ├───────────┼────────┼───────────┼──────────┼──────────────┤
  │     1     │   1    │     1     │  1/1=1.0 │    1.0       │
  │     2     │   0    │     1     │  1/2=0.5 │    0.0       │
  │     3     │   1    │     2     │  2/3=0.67│    0.67      │
  │     4     │   0    │     2     │  2/4=0.5 │    0.0       │
  │     5     │   1    │     3     │  3/5=0.6 │    0.6       │
  │     6     │   0    │     3     │  3/6=0.5 │    0.0       │
  └───────────┴────────┴───────────┴──────────┴──────────────┘

  AP@6 = (1/3) × (1.0 + 0 + 0.67 + 0 + 0.6 + 0)
       = (1/3) × 2.27
       = 0.756

Intuition:
  AP rewards:
  ✓ Having relevant items at the top (high precision early)
  ✓ Finding ALL relevant items (not just some)
  
  AP penalizes:
  ✗ Relevant items appearing late in the list
  ✗ Missing relevant items entirely
```

### 4.3 AP vs. Precision@K

```
Two systems with same Precision@3 = 2/3:

  System A:
  Position 1: RELEVANT    ← Precision@1 = 1.0
  Position 2: RELEVANT    ← Precision@2 = 1.0
  Position 3: NOT         ← Precision@3 = 0.67
  AP@3 = (1.0 + 1.0 + 0) / 3 = 0.667  (higher AP)

  System B:
  Position 1: NOT         ← Precision@1 = 0.0
  Position 2: RELEVANT    ← Precision@2 = 0.5
  Position 3: RELEVANT    ← Precision@3 = 0.67
  AP@3 = (0 + 0.5 + 0.67) / 3 = 0.389  (lower AP)

Same Precision@K but different AP!
AP captures the ORDERING of relevant items.
```

---

<a name="mrr"></a>
## 5. MRR@K — Mean Reciprocal Rank

### 5.1 Definition

```
MRR focuses on the position of the FIRST relevant item.

  RR@K = 1 / rank_of_first_relevant_item  (if found in top K)
  RR@K = 0                                 (if no relevant item in top K)

  MRR@K = (1/Q) Σ(q=1 to Q) RR@K(q)

MRR is ideal when the user only cares about the first correct result:
  • Question answering (which answer is correct?)
  • Navigation queries ("Facebook login")
  • Factoid queries ("capital of France")
```

### 5.2 Worked Example

```
Query: "python for loop syntax"

  System A:                            System B:
  1. Python tutorial  ✓ RELEVANT       1. Java tutorial
  2. C++ tutorial                      2. Ruby tutorial
  3. Java tutorial                     3. Python tutorial ✓ RELEVANT

  RR@3(A) = 1/1 = 1.0
  RR@3(B) = 1/3 = 0.333

  MRR strongly rewards putting ANY relevant item at position 1.

Comparison with NDCG:
  If both systems have 3 relevant items in positions 1-10:
  • MRR only cares about position 1
  • NDCG cares about positions 1, 2, 3, ... (all positions)
```

### 5.3 Metric Comparison

```
  System rankings for 3 queries:
  ┌────────┬───────────────────────────────┬────────┬────────┬────────┐
  │ Query  │ Ranked Relevance             │ NDCG@5 │ MAP@5  │ MRR@5  │
  ├────────┼───────────────────────────────┼────────┼────────┼────────┤
  │   Q1   │ [4, 3, 2, 1, 0]  (perfect!) │  1.000 │  1.000 │  1.000 │
  │   Q2   │ [3, 0, 0, 4, 2]             │  0.748 │  0.500 │  1.000 │
  │   Q3   │ [0, 0, 3, 0, 0]             │  0.350 │  0.167 │  0.333 │
  ├────────┼───────────────────────────────┼────────┼────────┼────────┤
  │ Mean   │                              │  0.699 │  0.556 │  0.778 │
  └────────┴───────────────────────────────┴────────┴────────┴────────┘

  Note: Q2 has high MRR (relevant at pos 1) but low MAP
  because the other relevant items are at positions 4 and 5.
```

---

<a name="other-offline"></a>
## 6. Other Offline Metrics

### 6.1 Recall@K and Hit Rate@K

```
Recall@K = (# relevant items in top K) / (total # relevant items)

Hit Rate@K = 1 if at least one relevant item in top K, else 0
  (equivalent to Recall@K > 0 for binary case)

Mean Reciprocal Rank of Hit Rate = proportion of queries with at least
one relevant result in top K.

Example:
  Total relevant items for query: R = 5
  Top K = 10, found 3 relevant items
  
  Recall@10 = 3/5 = 0.6
  
  If no relevant items in top 10: Recall@10 = 0/5 = 0

  ┌─────────────────────────────────────────┐
  │  Recall@K grows with K:                │
  │                                         │
  │  1.0 ┤                    ┌──────       │
  │      │               ┌───┘             │
  │  0.5 ┤          ┌────┘                  │
  │      │     ┌────┘                       │
  │  0.0 ┤─────┘                            │
  │      └──┬────┬────┬────┬────┬────┬──    │
  │         1    5    10   20   50  100    │
  │                    K                     │
  └─────────────────────────────────────────┘
```

### 6.2 Coverage, Diversity, Novelty

```
┌──────────────────────────────────────────────────────────────────┐
│  Beyond Accuracy: Beyond-Relevance Metrics                       │
├──────────────────┬───────────────────────────────────────────────┤
│ Metric           │ Formula / Description                         │
├──────────────────┼───────────────────────────────────────────────┤
│ Catalog Coverage │ % of items recommended at least once          │
│                  │ = |∪ᵤ Rec(u)| / |Catalog|                     │
├──────────────────┼───────────────────────────────────────────────┤
│ User Coverage    │ % of users receiving recommendations          │
│                  │ = |users with recs| / |all users|             │
├──────────────────┼───────────────────────────────────────────────┤
│ Intra-list       │ Average pairwise dissimilarity in top-K      │
│ Diversity        │ = (2/K(K-1)) Σᵢ<ⱼ d(itemᵢ, itemⱼ)          │
├──────────────────┼───────────────────────────────────────────────┤
│ ILS (Intra-List  │ 1 - Intra-list similarity                   │
│ Similarity)      │ Higher ILS = more diverse                    │
├──────────────────┼───────────────────────────────────────────────┤
│ Novelty          │ Average inverse popularity of recs           │
│                  │ = (1/K) Σᵢ -log₂(popularity(itemᵢ))         │
├──────────────────┼───────────────────────────────────────────────┤
│ Serendipity      │ Relevant + unexpected recommendations        │
│                  │ serendipity = relevance × unexpectedness     │
├──────────────────┼───────────────────────────────────────────────┤
│ Calibration      │ P(relevant|predicted_score=s) ≈ s           │
│                  │ Are predicted scores well calibrated?         │
└──────────────────┴───────────────────────────────────────────────┘
```

### 6.3 Calibration and Fairness Metrics

```
Calibration metric:
  ECE (Expected Calibration Error):
  ECE = Σ(b=1 to B) (n_b / n) × |accuracy_b - confidence_b|

  Divide predictions into B bins by confidence.
  For each bin, compare actual accuracy with average confidence.

Fairness metrics:
  ┌────────────────────────────────────────────────────┐
  │ Demographic Parity:                               │
  │   P(rec_item_in_group_A) = P(rec_item_in_group_B) │
  │                                                    │
  │ Equal Opportunity:                                │
  │   P(rec_relevant | group_A) = P(rec_relevant | B) │
  │                                                    │
  │ Exposure Equity:                                   │
  │   Each item gets fair exposure across users       │
  │   (important for multi-sided platforms)           │
  └────────────────────────────────────────────────────┘
```

---

<a name="online"></a>
## 7. Online Evaluation Metrics

### 7.1 Click-Through Rate (CTR)

```
CTR = (# clicks) / (# impressions) × 100%

Formula:
  CTR = clicks / impressions

  For a specific position k:
  CTR@k = clicks_at_position_k / impressions_at_position_k

  Weighted CTR:
  CTR_w = Σ(k=1 to K) p_k × CTR@k
  where p_k = probability of user scrolling to position k

Position bias in CTR:
  ┌──────────────────────────────────────────┐
  │  Position │  Avg CTR  │  Bias Factor    │
  ├──────────┼───────────┼─────────────────┤
  │     1    │   15-25%  │  Very high      │
  │     2    │   8-15%   │  High           │
  │     3    │   5-10%   │  Moderate       │
  │     5    │   2-5%    │  Moderate       │
  │    10    │   0.5-2%  │  Low            │
  │    20    │   0.1-0.5%│  Very low       │
  └──────────┴───────────┴─────────────────┘

  Users click position 1 items much more regardless of relevance!
  → Need position debiasing (IPS, PBM models)
```

### 7.2 Conversion Rate

```
Conversion Rate = (# conversions) / (# clicks or impressions) × 100%

Types:
  CTR → CVR (Click-to-Conversion):
    CVR = purchases / clicks
    
  Impression-to-Conversion:
    CVR = purchases / impressions

  Funnel analysis:
  ┌──────────────────────────────────────────────┐
  │ Impressions (1,000,000)                      │
  │     │ 10% CTR                                │
  │     ▼                                        │
  │ Clicks (100,000)                             │
  │     │ 5% CVR                                 │
  │     ▼                                        │
  │ Add to Cart (5,000)                          │
  │     │ 60% completion                          │
  │     ▼                                        │
  │ Purchases (3,000)                            │
  │     │ $50 avg order value                    │
  │     ▼                                        │
  │ Revenue ($150,000)                           │
  └──────────────────────────────────────────────┘
```

### 7.3 Session Metrics

```
Session-level metrics capture the quality of an entire interaction session:

┌─────────────────────────────────────────────────────────────────┐
│  Session Metric          │ Definition                           │
├─────────────────────────┼──────────────────────────────────────┤
│ Session Length           │ # of items viewed in a session       │
│                         │ Longer = more engaged                 │
├─────────────────────────┼──────────────────────────────────────┤
│ Session Success Rate    │ % of sessions with ≥1 click           │
│                         │ SR = sessions_with_clicks / total    │
├─────────────────────────┼──────────────────────────────────────┤
│ Dwell Time              │ Time spent on a clicked item          │
│                         │ Longer = more useful                  │
├─────────────────────────┼──────────────────────────────────────┤
│ Bounce Rate             │ % of sessions with 0 interactions     │
│                         │ Lower = better                        │
├─────────────────────────┼──────────────────────────────────────┤
│ Click Position          │ Average position of clicked items    │
│                         │ Lower = better (found what they need) │
├─────────────────────────┼──────────────────────────────────────┤
│ Last Click Position     │ Furthest position user scrolled to   │
│                         │ Higher = exploring more results       │
├─────────────────────────┼──────────────────────────────────────┤
│ Scroll Depth            │ % of page scrolled                    │
│                         │ Higher = more engaged                 │
├─────────────────────────┼──────────────────────────────────────┤
│ Search Satisfaction     │ User-reported (thumbs up/down)        │
│                         │ Most direct signal                    │
└─────────────────────────┴──────────────────────────────────────┘
```

### 7.4 Engagement Metrics

```
Long-term engagement (requires longitudinal tracking):

  DAU/MAU Ratio (Stickiness):
    stickiness = DAU / MAU
    > 20% = excellent, > 10% = good

  Retention:
    Day-1 retention: % of users who return next day
    Day-7 retention: % who return after 7 days
    Day-30 retention: % who return after 30 days

  Session frequency:
    Sessions per user per week/month

  Revenue per user:
    ARPU = total_revenue / active_users

  ┌──────────────────────────────────────────────────┐
  │  Engagement Decay After Change:                 │
  │                                                  │
  │  Metric      │  Before │ After 7d │ After 30d   │
  │  CTR         │  5.2%   │  5.1%   │  4.9%       │
  │  Conv Rate   │  2.1%   │  2.0%   │  2.0%       │
  │  Session Len │  8.3    │  8.1    │  7.9        │
  │  Retention   │  45%    │  44%    │  43%        │
  │                                                  │
  │  Monitor: short-term lift may decay over time!  │
  └──────────────────────────────────────────────────┘
```

---

<a name="selection"></a>
## 8. Metric Selection Guide

### 8.1 When to Use Each Metric

```
┌──────────────────────┬──────────────────────────────────────────┐
│ Metric               │ When to Use                              │
├──────────────────────┼──────────────────────────────────────────┤
│ NDCG@K               │ Graded relevance, position-sensitive     │
│                      │ Web search, document ranking             │
│                      │ Standard in most LTR competitions        │
├──────────────────────┼──────────────────────────────────────────┤
│ MAP@K                │ Binary relevance, care about all rel     │
│                      │ items being found                        │
│                      │ Information retrieval, document ranking  │
│                      │ Good when you need ALL relevant results  │
├──────────────────────┼──────────────────────────────────────────┤
│ MRR@K                │ First relevant result is most important  │
│                      │ QA systems, navigation queries           │
│                      │ Voice assistants, chatbots               │
├──────────────────────┼──────────────────────────────────────────┤
│ Hit Rate@K / Recall  │ Any relevant item is OK                  │
│                      │ Recommendation systems                   │
│                      │ Explore accuracy, coverage              │
├──────────────────────┼──────────────────────────────────────────┤
│ CTR                  │ Online, click-based evaluation           │
│                      │ Ad ranking, feed ranking                 │
│                      │ Quick feedback signal                    │
├──────────────────────┼──────────────────────────────────────────┤
│ Conversion Rate      │ E-commerce, lead generation              │
│                      │ Bottom-line business impact              │
├──────────────────────┼──────────────────────────────────────────┤
│ Session Success Rate │ Sequential recommendation               │
│                      │ Music, video, content feeds             │
├──────────────────────┼──────────────────────────────────────────┤
│ NDCG + Diversity     │ When accuracy alone isn't enough        │
│ + Novelty            │ Need balanced recommendations           │
└──────────────────────┴──────────────────────────────────────────┘
```

### 8.2 Common Pitfalls

```
┌──────────────────────────────────────────────────────────────────┐
│  PITFALL                          │ HOW TO AVOID                  │
├───────────────────────────────────┼──────────────────────────────┤
│ Optimizing NDCG offline doesn't   │ Use online A/B tests to       │
│ guarantee online CTR improvement  │ validate offline findings     │
├───────────────────────────────────┼──────────────────────────────┤
│ Position bias inflates CTR        │ Use position debiasing        │
│ (top items always get more clicks)│ (IPS, PBM, propensity models)│
├───────────────────────────────────┼──────────────────────────────┤
│ Popularity bias: popular items    │ Monitor diversity metrics     │
│ always win offline metrics        │ alongside accuracy            │
├───────────────────────────────────┼──────────────────────────────┤
│ Metric gaming: models optimize    │ Use guardrail metrics          │
│ one metric at expense of others   │ (e.g., diversity, coverage)  │
├───────────────────────────────────┼──────────────────────────────┤
│ Cold-start items never evaluated  │ Evaluate separately on        │
│ (no clicks → no offline signal)   │ cold-start items              │
├───────────────────────────────────┼──────────────────────────────┤
│ Train-test temporal leakage       │ Use temporal splitting         │
│ (future data in training)         │ (train on past, test on future)│
├───────────────────────────────────┼──────────────────────────────┤
│ Single metric tunnel vision       │ Report a dashboard of         │
│                                   │ complementary metrics         │
└───────────────────────────────────┴──────────────────────────────┘
```

---

<a name="python"></a>
## 9. Python Implementations

```python
import numpy as np
from typing import List, Union

# ============================================================
# 9.1 NDCG@K
# ============================================================
def dcg_at_k(relevances: np.ndarray, k: int, method='exp') -> float:
    """
    Compute Discounted Cumulative Gain at position k.
    
    Parameters:
        relevances: Array of relevance grades
        k: Cutoff position
        method: 'exp' (2^rel-1) or 'linear' (rel)
    """
    relevances = np.array(relevances, dtype=float)[:k]
    if len(relevances) == 0:
        return 0.0
    
    if method == 'exp':
        gains = 2 ** relevances - 1
    else:
        gains = relevances
    
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(relevances: np.ndarray, k: int, method='exp') -> float:
    """Compute Normalized DCG at position k."""
    dcg = dcg_at_k(relevances, k, method)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k, method)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(all_relevances: List[np.ndarray], k: int,
                    method='exp') -> float:
    """Compute Mean NDCG@K across multiple queries."""
    scores = [ndcg_at_k(rel, k, method) for rel in all_relevances]
    return np.mean(scores)


# ============================================================
# 9.2 MAP@K
# ============================================================
def average_precision_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Average Precision at position k.
    
    Parameters:
        relevances: Binary or graded relevance array
        k: Cutoff position
    """
    relevances = np.array(relevances)[:k]
    n_relevant = np.sum(relevances > 0)
    
    if n_relevant == 0:
        return 0.0
    
    # Cumulative count of relevant items
    cumsum_relevant = np.cumsum(relevances > 0)
    
    # Precision at each position
    positions = np.arange(1, len(relevances) + 1)
    precisions = cumsum_relevant / positions
    
    # Average precision: only sum at relevant positions
    ap = np.sum(precisions * (relevances > 0)) / n_relevant
    return ap


def mean_average_precision_at_k(all_relevances: List[np.ndarray],
                                  k: int) -> float:
    """Compute Mean Average Precision at K across queries."""
    scores = [average_precision_at_k(rel, k) for rel in all_relevances]
    return np.mean(scores)


# ============================================================
# 9.3 MRR@K
# ============================================================
def reciprocal_rank_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Reciprocal Rank at position k.
    
    Returns 1/rank of first relevant item, or 0 if none in top K.
    """
    relevances = np.array(relevances)[:k]
    
    for rank, rel in enumerate(relevances, 1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_at_k(all_relevances: List[np.ndarray],
                                k: int) -> float:
    """Compute Mean Reciprocal Rank at K across queries."""
    scores = [reciprocal_rank_at_k(rel, k) for rel in all_relevances]
    return np.mean(scores)


# ============================================================
# 9.4 Recall@K and Hit Rate@K
# ============================================================
def recall_at_k(relevances: np.ndarray, k: int,
                total_relevant: int = None) -> float:
    """Compute Recall@K."""
    relevances = np.array(relevances)[:k]
    if total_relevant is None:
        total_relevant = np.sum(np.array(relevances) > 0)
    if total_relevant == 0:
        return 0.0
    return np.sum(relevances > 0) / total_relevant


def hit_rate_at_k(all_relevances: List[np.ndarray], k: int) -> float:
    """Compute Hit Rate@K (proportion of queries with ≥1 relevant)."""
    hits = 0
    for rel in all_relevances:
        if np.sum(np.array(rel)[:k] > 0) > 0:
            hits += 1
    return hits / len(all_relevances)


# ============================================================
# 9.5 Diversity and Novelty
# ============================================================
def intra_list_diversity(item_features: np.ndarray, k: int) -> float:
    """
    Compute Intra-List Diversity.
    Average pairwise distance between top-K items.
    
    Parameters:
        item_features: Feature matrix (n_items, n_features)
        k: Number of items in the list
    """
    from sklearn.metrics.pairwise import cosine_distances
    
    top_k_features = item_features[:k]
    n = len(top_k_features)
    
    if n < 2:
        return 0.0
    
    # Pairwise distances
    distances = cosine_distances(top_k_features)
    
    # Average upper triangle
    sum_dist = np.sum(np.triu(distances, k=1))
    ild = sum_dist / (n * (n - 1) / 2)
    
    return ild


def novelty_at_k(recommended_items: List[int],
                 item_popularity: dict, k: int) -> float:
    """
    Compute Novelty@K.
    Average inverse popularity of recommended items.
    
    Higher novelty = recommending less popular (more surprising) items.
    """
    items = recommended_items[:k]
    novelty_scores = []
    
    for item in items:
        pop = item_popularity.get(item, 1)
        novelty_scores.append(-np.log2(pop))
    
    return np.mean(novelty_scores)


# ============================================================
# 9.6 Complete Evaluation Dashboard
# ============================================================
def evaluate_recommender(recommendations, ground_truth, k_values=[5, 10, 20]):
    """
    Comprehensive evaluation of a recommender system.
    
    Parameters:
        recommendations: Dict[user_id, List[item_id]] (ranked)
        ground_truth: Dict[user_id, set(item_id)] (relevant items)
        k_values: List of K values to evaluate
    """
    results = {}
    
    for k in k_values:
        ndcg_scores = []
        map_scores = []
        mrr_scores = []
        recall_scores = []
        hr_scores = []
        
        for user in ground_truth:
            rel_items = ground_truth[user]
            rec_items = recommendations.get(user, [])
            
            # Create relevance array
            relevances = np.array([1 if item in rel_items else 0
                                   for item in rec_items])
            
            ndcg_scores.append(ndcg_at_k(relevances, k))
            map_scores.append(average_precision_at_k(relevances, k))
            mrr_scores.append(reciprocal_rank_at_k(relevances, k))
            recall_scores.append(recall_at_k(
                relevances, k, len(rel_items)))
            hr_scores.append(1 if np.sum(relevances[:k]) > 0 else 0)
        
        results[f'NDCG@{k}'] = np.mean(ndcg_scores)
        results[f'MAP@{k}'] = np.mean(map_scores)
        results[f'MRR@{k}'] = np.mean(mrr_scores)
        results[f'Recall@{k}'] = np.mean(recall_scores)
        results[f'HR@{k}'] = np.mean(hr_scores)
    
    return results


# ============================================================
# 9.7 Evaluation Demo
# ============================================================
def evaluation_demo():
    """Demonstrate all evaluation metrics."""
    np.random.seed(42)
    
    # Simulate 100 queries with relevance grades
    queries = []
    for _ in range(100):
        n_docs = np.random.randint(10, 50)
        rels = np.random.choice([0, 0, 0, 1, 1, 2, 3, 4], size=n_docs)
        queries.append(rels)
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for k in [5, 10, 20]:
        print(f"\n--- K = {k} ---")
        print(f"  NDCG@{k}:  {mean_ndcg_at_k(queries, k):.4f}")
        print(f"  MAP@{k}:   {mean_average_precision_at_k(queries, k):.4f}")
        print(f"  MRR@{k}:   {mean_reciprocal_rank_at_k(queries, k):.4f}")
        print(f"  HR@{k}:    {hit_rate_at_k(queries, k):.4f}")
    
    # Detailed single query analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS FOR SINGLE QUERY")
    print("=" * 60)
    
    query_rels = np.array([4, 3, 0, 2, 0, 1, 0, 3, 0, 0])
    
    print(f"Relevance grades: {query_rels}")
    print(f"  NDCG@5:  {ndcg_at_k(query_rels, 5):.4f}")
    print(f"  NDCG@10: {ndcg_at_k(query_rels, 10):.4f}")
    print(f"  MAP@5:   {average_precision_at_k(query_rels, 5):.4f}")
    print(f"  MAP@10:  {average_precision_at_k(query_rels, 10):.4f}")
    print(f"  MRR@5:   {reciprocal_rank_at_k(query_rels, 5):.4f}")
    print(f"  MRR@10:  {reciprocal_rank_at_k(query_rels, 10):.4f}")
    
    # Ideal ranking
    ideal = np.sort(query_rels)[::-1]
    print(f"\nIdeal ranking:  {ideal}")
    print(f"  NDCG@10: {ndcg_at_k(ideal, 10):.4f}")
```

---

<a name="exercises"></a>
## 10. Exercises

### Exercise 1: Manual NDCG Calculation (Beginner)
```python
# Given this ranking with relevance grades [3, 0, 2, 1, 0]:
# 1. Calculate DCG@5 manually
# 2. Calculate IDCG@5
# 3. Calculate NDCG@5
# 4. Verify your answer with the implementation
```

### Exercise 2: Build an Evaluation Framework (Intermediate)
```python
# Create a comprehensive evaluation framework:
# 1. Generate synthetic recommendation data for 500 users
# 2. Implement three "models": random, popular, personalized
# 3. Evaluate all models using NDCG@K, MAP@K, MRR@K, HR@K
# 4. Create a comparison table
# 5. Plot metrics vs. K for each model
```

### Exercise 3: Diversity-Accuracy Tradeoff (Intermediate)
```python
# Explore the tradeoff between accuracy and diversity:
# 1. Create a model that only recommends popular items
# 2. Create a model that randomly recommends items
# 3. Create a hybrid with varying diversity weight
# 4. Plot NDCG@10 vs. Diversity for different weights
# 5. Find the Pareto-optimal point
```

### Exercise 4: Position Bias Simulation (Advanced)
```python
# Simulate position bias and its effect on offline metrics:
# 1. Generate "true" relevance for 1000 queries
# 2. Simulate clicks with position bias: P(click) = rel × pos_bias
# 3. Build a model on biased click data
# 4. Evaluate on true relevance (unbiased)
# 5. Implement IPS (Inverse Propensity Scoring) correction
# 6. Compare: no correction vs. IPS correction
```

### Exercise 5: MAP vs. NDCG Comparison (Advanced)
```python
# Create scenarios where MAP and NDCG disagree:
# 1. Find a ranking where MAP > NDCG for another ranking's items
# 2. Find a ranking where NDCG > MAP
# 3. Explain why they differ
# 4. Create 10 such examples and analyze patterns
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    METRIC QUICK REFERENCE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  NDCG@K = DCG@K / IDCG@K                                       │
│    DCG = Σ (2^rel - 1) / log₂(pos + 1)                         │
│    Best for: Graded relevance, position-sensitive ranking       │
│                                                                  │
│  MAP@K = Mean of Average Precision                               │
│    AP = (1/R) Σ P(k) × rel(k)                                   │
│    Best for: Binary relevance, need ALL relevant items          │
│                                                                  │
│  MRR@K = Mean of 1/rank_first_relevant                          │
│    Best for: Single correct answer scenarios                    │
│                                                                  │
│  Recall@K = relevant_in_top_K / total_relevant                  │
│    Best for: Coverage-focused evaluation                        │
│                                                                  │
│  CTR = clicks / impressions                                     │
│    Online metric; needs position debiasing                      │
│                                                                  │
│  Always report multiple metrics + diversity + coverage          │
│  Validate offline results with online A/B tests                 │
└─────────────────────────────────────────────────────────────────┘
```
