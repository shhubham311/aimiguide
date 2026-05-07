# Permutations & Combinations

## Why This Matters in AI/ML

Permutations and combinations are the **foundational counting principles** behind every probabilistic model in machine learning. When you build a neural network that classifies images into 10 categories, the number of possible output sequences, the ways to arrange features in an attention mechanism, and the combinatorial explosion of hyperparameter search spaces all rely on these principles.

Specific applications include:

- **Feature selection**: Choosing k features from n available features uses combinations — C(n, k).
- **Cross-validation**: Computing the number of ways to split data into training/validation folds.
- **Ensemble methods**: Counting the number of possible model combinations in bagging/boosting.
- **Attention mechanisms**: The number of possible attention patterns scales combinatorially with sequence length.
- **Neural architecture search (NAS)**: Exploring the space of possible network architectures is inherently combinatorial.
- **Clustering**: The number of possible cluster assignments for n data points into k clusters.

---

## The Fundamental Counting Principle

**Rule**: If Task A can be done in `m` ways and Task B can be done in `n` ways, then both tasks together can be done in `m × n` ways.

```
Task A: 3 options     Task B: 4 options
  [a1] ────────────────── [b1] ──► outcome 1
  [a2] ────────────────── [b2] ──► outcome 2
  [a3] ────────────────── [b3] ──► outcome 3
                          [b4] ──► outcome 4

Total outcomes = 3 × 4 = 12
```

**Generalized**: For k independent tasks with n₁, n₂, ..., nₖ options respectively:

```
Total = n₁ × n₂ × ... × nₖ
```

### Example
A neural network has:
- 3 possible activation functions (ReLU, Sigmoid, Tanh)
- 4 possible learning rates (0.001, 0.01, 0.1, 1.0)
- 2 possible optimizers (SGD, Adam)

Total configurations = 3 × 4 × 2 = **24**

---

## Factorial Notation

The factorial of n, denoted n!, is the product of all positive integers up to n:

```
n! = n × (n-1) × (n-2) × ... × 2 × 1
```

With the convention that `0! = 1`.

| n  | n!        |
|----|-----------|
| 0  | 1         |
| 1  | 1         |
| 5  | 120       |
| 10 | 3,628,800 |
| 20 | 2.4 × 10¹⁸ |

---

## Permutations

A **permutation** is an ordered arrangement of objects. When order matters, we use permutations.

### Permutation: All Items (n items, arranged in order)

```
P(n) = n!
```

**Example**: Arrange 5 books on a shelf:
```
5! = 5 × 4 × 3 × 2 × 1 = 120 arrangements
```

### Permutation: Selecting k from n (order matters)

```
         n!
P(n, k) = --------
         (n - k)!
```

Also written as nPk.

**Visual explanation for P(5, 3):**

```
Slot 1: 5 choices (A, B, C, D, E)
Slot 2: 4 remaining choices
Slot 3: 3 remaining choices

    ┌───┐   ┌───┐   ┌───┐
    │ 5 │ → │ 4 │ → │ 3 │  = 5 × 4 × 3 = 60
    └───┘   └───┘   └───┘

Formula: 5! / (5-3)! = 120 / 2 = 60
```

### Permutation with Repetition

When items can repeat:

```
P(n, k) with repetition = nᵏ
```

**Example**: Number of possible 4-digit PIN codes (digits 0-9):
```
10⁴ = 10,000 possible PINs
```

### Permutation of Multiset (items with duplicates)

When some items are identical:

```
         n!
P = ────────────────────
     n₁! × n₂! × ... × nₖ!
```

Where n₁, n₂, ..., nₖ are the counts of each distinct item.

**Example**: Arrange the letters in "MISSISSIPPI":
```
Total letters: n = 11
M: 1, I: 4, S: 4, P: 2

P = 11! / (1! × 4! × 4! × 2!) = 34,650
```

---

## Combinations

A **combination** is a selection of objects where **order does not matter**. This is the key distinction from permutations.

```
         n!          n!
C(n, k) = -------- = ----------
         k!(n-k)!   P(n,k)/k!
```

Also written as "n choose k" or **(n C k)**.

**Visual: Permutation vs Combination**

```
Select 2 from {A, B, C}:

PERMUTATIONS (order matters):     COMBINATIONS (order doesn't matter):
{A,B}, {B,A}, {A,C}, {C,A},      {A,B}, {A,C}, {B,C}
{B,C}, {C,B}
= 6 outcomes                      = 3 outcomes

P(3,2) = 3! / 1! = 6             C(3,2) = 3! / (2! × 1!) = 3
```

### Key Relationship

```
C(n, k) = P(n, k) / k!

Combinations divide out the k! orderings of each selection.
```

### Pascal's Triangle and Binomial Coefficients

```
                1
              1   1
            1   2   1
          1   3   3   1
        1   4   6   4   1
      1   5  10  10   5   1

Row n gives C(n, 0), C(n, 1), ..., C(n, n)

Key properties:
- C(n, k) = C(n, n-k)       [Symmetry]
- C(n, k) = C(n-1, k-1) + C(n-1, k)  [Pascal's Rule]
- Sum of row n = 2ⁿ
```

---

## Python Implementation

```python
import math
from itertools import permutations, combinations

# ---- Factorial ----
def factorial(n):
    """Compute n! iteratively."""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# ---- Permutations ----
def nPr(n, k):
    """Number of ways to arrange k items from n (order matters)."""
    return math.factorial(n) // math.factorial(n - k)

# ---- Combinations ----
def nCr(n, k):
    """Number of ways to choose k items from n (order doesn't matter)."""
    if k > n:
        return 0
    if k > n - k:
        k = n - k  # Optimize using symmetry
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

# ---- Using itertools for enumeration ----
items = ['cat', 'dog', 'bird']

print("Permutations of 2 from 3:")
for p in permutations(items, 2):
    print(f"  {p}")

print("\nCombinations of 2 from 3:")
for c in combinations(items, 2):
    print(f"  {c}")

# ---- ML Application: Feature Selection Combinatorics ----
print("\n=== ML Feature Selection ===")
n_features = 20
k_select = 5
print(f"Total features: {n_features}, Selecting: {k_select}")
print(f"Number of feature subsets: C({n_features}, {k_select}) = {nCr(n_features, k_select):,}")
print(f"This is why exhaustive search is infeasible for large feature sets!")

# ---- ML Application: Hyperparameter Search Space ----
print("\n=== Hyperparameter Search Space ===")
layer_sizes = [64, 128, 256, 512]        # 4 options
learning_rates = [0.001, 0.01, 0.1]      # 3 options
batch_sizes = [16, 32, 64, 128]          # 4 options
dropout_rates = [0.1, 0.2, 0.3, 0.5]    # 4 options

total_configs = len(layer_sizes) * len(learning_rates) * len(batch_sizes) * len(dropout_rates)
print(f"Total hyperparameter configurations: {total_configs}")
print(f"At 10 min per training run: {total_configs * 10 / 60:.1f} hours for grid search")

# ---- Permutation with repetition: One-hot encoding space ----
n_classes = 10
n_positions = 3
print(f"\nOne-hot sequences for {n_classes} classes, {n_positions} positions: {n_classes ** n_positions:,}")
```

**Output:**
```
Permutations of 2 from 3:
  ('cat', 'dog')
  ('cat', 'bird')
  ('dog', 'cat')
  ('dog', 'bird')
  ('bird', 'cat')
  ('bird', 'dog')

Combinations of 2 from 3:
  ('cat', 'dog')
  ('cat', 'bird')
  ('dog', 'bird')

=== ML Feature Selection ===
Total features: 20, Selecting: 5
Number of feature subsets: C(20, 5) = 15,504
This is why exhaustive search is infeasible for large feature sets!

=== Hyperparameter Search Space ===
Total hyperparameter configurations: 192
At 10 min per training run: 32.0 hours for grid search

One-hot sequences for 10 classes, 3 positions: 1,000
```

---

## Real-World ML Connections

### 1. Feature Selection
When building a model with 50 features and you want to select the best 10, the number of possible subsets is C(50, 10) ≈ 10 billion. This combinatorial explosion is why we use greedy methods (forward/backward selection) or regularization (L1/Lasso) instead of brute-force search.

### 2. Cross-Validation
K-fold cross-validation involves:
- Splitting n data points into k folds: the number of possible splits is C(n-1, k-1).
- Stratified k-fold adds constraints that make counting more complex.

### 3. Ensemble Diversity
In a bagging ensemble with B base models trained on bootstrap samples, the probability that any specific data point is omitted from a bootstrap sample of size n is:

```
P(omitted) = (1 - 1/n)ⁿ → 1/e ≈ 0.368 as n → ∞
```

This ~36.8% forms the **out-of-bag (OOB)** error estimate.

### 4. Neural Architecture Search (NAS)
The search space of possible neural network architectures grows factorially with depth and exponentially with branching. Modern NAS methods use reinforcement learning or evolutionary strategies to navigate this combinatorial space.

---

## Exercises with Solutions

### Exercise 1
A dataset has 15 features. How many ways can you select exactly 3 features for a model?

**Solution:**
```
C(15, 3) = 15! / (3! × 12!) = (15 × 14 × 13) / (3 × 2 × 1) = 455
```

### Exercise 2
How many ways can 8 images be arranged in a sequence for a data augmentation pipeline if only 5 are selected and order matters?

**Solution:**
```
P(8, 5) = 8! / (8-5)! = 8! / 3! = 8 × 7 × 6 × 5 × 4 = 6,720
```

### Exercise 3
A model evaluates sequences of length 4 from an alphabet of 26 tokens (with repetition allowed). How many possible sequences exist?

**Solution:**
```
26⁴ = 456,976 possible sequences
```

### Exercise 4
A confusion matrix for a 5-class classification problem has 25 cells. How many distinct ways can the predicted labels be arranged for a fixed set of true labels, if exactly one prediction per class must be correct?

**Solution:**
This is a derangement-like problem. We need the number of permutations of 5 items where at least one stays in place. The total number of permutations is 5! = 120. The number where none stays in place (derangement) is:
```
D(5) = 5! × (1 - 1/1! + 1/2! - 1/3! + 1/4! - 1/5!) = 44
```
So permutations with at least one fixed point = 120 - 44 = **76**.

### Exercise 5 (Advanced)
Prove that C(n, k) = C(n, n-k) using both algebraic and combinatorial arguments.

**Solution:**

*Algebraic proof:*
```
C(n, k) = n! / (k!(n-k)!)
C(n, n-k) = n! / ((n-k)!(n-(n-k))!) = n! / ((n-k)!k!) = n! / (k!(n-k)!)
Therefore C(n, k) = C(n, n-k). ✓
```

*Combinatorial proof:* Choosing k items to **include** in a subset is equivalent to choosing (n-k) items to **exclude**. Every selection of k items uniquely determines a selection of n-k items to leave out, establishing a bijection between the two sets.

---

## Summary Table

| Concept | Formula | Order Matters? | Repetition? |
|---------|---------|:-:|:-:|
| Fundamental Principle | n₁ × n₂ × ... × nₖ | — | No constraint |
| Permutation (all) | n! | Yes | No |
| Permutation (select k) | n!/(n-k)! | Yes | No |
| Permutation (with rep.) | nᵏ | Yes | Yes |
| Combination | n!/(k!(n-k)!) | No | No |
| Combination (with rep.) | C(n+k-1, k) | No | Yes |

---

## Key Takeaways

1. **Order is the deciding factor**: If {A,B} and {B,A} are different outcomes → permutation. If they're the same → combination.
2. **Always divide by k!** to go from permutations to combinations — you're removing the k! internal orderings.
3. **Combinatorial explosion** is a real challenge in ML: feature selection, architecture search, and hyperparameter tuning all face it.
4. **The fundamental counting principle** is your Swiss army knife — decompose complex problems into sequential independent choices.
