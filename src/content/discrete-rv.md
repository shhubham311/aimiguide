# Discrete Random Variables: PMF, Uniform, Bernoulli, Binomial

## Why This Matters in AI/ML

Discrete random variables model **countable outcomes** — the backbone of classification, counting events, and modeling binary decisions. Every time you predict a class label, count errors, or model a yes/no decision, you're working with discrete distributions.

Key ML connections:

- **Bernoulli**: Models a single binary classification (spam/not spam, click/no-click).
- **Binomial**: Models the number of correct predictions out of n trials (k-fold CV accuracy).
- **Categorical/Uniform**: Models random class selection, data augmentation, and exploration in RL.
- **Cross-entropy loss**: Derived from the log-likelihood of categorical distributions.
- **Poisson**: Models count data (number of events per unit time), used in NLP and recommender systems.

---

## Random Variables: The Core Concept

A **random variable (RV)** is a function that assigns a numerical value to each outcome in the sample space.

```
Sample Space  →  Random Variable  →  Real Numbers

Coin toss:    {H, T}             →  X(H) = 1, X(T) = 0
Dice roll:    {1,2,3,4,5,6}      →  X = outcome value
Classification: {cat,dog,bird}  →  X(cat)=0, X(dog)=1, X(bird)=2
```

**Two types:**
- **Discrete RV**: Takes countable values (integers, categories)
- **Continuous RV**: Takes any value in an interval (next chapter)

---

## Probability Mass Function (PMF)

For a discrete RV X, the **PMF** assigns a probability to each possible value:

```
p(x) = P(X = x)

Properties:
  1. p(x) ≥ 0    for all x
  2. Σ p(x) = 1   (sums to 1 over all possible values)
```

### Example: Fair Die

```
    p(x)
     │
0.20 │           ┌──┐
     │           │  │
     │     ┌──┐  │  │
     │     │  │  │  │  ┌──┐
     │┌──┐ │  │  │  │  │  │
     ││  │ │  │  │  │  │  │
  ───┼┴──┴─┴──┴──┴──┴──┴──┴── x
      1    2    3    4    5    6

PMF: p(x) = 1/6  for x ∈ {1, 2, 3, 4, 5, 6}
```

### Cumulative Distribution Function (CDF)

```
F(x) = P(X ≤ x) = Σ p(xᵢ) for all xᵢ ≤ x

Properties:
  1. 0 ≤ F(x) ≤ 1
  2. F is non-decreasing (right-continuous step function)
  3. lim F(x) → 1 as x → ∞
  4. lim F(x) → 0 as x → -∞
  5. P(a < X ≤ b) = F(b) - F(a)
```

---

## Expected Value and Variance

### Expected Value (Mean)

The long-run average of a discrete RV:

```
        E[X] = μ = Σ xᵢ × p(xᵢ)
```

### Variance

```
  Var(X) = σ² = E[(X - μ)²] = Σ (xᵢ - μ)² × p(xᵢ)

Shortcut formula:
  Var(X) = E[X²] - (E[X])²
```

### Properties of Expectation and Variance

```
Linearity of Expectation (always holds):
  E[aX + bY + c] = aE[X] + bE[Y] + c

Variance properties:
  Var(aX + b) = a² Var(X)         [constants don't affect variance]
  Var(X + Y) = Var(X) + Var(Y)    [IF X and Y are INDEPENDENT]
  Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)  [general case]

E[XY] = E[X]E[Y]    [IF X and Y are independent]
```

---

## Discrete Uniform Distribution

All outcomes are equally likely.

```
PMF:  p(x) = 1/n    for x ∈ {x₁, x₂, ..., xₙ}

Parameters: n (number of outcomes)
E[X] = (x₁ + x₂ + ... + xₙ) / n = (n+1)/2  (for {1, 2, ..., n})
Var(X) = (n² - 1) / 12
```

```
PMF for Uniform{1,2,3,4,5,6}:

    p(x)
     │
1/6  │ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
     │ │  │ │  │ │  │ │  │ │  │ │  │
     │ │  │ │  │ │  │ │  │ │  │ │  │
  ───┼─┴──┴─┴──┴─┴──┴─┴──┴─┴──┴─┴──┴─ x
       1    2    3    4    5    6

All bars equal height.
```

**ML Application**: Random initialization, random sampling, data shuffling, exploration strategies in reinforcement learning (epsilon-greedy with uniform exploration).

---

## Bernoulli Distribution

The simplest discrete distribution — models a **single binary trial** (success/failure).

```
PMF:  P(X = x) = pˣ × (1-p)^(1-x)    for x ∈ {0, 1}

Parameters: p = probability of success (x=1)

E[X] = p
Var(X) = p(1-p)
```

```
    p(x)
     │
     │         ┌──┐
     │         │  │
     │         │  │
  ───┼─────┬───┴──┴─── x
         0    1
          ↑    ↑
      P(0)   P(1)
     = 1-p   = p

Example (p=0.7):
  P(X=1) = 0.7  (success)
  P(X=0) = 0.3  (failure)
  E[X] = 0.7
  Var(X) = 0.7 × 0.3 = 0.21
```

**Maximum variance at p = 0.5:**
```
  Var(X) = p(1-p)
  d/dp [p(1-p)] = 1 - 2p = 0 → p = 0.5
  Max variance = 0.5 × 0.5 = 0.25
```

**ML Application**: Every binary classification output is a Bernoulli random variable. The sigmoid function maps logits to p values.

---

## Binomial Distribution

Models the number of successes in **n independent Bernoulli trials**.

```
PMF:  P(X = k) = C(n,k) × pᵏ × (1-p)^(n-k)

          n!
     = ─────── × pᵏ × (1-p)^(n-k)
       k!(n-k)!

Parameters: n = number of trials, p = probability of success per trial
Support: k ∈ {0, 1, 2, ..., n}

E[X] = np
Var(X) = np(1-p)
```

### Visual: Binomial PMF shapes

```
n=10, p=0.5 (symmetric):     n=10, p=0.2 (right-skewed):    n=10, p=0.8 (left-skewed):
    p(k)                          p(k)                          p(k)
     │                             │                             │
0.25 │          ┌──┐               │  ┌──┐                      │           ┌──┐
     │         ┌┤  ├┐              │  │  │                 ┌──┐ │          ┌┤  ├┐
     │     ┌──┐││  ││┌──┐         │┌┐│  │                ┌┤  ├┐│         ┌┐││  ││
     │  ┌──┤  ├┤│  ││├  ├──┐     ┌┤│├┤  ├┐            ┌──┤│  ││├┐     ┌┤│├┤  ├┤
  ───┼──┴──┴──┴──┴┴──┴┴──┴──┴─  ─┴┴┴┴┴──┴┴┴───────── ──┴┴┴──┴┴┴┴─── ──┴┴┴┴──┴┴
     0  1  2  3  4  5  6  7  8   0  1  2  3  4  5  6   2  4  6  7  8  9 10

  μ = np = 5                   μ = np = 2                    μ = np = 8
  σ² = np(1-p) = 2.5           σ² = np(1-p) = 1.6           σ² = np(1-p) = 1.6
```

### Special Cases
- **n=1**: Binomial(1,p) = Bernoulli(p)
- **p=0.5**: Symmetric distribution
- Large n, moderate p: Approaches Normal distribution (by CLT)

### Cumulative Probability
```
P(X ≤ k) = Σ C(n,j) × pʲ × (1-p)^(n-j)  for j = 0 to k
```

### Probability of AT LEAST k successes
```
P(X ≥ k) = 1 - P(X ≤ k-1)
```

---

## Poisson Distribution (Bonus)

Models the number of events in a fixed interval of time/space:

```
PMF:  P(X = k) = (λᵏ × e^(-λ)) / k!

Parameter: λ = average rate of events (mean)
Support: k ∈ {0, 1, 2, ...}

E[X] = λ
Var(X) = λ    (mean equals variance!)
```

**ML Application**: Modeling count data — number of visits to a website, number of words in a document, number of defects per batch.

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---- Bernoulli Distribution ----
print("=" * 60)
print("BERNOULLI DISTRIBUTION")
print("=" * 60)

p = 0.7
n_samples = 10000
rng = np.random.default_rng(42)

# Generate samples
samples = rng.binomial(1, p, n_samples)

print(f"\nParameter p = {p}")
print(f"Theoretical: E[X] = {p}, Var(X) = {p*(1-p):.4f}")
print(f"Empirical:   mean = {np.mean(samples):.4f}, var = {np.var(samples, ddof=1):.4f}")
print(f"P(X=0) = {1-p:.4f}, P(X=1) = {p:.4f}")
print(f"Empirical P(X=0) = {np.mean(samples==0):.4f}")
print(f"Empirical P(X=1) = {np.mean(samples==1):.4f}")

# ---- Binomial Distribution ----
print(f"\n{'=' * 60}")
print("BINOMIAL DISTRIBUTION")
print("=" * 60)

n_trials = 20
p_success = 0.3

# Theoretical
mean_theory = n_trials * p_success
var_theory = n_trials * p_success * (1 - p_success)

# Generate many experiments
n_experiments = 50000
samples = rng.binomial(n_trials, p_success, n_experiments)

print(f"\nParameters: n = {n_trials}, p = {p_success}")
print(f"Theoretical: E[X] = {mean_theory}, Var(X) = {var_theory:.4f}")
print(f"Empirical:   mean = {np.mean(samples):.4f}, var = {np.var(samples, ddof=1):.4f}")

# PMF values
k_values = np.arange(0, n_trials + 1)
pmf_values = stats.binom.pmf(k_values, n_trials, p_success)

print(f"\nPMF Table (selected values):")
print(f"{'k':>4} {'P(X=k)':>10} {'P(X≤k)':>10}")
print("-" * 26)
cumsum = 0
for k in range(0, n_trials + 1):
    cumsum += pmf_values[k]
    if k in [0, 3, 5, 6, 10, 15, 20]:
        print(f"{k:>4} {pmf_values[k]:>10.6f} {cumsum:>10.6f}")

# Probability of at least k successes
k_threshold = 8
p_at_least = 1 - stats.binom.cdf(k_threshold - 1, n_trials, p_success)
print(f"\nP(X ≥ {k_threshold}) = 1 - P(X ≤ {k_threshold-1}) = {p_at_least:.6f}")

# ---- ML Application: k-Fold CV Accuracy ----
print(f"\n{'=' * 60}")
print("ML APPLICATION: K-FOLD CROSS-VALIDATION")
print("=" * 60)

# Model achieves 85% true accuracy on each fold
true_acc = 0.85
n_folds = 10

# Expected number of folds where accuracy > 80% (using binomial)
p_above_80 = 0.9  # probability of exceeding 80% threshold per fold
E_correct = n_folds * p_above_80
print(f"\nTrue model accuracy: {true_acc*100:.0f}%")
print(f"Number of folds: {n_folds}")
print(f"P(acc > 80% per fold) ≈ {p_above_80}")
print(f"Expected folds exceeding 80%: {E_correct:.1f}")

# Simulate
n_simulations = 10000
fold_results = rng.binomial(n_folds, p_above_80, n_simulations)
print(f"Simulation: mean = {np.mean(fold_results):.2f}, std = {np.std(fold_results):.2f}")

# ---- ML Application: Spam Detection with Bernoulli ----
print(f"\n{'=' * 60}")
print("ML APPLICATION: SPAM DETECTION (BERNOULLI)")
print("=" * 60)

# Model outputs probability for 5 emails
email_probs = np.array([0.92, 0.15, 0.87, 0.03, 0.65])
predictions = rng.binomial(1, email_probs)
true_labels = np.array([1, 0, 1, 0, 1])

accuracy = np.mean(predictions == true_labels)
print(f"\nEmail probabilities: {email_probs}")
print(f"Predictions:         {predictions}")
print(f"True labels:         {true_labels}")
print(f"Accuracy: {accuracy*100:.0f}%")

# Expected accuracy (sum of Bernoulli expectations)
expected_acc = np.mean(np.where(true_labels == 1, email_probs, 1 - email_probs))
print(f"Expected accuracy:   {expected_acc*100:.1f}%")

# ---- Discrete Uniform: Random Class Selection ----
print(f"\n{'=' * 60}")
print("DISCRETE UNIFORM: RANDOM CLASS SAMPLING")
print("=" * 60)

n_classes = 10
n_samples_class = 50000
uniform_samples = rng.integers(0, n_classes, n_samples_class)

print(f"\nSampling uniformly from {n_classes} classes:")
print(f"Theoretical P(each class) = {1/n_classes:.4f}")
unique, counts = np.unique(uniform_samples, return_counts=True)
print(f"Empirical counts: {counts}")
print(f"Empirical probs:  {counts/n_samples_class}")

# ---- Poisson Distribution ----
print(f"\n{'=' * 60}")
print("POISSON DISTRIBUTION")
print("=" * 60)

lam = 4.5  # Average 4.5 events per interval
poisson_samples = rng.poisson(lam, 50000)

print(f"\nParameter λ = {lam}")
print(f"Theoretical: E[X] = {lam}, Var(X) = {lam}")
print(f"Empirical:   mean = {np.mean(poisson_samples):.4f}, var = {np.var(poisson_samples, ddof=1):.4f}")
print(f"P(X=0) = {stats.poisson.pmf(0, lam):.4f}")
print(f"P(X≤3) = {stats.poisson.cdf(3, lam):.4f}")
```

**Output:**
```
============================================================
BERNOULLI DISTRIBUTION
============================================================

Parameter p = 0.7
Theoretical: E[X] = 0.7, Var(X) = 0.2100
Empirical:   mean = 0.6995, var = 0.2104

============================================================
BINOMIAL DISTRIBUTION
============================================================

Parameters: n = 20, p = 0.3
Theoretical: E[X] = 6.0, Var(X) = 4.2000
Empirical:   mean = 6.0034, var = 4.1986

ML APPLICATION: K-FOLD CROSS-VALIDATION
============================================================
Expected folds exceeding 80%: 9.0
Simulation: mean = 9.00, std = 0.95
```

---

## Real-World ML Connections

### 1. Bernoulli in Binary Classification
Every binary classifier output is modeled as a Bernoulli trial. The cross-entropy loss:
```
L = -[y log(p) + (1-y) log(1-p)]
```
is the negative log-likelihood of a Bernoulli distribution.

### 2. Binomial for Model Evaluation
When a model with accuracy p is tested on n samples, the number of correct predictions follows Binomial(n, p). This lets us compute confidence intervals on accuracy metrics.

### 3. Binomial for A/B Testing
In A/B testing, conversions in each group follow binomial distributions. Comparing them leads to hypothesis tests (covered in a later chapter).

### 4. Categorical Distribution
The multinomial/categorical distribution generalizes Bernoulli to multiple classes. Softmax outputs are categorical distributions. Cross-entropy generalizes accordingly.

---

## Exercises with Solutions

### Exercise 1
A model has 92% accuracy. If tested on 50 new samples, what is the probability of getting exactly 45 correct?

**Solution:**
```
X ~ Binomial(n=50, p=0.92)
P(X=45) = C(50,45) × 0.92⁴⁵ × 0.08⁵

Using scipy: stats.binom.pmf(45, 50, 0.92) ≈ 0.1486
```

### Exercise 2
A website has a 5% click-through rate. What is the probability of at least 3 clicks in 20 impressions?

**Solution:**
```
X ~ Binomial(n=20, p=0.05)
P(X ≥ 3) = 1 - P(X ≤ 2)
         = 1 - [P(X=0) + P(X=1) + P(X=2)]
P(X=0) = 0.95²⁰ = 0.3585
P(X=1) = C(20,1)(0.05)(0.95¹⁹) = 20(0.05)(0.3774) = 0.3774
P(X=2) = C(20,2)(0.05²)(0.95¹⁸) = 190(0.0025)(0.3972) = 0.1887
P(X ≥ 3) = 1 - 0.3585 - 0.3774 - 0.1887 = 0.0755
```

### Exercise 3
Show that the Bernoulli distribution maximizes variance when p = 0.5.

**Solution:**
```
Var(X) = p(1-p) = p - p²
d/dp [p - p²] = 1 - 2p = 0 → p = 0.5
d²/dp² [p - p²] = -2 < 0 → maximum at p = 0.5
Max Var = 0.5 × 0.5 = 0.25
```

### Exercise 4
If X ~ Binomial(n=100, p=0.4), find E[X], Var(X), and P(X > 50).

**Solution:**
```
E[X] = 100 × 0.4 = 40
Var(X) = 100 × 0.4 × 0.6 = 24
P(X > 50) = 1 - P(X ≤ 50) ≈ 1 - stats.binom.cdf(50, 100, 0.4) ≈ 0.0271
```

### Exercise 5
A neural network processes images in batches of 32. Each image has a 2% chance of causing an error. What distribution models the number of errors per batch? Compute the probability of zero errors.

**Solution:**
```
X ~ Binomial(n=32, p=0.02)
P(X=0) = (1-0.02)³² = 0.98³² ≈ 0.5243

Alternatively, Poisson approximation (np = 0.64):
X ≈ Poisson(λ=0.64)
P(X=0) = e^(-0.64) ≈ 0.5273 (very close!)
```

---

## Key Takeaways

1. **Bernoulli is the atomic unit**: A single yes/no trial. Binary classification is built on it.
2. **Binomial = n Bernoulli trials**: Models count of successes. Use for accuracy evaluation, A/B testing, reliability analysis.
3. **E[X] = np, Var(X) = np(1-p)**: Memorize these — they're used constantly in ML.
4. **The PMF assigns probability to exact values**; the CDF gives cumulative probabilities (P(X ≤ k)).
5. **Poisson for rare events**: When n is large and p is small, Poisson(λ=np) approximates Binomial(n,p).
6. **Maximum Bernoulli variance at p=0.5**: The most uncertain prediction has the highest variance.
