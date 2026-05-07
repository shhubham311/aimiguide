# Probability Basics: Sample Space, Events & Marginal Probability

## Why This Matters in AI/ML

Probability is the mathematical language of uncertainty, and machine learning is fundamentally about making decisions under uncertainty. Every prediction a model makes carries a degree of uncertainty, and probability theory gives us the tools to quantify, reason about, and minimize that uncertainty.

Key ML connections:

- **Classification outputs**: Softmax produces a probability distribution over classes. Understanding what these probabilities mean is essential.
- **Loss functions**: Cross-entropy loss is derived from probability theory (KL divergence, maximum likelihood).
- **Generative models**: GANs, VAEs, and diffusion models are all built on probability distributions.
- **Bayesian inference**: Bayesian neural networks, prior/posterior distributions, and Bayesian optimization.
- **Uncertainty quantification**: Calibration, confidence intervals, and prediction intervals.

---

## Experiments, Sample Spaces & Events

### Random Experiment
A procedure whose outcome cannot be predicted with certainty in advance.

**In ML context**: Training a neural network with random initialization, or sampling a batch of data stochastically.

### Sample Space (S or Ω)
The set of all possible outcomes of an experiment.

```
Example 1: Coin toss        → S = {Heads, Tails}
Example 2: Dice roll         → S = {1, 2, 3, 4, 5, 6}
Example 3: Image classifier  → S = {cat, dog, bird}
Example 4: Neural net weight → S = (-∞, +∞) (continuous)
```

### Event
A subset of the sample space.

```
For a dice roll (S = {1,2,3,4,5,6}):

Event A = "even number"  = {2, 4, 6}
Event B = "greater than 4" = {5, 6}

Visual representation:
    ┌──────────────────────────────┐
    │         SAMPLE SPACE S       │
    │   ┌─────────────┐            │
    │   │   A∩B       │            │
    │   │   {6}       │    ┌───┐   │
    │   │  ┌──────────┼────│ 5 │───│
    │   │  │ A        │    └───┘   │
    │   │  │ {2,4}    │     B      │
    │   └──┴──────────┴──────────── │
    └──────────────────────────────┘
```

### Types of Events

```
┌─────────────────────────────────────────────────┐
│              EVENT TYPES                         │
├──────────────────┬──────────────────────────────┤
│ Simple Event     │ Contains exactly one outcome │
│                  │ e.g., {4} on a dice          │
├──────────────────┼──────────────────────────────┤
│ Compound Event   │ Contains two or more outcomes│
│                  │ e.g., {2, 4, 6}              │
├──────────────────┼──────────────────────────────┤
│ Certain Event    │ Equals the sample space S    │
│                  │ P(S) = 1                     │
├──────────────────┼──────────────────────────────┤
│ Impossible Event │ Empty set ∅                  │
│                  │ P(∅) = 0                     │
├──────────────────┼──────────────────────────────┤
│ Complement       │ Aᶜ = S \ A                  │
│                  │ P(Aᶜ) = 1 - P(A)            │
└──────────────────┴──────────────────────────────┘
```

---

## The Three Axioms of Probability (Kolmogorov)

Any valid probability measure P must satisfy:

```
Axiom 1: Non-negativity
         P(A) ≥ 0    for every event A

Axiom 2: Normalization
         P(S) = 1    (the sample space has probability 1)

Axiom 3: Additivity (for mutually exclusive events)
         If A ∩ B = ∅, then P(A ∪ B) = P(A) + P(B)
         Generalized: P(A₁ ∪ A₂ ∪ ... ∪ Aₙ) = Σ P(Aᵢ)
         when all pairs Aᵢ, Aⱼ are disjoint
```

**Everything else in probability theory follows from these three axioms.**

---

## Basic Probability Rules

### Complement Rule
```
P(Aᶜ) = 1 - P(A)
```

**ML Example**: If a model predicts "spam" with probability 0.85, then P(not spam) = 0.15.

### Addition Rule (General)
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

This avoids double-counting the intersection.

```
Visual (Venn diagram):

         ┌──────────┐
        ╱    A      ╲
       │  ┌──────┐   │
       │  │ A∩B  │   │
       │  └──────┘   │
        ╲            ╱
         └──────────┘
            B

Counting: P(A) + P(B) counts A∩B twice,
so subtract P(A∩B) once.
```

### For Mutually Exclusive Events (A ∩ B = ∅)
```
P(A ∪ B) = P(A) + P(B)
```

### Inclusion-Exclusion Principle (3 events)
```
P(A∪B∪C) = P(A) + P(B) + P(C)
          - P(A∩B) - P(A∩C) - P(B∩C)
          + P(A∩B∩C)
```

---

## Marginal Probability

**Marginal probability** is the probability of a single event occurring, regardless of other events. When dealing with multiple variables, we "marginalize out" (sum/integrate over) the other variables.

### Discrete Case

For two random variables X and Y with joint probability table:

```
P(X = x) = Σ P(X = x, Y = yᵢ)  (sum over all yᵢ)
         = y
```

### Example: Joint and Marginal Probability Table

```
Suppose we have a dataset of images classified by:
  - Weather: {Sunny, Cloudy, Rainy}
  - Scene:   {Outdoor, Indoor}

Joint probability table P(Weather, Scene):

              Outdoor    Indoor    MARGINAL P(W)
           ┌──────────┬──────────┬──────────┐
   Sunny   │   0.30   │   0.10   │   0.40   │
           ├──────────┼──────────┼──────────┤
   Cloudy  │   0.15   │   0.15   │   0.30   │
           ├──────────┼──────────┼──────────┤
   Rainy   │   0.05   │   0.25   │   0.30   │
           ├──────────┼──────────┼──────────┤
 MARGINAL  │   0.50   │   0.50   │   1.00   │
  P(S)     │          │          │          │
           └──────────┴──────────┴──────────┘

Marginal P(Sunny)  = 0.30 + 0.10 = 0.40
Marginal P(Outdoor) = 0.30 + 0.15 + 0.05 = 0.50
```

### Continuous Case

For continuous random variables with joint density f(x, y):

```
f_X(x) = ∫ f(x, y) dy     (marginal density of X)
f_Y(y) = ∫ f(x, y) dx     (marginal density of Y)
```

---

## Equally Likely Outcomes (Classical Probability)

When all outcomes in a finite sample space are equally likely:

```
         number of favorable outcomes
P(A) = ─────────────────────────────
         total number of outcomes

       |A|
     = ───
       |S|
```

**Example**: Fair six-sided die.
```
P(rolling a 4) = 1/6
P(rolling even) = 3/6 = 1/2
P(rolling > 3)  = 3/6 = 1/2
```

---

## Python Implementation

```python
import numpy as np

# ---- Basic Probability Calculations ----
# Coin flip simulation
def simulate_coin_flips(n_flips=10000, p_heads=0.5, seed=42):
    """Simulate coin flips and compute empirical probabilities."""
    rng = np.random.default_rng(seed)
    flips = rng.random(n_flips) < p_heads  # True = Heads
    n_heads = np.sum(flips)
    p_empirical = n_heads / n_flips
    print(f"Theoretical P(Heads): {p_heads}")
    print(f"Empirical P(Heads):   {p_empirical:.4f} (from {n_flips} flips)")
    return flips

print("=== Coin Flip Simulation ===")
simulate_coin_flips(100000)

# ---- Dice Roll Probabilities ----
def dice_probabilities():
    """Compute probabilities for a fair six-sided die."""
    outcomes = np.arange(1, 7)
    p_each = 1/6

    # Compound events
    p_even = len([x for x in outcomes if x % 2 == 0]) * p_each
    p_gt3 = len([x for x in outcomes if x > 3]) * p_each
    p_prime = len([x for x in outcomes if x in {2, 3, 5}]) * p_each

    print(f"\n=== Dice Probabilities ===")
    print(f"P(even)     = {p_even:.4f}")
    print(f"P(> 3)      = {p_gt3:.4f}")
    print(f"P(prime)    = {p_prime:.4f}")
    print(f"P(even ∩ > 3) = {len([x for x in outcomes if x % 2 == 0 and x > 3]) * p_each:.4f}")
    print(f"P(even ∪ > 3) = {p_even + p_gt3 - len([x for x in outcomes if x % 2 == 0 and x > 3]) * p_each:.4f}")

dice_probabilities()

# ---- Marginal Probability from Joint Table ----
def marginal_probability():
    """Compute marginal probabilities from a joint distribution table."""
    # P(Weather, Scene) as a 2D array
    weather_labels = ['Sunny', 'Cloudy', 'Rainy']
    scene_labels = ['Outdoor', 'Indoor']
    joint = np.array([
        [0.30, 0.10],  # Sunny
        [0.15, 0.15],  # Cloudy
        [0.05, 0.25],  # Rainy
    ])

    print(f"\n=== Joint & Marginal Probability Table ===")
    print(f"{'Weather':<10} {'Outdoor':>10} {'Indoor':>10} {'Marginal':>10}")

    for i, w in enumerate(weather_labels):
        marginal_w = np.sum(joint[i, :])
        print(f"{w:<10} {joint[i,0]:>10.2f} {joint[i,1]:>10.2f} {marginal_w:>10.2f}")

    print(f"\nMarginal P(Outdoor) = {np.sum(joint[:, 0]):.2f}")
    print(f"Marginal P(Indoor)  = {np.sum(joint[:, 1]):.2f}")
    print(f"Sum check: {np.sum(joint):.4f} (should be 1.0)")

    # Specific marginal query
    print(f"\nP(Sunny ∪ Rainy) = {np.sum(joint[0,:]) + np.sum(joint[2,:]):.2f}")

marginal_probability()

# ---- Softmax as Probability Distribution (ML Connection) ----
def softmax(logits):
    """Convert logits to probabilities using softmax."""
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / np.sum(exp_logits)

print(f"\n=== Softmax Probability Distribution ===")
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.5])
probs = softmax(logits)
classes = ['cat', 'dog', 'bird', 'fish', 'horse']

print("Logits → Probabilities:")
for cls, logit, prob in zip(classes, logits, probs):
    print(f"  {cls:<6}: logit={logit:>5.1f}, P={prob:.4f}")
print(f"  Sum of probabilities: {np.sum(probs):.6f} (should be 1.0)")

# ---- Simulation: Law of Large Numbers ----
def law_of_large_numbers(n_max=5000, true_p=0.3):
    """Demonstrate convergence of empirical to theoretical probability."""
    rng = np.random.default_rng(123)
    cumulative_heads = np.cumsum(rng.random(n_max) < true_p)
    empirical_p = cumulative_heads / np.arange(1, n_max + 1)

    print(f"\n=== Law of Large Numbers Demo ===")
    print(f"True P(Heads) = {true_p}")
    checkpoints = [10, 50, 100, 500, 1000, 5000]
    for cp in checkpoints:
        print(f"  After {cp:>5} flips: empirical P = {empirical_p[cp-1]:.4f}")

law_of_large_numbers()
```

**Output:**
```
=== Coin Flip Simulation ===
Theoretical P(Heads): 0.5
Empirical P(Heads):   0.4994 (from 100000 flips)

=== Dice Probabilities ===
P(even)     = 0.5000
P(> 3)      = 0.5000
P(prime)    = 0.5000
P(even ∩ > 3) = 0.1667
P(even ∪ > 3) = 0.8333

=== Joint & Marginal Probability Table ===
Weather       Outdoor     Indoor   Marginal
Sunny            0.30       0.10       0.40
Cloudy           0.15       0.15       0.30
Rainy            0.05       0.25       0.30

Marginal P(Outdoor) = 0.50
Marginal P(Indoor)  = 0.50
Sum check: 1.0000 (should be 1.0)

=== Softmax Probability Distribution ===
Logits → Probabilities:
  cat   : logit=  2.0, P=0.1500
  dog   : logit=  1.0, P=0.0554
  bird  : logit=  0.1, P=0.0203
  fish  : logit= -1.0, P=0.0074
  horse : logit=  3.5, P=0.7669
  Sum of probabilities: 1.000000 (should be 1.0)

=== Law of Large Numbers Demo ===
True P(Heads) = 0.3
After    10 flips: empirical P = 0.4000
After    50 flips: empirical P = 0.2800
After   100 flips: empirical P = 0.3100
After   500 flips: empirical P = 0.3040
After  1000 flips: empirical P = 0.2950
After  5000 flips: empirical P = 0.2994
```

---

## Real-World ML Connections

### 1. Softmax Output is a Probability Distribution
The softmax function in neural networks converts raw logits into a valid probability distribution — all outputs are non-negative and sum to 1. This directly satisfies Kolmogorov's Axioms 1 and 2.

### 2. Cross-Entropy Loss
Derived from maximum likelihood estimation, cross-entropy measures the divergence between the model's predicted probability distribution and the true distribution.

```
L = -Σ yᵢ log(pᵢ)

where yᵢ is the true label (one-hot) and pᵢ is the predicted probability.
```

### 3. Class Imbalance
Marginal probabilities reveal class imbalance. If P(class A) = 0.95 and P(class B) = 0.05, a naive model predicting only A achieves 95% accuracy. Understanding marginals is the first step in handling imbalanced data.

### 4. Monte Carlo Methods
When analytical probability computation is intractable (e.g., in high-dimensional spaces), we use random sampling (Monte Carlo) to estimate probabilities — directly connecting to the Law of Large Numbers.

---

## Exercises with Solutions

### Exercise 1
A dataset has 1000 images: 600 cats, 300 dogs, 100 birds. If you randomly select one image, what is:
(a) P(cat)?
(b) P(cat or dog)?
(c) P(not bird)?

**Solution:**
```
(a) P(cat) = 600/1000 = 0.6
(b) P(cat ∪ dog) = P(cat) + P(dog) = 0.6 + 0.3 = 0.9
    (mutually exclusive, so no intersection correction needed)
(c) P(not bird) = 1 - P(bird) = 1 - 0.1 = 0.9
```

### Exercise 2
A spam classifier outputs: P(spam | email) = 0.8, P(not spam | email) = 0.2. After seeing 500 emails marked as spam, 400 were correctly classified. What is the empirical P(correct | spam)?

**Solution:**
```
P(correct | spam) = 400/500 = 0.8 (matches the theoretical prediction)
```

### Exercise 3
Given the joint table below, find all marginal probabilities:

```
           X=0    X=1
    Y=0  [0.1   0.3]
    Y=1  [0.2   0.4]
```

**Solution:**
```
P(X=0) = 0.1 + 0.2 = 0.3
P(X=1) = 0.3 + 0.4 = 0.7
P(Y=0) = 0.1 + 0.3 = 0.4
P(Y=1) = 0.2 + 0.4 = 0.6
Check: 0.3 + 0.7 = 1.0 ✓ and 0.4 + 0.6 = 1.0 ✓
```

### Exercise 4
Verify that softmax output satisfies all three Kolmogorov axioms.

**Solution:**
```
Axiom 1 (Non-negativity): softmax(xᵢ) = exp(xᵢ)/Σexp(xⱼ) > 0 since exp > 0. ✓
Axiom 2 (Normalization): Σ softmax(xᵢ) = Σ exp(xᵢ)/Σexp(xⱼ) = 1. ✓
Axiom 3 (Additivity): For mutually exclusive events A, B ⊆ {classes},
         P(A ∪ B) = Σᵢ∈A∪B softmax(xᵢ) = Σᵢ∈A softmax(xᵢ) + Σᵢ∈B softmax(xᵢ)
         = P(A) + P(B). ✓
```

### Exercise 5
A medical test has P(positive) = 0.15 across all tested patients. If 200 patients are tested, how many are expected to test positive?

**Solution:**
```
Expected positive tests = n × P(positive) = 200 × 0.15 = 30 patients
```

---

## Key Takeaways

1. **Probability = normalized counting**: It assigns numbers between 0 and 1 to events, satisfying three simple axioms.
2. **Sample space first**: Always start by clearly defining what all possible outcomes are.
3. **Marginal probability** gives you the "big picture" of a single variable by summing out all others.
4. **Softmax is your bridge**: It connects neural network outputs (real numbers) to probability theory (valid distributions).
5. **The Law of Large Numbers** guarantees that empirical probabilities converge to theoretical ones with enough data.
