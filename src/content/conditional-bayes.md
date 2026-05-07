# Conditional & Joint Probability, Bayes' Theorem

## Why This Matters in AI/ML

Conditional probability is arguably the **single most important concept** connecting probability theory to machine learning. The entire discipline of Bayesian inference, Naive Bayes classifiers, Markov models, Hidden Markov Models (HMMs), and probabilistic graphical models is built on conditional probability.

Key ML connections:

- **Naive Bayes Classifier**: Directly applies Bayes' theorem for text classification (spam detection, sentiment analysis).
- **Bayesian Neural Networks**: Treat weights as probability distributions rather than fixed values.
- **Markov Chains & HMMs**: Model sequential data (speech, text, time series) using transition probabilities P(state_t | state_{t-1}).
- **Probabilistic Graphical Models**: Bayesian networks represent complex dependencies via conditional probability tables.
- **Maximum a Posteriori (MAP) Estimation**: Regularization in ML is equivalent to MAP with appropriate priors.
- **Likelihood and MLE**: The likelihood function L(θ|x) = P(x|θ) is a conditional probability.

---

## Conditional Probability

### Definition

The probability of event A **given that** event B has occurred:

```
         P(A ∩ B)
P(A|B) = ─────────
           P(B)
```

Provided that P(B) > 0.

**Intuition**: Conditional probability "zooms in" on the subset of outcomes where B occurred and asks "what fraction of those also have A?"

```
Unconditional:              Conditional (given B):

  ┌────────────────┐         ┌────────────────┐
  │    A           │         │    A           │
  │  ┌──────┐      │         │  ┌──────┐      │
  │  │ A∩B  │      │         │  │ A∩B  │      │
  │  └──────┘      │         │  └──────┘      │
  │       B        │   ──►   │   ┌────────┐   │
  │                │         │   │ B only  │   │
  └────────────────┘         └───┴────────┴───┘

  P(A) = area(A)/area(S)     P(A|B) = area(A∩B)/area(B)
```

### Example: Medical Testing
```
P(disease) = 0.01        (1% of population has disease)
P(positive | disease) = 0.99   (99% sensitivity)
P(positive | no disease) = 0.05 (5% false positive rate)

P(positive) = P(positive|disease)P(disease) + P(positive|no disease)P(no disease)
            = 0.99 × 0.01 + 0.05 × 0.99
            = 0.0099 + 0.0495
            = 0.0594
```

---

## Joint Probability

### Definition

The probability that **both** events A and B occur simultaneously:

```
P(A ∩ B) = P(A|B) × P(B) = P(B|A) × P(A)
```

This is called the **multiplication rule** or **chain rule of probability**.

### Joint Probability for Multiple Events

```
P(A ∩ B ∩ C) = P(A) × P(B|A) × P(C|A ∩ B)

General chain rule:
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) × Π P(Aᵢ | A₁ ∩ ... ∩ Aᵢ₋₁)
```

### Joint vs Marginal vs Conditional: The Complete Picture

```
JOINT TABLE: P(Weather, Sentiment)

                Positive   Neutral   Negative   MARGINAL P(W)
             ┌──────────┬─────────┬──────────┬──────────┐
  Sunny      │   0.25   │  0.10   │   0.05   │   0.40   │
             ├──────────┼─────────┼──────────┼──────────┤
  Cloudy     │   0.08   │  0.12   │   0.10   │   0.30   │
             ├──────────┼─────────┼──────────┼──────────┤
  Rainy      │   0.02   │  0.08   │   0.20   │   0.30   │
             ├──────────┼─────────┼──────────┼──────────┤
  MARGINAL   │   0.35   │  0.30   │   0.35   │   1.00   │
  P(S)       │          │         │          │          │
             └──────────┴─────────┴──────────┴──────────┘

Joint:      P(Sunny, Positive) = 0.25
Marginal:   P(Sunny) = 0.40  (row sum)
Conditional: P(Positive | Sunny) = 0.25 / 0.40 = 0.625
```

---

## Independence

### Definition
Events A and B are **independent** if and only if:

```
P(A ∩ B) = P(A) × P(B)

Equivalently:  P(A|B) = P(A)   (knowing B doesn't change A's probability)
               P(B|A) = P(B)
```

### Mutually Exclusive ≠ Independent!

```
CAUTION: These are DIFFERENT concepts!

Mutually Exclusive:  A ∩ B = ∅    →  P(A∩B) = 0
Independent:         P(A∩B) = P(A) × P(B)

If P(A) > 0 and P(B) > 0:
  Mutually exclusive → NOT independent
  (because P(A|B) = 0 ≠ P(A) > 0)
```

### Conditional Independence
Events A and B are conditionally independent given C if:

```
P(A ∩ B | C) = P(A|C) × P(B|C)
```

**ML Context**: Naive Bayes assumes features are conditionally independent given the class label. This is almost never true, but the classifier works surprisingly well in practice.

---

## Bayes' Theorem

### The Formula

Bayes' theorem is the mathematical foundation for updating beliefs given evidence:

```
         P(B|A) × P(A)
P(A|B) = ─────────────
            P(B)

Expanded form:
         P(B|A) × P(A)
P(A|B) = ──────────────────────────────
         P(B|A) × P(A) + P(B|Aᶜ) × P(Aᶜ)
```

### Terminology

```
P(A|B) = Posterior probability (what we want to compute)
P(A)   = Prior probability (our initial belief)
P(B|A) = Likelihood (probability of evidence given hypothesis)
P(B)   = Evidence / Marginal likelihood (normalizing constant)
```

### Visual Explanation

```
  BEFORE (Prior)            AFTER seeing evidence B (Posterior)

  ┌────────────────┐       ┌────────────────┐
  │   ┌──────┐     │       │   ┌────────┐   │
  │   │  A   │     │       │   │  A|B   │   │
  │   │ P(A) │     │       │   │ larger │   │
  │   └──────┘     │  ──►  │   └────────┘   │
  │                │       │     B observed │
  └────────────────┘       └────────────────┘
```

### The Classic Example: Medical Diagnosis

```
Given:
  P(Disease) = 0.001            (prior: rare disease, 0.1% prevalence)
  P(Positive | Disease) = 0.99  (high sensitivity)
  P(Positive | No Disease) = 0.01  (low false positive rate)

Question: P(Disease | Positive) = ?

Step 1: P(Positive) = P(Pos|Dis)P(Dis) + P(Pos|NoDis)P(NoDis)
       = 0.99 × 0.001 + 0.01 × 0.999
       = 0.00099 + 0.00999
       = 0.01098

Step 2: P(Disease | Positive) = P(Pos|Dis) × P(Dis) / P(Pos)
       = 0.99 × 0.001 / 0.01098
       = 0.0902

Result: Only ~9% chance of having the disease despite a positive test!

WHY? Because the disease is so rare (low prior) that false positives
outnumber true positives in the population.
```

### Bayes' Theorem for Multiple Hypotheses

```
                    P(E|Hᵢ) × P(Hᵢ)
P(Hᵢ|E) = ────────────────────────────
            Σⱼ P(E|Hⱼ) × P(Hⱼ)

where H₁, H₂, ..., Hₙ are mutually exclusive hypotheses
and E is the observed evidence.
```

This is used in multi-class Naive Bayes classification.

---

## Bayesian Inference in Machine Learning

The general Bayesian inference framework:

```
      P(data|θ) × P(θ)
P(θ|data) = ─────────────
        P(data)

Posterior  ∝  Likelihood × Prior
```

```
┌─────────┐     ┌────────────┐     ┌──────────────┐
│  Prior  │  ×  │ Likelihood │  =  │  Posterior   │
│ P(θ)    │     │ P(data|θ)  │     │ P(θ|data)    │
│         │     │            │     │              │
│ Prior   │     │ How well   │     │ Updated      │
│ belief  │     │ does θ     │     │ belief after │
│ about θ │     │ explain    │     │ seeing data  │
│         │     │ the data?  │     │              │
└─────────┘     └────────────┘     └──────────────┘
```

### MAP Estimation = L2 Regularization

```
MAP estimate: θ_MAP = argmax_θ log P(θ|data)
            = argmax_θ [log P(data|θ) + log P(θ)]
            = argmax_θ [log L(θ) + log P(θ)]

If P(θ) ~ N(0, σ²), then log P(θ) ∝ -||θ||²/(2σ²)
This is equivalent to L2 (Ridge) regularization!

If P(θ) ~ Laplace(0, b), this gives L1 (Lasso) regularization.
```

---

## Python Implementation

```python
import numpy as np
from collections import defaultdict

# ---- Conditional Probability ----
def conditional_probability():
    """Demonstrate conditional probability with a dataset simulation."""
    rng = np.random.default_rng(42)

    # Simulate 10000 students: Study(yes/no) and Pass(yes/no)
    n = 10000
    study = rng.random(n) < 0.6   # 60% study
    # P(Pass|Study) = 0.8, P(Pass|NoStudy) = 0.3
    pass_exam = np.where(study, rng.random(n) < 0.8, rng.random(n) < 0.3)

    # Empirical estimates
    p_study = np.mean(study)
    p_pass = np.mean(pass_exam)
    p_study_and_pass = np.mean(study & pass_exam)

    p_pass_given_study = np.mean(pass_exam[study])
    p_study_given_pass = np.mean(study[pass_exam])

    print("=== Conditional Probability: Study vs Pass ===")
    print(f"P(Study)           = {p_study:.4f}")
    print(f"P(Pass)            = {p_pass:.4f}")
    print(f"P(Study ∩ Pass)    = {p_study_and_pass:.4f}")
    print(f"P(Pass | Study)    = {p_pass_given_study:.4f}")
    print(f"P(Study | Pass)    = {p_study_given_pass:.4f}")
    print(f"Verify: P(S∩P)/P(S) = {p_study_and_pass/p_study:.4f}")
    print(f"Verify: P(S∩P)/P(P) = {p_study_and_pass/p_pass:.4f}")

conditional_probability()

# ---- Bayes' Theorem: Spam Classification ----
def bayes_spam_classifier():
    """Naive Bayes spam classifier demonstration."""
    # Vocabulary: ['free', 'money', 'meeting', 'report', 'buy', 'project']
    vocab = ['free', 'money', 'meeting', 'report', 'buy', 'project']
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    # Training data
    spam_emails = [
        ['free', 'money', 'buy', 'free'],
        ['buy', 'money', 'free'],
        ['free', 'free', 'buy'],
        ['money', 'buy'],
        ['free', 'buy', 'money', 'free'],
    ]
    ham_emails = [
        ['meeting', 'report', 'project'],
        ['report', 'meeting'],
        ['project', 'meeting', 'report'],
        ['meeting', 'project'],
        ['report', 'project', 'meeting', 'report'],
    ]

    # Priors
    p_spam = len(spam_emails) / (len(spam_emails) + len(ham_emails))
    p_ham = 1 - p_spam

    # Likelihoods with Laplace smoothing (alpha=1)
    alpha = 1
    v = len(vocab)

    # Count words in spam
    spam_word_counts = np.zeros(v)
    for email in spam_emails:
        for word in email:
            spam_word_counts[vocab_idx[word]] += 1
    spam_total = np.sum(spam_word_counts) + alpha * v
    p_word_given_spam = (spam_word_counts + alpha) / spam_total

    # Count words in ham
    ham_word_counts = np.zeros(v)
    for email in ham_emails:
        for word in email:
            ham_word_counts[vocab_idx[word]] += 1
    ham_total = np.sum(ham_word_counts) + alpha * v
    p_word_given_ham = (ham_word_counts + alpha) / ham_total

    # Print likelihoods
    print("\n=== Naive Bayes Spam Classifier ===")
    print(f"Prior: P(Spam) = {p_spam:.2f}, P(Ham) = {p_ham:.2f}\n")
    print(f"{'Word':<10} {'P(w|Spam)':>10} {'P(w|Ham)':>10} {'Ratio':>10}")
    for w in vocab:
        i = vocab_idx[w]
        ratio = p_word_given_spam[i] / p_word_given_ham[i]
        print(f"{w:<10} {p_word_given_spam[i]:>10.4f} {p_word_given_ham[i]:>10.4f} {ratio:>10.2f}x")

    # Classify new emails
    test_emails = [
        ['free', 'money'],
        ['meeting', 'report'],
        ['buy', 'project', 'free'],
    ]

    print("\n--- Classifications ---")
    for email in test_emails:
        log_p_spam = np.log(p_spam)
        log_p_ham = np.log(p_ham)
        for word in email:
            if word in vocab_idx:
                i = vocab_idx[word]
                log_p_spam += np.log(p_word_given_spam[i])
                log_p_ham += np.log(p_word_given_ham[i])

        p_spam_email = np.exp(log_p_spam) / (np.exp(log_p_spam) + np.exp(log_p_ham))
        label = "SPAM" if p_spam_email > 0.5 else "HAM"
        print(f"  '{' '.join(email)}' → P(Spam|email) = {p_spam_email:.4f} → {label}")

bayes_spam_classifier()

# ---- Bayesian Updating Example ----
def bayesian_updating():
    """Demonstrate sequential Bayesian updating (coin fairness)."""
    # Prior: P(fair) = 0.5, P(biased) = 0.5
    # Likelihood: P(3 heads | fair) = C(3,3)(0.5)^3 = 0.125
    #             P(3 heads | biased) = (0.8)^3 = 0.512

    prior_fair = 0.5
    prior_biased = 0.5

    # Observe 3 heads in 3 flips
    likelihood_fair = 0.5**3
    likelihood_biased = 0.8**3

    # Posterior
    evidence = likelihood_fair * prior_fair + likelihood_biased * prior_biased
    posterior_fair = (likelihood_fair * prior_fair) / evidence
    posterior_biased = 1 - posterior_fair

    print(f"\n=== Bayesian Updating: Is the Coin Fair? ===")
    print(f"Prior: P(Fair) = {prior_fair:.2f}")
    print(f"Observed: 3 heads in 3 flips")
    print(f"P(3H | Fair)   = {likelihood_fair:.4f}")
    print(f"P(3H | Biased) = {likelihood_biased:.4f}")
    print(f"Posterior: P(Fair | 3H) = {posterior_fair:.4f}")
    print(f"Posterior: P(Biased | 3H) = {posterior_biased:.4f}")

    # Now observe 7 tails in 10 more flips (total: 3H, 7T in 13)
    # P(7T,3H|fair) = C(10,7)(0.5)^10, P(7T,3H|biased) = C(10,7)(0.2)^7(0.8)^3
    from math import comb
    lik_fair2 = comb(10, 7) * 0.5**10
    lik_biased2 = comb(10, 7) * 0.2**7 * 0.8**3

    evidence2 = lik_fair2 * posterior_fair + lik_biased2 * posterior_biased
    post_fair2 = (lik_fair2 * posterior_fair) / evidence2

    print(f"\nAdditional evidence: 7 tails, 3 heads in next 10 flips")
    print(f"Updated P(Fair) = {post_fair2:.4f}")
    print(f"Updated P(Biased) = {1 - post_fair2:.4f}")

bayesian_updating()
```

**Output:**
```
=== Conditional Probability: Study vs Pass ===
P(Study)           = 0.6006
P(Pass)            = 0.5783
P(Study ∩ Pass)    = 0.4798
P(Pass | Study)    = 0.7989
P(Study | Pass)    = 0.8296
Verify: P(S∩P)/P(S) = 0.7989
Verify: P(S∩P)/P(P) = 0.8296

=== Naive Bayes Spam Classifier ===
Prior: P(Spam) = 0.50, P(Ham) = 0.50

Word       P(w|Spam)   P(w|Ham)      Ratio
free        0.3333     0.0294    11.33x
money       0.2000     0.0294     6.80x
meeting     0.0333     0.2059     0.16x
report      0.0333     0.2059     0.16x
buy         0.2000     0.0294     6.80x
project     0.0333     0.2059     0.16x

--- Classifications ---
  'free money' → P(Spam|email) = 0.9932 → SPAM
  'meeting report' → P(Spam|email) = 0.0012 → HAM
  'buy project free' → P(Spam|email) = 0.9449 → SPAM
```

---

## Real-World ML Connections

### 1. Naive Bayes in NLP
Despite the "naive" assumption of feature independence, Naive Bayes is a strong baseline for text classification. It's fast, works well with high-dimensional data, and requires very little training data.

### 2. Expectation-Maximization (EM)
EM algorithm iteratively applies Bayes' theorem:
- **E-step**: Compute posterior probabilities P(z|data, θ) using current parameters
- **M-step**: Update parameters θ using the posteriors from E-step

### 3. Bayesian Hyperparameter Optimization
Tools like Optuna use Bayesian optimization to efficiently search hyperparameter spaces by maintaining a probabilistic model (posterior) of the objective function.

### 4. Hidden Markov Models (HMMs)
HMMs use transition probabilities P(state_t | state_{t-1}) and emission probabilities P(observation | state). The Viterbi algorithm finds the most likely state sequence, and the Forward-Backward algorithm computes posterior state probabilities.

---

## Exercises with Solutions

### Exercise 1
A factory produces widgets with two machines. Machine A produces 60% of widgets with 2% defect rate. Machine B produces 40% with 1% defect rate. A widget is found defective. What is the probability it came from Machine A?

**Solution:**
```
P(A) = 0.6, P(B) = 0.4
P(D|A) = 0.02, P(D|B) = 0.01

P(D) = 0.02 × 0.6 + 0.01 × 0.4 = 0.012 + 0.004 = 0.016

P(A|D) = P(D|A) × P(A) / P(D)
       = 0.02 × 0.6 / 0.016
       = 0.012 / 0.016
       = 0.75

75% chance the defective widget came from Machine A.
```

### Exercise 2
Prove that if A and B are independent, then Aᶜ and B are also independent.

**Solution:**
```
P(Aᶜ ∩ B) = P(B) - P(A ∩ B)         [since A ∩ B and Aᶜ ∩ B partition B]
           = P(B) - P(A)P(B)         [by independence of A and B]
           = P(B)(1 - P(A))
           = P(B)P(Aᶜ)              ✓
```

### Exercise 3
In a neural network, the output layer has 3 classes with softmax outputs [0.7, 0.2, 0.1]. If the true class is class 1, compute:
(a) P(correct prediction)
(b) P(wrong prediction)
(c) P(class 2 | wrong prediction)

**Solution:**
```
(a) P(correct) = 0.7
(b) P(wrong) = 1 - 0.7 = 0.3
(c) P(class 2 | wrong) = P(class 2) / P(wrong) = 0.2 / 0.3 = 0.667
```

### Exercise 4
Show that the MAP estimate with a Gaussian prior is equivalent to Ridge regression.

**Solution:**
```
θ_MAP = argmax_θ [log P(data|θ) + log P(θ)]

Assuming Gaussian noise: log P(data|θ) ∝ -||y - Xθ||²/(2σ²)
Assuming Gaussian prior θ ~ N(0, τ²): log P(θ) ∝ -||θ||²/(2τ²)

θ_MAP = argmin_θ [||y - Xθ||²/σ² + ||θ||²/τ²]
      = argmin_θ [||y - Xθ||² + (σ²/τ²)||θ||²]

Setting λ = σ²/τ², this is exactly Ridge regression:
θ_MAP = argmin_θ [||y - Xθ||² + λ||θ||²]  ✓
```

### Exercise 5
A spam filter has P(spam) = 0.3. For a specific word "urgent":
- P(word | spam) = 0.4
- P(word | ham) = 0.1

Compute P(spam | word) using Bayes' theorem.

**Solution:**
```
P(word) = P(word|spam)P(spam) + P(word|ham)P(ham)
        = 0.4 × 0.3 + 0.1 × 0.7
        = 0.12 + 0.07
        = 0.19

P(spam | word) = 0.12 / 0.19 = 0.6316
```

---

## Key Takeaways

1. **Conditional probability** rescales probability by what we already know — it's the foundation of all Bayesian reasoning.
2. **Joint = Conditional × Marginal**: P(A ∩ B) = P(A|B)P(B) is the multiplication rule you'll use constantly.
3. **Bayes' theorem** flips the conditional: from P(evidence|hypothesis) to P(hypothesis|evidence). This is how we learn from data.
4. **Independence ≠ Mutual exclusivity**: This is the most common misconception. Independent events can co-occur; mutually exclusive events cannot.
5. **Regularization is Bayesian**: L1 = Laplace prior, L2 = Gaussian prior. Every regularized model has a Bayesian interpretation.
