# Confidence Intervals, Hypothesis Testing & A/B Testing — Complete Guide

## Table of Contents
1. [Foundations of Statistical Inference](#foundations)
2. [Confidence Intervals](#confidence-intervals)
3. [Hypothesis Testing Framework](#hypothesis-testing)
4. [Common Statistical Tests](#statistical-tests)
5. [Chi-Squared Test](#chi-squared)
6. [A/B Testing](#ab-testing)
7. [Python Implementations](#python)
8. [ML Connections](#ml-connections)
9. [Exercises](#exercises)

---

<a name="foundations"></a>
## 1. Foundations of Statistical Inference

### 1.1 Population vs. Sample

```
POPULATION (N, often unknown/infinite)        SAMPLE (n, observed data)
┌──────────────────────────────┐           ┌──────────────────┐
│  μ = population mean         │           │  x̄ = sample mean │
│  σ = population std dev      │  ◄─────── │  s = sample std  │
│  p = population proportion   │  sample   │  p̂ = sample prop │
│  ...                          │           │  ...             │
└──────────────────────────────┘           └──────────────────┘

Inference: Use sample statistics to draw conclusions about population parameters.

Key principle: Sample statistics are random variables with their own distributions.
  x̄ ~ N(μ, σ²/n)       (by CLT, for large n)
  s² is an estimator of σ²
```

### 1.2 The Central Limit Theorem (CLT) — The Foundation

```
Central Limit Theorem:
  For ANY population with mean μ and variance σ², the sampling distribution
  of the sample mean approaches a normal distribution as n → ∞:

    √n (x̄ - μ) → N(0, σ²)    as n → ∞

  Or equivalently:
    x̄ ~ N(μ, σ²/n)           for large n

  Rule of thumb: n ≥ 30 is usually "large enough"

Visual illustration:
  Population distribution    →    Sampling distribution of x̄
  ┌──────────────────┐         ┌──────────────────┐
  │  (can be any     │         │  (always bell-   │
  │   shape!)        │         │   shaped!)       │
  │    __            │         │    /‾‾\          │
  │   /  \___        │  CLT    │   /    \         │
  │  /       \___    │ ────►   │  /      \        │
  │ /            \___│         │ /        \____    │
  └──────────────────┘         └──────────────────┘
```

### 1.3 Standard Error

```
Standard Error (SE) = Standard deviation of the sampling distribution

  SE(x̄)    = σ / √n           (mean, σ known)
  SE(x̄)    = s / √n           (mean, σ estimated)
  SE(p̂)    = √(p(1-p) / n)   (proportion)
  SE(x̄₁-x̄₂)= √(s₁²/n₁ + s₂²/n₂)  (difference of means)

Relationship to confidence intervals:
  CI = point_estimate ± z* × SE

Standard Error vs. Standard Deviation:
  ┌─────────────────────┬──────────────────────────────┐
  │ Standard Deviation  │ Standard Error               │
  ├─────────────────────┼──────────────────────────────┤
  │ Variability of      │ Variability of               │
  │ individual data     │ sample statistic             │
  │ points              │ (estimator)                  │
  │                     │                              │
  │ s = √(Σ(xi-x̄)²/(n-1))│ SE = s/√n                  │
  │                     │                              │
  │ Doesn't change      │ Decreases with √n            │
  │ much with n         │ (more data → more precision) │
  └─────────────────────┴──────────────────────────────┘
```

---

<a name="confidence-intervals"></a>
## 2. Confidence Intervals

### 2.1 Definition and Interpretation

```
A confidence interval gives a range of plausible values for a population parameter.

  CI = point_estimate ± margin_of_error
     = x̄ ± z_(α/2) × SE(x̄)

Confidence Level (1 - α):
  - 90% CI: α = 0.10, z = 1.645
  - 95% CI: α = 0.05, z = 1.960
  - 99% CI: α = 0.01, z = 2.576

CORRECT Interpretation:
  "If we repeated the sampling process many times, approximately 95% of the
   constructed confidence intervals would contain the true parameter."

INCORRECT Interpretation:
  "There is a 95% probability that the true parameter lies in this interval."
  (The parameter is fixed; the interval is random!)

Visual:
  Repeated sampling and CI construction (95% CIs):

  ───|──────|──────|──────|──────|──────|──────|──────|──────|───
      ══════════════════                                    (contains μ) ✓
          ══════════════════                                 (contains μ) ✓
                    ══════════════════                      (contains μ) ✓
                 ═══════════════════════════                (contains μ) ✓
                           ══════════════════               (contains μ) ✓
        ══════════════════════════════════                  (contains μ) ✓
                          ══════════════════                (contains μ) ✓
                                  ══════════════════        (contains μ) ✓
                              ═════════════════             (misses μ) ✗
                    ══════════════════                      (contains μ) ✓
```

### 2.2 Types of Confidence Intervals

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE INTERVALS                          │
├──────────────────┬──────────────────────────────────────────────┤
│ Parameter        │ Formula                                     │
├──────────────────┼──────────────────────────────────────────────┤
│ Mean (σ known)   │ x̄ ± z_(α/2) · σ/√n                         │
│ Mean (σ unknown) │ x̄ ± t_(α/2,n-1) · s/√n                     │
│ Proportion       │ p̂ ± z_(α/2) · √(p̂(1-p̂)/n)                │
│ Difference       │ (x̄₁-x̄₂) ± z·√(s₁²/n₁+s₂²/n₂)            │
│   of means       │                                             │
│ Variance         │ ((n-1)s²/χ²_(α/2), (n-1)s²/χ²_(1-α/2))    │
│ Correlation      │ r ± z_(α/2)·SE(r)  (Fisher z-transformation)│
├──────────────────┼──────────────────────────────────────────────┤
│ Key choice:      │ z-distribution (σ known, n large)           │
│                  │ t-distribution (σ unknown)                  │
└──────────────────┴──────────────────────────────────────────────┘
```

### 2.3 CI for Mean — t-Distribution

```
When σ is unknown (which is almost always), use the t-distribution:

  x̄ ± t_(α/2, df) × s/√n

  where df = n - 1 (degrees of freedom)

t-distribution vs. Normal distribution:
  ┌──────────────────────────────────────────┐
  │   t-distribution properties:             │
  │                                          │
  │   • Heavier tails than Normal            │
  │   • More uncertainty (less info on σ)    │
  │   • Approaches Normal as n → ∞          │
  │   • Parameterized by degrees of freedom  │
  │                                          │
  │   df = 5:   ╱‾╲                           │
  │              ╱  ╲  (much heavier tails)   │
  │   df = 30:  /‾‾\                          │
  │             /    ╲ (slightly heavier)     │
  │   Normal:  /‾‾‾‾\                         │
  │            /      ╲(reference)            │
  └──────────────────────────────────────────┘

  ┌────────┬──────────────────────────────────┐
  │  df    │  95% CI critical values          │
  ├────────┼──────────────────────────────────┤
  │  1     │  12.706                          │
  │  5     │  2.571                           │
  │  10    │  2.228                           │
  │  30    │  2.042                           │
  │  60    │  2.000                           │
  │  ∞     │  1.960 (= z)                     │
  └────────┴──────────────────────────────────┘
```

### 2.4 Sample Size Determination

```
To determine required sample size for a desired margin of error E:

  For means:     n ≥ (z_(α/2) × σ / E)²
  For proportions: n ≥ (z_(α/2) / E)² × p̂(1-p̂)
                  (use p̂ = 0.5 for maximum conservativeness)

Example: 95% CI for a proportion with E = 0.03:
  n ≥ (1.96 / 0.03)² × 0.5 × 0.5 = 1067.1 → n = 1068

Trade-off:
  ┌──────────────────────────────────────────────┐
  │  Narrower CI (smaller E)  →  Need larger n  │
  │  Higher confidence (smaller α) → Larger n    │
  │  More variability (larger σ)  →  Need larger n│
  └──────────────────────────────────────────────┘
```

---

<a name="hypothesis-testing"></a>
## 3. Hypothesis Testing Framework

### 3.1 The Hypothesis Testing Procedure

```
┌─────────────────────────────────────────────────────────────────────┐
│                HYPOTHESIS TESTING WORKFLOW                           │
│                                                                     │
│  1. State Hypotheses                                                │
│     H₀ (null):      status quo, no effect, no difference            │
│     H₁ (alternative): what we want to prove                         │
│                                                                     │
│  2. Choose Significance Level α                                      │
│     Typically α = 0.05 (5%)                                         │
│                                                                     │
│  3. Select Test Statistic                                           │
│     Z-test, t-test, chi-squared, F-test, etc.                       │
│                                                                     │
│  4. Compute Test Statistic and p-value                              │
│                                                                     │
│  5. Make Decision                                                   │
│     if p-value < α: Reject H₀                                       │
│     if p-value ≥ α: Fail to reject H₀                               │
│                                                                     │
│  6. State Conclusion in Context                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Type I and Type II Errors

```
                    Truth
               ┌──────────────┬──────────────┐
               │  H₀ True     │  H₀ False    │
       ┌───────┼──────────────┼──────────────┤
Decision│Reject │  Type I Error│  Correct!    │
       │H₀     │  (False Pos) │  (Power)     │
       ├───────┼──────────────┼──────────────┤
       │Fail to│  Correct!    │  Type II Error│
       │Reject │  (True Neg)  │  (False Neg) │
       │H₀     │              │              │
       └───────┴──────────────┴──────────────┘

  α = P(Type I Error)  = P(Reject H₀ | H₀ is True)
  β = P(Type II Error) = P(Fail to Reject | H₀ is False)
  Power = 1 - β = P(Reject H₀ | H₀ is False)

Visual representation (normal distribution under H₀):

  ──────────────────────────────────────────────────►
  ←── Fail to Reject H₀ ──→←── Reject H₀ ──→
  ────────┬──────────────┬──────────────┬────────
          │              │     α/2      │
          │   1-α        │    (right    │
          │              │     tail)    │
          │  ┌──────┐    │              │
          │  │██████│    │     ┌──┐     │
          │  │██████│    │     │██│     │
          │  │██████│    │     │██│     │
  ────────┴──┴──────┴────┴─────┴──┴─────┴────────
            -z_(α/2)    μ₀   z_(α/2)

  Power diagram:
  ┌─────────────────────────────────────────────────┐
  │                                                  │
  │  Under H₀:    ╱‾‾‾‾‾╲                             │
  │             ╱  ▓▓▓▓▓ ╲  ← α region              │
  │            ╱  ▓▓▓▓▓▓▓  ╲                        │
  │                                                  │
  │  Under H₁:       ╱‾‾‾‾‾╲                        │
  │                ╱▓▓▓▓▓▓▓▓╲  ← Power region       │
  │               ╱ ▓▓▓▓▓▓▓▓ ╲                      │
  │              ╱   ▓▓▓▓▓▓▓   ╲ ← β region         │
  │                                                  │
  │  Factors affecting power:                       │
  │  • Larger sample size n → more power            │
  │  • Larger effect size → more power              │
  │  • Smaller α → less power                       │
  │  • Smaller σ → more power                       │
  └─────────────────────────────────────────────────┘
```

### 3.3 P-Values

```
Definition:
  The p-value is the probability of observing a test statistic at least
  as extreme as the one computed, assuming H₀ is true.

  p-value = P(|T| ≥ |t_obs| | H₀)

Interpretation scale:
  ┌───────────────────────────────────────────┐
  │  p < 0.001  : Very strong evidence        │
  │  p < 0.01   : Strong evidence             │
  │  p < 0.05   : Moderate evidence           │
  │  p < 0.10   : Weak evidence               │
  │  p ≥ 0.10   : No evidence against H₀      │
  └───────────────────────────────────────────┘

Common Misconception:
  "The p-value is the probability that H₀ is true."
  WRONG! The p-value is P(data | H₀), NOT P(H₀ | data).

  To get P(H₀ | data), you need Bayesian methods:
    P(H₀ | data) = P(data | H₀) · P(H₀) / P(data)

One-tailed vs. Two-tailed tests:
  ┌──────────────────────────────────────────────────────┐
  │  Two-tailed: H₁: μ ≠ μ₀    (difference in either    │
  │               direction)                             │
  │    p-value = 2 × P(T ≥ |t_obs|)                     │
  │                                                      │
  │  One-tailed: H₁: μ > μ₀     (specific direction)    │
  │    p-value = P(T ≥ t_obs)                           │
  │                                                      │
  │  Use two-tailed unless you have a strong reason      │
  │  for a directional hypothesis.                       │
  └──────────────────────────────────────────────────────┘
```

---

<a name="statistical-tests"></a>
## 4. Common Statistical Tests

### 4.1 Z-Test (Known Population Variance)

```
When to use:
  • Population variance σ² is known
  • Sample size is large (n ≥ 30)
  • Testing a population mean

Formula:
  Z = (x̄ - μ₀) / (σ / √n)

  Under H₀: Z ~ N(0, 1)

Decision rule:
  |Z| > z_(α/2) → Reject H₀

Example:
  A factory claims their lightbulbs last 1000 hours (σ = 100).
  Sample of 50 bulbs: x̄ = 980 hours.
  Test at α = 0.05: H₀: μ = 1000, H₁: μ ≠ 1000

  Z = (980 - 1000) / (100/√50) = -20/14.14 = -1.414
  
  Critical value: z_(0.025) = 1.96
  |Z| = 1.414 < 1.96 → Fail to reject H₀
  
  p-value = 2 × P(Z < -1.414) = 2 × 0.0786 = 0.157
```

### 4.2 One-Sample t-Test (Unknown Population Variance)

```
When to use:
  • Population variance σ² is unknown
  • Use sample standard deviation s instead
  • Works for any sample size (but better for n ≥ 30)

Formula:
  t = (x̄ - μ₀) / (s / √n)
  df = n - 1

  Under H₀: t ~ t_(n-1)

Example:
  A study measures cholesterol levels (n=25):
  x̄ = 210 mg/dL, s = 30 mg/dL
  Is the mean different from 200? (α = 0.05)

  t = (210 - 200) / (30/√25) = 10/6 = 1.667
  
  Critical value: t_(0.025, 24) = 2.064
  |t| = 1.667 < 2.064 → Fail to reject H₀
  
  p-value = 2 × P(t₂₄ > 1.667) ≈ 0.108
```

### 4.3 Two-Sample t-Test (Comparing Two Means)

```
When to use:
  • Comparing means of two independent groups

  ┌──────────────────────────────────────────────────────┐
  │  Equal variances (Welch's is safer in general):      │
  │                                                      │
  │  Student's t (equal variances assumed):              │
  │    t = (x̄₁ - x̄₂) / (s_p × √(1/n₁ + 1/n₂))        │
  │    s_p² = ((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)     │
  │    df = n₁ + n₂ - 2                                 │
  │                                                      │
  │  Welch's t (unequal variances):                      │
  │    t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)            │
  │    df = (s₁²/n₁ + s₂²/n₂)² /                       │
  │         [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]     │
  └──────────────────────────────────────────────────────┘

Paired t-test (before/after, matched pairs):
  t = d̄ / (s_d / √n)
  df = n - 1
  where dᵢ = x₁ᵢ - x₂ᵢ (paired differences)
```

### 4.4 Test Selection Flowchart

```
What are you testing?
  │
  ├── One mean ───────────┬── σ known ────────► Z-test
  │                      └── σ unknown ─────► t-test (one-sample)
  │
  ├── Two means ──────────┬── Independent ────► t-test (two-sample)
  │                      └── Paired ─────────► t-test (paired)
  │
  ├── Proportion(s) ──────┬── One proportion ──► Z-test for proportion
  │                      └── Two proportions ► Z-test (two-proportion)
  │
  ├── Variance(s) ────────┬── One variance ───► Chi-squared test
  │                      └── Two variances ──► F-test
  │
  ├── Categorical data ──┬── Goodness of fit ► Chi-squared GOF
  │                      └── Independence ───► Chi-squared test
  │
  └── Distribution ──────┬── Normality ──────► Shapiro-Wilk
                         └── Non-parametric ► Mann-Whitney, Kruskal-Wallis
```

---

<a name="chi-squared"></a>
## 5. Chi-Squared Test

### 5.1 Chi-Squared Goodness of Fit

```
Purpose: Test if observed frequencies match expected (theoretical) frequencies.

Formula:
  χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ

  Oᵢ = observed frequency for category i
  Eᵢ = expected frequency for category i
  df = k - 1 (k = number of categories)

Example: Is a die fair?
  Observed rolls: [18, 22, 16, 24, 19, 21] (total = 120)
  Expected (fair):  120/6 = 20 for each face

  χ² = (18-20)²/20 + (22-20)²/20 + (16-20)²/20 + 
        (24-20)²/20 + (19-20)²/20 + (21-20)²/20
      = 4/20 + 4/20 + 16/20 + 16/20 + 1/20 + 1/20
      = 42/20 = 2.1

  df = 6 - 1 = 5
  Critical value χ²_(0.05, 5) = 11.07
  2.1 < 11.07 → Fail to reject H₀ (die appears fair)
```

### 5.2 Chi-Squared Test of Independence

```
Purpose: Test if two categorical variables are independent.

Contingency table:
                Variable B
               ┌──────┬──────┬──────┬──────┐
               │ B₁   │ B₂   │ B₃   │ Total│
        ┌──────┼──────┼──────┼──────┼──────┤
        │ A₁   │ O₁₁  │ O₁₂  │ O₁₃  │ R₁   │
Var A   ├──────┼──────┼──────┼──────┼──────┤
        │ A₂   │ O₂₁  │ O₂₂  │ O₂₃  │ R₂   │
        ├──────┼──────┼──────┼──────┼──────┤
        │ Total│ C₁   │ C₂   │ C₃   │  N   │
        └──────┴──────┴──────┴──────┴──────┘

Formula:
  χ² = Σᵢ Σⱼ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ

  Eᵢⱼ = (Rᵢ × Cⱼ) / N
  df = (r - 1)(c - 1)

Example: Gender vs. Product Preference
  ┌────────┬────────┬────────┬────────┐
          │  Prod A │  Prod B │  Prod C │
  ┌───────┼─────────┼─────────┼─────────┤
  │ Male  │   20    │   30    │   50    │  (100)
  │Female │   40    │   30    │   30    │  (100)
  └───────┼─────────┼─────────┼─────────┘
            (60)     (60)      (80)     (200)

  E(Male, A)  = 100×60/200 = 30
  E(Male, B)  = 100×60/200 = 30
  E(Male, C)  = 100×80/200 = 40
  E(Fem, A)   = 100×60/200 = 30
  E(Fem, B)   = 100×60/200 = 30
  E(Fem, C)   = 100×80/200 = 40

  χ² = (20-30)²/30 + (30-30)²/30 + (50-40)²/40 +
        (40-30)²/30 + (30-30)²/30 + (30-40)²/40
      = 100/30 + 0 + 100/40 + 100/30 + 0 + 100/40
      = 3.33 + 2.5 + 3.33 + 2.5 = 11.67

  df = (2-1)(3-1) = 2
  χ²_(0.05, 2) = 5.991
  11.67 > 5.991 → Reject H₀ (gender and preference are associated)
```

### 5.3 Chi-Squared Test Assumptions

```
  ┌──────────────────────────────────────────────────┐
  │ Assumptions for Chi-Squared Test:               │
  │                                                  │
  │ 1. Observations are independent                 │
  │ 2. Expected frequency in each cell ≥ 5          │
  │ 3. Sample size is sufficiently large            │
  │ 4. Categories are mutually exclusive            │
  │                                                  │
  │ If E < 5 in some cells:                         │
  │  • Combine categories                           │
  │  • Use Fisher's Exact Test (for 2×2 tables)    │
  │  • Use Monte Carlo simulation                   │
  └──────────────────────────────────────────────────┘
```

---

<a name="ab-testing"></a>
## 6. A/B Testing

### 6.1 A/B Testing Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    A/B TESTING PIPELINE                          │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Design  │───►│  Run     │───►│ Analyze  │───►│  Decide  │ │
│  │  Test    │    │ Experiment│   │ Results  │    │ & Deploy │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
│  Step 1: Design                                                │
│    • Define metric (primary, secondary, guardrail)             │
│    • Determine MDE (Minimum Detectable Effect)                 │
│    • Calculate sample size                                     │
│    • Randomization strategy                                    │
│                                                                 │
│  Step 2: Run                                                   │
│    • Split traffic (50/50 or other)                            │
│    • Monitor for SRM (Sample Ratio Mismatch)                   │
│    • Run for pre-determined duration                           │
│                                                                 │
│  Step 3: Analyze                                               │
│    • Check for practical significance                          │
│    • Check for statistical significance                        │
│    • Segment analysis                                          │
│                                                                 │
│  Step 4: Decide                                                │
│    • Ship the winning variant                                  │
│    • Iterate with new hypotheses                               │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Sample Size Calculation for A/B Tests

```
For comparing two proportions (click-through rates, conversion rates):

  n = (z_(α/2) + z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²

  Where:
  - p₁ = baseline conversion rate
  - p₂ = expected conversion rate (p₁ + MDE)
  - α = significance level (typically 0.05)
  - β = Type II error rate (typically 0.20, power = 0.80)
  - n = sample size per group

Example:
  Baseline CTR: p₁ = 0.10
  MDE: 2% relative improvement → p₂ = 0.102
  α = 0.05, Power = 0.80

  n = (1.96 + 0.842)² × (0.10×0.90 + 0.102×0.898) / (0.002)²
    ≈ 7.849 × 0.1830 / 0.000004
    ≈ 358,530 per group
  → Total: ~717,060 users needed!

  For comparing two means:
  n = 2 × (z_(α/2) + z_β)² × σ² / δ²

  Where δ = minimum detectable effect (MDE)
```

### 6.3 Common A/B Test Pitfalls

```
┌─────────────────────────────────────────────────────────────┐
│  PITFALL                      │  SOLUTION                    │
├───────────────────────────────┼──────────────────────────────┤
│  Peeking at results early     │  Pre-commit to sample size   │
│  (inflates Type I error)      │  or use sequential testing   │
├───────────────────────────────┼──────────────────────────────┤
│  Multiple testing problem     │  Bonferroni correction:      │
│  (testing many metrics)       │  α_adj = α / k               │
│                               │  Or use BH procedure          │
├───────────────────────────────┼──────────────────────────────┤
│  Sample Ratio Mismatch (SRM)  │  Use chi-squared test to     │
│  (unequal randomization)      │  check 50/50 split            │
├───────────────────────────────┼──────────────────────────────┤
│  Novelty effect               │  Run test long enough for     │
│  (users like anything new)    │  novelty to wear off          │
├───────────────────────────────┼──────────────────────────────┤
│  Simpson's Paradox            │  Always segment analysis      │
│  (treatment looks good        │  before concluding            │
│   overall but bad in key      │                              │
│   segments)                   │                              │
├───────────────────────────────┼──────────────────────────────┤
│  Insufficient power           │  Do power analysis BEFORE     │
│  (can't detect real effects)  │  running the experiment       │
├───────────────────────────────┼──────────────────────────────┤
│  Crossover / contamination    │  Ensure treatment groups      │
│  (users in A see B's feature) │  are truly isolated           │
└───────────────────────────────┴──────────────────────────────┘
```

### 6.4 A/B Testing in Machine Learning

```
┌──────────────────────────────────────────────────────────────┐
│  Where A/B Testing Meets ML:                                  │
│                                                              │
│  1. Model Evaluation:                                        │
│     A: Current production model                              │
│     B: New candidate model                                   │
│     Metric: CTR, conversion, revenue, engagement             │
│                                                              │
│  2. Feature Testing:                                         │
│     A: Model without new feature                             │
│     B: Model with new feature                                │
│                                                              │
│  3. Bandit Testing (multi-armed):                            │
│     Instead of fixed split, dynamically allocate traffic     │
│     to better-performing variants                            │
│                                                              │
│  4. Interleaving (for ranking):                              │
│     Mix results from A and B, measure clicks per ranking    │
│     More sensitive than traditional A/B for rankers          │
│                                                              │
│  5. CUPED (Controlled-experiment Using Pre-Experiment Data):│
│     Reduce variance using pre-experiment data               │
│     θ_adj = θ - γ(X - E[X])                                 │
│     Dramatically reduces required sample size                │
└──────────────────────────────────────────────────────────────┘
```

---

<a name="python"></a>
## 7. Python Implementations

### 7.1 Confidence Intervals

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================
# 7.1.1 Confidence Interval for Mean
# ============================================================
def confidence_interval_mean(data, confidence=0.95):
    """Compute confidence interval for the population mean."""
    n = len(data)
    x_bar = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    
    # t-distribution for unknown variance
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_crit * se
    
    return (x_bar - margin, x_bar + margin, x_bar, margin)


def ci_demo():
    """Demonstrate confidence interval concepts."""
    np.random.seed(42)
    
    # Generate data from unknown distribution
    population = np.random.exponential(scale=5, size=100000)
    true_mean = population.mean()
    
    print(f"True population mean: {true_mean:.4f}")
    print()
    
    # Compute CI for different sample sizes
    print(f"{'n':>5} | {'CI Lower':>10} | {'CI Upper':>10} | {'x̄':>8} | "
          f"{'Width':>8} | {'Contains μ?':>12}")
    print("-" * 65)
    
    for n in [10, 30, 100, 500, 1000]:
        sample = np.random.choice(population, size=n)
        ci_lower, ci_upper, x_bar, margin = confidence_interval_mean(sample)
        contains = "YES ✓" if ci_lower <= true_mean <= ci_upper else "NO ✗"
        print(f"{n:5d} | {ci_lower:10.4f} | {ci_upper:10.4f} | {x_bar:8.4f} | "
              f"{2*margin:8.4f} | {contains:>12}")


# ============================================================
# 7.1.2 Confidence Interval for Proportion
# ============================================================
def confidence_interval_proportion(p_hat, n, confidence=0.95):
    """Compute confidence interval for a population proportion."""
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    margin = z_crit * se
    
    return (p_hat - margin, p_hat + margin)


def proportion_ci_demo():
    """CI for proportions - like CTR, conversion rate."""
    # Example: 1000 users, 120 clicked
    n = 1000
    clicks = 120
    p_hat = clicks / n
    
    ci_90 = confidence_interval_proportion(p_hat, n, 0.90)
    ci_95 = confidence_interval_proportion(p_hat, n, 0.95)
    ci_99 = confidence_interval_proportion(p_hat, n, 0.99)
    
    print(f"Observed CTR: {p_hat:.4f} ({clicks}/{n})")
    print(f"  90% CI: ({ci_90[0]:.4f}, {ci_90[1]:.4f})")
    print(f"  95% CI: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")
    print(f"  99% CI: ({ci_99[0]:.4f}, {ci_99[1]:.4f})")


# ============================================================
# 7.1.3 Bootstrap Confidence Intervals
# ============================================================
def bootstrap_ci(data, statistic=np.mean, n_bootstraps=10000,
                 confidence=0.95, method='percentile'):
    """Compute bootstrap confidence interval for any statistic."""
    boot_stats = []
    n = len(data)
    
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))
    
    boot_stats = np.array(boot_stats)
    
    if method == 'percentile':
        alpha = (1 - confidence) / 2
        lower = np.percentile(boot_stats, alpha * 100)
        upper = np.percentile(boot_stats, (1 - alpha) * 100)
    elif method == 'normal':
        se = np.std(boot_stats, ddof=1)
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = np.mean(boot_stats) - z * se
        upper = np.mean(boot_stats) + z * se
    
    return lower, upper, boot_stats


def bootstrap_demo():
    """Bootstrap CI for median (non-parametric)."""
    np.random.seed(42)
    data = np.random.lognormal(mean=2, sigma=0.5, size=100)
    
    # Bootstrap CI for median
    ci_lower, ci_upper, boot_medians = bootstrap_ci(
        data, statistic=np.median, n_bootstraps=10000
    )
    
    print(f"Sample median: {np.median(data):.4f}")
    print(f"Bootstrap 95% CI for median: ({ci_lower:.4f}, {ci_upper:.4f})")
    
    # Visualize bootstrap distribution
    plt.figure(figsize=(10, 5))
    plt.hist(boot_medians, bins=50, density=True, alpha=0.7, color='steelblue')
    plt.axvline(np.median(data), color='red', linestyle='--', label='Sample median')
    plt.axvline(ci_lower, color='green', linestyle='-', label=f'95% CI: ({ci_lower:.2f}, {ci_upper:.2f})')
    plt.axvline(ci_upper, color='green', linestyle='-')
    plt.xlabel('Bootstrap Median')
    plt.ylabel('Density')
    plt.title('Bootstrap Distribution of Median')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bootstrap_ci.png', dpi=150)
    plt.show()
```

### 7.2 Hypothesis Tests

```python
# ============================================================
# 7.2.1 Z-Test and t-Test
# ============================================================
def z_test(sample_mean, pop_mean, pop_std, n, alpha=0.05, 
           alternative='two-sided'):
    """Perform a one-sample Z-test."""
    se = pop_std / np.sqrt(n)
    z_stat = (sample_mean - pop_mean) / se
    
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
    
    z_crit = stats.norm.ppf(1 - alpha/2) if alternative == 'two-sided' \
             else stats.norm.ppf(1 - alpha)
    
    reject = p_value < alpha
    
    return {
        'z_statistic': z_stat,
        'p_value': p_value,
        'critical_value': z_crit,
        'reject_h0': reject,
        'confidence_interval': (sample_mean - z_crit * se, 
                                sample_mean + z_crit * se)
    }


def t_test_one_sample(data, pop_mean=0, alpha=0.05, alternative='two-sided'):
    """Perform a one-sample t-test."""
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    se = s / np.sqrt(n)
    
    t_stat = (x_bar - pop_mean) / se
    df = n - 1
    
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_stat, df)
    elif alternative == 'less':
        p_value = stats.t.cdf(t_stat, df)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'reject_h0': p_value < alpha
    }


def t_test_two_sample(data1, data2, alpha=0.05, equal_var=False):
    """Perform a two-sample t-test."""
    result = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    return {
        't_statistic': result.statistic,
        'p_value': result.pvalue,
        'reject_h0': result.pvalue < alpha,
        'mean_diff': np.mean(data1) - np.mean(data2),
        'method': "Student's t" if equal_var else "Welch's t"
    }


def hypothesis_test_demo():
    """Comprehensive hypothesis testing demonstration."""
    np.random.seed(42)
    
    print("=" * 60)
    print("ONE-SAMPLE Z-TEST")
    print("=" * 60)
    # Factory example: bulbs should last 1000 hours, σ=100
    sample = np.random.normal(990, 100, 50)
    result = z_test(np.mean(sample), 1000, 100, 50)
    print(f"Z-statistic: {result['z_statistic']:.4f}")
    print(f"P-value:     {result['p_value']:.4f}")
    print(f"Reject H₀:   {result['reject_h0']}")
    
    print("\n" + "=" * 60)
    print("ONE-SAMPLE T-TEST (scipy)")
    print("=" * 60)
    result_scipy = stats.ttest_1samp(sample, 1000)
    print(f"t-statistic: {result_scipy.statistic:.4f}")
    print(f"P-value:     {result_scipy.pvalue:.4f}")
    
    print("\n" + "=" * 60)
    print("TWO-SAMPLE T-TEST")
    print("=" * 60)
    group_a = np.random.normal(25, 5, 100)
    group_b = np.random.normal(27, 5, 100)
    
    result = t_test_two_sample(group_a, group_b)
    print(f"Mean A: {np.mean(group_a):.2f}, Mean B: {np.mean(group_b):.2f}")
    print(f"Method:    {result['method']}")
    print(f"t-stat:    {result['t_statistic']:.4f}")
    print(f"P-value:   {result['p_value']:.4f}")
    print(f"Reject H₀: {result['reject_h0']}")
    
    print("\n" + "=" * 60)
    print("PAIRED T-TEST")
    print("=" * 60)
    before = np.random.normal(100, 15, 30)
    after = before + np.random.normal(5, 10, 30)  # treatment effect
    
    result_paired = stats.ttest_rel(before, after)
    print(f"Mean before: {np.mean(before):.2f}")
    print(f"Mean after:  {np.mean(after):.2f}")
    print(f"t-statistic: {result_paired.statistic:.4f}")
    print(f"P-value:     {result_paired.pvalue:.4f}")


# ============================================================
# 7.2.2 Chi-Squared Tests
# ============================================================
def chi_squared_gof(observed, expected=None, alpha=0.05):
    """Chi-squared goodness of fit test."""
    observed = np.array(observed)
    n = np.sum(observed)
    
    if expected is None:
        expected = np.full_like(observed, n / len(observed), dtype=float)
    
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'df': df,
        'reject_h0': p_value < alpha
    }


def chi_squared_independence(observed, alpha=0.05):
    """Chi-squared test of independence for contingency table."""
    observed = np.array(observed)
    
    # scipy implementation
    chi2, p, df, expected = stats.chi2_contingency(observed)
    
    return {
        'chi2_statistic': chi2,
        'p_value': p,
        'df': df,
        'expected': expected,
        'reject_h0': p < alpha
    }


def chi_squared_demo():
    """Chi-squared test demonstrations."""
    print("=" * 60)
    print("CHI-SQUARED GOODNESS OF FIT")
    print("=" * 60)
    
    # Die fairness test
    observed = [18, 22, 16, 24, 19, 21]
    result = chi_squared_gof(observed)
    print(f"Observed: {observed}")
    print(f"χ² = {result['chi2_statistic']:.4f}")
    print(f"p-value = {result['p_value']:.4f}")
    print(f"Reject H₀: {result['reject_h0']}")
    
    # scipy version
    result_scipy = stats.chisquare(observed)
    print(f"scipy χ² = {result_scipy.statistic:.4f}, p = {result_scipy.pvalue:.4f}")
    
    print("\n" + "=" * 60)
    print("CHI-SQUARED TEST OF INDEPENDENCE")
    print("=" * 60)
    
    # Gender vs. Product preference
    observed_table = np.array([
        [20, 30, 50],  # Male
        [40, 30, 30]   # Female
    ])
    
    result = chi_squared_independence(observed_table)
    print(f"Observed table:\n{observed_table}")
    print(f"\nExpected frequencies:\n{np.round(result['expected'], 2)}")
    print(f"\nχ² = {result['chi2_statistic']:.4f}")
    print(f"p-value = {result['p_value']:.4f}")
    print(f"df = {result['df']}")
    print(f"Reject H₀: {result['reject_h0']}")
    
    # Cramér's V (effect size)
    n = np.sum(observed_table)
    min_dim = min(observed_table.shape) - 1
    cramers_v = np.sqrt(result['chi2_statistic'] / (n * min_dim))
    print(f"Cramér's V (effect size): {cramers_v:.4f}")
    # Interpretation: 0.1=small, 0.3=medium, 0.5=large


# ============================================================
# 7.2.3 ANOVA (F-Test for Multiple Groups)
# ============================================================
def anova_test(*groups, alpha=0.05):
    """One-way ANOVA test."""
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta-squared)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'reject_h0': p_value < alpha
    }


def anova_demo():
    """ANOVA demonstration."""
    np.random.seed(42)
    
    # Three teaching methods
    method_a = np.random.normal(75, 10, 30)
    method_b = np.random.normal(80, 10, 30)
    method_c = np.random.normal(85, 10, 30)
    
    result = anova_test(method_a, method_b, method_c)
    print(f"F-statistic: {result['f_statistic']:.4f}")
    print(f"P-value:     {result['p_value']:.6f}")
    print(f"η² (effect): {result['eta_squared']:.4f}")
    print(f"Reject H₀:   {result['reject_h0']}")
    
    # Post-hoc: Tukey's HSD
    from scipy.stats import tukey_hsd
    all_data = np.concatenate([method_a, method_b, method_c])
    groups = np.concatenate([
        np.full(30, 'A'), np.full(30, 'B'), np.full(30, 'C')
    ])
    result_tukey = tukey_hsd(groups, all_data)
    print(f"\nTukey HSD results:\n{result_tukey}")
```

### 7.3 A/B Testing Implementation

```python
# ============================================================
# 7.3.1 A/B Test for Proportions (CTR Comparison)
# ============================================================
def ab_test_proportions(n_a, conversions_a, n_b, conversions_b, 
                         alpha=0.05):
    """A/B test comparing two proportions."""
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)
    
    # Z-test for proportions
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z_stat = (p_b - p_a) / se
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Confidence interval for the difference
    se_diff = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    ci = (p_b - p_a - 1.96*se_diff, p_b - p_a + 1.96*se_diff)
    
    # Lift
    relative_lift = (p_b - p_a) / p_a * 100
    
    return {
        'p_a': p_a, 'p_b': p_b,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_diff': ci,
        'relative_lift': relative_lift,
        'significant': p_value < alpha
    }


def ab_test_sample_size(p_baseline, mde_relative, alpha=0.05, power=0.80):
    """Calculate required sample size per group for A/B test."""
    p_a = p_baseline
    p_b = p_baseline * (1 + mde_relative)
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = (z_alpha + z_beta)**2 * (p_a*(1-p_a) + p_b*(1-p_b)) / (p_b - p_a)**2
    
    return int(np.ceil(n))


def ab_testing_demo():
    """Complete A/B testing demonstration."""
    print("=" * 60)
    print("A/B TEST: COMPARING CONVERSION RATES")
    print("=" * 60)
    
    # Scenario: E-commerce checkout button color
    n_control = 15000
    conv_control = 450    # 3.0%
    n_treatment = 15000
    conv_treatment = 540  # 3.6%
    
    result = ab_test_proportions(n_control, conv_control, 
                                  n_treatment, conv_treatment)
    
    print(f"Control:   {result['p_a']:.4f} ({conv_control}/{n_control})")
    print(f"Treatment: {result['p_b']:.4f} ({conv_treatment}/{n_treatment})")
    print(f"Z-statistic: {result['z_statistic']:.4f}")
    print(f"P-value:     {result['p_value']:.6f}")
    print(f"95% CI for diff: ({result['ci_diff'][0]:.4f}, {result['ci_diff'][1]:.4f})")
    print(f"Relative lift:  {result['relative_lift']:.2f}%")
    print(f"Statistically significant: {result['significant']}")
    
    print("\n" + "=" * 60)
    print("SAMPLE SIZE CALCULATION")
    print("=" * 60)
    
    baseline_ctr = 0.03  # 3% baseline
    for mde in [0.01, 0.02, 0.05, 0.10, 0.20]:
        n = ab_test_sample_size(baseline_ctr, mde)
        print(f"MDE {mde*100:5.1f}%: {n:>10,} users per group "
              f"({2*n:>12,} total)")


# ============================================================
# 7.3.2 Multiple Testing Correction
# ============================================================
def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    p_values = np.array(p_values)
    adjusted_alpha = alpha / len(p_values)
    significant = p_values < adjusted_alpha
    
    return {
        'original_alpha': alpha,
        'adjusted_alpha': adjusted_alpha,
        'significant': significant,
        'adjusted_p': p_values * len(p_values)  # Bonferroni adjusted p-values
    }


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg procedure for FDR control."""
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Find threshold
    thresholds = alpha * np.arange(1, n+1) / n
    
    # Largest k where p_(k) <= k * α / n
    significant_sorted = sorted_p <= thresholds
    if np.any(significant_sorted):
        max_k = np.max(np.where(significant_sorted))
        result = np.zeros(n, dtype=bool)
        result[sorted_indices[:max_k+1]] = True
    else:
        result = np.zeros(n, dtype=bool)
    
    return result


def multiple_testing_demo():
    """Demonstrate multiple testing corrections."""
    np.random.seed(42)
    
    # Simulate 20 tests (5 truly significant, 15 null)
    p_values = list(np.random.uniform(0.001, 0.04, 5))  # 5 real effects
    p_values.extend(np.random.uniform(0.05, 0.95, 15))  # 15 null
    p_values = np.array(p_values)
    
    print("Uncorrected (α=0.05):", np.sum(p_values < 0.05), "significant")
    
    bonf = bonferroni_correction(p_values)
    print("Bonferroni:", np.sum(bonf['significant']), "significant")
    
    bh = benjamini_hochberg(p_values)
    print("Benjamini-Hochberg:", np.sum(bh), "significant")
    
    print(f"\n{'Test':>4} | {'p-value':>10} | {'Uncorrected':>12} | "
          f"{'Bonferroni':>12} | {'BH':>8}")
    print("-" * 55)
    for i, p in enumerate(p_values):
        unc = p < 0.05
        bon = bonf['significant'][i]
        bho = bh[i]
        print(f"{i:4d} | {p:10.6f} | {'SIGNIF' if unc else 'ns':>12} | "
              f"{'SIGNIF' if bon else 'ns':>12} | {'SIGNIF' if bho else 'ns':>8}")
```

---

<a name="ml-connections"></a>
## 8. ML Connections

### 8.1 Statistical Tests in Model Evaluation

```
┌──────────────────────────────────────────────────────────────┐
│  Statistical Tests in ML Workflows:                           │
│                                                              │
│  Model Comparison:                                            │
│  ┌────────────────────────────────────────────────────┐      │
│  │  McNemar's Test: Compare two classifiers           │      │
│  │    H₀: Both models have the same error rate         │      │
│  │    Use: Paired comparison on same test set          │      │
│  │                                                      │      │
│  │  Paired t-test: Compare cross-validation scores     │      │
│  │    H₀: Models have equal mean performance            │      │
│  │    Use: k-fold CV results                           │      │
│  │                                                      │      │
│  │  Friedman Test: Compare multiple models             │      │
│  │    Non-parametric alternative to repeated ANOVA     │      │
│  │    Use: k-fold CV with >2 models                    │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  Feature Selection:                                           │
│  ┌────────────────────────────────────────────────────┐      │
│  │  Chi-squared: Feature vs. target relationship       │      │
│  │  t-test: Mean feature value differs by class       │      │
│  │  ANOVA: Feature variance across multiple classes   │      │
│  │  Mutual Information: Non-parametric dependency      │      │
│  └────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 Statistical Significance in ML Experiments

```python
def compare_models_cv(model1, model2, X, y, cv=5):
    """
    Compare two models using paired t-test on CV scores.
    
    Returns statistical test of whether model1 is significantly
    better than model2.
    """
    from sklearn.model_selection import cross_val_score
    
    scores1 = cross_val_score(model1, X, y, cv=cv, scoring='accuracy')
    scores2 = cross_val_score(model2, X, y, cv=cv, scoring='accuracy')
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Check if improvement is significant
    mean_diff = np.mean(scores1) - np.mean(scores2)
    
    return {
        'model1_mean': np.mean(scores1),
        'model2_mean': np.mean(scores2),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def permutation_test(model1, model2, X, y, n_permutations=1000):
    """
    Permutation test for model comparison.
    Non-parametric alternative when normality assumption is violated.
    """
    from sklearn.model_selection import cross_val_score
    
    scores1 = cross_val_score(model1, X, y, cv=5)
    scores2 = cross_val_score(model2, X, y, cv=5)
    observed_diff = np.mean(scores1) - np.mean(scores2)
    
    # Permutation: shuffle labels
    count = 0
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        s1 = cross_val_score(model1, X, y_perm, cv=5)
        s2 = cross_val_score(model2, X, y_perm, cv=5)
        perm_diff = np.mean(s1) - np.mean(s2)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    
    p_value = count / n_permutations
    return p_value
```

---

<a name="exercises"></a>
## 9. Exercises

### Exercise 1: CI Fundamentals (Beginner)
```python
# Generate 100 samples from a normal distribution with μ=50, σ=10.
# Compute 95% CI for the mean. Repeat 1000 times and count how
# many CIs contain the true mean. Should be approximately 950.
```

### Exercise 2: Power Analysis (Intermediate)
```python
# A drug company claims their new drug reduces blood pressure by
# 5 mmHg more than placebo (σ = 12 mmHg).
# What sample size per group is needed for 80% power at α = 0.05?
# Then simulate the experiment and verify the power empirically.
```

### Exercise 3: A/B Test Framework (Intermediate)
```python
# Build a complete A/B test analysis:
# 1. Simulate click data for control (CTR=2.5%) and treatment (CTR=3.0%)
# 2. Calculate sample size needed for 80% power
# 3. Run the test and compute results
# 4. Create a visualization of the difference distribution
# 5. Apply sequential testing (check after every 1000 users)
```

### Exercise 4: Chi-Squared Feature Selection (Advanced)
```python
# Use the Titanic dataset or similar:
# 1. Create a contingency table for each categorical feature vs. survival
# 2. Run chi-squared test for each feature
# 3. Rank features by statistical significance
# 4. Compare with sklearn's chi2 feature selection
# 5. Build a classifier with and without top features
```

### Exercise 5: Bootstrap Model Comparison (Advanced)
```python
# Compare two classifiers using bootstrap confidence intervals:
# 1. Train logistic regression and random forest on a dataset
# 2. Bootstrap the test set 1000 times
# 3. Compute accuracy difference for each bootstrap sample
# 4. Construct 95% CI for the accuracy difference
# 5. Determine if one model is significantly better
```

### Exercise 6: Multiple Testing in Feature Selection (Advanced)
```python
# When testing 100 features for significance:
# 1. Generate data with 5 truly informative features out of 100
# 2. Run t-tests for each feature
# 3. Apply no correction, Bonferroni, and BH
# 4. Calculate: true positives, false positives, false negatives
# 5. Compare FDR (false discovery rate) for each method
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                       KEY FORMULAS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Confidence Interval:                                            │
│    x̄ ± z_(α/2) × σ/√n         (σ known)                        │
│    x̄ ± t_(α/2,n-1) × s/√n     (σ unknown)                      │
│                                                                  │
│  Z-test:  Z = (x̄ - μ₀)/(σ/√n)                                  │
│  t-test:  t = (x̄ - μ₀)/(s/√n)                                  │
│  χ² test: χ² = Σ(Oᵢ - Eᵢ)²/Eᵢ                                 │
│                                                                  │
│  A/B Test (proportions):                                        │
│    Z = (p̂_B - p̂_A) / √(p̂_pool(1-p̂_pool)(1/nA+1/nB))        │
│                                                                  │
│  Bonferroni: α_adj = α/k                                         │
│                                                                  │
│  Power = 1 - β = P(Reject H₀ | H₁ is true)                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  CRITICAL CONCEPTS:                                              │
│  • p-value ≠ P(H₀ is true)                                      │
│  • Statistical significance ≠ Practical significance            │
│  • Always pre-register your analysis plan                       │
│  • Correct for multiple comparisons                              │
│  • Power analysis before collecting data                        │
└─────────────────────────────────────────────────────────────────┘
```
