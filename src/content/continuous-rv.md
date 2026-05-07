# Continuous Random Variables: PDF, CDF, and Key Distributions

## Why This Matters in AI/ML

Continuous random variables are the mathematical foundation of **every probabilistic ML model**. When a neural network outputs a probability score, when you compute the likelihood of data under a Gaussian, when you model waiting times — you're working with continuous distributions.

Key ML applications:
- **Gaussian Naive Bayes**: Assumes features follow normal distributions
- **Linear Regression**: Assumes errors are normally distributed
- **VAEs (Variational Autoencoders)**: Model data as samples from continuous distributions
- **Bayesian Optimization**: Uses acquisition functions based on continuous distributions
- **Anomaly Detection**: Flag data points with low probability density

## PDF vs CDF vs PMF

```
┌──────────────────────────────────────────────────────────┐
│              DISCRETE vs CONTINUOUS                       │
│                                                          │
│  DISCRETE (PMF):           CONTINUOUS (PDF):             │
│  P(X = x) = specific      f(x) = density               │
│  probability               P(X = x) = 0  (always!)      │
│                                                          │
│  Probability Mass          Probability Density            │
│  Function                  Function                      │
│                                                          │
│  Roll a die:                Measure height:              │
│  P(X=3) = 1/6              f(170cm) = 0.04             │
│  P(X=3.5) = 0              P(169.5 < X < 170.5) = f*Δx │
│                                                          │
│  CDF: F(x) = P(X ≤ x)     CDF: F(x) = P(X ≤ x)        │
│           = Σ P(X = xi)          = ∫₋∞ˣ f(t)dt        │
└──────────────────────────────────────────────────────────┘
```

## Probability Density Function (PDF)

The PDF f(x) satisfies:
- f(x) ≥ 0 for all x
- ∫₋∞^∞ f(x) dx = 1 (total probability = 1)
- P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx = F(b) - F(a)

**Key insight**: f(x) is NOT a probability. It's a density. The actual probability is the **area under the curve**.

## Cumulative Distribution Function (CDF)

F(x) = P(X ≤ x) = ∫₋∞ˣ f(t) dt

Properties:
- F(-∞) = 0, F(+∞) = 1
- Non-decreasing
- P(a < X ≤ b) = F(b) - F(a)

## The Normal Distribution (Gaussian)

The most important continuous distribution in ML.

### PDF Formula

```
f(x) = (1 / (σ√(2π))) × e^(-(x-μ)²/(2σ²))

Where:
  μ = mean (center of the distribution)
  σ = standard deviation (spread)
  σ² = variance
  π ≈ 3.14159
  e ≈ 2.71828
```

### The Bell Curve

```
        f(x)
         │
         │        ╱‾‾╲
         │      ╱      ╲
         │    ╱          ╲
         │   ╱            ╲
         │  ╱              ╲
         │ ╱                ╲
    ─────┼──────────────────────┼──── x
       μ-3σ  μ-σ  μ  μ+σ  μ+3σ
         
  68.2% of data within μ ± 1σ
  95.4% of data within μ ± 2σ
  99.7% of data within μ ± 3σ
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Plot normal distribution with different parameters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Different means, same variance
x = np.linspace(-10, 10, 300)
for mu in [-3, 0, 3]:
    axes[0].plot(x, norm.pdf(x, mu, 1), label=f'μ={mu}')
axes[0].set_title('Different Means (σ=1)')
axes[0].legend()

# Same mean, different variances
for sigma in [0.5, 1, 2]:
    axes[1].plot(x, norm.pdf(x, 0, sigma), label=f'σ={sigma}')
axes[1].set_title('Different Variances (μ=0)')
axes[1].legend()

# Standard Normal
x = np.linspace(-4, 4, 300)
axes[2].plot(x, norm.pdf(x, 0, 1), 'b-', linewidth=2)
axes[2].fill_between(x[x<=1], norm.pdf(x[x<=1], 0, 1), alpha=0.3)
axes[2].set_title('P(Z ≤ 1) ≈ 0.8413')
axes[2].annotate('84.13%', xy=(0.5, 0.15), fontsize=14)

plt.tight_layout()
plt.show()
```

## Standard Normal Distribution

Z = (X - μ) / σ follows N(0, 1)

```python
from scipy.stats import norm

# Key percentiles
print(f"P(Z ≤ 1.96) = {norm.cdf(1.96):.4f}")  # 0.9750
print(f"P(Z ≤ -1.96) = {norm.cdf(-1.96):.4f}")  # 0.0250
print(f"P(|Z| ≤ 1.96) = {norm.cdf(1.96) - norm.cdf(-1.96):.4f}")  # 0.9500
print(f"P(Z ≤ 0) = {norm.cdf(0):.4f}")  # 0.5000
print(f"P(Z > 2.58) = {1 - norm.cdf(2.58):.4f}")  # 0.0049
```

## Exponential Distribution

Models **time between events** (waiting times).

```
PDF: f(x) = λe^(-λx)  for x ≥ 0
CDF: F(x) = 1 - e^(-λx)
Mean: 1/λ
Variance: 1/λ²
```

```python
from scipy.stats import expon
import numpy as np

# Time between website visits (average: 10 minutes)
lam = 1/10  # rate parameter
x = np.linspace(0, 50, 300)

plt.figure(figsize=(10, 4))
plt.plot(x, expon.pdf(x, scale=10), 'b-', linewidth=2)
plt.fill_between(x[x<=20], expon.pdf(x[x<=20], scale=10), alpha=0.3)
plt.title('P(wait ≤ 20 min) = {:.4f}'.format(expon.cdf(20, scale=10)))
plt.xlabel('Minutes')
plt.ylabel('Density')
plt.show()

print(f"P(wait ≤ 10 min) = {expon.cdf(10, scale=10):.4f}")
print(f"P(wait > 20 min) = {1 - expon.cdf(20, scale=10):.4f}")
print(f"P(10 < wait < 30) = {expon.cdf(30, scale=10) - expon.cdf(10, scale=10):.4f}")
```

## Student's t-Distribution

Used when **sample size is small** and population σ is unknown. Has **heavier tails** than normal.

```
As n → ∞, t-distribution → normal distribution
Degrees of freedom: df = n - 1
```

```python
from scipy.stats import t, norm
import numpy as np

x = np.linspace(-5, 5, 300)
plt.figure(figsize=(10, 5))
for df in [1, 3, 10, 30]:
    plt.plot(x, t.pdf(x, df), label=f't (df={df})')
plt.plot(x, norm.pdf(x), 'k--', linewidth=2, label='Normal')
plt.title('t-distribution vs Normal')
plt.legend()
plt.show()

# 95% CI using t-distribution vs normal
from scipy.stats import t
df = 9  # 10 samples
t_critical = t.ppf(0.975, df)
z_critical = norm.ppf(0.975)
print(f"t critical (df=9):  ±{t_critical:.4f}")
print(f"z critical (normal): ±{z_critical:.4f}")
print(f"t gives wider interval → more conservative with small samples")
```

## Chi-Squared Distribution

Distribution of **sum of squared standard normals**. Used in:
- Goodness-of-fit tests
- Test of independence
- Variance estimation

```
χ²(k) = Z₁² + Z₂² + ... + Zₖ²  (k = degrees of freedom)
Mean = k, Variance = 2k
```

```python
from scipy.stats import chi2
import numpy as np

x = np.linspace(0, 20, 300)
plt.figure(figsize=(10, 5))
for df in [1, 2, 5, 10]:
    plt.plot(x, chi2.pdf(x, df), label=f'χ²(df={df})')
plt.title('Chi-Squared Distribution')
plt.legend()
plt.show()

# Chi-squared test example
from scipy.stats import chi2_contingency
observed = np.array([[50, 30, 20], [30, 40, 30]])  # 2x3 contingency table
chi2_stat, p_val, dof, expected = chi2_contingency(observed)
print(f"Chi-squared stat: {chi2_stat:.4f}")
print(f"p-value: {p_val:.4f}")
print(f"Degrees of freedom: {dof}")
```

## Poisson Distribution (as continuous approximation)

Models **count of events in a fixed interval**.

```
P(X = k) = (λ^k × e^(-λ)) / k!
Mean = Variance = λ
```

```python
from scipy.stats import poisson
import numpy as np

# Website gets average 5 requests per minute
lam = 5
k = np.arange(0, 20)

plt.figure(figsize=(10, 4))
plt.bar(k, poisson.pmf(k, lam), alpha=0.7, color='steelblue')
plt.title('Poisson Distribution (λ=5): Requests per minute')
plt.xlabel('Number of requests')
plt.ylabel('Probability')
plt.show()

print(f"P(exactly 5 requests) = {poisson.pmf(5, lam):.4f}")
print(f"P(≤ 3 requests) = {poisson.cdf(3, lam):.4f}")
print(f"P(> 8 requests) = {1 - poisson.cdf(8, lam):.4f}")
```

## Real ML Application: Gaussian Naive Bayes

```python
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
model = GaussianNB()
scores = cross_val_score(model, X, y, cv=5)
print(f"GaussianNB accuracy: {scores.mean():.4f} (±{scores.std():.4f})")

# The model assumes each feature follows a normal distribution
model.fit(X, y)
print(f"\nLearned parameters:")
print(f"Class priors (π): {model.class_prior_}")
print(f"Means (μ): {model.theta_.round(2)}")
print(f"Variances (σ²): {model.var_.round(2)}")
```

## 🧠 Exercises

### Exercise 1: A factory produces light bulbs with lifespan ~ N(1000, 200²) hours. What fraction lasts between 800 and 1200 hours? Between 600 and 1400 hours?

<details>
<summary>Solution</summary>

```python
from scipy.stats import norm
mu, sigma = 1000, 200

p1 = norm.cdf(1200, mu, sigma) - norm.cdf(800, mu, sigma)  # 0.6827
p2 = norm.cdf(1400, mu, sigma) - norm.cdf(600, mu, sigma)  # 0.9545

print(f"P(800 < X < 1200) = {p1:.4f}")  # 68.27%
print(f"P(600 < X < 1400) = {p2:.4f}")  # 95.45%
```

</details>

### Exercise 2: Customer service calls arrive exponentially with average 8 minutes between calls. What's the probability of waiting more than 15 minutes? Less than 3 minutes?

<details>
<summary>Solution</summary>

```python
from scipy.stats import expon
scale = 8  # 1/λ

p_gt15 = 1 - expon.cdf(15, scale=scale)  # 0.1496
p_lt3 = expon.cdf(3, scale=scale)  # 0.3127

print(f"P(wait > 15 min) = {p_gt15:.4f}")
print(f"P(wait < 3 min) = {p_lt3:.4f}")
```

</details>

### Exercise 3: Generate 1000 samples from N(50, 10²). Compute the sample mean and standard deviation. Use the t-distribution to construct a 95% confidence interval for the population mean.

<details>
<summary>Solution</summary>

```python
import numpy as np
from scipy.stats import t

np.random.seed(42)
sample = np.random.normal(50, 10, 1000)
n = len(sample)
x_bar = sample.mean()
s = sample.std(ddof=1)
se = s / np.sqrt(n)
t_crit = t.ppf(0.975, df=n-1)

ci = (x_bar - t_crit * se, x_bar + t_crit * se)
print(f"Sample mean: {x_bar:.4f}")
print(f"95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
print(f"True μ = 50, captured: {ci[0] <= 50 <= ci[1]}")
```

</details>
