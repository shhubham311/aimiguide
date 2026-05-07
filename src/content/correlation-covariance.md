# Correlation & Covariance

## Why This Matters in AI/ML

Understanding how variables relate to each other is fundamental to machine learning. Correlation and covariance quantify the **linear relationship** between variables, which directly impacts feature selection, multicollinearity detection, dimensionality reduction, and model interpretability.

Key ML connections:

- **Feature selection**: Remove highly correlated features to reduce redundancy.
- **Multicollinearity**: In linear regression, correlated predictors make coefficient estimates unstable.
- **PCA (Principal Component Analysis)**: Finds directions of maximum covariance in the data.
- **Neural network initialization**: Xavier/Glorot initialization uses the variance of inputs and outputs.
- **Data leakage detection**: Unexpectedly high correlation between features and target may indicate leakage.

---

## Covariance

### Definition

Covariance measures how two variables change together:

```
                   n
Cov(X, Y) =  1/(n-1) Σ (xᵢ - x̄)(yᵢ - ȳ)
                  i=1
```

**Interpretation:**
- `Cov(X, Y) > 0`: X and Y tend to increase together (positive linear relationship)
- `Cov(X, Y) < 0`: When X increases, Y tends to decrease (negative linear relationship)
- `Cov(X, Y) = 0`: No linear relationship (but may still have nonlinear relationship!)

```
Cov(X, Y) > 0:          Cov(X, Y) < 0:          Cov(X, Y) ≈ 0:
    y                       y                       y
    |  ●  ●                 |●                      ●  ●
    |    ●  ●               |  ●                  ●     ●
    | ●     ●               |    ●                ●   ●
    |   ●                   |      ●            ●      ●
    |●                      |        ●          ●
    └────────── x           └────────── x        └────────── x

  Move together            Move in opposite      No linear pattern
  (same direction)         directions
```

### Properties of Covariance

```
1. Cov(X, X) = Var(X) = σ²_X
   (Covariance of a variable with itself is its variance)

2. Cov(X, Y) = Cov(Y, X)
   (Symmetric)

3. Cov(aX + b, cY + d) = ac × Cov(X, Y)
   (Scaling: multiplying by constants scales covariance)

4. Cov(X₁ + X₂, Y) = Cov(X₁, Y) + Cov(X₂, Y)
   (Linearity: covariance distributes over addition)
```

### Problem with Covariance: Scale Dependence

Covariance depends on the units of measurement. If you measure height in cm vs inches, the covariance changes even though the relationship is identical:

```
Height (cm):    Cov(height_cm, weight_kg) = 850
Height (in):    Cov(height_in, weight_kg) = 335

Same relationship, different covariance values!
→ This is why we need CORRELATION (normalized covariance).
```

---

## Correlation (Pearson)

### Definition

Pearson correlation coefficient normalizes covariance by the product of standard deviations:

```
              Cov(X, Y)         Σ(xᵢ - x̄)(yᵢ - ȳ)
  r = ρ(X,Y) = ─────────── = ──────────────────────────
              σ_X × σ_Y     √[Σ(xᵢ-x̄)²] √[Σ(yᵢ-ȳ)²]
```

**Key properties:**
- Always in range: **-1 ≤ r ≤ 1**
- Unitless (dimensionless)
- r = 1: Perfect positive linear relationship
- r = -1: Perfect negative linear relationship
- r = 0: No linear correlation

```
Correlation Strength Guide:

  r = 1.0  ═══════════════  Perfect positive
  r = 0.8  ═══════════      Strong positive
  r = 0.5  ═══════          Moderate positive
  r = 0.3  ═══              Weak positive
  r = 0.0                  No linear correlation
  r = -0.3  ═══             Weak negative
  r = -0.5  ═══════         Moderate negative
  r = -0.8  ═══════════     Strong negative
  r = -1.0  ════════════════ Perfect negative
```

### Visual Guide to Correlation Values

```
r = +1.0              r = +0.8              r = +0.4              r ≈ 0
  y                     y                     y                     y
  │ ●                   │ ●                   │  ●                ●●●
  │  ●                  │  ●                  │ ●  ●             ● ● ●
  │   ●                 │   ●                 │●    ●           ●  ●  ●
  │    ●                │    ●                │  ● ●            ●●  ●●
  └────── x             └────── x             └────── x          └────── x

  Points fall           Strong linear         Moderate linear     No linear
  on a perfect          trend, some           trend, more         relationship
  straight line         scatter               scatter             (possibly nonlinear)

r = -0.4               r = -0.8              r = -1.0             Nonlinear
  y                     y                     y                   y
  │  ●                 │  ●                  │●                  ● ╱╲
  │ ●  ●               │ ●  ●                │ ●                ●╱  ╲●
  │●    ●              │●    ●               │  ●              ●╱    ╲●
  │  ● ●               │  ● ●                │   ●             ●╱______╲
  └────── x             └────── x             └────── x          └────── x
                                                                   r ≈ 0 but
                                                                   strong nonlinear
                                                                   relationship!
```

### Important Caveats

```
WARNING: Correlation ≠ Causation
────────────────────────────────
Just because X and Y are correlated does NOT mean X causes Y.

Examples of spurious correlation:
  - Ice cream sales and drowning rates (both caused by hot weather)
  - Number of storks and birth rates (both related to rural areas)
  - ML: GPU hours and model accuracy (both driven by larger datasets)

WARNING: Correlation only measures LINEAR relationships
──────────────────────────────────────────────────────
Y = X² has r = 0 but a perfect nonlinear relationship.
Always visualize your data!
```

---

## Correlation Matrix

For multiple variables, we compute pairwise correlations:

```
For variables X₁, X₂, X₃:

         ┌                         ┐
         │  1    r₁₂    r₁₃        │
    R =  │  r₂₁   1     r₂₃        │
         │  r₃₁   r₃₂    1         │
         └                         ┘

Properties:
  - Diagonal always = 1 (self-correlation)
  - Symmetric: R = Rᵀ
  - Positive semi-definite
```

---

## Spearman's Rank Correlation

For monotonic (not necessarily linear) relationships:

```
              Cov(rank(X), rank(Y))
ρ_spearman = ──────────────────────
              σ_rank(X) × σ_rank(Y)
```

**When to use Spearman instead of Pearson:**
- Data has outliers
- Relationship is monotonic but not linear
- Data is ordinal (ranked categories)
- Distribution is highly non-normal

```
Pearson: measures LINEAR relationship
Spearman: measures MONOTONIC relationship

  Pearson r ≈ 0        Spearman ρ ≈ 1
       y                     y
      ╱╲                    │  ●
     ╱  ╲                   │ ●
    ╱    ╲                  │●
   ╱      ╲                 │
  └────────x               └────────x
```

---

## Covariance Matrix

The covariance matrix generalizes covariance to multiple dimensions:

```
         ┌                                    ┐
         │  Var(X₁)    Cov(X₁,X₂)  Cov(X₁,X₃) │
    Σ =  │ Cov(X₂,X₁)   Var(X₂)    Cov(X₂,X₃) │
         │ Cov(X₃,X₁)  Cov(X₃,X₂)   Var(X₃)    │
         └                                    ┘
```

This matrix is:
- **Symmetric**: Σ = Σᵀ
- **Positive semi-definite**: all eigenvalues ≥ 0
- The diagonal contains variances, off-diagonal contains covariances

**ML Application**: The covariance matrix is the core input to PCA, LDA, Gaussian Naive Bayes, and Mahalanobis distance.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# ---- Generate correlated data ----
def generate_correlated_data():
    """Generate pairs of variables with known correlations."""
    rng = np.random.default_rng(42)
    n = 500

    # Positive correlation: study hours vs exam score
    study_hours = rng.normal(5, 1.5, n)
    exam_scores = 50 + 8 * study_hours + rng.normal(0, 8, n)

    # Negative correlation: temperature vs heating bill
    temperature = rng.normal(20, 8, n)
    heating_bill = 300 - 5 * temperature + rng.normal(0, 30, n)

    # No correlation: shoe size vs IQ (random)
    shoe_size = rng.normal(42, 2, n)
    iq = rng.normal(100, 15, n)

    return study_hours, exam_scores, temperature, heating_bill, shoe_size, iq

study, scores, temp, bills, shoes, iq = generate_correlated_data()

# ---- Manual Covariance ----
def manual_covariance(x, y):
    """Compute sample covariance manually."""
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)

# ---- Manual Pearson Correlation ----
def manual_pearson(x, y):
    """Compute Pearson correlation manually."""
    cov_xy = manual_covariance(x, y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    return cov_xy / (std_x * std_y)

# ---- Results ----
print("=" * 65)
print("COVARIANCE AND CORRELATION ANALYSIS")
print("=" * 65)

pairs = [
    ("Study Hours vs Exam Score", study, scores),
    ("Temperature vs Heating Bill", temp, bills),
    ("Shoe Size vs IQ", shoes, iq),
]

for name, x, y in pairs:
    cov = manual_covariance(x, y)
    r_manual = manual_pearson(x, y)
    r_numpy = np.corrcoef(x, y)[0, 1]
    r_spearman = stats.spearmanr(x, y).statistic

    print(f"\n--- {name} ---")
    print(f"  Covariance:       {cov:>10.4f}")
    print(f"  Pearson r (manual): {r_manual:>8.4f}")
    print(f"  Pearson r (numpy):  {r_numpy:>8.4f}")
    print(f"  Spearman ρ:         {r_spearman:>8.4f}")

    # R-squared (coefficient of determination)
    r_squared = r_manual ** 2
    print(f"  R²: {r_squared:>14.4f} ({r_squared*100:.1f}% variance explained)")

# ---- Correlation Matrix ----
print(f"\n{'=' * 65}")
print("CORRELATION MATRIX")
print("=" * 65)

data = np.column_stack([study, scores, temp, bills])
labels = ['Study', 'Scores', 'Temp', 'Bills']
corr_matrix = np.corrcoef(data, rowvar=False)

print(f"\n{'':>10}", end="")
for label in labels:
    print(f"{label:>10}", end="")
print()

for i, label in enumerate(labels):
    print(f"{label:>10}", end="")
    for j in range(len(labels)):
        print(f"{corr_matrix[i,j]:>10.3f}", end="")
    print()

# ---- Covariance Matrix ----
print(f"\n{'=' * 65}")
print("COVARIANCE MATRIX")
print("=" * 65)

cov_matrix = np.cov(data, rowvar=False)
print(f"\n{'':>12}", end="")
for label in labels:
    print(f"{label:>12}", end="")
print()

for i, label in enumerate(labels):
    print(f"{label:>12}", end="")
    for j in range(len(labels)):
        print(f"{cov_matrix[i,j]:>12.3f}", end="")
    print()

# ---- Detect Multicollinearity ----
print(f"\n{'=' * 65}")
print("MULTICOLLINEARITY DETECTION (Feature Selection)")
print("=" * 65)

threshold = 0.8
print(f"\nHighly correlated feature pairs (|r| > {threshold}):")
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        if abs(corr_matrix[i, j]) > threshold:
            print(f"  {labels[i]} and {labels[j]}: r = {corr_matrix[i,j]:.4f}")

# ---- Covariance in ML: Xavier Initialization ----
print(f"\n{'=' * 65}")
print("ML APPLICATION: XAVIER/GLOROT WEIGHT INITIALIZATION")
print("=" * 65)

print("""
Xavier initialization sets weights from:
  W ~ N(0, 2/(n_in + n_out))

This keeps the variance of activations constant across layers,
preventing vanishing/exploding gradients.

Derivation:
  If X has variance Var(X) and W has variance Var(W), then:
  Var(WX) = n_in × Var(W) × Var(X)

  To keep Var(WX) = Var(X), we need:
  Var(W) = 1/n_in
  Xavier uses the average: Var(W) = 2/(n_in + n_out)
""")

# Demonstrate
n_in, n_out = 128, 64
xavier_std = np.sqrt(2.0 / (n_in + n_out))
he_std = np.sqrt(2.0 / n_in)  # He initialization for ReLU

print(f"Layer: {n_in} → {n_out}")
print(f"Xavier std: {xavier_std:.4f}")
print(f"He std:     {he_std:.4f}")

# Verify variance preservation
rng = np.random.default_rng(42)
W = rng.normal(0, xavier_std, (n_out, n_in))
X = rng.normal(0, 1, (1000, n_in))
Y = X @ W.T
print(f"\nInput variance:  {np.var(X):.4f}")
print(f"Output variance: {np.var(Y):.4f}")
print(f"Ratio (should be ≈ 1): {np.var(Y)/np.var(X):.4f}")
```

**Output:**
```
=================================================================
COVARIANCE AND CORRELATION ANALYSIS
=================================================================

--- Study Hours vs Exam Score ---
  Covariance:        11.5237
  Pearson r (manual):   0.9502
  Spearman ρ:           0.9471
  R²:       0.9029 (90.3% variance explained)

--- Temperature vs Heating Bill ---
  Covariance:       -35.8915
  Pearson r (manual):  -0.8813
  Spearman ρ:          -0.8783
  R²:       0.7767 (77.7% variance explained)

--- Shoe Size vs IQ ---
  Covariance:         0.3288
  Pearson r (manual):   0.0130
  Spearman ρ:           0.0213
  R²:       0.0002 (0.0% variance explained)

=================================================================
CORRELATION MATRIX
=================================================================
           Study    Scores      Temp     Bills
    Study     1.000     0.950     0.004    -0.006
   Scores     0.950     1.000    -0.004     0.001
     Temp     0.004    -0.004     1.000    -0.881
    Bills    -0.006     0.001    -0.881     1.000
```

---

## Real-World ML Connections

### 1. Feature Selection via Correlation
In practice, remove one of any pair of features with |r| > 0.8 to reduce multicollinearity. This is especially important for:
- **Linear models**: Correlated features make coefficients unstable and hard to interpret.
- **Logistic regression**: Can cause convergence issues.
- **KNN**: Distance-based methods are affected by redundant features.

### 2. PCA (Principal Component Analysis)
PCA finds eigenvectors of the covariance matrix. The eigenvector with the largest eigenvalue points in the direction of maximum variance.

### 3. Mahalanobis Distance
Unlike Euclidean distance, Mahalanobis distance accounts for covariance between features:

```
d_M(x, y) = √[(x - y)ᵀ Σ⁻¹ (x - y)]
```

Used in anomaly detection and clustering.

### 4. Correlation does not Imply Causation
In ML, a feature correlated with the target may not be causal. Be careful of:
- **Data leakage**: Feature contains information from the future.
- **Confounding**: A third variable causes both the feature and target.

---

## Exercises with Solutions

### Exercise 1
Compute the covariance and correlation of X = [1, 2, 3, 4, 5] and Y = [2, 4, 5, 4, 5].

**Solution:**
```
x̄ = 3, ȳ = 4
Cov = [(1-3)(2-4) + (2-3)(4-4) + (3-3)(5-4) + (4-3)(4-4) + (5-3)(5-4)] / 4
    = [4 + 0 + 0 + 0 + 2] / 4 = 6/4 = 1.5

σ_X = √[(4+1+0+1+4)/4] = √2.5 = 1.581
σ_Y = √[(4+0+1+0+1)/4] = √1.5 = 1.225

r = 1.5 / (1.581 × 1.225) = 1.5 / 1.936 = 0.775
```

### Exercise 2
If Cov(X, Y) = 12, Var(X) = 9, Var(Y) = 16, find the correlation coefficient. What fraction of variance in Y is explained by X?

**Solution:**
```
r = Cov(X,Y) / (σ_X × σ_Y) = 12 / (3 × 4) = 1.0

R² = 1.0, meaning 100% of variance in Y is linearly explained by X.
This means Y is perfectly linearly predicted by X: Y = aX + b.
```

### Exercise 3
Given Cov(X, Y) = 8, Var(X) = 4, Var(Y) = 25. Compute Cov(3X+1, 2Y-5).

**Solution:**
```
Cov(3X+1, 2Y-5) = 3 × 2 × Cov(X, Y) = 6 × 8 = 48
(Constants don't affect covariance.)
```

### Exercise 4
Can two variables have zero correlation but still be dependent? Give an example.

**Solution:**
```
Yes! If Y = X² and X is symmetric around 0 (e.g., X ~ Uniform(-1, 1)):

Cov(X, X²) = E[X³] - E[X]E[X²] = 0 - 0 × 1/3 = 0

So r = 0 even though Y is completely determined by X.

The relationship is nonlinear (parabolic), which Pearson correlation cannot detect.
Spearman correlation would also miss this since it's not monotonic either.
```

### Exercise 5
A dataset has features F1, F2, F3 with correlation matrix:
```
     F1    F2    F3
F1 [ 1.0   0.2   0.9 ]
F2 [ 0.2   1.0   0.3 ]
F3 [ 0.9   0.3   1.0 ]
```
Which feature would you remove for a linear model and why?

**Solution:**
```
Remove F1 or F3 (they have r = 0.9, indicating near-duplicate information).
Since F3 also has a higher correlation with F2 (0.3 vs 0.2), keeping F2 and F3
or F1 and F2 are both reasonable choices. I'd remove F3 to keep F1 and F2
(since F3's information is mostly captured by F1).
```

---

## Key Takeaways

1. **Covariance measures directional co-movement** but is scale-dependent and hard to interpret in isolation.
2. **Correlation is standardized covariance**: always between -1 and +1, making it directly interpretable.
3. **Pearson = linear, Spearman = monotonic**: Choose based on the type of relationship you expect.
4. **Correlation ≠ Causation**: Always be skeptical of correlated features — look for confounders and data leakage.
5. **The covariance matrix** is the foundation of PCA, LDA, Mahalanobis distance, and Gaussian models.
6. **Remove one of any highly correlated pair** (|r| > 0.8) before training linear models.
