# Mean, Median, Mode, Variance & Standard Deviation

## Why This Matters in AI/ML

Descriptive statistics are the **first lens** through which we examine data. Before building any machine learning model, understanding the central tendency and spread of your features and targets is essential. These statistics directly impact model performance, preprocessing decisions, and how we interpret results.

Key ML connections:

- **Mean centering**: Required for PCA, standardization (z-scores), and many distance-based algorithms.
- **Variance/std dev**: The basis for regularization (minimize variance of weights), loss functions (MSE = variance of residuals), and understanding model stability.
- **Median robustness**: Used in outlier-resistant preprocessing and robust loss functions (MAE vs MSE).
- **Mode**: Relevant for categorical features, imbalanced datasets, and understanding dominant classes.
- **Normalization**: Z-score normalization uses mean and std dev; robust scaling uses median and IQR.

---

## Measures of Central Tendency

### 1. Mean (Arithmetic Average)

The mean is the sum of all values divided by the count:

```
          1    n
   μ =   ─  Σ  xᵢ
          n   i=1

Sample mean:
          1    n
   x̄ =   ─  Σ  xᵢ
          n   i=1
```

**Properties:**
- Uses every data point
- Sensitive to outliers
- Minimizes sum of squared deviations: Σ(xᵢ - x̄)² is minimized at x̄
- Algebraic: Σ(xᵢ - x̄) = 0 (deviations always sum to zero)

```
Data: [2, 4, 4, 4, 5, 5, 7, 9]
Mean = (2+4+4+4+5+5+7+9) / 8 = 40/8 = 5.0

  1   2   3   4   5   6   7   8   9
  ─●───●───●───●───▲───●───●───●──
                  mean

  With outlier: [2, 4, 4, 4, 5, 5, 7, 9, 100]
  Mean = 140/9 = 15.6  (pulled far right by outlier)
  ─●───●───●───●───▲────────────────●──
                           mean (skewed)
```

### 2. Weighted Mean

```
         Σ wᵢxᵢ
x̄_w = ────────
         Σ wᵢ
```

**ML Application**: In gradient descent with weighted loss functions, or computing macro/micro F1 scores.

### 3. Geometric Mean

```
x̄_g = (Π xᵢ)^(1/n) = exp((1/n) × Σ ln(xᵢ))
```

Used for growth rates and ratios. **Never use with negative or zero values.**

### 4. Median

The middle value when data is sorted. For n values:
- If n is odd: median = x_((n+1)/2)
- If n is even: median = (x_(n/2) + x_(n/2+1)) / 2

```
Data: [2, 4, 4, 4, 5, 5, 7, 9]
Sorted: 2, 4, 4, 4, |5|, 5, 7, 9
                    ▲
                Median = 4.5 (average of 4th and 5th values)

With outlier: [2, 4, 4, 4, 5, 5, 7, 9, 100]
Sorted: 2, 4, 4, 4, |5|, 5, 7, 9, 100
                    ▲
                Median = 5 (barely affected!)
```

**Why median is robust**: Changing the extreme values doesn't affect the middle position. This makes it ideal for datasets with outliers.

### 5. Mode

The most frequently occurring value(s).

```
Data: [2, 4, 4, 4, 5, 5, 7, 9]
          ▲ ▲ ▲
          mode = 4 (appears 3 times)

A dataset can be:
  Unimodal:  [1, 2, 3, 3, 3, 4, 5]     → one mode
  Bimodal:   [1, 1, 2, 3, 4, 4, 4, 5, 5, 5] → two modes
  Multimodal: [1,1, 2,2, 3,3]            → three modes
  No mode:    [1, 2, 3, 4, 5]            → all appear once
```

---

## Measures of Spread (Dispersion)

### 1. Range

```
Range = max(x) - min(x)
```

Simple but extremely sensitive to outliers.

### 2. Interquartile Range (IQR)

```
IQR = Q3 - Q1

Where:
  Q1 = 25th percentile (median of lower half)
  Q2 = 50th percentile (median)
  Q3 = 75th percentile (median of upper half)
```

```
Box Plot:
    ┌─────┬───┬────────────┬─────┬───┐
    │     │   │            │     │   │
  ──┼─────┼───┼────────────┼─────┼───┼──
    │   Q1   │     Q2      │  Q3   │
    │        │   (median)  │       │
    └───IQR──┘             └───────┘

  Whiskers typically extend to 1.5 × IQR
  Points beyond whiskers are outliers:
      ○        ○
       ────────┤─────────────────────
              Q1 Q2    Q3
```

### 3. Variance

Variance measures the average squared deviation from the mean:

**Population variance:**

```
         1    n
  σ² =   ─  Σ (xᵢ - μ)²
         N   i=1
```

**Sample variance (unbiased estimator):**

```
         1    n
  s² =   ─  Σ (xᵢ - x̄)²
         n-1  i=1
```

**Why n-1? (Bessel's Correction)**
The sample mean x̄ is closer to the data points than the true mean μ, so squared deviations around x̄ systematically underestimate the true variance. Dividing by n-1 corrects for this bias.

```
Visual: Deviations from mean

    x₁   x₂   x₃   x₄   x₅
    │    │    │    │    │
  ──●────●────●────●────●──── x̄
    │←d₁→│←d₂→│←d₃→│←d₄→│←d₅→│

  Variance = average of (d₁² + d₂² + d₃² + d₄² + d₅²)

  Large spread → large variance
  Small spread → small variance
```

### 4. Standard Deviation

```
σ = √σ²    (population std dev)
s = √s²    (sample std dev)
```

Standard deviation is in the **same units** as the original data, making it more interpretable than variance.

### Computational Formula (faster for hand calculation)

```
         n    2       2
s² = ──── (Σxᵢ ) - (Σxᵢ)
       n-1     n       n
```

---

## Mean vs Median: Skewness Implications

```
Right-skewed (positive skew):     Left-skewed (negative skew):      Symmetric:
    Mode  Median  Mean                Mean  Median  Mode           Mean = Median = Mode
     │     │     │←                  →│     │     │                   │
     ▼     ▼     ▼                   ▼     ▼     ▼                   ▼
  ───╱──────╱────────────           ─────────────╲──────╲──        ────╱╲───
  ╱                                                ╲                  ╱  ╲
 ╱                                                  ╲                ╱    ╲

  Examples: Income, house prices      Examples: Exam scores        Example: Heights,
  Mean > Median > Mode                (easy exam)                   Normal distribution

  ML Impact:                           ML Impact:                    ML Impact:
  Use median for                      Use median for                Mean is reliable
  preprocessing                      preprocessing
```

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# ---- Generate Realistic Data ----
def generate_data():
    """Generate ML-relevant dataset for demonstration."""
    rng = np.random.default_rng(42)

    # Simulate house prices (right-skewed, with outliers)
    prices = rng.lognormal(mean=12, sigma=0.5, size=1000)  # skewed
    # Add some outliers
    prices = np.append(prices, [5_000_000, 8_000_000, 12_000_000])

    # Simulate model accuracy scores (roughly normal)
    accuracies = rng.normal(loc=0.85, scale=0.05, size=200)

    # Simulate training loss over epochs
    epochs = np.arange(1, 101)
    loss = 2.0 * np.exp(-0.03 * epochs) + rng.normal(0, 0.01, size=100)
    loss = np.clip(loss, 0, None)

    return prices, accuracies, loss

prices, accuracies, loss = generate_data()

# ---- 1. Central Tendency ----
print("=" * 60)
print("MEASURES OF CENTRAL TENDENCY")
print("=" * 60)

print(f"\n--- House Prices ---")
print(f"Mean:    ${np.mean(prices):>15,.2f}")
print(f"Median:  ${np.median(prices):>15,.2f}")
mode_result = stats.mode(prices, keepdims=True)
print(f"Mode:    ${mode_result.mode[0]:>15,.2f}")
print(f"Geometric Mean: ${stats.gmean(prices):>13,.2f}")
print(f"\nMean/Median ratio: {np.mean(prices)/np.median(prices):.2f}")
print(f"(Ratio > 1 indicates RIGHT skew — outliers pull mean up)")

print(f"\n--- Model Accuracies ---")
print(f"Mean:    {np.mean(accuracies):.4f}")
print(f"Median:  {np.median(accuracies):.4f}")
print(f"Mode:    {stats.mode(np.round(accuracies, 2), keepdims=True).mode[0]:.4f}")

# ---- 2. Measures of Spread ----
print(f"\n{'=' * 60}")
print("MEASURES OF SPREAD")
print("=" * 60)

print(f"\n--- House Prices ---")
print(f"Range:      ${np.ptp(prices):>14,.2f}")
q1, q3 = np.percentile(prices, [25, 75])
iqr = q3 - q1
print(f"IQR:        ${iqr:>14,.2f}")
print(f"  Q1 (25%): ${q1:>14,.2f}")
print(f"  Q3 (75%): ${q3:>14,.2f}")
print(f"Variance:   ${np.var(prices):>14,.2f}")
print(f"Std Dev:    ${np.std(prices):>14,.2f}")
print(f"Sample Var: ${np.var(prices, ddof=1):>14,.2f}")
print(f"Sample SD:  ${np.std(prices, ddof=1):>14,.2f}")

# Detect outliers using IQR method
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
print(f"\nOutliers (1.5×IQR): {len(outliers)} points")
print(f"  Lower bound: ${lower_bound:,.2f}")
print(f"  Upper bound: ${upper_bound:,.2f}")

# ---- 3. Mean vs Median for Outlier Detection ----
print(f"\n{'=' * 60}")
print("MEAN vs MEDIAN: OUTLIER IMPACT")
print("=" * 60)

# Remove outliers and compare
clean_prices = prices[(prices >= lower_bound) & (prices <= upper_bound)]
print(f"\nWith outliers ({len(prices)} points):")
print(f"  Mean:   ${np.mean(prices):>12,.2f}")
print(f"  Median: ${np.median(prices):>12,.2f}")
print(f"  Std:    ${np.std(prices):>12,.2f}")

print(f"\nWithout outliers ({len(clean_prices)} points):")
print(f"  Mean:   ${np.mean(clean_prices):>12,.2f}")
print(f"  Median: ${np.median(clean_prices):>12,.2f}")
print(f"  Std:    ${np.std(clean_prices):>12,.2f}")

# ---- 4. Z-Score Normalization (ML Preprocessing) ----
print(f"\n{'=' * 60}")
print("Z-SCORE NORMALIZATION (ML PREPROCESSING)")
print("=" * 60)

# Raw features
raw = np.array([150, 160, 170, 180, 190, 200, 210, 220, 230, 240])

# Z-score normalization
z_scores = (raw - np.mean(raw)) / np.std(raw)

print(f"\nRaw data:     {raw}")
print(f"Mean: {np.mean(raw):.1f}, Std: {np.std(raw):.1f}")
print(f"Z-scores:     {np.round(z_scores, 2)}")
print(f"Z mean: {np.mean(z_scores):.6f}, Z std: {np.std(z_scores):.6f}")

# ---- 5. Robust Scaling (using median and IQR) ----
print(f"\n--- Robust Scaling (Median/IQR) ---")
robust_scaled = (raw - np.median(raw)) / iqr if iqr > 0 else raw
Q1_raw, Q3_raw = np.percentile(raw, [25, 75])
iqr_raw = Q3_raw - Q1_raw
robust_scaled = (raw - np.median(raw)) / iqr_raw
print(f"Robust scaled: {np.round(robust_scaled, 2)}")
print(f"Robust mean: {np.mean(robust_scaled):.6f}")
print(f"Robust median: {np.median(robust_scaled):.6f}")

# ---- 6. Coefficient of Variation ----
print(f"\n--- Coefficient of Variation (CV) ---")
cv_prices = np.std(prices) / np.mean(prices)
cv_acc = np.std(accuracies) / np.mean(accuracies)
print(f"CV (prices):    {cv_prices:.4f} (high relative variability)")
print(f"CV (accuracies): {cv_acc:.4f} (low relative variability)")

# ---- 7. Manual Computation ----
print(f"\n{'=' * 60}")
print("MANUAL VARIANCE COMPUTATION (Step-by-step)")
print("=" * 60)

data = np.array([4, 8, 6, 5, 3, 2, 8, 9, 2, 5])
n = len(data)
mean = np.mean(data)
deviations = data - mean
sq_devs = deviations ** 2

print(f"\nData: {data}")
print(f"n = {n}, Mean = {mean:.1f}\n")
print(f"{'xᵢ':>4} {'(xᵢ - x̄)':>10} {'(xᵢ - x̄)²':>12}")
print("-" * 28)
for x, d, sd in zip(data, deviations, sq_devs):
    print(f"{x:>4.0f} {d:>10.1f} {sd:>12.1f}")

print("-" * 28)
print(f"{'Sum':>4} {np.sum(deviations):>10.1f} {np.sum(sq_devs):>12.1f}")
print(f"\nPopulation variance = {np.sum(sq_devs)/n:.2f}")
print(f"Sample variance     = {np.sum(sq_devs)/(n-1):.2f}")
print(f"Std deviation       = {np.sqrt(np.sum(sq_devs)/(n-1)):.2f}")
```

---

## Real-World ML Connections

### 1. Feature Scaling
- **StandardScaler** (sklearn): Uses mean and std dev. `(x - mean) / std`
- **RobustScaler** (sklearn): Uses median and IQR. `(x - median) / IQR`
- **MinMaxScaler**: Uses min and max. `(x - min) / (max - min)`

Choose StandardScaler for approximately normal features, RobustScaler for features with outliers.

### 2. Loss Functions
- **MSE (Mean Squared Error)**: Minimizes the variance of residuals. Sensitive to outliers (squared penalty).
- **MAE (Mean Absolute Error)**: Minimizes around the median. Robust to outliers.
- **Huber Loss**: Combines MSE and MAE — quadratic near zero, linear for large errors.

### 3. Batch Normalization
Normalizes layer inputs using batch mean and variance:
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γ × x̂ + β
```
This stabilizes training and allows higher learning rates.

### 4. Explaining Variance (PCA)
PCA finds directions of maximum variance in data. The variance explained by each principal component determines how many components to keep. Typically we keep components explaining 95%+ of total variance.

### 5. Bias-Variance Tradeoff
```
Total Error = Bias² + Variance + Irreducible Noise

High bias  → underfitting (model too simple)
High variance → overfitting (model too complex, memorizes noise)

Goal: Minimize total error by balancing bias and variance.
```

---

## Exercises with Solutions

### Exercise 1
Compute the mean, median, mode, variance, and standard deviation of: [3, 7, 2, 9, 4, 7, 8, 7, 1, 5]

**Solution:**
```
Sorted: [1, 2, 3, 4, 5, 7, 7, 7, 8, 9]
Mean = 53/10 = 5.3
Median = (5 + 7)/2 = 6.0
Mode = 7 (appears 3 times)

Deviations from mean (5.3):
[-2.3, 1.7, -3.3, 3.7, -1.3, 1.7, 2.7, 1.7, -4.3, -0.3]
Squared: [5.29, 2.89, 10.89, 13.69, 1.69, 2.89, 7.29, 2.89, 18.49, 0.09]
Sum = 66.1

Sample variance = 66.1/9 = 7.344
Std dev = √7.344 = 2.71
```

### Exercise 2
Why is the sample variance divided by (n-1) instead of n? Show what happens with a tiny sample.

**Solution:**
```
With n=2 data points: x₁=3, x₂=7
True mean = 5, but we don't know it.
Sample mean x̄ = 5 (happens to match)

Using n:    s² = [(3-5)² + (7-5)²]/2 = 8/2 = 4
Using n-1:  s² = [(3-5)² + (7-5)²]/1 = 8/1 = 8

The true variance (known to us) = [(3-5)² + (7-5)²]/2 = 4.
But the sample variance using n systematically underestimates.
With n-1 (Bessel's correction), we get an unbiased estimator
ON AVERAGE across all possible samples.
```

### Exercise 3
A model has training accuracies: [0.92, 0.89, 0.95, 0.88, 0.91, 0.94, 0.87, 0.90] across 8 folds. Compute the mean ± std dev and identify if any fold is an outlier.

**Solution:**
```
Mean = 7.26/8 = 0.9075
Sorted: [0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.94, 0.95]
Q1 = 0.885, Q3 = 0.93
IQR = 0.045
Lower = 0.885 - 0.0675 = 0.8175
Upper = 0.93 + 0.0675 = 0.9975
No outliers — all values within bounds.

Std dev ≈ 0.027
Report: 90.75% ± 2.7% accuracy
```

### Exercise 4
Show that the mean minimizes the sum of squared deviations.

**Solution:**
```
Let f(c) = Σ(xᵢ - c)²
df/dc = -2 Σ(xᵢ - c) = 0
→ Σ(xᵢ - c) = 0
→ Σxᵢ = nc
→ c = (1/n)Σxᵢ = x̄

d²f/dc² = 2n > 0, so this is a minimum.

Similarly, the median minimizes the sum of absolute deviations:
g(c) = Σ|xᵢ - c| is minimized at c = median(x).
```

### Exercise 5
Compare z-score normalization and robust scaling on data with an outlier: [10, 12, 11, 13, 12, 100]

**Solution:**
```
Z-score:
  mean = 26.33, std = 34.88
  z(10) = -0.47, z(100) = 2.12, z(12) = -0.41

Robust:
  median = 12, Q1=11, Q3=12.5, IQR=1.5
  r(10) = -1.33, r(100) = 58.67, r(12) = 0.0

Z-score spreads all points including outliers.
Robust scaling keeps most data near 0 while clearly flagging outliers.
```

---

## Key Takeaways

1. **Mean is great for symmetric data** but gets destroyed by outliers. Always compare mean and median.
2. **Median is your outlier-resistant friend**. Use it for skewed distributions (income, prices, reaction times).
3. **Variance and std dev measure spread** — high variance means data is spread out, which impacts model generalization.
4. **Use n-1 for sample variance** (Bessel's correction) to get an unbiased estimator.
5. **In ML preprocessing**: StandardScaler (mean/std) for normal data, RobustScaler (median/IQR) for data with outliers.
6. **The bias-variance tradeoff** is the central challenge of ML — it's fundamentally about managing variance.
