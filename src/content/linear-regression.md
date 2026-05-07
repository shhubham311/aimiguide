# Linear Regression & Ridge Regression

## Why This Topic Matters

Linear regression is arguably the most important algorithm in all of machine learning—not because it is the most powerful, but because it is the most **fundamental**. It is the first model you should try on any regression problem. It is the basis for understanding gradient descent, regularization, cost functions, and the bias-variance tradeoff. Even advanced models like neural networks are, at their core, chains of linear transformations with nonlinear activations. If you understand linear regression deeply, you have a foundation for everything else.

Ridge regression extends linear regression with **L2 regularization**, introducing the critical concept of penalizing model complexity to prevent overfitting. This idea—adding a penalty term to the loss function—is reused in Lasso, Elastic Net, SVMs, and neural networks. Mastering Ridge is your gateway to understanding modern regularization techniques.

---

## Simple Linear Regression

### The Model

Simple linear regression models the relationship between one input feature `x` and one continuous target `y`:

```
ŷ = w₀ + w₁·x
```

where:
- `ŷ` = predicted value
- `w₀` = intercept (bias term) — the predicted value when `x = 0`
- `w₁` = slope (weight/coefficient) — how much `y` changes per unit change in `x`
- `x` = input feature

### Visual: The Line of Best Fit

```
y (Price)
│         *
│      *     ●          *
│   ●           ●    ŷ = w₀ + w₁·x
│●        ●
│     ●          ●        (line of best fit)
│  ●                ●
│
└──────────────────────────── x (Square Footage)
```

The goal is to find the line that **minimizes the total squared error** between predictions `ŷ` and actual values `y`.

---

## Multiple Linear Regression

When you have `p` features, the model extends naturally:

```
ŷ = w₀ + w₁·x₁ + w₂·x₂ + ... + wₚ·xₚ
```

In matrix form, for `n` samples with `p` features:

```
ŷ = Xw

where:
    ŷ = [ŷ₁, ŷ₂, ..., ŷₙ]ᵀ      (n × 1 vector of predictions)
    X = [1, x₁₁, x₁₂, ..., x₁ₚ]  (n × p+1 design matrix, with column of 1s for intercept)
        [1, x₂₁, x₂₂, ..., x₂ₚ]
        [      ...              ]
        [1, xₙ₁, xₙ₂, ..., xₙₚ]
    w = [w₀, w₁, w₂, ..., wₚ]ᵀ  (p+1 × 1 vector of coefficients)
```

---

## The Cost Function: Ordinary Least Squares (OLS)

The objective of linear regression is to minimize the **Residual Sum of Squares (RSS)**, also called the **cost function** `J(w)`:

```
        n
J(w) = Σ (yᵢ - ŷᵢ)² = Σ (yᵢ - xᵢᵀw)²
       i=1
```

Or equivalently, using the **Mean Squared Error (MSE)**:

```
J(w) = (1/n) · Σ (yᵢ - ŷᵢ)² = (1/n) · ||y - Xw||²
```

where:
- `yᵢ` = actual target value for sample `i`
- `ŷᵢ` = predicted value for sample `i`
- `xᵢᵀw` = dot product of feature vector and weights for sample `i`
- `||·||²` = L2 norm squared (sum of squared elements)
- `n` = number of training samples

### The Closed-Form Solution

Setting the derivative of `J(w)` with respect to `w` to zero yields the **Normal Equation**:

```
∂J/∂w = -2Xᵀ(y - Xw) = 0

Solving:
    w = (XᵀX)⁻¹Xᵀy
```

where:
- `XᵀX` = (p+1 × p+1) matrix of feature cross-products
- `(XᵀX)⁻¹` = inverse of `XᵀX` (must be invertible — requires no perfect multicollinearity)
- `Xᵀy` = (p+1 × 1) vector of feature-target cross-products

**Limitations of the Normal Equation:**
- Computational cost: `O(p³)` for matrix inversion (expensive for many features)
- Requires `XᵀX` to be invertible (fails with correlated features or `p > n`)
- That's why we use gradient descent or regularization in practice

---

## Assumptions of Linear Regression

```
┌──────────────────────────────────────────────────────────────┐
│  LINEARITY:      Relationship between X and y is linear      │
│  INDEPENDENCE:   Errors are independent of each other        │
│  HOMOSCEDASTICITY: Constant variance of errors               │
│  NORMALITY:      Errors are normally distributed (for stats) │
│  NO MULTICOLLINEARITY: Features are not highly correlated    │
└──────────────────────────────────────────────────────────────┘

   Violating these assumptions doesn't necessarily make the model
   useless, but it affects the reliability of predictions and the
   validity of statistical inference (p-values, confidence intervals).
```

---

## Ridge Regression (L2 Regularization)

### The Problem with OLS

When features are highly correlated (multicollinearity) or when `p > n` (more features than samples), the OLS solution becomes unstable: small changes in data cause large changes in coefficients. This is **overfitting**.

### The Solution: Add a Penalty

Ridge regression adds an L2 penalty to the cost function:

```
J_ridge(w) = ||y - Xw||² + α||w||²
             ─────────────   ───────
               Data fit        Complexity penalty

where:
    α = regularization strength (hyperparameter, α ≥ 0)
    ||w||² = w₁² + w₂² + ... + wₚ²  (excludes intercept w₀)
```

- When `α = 0`: Ridge becomes OLS (no regularization).
- When `α → ∞`: All weights shrink to zero (underfitting).
- **Optimal α** balances bias and variance.

### The Ridge Solution

```
w_ridge = (XᵀX + αI)⁻¹Xᵀy

where:
    I = identity matrix (p+1 × p+1)
    αI is added to the diagonal, making XᵀX + αI always invertible!
```

### Why Ridge Works: Visual Intuition

```
Coefficient Values (without regularization):
    w₁ = 45000    (very large!)
    w₂ = -38000   (very large!)
    w₃ = 0.5

Coefficient Values (with Ridge, α=10):
    w₁ = 120      (shrunk toward zero)
    w₂ = -95      (shrunk toward zero)
    w₃ = 0.4

The L2 penalty "pulls" all coefficients toward zero,
reducing model variance at the cost of slightly more bias.
```

### Regularization Path: How Coefficients Shrink

```
w₁ ─────╲
        ╲
w₂      ╲─────╲
               ╲
w₃              ╲──────────
                           ╲
w₄                        ╲────────────
                                        ╲
w₅                                     ╲───────
    0    10    100   1000   10000  → α (log scale)

As α increases, all coefficients shrink smoothly toward 0.
Contrast with Lasso: some coefficients hit exactly 0 (sparsity).
```

---

## Python Implementation: California Housing Dataset

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the California Housing dataset (20,640 houses, 8 features)
housing = fetch_california_housing()
X = housing.data
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {housing.feature_names}")
print(f"Target: Median house value (in $100,000s)")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 1: Scale features (critical for regularized models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODEL 1: Ordinary Linear Regression (OLS)
# ============================================================
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)
y_pred_ols = ols.predict(X_test_scaled)

print("\n" + "=" * 50)
print("ORDINARY LINEAR REGRESSION (OLS)")
print("=" * 50)
print(f"MSE:  {mean_squared_error(y_test, y_pred_ols):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_ols):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_ols):.4f}")
print(f"\nCoefficients:")
for name, coef in zip(housing.feature_names, ols.coef_):
    print(f"  {name:>15s}: {coef:+.4f}")

# ============================================================
# MODEL 2: Ridge Regression
# ============================================================
# Find the best alpha using cross-validation
from sklearn.linear_model import RidgeCV

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
best_alpha = ridge_cv.alpha_

ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("\n" + "=" * 50)
print(f"RIDGE REGRESSION (alpha={best_alpha})")
print("=" * 50)
print(f"MSE:  {mean_squared_error(y_test, y_pred_ridge):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_ridge):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_ridge):.4f}")
print(f"\nCoefficients (Ridge vs OLS):")
for name, ols_c, ridge_c in zip(housing.feature_names, ols.coef_, ridge.coef_):
    print(f"  {name:>15s}: OLS={ols_c:+.4f}  Ridge={ridge_c:+.4f}")

# ============================================================
# COMPARISON: Cross-validation scores
# ============================================================
ols_scores = cross_val_score(LinearRegression(), X_train_scaled, y_train,
                              cv=5, scoring='r2')
ridge_scores = cross_val_score(Ridge(alpha=best_alpha), X_train_scaled, y_train,
                                cv=5, scoring='r2')

print("\n" + "=" * 50)
print("5-FOLD CROSS-VALIDATION (R²)")
print("=" * 50)
print(f"OLS:   Mean={ols_scores.mean():.4f}, Std={ols_scores.std():.4f}")
print(f"Ridge: Mean={ridge_scores.mean():.4f}, Std={ridge_scores.std():.4f}")
```

**Expected Output:**
```
Dataset shape: (20640, 8)
Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

==================================================
ORDINARY LINEAR REGRESSION (OLS)
==================================================
MSE:  0.5559
RMSE: 0.7456
MAE:  0.5332
R²:   0.5757

Coefficients:
       MedInc: +0.8296
      HouseAge: +0.1188
     AveRooms: -0.2655
    AveBedrms: +0.3057
   Population: -0.0047
     AveOccup: -0.0395
     Latitude: -0.9001
    Longitude: -0.8709

==================================================
RIDGE REGRESSION (alpha=1.0)
==================================================
MSE:  0.5559
RMSE: 0.7456
MAE:  0.5333
R²:   0.5757

==================================================
5-FOLD CROSS-VALIDATION (R²)
==================================================
OLS:   Mean=0.5926, Std=0.0254
Ridge: Mean=0.5927, Std=0.0244
```

Note: On this well-conditioned dataset, Ridge and OLS perform similarly. Ridge shines when there is multicollinearity or when `p` is large relative to `n`.

---

## Gradient Descent: The Algorithmic Alternative

Instead of computing the closed-form solution, we can iteratively update weights:

```
Initialize: w = 0 (or random small values)
Repeat until convergence:
    w ← w - η · ∂J/∂w

For OLS:
    ∂J/∂w = -(2/n) · Xᵀ(y - Xw)

For Ridge:
    ∂J/∂w = -(2/n) · Xᵀ(y - Xw) + (2α/n) · w
```

where `η` (eta) is the **learning rate** — a hyperparameter controlling step size.

### Visual: Gradient Descent Convergence

```
    Cost J(w)
        │
    ●───┐
    │    ╲
    │     ╲  ●
    │      ╲╱ ╲   ← finding minimum
    │       ╲   ╲
    │        ╲    ╲●
    │         ╲    ╱
    │          ●──╱
    │             ●  ← convergence at global minimum
    └──────────────────── w

    Learning rate too small:   Slow convergence (many iterations)
    Learning rate too large:   Overshooting / divergence
    Learning rate just right:  Efficient convergence
```

---

## Exercises

### Exercise 1: Implement Linear Regression from Scratch

```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n, p = X.shape
        self.w = np.zeros(p)
        self.b = 0

        for i in range(self.n_iter):
            # Forward pass
            y_pred = X @ self.w + self.b

            # Compute gradients
            dw = -(2/n) * X.T @ (y - y_pred)
            db = -(2/n) * np.sum(y - y_pred)

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 200 == 0:
                mse = np.mean((y - y_pred) ** 2)
                print(f"Iter {i:4d}: MSE = {mse:.6f}")

    def predict(self, X):
        return X @ self.w + self.b

# Test against sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=500, n_features=3, noise=10, random_state=42)

# Our implementation
model_scratch = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model_scratch.fit(X, y)

# Sklearn implementation
model_sk = LinearRegression()
model_sk.fit(X, y)

print(f"\nOur weights:       {model_scratch.w.round(4)}")
print(f"Sklearn weights:   {model_sk.coef_.round(4)}")
print(f"Our intercept:     {model_scratch.b:.4f}")
print(f"Sklearn intercept: {model_sk.intercept_:.4f}")
```

### Exercise 2: Ridge Regularization Strength Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)

# Sweep alpha values on log scale
alphas = np.logspace(-3, 6, 50)
cv_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

best_idx = np.argmax(cv_scores)
print(f"Best alpha: {alphas[best_idx]:.4f}")
print(f"Best R²: {cv_scores[best_idx]:.4f}")
```

### Solution Notes

The curve of `R²` vs. `log(alpha)` is typically inverted-U shaped:
- Too small `α`: OLS-like, potentially overfitting
- Too large `α`: all coefficients near zero, underfitting
- Sweet spot: optimal bias-variance tradeoff

---

## Key Takeaways

1. **Linear regression** models the relationship `y = Xw` by minimizing squared error.
2. The **Normal Equation** `w = (XᵀX)⁻¹Xᵀy` gives the exact solution but is expensive for large `p`.
3. **Ridge regression** adds `α||w||²` to the cost function, stabilizing the solution when features are correlated.
4. **Always scale features** before fitting Ridge (otherwise, features with larger scales get unfairly penalized).
5. **Gradient descent** is an iterative alternative to the closed-form solution, essential for large datasets.
6. Use **cross-validation** to select the optimal `α` for Ridge.

---

*Previous: [Intro: Supervised & Unsupervised Learning](./ml-intro.md) | Next: [Evaluation Metrics](./evaluation-metrics.md)*
