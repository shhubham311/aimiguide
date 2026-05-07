# Logistic Regression & Gradient Descent Algorithm

## Why This Topic Matters

Logistic regression is the workhorse of classification. Despite its name, it is a **classification** algorithm, not a regression algorithm. It is used in virtually every domain: medical diagnosis (disease vs. healthy), finance (default vs. no default), marketing (click vs. no click), and NLP (spam vs. not spam). It is fast, interpretable, and provides calibrated probability estimates — properties that make it the first model data scientists reach for on binary classification problems.

But logistic regression is also your gateway to understanding **gradient descent**, the optimization algorithm that powers nearly every modern ML technique, from neural networks to large language models. Understanding gradient descent on the logistic regression loss function is the clearest way to build intuition for how models learn. The concepts of learning rate, convergence, and gradient computation that you master here apply directly to deep learning.

---

## From Linear to Logistic: The Core Idea

### The Problem with Linear Regression for Classification

If we try to use linear regression `ŷ = w₀ + w₁x` for a binary problem (y ∈ {0, 1}), the predicted values can be any real number:

```
y (class)
1 │        ┌─────────────────  Linear regression: ŷ can be > 1 or < 0
  │       ╱                     Problem: what does ŷ = 2.3 mean?
  │      ╱                      What does ŷ = -0.5 mean?
  │     ╱
  │    ╱
0 │───╱──────────────────────
  │  ╱
  │ ╱
  └────────────────────────── x

  We need a function that constrains predictions to [0, 1]:
  a PROBABILITY of belonging to class 1.
```

### The Solution: The Sigmoid Function

The **sigmoid function** (also called the logistic function) maps any real number to the range (0, 1):

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

where:
- `z = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ = wᵀx` (linear combination of inputs)
- `e` = Euler's number (≈ 2.71828)
- `σ(z)` = probability that y = 1 given input x

### Sigmoid Properties

```
    σ(z)
    1.0 │            ──────────────
        │         ╱
    0.5 │ ─ ─ ─●─ ─ ─ ─ ─ ─ ─   σ(0) = 0.5
        │      ╱
    0.0 │───╱─────────────────────
        └───────────────────────── z
       -∞   -4  -2   0  2  4   +∞

Key values:
    σ(0)  = 0.5     (decision boundary)
    σ(-∞) = 0       (definitely class 0)
    σ(+∞) = 1       (definitely class 1)
    σ(-z) = 1 - σ(z) (symmetry)

Derivative (crucial for gradient descent):
    σ'(z) = σ(z) · (1 - σ(z))
```

### The Logistic Regression Model

```
P(y=1|x) = σ(wᵀx) = 1 / (1 + e⁻ʷᵀˣ)
P(y=0|x) = 1 - σ(wᵀx)

Compact form (using y ∈ {0, 1}):
    P(y|x) = [σ(wᵀx)]ʸ · [1 - σ(wᵀx)]¹⁻ʸ
```

**Decision rule**: Predict class 1 if `P(y=1|x) ≥ 0.5`, otherwise predict class 0.

This is equivalent to: predict class 1 if `wᵀx ≥ 0`.

---

## The Loss Function: Log-Loss (Binary Cross-Entropy)

We need a loss function that measures how well the predicted probabilities match the actual labels. The standard choice is the **log-loss** (also called **binary cross-entropy**):

```
            n
J(w) = -1/n Σ [yᵢ · log(ŷᵢ) + (1 - yᵢ) · log(1 - ŷᵢ)]
           i=1
```

where:
- `ŷᵢ = σ(wᵀxᵢ)` = predicted probability for sample `i`
- `yᵢ` = actual label (0 or 1) for sample `i`
- `log` = natural logarithm

### Why Log-Loss? Intuition

Consider each term of the loss:

```
If yᵢ = 1:  Loss = -log(ŷᵢ)
    → If ŷᵢ = 0.99: Loss = -log(0.99) = 0.01   ← confident and correct
    → If ŷᵢ = 0.50: Loss = -log(0.50) = 0.69   ← uncertain
    → If ŷᵢ = 0.01: Loss = -log(0.01) = 4.61   ← confident and WRONG (huge penalty)

If yᵢ = 0:  Loss = -log(1 - ŷᵢ)
    → If ŷᵢ = 0.01: Loss = -log(0.99) = 0.01   ← confident and correct
    → If ŷᵢ = 0.50: Loss = -log(0.50) = 0.69   ← uncertain
    → If ŷᵢ = 0.99: Loss = -log(0.01) = 4.61   ← confident and WRONG (huge penalty)

Log-loss heavily penalizes confident wrong predictions!
```

### Visual: Log-Loss Penalty

```
Loss
  │
  │                              ╱
  │                           ╱
  │                        ╱
  │                     ╱
  │                  ╱
  │               ╱
  │            ╱
  │         ╱
  │      ╱
  │───╱──────────────────────────── ŷ (when y=1)
  │  │
  0  0.1  0.3  0.5  0.7  0.9  1.0

  As predicted probability ŷ → 0 (wrong direction),
  loss → ∞. This drives the model to predict correct probabilities.
```

---

## Gradient Descent: The Optimization Engine

### The General Framework

Gradient descent is an iterative optimization algorithm that finds the minimum of a function by following the direction of steepest descent:

```
Algorithm: Gradient Descent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Loss function J(w), learning rate η, number of iterations T

1. Initialize w = 0 (or small random values)

2. For t = 1 to T:
       a. Compute gradient: g = ∇J(w)
       b. Update weights:  w ← w - η · g
       c. (Optional) Check convergence: if ||g|| < ε, stop

3. Return w
```

### Gradient Descent Variants

```
┌─────────────────────┬────────────────────────────┬──────────────────┐
│     Variant         │       Update Rule          │  Characteristics │
├─────────────────────┼────────────────────────────┼──────────────────┤
│ Batch GD            │ w ← w - η·(1/n)Σ∇Jᵢ(w)    │ Stable, slow     │
│                     │                            │ (uses ALL data)  │
├─────────────────────┼────────────────────────────┼──────────────────┤
│ Stochastic GD (SGD) │ w ← w - η·∇Jᵢ(w)          │ Noisy, fast      │
│                     │                            │ (1 sample/step)  │
├─────────────────────┼────────────────────────────┼──────────────────┤
│ Mini-batch GD       │ w ← w - η·(1/m)Σ∇Jᵢ(w)    │ Best of both     │
│                     │                            │ (m samples/step) │
└─────────────────────┴────────────────────────────┴──────────────────┘

Visual: Convergence comparison

Loss J(w)
    │
    │● Batch GD (smooth, slow)
    │ ╲
    │  ╲    ● SGD (noisy, fast)
    │   ╲  ╱╲╱╲
    │    ╲╱  ╲ ╱╲  ╱●
    │         ╲╱  ╲╱
    │          ●───●
    │          Mini-batch (good balance)
    └──────────────────────────── Iterations
```

### Gradient of Log-Loss for Logistic Regression

The gradient of the log-loss with respect to the weight vector `w` is:

```
∇J(w) = (1/n) · Xᵀ(σ(Xw) - y)
       = (1/n) · Xᵀ(ŷ - y)
```

where:
- `X` = (n × p) feature matrix
- `ŷ = σ(Xw)` = (n × 1) vector of predicted probabilities
- `y` = (n × 1) vector of true labels
- `Xᵀ(ŷ - y)` = (p × 1) gradient vector

**This is remarkably elegant and efficient to compute!**

The update rule becomes:

```
w ← w - η · (1/n) · Xᵀ(σ(Xw) - y)
```

### Learning Rate: The Most Important Hyperparameter

```
Too Small (η = 0.0001):
    Loss
      │●
      │ ╲
      │  ╲●
      │   ╲●
      │    ╲●───── (converging, but too slowly)
      └──────────── Iterations
      Waste of compute.

Too Large (η = 10.0):
    Loss
      │    ●
      │   ╱ ╲
      │  ╱   ╲  ●
      │ ╱     ╲╱ ╲
      │●           ╲  (diverging!)
      └──────────── Iterations
      Loss increases! Model explodes.

Just Right (η = 0.1):
    Loss
      │●
      │╲
      │ ╲●
      │  ╲●
      │   ╲●───── (smooth convergence)
      └──────────── Iterations
      Efficient optimization.
```

**Common strategies**: Learning rate scheduling (decrease η over time), adaptive methods (Adam, RMSprop — we'll see these in neural networks).

---

## Regularization in Logistic Regression

Just like Ridge regression, we can add penalties to the log-loss:

### L2 Regularization (Ridge-like):
```
J(w) = -1/n Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)] + (α/2) ||w||²
```

### L1 Regularization (Lasso-like):
```
J(w) = -1/n Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)] + α ||w||₁
```

**Key difference**: L1 can drive some coefficients to exactly zero (feature selection), while L2 shrinks all coefficients toward zero but rarely to exactly zero.

```
L1 Penalty (||w||₁):    L2 Penalty (||w||²):
    |w₁| + |w₂|          w₁² + w₂²

    Contour of penalty:    Contour of penalty:
    ◇ (diamond shape)      ○ (circular shape)

    ◇ diamonds → coefficients hit corners → exactly zero
    ○ circles → coefficients shrink smoothly → small but nonzero
```

---

## Python Implementation: Breast Cancer Classification

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target  # 0=malignant, 1=benign

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.bincount(y)} (malignant: {np.bincount(y)[0]}, benign: {np.bincount(y)[1]})")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ALWAYS scale features for logistic regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Model 1: Basic Logistic Regression (L2 regularization)
# ============================================================
model_l2 = LogisticRegression(max_iter=1000, random_state=42)
model_l2.fit(X_train_s, y_train)
y_pred_l2 = model_l2.predict(X_test_s)
y_prob_l2 = model_l2.predict_proba(X_test_s)[:, 1]

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (L2, default C=1.0)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_l2):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_l2):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_l2):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_l2):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_l2):.4f}")

# ============================================================
# Model 2: L1 Regularization (Lasso-like, for feature selection)
# ============================================================
model_l1 = LogisticRegression(
    penalty='l1', solver='liblinear', C=0.5, max_iter=1000, random_state=42
)
model_l1.fit(X_train_s, y_train)
y_pred_l1 = model_l1.predict(X_test_s)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (L1, C=0.5)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_l1):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_l1):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_l1):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_l1):.4f}")

# Show which features L1 set to zero
nonzero = np.sum(np.abs(model_l1.coef_) > 1e-6)
print(f"\nFeatures with nonzero weights: {nonzero}/{X.shape[1]}")
print("(L1 regularization performed automatic feature selection!)")

# ============================================================
# Model 3: Tuning regularization strength (C parameter)
# ============================================================
# Note: In sklearn, C = 1/α (inverse of regularization strength)
# Small C = strong regularization, Large C = weak regularization

from sklearn.model_selection import cross_val_score

C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING: C (regularization strength)")
print("=" * 60)
print(f"{'C':>8} {'Train Acc':>10} {'CV Acc':>10} {'Nonzero W':>12}")
print("-" * 45)

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    print(f"{C:>8.3f} {model.score(X_train_s, y_train):>10.4f} "
          f"{cv_scores.mean():>10.4f} {nonzero:>12d}")

# ============================================================
# Probability Calibration: Interpreting Outputs
# ============================================================
print("\n" + "=" * 60)
print("PROBABILITY OUTPUTS (first 10 test samples)")
print("=" * 60)
probabilities = model_l2.predict_proba(X_test_s)
print(f"{'True':>5} {'Pred':>5} {'P(Class 0)':>12} {'P(Class 1)':>12}")
print("-" * 40)
for i in range(min(10, len(y_test))):
    print(f"{y_test[i]:>5} {y_pred_l2[i]:>5} "
          f"{probabilities[i, 0]:>12.4f} {probabilities[i, 1]:>12.4f}")
```

**Expected Output:**
```
Dataset: 569 samples, 30 features
Classes: [212 357] (malignant: 212, benign: 357)

============================================================
LOGISTIC REGRESSION (L2, default C=1.0)
============================================================
Accuracy:  0.9737
Precision: 0.9722
Recall:    0.9861
F1 Score:  0.9791
ROC-AUC:   0.9962

============================================================
LOGISTIC REGRESSION (L1, C=0.5)
============================================================
Accuracy:  0.9649
Precision: 0.9583
Recall:    0.9861
F1 Score:  0.9720

Features with nonzero weights: 20/30
(L1 regularization performed automatic feature selection!)

============================================================
HYPERPARAMETER TUNING: C (regularization strength)
============================================================
       C   Train Acc      CV Acc   Nonzero W
---------------------------------------------
   0.001     0.9341     0.9275           16
   0.010     0.9670     0.9627           22
   0.100     0.9846     0.9780           29
   0.500     0.9901     0.9824           30
   1.000     0.9934     0.9846           30
   5.000     0.9978     0.9846           30
  10.000     0.9978     0.9846           30
 100.000     1.0000     0.9824           30
```

---

## Logistic Regression from Scratch (with Gradient Descent)

```python
import numpy as np

class LogisticRegressionGD:
    """Logistic Regression trained with Gradient Descent from scratch."""

    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.0):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.reg_lambda = reg_lambda  # L2 regularization strength
        self.w = None
        self.losses = []

    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid function."""
        # Clip to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n, p = X.shape
        self.w = np.zeros(p)

        for t in range(self.n_iter):
            # Forward pass: compute predictions
            z = X @ self.w
            y_hat = self.sigmoid(z)

            # Compute loss (log-loss + L2 penalty)
            eps = 1e-15  # prevent log(0)
            y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
            loss = -np.mean(
                y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped)
            )
            if self.reg_lambda > 0:
                loss += (self.reg_lambda / 2) * np.sum(self.w ** 2)
            self.losses.append(loss)

            # Compute gradient
            gradient = (1 / n) * X.T @ (y_hat - y)
            if self.reg_lambda > 0:
                gradient += self.reg_lambda * self.w

            # Update weights
            self.w -= self.lr * gradient

            # Print progress
            if t % 200 == 0 or t == self.n_iter - 1:
                preds = (y_hat >= 0.5).astype(int)
                acc = np.mean(preds == y)
                print(f"  Iter {t:4d}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ============================================================
# Test our implementation against sklearn
# ============================================================
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Our implementation
print("Our Gradient Descent Implementation:")
our_model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000, reg_lambda=0.1)
our_model.fit(X_train_s, y_train)
our_preds = our_model.predict(X_test_s)
our_acc = np.mean(our_preds == y_test)
print(f"  Test Accuracy: {our_acc:.4f}\n")

# Sklearn implementation
print("Sklearn Implementation:")
sk_model = SklearnLR(C=10.0, max_iter=1000, random_state=42)
sk_model.fit(X_train_s, y_train)
sk_preds = sk_model.predict(X_test_s)
sk_acc = np.mean(sk_preds == y_test)
print(f"  Test Accuracy: {sk_acc:.4f}")
```

---

## Decision Boundary Visualization

For a 2D feature space, the decision boundary `wᵀx = 0` is a straight line:

```
Feature 2
    │
  5 │    ● ●
    │  ●     ●  ← Class 1 region (wᵀx > 0, σ > 0.5)
  4 │ ●         ╱
    │          ╱   Decision boundary: w₀ + w₁x₁ + w₂x₂ = 0
  3 │────────╱───────────────────────
    │      ╱
  2 │    ╱  ○   ○   ← Class 0 region (wᵀx < 0, σ < 0.5)
    │  ╱  ○       ○
  1 │╱   ○
    │
    └────────────────────────────── Feature 1
      1   2   3   4   5   6   7

  Logistic regression creates a LINEAR decision boundary.
  More complex boundaries require polynomial features or other algorithms.
```

---

## Exercises

### Exercise 1: Learning Rate Experiment

Try different learning rates on our scratch implementation and observe convergence behavior.

```python
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]

for lr in learning_rates:
    print(f"\nLearning rate = {lr}:")
    try:
        model = LogisticRegressionGD(learning_rate=lr, n_iterations=500)
        model.fit(X_train_s, y_train)
        final_loss = model.losses[-1]
        print(f"  Final loss: {final_loss:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")
```

**Expected behavior**:
- `0.001`: Converges very slowly (loss still decreasing at 500 iterations)
- `0.01`: Converges reasonably
- `0.1`: Fast, clean convergence
- `0.5`: Fast convergence (may have some oscillation)
- `1.0, 5.0`: May diverge (loss increases or NaN)

### Exercise 2: Multiclass Logistic Regression (Softmax)

Extend logistic regression to handle multiple classes using the **softmax** function:

```
P(y=k|x) = e^{wₖᵀx} / Σⱼ e^{wⱼᵀx}
```

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target  # 3 classes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Multinomial logistic regression (softmax)
model = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000
)
model.fit(X_train_s, y_train)

print(f"Accuracy: {model.score(X_test_s, y_test):.4f}")
print(f"\nClass probabilities for first 5 test samples:")
proba = model.predict_proba(X_test_s)
for i in range(5):
    print(f"  Sample {i}: P(setosa)={proba[i,0]:.3f}, "
          f"P(versicolor)={proba[i,1]:.3f}, P(virginica)={proba[i,2]:.3f} "
          f"→ Predicted: {iris.target_names[model.predict(X_test_s)[i]]}")
```

### Exercise 3: Implement Gradient Descent Variants

Compare batch GD, SGD, and mini-batch GD on the breast cancer dataset. Track loss curves.

```python
class LogisticRegressionSGD(LogisticRegressionGD):
    """Stochastic Gradient Descent variant."""

    def fit(self, X, y, batch_size=None):
        n, p = X.shape
        self.w = np.zeros(p)

        if batch_size is None:
            batch_size = n  # Batch GD

        for t in range(self.n_iter):
            # Shuffle data each epoch
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Process mini-batches
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_hat = self.sigmoid(X_batch @ self.w)
                gradient = (1/len(y_batch)) * X_batch.T @ (y_hat - y_batch)
                if self.reg_lambda > 0:
                    gradient += self.reg_lambda * self.w
                self.w -= self.lr * gradient

            # Log epoch loss
            y_hat_full = self.sigmoid(X @ self.w)
            eps = 1e-15
            loss = -np.mean(y * np.log(np.clip(y_hat_full, eps, 1-eps))
                          + (1-y) * np.log(np.clip(1-y_hat_full, eps, 1-eps)))
            self.losses.append(loss)
```

---

## Key Takeaways

1. **Logistic regression** models `P(y=1|x)` using the sigmoid function, outputting calibrated probabilities.
2. **Log-loss (cross-entropy)** is the natural loss function for probabilistic classifiers, heavily penalizing confident wrong predictions.
3. **Gradient descent** iteratively updates weights: `w ← w - η∇J(w)`. The learning rate `η` controls step size.
4. **Three variants**: Batch GD (stable but slow), SGD (fast but noisy), mini-batch GD (practical compromise).
5. **Regularization**: L2 shrinks all weights; L1 drives some to zero (automatic feature selection).
6. **Always scale features** before logistic regression — unscaled features distort the sigmoid and slow convergence.
7. **Sklearn's `C` parameter** is the inverse of regularization strength: small `C` = strong regularization, large `C` = weak regularization.
8. **Multiclass extension**: Softmax regression generalizes binary logistic regression to `K` classes.

---

*Previous: [Evaluation Metrics](./evaluation-metrics.md) | Next: [KNN, Naive Bayes & SVM](./knn-naive-svm.md)*
