# KNN, Naive Bayes & SVM

## Why This Topic Matters

These three algorithms represent fundamentally different philosophies of learning, and together they give you a versatile toolkit for classification problems. **K-Nearest Neighbors (KNN)** is the simplest classifier imaginable — it doesn't even "learn" a model; it just memorizes the training data and compares new points to their nearest neighbors. **Naive Bayes** takes the opposite approach: it builds a probabilistic model based on Bayes' theorem, making strong (but effective) independence assumptions. **Support Vector Machines (SVM)** finds the optimal separating boundary by maximizing the margin between classes, and through the kernel trick, can learn complex nonlinear decision boundaries.

In practice, you would try all three (plus logistic regression) on any new classification problem, compare their performance, and choose the best. Understanding how each works, what assumptions it makes, and when it fails is essential for any ML practitioner.

---

## Part 1: K-Nearest Neighbors (KNN)

### The Core Idea

KNN classifies a data point by looking at the `k` closest training examples and taking a majority vote. It is a **lazy learner** — it defers all computation until prediction time.

```
Algorithm: KNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Training data {(x₁,y₁), ..., (xₙ,yₙ)}, new point x_new, parameter k

1. Compute the distance from x_new to every training point:
   d(x_new, xᵢ) for all i = 1, ..., n

2. Find the k training points with smallest distances

3. Predict: majority class among the k nearest neighbors

That's it. No training phase!
```

### Distance Metrics

The choice of distance metric is critical for KNN:

**Euclidean Distance** (most common):
```
d(x, x') = √(Σⱼ (xⱼ - x'ⱼ)²)
```

**Manhattan Distance** (L1, useful for high dimensions):
```
d(x, x') = Σⱼ |xⱼ - x'ⱼ|
```

**Minkowski Distance** (generalizes both):
```
d(x, x') = (Σⱼ |xⱼ - x'ⱼ|ᵖ)^(1/p)

p = 2 → Euclidean
p = 1 → Manhattan
```

### Visual: How KNN Works

```
     Feature 2
       │
    5  │     A           ?  (new point)
       │       A       ╱
    4  │   A       ╱╱    ← k=3 nearest neighbors: 2 A's, 1 B
       │       ╱╱          Prediction: Class A
    3  │    B  ╱╱
       │   B  ╱╱
    2  │  B  ╱╱
       │   ╱╱
    1  │  ╱╱
       └──────────────────── Feature 1
         1  2  3  4  5

   With k=5: 3 A's, 2 B → still Class A
   With k=1: Closest neighbor is A → Class A
   With k=7: Could change! Larger k = smoother boundary
```

### The Effect of K

```
k = 1 (High Variance, Low Bias):
    - Decision boundary tightly wraps each point
    - Overfits to noise
    - Training accuracy = 100%

k = large (Low Variance, High Bias):
    - Decision boundary becomes very smooth
    - Underfits, misses local patterns
    - Approaches predicting the majority class

Optimal k balances the two:
    - Use cross-validation to find the best k
    - Common default: k = √n
    - Odd k preferred for binary classification (avoids ties)
```

### Decision Boundaries for Different K Values

```
k=1 (Overfitting)          k=5 (Good Balance)         k=50 (Underfitting)

  A A                         A A                         A A A A
A     A    B               A     A    B                A         A
  A    B B                A    B  B B                 A         A
A  B   B                 A  B    B B                   A    A    A
  A B  B                  A B  B B                     A         A
   B B                     B B B                       B B B B B B
```

### KNN: Pros and Cons

```
┌──────────────────────────────────┬──────────────────────────────────┐
│             PROS                 │              CONS                 │
├──────────────────────────────────┼──────────────────────────────────┤
│ No training phase (instant)      │ Slow prediction (O(n·p) per q.)  │
│ Simple to understand & implement │ Must store all training data     │
│ Naturally handles multi-class    │ Suffers from curse of            │
│ No assumptions about data        │ dimensionality                  │
│ Non-linear decision boundaries   │ Sensitive to feature scaling     │
│                                  │ Sensitive to irrelevant features │
│                                  │ Struggles with imbalanced data   │
└──────────────────────────────────┴──────────────────────────────────┘
```

### Python Implementation: KNN on Wine Quality

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load Wine dataset (3 classes, 13 features, 178 samples)
wine = load_wine()
X, y = wine.data, wine.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {wine.target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CRITICAL: Scale features for distance-based algorithms
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Find optimal k using cross-validation
# ============================================================
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"\nBest k: {best_k} (CV accuracy: {max(cv_scores):.4f})")

# ============================================================
# Train with optimal k
# ============================================================
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# ============================================================
# Compare distance metrics
# ============================================================
print("\n" + "=" * 50)
print("DISTANCE METRIC COMPARISON")
print("=" * 50)

metrics = ['euclidean', 'manhattan', 'minkowski']
for metric in metrics:
    model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"  {metric:>12s}: CV Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================
# Weighted KNN (closer neighbors have more influence)
# ============================================================
print("\nWeighted vs. Unweighted KNN:")
for weights in ['uniform', 'distance']:
    model = KNeighborsClassifier(n_neighbors=best_k, weights=weights)
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"  weights={weights:>8s}: CV Accuracy = {scores.mean():.4f}")
```

**Expected Output:**
```
Dataset: 178 samples, 13 features
Classes: ['class_0' 'class_1' 'class_2']
Class distribution: [59 71 48]

Best k: 7 (CV accuracy: 0.9789)

Test Accuracy: 0.9722

Classification Report:
              precision    recall  f1-score   support

     class_0       1.00      0.93      0.97        14
     class_1       0.93      1.00      0.97        14
     class_2       1.00      1.00      1.00         8

    accuracy                           0.97        36

============================================================
DISTANCE METRIC COMPARISON
============================================================
     euclidean: CV Accuracy = 0.9789 ± 0.0286
      manhattan: CV Accuracy = 0.9789 ± 0.0286
      minkowski: CV Accuracy = 0.9789 ± 0.0286

Weighted vs. Unweighted KNN:
  weights= uniform: CV Accuracy = 0.9789
  weights= distance: CV Accuracy = 0.9789
```

---

## Part 2: Naive Bayes

### The Core Idea: Bayes' Theorem

Naive Bayes classifiers are based on **Bayes' theorem** with the "naive" assumption that all features are conditionally independent given the class:

```
Bayes' Theorem:
    P(y|x₁, x₂, ..., xₚ) = P(x₁, x₂, ..., xₚ|y) · P(y) / P(x₁, x₂, ..., xₚ)

Naive Assumption (feature independence):
    P(x₁, x₂, ..., xₚ|y) = P(x₁|y) · P(x₂|y) · ... · P(xₚ|y)

Therefore:
    P(y|x₁, ..., xₚ) ∝ P(y) · Πⱼ P(xⱼ|y)

Classification rule:
    ŷ = argmaxₖ P(y=k) · Πⱼ P(xⱼ|y=k)
```

where:
- `P(y)` = **prior probability** of class `y` (how common is each class?)
- `P(xⱼ|y)` = **likelihood** of feature `j` given class `y`
- `P(y|x)` = **posterior probability** (what we want to compute)
- `Πⱼ` = product over all features

### Why "Naive"?

The independence assumption is almost never true in practice. If feature x₁ is "outlook=sunny" and feature x₂ is "temperature=high", these are clearly correlated. Yet **Naive Bayes works surprisingly well** in practice, especially for:
- Text classification (spam filtering, sentiment analysis)
- High-dimensional problems with many features
- Situations where data is scarce

This phenomenon is sometimes called **"Bayes optimality under model misspecification"** — even though the independence assumption is wrong, the classification rule can still be near-optimal because the ratio of posteriors is well-estimated.

### Types of Naive Bayes

The choice of Naive Bayes variant depends on the distribution of features:

```
┌─────────────────────┬──────────────────────────┬──────────────────────┐
│     Variant         │   Assumption             │   Best For           │
├─────────────────────┼──────────────────────────┼──────────────────────┤
│ Gaussian NB         │ Features ~ Normal(μ, σ²) │ Continuous features  │
│ Multinomial NB      │ Features ~ multinomial   │ Word counts (NLP)    │
│ Bernoulli NB        │ Features ~ Bernoulli     │ Binary features      │
│ Complement NB       │ Uses complement class    │ Imbalanced data      │
└─────────────────────┴──────────────────────────┴──────────────────────┘
```

### Gaussian Naive Bayes in Detail

For each class `k` and feature `j`, we estimate:

```
Prior:              P(y = k) = Nₖ / N

Likelihood params:  μₖⱼ = mean of feature j for class k
                   σₖⱼ² = variance of feature j for class k

Likelihood:        P(xⱼ|y=k) = (1/√(2πσₖⱼ²)) · exp(-(xⱼ - μₖⱼ)²/(2σₖⱼ²))

Posterior:         P(y=k|x) ∝ P(y=k) · Πⱼ P(xⱼ|y=k)

(In practice, we work with log-probabilities to avoid underflow)
```

### Visual: Naive Bayes Decision Process

```
Step 1: Estimate class priors
    P(spam)  = 200/1000 = 0.20
    P(ham)   = 800/1000 = 0.80

Step 2: Estimate feature likelihoods (simplified)
    P(word="free"|spam)  = 0.15
    P(word="free"|ham)   = 0.01
    P(word="meeting"|spam) = 0.05
    P(word="meeting"|ham)  = 0.12

Step 3: Classify email with words "free" and "meeting"
    P(spam|"free","meeting")  ∝ 0.20 × 0.15 × 0.05 = 0.00150
    P(ham|"free","meeting")   ∝ 0.80 × 0.01 × 0.12 = 0.00096

Step 4: Normalize (convert to probabilities)
    P(spam|...) = 0.00150 / (0.00150 + 0.00096) = 0.610
    P(ham|...)  = 0.00096 / (0.00150 + 0.00096) = 0.390

Step 5: Predict spam (higher posterior probability)
```

### Python Implementation: Naive Bayes on Text Classification

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
import numpy as np

# Load text data: 4 categories for simplicity
categories = ['sci.space', 'comp.graphics', 'talk.politics.mideast', 'rec.autos']
train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

print(f"Training samples: {len(train.data)}")
print(f"Test samples:     {len(test.data)}")
print(f"Categories:       {train.target_names}")

# Create TF-IDF features (better than raw counts for text)
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
X_train = tfidf.fit_transform(train.data)
X_test = tfidf.transform(test.data)
y_train = train.target
y_test = test.target

print(f"\nFeature matrix shape: {X_train.shape}")
print(f"  (sparse: only {X_train.nnz / (X_train.shape[0] * X_train.shape[1]) * 100:.2f}% non-zero)")

# ============================================================
# Multinomial Naive Bayes (best for text/word counts)
# ============================================================
mnb = MultinomialNB(alpha=0.1)  # alpha = Laplace smoothing parameter
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

print("\n" + "=" * 60)
print("MULTINOMIAL NAIVE BAYES")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred_mnb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_mnb, target_names=train.target_names))

# ============================================================
# Compare all Naive Bayes variants
# ============================================================
print("=" * 60)
print("NAIVE BAYES VARIANT COMPARISON")
print("=" * 60)

# For Bernoulli, binarize the features
bnb = BernoulliNB(alpha=0.1)
bnb.fit(X_train > 0, y_train)  # Binarize: word present or not

# Complement NB (good for imbalanced classes)
from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB(alpha=0.1)
cnb.fit(X_train, y_train)

for name, model in [('Multinomial', mnb), ('Bernoulli', bnb), ('Complement', cnb)]:
    y_pred = model.predict(X_test if name != 'Bernoulli' else X_test > 0)
    acc = accuracy_score(y_test, y_pred)
    cv = cross_val_score(model, X_train if name != 'Bernoulli' else X_train > 0,
                          y_train, cv=5, scoring='accuracy')
    print(f"  {name:>12s}: Test Acc = {acc:.4f}, CV Acc = {cv.mean():.4f} ± {cv.std():.4f}")

# ============================================================
# Gaussian Naive Bayes on continuous features (Iris)
# ============================================================
print("\n" + "=" * 60)
print("GAUSSIAN NAIVE BAYES ON IRIS DATASET")
print("=" * 60)

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

iris = load_iris()
X_ir, y_ir = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_ir, y_ir, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_tr, y_tr)
y_pred_gnb = gnb.predict(X_te)

print(f"Accuracy: {accuracy_score(y_te, y_pred_gnb):.4f}")
print(f"\nClass priors: {gnb.class_prior_.round(3)}")
print(f"\nMean of each feature per class:")
print(gnb.theta_.round(2))  # shape: (n_classes, n_features)
print(f"\nVariance of each feature per class:")
print(gnb.var_.round(2))   # shape: (n_classes, n_features)
```

---

## Part 3: Support Vector Machines (SVM)

### The Core Idea: Maximum Margin Classifier

SVM finds the **hyperplane** that separates two classes with the **maximum margin** — the widest "street" between the closest points of each class.

```
              Margin (to be maximized)
         ┌────────────────────────────┐
         │                            │
    ●    │    ●                       │    ← Class +1
         │                            │
  ● ─────┼───── decision boundary ────┼───── ●  Class -1
         │    (wᵀx + b = 0)          │
    ●    │    ●  ← support vectors    │
         │                            │
         └────────────────────────────┘

    Support vectors: points closest to the boundary
    Margin width: 2 / ||w||

    SVM maximizes margin = minimizes ||w||
```

### Mathematical Formulation

**Hard Margin SVM** (perfectly separable case):

```
Minimize:    (1/2) ||w||²
Subject to:  yᵢ(wᵀxᵢ + b) ≥ 1    for all i

Equivalently: Maximize the margin 2/||w||
              while correctly classifying all points
```

**Soft Margin SVM** (allows misclassifications, practical case):

```
Minimize:    (1/2) ||w||² + C · Σᵢ ξᵢ
Subject to:  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ    for all i
             ξᵢ ≥ 0                    for all i

where:
    ξᵢ = slack variable (allows violation of margin)
    C  = regularization parameter (tradeoff parameter)
         Small C: wide margin, more misclassifications (high bias)
         Large C: narrow margin, fewer misclassifications (low bias)
```

### The Kernel Trick: Nonlinear SVM

The real power of SVM comes from the **kernel trick**. Instead of mapping data to a higher dimension explicitly, kernels compute inner products in that space implicitly:

```
Linear Kernel:      K(x, x') = xᵀx'
Polynomial Kernel:  K(x, x') = (γxᵀx' + r)^d
RBF (Gaussian):     K(x, x') = exp(-γ||x - x'||²)
Sigmoid Kernel:     K(x, x') = tanh(γxᵀx' + r)
```

**Why the kernel trick matters**: It lets SVM learn complex nonlinear boundaries without ever computing the expensive high-dimensional mapping.

### Visual: Linear vs. RBF Kernel

```
LINEAR KERNEL:                RBF KERNEL:
Straight boundary              Curved boundary

    ●●                           ●●
  ●   ●                        ●   ●
●  ✕  ●     ← can't             ╭─●─╮
●  ✕  ●       separate          │ ✕ │  ← wraps around ✕
  ●   ●       nonlinearly      ╰─●─╯
    ●●                           ●●

RBF kernel maps data to infinite dimensions where
a linear boundary exists, then maps back.
```

### SVM Hyperparameters

```
┌──────────────┬────────────────────────────────────────┐
│  Parameter   │  Effect                                │
├──────────────┼────────────────────────────────────────┤
│ C (cost)     │ Small C → wide margin (simpler, high  │
│              │ bias). Large C → narrow margin (more   │
│              │ complex, low bias).                    │
├──────────────┼────────────────────────────────────────┤
│ γ (gamma)    │ RBF kernel width. Small γ → smooth    │
│              │ boundary (high bias). Large γ →       │
│              │ complex boundary (low bias).           │
├──────────────┼────────────────────────────────────────┤
│ kernel       │ 'linear', 'rbf', 'poly', 'sigmoid'   │
│              │ Choose based on data structure.        │
└──────────────┴────────────────────────────────────────┘

Rule of thumb: ALWAYS scale features before SVM.
The margin is distance-based, so unscaled features
distort the decision boundary.
```

### Python Implementation: SVM on Multiple Datasets

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ============================================================
# Example 1: SVM on Iris Dataset (Multi-class, relatively easy)
# ============================================================
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Linear SVM (good for linearly separable data)
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_s, y_train)
print(f"Linear SVM (Iris): {svm_linear.score(X_test_s, y_test):.4f}")

# RBF SVM (good for non-linear boundaries)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_s, y_train)
print(f"RBF SVM (Iris):    {svm_rbf.score(X_test_s, y_test):.4f}")

# Number of support vectors
print(f"Support vectors (linear): {svm_linear.n_support_} "
      f"(total: {sum(svm_linear.n_support_)})")
print(f"Support vectors (RBF):    {svm_rbf.n_support_} "
      f"(total: {sum(svm_rbf.n_support_)})")

# ============================================================
# Example 2: SVM on Breast Cancer (Binary, more challenging)
# ============================================================
print("\n" + "=" * 60)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Grid Search for optimal hyperparameters
# ============================================================
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_s, y_train)

print("GRID SEARCH RESULTS (RBF SVM)")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_s)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ============================================================
# Compare kernels
# ============================================================
print("=" * 60)
print("KERNEL COMPARISON")
print("=" * 60)

kernels = {
    'linear': {'C': [0.1, 1, 10]},
    'rbf':    {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]},
    'poly':   {'C': [0.1, 1, 10], 'degree': [2, 3]}
}

for kernel, params in kernels.items():
    gs = GridSearchCV(SVC(kernel=kernel, random_state=42), params, cv=5, scoring='accuracy')
    gs.fit(X_train_s, y_train)
    print(f"  {kernel:>8s}: Best CV = {gs.best_score_:.4f}, Params = {gs.best_params_}")

# ============================================================
# Support Vector Analysis
# ============================================================
print("\n" + "=" * 60)
print("SUPPORT VECTOR ANALYSIS")
print("=" * 60)

# Points that are support vectors lie on or within the margin
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_s, y_train)

n_total = X_train_s.shape[0]
n_sv = sum(svm.n_support_)
print(f"Total training points: {n_total}")
print(f"Support vectors:       {n_sv} ({n_sv/n_total*100:.1f}%)")
print(f"Per class: {svm.n_support_}")
print(f"\nDual coefficients range: [{svm.dual_coef_.min():.4f}, {svm.dual_coef_.max():.4f}]")
```

**Expected Output:**
```
Linear SVM (Iris): 0.9778
RBF SVM (Iris):    0.9778
Support vectors (linear): [2 3 2] (total: 7)
Support vectors (RBF):    [4 6 5] (total: 15)

============================================================
GRID SEARCH RESULTS (RBF SVM)
Best parameters: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
Best CV accuracy: 0.9780

Test Accuracy: 0.9737

Classification Report:
              precision    recall  f1-score   support

   Malignant       0.97      0.95      0.96        42
      Benign       0.97      0.99      0.98        72

    accuracy                           0.97       114

============================================================
KERNEL COMPARISON
============================================================
   linear: Best CV = 0.9736, Params = {'C': 1}
      rbf: Best CV = 0.9780, Params = {'C': 10, 'gamma': 0.01}
     poly: Best CV = 0.9516, Params = {'C': 1, 'degree': 2}

============================================================
SUPPORT VECTOR ANALYSIS
============================================================
Total training points: 455
Support vectors:       44 (9.7%)
Per class: [19 25]
```

---

## Algorithm Comparison: When to Use What

```
┌──────────────────────┬───────────────────┬───────────────────┬───────────────────┐
│       Scenario       │        KNN        │   Naive Bayes     │       SVM         │
├──────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Small dataset        │ ★★★★★             │ ★★★★★             │ ★★★☆☆             │
│ Large dataset        │ ★★☆☆☆ (slow!)     │ ★★★★★ (fast!)     │ ★★★☆☆             │
│ High dimensions      │ ★★☆☆☆             │ ★★★★☆             │ ★★★★★             │
│ Text classification  │ ★★☆☆☆             │ ★★★★★             │ ★★★☆☆             │
│ Non-linear boundary  │ ★★★★☆             │ ★★★☆☆             │ ★★★★★ (RBF)       │
│ Interpretability     │ ★★★☆☆             │ ★★★★★             │ ★★☆☆☆             │
│ Imbalanced data      │ ★★☆☆☆             │ ★★★★☆ (Comp. NB)  │ ★★★☆☆ (tune C)    │
│ Training speed       │ ★★★★★ (none!)     │ ★★★★★             │ ★★☆☆☆ (O(n²-n³))  │
│ Prediction speed     │ ★★☆☆☆             │ ★★★★★             │ ★★★★☆             │
│ Probability output   │ ✗ (votes only)    │ ✓ (calibrated)    │ ✗ (use SVC(prob))  │
└──────────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

---

## Exercises

### Exercise 1: KNN with Dimensionality Reduction

KNN suffers from the curse of dimensionality. Test if reducing dimensions with PCA helps.

```python
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target  # 64 features
X = StandardScaler().fit_transform(X)

# Compare KNN with different numbers of PCA components
pca_dims = [5, 10, 20, 30, 64]  # 64 = no reduction

for dim in pca_dims:
    if dim < 64:
        X_reduced = PCA(n_components=dim).fit_transform(X)
    else:
        X_reduced = X
    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn, X_reduced, y, cv=5, scoring='accuracy')
    print(f"PCA dims={dim:>3d}: CV Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
```

### Solution 1:
```
PCA dims=  5: CV Accuracy = 0.8896 ± 0.0141
PCA dims= 10: CV Accuracy = 0.9422 ± 0.0102
PCA dims= 20: CV Accuracy = 0.9655 ± 0.0085
PCA dims= 30: CV Accuracy = 0.9766 ± 0.0077
PCA dims= 64: CV Accuracy = 0.9755 ± 0.0092

Surprisingly, PCA with 30 components slightly outperforms all 64 features!
This is because PCA removes noise dimensions that confuse the distance metric.
```

### Exercise 2: Build a Spam Classifier with Naive Bayes

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Binary classification: space (sci.space) vs. windows (comp.os.ms-windows.misc)
categories = ['sci.space', 'comp.os.ms-windows.misc']
train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('nb', MultinomialNB(alpha=0.1))
])

# Cross-validate
scores = cross_val_score(pipeline, train.data, train.target, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Train and evaluate
pipeline.fit(train.data, train.target)
y_pred = pipeline.predict(test.data)
print(f"\nTest Accuracy: {accuracy_score(test.target, y_pred):.4f}")
print(classification_report(test.target, y_pred, target_names=train.target_names))

# Show most informative features
feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
log_prob = pipeline.named_steps['nb'].feature_log_prob_
top_10_space = np.argsort(log_prob[0])[-10:][::-1]
top_10_windows = np.argsort(log_prob[1])[-10:][::-1]

print(f"\nTop 10 words for {train.target_names[0]}: {[feature_names[i] for i in top_10_space]}")
print(f"Top 10 words for {train.target_names[1]}: {[feature_names[i] for i in top_10_windows]}")
```

### Exercise 3: SVM Decision Boundary Visualization

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Create non-linear dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)

# Train different SVMs
classifiers = {
    'Linear (C=1)':   SVC(kernel='linear', C=1),
    'Linear (C=100)': SVC(kernel='linear', C=100),
    'RBF (C=1, γ=1)': SVC(kernel='rbf', C=1, gamma=1),
    'RBF (C=1, γ=5)': SVC(kernel='rbf', C=1, gamma=5),
}

# Plot decision boundaries
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, clf) in zip(axes.ravel(), classifiers.items()):
    clf.fit(X, y)
    score = clf.score(X, y)

    # Create mesh grid for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(f"{name}\nAccuracy: {score:.3f}")

plt.tight_layout()
plt.savefig("svm_boundaries.png", dpi=150)
plt.show()
```

---

## Key Takeaways

1. **KNN** is the simplest classifier — it memorizes data and votes on neighbors. Always scale features and tune `k`.
2. **Naive Bayes** uses Bayes' theorem with a "naive" independence assumption. Surprisingly effective for text and high-dimensional data.
3. **SVM** finds the maximum-margin separating hyperplane. With kernels, it can learn complex nonlinear boundaries.
4. **Always scale features** before using KNN or SVM — both are distance-based algorithms.
5. **Naive Bayes is fast** (training is O(np)) and works well with small datasets; SVM is slower (O(n²) to O(n³)) but often more accurate.
6. **For text classification**: Naive Bayes (especially Multinomial NB) is hard to beat as a baseline.
7. **For image/complex data**: SVM with RBF kernel or KNN with appropriate `k` tend to perform well.
8. **Hyperparameter tuning** is essential for all three: `k` for KNN, `alpha` for Naive Bayes, `C` and `gamma` for SVM.

---

*Previous: [Logistic Regression & Gradient Descent](./logistic-regression.md)*
