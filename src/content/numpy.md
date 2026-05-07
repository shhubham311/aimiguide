# NumPy & Implementing Basic Algorithms

## Why NumPy Matters in AI/ML

NumPy is the computational backbone of every Python ML framework. When you call `torch.tensor()` in PyTorch or `pd.DataFrame` in pandas, you are ultimately working with NumPy-compatible array interfaces. The entire field of scientific computing in Python stands on three letters: **N**umerical **Py**thon.

The reason is **vectorization**: instead of looping over data element-by-element in Python (slow), NumPy pushes operations down to optimized C code that processes entire arrays at once (100x faster). A matrix multiplication that takes 10 seconds with Python lists takes 0.1 seconds with NumPy.

Understanding NumPy means understanding:
- How data is stored and accessed in memory
- How linear algebra operations map to array operations
- How to implement ML algorithms from scratch

---

## 1. Creating Arrays

```python
import numpy as np

# From Python lists
a = np.array([1, 2, 3, 4, 5])
print(type(a))          # <class 'numpy.ndarray'>
print(a.dtype)          # int64
print(a.shape)          # (5,)
print(a.ndim)           # 1

# Multi-dimensional arrays
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matrix.shape)     # (2, 3)
print(matrix.ndim)      # 2
print(matrix.size)      # 6

# Special arrays — frequently used in ML
zeros = np.zeros((3, 4))           # All zeros — weight initialization
ones = np.ones((2, 3))             # All ones — bias initialization
empty = np.empty((2, 2))           # Uninitialized — fast but random values
identity = np.eye(4)               # Identity matrix — used in PCA, linear regression
random = np.random.randn(3, 3)     # Standard normal — weight initialization (He/Xavier)

# Ranges and sequences
x = np.arange(0, 1, 0.1)          # [0.0, 0.1, 0.2, ..., 0.9]
x = np.linspace(0, 10, 50)        # 50 evenly spaced points from 0 to 10

# Reshaping — crucial for ML (images, sequences)
data = np.arange(24)
data_3d = data.reshape(2, 3, 4)   # 2 batches, 3 channels, 4 features
data_2d = data.reshape(6, -1)      # -1: NumPy infers the dimension
```

### Array Memory Layout

```
np.array([[1, 2, 3],
          [4, 5, 6]])

Memory (C-order, row-major):  [1, 2, 3, 4, 5, 6]  ← contiguous in memory
Strides: (24 bytes, 8 bytes)    3 elements * 8 bytes per float64

This contiguous layout is what makes NumPy fast:
- Cache-friendly access patterns
- SIMD (Single Instruction, Multiple Data) operations
- No Python object overhead per element
```

---

## 2. Array Operations — Vectorization

Vectorization replaces Python loops with array operations. This is the single most important NumPy concept.

```python
# SLOW: Python loop
import time
data = np.random.randn(1_000_000)

start = time.time()
result_loop = [x * 2 + 1 for x in data]
print(f"Loop: {time.time() - start:.4f}s")       # ~120ms

# FAST: NumPy vectorized
start = time.time()
result_vec = data * 2 + 1
print(f"Vectorized: {time.time() - start:.4f}s")  # ~2ms (60x faster)
```

### Element-wise operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Arithmetic
print(a + b)     # [11, 22, 33, 40]
print(a * b)     # [10, 40, 90, 160]  (element-wise, NOT matrix multiply!)
print(a ** 2)    # [1, 4, 9, 16]

# Mathematical functions (ufuncs — universal functions)
print(np.exp(a))     # e^x
print(np.log(b))     # natural log
print(np.sqrt(a))    # square root
print(np.abs(a))     # absolute value

# Comparison operations (return boolean arrays)
print(a > 2)                    # [False, False, True, True]
print(np.where(a > 2, a, 0))   # [0, 0, 3, 4] — conditional selection

# Statistical operations
print(a.mean())    # 2.5
print(a.std())     # 1.118...
print(a.sum())     # 10
print(np.percentile(a, 50))  # median = 2.5
```

---

## 3. Broadcasting

Broadcasting lets NumPy perform operations on arrays of different shapes without explicit looping. It is how you add a bias vector to a batch of predictions, or normalize all features by their column means.

### Broadcasting Rules

```
Rule 1: If arrays have different ndim, prepend 1s to the shape of the smaller array.
Rule 2: Arrays with size 1 along a dimension act as if they had the size of the other array.
Rule 3: Dimensions must be equal or one of them must be 1 (otherwise error).

Example: (4, 3) + (3,)
  Step 1: (3,) → (1, 3)        [prepend 1]
  Step 2: (1, 3) → (4, 3)       [broadcast along axis 0]
  Result:  (4, 3) operation

Example: (4, 1) + (1, 3)
  Step 1: Already compatible
  Step 2: (4, 1) → (4, 3)       [broadcast along axis 1]
          (1, 3) → (4, 3)       [broadcast along axis 0]
  Result:  (4, 3) operation
```

```python
# Batch normalization (add bias to each sample)
X = np.random.randn(100, 5)      # 100 samples, 5 features
bias = np.array([0.1, -0.2, 0.3, -0.1, 0.0])  # shape (5,)
X_biased = X + bias              # (100, 5) + (5,) → (100, 5)

# Feature-wise scaling
feature_means = X.mean(axis=0)   # shape (5,) — mean of each column
feature_stds = X.std(axis=0)     # shape (5,)
X_normalized = (X - feature_means) / feature_stds  # Z-score normalization

# Outer product via broadcasting
v = np.array([1, 2, 3]).reshape(3, 1)  # column vector: shape (3, 1)
w = np.array([10, 20, 30])             # row vector:    shape (3,)
print(v * w)  # shape (3, 3)
# [[10, 20, 30],
#  [20, 40, 60],
#  [30, 60, 90]]
```

---

## 4. Linear Algebra with NumPy

Linear algebra is the mathematics of ML. Every model — from linear regression to transformers — is built on matrix operations.

### Matrix Operations

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Element-wise vs matrix multiplication
print(A * B)          # Element-wise (Hadamard product)
# [[ 5, 12],
#  [21, 32]]

print(A @ B)          # Matrix multiplication (dot product)
# [[19, 22],
#  [43, 50]]

print(np.dot(A, B))   # Same as @ for 2D arrays

# Transpose
print(A.T)
# [[1, 3],
#  [2, 4]]

# Matrix inverse and determinant
A_inv = np.linalg.inv(A)
print(A @ A_inv)      # Should be ~identity matrix
print(np.linalg.det(A))  # -2.0

# Solve linear systems: Ax = b
b = np.array([5, 6])
x = np.linalg.solve(A, b)  # x = A^(-1) * b
print(x)  # [-4.  4.5]

# Eigenvalues and eigenvectors (PCA, spectral clustering)
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### Linear Algebra in ML Context

```
Linear Regression:  y = Xw + b
  X: (n_samples, n_features)  — the data matrix
  w: (n_features,)             — the weight vector
  b: (n_samples,)              — the bias (broadcast)

Forward pass of a neural network layer:
  h = X @ W + b
  X: (batch_size, input_dim)
  W: (input_dim, hidden_dim)
  b: (hidden_dim,)
  h: (batch_size, hidden_dim)

Principal Component Analysis:
  X_centered = X - X.mean(axis=0)
  covariance = (X_centered.T @ X_centered) / (n - 1)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance)
  X_pca = X_centered @ eigenvectors[:, :k]
```

---

## 5. Indexing, Slicing, and Advanced Selection

```python
X = np.array([[10, 20, 30, 40],
              [50, 60, 70, 80],
              [90, 100, 110, 120]])

# Basic indexing — row, column
print(X[0, 1])      # 20 (row 0, column 1)
print(X[1, :])      # [50, 60, 70, 80] (entire row 1)
print(X[:, 2])      # [30, 70, 110] (entire column 2)

# Slicing
print(X[0:2, 1:3])  # [[20, 30], [60, 70]]
print(X[::-1])      # Reverse rows

# Boolean indexing — EXTREMELY useful in ML
data = np.random.randn(1000, 5)
print(data[data[:, 0] > 0].shape)     # Rows where first column > 0
print(data[(data[:, 0] > 0) & (data[:, 1] < 0)].shape)  # Compound conditions

# Fancy indexing — select specific rows/columns
print(data[[0, 5, 10, 15]])           # Specific rows
print(data[:, [0, 2, 4]])             # Specific columns

# np.where — conditional replacement
labels = np.array([0, 1, 0, 1, 1, 0, 1])
class_names = np.where(labels == 1, "positive", "negative")
# ['negative', 'positive', 'negative', 'positive', 'positive', 'negative', 'positive']

# np.argsort — get sorted indices (used for ranking features)
importances = np.array([0.35, 0.12, 0.08, 0.25, 0.20])
sorted_idx = np.argsort(importances)[::-1]  # Descending
print(importances[sorted_idx])  # [0.35, 0.25, 0.20, 0.12, 0.08]
```

---

## 6. Implementing Algorithms from Scratch

### 6.1 Linear Regression with Gradient Descent

The fundamental ML algorithm, implemented purely with NumPy:

```python
class LinearRegressionNumPy:
    """
    Linear Regression: y = Xw + b
    Loss: MSE = (1/2n) * ||Xw + b - y||^2
    Gradient: dw = (1/n) * X.T @ (Xw + b - y)
               db = (1/n) * sum(Xw + b - y)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient descent
        for i in range(self.n_iter):
            # Forward pass (vectorized)
            y_pred = X @ self.weights + self.bias

            # Compute loss
            error = y_pred - y
            mse = np.mean(error ** 2) / 2
            self.loss_history.append(mse)

            # Compute gradients (vectorized)
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and (i + 1) % 100 == 0:
                print(f"  Epoch {i+1}/{self.n_iter}, Loss: {mse:.6f}")

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def r_squared(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot


# Demo with real data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)

# Split
split = 400
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train
model = LinearRegressionNumPy(learning_rate=0.01, n_iterations=1000, verbose=True)
model.fit(X_train, y_train)

# Evaluate
r2 = model.r_squared(X_test, y_test)
print(f"\nR² Score: {r2:.4f}")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")
```

### 6.2 K-Nearest Neighbors from Scratch

```python
class KNNNumPy:
    """k-NN classifier using NumPy vectorized operations."""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        # Compute all pairwise distances at once (vectorized)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)       # (n_test, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1)       # (n_train,)
        cross = X @ self.X_train.T                          # (n_test, n_train)

        distances = np.sqrt(
            X_sq + train_sq - 2 * cross + 1e-10  # epsilon for numerical stability
        )

        # Get k nearest neighbors for each test point
        k_nearest_idx = np.argpartition(distances, self.k, axis=1)[:, :self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_nearest_idx]  # (n_test, k)

        # Majority vote
        predictions = []
        for labels in k_nearest_labels:
            unique, counts = np.unique(labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])

        return np.array(predictions)


# Demo
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNNNumPy(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"k-NN Accuracy: {accuracy:.2%}")
```

### 6.3 K-Means Clustering from Scratch

```python
class KMeansNumPy:
    """
    K-Means clustering using NumPy.
    Objective: Minimize within-cluster sum of squares.
    """

    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize centroids using k-means++ strategy
        self.centroids = np.empty((self.k, n_features))
        self.centroids[0] = X[self.rng.randint(n_samples)]

        for i in range(1, self.k):
            # Compute distances to nearest existing centroid
            distances = np.min(
                np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :i, :]) ** 2, axis=2),
                axis=1
            )
            # Probability proportional to distance squared
            probs = distances / distances.sum()
            self.centroids[i] = X[self.rng.choice(n_samples, p=probs)]

        # Iterative refinement
        for iteration in range(self.max_iter):
            # Assignment step: assign each point to nearest centroid
            # Using broadcasting: (n_samples, 1, n_features) - (1, k, n_features)
            distances = np.sum(
                (X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2,
                axis=2
            )  # shape: (n_samples, k)
            self.labels_ = np.argmin(distances, axis=1)

            # Update step: recompute centroids
            new_centroids = np.array([
                X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i)
                else self.centroids[i]
                for i in range(self.k)
            ])

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        # Compute inertia (within-cluster sum of squares)
        self.inertia_ = sum(
            np.sum((X[self.labels_ == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        )

        return self

    def predict(self, X):
        distances = np.sum(
            (X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )
        return np.argmin(distances, axis=1)


# Demo
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)

kmeans = KMeansNumPy(n_clusters=3)
kmeans.fit(X)

print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Centroids:\n{kmeans.centroids}")

# Compare with sklearn
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y_true, kmeans.labels_)
print(f"Adjusted Rand Index: {ari:.4f}")
```

---

## 7. Performance: NumPy vs Pure Python

```
Benchmark: Matrix multiplication of (500, 500) matrices

Python nested loops:      ~12,000 ms
NumPy (np.dot / @):          ~1.5 ms
Speedup:                     ~8,000x
```

```
Benchmark: Element-wise operations on 1,000,000 elements

Python list comprehension:    ~85 ms
NumPy vectorized:              ~1.5 ms
Speedup:                       ~57x
```

The lesson is clear: **never loop over NumPy arrays in Python**. Always find a vectorized equivalent.

---

## 8. Common Patterns in ML

```python
# One-hot encoding
n_classes = 3
labels = np.array([0, 1, 2, 1, 0, 2])
one_hot = np.eye(n_classes)[labels]
# [[1,0,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0], [0,0,1]]

# Softmax function
def softmax(logits):
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=-1, keepdims=True)

scores = np.array([[2.0, 1.0, 0.1]])
print(softmax(scores))  # [[0.659, 0.242, 0.099]]

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip to avoid overflow

# Cross-entropy loss
def cross_entropy(probs, targets):
    """Numerically stable cross-entropy."""
    n = len(targets)
    log_probs = -np.log(probs[np.arange(n), targets] + 1e-12)
    return np.mean(log_probs)

# Batch normalization
def batch_normalize(X, eps=1e-8):
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    return (X - mean) / np.sqrt(var + eps), mean, var
```

---

## 9. Exercises

### Exercise 1: Implement PCA from Scratch

```python
def pca(X, n_components=2):
    """Principal Component Analysis from scratch using NumPy."""
    # Step 1: Center the data
    X_centered = X - X.mean(axis=0)

    # Step 2: Compute covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort by descending eigenvalue
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Step 5: Project onto top components
    W = eigenvectors[:, :n_components]
    X_pca = X_centered @ W

    # Variance explained
    variance_explained = eigenvalues[:n_components] / eigenvalues.sum()

    return X_pca, variance_explained

# Test
from sklearn.datasets import load_iris
iris = load_iris()
X_pca, var_exp = pca(iris.data, n_components=2)
print(f"Variance explained: {var_exp}")
print(f"Shape: {X_pca.shape}")  # (150, 2)
```

### Exercise 2: Implement a Neural Network Layer

Create a `DenseLayer` class with forward and backward pass using NumPy matrix operations.

### Exercise 3: Image Processing

Load an image as a NumPy array, convert to grayscale using the formula: `gray = 0.299*R + 0.587*G + 0.114*B`, apply a threshold, and save.

---

## Connections to Other ML Topics

| NumPy Concept | ML Application |
|--------------|----------------|
| `@` operator | Neural network forward passes, linear models |
| Broadcasting | Adding biases, batch normalization |
| `np.linalg` | PCA, linear regression, SVD, eigenvalues |
| Boolean indexing | Filtering data, masking inattention |
| `np.where` | Thresholding, conditional transforms |
| `np.unique` | Computing class distributions, metrics |
| Reshape/transpose | Reshaping images, sequence-to-matrix conversion |

---

## Key Takeaways

1. **Vectorization is everything** — replace Python loops with array operations for 50-8000x speedup.
2. **Broadcasting** eliminates explicit loops when combining arrays of different shapes.
3. **`@` (matmul)** is the core operation for neural networks and linear models.
4. **`np.linalg`** provides all the linear algebra needed for PCA, regression, and SVD.
5. **Implementing algorithms from scratch** (linear regression, k-NN, k-means) builds deep understanding.
6. **NumPy arrays** are the universal data interchange format between all ML libraries.
