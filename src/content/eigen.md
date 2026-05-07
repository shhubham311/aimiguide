# Eigenvalues & Eigenvectors

## Why This Matters in AI/ML

Eigenvalues and eigenvectors are among the most important concepts in applied linear algebra and machine learning. **PCA** (Principal Component Analysis) is entirely built on eigendecomposition of the covariance matrix. **Spectral clustering** uses eigenvectors of graph Laplacians. **PageRank** is an eigenvector problem. The **convergence rate of iterative methods** depends on eigenvalue ratios. In deep learning, the **Hessian eigenvalues** determine the loss landscape geometry, affecting optimization. Understanding eigendecomposition unlocks the ability to analyze, compress, and improve ML systems.

---

## 1. Definitions

### 1.1 Core Definition

For a square matrix A ∈ ℝ^(n×n), a scalar λ and a nonzero vector v satisfying:

```
Av = λv
```

are called an **eigenvalue** and its corresponding **eigenvector** of A.

**Interpretation:** The matrix A acts on the eigenvector v by simply scaling it by λ — the direction is unchanged.

```
  Geometric View:

  Before A:         After A:
  
     ^                  ^   Av = 2v
     | v                |   ↗ (stretched by λ=2)
     | ↗                |  /
     |/                 | /
     +------>           +------>
     
  For λ = 1: vector unchanged
  For λ > 1: vector stretched
  For 0 < λ < 1: vector compressed  
  For λ < 0: vector flipped and scaled
```

### 1.2 Characteristic Equation

Rearranging Av = λv:

```
Av - λv = 0
(A - λI)v = 0

For nonzero v to exist: det(A - λI) = 0
```

This is the **characteristic polynomial** — an n-th degree polynomial whose roots are the eigenvalues.

### 1.3 Finding Eigenvalues: Step by Step

**Example:** A = [[4, 1], [2, 3]]

```
Step 1: Form A - λI
A - λI = [[4-λ, 1], [2, 3-λ]]

Step 2: Compute determinant
det(A - λI) = (4-λ)(3-λ) - 2 = λ² - 7λ + 10 = 0

Step 3: Solve characteristic equation
λ² - 7λ + 10 = (λ-5)(λ-2) = 0
λ₁ = 5, λ₂ = 2

Step 4: Find eigenvectors
For λ₁ = 5: (A - 5I)v = 0
  [[-1, 1],[2, -2]]v = 0  →  v₁ = [1, 1]ᵀ

For λ₂ = 2: (A - 2I)v = 0
  [[2, 1],[2, 1]]v = 0  →  v₂ = [1, -2]ᵀ
```

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"\nλ = {lam}: Av = {A @ v}, λv = {lam * v}")
    print(f"  ||Av - λv|| = {np.linalg.norm(A @ v - lam * v):.2e}")
```

---

## 2. Properties of Eigenvalues

### 2.1 Fundamental Properties

```
1. trace(A) = Σ λᵢ              (sum of eigenvalues = trace)
2. det(A) = Π λᵢ                (product of eigenvalues = determinant)
3. A is singular ⇔ some λᵢ = 0
4. Eigenvalues of Aᵏ are λᵢᵏ
5. Eigenvalues of A⁻¹ are 1/λᵢ (if invertible)
6. Eigenvalues of A + cI are λᵢ + c
7. If A is triangular, eigenvalues are the diagonal entries
```

### 2.2 Special Matrices

| Matrix Type | Eigenvalue Properties |
|-------------|----------------------|
| **Symmetric** | All eigenvalues are **real** |
| **Orthogonal** | All eigenvalues have **|λ| = 1** |
| **Positive Definite** | All eigenvalues **> 0** |
| **Positive Semidefinite** | All eigenvalues **≥ 0** |
| **Diagonal** | Eigenvalues = diagonal entries |
| **Nilpotent (Aᵏ = 0)** | All eigenvalues = 0 |
| **Projection** | Eigenvalues are 0 or 1 |

### 2.3 Spectral Theorem

**Theorem:** Every symmetric matrix A ∈ ℝ^(n×n) can be decomposed as:

```
A = QΛQᵀ

where:
Q = orthogonal matrix of eigenvectors (QᵀQ = I)
Λ = diagonal matrix of eigenvalues
```

This is the **eigendecomposition** and is one of the most important results in linear algebra.

```
  Spectral Decomposition Visualization:

  A = QΛQᵀ

  Q (rotation)  Λ (scaling)  Qᵀ (rotation back)
  
    ↻             ↕              ↺
  
  Input → Rotate → Scale along axes → Rotate back → Output

  The eigenvectors define the "principal axes" of the transformation.
```

---

## 3. Diagonalization

### 3.1 Definition

A matrix A is **diagonalizable** if:

```
A = PDP⁻¹

where D = diag(λ₁, ..., λₙ) and P has eigenvectors as columns.
```

### 3.2 When is Diagonalization Possible?

A matrix is diagonalizable if and only if it has **n linearly independent eigenvectors**. This is always true for:
- Symmetric matrices (orthogonal eigenvectors!)
- Matrices with n distinct eigenvalues

```python
import numpy as np

def diagonalize(A):
    """Compute eigendecomposition and verify A = PDP⁻¹."""
    eigenvalues, P = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    
    # Verify: A = P @ D @ P⁻¹
    A_reconstructed = P @ D @ np.linalg.inv(P)
    
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")
    return eigenvalues, P, D

# Example 1: Symmetric matrix (always diagonalizable)
A = np.array([[2, 1], [1, 3]])
print("Symmetric matrix:")
diagonalize(A)

# Example 2: Matrix with repeated eigenvalue
B = np.array([[2, 1], [0, 2]])  # Jordan block
print("\nNon-diagonalizable matrix (Jordan block):")
try:
    eigenvalues, P, D = diagonalize(B)
except Exception as e:
    print(f"Error: {e}")
# Eigenvalue 2 has multiplicity 2 but only 1 eigenvector!
```

---

## 4. Applications in Machine Learning

### 4.1 Principal Component Analysis (PCA)

PCA finds the directions of maximum variance by computing eigenvectors of the covariance matrix.

```
Steps:
1. Center data: X_centered = X - μ
2. Compute covariance: Σ = (1/n) X_centeredᵀ X_centered
3. Eigendecompose: Σ = QΛQᵀ
4. Sort eigenvalues: λ₁ ≥ λ₂ ≥ ... ≥ λₙ
5. Project onto top k eigenvectors: Z = X_centered @ Q[:, :k]
```

```
  PCA Visualization (2D data):

  Original data:          After PCA:
  
      ^ x   x                ^
     x  x   x               |  PC2 (small variance)
    x   x   x   x           |  x x
    x  x    x  x x     →    |   x x x
     x  x  x   x            +---------> PC1 (large variance)
      +---------->            x  x  x x x x
      x₁   x₂                x  x x   x  x
  
  PC1 captures the direction of maximum variance.
  PC2 captures the remaining (perpendicular) variance.
```

```python
import numpy as np

def pca(X, n_components=2):
    """Principal Component Analysis from scratch."""
    # Step 1: Center the data
    X_centered = X - X.mean(axis=0)
    
    # Step 2: Compute covariance matrix
    n = X.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)
    
    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Step 4: Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Project data
    X_pca = X_centered @ eigenvectors[:, :n_components]
    
    # Variance explained
    variance_explained = eigenvalues / eigenvalues.sum()
    
    return X_pca, eigenvalues, eigenvectors, variance_explained

# Generate correlated 2D data
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 3]]  # Correlated features
X = np.random.multivariate_normal(mean, cov, 200)

X_pca, eigvals, eigvecs, var_exp = pca(X, n_components=2)

print(f"Eigenvalues: {eigvals}")
print(f"Variance explained: {var_exp}")
print(f"Eigenvectors:\n{eigvecs}")
print(f"Data shape: {X.shape} → PCA shape: {X_pca.shape}")
```

### 4.2 Spectral Clustering

Uses eigenvectors of the graph Laplacian to cluster data:

```
Steps:
1. Build similarity matrix W (Wᵢⱼ = similarity between points i and j)
2. Compute degree matrix D = diag(Σⱼ Wᵢⱼ)
3. Compute graph Laplacian: L = D - W (or normalized: L_sym = D^(-1/2)LD^(-1/2))
4. Find k smallest eigenvectors of L
5. Cluster the rows using k-means
```

### 4.3 PageRank (Google's Original Algorithm)

PageRank finds the eigenvector corresponding to eigenvalue 1 of the **transition matrix**:

```
r = Mr

where M is the column-stochastic transition matrix and r is the PageRank vector.
```

```python
import numpy as np

def pagerank(links, damping=0.85, tol=1e-10, max_iter=1000):
    """
    Compute PageRank using power iteration.
    links: dict mapping page -> list of linked pages
    """
    pages = sorted(links.keys())
    n = len(pages)
    page_idx = {p: i for i, p in enumerate(pages)}
    
    # Build transition matrix
    M = np.zeros((n, n))
    for page, linked_pages in links.items():
        if len(linked_pages) == 0:
            # Dangling node: distribute equally
            M[:, page_idx[page]] = 1.0 / n
        else:
            for target in linked_pages:
                M[page_idx[target], page_idx[page]] = 1.0 / len(linked_pages)
    
    # Add damping
    M = damping * M + (1 - damping) / n * np.ones((n, n))
    
    # Power iteration to find dominant eigenvector
    r = np.ones(n) / n
    for _ in range(max_iter):
        r_new = M @ r
        r_new = r_new / r_new.sum()
        if np.linalg.norm(r_new - r) < tol:
            return r_new, pages
        r = r_new
    return r, pages

# Simple web graph
links = {
    'A': ['B', 'C'],
    'B': ['C'],
    'C': ['A'],
    'D': ['C']
}

scores, pages = pagerank(links)
print("PageRank scores:")
for page, score in sorted(zip(pages, scores), key=lambda x: -x[1]):
    print(f"  {page}: {score:.4f}")
```

### 4.4 Hessian Eigenvalues in Deep Learning

The eigenvalues of the Hessian matrix of the loss function determine:

```
- Condition number κ = λ_max / λ_min:
  High κ → slow convergence, need adaptive optimizers
  
- Eigenvalue spectrum:
  All positive → local minimum
  Mixed signs  → saddle point
  Negative     → local maximum

- Gradient noise proportional to ‖H‖ = λ_max:
  Large eigenvalues → training instability
```

---

## 5. Computing Eigenvalues: Power Iteration

### 5.1 Power Iteration (for dominant eigenvalue)

```
Algorithm:
1. Start with random v₀
2. v_{k+1} = A v_k / ‖A v_k‖
3. Repeat until convergence
4. λ = vᵀAv (Rayleigh quotient)
```

```python
import numpy as np

def power_iteration(A, max_iter=1000, tol=1e-10):
    """Find the dominant eigenvalue and eigenvector."""
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(max_iter):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    
    # Rayleigh quotient for eigenvalue
    eigenvalue = v @ A @ v
    return eigenvalue, v

# Example
A = np.array([[4, 1],
              [2, 3]])

eigenvalue, eigenvector = power_iteration(A)
print(f"Power iteration: λ = {eigenvalue:.6f}")
print(f"Eigenvector: {eigenvector}")
print(f"NumPy:        λ = {np.linalg.eigvals(A)[0]:.6f}")
```

### 5.2 Inverse Iteration (for smallest eigenvalue)

Replace A with A⁻¹ in power iteration to find the smallest eigenvalue.

### 5.3 QR Algorithm (complete eigendecomposition)

The standard algorithm used by NumPy/MATLAB:

```
Algorithm:
1. Compute QR decomposition: A = QR
2. Update: A ← RQ (= QᵀAQ)
3. Repeat until A converges to upper triangular
4. Diagonal entries are eigenvalues
```

---

## 6. Generalized Eigenvalue Problem

```
Ax = λBx

where A and B are both symmetric and B is positive definite.

Solution: B⁻¹Ax = λx (reduce to standard problem)
Or: B^(-1/2) A B^(-1/2) y = λy where x = B^(-1/2) y
```

**ML Application:** **Linear Discriminant Analysis (LDA)** maximizes the ratio:
```
J(w) = (wᵀS_B w) / (wᵀS_W w)
```
This is a generalized eigenvalue problem: S_B w = λ S_W w.

---

## 7. Exercises with Solutions

### Exercise 1
Find the eigenvalues and eigenvectors of A = [[3, 0], [0, 7]].

**Solution:** Since A is diagonal, the eigenvalues are the diagonal entries: λ₁ = 3, λ₂ = 7.

```
For λ = 3: (A - 3I)v = 0 → [[0,0],[0,4]]v = 0 → v = [1, 0]ᵀ
For λ = 7: (A - 7I)v = 0 → [[-4,0],[0,0]]v = 0 → v = [0, 1]ᵀ
```

### Exercise 2
Show that if v is an eigenvector of A with eigenvalue λ, then v is also an eigenvector of A² with eigenvalue λ².

**Solution:**
```
A²v = A(Av) = A(λv) = λ(Av) = λ(λv) = λ²v ✓
```

### Exercise 3
A 2×2 matrix A has trace 8 and determinant 15. Find its eigenvalues.

**Solution:**
```
trace(A) = λ₁ + λ₂ = 8
det(A) = λ₁ · λ₂ = 15

Solving: λ² - 8λ + 15 = 0
(λ - 3)(λ - 5) = 0

λ₁ = 3, λ₂ = 5
```

### Exercise 4
For the covariance matrix Σ = [[4, 2], [2, 3]], find the principal component direction.

**Solution:**
```
det(Σ - λI) = (4-λ)(3-λ) - 4 = λ² - 7λ + 8 = 0
λ = (7 ± √17)/2
λ₁ = (7 + √17)/2 ≈ 5.562  (principal eigenvalue)
λ₂ = (7 - √17)/2 ≈ 1.438

For λ₁ ≈ 5.562:
(Σ - λ₁I)v = 0
[4-5.562, 2; 2, 3-5.562][v₁; v₂] = 0
[-1.562, 2; 2, -2.562][v₁; v₂] = 0
v₁ ≈ [2, 1.562]ᵀ → normalized: [0.788, 0.616]ᵀ

The first principal component direction is approximately [0.788, 0.616]ᵀ.
This direction captures 5.562/7 ≈ 79.5% of total variance.
```

---

## Summary

| Concept | Formula | ML Application |
|---------|---------|----------------|
| **Eigenvalue** | det(A - λI) = 0 | Solving characteristic equation |
| **Eigenvector** | (A - λI)v = 0 | Direction of no rotation |
| **Spectral Theorem** | A = QΛQᵀ (symmetric A) | Foundation of PCA |
| **PCA** | Σv = λv | Dimensionality reduction |
| **Trace** | tr(A) = Σλᵢ | Total variance |
| **Determinant** | det(A) = Πλᵢ | Invertibility check |
| **Power Iteration** | v ← Av/‖Av‖ | Finding dominant eigenvalue |
| **PageRank** | r = Mr | Dominant eigenvector of link matrix |
