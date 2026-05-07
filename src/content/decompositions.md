# SVD & LU Decomposition — Complete Guide

## Table of Contents
1. [Introduction to Matrix Decomposition](#introduction)
2. [Singular Value Decomposition (SVD)](#svd)
3. [SVD Applications](#svd-applications)
4. [LU Decomposition](#lu-decomposition)
5. [LU Decomposition Applications](#lu-applications)
6. [Python Implementations](#python)
7. [ML Connections](#ml-connections)
8. [Exercises](#exercises)

---

<a name="introduction"></a>
## 1. Introduction to Matrix Decomposition

Matrix decomposition (factorization) is the process of breaking down a matrix into a product of simpler matrices. This is one of the most fundamental tools in linear algebra with far-reaching applications in machine learning, data science, and engineering.

### Why Decompose Matrices?

```
Original Matrix A
       │
       ├─── SVD  ───►  U · Σ · Vᵀ       (dimensionality reduction, compression)
       ├─── LU   ───►  L · U            (solving linear systems)
       ├─── QR   ───►  Q · R            (least squares, eigenvalues)
       ├─── Chol ───►  L · Lᵀ           (positive definite systems)
       └─── Eig  ───►  Q · Λ · Q⁻¹     (diagonalization, PCA)
```

### When to Use Which Decomposition?

| Decomposition | Matrix Type | Primary Use | Complexity |
|--------------|-------------|-------------|------------|
| SVD | Any m×n | Compression, PCA, Recommenders | O(mn²) |
| LU | Square n×n | Solving Ax=b | O(n³/3) |
| QR | Any m×n | Least squares | O(mn²) |
| Cholesky | Symmetric PD | Fast solving | O(n³/6) |
| Eigen | Square n×n | Diagonalization | O(n³) |

---

<a name="svd"></a>
## 2. Singular Value Decomposition (SVD)

### 2.1 The Fundamental Theorem

**Every matrix A ∈ ℝ^(m×n) can be factored as:**

```
         A = U · Σ · Vᵀ

Where:
  - A is m × n  (the original matrix)
  - U is m × m  (left singular vectors — orthogonal)
  - Σ is m × n  (diagonal matrix of singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
  - Vᵀ is n × n (right singular vectors — orthogonal)

Properties:
  - UᵀU = Iₘ   (columns of U are orthonormal)
  - VᵀV = Iₙ   (columns of V are orthonormal)
  - σᵢ ≥ 0     (singular values are non-negative)
  - rank(A) = number of non-zero singular values
```

### 2.2 Geometric Interpretation

SVD decomposes any linear transformation into three simple steps:

```
   Input space ℝⁿ         Hidden space ℝʳ          Output space ℝᵐ
  ┌──────────┐          ┌──────────┐            ┌──────────┐
  │    x     │  ──Vᵀ──► │   Vᵀx    │  ──Σ───►  │   ΣVᵀx  │  ──U──►  Ax
  │  (n×1)   │          │  (n×1)   │            │  (r×1)   │          (m×1)
  └──────────┘          └──────────┘            └──────────┘          └─────┘
  
  Step 1: Vᵀ rotates   Step 2: Σ scales       Step 3: U rotates
  into the basis of     each coordinate by     into the output
  right singular         its singular value     coordinate system
  vectors
```

### 2.3 Relationship to Eigenvalues

```
                    AᵀA = V Σᵀ Σ Vᵀ = V (Σ²) Vᵀ
                    AAᵀ = U Σ Σᵀ Uᵀ = U (Σ²) Uᵀ

The singular values of A are the square roots of the eigenvalues of AᵀA:

  σᵢ = √(λᵢ(AᵀA))    for i = 1, 2, ..., min(m, n)

The columns of V are eigenvectors of AᵀA.
The columns of U are eigenvectors of AAᵀ.
```

### 2.4 Compact and Truncated SVD

```
Full SVD (m × n):
┌─────┐   ┌─────────────┐   ┌─────┐
│  U  │ · │  Σ (m × n)  │ · │  Vᵀ │
│m × m│   │             │   │n × n│
└─────┘   └─────────────┘   └─────┘

Compact SVD (keep only r = rank(A) non-zero singular values):
┌─────┐   ┌────────┐   ┌─────┐
│ Û   │ · │ Σ̂ (r×r)│ · │ V̂ᵀ  │
│m × r│   │        │   │r × n│
└─────┘   └────────┘   └─────┘

Truncated SVD (keep top k < r singular values):
┌─────┐   ┌────────┐   ┌─────┐
│ Û_k │ · │ Σ̂_k    │ · │ V̂ᵀ_k│
│m × k│   │  (k×k) │   │k × n│
└─────┘   └────────┘   └─────┘

Low-rank approximation:
  A_k = U_k · Σ_k · V_kᵀ   (best rank-k approximation in Frobenius norm)
```

### 2.5 Key Properties

```
Frobenius Norm from SVD:
  ||A||_F = √(σ₁² + σ₂² + ... + σᵣ²)

Spectral (Operator) Norm:
  ||A||₂ = σ₁  (largest singular value)

Nuclear Norm:
  ||A||_* = σ₁ + σ₂ + ... + σᵣ  (sum of singular values)

Condition Number:
  κ(A) = σ₁ / σᵣ  (ratio of largest to smallest singular value)

Pseudo-Inverse (Moore-Penrose):
  A⁺ = V Σ⁺ Uᵀ  where Σ⁺ replaces σᵢ with 1/σᵢ for non-zero entries

Eckart-Young Theorem:
  A_k = argmin_{rank(B)=k} ||A - B||_F
  
  The error of rank-k approximation:
  ||A - A_k||_F = √(σ_{k+1}² + ... + σᵣ²)
```

---

<a name="svd-applications"></a>
## 3. SVD Applications

### 3.1 Principal Component Analysis (PCA) via SVD

```
Data Matrix: X ∈ ℝ^(n × d)  (n samples, d features)
Step 1: Center the data
  X_centered = X - μ  (subtract mean of each feature)

Step 2: Compute SVD
  X_centered = U Σ Vᵀ

Step 3: Principal components are columns of V
  - PC₁ = v₁  (direction of maximum variance)
  - PC₂ = v₂  (direction of next maximum variance, orthogonal to PC₁)
  - ...

Step 4: Project data onto top k components
  Z = X_centered · V_k  where V_k = [v₁ | v₂ | ... | v_k]

Variance explained by component i:
  Var(PCᵢ) = σᵢ² / (n - 1)

Total variance explained ratio (top k components):
  R² = (σ₁² + σ₂² + ... + σₖ²) / (σ₁² + σ₂² + ... + σᵣ²)

                Original Space (d dimensions)
                ┌───────────────────────┐
                │  ●  ●     ●           │
                │    ●  ●     ●         │
                │  ●    ●       ●       │────── SVD ────► PC₁ (max variance)
                │     ●   ●  ●    ●     │                PC₂ (2nd max variance)
                │  ●     ●     ●   ●    │
                └───────────────────────┘
```

### 3.2 Recommender Systems (Matrix Factorization)

```
User-Item Rating Matrix R (m users × n items):

           Item₁  Item₂  Item₃  Item₄
  User₁  [  5      ?      3      ?  ]
  User₂  [  ?      4      ?      2  ]
  User₃  [  3      ?      5      4  ]
  User₄  [  ?      3      4      ?  ]

SVD Factorization:
  R ≈ U · Σ · Vᵀ
  
  U (m × k): User latent factors
  Σ (k × k): Strength of each latent factor
  Vᵀ (k × n): Item latent factors

Interpretation of latent factors:
  Factor 1: "Action vs. Romance" preference
  Factor 2: "Old vs. New" preference
  Factor 3: "Critical vs. Casual" preference
  
  Each user u gets a vector:  u⃗ = U[u, :]  (user embedding)
  Each item i gets a vector:  v⃗ = V[i, :]  (item embedding)
  
  Predicted rating:  r̂(u,i) = u⃗ · v⃗
```

### 3.3 Image Compression

```
Original image as matrix A (m × n pixels):

  Full SVD: m×m + m×n + n×n = m² + mn + n² total values
  
  Rank-k approximation:
  k×m + k + k×n = k(m + 1 + n) values to store

  Compression ratio = (m + 1 + n)k / (mn)

Example: 1000 × 1000 image with k=50
  Original:  1,000,000 values
  Compressed: 50 × (1000 + 1 + 1000) = 100,050 values
  Ratio: ~10:1 compression

Visual quality:
  k = 10   ████░░░░░░  (blurry, major features only)
  k = 30   ██████░░░░  (good quality, minor artifacts)
  k = 50   ████████░░  (very good quality)
  k = 100  █████████░  (excellent quality)
  k = 200  ██████████  (near-perfect)
```

### 3.4 LSA (Latent Semantic Analysis) for NLP

```
Term-Document Matrix A (vocabulary × documents):

            Doc₁  Doc₂  Doc₃  Doc₄  Doc₅
  cat      [  2     0     1     0     0  ]
  dog      [  1     3     0     1     0  ]
  pet      [  1     1     0     1     0  ]
  car      [  0     0     2     0     3  ]
  engine   [  0     0     1     0     2  ]

After SVD truncation to k=2:
  U₂ (vocab × 2) captures "topic embeddings" for words
  V₂ (2 × docs)  captures "topic proportions" for documents
  
  Topic 1 ≈ "animals/pets"
  Topic 2 ≈ "automobiles"

Words similar in meaning map to nearby points in the latent space.
```

### 3.5 Noise Reduction

```
If signal matrix A = A_signal + A_noise where:
  - A_signal has rank r_signal (low)
  - A_noise spreads across all singular values

Then truncated SVD with k ≈ r_signal preserves the signal:

  Singular Value Spectrum:
  σ  │  ■■
     │  ■■ ■■
     │  ■■ ■■ ░░░░░░░░░░░░░░░░  ← noise floor
     │  ■■ ■■ ░░░░░░░░░░░░░░░░
     └──┴──┴──┴──┴──┴──┴──┴──┴──►  index
       1  2  3  4  5  6  ...  r

  Keep σ₁ through σ_k (above the "elbow")
  Discard σ_{k+1} through σᵣ (noise)
```

---

<a name="lu-decomposition"></a>
## 4. LU Decomposition

### 4.1 Definition

**For a square matrix A ∈ ℝ^(n×n), LU decomposition factors it as:**

```
  A = L · U

Where:
  - L is lower triangular with 1's on the diagonal (unit lower triangular)
  - U is upper triangular

Visual representation for n = 4:

  ┌                   ┐   ┌                   ┐   ┌                   ┐
  │ a₁₁  a₁₂  a₁₃  a₁₄│   │  1    0    0    0  │   │ u₁₁  u₁₂  u₁₃  u₁₄│
  │                   │   │                   │   │                   │
  │ a₂₁  a₂₂  a₂₃  a₂₄│ = │ l₂₁   1    0    0  │ × │  0   u₂₂  u₂₃  u₂₄│
  │                   │   │                   │   │                   │
  │ a₃₁  a₃₂  a₃₃  a₃₄│   │ l₃₁  l₃₂   1    0  │   │  0    0   u₃₃  u₃₄│
  │                   │   │                   │   │                   │
  │ a₄₁  a₄₂  a₄₃  a₄₄│   │ l₄₁  l₄₂  l₄₃   1  │   │  0    0    0   u₄₄│
  └                   ┘   └                   ┘   └                   ┘
```

### 4.2 Existence and Uniqueness

```
LU decomposition exists and is unique if and only if all leading principal 
minors are non-zero.

Leading principal minor of order k:
  Mₖ = det(A[1:k, 1:k])  ≠ 0  for all k = 1, 2, ..., n

  ┌───────┐
  │ a₁₁   │  ← M₁ = a₁₁ ≠ 0
  │ a₂₁ a₂₂│ ← M₂ = a₁₁a₂₂ - a₁₂a₂₁ ≠ 0
  │ ...    │  ← M₃ ≠ 0
  │       │  
  └───────┘

If A is singular or has a zero leading principal minor, use pivoting:

  PA = LU    (where P is a permutation matrix)

Partial pivoting: at each step, swap rows to put the largest element
on the diagonal. This improves numerical stability.
```

### 4.3 Doolittle's Algorithm

```
Doolittle's method computes L and U directly:

For each column j = 1, 2, ..., n:

  Upper triangular (U):
    For each row i = 1, 2, ..., j:
      uᵢⱼ = aᵢⱼ - Σ(k=1 to i-1) lᵢₖ · uₖⱼ

  Lower triangular (L):
    For each row i = j+1, j+2, ..., n:
      lᵢⱼ = (aᵢⱼ - Σ(k=1 to j-1) lᵢₖ · uₖⱼ) / uⱼⱼ

Example (3×3):

  A = ┌         ┐
      │ 2  1  1 │
      │ 4  3  3 │
      │ 8  7  9 │
      └         ┘

  Step 1 (j=1):
    u₁₁ = 2,  u₁₂ = 1,  u₁₃ = 1
    l₂₁ = 4/2 = 2
    l₃₁ = 8/2 = 4

  Step 2 (j=2):
    u₂₂ = 3 - 2(1) = 1
    u₂₃ = 3 - 2(1) = 1
    l₃₂ = (7 - 4(1)) / 1 = 3

  Step 3 (j=3):
    u₃₃ = 9 - 4(1) - 3(1) = 2

  Result:
    L = ┌         ┐      U = ┌         ┐
        │ 1  0  0 │          │ 2  1  1 │
        │ 2  1  0 │          │ 0  1  1 │
        │ 4  3  1 │          │ 0  0  2 │
        └         ┘          └         ┘

  Verification: L × U = A ✓
```

### 4.4 Solving Linear Systems with LU

```
Given: A·x = b

Step 1: Factor  A = L·U

Step 2: Solve L·y = b   (forward substitution)
  y₁ = b₁
  yᵢ = bᵢ - Σ(j=1 to i-1) lᵢⱼ · yⱼ    for i = 2, ..., n

Step 3: Solve U·x = y   (backward substitution)
  xₙ = yₙ / uₙₙ
  xᵢ = (yᵢ - Σ(j=i+1 to n) uᵢⱼ · xⱼ) / uᵢᵢ   for i = n-1, ..., 1

Complexity analysis:
  - LU factorization:    O(n³/3)
  - Forward substitution: O(n²)
  - Backward substitution: O(n²)
  
  Total for first system: O(n³)
  For subsequent systems with same A, different b: only O(n²)!
  
  This is the key advantage over Gaussian elimination for multiple RHS.
```

### 4.5 Computing the Determinant

```
det(A) = det(L) · det(U)

det(L) = 1 (product of diagonal, which are all 1)
det(U) = u₁₁ · u₂₂ · ... · uₙₙ (product of diagonal)

Therefore:
  det(A) = Π(i=1 to n) uᵢᵢ

For PA = LU:
  det(A) = (-1)^(number of row swaps) · Π(i=1 to n) uᵢᵢ

Complexity: O(n³) to factor, then O(n) for the product.
Much faster than cofactor expansion which is O(n!).
```

### 4.6 Computing the Matrix Inverse

```
To find A⁻¹, solve A · X = I column by column:

  A · [x₁ | x₂ | ... | xₙ] = [e₁ | e₂ | ... | eₙ]

For each column j:
  1. Forward substitution:  L · yⱼ = eⱼ
  2. Backward substitution: U · xⱼ = yⱼ

Total cost: O(n³) factorization + n × O(n²) = O(n³)
```

---

<a name="lu-applications"></a>
## 5. LU Decomposition Applications

### 5.1 Solving Multiple Linear Systems

```
Common scenario in physics simulations, circuit analysis:

  Same coefficient matrix A, different right-hand sides b₁, b₂, ..., bₖ

  ┌─────────────────────────────────────────┐
  │ Factor A = LU:        O(n³)           │  ← done ONCE
  │                                          │
  │ For each bᵢ:                              │
  │   Solve L·y = bᵢ:     O(n²)           │
  │   Solve U·x = y:      O(n²)           │
  │                                          │
  │ Total: O(n³) + k·O(n²)                 │
  └─────────────────────────────────────────┘

  vs. Gaussian elimination for each system: k × O(n³)
  
  Speedup when k << n: roughly k times faster!
```

### 5.2 LU in Iterative Refinement

```
To improve accuracy of a solution x₀ to Ax = b:

  For iteration j = 1, 2, ..., max_iter:
    rⱼ = b - A·xⱼ₋₁       (compute residual)
    A·Δⱼ = rⱼ              (solve using existing LU)
    xⱼ = xⱼ₋₁ + Δⱼ        (update solution)

This converges quadratically when A is well-conditioned.
```

### 5.3 LU in Conditioning Analysis

```
The condition number of A can be estimated from LU factors:

  ||A|| ≈ ||U|| (upper bound)
  ||A⁻¹|| can be estimated from L and U
  
  κ(A) = ||A|| · ||A⁻¹||

Good conditioning:  κ(A) ≈ 1      (singular values similar)
Poor conditioning:  κ(A) >> 1     (some singular values very small)

  κ < 10²    : Excellent
  κ < 10⁴    : Good
  κ < 10⁶    : Fair
  κ < 10⁸    : Poor
  κ ≥ 10⁸    : Very problematic
```

---

<a name="python"></a>
## 6. Python Implementations

### 6.1 SVD with NumPy and SciPy

```python
import numpy as np
from scipy import linalg
from scipy.linalg import svd, lu, lu_factor, lu_solve
import matplotlib.pyplot as plt

# ============================================================
# 6.1.1 Basic SVD
# ============================================================
def basic_svd_demo():
    """Demonstrate basic SVD computation."""
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=float)
    
    # Full SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    
    print("Matrix A shape:", A.shape)
    print("U shape:", U.shape)       # (4, 4) - left singular vectors
    print("s shape:", s.shape)        # (3,)  - singular values
    print("Vt shape:", Vt.shape)     # (3, 3) - right singular vectors (transposed)
    
    # Singular values
    print("\nSingular values:", s)
    print("Frobenius norm:", np.linalg.norm(A, 'fro'))
    print("From SVD:", np.sqrt(np.sum(s**2)))  # Should match!
    
    # Reconstruct A
    Sigma = np.zeros(A.shape)
    np.fill_diagonal(Sigma, s)
    A_reconstructed = U @ Sigma @ Vt
    print("\nReconstruction error:", np.linalg.norm(A - A_reconstructed))
    
    # Compact SVD
    U_compact, s_compact, Vt_compact = np.linalg.svd(A, full_matrices=False)
    print("\nCompact SVD:")
    print("U shape:", U_compact.shape)  # (4, 3)
    print("Vt shape:", Vt_compact.shape)  # (3, 3)
    
    return U, s, Vt


# ============================================================
# 6.1.2 Truncated SVD for Low-Rank Approximation
# ============================================================
def truncated_svd_demo():
    """Demonstrate truncated SVD and low-rank approximation."""
    np.random.seed(42)
    A = np.random.randn(100, 50)
    
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    print("Original matrix shape:", A.shape)
    print("All singular values:", s[:10])
    
    # Approximation with different ranks
    ranks = [5, 10, 20, 30]
    for k in ranks:
        A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - A_k, 'fro')
        rel_error = error / np.linalg.norm(A, 'fro')
        variance_explained = np.sum(s[:k]**2) / np.sum(s**2)
        print(f"  Rank {k:2d}: error = {error:.4f}, "
              f"relative = {rel_error:.4f}, "
              f"variance explained = {variance_explained:.4f}")
    
    # Scree plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(s, 'bo-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values (Scree Plot)')
    
    plt.subplot(1, 2, 2)
    cumulative = np.cumsum(s**2) / np.sum(s**2)
    plt.plot(cumulative, 'ro-')
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('svd_analysis.png', dpi=150)
    plt.show()


# ============================================================
# 6.1.3 PCA via SVD
# ============================================================
def pca_via_svd(X, n_components=2):
    """Perform PCA using SVD on data matrix X (n_samples, n_features)."""
    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Step 2: Compute SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Step 3: Principal components
    components = Vt[:n_components]  # (n_components, n_features)
    
    # Step 4: Project data
    X_transformed = X_centered @ components.T  # (n_samples, n_components)
    
    # Variance explained
    total_var = np.sum(s**2) / (len(X) - 1)
    explained_var = s[:n_components]**2 / (len(X) - 1)
    explained_ratio = explained_var / total_var
    
    return X_transformed, components, explained_ratio


def pca_demo():
    """Full PCA demonstration."""
    from sklearn.datasets import load_iris
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Apply PCA
    X_pca, components, ratio = pca_via_svd(X, n_components=2)
    
    print("Original shape:", X.shape)
    print("Reduced shape:", X_pca.shape)
    print("Variance explained by PC1: {:.2%}".format(ratio[0]))
    print("Variance explained by PC2: {:.2%}".format(ratio[1]))
    print("Total variance explained:  {:.2%}".format(sum(ratio)))
    
    # Verify with sklearn
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_sklearn = pca.fit_transform(X)
    print("\nsklearn PCA variance ratio:", pca.explained_variance_ratio_)
    
    return X_pca


# ============================================================
# 6.1.4 Image Compression with SVD
# ============================================================
def image_compression_demo():
    """Compress an image using truncated SVD."""
    from sklearn.datasets import load_sample_image
    
    # Load and preprocess
    china = load_sample_image('china.jpg')
    china_gray = np.mean(china, axis=2) / 255.0  # Convert to grayscale
    
    print("Image shape:", china_gray.shape)
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(china_gray, full_matrices=False)
    print(f"Singular values: {len(s)} total")
    
    # Compress with different ranks
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    ranks = [5, 20, 50, 100, 200, len(s)]
    for idx, k in enumerate(ranks):
        compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        storage = k * (china_gray.shape[0] + 1 + china_gray.shape[1])
        original = china_gray.shape[0] * china_gray.shape[1]
        compression_ratio = original / storage
        
        axes[idx].imshow(compressed, cmap='gray')
        axes[idx].set_title(f'Rank {k}\nCompression: {compression_ratio:.1f}x')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_compression.png', dpi=150)
    plt.show()


# ============================================================
# 6.1.5 Recommender System with SVD
# ============================================================
def recommender_svd_demo():
    """Build a simple recommender system using SVD."""
    # User-item rating matrix (users × items)
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ], dtype=float)
    
    # Mean centering (optional but common)
    user_mean = np.mean(R[R > 0])  # global mean
    R_centered = R.copy()
    R_centered[R_centered == 0] = np.nan
    row_means = np.nanmean(R_centered, axis=1, keepdims=True)
    R_centered = np.nan_to_num(R_centered - row_means)
    
    # SVD
    k = 2  # latent factors
    U, s, Vt = np.linalg.svd(R_centered, full_matrices=False)
    
    # Truncated
    U_k = U[:, :k]
    S_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]
    
    # Predict all ratings
    R_pred = U_k @ S_k @ Vt_k + row_means
    
    print("Original ratings:")
    print(R)
    print("\nPredicted ratings (0 = unseen):")
    print(np.round(R_pred, 2))
    print("\nPredicted for unseen items (where original was 0):")
    mask = R == 0
    predictions = np.round(R_pred[mask], 2)
    print(predictions)
    
    # User and item embeddings
    print("\nUser embeddings (U_k × S_k^(1/2)):")
    user_embeddings = U_k @ np.sqrt(S_k)
    print(np.round(user_embeddings, 3))
    
    print("\nItem embeddings (S_k^(1/2) × Vt_k):")
    item_embeddings = np.sqrt(S_k) @ Vt_k
    print(np.round(item_embeddings, 3))
    
    return R_pred
```

### 6.2 LU Decomposition with NumPy and SciPy

```python
# ============================================================
# 6.2.1 Basic LU Decomposition
# ============================================================
def basic_lu_demo():
    """Demonstrate LU decomposition."""
    A = np.array([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9]
    ], dtype=float)
    
    # Using scipy.linalg.lu
    P, L, U = linalg.lu(A)
    
    print("Original matrix A:")
    print(A)
    print("\nPermutation matrix P:")
    print(P)
    print("\nLower triangular L:")
    print(L)
    print("\nUpper triangular U:")
    print(U)
    print("\nVerification P @ L @ U:")
    print(P @ L @ U)
    print("Matches A:", np.allclose(P @ L @ U, A))
    
    return P, L, U


# ============================================================
# 6.2.2 Solving Linear Systems
# ============================================================
def solve_linear_system_demo():
    """Solve Ax = b using LU decomposition."""
    A = np.array([
        [3, 2, -1],
        [2, -2, 4],
        [-1, 0.5, -1]
    ], dtype=float)
    
    b = np.array([1, -2, 0], dtype=float)
    
    # Method 1: Direct solve (for reference)
    x_direct = np.linalg.solve(A, b)
    print("Direct solve:", x_direct)
    
    # Method 2: LU factorization
    lu_piv = lu_factor(A)  # Returns (LU, piv) tuple
    x_lu = lu_solve(lu_piv, b)
    print("LU solve:   ", x_lu)
    
    # Method 3: Manual LU
    P, L, U = linalg.lu(A)
    
    # Solve PA = LU → Ax = b → PAx = Pb → LUx = Pb
    Pb = P @ b
    
    # Forward substitution: Ly = Pb
    y = linalg.solve_triangular(L, Pb, lower=True)
    
    # Back substitution: Ux = y
    x_manual = linalg.solve_triangular(U, y, lower=False)
    
    print("Manual LU:   ", x_manual)
    print("Verification:", A @ x_manual)
    
    # Timing comparison for multiple RHS
    print("\n--- Timing for 100 different b vectors ---")
    import time
    
    # Generate 100 right-hand sides
    B = np.random.randn(3, 100)
    
    t0 = time.time()
    X_direct = np.linalg.solve(A, B)
    t_direct = time.time() - t0
    
    t0 = time.time()
    lu_piv = lu_factor(A)
    X_lu = lu_solve(lu_piv, B)
    t_lu = time.time() - t0
    
    print(f"Direct solve:    {t_direct*1000:.3f} ms")
    print(f"LU factor+solve: {t_lu*1000:.3f} ms")
    
    return x_lu


# ============================================================
# 6.2.3 Computing Determinant via LU
# ============================================================
def determinant_via_lu(A):
    """Compute determinant using LU decomposition."""
    P, L, U = linalg.lu(A)
    
    # det(P) = (-1)^(number of row swaps)
    n_swaps = len(np.where(np.diag(P) != np.arange(len(P)))[0])
    sign = (-1) ** n_swaps
    
    # det(L) = 1, det(U) = product of diagonal
    det_U = np.prod(np.diag(U))
    
    return sign * det_U


def determinant_demo():
    """Compare determinant computation methods."""
    np.random.seed(42)
    A = np.random.randn(5, 5)
    
    det_lu = determinant_via_lu(A)
    det_numpy = np.linalg.det(A)
    
    print(f"det(A) via LU:  {det_lu:.6f}")
    print(f"det(A) via numpy: {det_numpy:.6f}")
    print(f"Match: {np.isclose(det_lu, det_numpy)}")


# ============================================================
# 6.2.4 Matrix Inverse via LU
# ============================================================
def inverse_via_lu(A):
    """Compute matrix inverse using LU decomposition."""
    n = A.shape[0]
    I = np.eye(n)
    
    P, L, U = linalg.lu(A)
    PI = P @ I
    
    # Solve for each column
    A_inv = np.zeros_like(A)
    for j in range(n):
        # Forward substitution: Ly = (PI)[:,j]
        y = linalg.solve_triangular(L, PI[:, j], lower=True)
        # Back substitution: Ux = y
        A_inv[:, j] = linalg.solve_triangular(U, y, lower=False)
    
    return A_inv


def inverse_demo():
    """Compare inverse computation methods."""
    np.random.seed(42)
    A = np.random.randn(4, 4)
    
    inv_lu = inverse_via_lu(A)
    inv_numpy = np.linalg.inv(A)
    
    print("LU inverse:\n", np.round(inv_lu, 4))
    print("\nNumPy inverse:\n", np.round(inv_numpy, 4))
    print("\nA @ A_inv (LU):\n", np.round(A @ inv_lu, 10))
    print("\nMax error:", np.max(np.abs(A @ inv_lu - np.eye(4))))


# ============================================================
# 6.2.5 Complete LU Factorization (Doolittle's Method)
# ============================================================
def doolittle_lu(A):
    """Manual implementation of Doolittle's LU decomposition."""
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for j in range(n):
        # Upper triangular
        for i in range(j + 1):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])
        
        # Lower triangular
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * U[:j, j])) / U[j, j]
    
    return L, U


def doolittle_demo():
    """Demonstrate manual LU implementation."""
    A = np.array([
        [2, 1, 1, 0],
        [4, 3, 3, 1],
        [8, 7, 9, 5],
        [6, 7, 9, 8]
    ], dtype=float)
    
    L, U = doolittle_lu(A)
    
    print("L (Doolittle):")
    print(np.round(L, 4))
    print("\nU (Doolittle):")
    print(np.round(U, 4))
    print("\nL @ U:")
    print(np.round(L @ U, 4))
    print("\nOriginal A:")
    print(A)
    print("Match:", np.allclose(L @ U, A))


# ============================================================
# 6.2.6 LU with Partial Pivoting
# ============================================================
def lu_with_pivoting(A):
    """LU decomposition with partial pivoting."""
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)
    
    for k in range(n):
        # Find pivot
        max_idx = k + np.argmax(np.abs(U[k:, k]))
        
        # Swap rows
        if max_idx != k:
            U[[k, max_idx]] = U[[max_idx, k]]
            P[[k, max_idx]] = P[[max_idx, k]]
            if k > 0:
                L[[k, max_idx], :k] = L[[max_idx, k], :k]
        
        # Elimination
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return P, L, U
```

### 6.3 Comparison and Performance Analysis

```python
# ============================================================
# 6.3.1 SVD vs LU Comparison
# ============================================================
def decomposition_comparison():
    """Compare different decomposition methods."""
    import time
    
    sizes = [50, 100, 200, 500, 1000]
    
    print("Matrix Size | SVD Time | LU Time | Det (SVD) | Det (LU) | Condition #")
    print("-" * 80)
    
    for n in sizes:
        A = np.random.randn(n, n)
        
        # SVD timing
        t0 = time.time()
        U, s, Vt = np.linalg.svd(A)
        t_svd = time.time() - t0
        
        # LU timing
        t0 = time.time()
        lu_piv = lu_factor(A)
        t_lu = time.time() - t0
        
        # Determinant
        det_svd = np.prod(s) * np.linalg.det(U @ Vt)
        det_lu = np.prod(np.diag(lu_piv[0]))
        
        # Condition number
        cond = s[0] / s[-1]
        
        print(f"{n:11d} | {t_svd*1000:8.2f} | {t_lu*1000:6.2f} | "
              f"{det_svd:9.4f} | {det_lu:8.4f} | {cond:11.4f}")


# ============================================================
# 6.3.2 Numerical Stability Analysis
# ============================================================
def numerical_stability_demo():
    """Demonstrate importance of pivoting in LU."""
    # Hilbert matrix (notoriously ill-conditioned)
    def hilbert(n):
        return np.array([[1.0 / (i + j - 1) for j in range(1, n+1)]
                         for i in range(1, n+1)])
    
    for n in [5, 10, 15]:
        H = hilbert(n)
        cond = np.linalg.cond(H)
        
        # LU without pivoting (using our Doolittle)
        try:
            L, U = doolittle_lu(H)
            error_no_pivot = np.linalg.norm(L @ U - H)
        except:
            error_no_pivot = float('inf')
        
        # LU with pivoting (scipy)
        P, L_p, U_p = linalg.lu(H)
        error_pivot = np.linalg.norm(P @ L_p @ U_p - H)
        
        print(f"Hilbert({n:2d}): κ = {cond:.2e}, "
              f"error no-pivot = {error_no_pivot:.2e}, "
              f"error pivot = {error_pivot:.2e}")
```

---

<a name="ml-connections"></a>
## 7. ML Connections

### 7.1 SVD in Machine Learning Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    ML Pipeline with SVD                       │
│                                                              │
│  Raw Data                                                    │
│     │                                                        │
│     ▼                                                        │
│  ┌─────────┐    SVD     ┌──────────┐     ┌──────────────┐  │
│  │Feature  │───────────►│Dimension │────►│ML Model      │  │
│  │Matrix   │            │Reduction │     │(SVM, RF, NN) │  │
│  └─────────┘            └──────────┘     └──────────────┘  │
│                                                              │
│  Applications:                                               │
│  • PCA: Feature extraction for classification               │
│  • LSA: Document similarity for clustering                  │
│  • Recommenders: Collaborative filtering                    │
│  • Image: Feature extraction for CNN pre-processing         │
│  • NLP: Word embeddings (SVD on co-occurrence)             │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 LU in ML Applications

```
LU decomposition appears in several ML contexts:

1. Logistic Regression (Newton's Method):
   H·Δθ = -∇J  where H is the Hessian
   → LU factorize H once, solve for each Newton step

2. Gaussian Processes:
   K·α = y  where K is the kernel matrix
   → LU factorize K, use for prediction

3. Kalman Filters:
   State update involves solving linear systems
   → LU factorize the covariance matrix

4. Interior Point Methods (SVM, LP):
   Each iteration solves a linear system
   → LU factorization is the core solver

5. Finite Element Methods (Physics-Informed NNs):
   Discretize PDE → sparse linear system
   → Sparse LU (SuperLU, UMFPACK)
```

### 7.3 Truncated SVD as Regularization

```
In ill-posed problems (e.g., solving Ax=b with noisy A):

  Direct solution: x = A⁺b = VΣ⁺Uᵀb
  
  Problem: Small singular values amplify noise enormously
  
  Tikhonov regularization:
    x_reg = (AᵀA + λI)⁻¹Aᵀb  ← adds λ to diagonal of AᵀA
    
  Equivalent via SVD:
    x_reg = V · diag(σᵢ/(σᵢ² + λ)) · Uᵀb
    
  Truncated SVD:
    x_trunc = V_k · diag(1/σᵢ) · U_kᵀb  ← simply discards small σᵢ
    
  Both approaches are forms of regularization!
  
  Filter factors:
  ┌─────────────────────────────────────────────┐
  │ σ large → Tikhonov: σ/(σ²+λ) ≈ 1/σ        │
  │           Truncated:  1/σ                    │
  │ σ small → Tikhonov: σ/(σ²+λ) ≈ σ/λ ≈ 0   │
  │           Truncated:  0                     │
  └─────────────────────────────────────────────┘
```

### 7.4 Randomized SVD for Large-Scale ML

```
For very large matrices (common in deep learning):

  Randomized SVD Algorithm:
  1. Draw random matrix Ω ∈ ℝ^(n×(k+p))   [p = oversampling]
  2. Form Y = A · Ω                         [O(mnk)]
  3. QR factorize Y = QR                     [O(n(k+p)²)]
  4. Form B = Qᵀ · A                         [O(mn(k+p))]
  5. Compute SVD of small B = Û·Ŝ·V̂ᵀ        [O(n(k+p)²)]
  6. U = Q · Û                              [O(m(k+p)²)]
  
  Total: O(mnk) vs O(mn²) for full SVD
  
  When k << min(m,n), this is dramatically faster!
  
  Used in:
  - sklearn.decomposition.TruncatedSVD
  - Facebook's FAISS library
  - Large-scale PCA
  - Neural network compression
```

---

<a name="exercises"></a>
## 8. Exercises

### Exercise 1: SVD Basics (Beginner)

```python
# Problem: Given matrix A, compute its SVD and verify the properties
A = np.array([[3, 2, 2],
              [2, 3, -2]])

# TODO:
# 1. Compute U, s, Vt using np.linalg.svd
# 2. Verify U is orthogonal (U^T U = I)
# 3. Verify V is orthogonal (V^T V = I)
# 4. Reconstruct A from U, s, Vt
# 5. Compute ||A||_F from singular values and verify
```

### Exercise 2: Image Compression (Intermediate)

```python
# Problem: Compress an image and find the rank needed for 95% quality
from sklearn.datasets import load_sample_image

image = load_sample_image('china.jpg')
# Convert to grayscale and normalize
gray = np.mean(image, axis=2) / 255.0

# TODO:
# 1. Compute SVD of the image matrix
# 2. Plot the singular values (scree plot)
# 3. Find the minimum rank k such that ||A - A_k||_F / ||A||_F < 0.05
# 4. Display the compressed image for k = 10, 50, 100
# 5. Calculate compression ratio for each k
```

### Exercise 3: PCA from Scratch (Intermediate)

```python
# Problem: Implement PCA using SVD (without sklearn)
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

# TODO:
# 1. Center the data (subtract mean)
# 2. Compute SVD of centered data
# 3. Find the number of components needed for 90% variance
# 4. Project data to 2D and plot with different class colors
# 5. Compare with sklearn PCA results
```

### Exercise 4: LU Factorization (Intermediate)

```python
# Problem: Implement Gaussian elimination with partial pivoting
def lu_solve_system(A, b):
    """
    Solve Ax = b using LU decomposition with partial pivoting.
    
    Returns: x (solution vector), L (lower triangular), U (upper triangular)
    """
    # TODO: Implement the algorithm
    pass

# Test:
A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], dtype=float)
b = np.array([5], [-2], [9], dtype=float)
# Expected solution: x ≈ [1.442, -0.885, 4.010]
```

### Exercise 5: Recommender System (Advanced)

```python
# Problem: Build a complete recommender system with SVD
# Dataset: MovieLens 100K (or use synthetic data)

# TODO:
# 1. Load and preprocess user-item rating matrix
# 2. Apply SVD with different k values (5, 10, 20, 50)
# 3. Handle missing values properly
# 4. Evaluate using RMSE on a train/test split
# 5. Implement bias terms (user bias + item bias + global mean)
# 6. Compare with simple baseline (mean rating)
```

### Exercise 6: Doolittle's Algorithm (Advanced)

```python
# Problem: Implement Doolittle's method with error analysis
def doolittle_with_analysis(A):
    """
    Compute LU decomposition and return error analysis.
    Returns: L, U, forward_error, backward_error
    """
    # TODO:
    # 1. Implement Doolittle's algorithm
    # 2. Compute ||A - LU||_F (backward error)
    # 3. Compare with scipy.linalg.lu result
    # 4. Test on Hilbert matrices of increasing size
    pass

# Test on increasingly ill-conditioned matrices
for n in range(2, 12):
    H = hilbert(n)
    L, U = doolittle_with_analysis(H)
    print(f"n={n:2d}, error={error:.2e}, cond={cond:.2e}")
```

### Exercise 7: Randomized SVD (Advanced)

```python
# Problem: Implement randomized SVD and compare with exact SVD
def randomized_svd(A, k, n_oversamples=10, n_iter=2):
    """
    Compute approximate rank-k SVD using randomized algorithm.
    
    Parameters:
        A: input matrix (m, n)
        k: target rank
        n_oversamples: additional samples for accuracy
        n_iter: number of power iterations
    
    Returns: U_k, s_k, Vt_k
    """
    # TODO: Implement the randomized SVD algorithm
    # Hint: Use QR factorization and power iterations
    pass

# Compare accuracy and speed
for size in [(1000, 500), (5000, 2000), (10000, 5000)]:
    A = np.random.randn(*size)
    # Time and compare exact vs randomized SVD
```

### Exercise 8: SVD for Topic Modeling (Advanced)

```python
# Problem: Use SVD for topic modeling on text data
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing deals with text understanding",
    "Computer vision enables machines to see and understand images",
    "Reinforcement learning trains agents through rewards",
    # ... add more documents
]

# TODO:
# 1. Create TF-IDF matrix
# 2. Apply truncated SVD (k=3 topics)
# 3. Identify top words for each topic
# 4. Classify documents by dominant topic
# 5. Compare with NMF (Non-negative Matrix Factorization)
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    KEY TAKEAWAYS                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SVD:   A = UΣVᵀ                                               │
│  • Works for ANY matrix (not just square)                       │
│  • Best low-rank approximation (Eckart-Young theorem)          │
│  • Foundation of PCA, LSA, Recommender Systems                 │
│  • Complexity: O(mn²) or O(mn min(m,n))                        │
│                                                                  │
│  LU:    PA = LU                                                  │
│  • Only for square matrices                                      │
│  • Efficient for solving multiple systems with same A           │
│  • Need pivoting for numerical stability                        │
│  • Complexity: O(n³/3) for factorization                       │
│                                                                  │
│  ML Connections:                                                 │
│  • SVD → PCA, dimensionality reduction, recommender systems    │
│  • LU → Solving linear systems in optimization, GP, Kalman     │
│  • Both fundamental to numerical linear algebra in ML           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
