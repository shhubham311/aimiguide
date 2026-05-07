# Matrices: Types & Properties

## Why This Matters in AI/ML

Matrices are the fundamental data structure of machine learning. Your entire dataset is a matrix. A neural network's weights are matrices. Convolution operations are matrix multiplications. The covariance matrix captures feature relationships. The Jacobian and Hessian encode gradient information. Understanding matrix types and operations is non-negotiable for anyone working in ML — every `forward()` pass is a sequence of matrix operations.

---

## 1. Matrix Basics

### 1.1 Definition

A matrix A is a rectangular array of numbers with m rows and n columns:

```
A ∈ ℝ^(m×n) = [a₁₁  a₁₂  ...  a₁ₙ ]
              [a₂₁  a₂₂  ...  a₂ₙ ]
              [ ...  ...  ...  ... ]
              [aₘ₁  aₘ₂  ...  aₘₙ ]
```

- **Entry aᵢⱼ**: element in row i, column j
- **Row vector:** 1 × n matrix
- **Column vector:** m × 1 matrix
- **Square matrix:** m = n

### 1.2 Matrix Operations

```
Addition:        (A + B)ᵢⱼ = aᵢⱼ + bᵢⱼ         [same dimensions]
Scalar Multiply: (αA)ᵢⱼ  = α · aᵢⱼ
Transpose:       (Aᵀ)ᵢⱼ   = aⱼᵢ                 [flips rows and columns]
Multiplication:  (AB)ᵢⱼ  = Σₖ aᵢₖ bₖⱼ           [A: m×n, B: n×p → AB: m×p]
```

### 1.3 Matrix Multiplication: The Key Operation

```
  A (2×3)          B (3×2)         C = AB (2×2)

  [a b c]    ×    [p q]     =     [ap+br+cs  aq+bt+cu]
  [d e f]         [r s]           [dp+er+fs  dq+es+fu]
                  [t u]

  Each entry is a dot product of a row of A with a column of B.
```

**Important properties:**
```
AB ≠ BA in general         (matrix multiplication is NOT commutative)
(AB)C = A(BC)             (associative)
A(B+C) = AB + AC          (distributive)
(AB)ᵀ = BᵀAᵀ             (reverse order for transpose!)
```

```python
import numpy as np

# Matrix multiplication in NumPy
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

B = np.array([[7,  8],
              [9, 10],
              [11, 12]])    # 3×2

C = A @ B  # equivalent to np.dot(A, B) or A.dot(B)
print(f"A @ B = \n{C}")
print(f"Shape: {C.shape}")

# Transpose properties
print(f"\n(AB)ᵀ = \n{C.T}")
print(f"\nBᵀ @ Aᵀ = \n{B.T @ A.T}")  # Same as (AB)ᵀ
```

---

## 2. Types of Matrices

### 2.1 Square Matrix

A matrix with m = n (same number of rows and columns).

### 2.2 Diagonal Matrix

All off-diagonal entries are zero:

```
D = [d₁  0   0  ]
    [0   d₂  0  ]
    [0   0   d₃ ]
```

**Notation:** diag(d₁, d₂, ..., dₙ)

### 2.3 Identity Matrix

The diagonal matrix with all 1s on the diagonal: Iₙ

```
I = [1 0 0]
    [0 1 0]
    [0 0 1]
```

**Property:** AI = IA = A for any compatible A.

### 2.4 Symmetric Matrix

A square matrix where Aᵀ = A (mirror image across the diagonal):

```
S = [1   2   3 ]
    [2   5   6 ]
    [3   6   9 ]
```

**ML Example:** The covariance matrix Σ = (1/n)XᵀX is always symmetric.

### 2.5 Triangular Matrices

```
Upper Triangular:          Lower Triangular:
[1  2  3 ]                 [1  0  0 ]
[0  5  6 ]                 [4  5  0 ]
[0  0  9 ]                 [7  8  9 ]
```

### 2.6 Orthogonal Matrix

A square matrix Q where QᵀQ = I (columns are orthonormal vectors).

```
Properties:
- Qᵀ = Q⁻¹ (inverse = transpose)
- ‖Qx‖ = ‖x‖ (preserves lengths)
- Qx · Qy = x · y (preserves inner products)
- det(Q) = ±1
```

**ML Applications:** Rotation matrices, reflection matrices, PCA transformation.

```python
import numpy as np

def is_orthogonal(Q, tol=1e-10):
    """Check if Q is orthogonal."""
    return np.allclose(Q.T @ Q, np.eye(Q.shape[0]), atol=tol)

# 2D Rotation matrix (always orthogonal)
theta = np.pi / 4  # 45 degrees
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print(f"Rotation matrix Q:\n{Q}")
print(f"Is orthogonal: {is_orthogonal(Q)}")
print(f"QᵀQ:\n{np.round(Q.T @ Q, 10)}")

# Verify preservation of norm
v = np.array([3.0, 4.0])
print(f"\n||v|| = {np.linalg.norm(v):.4f}")
print(f"||Qv|| = {np.linalg.norm(Q @ v):.4f}")  # Should be the same!
```

### 2.7 Positive Definite Matrix

A symmetric matrix A is **positive definite** if xᵀAx > 0 for all nonzero x ∈ ℝⁿ.

```
Properties:
- All eigenvalues > 0
- All leading principal minors > 0 (Sylvester's criterion)
- A⁻¹ exists and is also positive definite
- xᵀAx defines an inner product

Positive Semidefinite (PSD): xᵀAx ≥ 0 for all x
  - All eigenvalues ≥ 0
  - May not be invertible
```

**ML Examples:**
- Covariance matrix is always PSD
- Hessian of convex loss function is PSD
- XᵀX (normal equations matrix) is always PSD

### 2.8 Sparse Matrix

Most entries are zero. Common in NLP (TF-IDF matrices), recommender systems (user-item matrices), and graph adjacency matrices.

```python
import numpy as np
from scipy import sparse

# Create sparse matrices (efficient for large, mostly-zero data)
row = np.array([0, 1, 2, 3])
col = np.array([0, 1, 2, 3])
data = np.array([1.0, 2.0, 3.0, 4.0])

# Dense: 4×4 = 16 entries, 12 of them zero
A_dense = np.eye(4) * [1, 2, 3, 4]

# Sparse: stores only non-zero entries
A_sparse = sparse.csr_matrix(A_dense)

print(f"Dense memory: {A_dense.nbytes} bytes")
print(f"Sparse memory: {A_sparse.data.nbytes + A_sparse.indptr.nbytes + A_sparse.indices.nbytes} bytes")
print(f"Sparse ratio: {(A_sparse.data.nbytes + A_sparse.indptr.nbytes + A_sparse.indices.nbytes) / A_dense.nbytes:.2%}")
```

---

## 3. Matrix Inverse

### 3.1 Definition

For a square matrix A, the inverse A⁻¹ satisfies:

```
A · A⁻¹ = A⁻¹ · A = I
```

A matrix is **invertible** (nonsingular) if and only if det(A) ≠ 0.

### 3.2 Computing the Inverse

```
For 2×2 matrix A = [[a, b], [c, d]]:

A⁻¹ = (1/(ad - bc)) · [[d, -b], [-c, a]]
```

For larger matrices: use Gaussian elimination or compute adjugate.

### 3.3 Properties of Inverse

```
(A⁻¹)⁻¹ = A
(Aᵀ)⁻¹ = (A⁻¹)ᵀ
(AB)⁻¹ = B⁻¹A⁻¹     (reverse order!)
(αA)⁻¹ = (1/α)A⁻¹
det(A⁻¹) = 1/det(A)
```

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

A_inv = np.linalg.inv(A)
print(f"A⁻¹:\n{A_inv}")
print(f"\nA @ A⁻¹:\n{np.round(A @ A_inv, 10)}")  # Should be identity

# Verify 2×2 formula
det_A = 1*4 - 2*3
A_inv_formula = (1/det_A) * np.array([[4, -2], [-3, 1]])
print(f"\nFormula result:\n{A_inv_formula}")
```

---

## 4. Matrix Trace

The **trace** of a square matrix is the sum of its diagonal elements:

```
tr(A) = Σᵢ aᵢᵢ
```

**Properties:**
```
tr(A + B) = tr(A) + tr(B)
tr(αA) = α · tr(A)
tr(AB) = tr(BA)           (even if AB ≠ BA!)
tr(Aᵀ) = tr(A)
tr(Iₙ) = n
```

**ML Application:** The trace of the covariance matrix equals the sum of feature variances (= total variance captured by all features).

---

## 5. Special Matrix Operations in ML

### 5.1 Element-wise Operations (Hadamard Product)

```
(A ⊙ B)ᵢⱼ = aᵢⱼ · bᵢⱼ

Not the same as matrix multiplication!
```

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix multiply A@B:\n{A @ B}")
print(f"Element-wise A*B:\n{A * B}")  # NumPy uses * for element-wise
```

### 5.2 Outer Product

For column vectors u ∈ ℝᵐ, v ∈ ℝⁿ:

```
u ⊗ v = uvᵀ ∈ ℝ^(m×n)

(uvᵀ)ᵢⱼ = uᵢ · vⱼ
```

```python
u = np.array([1, 2, 3])
v = np.array([4, 5])
print(f"Outer product (3×2):\n{np.outer(u, v)}")
```

### 5.3 Kronecker Product

```
A ⊗ B: replace each aᵢⱼ in A with the block aᵢⱼ · B
```

Used in tensor decomposition and quantum computing.

### 5.4 Broadcasting

NumPy automatically expands dimensions for element-wise operations:

```python
# Broadcasting: add a vector to each row of a matrix
X = np.array([[1, 2, 3],
              [4, 5, 6]])   # 2×3
b = np.array([10, 20, 30])  # (3,)

print(f"X + b:\n{X + b}")
# b is "broadcast" to [[10,20,30],[10,20,30]] then added
```

---

## 6. Norms of Matrices

### 6.1 Frobenius Norm

```
‖A‖_F = √(Σᵢ Σⱼ |aᵢⱼ|²) = √(tr(AᵀA))
```

### 6.2 Operator Norm (Spectral Norm)

```
‖A‖₂ = max(‖Ax‖ / ‖x‖) = σ_max(A)
```

The largest singular value of A. This is the induced L2 norm.

### 6.3 Nuclear Norm

```
‖A‖_* = Σᵢ σᵢ
```

Sum of all singular values. Used in **matrix completion** and **low-rank matrix factorization** (e.g., collaborative filtering).

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Frobenius norm: {np.linalg.norm(A, 'fro'):.4f}")
print(f"Spectral norm:  {np.linalg.norm(A, 2):.4f}")
print(f"Nuclear norm:   {np.linalg.norm(A, 'nuc'):.4f}")

# Verify Frobenius norm
manual_frob = np.sqrt(np.sum(A**2))
print(f"Manual Frobenius: {manual_frob:.4f}")
```

---

## 7. Exercises with Solutions

### Exercise 1
Compute AB and BA for A = [[1, 0], [2, 3]] and B = [[0, 1], [1, 0]].

**Solution:**
```
AB = [[1·0+0·1, 1·1+0·0],    = [[0, 1],
     [2·0+3·1, 2·1+3·0]]       [3, 2]]

BA = [[0·1+1·2, 0·0+1·3],    = [[2, 3],
     [1·1+0·2, 1·0+0·3]]       [1, 0]]

AB ≠ BA — matrix multiplication is not commutative!
```

### Exercise 2
Show that (AB)ᵀ = BᵀAᵀ for A ∈ ℝ^(m×n), B ∈ ℝ^(n×p).

**Solution:**
```
((AB)ᵀ)ᵢⱼ = (AB)ⱼᵢ = Σₖ aⱼₖ bₖᵢ

(BᵀAᵀ)ᵢⱼ = Σₖ (Bᵀ)ᵢₖ (Aᵀ)ₖⱼ = Σₖ bₖᵢ aⱼₖ = Σₖ aⱼₖ bₖᵢ

Therefore (AB)ᵀ = BᵀAᵀ ✓
```

### Exercise 3
Prove that the covariance matrix C = (1/n)XᵀX is always symmetric and positive semidefinite.

**Solution:**
```
Symmetric: Cᵀ = ((1/n)XᵀX)ᵀ = (1/n)Xᵀ(Xᵀ)ᵀ = (1/n)XᵀX = C ✓

PSD: For any v ∈ ℝⁿ:
vᵀCv = (1/n) vᵀXᵀXv = (1/n)(Xv)ᵀ(Xv) = (1/n)‖Xv‖² ≥ 0 ✓
```

### Exercise 4
Find the inverse of A = [[2, 1], [5, 3]] and verify A·A⁻¹ = I.

**Solution:**
```
det(A) = 2·3 - 1·5 = 6 - 5 = 1

A⁻¹ = (1/1) · [[3, -1], [-5, 2]] = [[3, -1], [-5, 2]]

Verify: [[2,1],[5,3]] · [[3,-1],[-5,2]]
      = [[6-5, -2+2], [15-15, -5+6]]
      = [[1, 0], [0, 1]] ✓
```

---

## Summary

| Matrix Type | Key Property | ML Application |
|-------------|-------------|----------------|
| **Diagonal** | aᵢⱼ = 0 for i ≠ j | Scaling transformations |
| **Identity** | AI = IA = A | Neutral element |
| **Symmetric** | Aᵀ = A | Covariance matrix, Hessian |
| **Orthogonal** | QᵀQ = I | Rotations, PCA |
| **Positive Definite** | xᵀAx > 0 | Convexity, kernel matrices |
| **Sparse** | Most entries = 0 | NLP features, recommender systems |
| **Triangular** | Upper or lower zeros | LU decomposition |
