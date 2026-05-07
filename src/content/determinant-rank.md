# Determinant, Rank & Projections

## Why This Matters in AI/ML

These three linear algebra concepts are fundamental to understanding how ML algorithms work:

- **Determinants**: Tell you if a matrix is invertible (critical for solving linear systems in regression)
- **Rank**: Reveals the effective dimensionality of your data (connects to PCA, feature selection)
- **Projections**: The mathematical foundation of least squares regression, dimensionality reduction, and recommendation systems

## Determinant

### Intuition

The determinant of a matrix tells you **how much the matrix scales area/volume** when it transforms space.

```
Original square:          After transformation by A:
┌─────┐                  ┌──────────┐
│     │                  │  ╱        │
│     │  → det(A) = 2   │ ╱  Area   │
│     │                  │╱ doubled! │
└─────┘                  └──────────┘

det(A) > 0: Preserves orientation
det(A) < 0: Flips orientation  
det(A) = 0: Collapses dimension (NOT invertible!)
```

### Formulas

**2×2 Matrix:**
```
det(A) = |a  b| = ad - bc
        |c  d|
```

**3×3 Matrix (Rule of Sarrus):**
```
det(A) = |a b c| = aei + bfg + cdh - ceg - bdi - afh
        |d e f|
        |g h i|
```

**General (Cofactor Expansion):**
```
det(A) = Σⱼ₌₁ⁿ (-1)^(i+j) × aᵢⱼ × Mᵢⱼ

Where Mᵢⱼ is the minor (determinant of the submatrix after removing row i, col j)
```

### Key Properties

```
det(AB) = det(A) × det(B)
det(Aᵀ) = det(A)
det(A⁻¹) = 1/det(A)
det(kA) = kⁿ × det(A)  (n×n matrix scaled by k)
det(I) = 1
```

### Python Implementation

```python
import numpy as np

A = np.array([[3, 1], [2, 4]])
print(f"det(A) = {np.linalg.det(A):.2f}")  # 10.0

# Verify: 3×4 - 1×2 = 12 - 2 = 10 ✓

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"det(B) = {np.linalg.det(B):.6f}")  # ≈ 0 (singular!)

# Why does this matter? 
try:
    B_inv = np.linalg.inv(B)
except np.linalg.LinAlgError:
    print("B is NOT invertible (det ≈ 0) — this means the system has no unique solution")
```

### ML Connection: Invertibility in Linear Regression

```python
# In linear regression: w = (XᵀX)⁻¹Xᵀy
# XᵀX must be invertible, which means det(XᵀX) ≠ 0

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

XtX = X.T @ X
det_XtX = np.linalg.det(XtX)
print(f"det(XᵀX) = {det_XtX:.4e}")  # Large number → invertible

# If features are collinear:
X[:, 1] = X[:, 0] + X[:, 2]  # Make column 1 = col 0 + col 2
XtX_bad = X.T @ X
det_XtX_bad = np.linalg.det(XtX_bad)
print(f"det(XᵀX) with collinearity = {det_XtX_bad:.6e}")  # ≈ 0!
print("→ Need regularization (Ridge) to handle this!")
```

## Matrix Rank

### Definition

The **rank** of a matrix is the:
- Maximum number of **linearly independent rows**
- Maximum number of **linearly independent columns**  
- Dimension of the **column space** (range)
- Dimension of the **row space**

```
rank(A) ≤ min(m, n)  for an m×n matrix
```

### Intuition

```
Rank = Number of "dimensions" your data actually spans

3D data that lies on a 2D plane → rank = 2
3D data that lies on a 1D line → rank = 1
3D data that fills 3D space → rank = 3

┌──────────────────────────────────────┐
│  A = [1  2  3]    rank(A) = 1      │
│      [2  4  6]    (row 2 = 2×row 1)│
│      [3  6  9]    (row 3 = 3×row 1)│
│                                      │
│  All rows are linearly dependent!   │
│  Only 1 independent direction.      │
└──────────────────────────────────────┘
```

### Properties

```
rank(A) = rank(Aᵀ)
rank(AB) ≤ min(rank(A), rank(B))
rank(A + B) ≤ rank(A) + rank(B)
rank(A) + rank(B) - n ≤ rank(AB)  (Sylvester's inequality)
A is invertible ⟺ rank(A) = n  (full rank)
nullity(A) = n - rank(A)  (Rank-Nullity Theorem)
```

### Python

```python
import numpy as np

# Full rank matrix
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(f"rank(I₃) = {np.linalg.matrix_rank(A)}")  # 3

# Rank-deficient matrix
B = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
print(f"rank(B) = {np.linalg.matrix_rank(B)}")  # 1

# ML: Check if features are linearly independent
from sklearn.datasets import load_iris
X, _ = load_iris(return_X_y=True)
print(f"Iris features: {X.shape}")
print(f"Rank of feature matrix: {np.linalg.matrix_rank(X)}")  # 4 (full rank)
print(f"→ All 4 features carry independent information")
```

### ML Application: Feature Redundancy Detection

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create features with redundancy
np.random.seed(42)
X = np.column_stack([
    np.random.normal(0, 1, 100),      # Feature 1 (independent)
    np.random.normal(0, 1, 100),      # Feature 2 (independent)
    3 * X[:, 0] + 2,                  # Feature 3 = 3×F1 + 2 (dependent!)
    X[:, 1] + X[:, 0],               # Feature 4 = F1 + F2 (dependent!)
])

X = StandardScaler().fit_transform(X)
rank = np.linalg.matrix_rank(X)
print(f"Feature matrix shape: {X.shape}")
print(f"Rank: {rank}")  # 2 (only 2 independent features out of 4!)

# Singular values reveal this
U, S, Vt = np.linalg.svd(X)
print(f"Singular values: {S.round(4)}")
print(f"Only 2 significant values → rank = 2")
print(f"→ PCA would correctly identify 2 principal components")
```

## Projections

### The Core Idea

A projection takes a vector and finds its **shadow** on a line (or plane). It's the closest point on that line/plane to the original vector.

```
Orthogonal Projection of b onto a:

      b
     /|
    / |
   /  | b - proj
  /   |
 /____|
a    projₐ(b)

projₐ(b) = (a·b / a·a) × a
```

### Projection Formulas

**Projection of vector b onto vector a:**
```
projₐ(b) = (aᵀb / aᵀa) × a
```

**Projection onto a subspace (column space of A):**
```
projᵥ(b) = A(AᵀA)⁻¹Aᵀb
```

This is EXACTLY the least squares regression formula!

### Why Projections Matter: Least Squares = Projection

```
┌──────────────────────────────────────────────────────────┐
│          LEAST SQUARES REGRESSION = PROJECTION            │
│                                                          │
│  We want to find ŷ = Xw that is CLOSEST to y           │
│  "Closest" means minimizing ||y - Xw||²                  │
│                                                          │
│  The solution: Xw = proj₍col(X)₎(y)                    │
│                                                          │
│  Geometrically: we project y onto the column space of X │
│  The residual (y - ŷ) is PERPENDICULAR to the prediction│
│                                                          │
│         y                                                │
│        /|                                                │
│       / | (residual: y - ŷ)                              │
│      /  |                                               │
│     /   |  ŷ = proj(y)                                  │
│    /____|                                               │
│   col(X)                                               │
│                                                          │
│  Normal equations: XᵀX w = Xᵀy                          │
│  Solution: w = (XᵀX)⁻¹ Xᵀy                              │
└──────────────────────────────────────────────────────────┘
```

### Python: Projections in Action

```python
import numpy as np

# Project vector b onto vector a
a = np.array([2, 1])
b = np.array([4, 5])

proj = (a @ b) / (a @ a) * a
residual = b - proj

print(f"Vector a: {a}")
print(f"Vector b: {b}")
print(f"Projection of b onto a: {proj.round(4)}")
print(f"Residual (b - proj): {residual.round(4)}")
print(f"Residual ⊥ a? {np.isclose(residual @ a, 0)}")  # True!

# This IS least squares regression with one feature!
X = a.reshape(-1, 1)
y = b
w = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"\nRegression weight: {w[0]:.4f}")
print(f"Prediction: {w[0] * a}")  # Same as projection!
```

### Gram-Schmidt Process: Building Orthogonal Bases

```python
def gram_schmidt(vectors):
    """Convert a set of vectors into an orthonormal basis."""
    orthogonal = []
    for v in vectors:
        w = v.copy().astype(float)
        for u in orthogonal:
            w -= (u @ w) / (u @ u) * u  # Subtract projections
        orthogonal.append(w)
    
    # Normalize
    orthonormal = [u / np.linalg.norm(u) for u in orthogonal]
    return np.array(orthonormal)

# Create a basis from non-orthogonal vectors
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])
v3 = np.array([0, 1, 1])

ortho = gram_schmidt([v1, v2, v3])
print("Orthonormal basis:")
for i, u in enumerate(ortho):
    print(f"  u{i+1} = {u.round(4)}")
    print(f"    ||u{i+1}|| = {np.linalg.norm(u):.4f}")

# Verify orthogonality
print(f"\nu1·u2 = {ortho[0] @ ortho[1]:.4f}")  # ≈ 0
print(f"u1·u3 = {ortho[0] @ ortho[2]:.4f}")  # ≈ 0
print(f"u2·u3 = {ortho[1] @ ortho[2]:.4f}")  # ≈ 0
```

## 🧠 Exercises

### Exercise 1: Compute the determinant of A = [[2, 3], [1, 4]]. Is A invertible? Find A⁻¹.

<details>
<summary>Solution</summary>

```python
import numpy as np
A = np.array([[2, 3], [1, 4]])
det = 2*4 - 3*1  # 5
print(f"det(A) = {det}")  # 5 ≠ 0, so invertible
A_inv = np.array([[4, -3], [-1, 2]]) / 5
print(f"A⁻¹ = \n{A_inv}")
```

</details>

### Exercise 2: Find the rank of [[1,2,3],[4,5,6],[5,7,9]] without using numpy.

<details>
<summary>Solution</summary>

Row 3 = Row 1 + Row 2, so one row is dependent.
Only 2 independent rows → rank = 2.

```python
import numpy as np
M = np.array([[1,2,3],[4,5,6],[5,7,9]])
print(np.linalg.matrix_rank(M))  # 2
```

</details>

### Exercise 3: Project the vector b = [3, 7] onto the line spanned by a = [1, 2]. Verify the residual is orthogonal to a.

<details>
<summary>Solution</summary>

```python
import numpy as np
a = np.array([1, 2])
b = np.array([3, 7])
proj = (a @ b) / (a @ a) * a  # (17/5) × [1,2] = [3.4, 6.8]
residual = b - proj  # [-0.4, 0.2]
print(f"Projection: {proj}")
print(f"Residual: {residual}")
print(f"Orthogonal? {np.isclose(residual @ a, 0)}")  # True
```

</details>
