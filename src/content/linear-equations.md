# System of Linear Equations & Solutions

## Why This Matters in AI/ML

Systems of linear equations are solved every time you train a linear model. The **normal equations** (XᵀX)w = Xᵀy for linear regression, the **update equations** in iterative methods, and the **forward/backward passes** in neural networks all involve solving systems of linear equations. Understanding when solutions exist, when they are unique, and how to find them efficiently is essential for implementing and debugging ML algorithms.

---

## 1. Formulation

### 1.1 Matrix Form

A system of m linear equations in n unknowns:

```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
  ...           ...              ...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ
```

Written compactly as:

```
Ax = b

where:
A ∈ ℝ^(m×n) = coefficient matrix
x ∈ ℝⁿ     = unknown vector
b ∈ ℝᵐ     = right-hand side vector
```

### 1.2 Geometric Interpretation

```
2 Equations, 2 Unknowns (unique solution):

Line 1: x + y = 3     /
                    / • (1, 2)
                  /     \ 
                /         \
Line 2: x - y = -1       \ 
                            \
  Two lines intersect at one point = unique solution.

  2 Equations, 2 Unknowns (no solution):

Line 1: x + y = 3     /
                    /
                  /
Line 2: x + y = 5   (parallel line above)

  Parallel lines = no solution (inconsistent).

  2 Equations, 2 Unknowns (infinitely many):

Line 1: x + y = 3     /
                    /
                  /
Line 2: 2x + 2y = 6 (same line!)

  Coincident lines = infinitely many solutions.
```

---

## 2. Solution Theory

### 2.1 The Three Possibilities

| Case | Condition | Solutions |
|------|-----------|-----------|
| **Unique** | rank(A) = rank([A\|b]) = n | Exactly one |
| **Infinite** | rank(A) = rank([A\|b]) < n | Infinitely many (parametrized) |
| **None** | rank(A) < rank([A\|b]) | No solution (inconsistent) |

where rank([A\|b]) is the rank of the **augmented matrix**.

### 2.2 Existence: When Does a Solution Exist?

A solution exists if and only if:

```
b ∈ Col(A)    (b is in the column space of A)
```

Equivalently: rank(A) = rank([A\|b]).

### 2.3 Uniqueness: When Is the Solution Unique?

If a solution exists, it is unique if and only if:

```
Nul(A) = {0}    (null space contains only zero)
```

Equivalently: rank(A) = n (number of columns).

### 2.4 Complete Classification

```
For A ∈ ℝ^(m×n):

m = n = rank(A)     → Unique solution    (square, full rank)
m = n, rank(A) < n  → Infinite or none   (square, singular)
m > n, rank(A) = n  → Unique solution    (overdetermined, full rank)
m > n, rank(A) < n  → Infinite or none   (overdetermined)
m < n, rank(A) = m  → Infinite solutions (underdetermined, full row rank)
m < n, rank(A) < m  → Infinite or none   (underdetermined)
```

---

## 3. Gaussian Elimination

### 3.1 Row Operations

Three elementary row operations that preserve the solution set:

```
1. Swap two rows:      Rᵢ ↔ Rⱼ
2. Multiply row by c:  Rᵢ → c·Rᵢ    (c ≠ 0)
3. Add multiple of row: Rᵢ → Rᵢ + c·Rⱼ
```

### 3.2 Row Echelon Form (REF)

A matrix is in REF if:
1. All zero rows are at the bottom
2. The first nonzero entry (pivot) in each row is to the right of the pivot above
3. Pivot entries can be any nonzero value

### 3.3 Reduced Row Echelon Form (RREF)

A matrix is in RREF if:
1. It is in REF
2. Each pivot is 1
3. Each pivot is the only nonzero entry in its column

```
Example:

Original:                RREF:
[2  4  -2  | 6]         [1  2  0  | 4]
[1  3   1  | 4]   →     [0  1  1  | 0]
[3  7   0  | 10]        [0  0  0  | 0]

Solution: x₃ = t (free), x₂ = -t, x₁ = 4 + 2t
General: [x₁, x₂, x₃]ᵀ = [4, 0, 0]ᵀ + t[-2, -1, 1]ᵀ
```

```python
import numpy as np

def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination with back substitution."""
    # Augmented matrix
    Ab = np.column_stack([A.astype(float), b.astype(float)])
    n = len(b)
    
    # Forward elimination
    for col in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(Ab[col:, col])) + col
        Ab[[col, max_row]] = Ab[[max_row, col]]
        
        if abs(Ab[col, col]) < 1e-12:
            continue
        
        # Eliminate below
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]
    
    # Back substitution
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        if abs(Ab[row, row]) < 1e-12:
            continue
        x[row] = (Ab[row, -1] - np.dot(Ab[row, row+1:n], x[row+1:])) / Ab[row, row]
    
    return x

# Example
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

x = gaussian_elimination(A, b)
print(f"Solution: x = {x}")
print(f"Verify: Ax = {A @ x}")  # Should equal b
```

---

## 4. Solving Methods

### 4.1 Direct Methods

**LU Decomposition** (see separate lesson):
```
Ax = b  →  LUx = b  →  Ly = b, Ux = y
```

**Matrix Inverse:**
```
x = A⁻¹b
```
Computationally expensive (O(n³) for inversion, O(n²) for multiplication).

### 4.2 Iterative Methods

Used for large, sparse systems (common in deep learning and scientific computing).

**Jacobi Method:**
```
x⁽ᵏ⁺¹⁾_i = (bᵢ - Σⱼ≠ᵢ aᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ
```

**Gauss-Seidel Method:**
```
x⁽ᵏ⁺¹⁾_i = (bᵢ - Σⱼ<ᵢ aᵢⱼxⱼ⁽ᵏ⁺¹⁾ - Σⱼ>ᵢ aᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ
```
Uses updated values immediately (faster convergence than Jacobi).

```python
import numpy as np

def jacobi(A, b, max_iter=100, tol=1e-10):
    """Jacobi iterative method for solving Ax = b."""
    n = len(b)
    x = np.zeros(n)
    D = np.diag(np.diag(A))        # Diagonal of A
    R = A - D                       # Rest of A
    
    for k in range(max_iter):
        x_new = np.linalg.solve(D, b - R @ x)
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def gauss_seidel(A, b, max_iter=100, tol=1e-10):
    """Gauss-Seidel iterative method."""
    n = len(b)
    x = np.zeros(n)
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
        
        if np.linalg.norm(x - x_old) < tol:
            return x, k + 1
    return x, max_iter

# Example: diagonally dominant system
A = np.array([[10, -1, 2],
              [-1, 11, -1],
              [2, -1, 10]])
b = np.array([6, 25, -11])

x_jac, k_jac = jacobi(A, b)
x_gs, k_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)

print(f"Jacobi:      x = {x_jac}, iterations: {k_jac}")
print(f"Gauss-Seidel: x = {x_gs}, iterations: {k_gs}")
print(f"Exact:       x = {x_exact}")
```

```
  Convergence of Iterative Methods:

  Error vs Iteration:
  
  Error
    ^
  1 |*
    | *
    |  *
    |   *
    |     *  Gauss-Seidel (faster)
    |       *
    |         *
  0.1|-----------*-----*------- Jacobi
    |                         *
    +---+---+---+---+---+---+---> Iteration
    0   5   10  15  20  25  30
```

---

## 5. The Overdetermined System: Least Squares

### 5.1 The Problem

When m > n (more equations than unknowns), a solution may not exist. Instead, find the **best approximate solution**:

```
Minimize: ‖Ax - b‖²
```

This is the **least squares problem** — the foundation of linear regression.

### 5.2 Normal Equations

Setting the gradient to zero:

```
AᵀA x = Aᵀb

Solution: x* = (AᵀA)⁻¹ Aᵀb
```

```
  Geometric Interpretation:

  b is projected onto Col(A):
  
       • b
      /|
     / | (b - Ax*)  residual
    /  |
   /   |
  •----•
 Ax*  Col(A)
 
  Ax* is the closest point to b in Col(A).
  The residual (b - Ax*) is orthogonal to Col(A).
```

### 5.3 Linear Regression as Least Squares

```python
import numpy as np

# Generate data: y = 2x + 1 + noise
np.random.seed(42)
m = 100  # number of samples
X = 2 * np.random.rand(m, 1)
y = 2 * X + 1 + 0.5 * np.random.randn(m, 1)

# Add intercept: X_aug = [1, x]
X_aug = np.column_stack([np.ones(m), X])

# Normal equations: (XᵀX)w = Xᵀy
w = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
print(f"Normal equations: w₀ = {w[0,0]:.4f}, w₁ = {w[1,0]:.4f}")

# Compare with NumPy's built-in
w_np, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
print(f"NumPy lstsq:      w₀ = {w_np[0,0]:.4f}, w₁ = {w_np[1,0]:.4f}")

# Mean Squared Error
y_pred = X_aug @ w
mse = np.mean((y - y_pred)**2)
print(f"MSE: {mse:.4f}")
```

---

## 6. Underdetermined Systems

### 6.1 The Problem

When m < n (fewer equations than unknowns), if a solution exists, there are infinitely many.

### 6.2 Minimum Norm Solution

Among all solutions, choose the one with minimum norm:

```
Minimize: ‖x‖²
Subject to: Ax = b

Solution: x* = Aᵀ(AAᵀ)⁻¹b
```

```python
import numpy as np

# Underdetermined system: 2 equations, 3 unknowns
A = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([1, 2])

# General solution: many solutions exist
# Minimum norm solution using pseudo-inverse
x_min_norm = np.linalg.pinv(A) @ b
print(f"Minimum norm solution: {x_min_norm}")
print(f"Norm: {np.linalg.norm(x_min_norm):.4f}")

# Verify it satisfies Ax = b
print(f"Ax: {A @ x_min_norm} (should be [1, 2])")

# Compare with another solution
x_other = np.array([1, -1, 0.333])
print(f"\nAnother solution: {x_other}")
print(f"Ax: {A @ x_other}")
print(f"Norm: {np.linalg.norm(x_other):.4f} (larger!)")
```

### 6.3 Regularized Solutions

In ML, we often prefer:

```
Minimize: ‖Ax - b‖² + λ‖x‖²

Solution: x* = (AᵀA + λI)⁻¹Aᵀb  (Ridge regression!)
```

This always has a unique solution, even when AᵀA is singular.

---

## 7. Condition Number and Numerical Stability

### 7.1 Condition Number

```
κ(A) = ‖A‖ · ‖A⁻¹‖ = σ_max / σ_min
```

**Interpretation:**
- κ = 1: perfectly conditioned (orthogonal matrix)
- κ >> 1: ill-conditioned (small changes in b cause large changes in x)

```python
import numpy as np

# Well-conditioned vs ill-conditioned
A_good = np.array([[1, 0.1], [0.1, 1]])
A_bad = np.array([[1, 0.999], [0.999, 1]])

print(f"Well-conditioned: κ = {np.linalg.cond(A_good):.2f}")
print(f"Ill-conditioned:  κ = {np.linalg.cond(A_bad):.2f}")

# Effect on solution accuracy
b = np.array([1, 1])
b_noisy = b + 1e-10 * np.random.randn(2)

x_exact = np.linalg.solve(A_good, b)
x_noisy = np.linalg.solve(A_good, b_noisy)
print(f"\nWell-conditioned perturbation: {np.linalg.norm(x_exact - x_noisy):.2e}")

x_exact = np.linalg.solve(A_bad, b)
x_noisy = np.linalg.solve(A_bad, b_noisy)
print(f"Ill-conditioned perturbation:  {np.linalg.norm(x_exact - x_noisy):.2e}")
```

### 7.2 ML Implications

- **Feature scaling** dramatically affects condition number
- **Multicollinearity** causes high condition numbers → unstable coefficients
- **Regularization** (Ridge) improves conditioning: κ(AᵀA + λI) << κ(AᵀA)

---

## 8. Exercises with Solutions

### Exercise 1
Solve using Gaussian elimination:
```
x + 2y - z = 3
2x - y + z = 1
3x + y - 2z = 4
```

**Solution:** Augmented matrix:
```
[1  2 -1 | 3]     RREF:     [1  0  0 | 7/6]
[2 -1  1 | 1]  →          →  [0  1  0 | 1/6]
[3  1 -2 | 4]              [0  0  1 | 1/3]

x = 7/6, y = 1/6, z = 1/3
```

### Exercise 2
Determine the number of solutions for:
```
x + y + z = 1
2x + 2y + 2z = 3
```

**Solution:**
```
A = [[1,1,1],[2,2,2]], rank(A) = 1
[A|b] = [[1,1,1,1],[2,2,2,3]], rank = 2

Since rank(A) ≠ rank([A|b]), no solution exists.
(Geometrically: two parallel planes in ℝ³.)
```

### Exercise 3
Set up the normal equations for fitting y = w₀ + w₁x to the data points (1,2), (2,3), (3,5).

**Solution:**
```
Design matrix:  X = [[1,1],[1,2],[1,3]]
Targets:        y = [2, 3, 5]

Normal equations: XᵀXw = Xᵀy

XᵀX = [[3, 6],[6, 14]]
Xᵀy = [10, 23]

[3 6 ][w₀]   [10]
[6 14][w₁] = [23]

Solving: w₀ = 1/3, w₁ = 3/2

Best fit line: y = 1/3 + (3/2)x
```

### Exercise 4
Why does Ridge regression (adding λI to XᵀX) help when X has linearly dependent columns?

**Solution:** If X has dependent columns, rank(X) < n, so XᵀX is singular (not invertible). Adding λI shifts all eigenvalues by λ:

```
eigvals(XᵀX + λI) = eigvals(XᵀX) + λ

Zero eigenvalues become λ > 0, making XᵀX + λI positive definite and invertible.
Condition number improves from ∞ to σ_max²/λ + 1.
```

---

## Summary

| Concept | Formula | When to Use |
|---------|---------|-------------|
| **Direct Solve** | x = A⁻¹b | Small, square, well-conditioned |
| **LU Decomposition** | LUx = b | Efficient for multiple b vectors |
| **Jacobi/Gauss-Seidel** | Iterative | Large, sparse, diagonally dominant |
| **Least Squares** | x = (AᵀA)⁻¹Aᵀb | Overdetermined (m > n) |
| **Min Norm** | x = Aᵀ(AAᵀ)⁻¹b | Underdetermined (m < n) |
| **Ridge** | x = (AᵀA + λI)⁻¹Aᵀb | Ill-conditioned or singular |
