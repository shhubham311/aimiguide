# Vector Space, Subspaces, Linear Dependence & Independence

## Why This Matters in AI/ML

Vector spaces are the language of data representation in machine learning. Every dataset, every feature vector, every word embedding lives in a vector space. Understanding subspaces helps you grasp **dimensionality reduction** (PCA), **feature spaces** in SVMs, and **latent spaces** in generative models. Linear independence tells you whether features carry redundant information — the foundation of **regularization**, **feature selection**, and **collinearity diagnostics**. Without vector space theory, you cannot understand why some models fail or how to fix them.

---

## 1. Vectors: The Building Blocks

### 1.1 What is a Vector?

A vector is an ordered list of numbers. In ML, a vector represents a data point with n features:

```
x = [x₁, x₂, ..., xₙ]ᵀ
```

**Example:** A house with 3 features (sqft, bedrooms, age):

```
x = [2000, 3, 15]ᵀ
```

### 1.2 Vector Operations

```
Addition:        x + y = [x₁+y₁, x₂+y₂, ..., xₙ+yₙ]ᵀ
Scalar Multiply: αx    = [αx₁, αx₂, ..., αxₙ]ᵀ
Dot Product:     x·y   = Σᵢ xᵢyᵢ  (= xᵀy in matrix notation)
Norm (Length):   ‖x‖   = √(x·x) = √(Σᵢ xᵢ²)
```

```
  Geometric View (2D):

  y
  ^
  |      • (3,4)    ||(3,4)|| = √(9+16) = 5
  |     /|
  |  4 / |
  |   /  |
  |  /   | 4
  | /    |
  |/     |
  +------•----> x
  0      3

  Unit vectors:
  ê₁ = [1,0], ê₂ = [0,1]
  û = x/||x|| = [3/5, 4/5]
```

---

## 2. Vector Spaces: Formal Definition

### 2.1 Definition

A **vector space** V over a field F (usually real numbers ℝ) is a set equipped with two operations:

1. **Vector addition:** + : V × V → V
2. **Scalar multiplication:** · : F × V → V

satisfying **8 axioms** for all u, v, w ∈ V and α, β ∈ F:

| # | Axiom | Description |
|---|-------|-------------|
| 1 | u + v = v + u | Commutativity of addition |
| 2 | (u + v) + w = u + (v + w) | Associativity of addition |
| 3 | ∃0 ∈ V : v + 0 = v | Additive identity (zero vector) |
| 4 | ∃(-v) ∈ V : v + (-v) = 0 | Additive inverse |
| 5 | α(βv) = (αβ)v | Associativity of scalar mult. |
| 6 | 1·v = v | Scalar multiplicative identity |
| 7 | α(u + v) = αu + αv | Distributivity over vector addition |
| 8 | (α + β)v = αv + βv | Distributivity over scalar addition |

### 2.2 Important Examples

```
ℝⁿ        : n-dimensional real vectors (the main ML vector space)
ℂⁿ        : n-dimensional complex vectors
Pₙ        : Polynomials of degree ≤ n
C[a,b]    : Continuous functions on [a,b]
M(m,n)    : m×n matrices
{0}       : The trivial vector space (just the zero vector)
```

### 2.3 The ML Perspective

In ML, your data lives in ℝⁿ where n = number of features:

```
Dataset: X ∈ ℝ^(m×n)
- m = number of samples
- n = number of features (dimension of the feature space)
- Each row xᵢ ∈ ℝⁿ is a point in the vector space
```

---

## 3. Subspaces

### 3.1 Definition

A **subspace** W of a vector space V is a subset of V that is itself a vector space under the same operations.

### 3.2 Subspace Test

W is a subspace of V if and only if:
1. **Zero vector:** 0 ∈ W
2. **Closed under addition:** u, v ∈ W → u + v ∈ W
3. **Closed under scalar multiplication:** u ∈ W, α ∈ F → αu ∈ W

### 3.3 Important Subspaces of ℝⁿ

```
1. The trivial subspaces: {0} and ℝⁿ itself
2. Lines through the origin: {αv : α ∈ ℝ} for fixed v ≠ 0
3. Planes through the origin: {αu + βv : α, β ∈ ℝ} for fixed u, v
4. Hyperplanes through the origin
5. Column space (range) of a matrix
6. Null space (kernel) of a matrix
```

```
  Subspaces in ℝ³:

     z                   z                    z
     |                   |                    |
     |   W: plane        |  W: line           |  W = ℝ³
     |   through         |  through           |  (everything)
     |   origin          |  origin            |
     |  /                |   /                |
     | /                 |  /                 |
     |/__________________ |/__________________|/________________
    /|                   /|                   /|
   / |                  / |                  / |
     x                   x                    x
  y                     y                    y

  W = {αv₁ + βv₂ :     W = {αv : α∈ℝ}     W = ℝ³ (dim = 3)
    α, β ∈ ℝ}          (dim = 1)            (dim = 3)
  (dim = 2)
```

### 3.4 Column Space and Null Space (Crucial for ML)

Given a matrix A ∈ ℝ^(m×n):

**Column Space (Range):**
```
Col(A) = {Ax : x ∈ ℝⁿ} ⊆ ℝᵐ
```
The set of all possible outputs of the linear transformation A.
- **In ML:** The column space represents all predictions a linear model can make.

**Null Space (Kernel):**
```
Nul(A) = {x ∈ ℝⁿ : Ax = 0} ⊆ ℝⁿ
```
The set of all inputs that produce zero output.
- **In ML:** Null space vectors represent features that the model completely ignores.

**Fundamental Theorem of Linear Algebra:**
```
dim(Col(A)) + dim(Nul(A)) = n
```

---

## 4. Linear Dependence and Independence

### 4.1 Definition

Vectors v₁, v₂, ..., vₖ are **linearly independent** if the only solution to:

```
α₁v₁ + α₂v₂ + ... + αₖvₖ = 0
```

is α₁ = α₂ = ... = αₖ = 0.

They are **linearly dependent** if there exist coefficients (not all zero) satisfying the equation.

### 4.2 Geometric Interpretation

```
  2D Examples:

  Independent (span ℝ²):    Dependent (span a line):
  
     ^                          ^
     |  v₂                      |  v₂
     |  /                       |  /
     | /                        | /
     |/ v₁                      |/ v₁
     +------->                  +------->
     v₂ is NOT a scalar         v₂ = 2v₁ (redundant!)
     multiple of v₁

  3D Examples:
  
  Independent (span ℝ³):       Dependent (span a plane):
  
  v₃                           v₃
  |   ↗ v₂                      |   ↗ v₂
  |  /                          |  /
  | /                           | /
  |/  v₁                        |/  v₁
  +----->                       +----->
  
  No vector can be written       v₃ = αv₁ + βv₂ (lies in
  as a combination of others     the span of v₁, v₂)
```

### 4.3 Testing Linear Independence

Form the matrix with vectors as columns and check rank:

```python
import numpy as np

def are_independent(vectors):
    """Check if a list of vectors is linearly independent."""
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    n_cols = A.shape[1]
    is_indep = rank == n_cols
    return is_indep, rank, n_cols

# Example 1: Independent vectors in ℝ³
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
ind, rank, n = are_independent([v1, v2, v3])
print(f"Standard basis: independent={ind}, rank={rank}, n={n}")

# Example 2: Dependent vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # v2 = 2*v1
ind, rank, n = are_independent([v1, v2])
print(f"v2=2*v1: independent={ind}, rank={rank}, n={n}")

# Example 3: ML feature matrix
np.random.seed(42)
X = np.random.randn(100, 5)
X[:, 3] = X[:, 0] + X[:, 1]  # Feature 3 is dependent!
ind, rank, n = are_independent([X[:, i] for i in range(5)])
print(f"Features with redundancy: independent={ind}, rank={rank}, n={n}")
```

### 4.4 ML Connection: Multicollinearity

When features are linearly dependent (or nearly so):

```
Problems:
- Coefficients become unstable (small data changes → large weight changes)
- The design matrix XᵀX becomes singular or near-singular
- Individual feature importance cannot be determined

Solutions:
- Remove redundant features (feature selection)
- Use regularization (Ridge: adds λI to make XᵀX invertible)
- Use PCA to decorrelate features
```

---

## 5. Basis and Dimension

### 5.1 Basis

A **basis** for a vector space V is a set of vectors that:
1. Spans V (every vector in V can be written as a combination)
2. Is linearly independent (no redundancy)

**Key properties:**
- Every vector space has a basis
- The number of vectors in any basis is the same → this is the **dimension**

### 5.2 Common Bases

```
Standard basis of ℝ³: e₁ = [1,0,0]ᵀ, e₂ = [0,1,0]ᵀ, e₃ = [0,0,1]ᵀ

Polynomial basis of P₂: {1, x, x²}
  Any quadratic ax² + bx + c = a·x² + b·x + c·1

Fourier basis: {1, sin(x), cos(x), sin(2x), cos(2x), ...}
  Used in signal processing and feature engineering
```

### 5.3 Dimension

```
dim(ℝⁿ) = n
dim({0}) = 0
dim(Pₙ) = n + 1
dim(Col(A)) = rank(A)
```

### 5.4 Change of Basis (Critical for PCA, Word Embeddings)

The same vector can have different coordinate representations in different bases:

```
  Standard Basis:       Rotated Basis:

  v = [3, 1]ᵀ in ê₁,ê₂  v = [a, b]ᵀ in u₁,u₂

     ^ ê₂                ^ u₂
     |  • (3,1)           |  • (3,1)
     | /                  | /
     |/                   |/
     +----> ê₁            +----> u₁

  The vector is the SAME geometric object.
  Only its coordinates change with the basis.
```

```python
import numpy as np

# Change of basis in 2D
v = np.array([3.0, 1.0])  # Vector in standard basis

# New basis: u1 = [1,1], u2 = [-1,1]
B = np.array([[1, -1],   # Columns are new basis vectors
              [1,  1]])

# Transform to new coordinates: v_new = B^{-1} v
v_new = np.linalg.inv(B) @ v
print(f"Standard coords: {v}")
print(f"New basis coords: {v_new}")
print(f"Verify: B @ v_new = {B @ v_new}  (should equal v)")

# ML application: PCA is essentially a change of basis
# to the eigenbasis of the covariance matrix
```

---

## 6. Span

### 6.1 Definition

The **span** of a set of vectors {v₁, ..., vₖ} is the set of all linear combinations:

```
span{v₁, ..., vₖ} = {α₁v₁ + α₂v₂ + ... + αₖvₖ : αᵢ ∈ ℝ}
```

### 6.2 Span in ML

```
The span of feature vectors determines the "expressiveness" of a model:

- n linearly independent features span ℝⁿ → model can represent any
  linear function on ℝⁿ
- Fewer than n independent features → model has blind spots
- Kernel trick: maps data to span of kernel functions (possibly ∞-dimensional)
```

---

## 7. Inner Product Spaces

### 7.1 Definition

An **inner product** on V is a function ⟨·,·⟩: V × V → ℝ satisfying:

```
1. ⟨x, y⟩ = ⟨y, x⟩           (Symmetry)
2. ⟨αx, y⟩ = α⟨x, y⟩         (Linearity in first argument)
3. ⟨x + z, y⟩ = ⟨x, y⟩ + ⟨z, y⟩  (Additivity)
4. ⟨x, x⟩ ≥ 0, = 0 iff x = 0 (Positive definiteness)
```

### 7.2 The Standard Inner Product (Dot Product)

```
⟨x, y⟩ = x₁y₁ + x₂y₂ + ... + xₙyₙ = xᵀy
```

### 7.3 Orthogonality

Two vectors are **orthogonal** if ⟨x, y⟩ = 0.

```
  Orthogonal vectors in ℝ²:
  
  ^      • v = [0, 3]
  |      |
  |      |
  |      |
  +------•-----> u = [4, 0]
  
  u · v = 4·0 + 0·3 = 0  (orthogonal!)
```

### 7.4 ML Applications of Inner Products

**Cosine Similarity:**
```
cos(θ) = (x · y) / (‖x‖ · ‖y⟩)
```
Used for comparing text embeddings, recommendation systems, document similarity.

**Kernel Functions:**
```
K(x, y) = ⟨φ(x), φ(y)⟩
```
Kernels compute inner products in high-dimensional feature spaces implicitly.

```python
import numpy as np

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Document embeddings (simplified)
doc1 = np.array([1, 1, 0, 0, 1])  # "machine learning algorithms"
doc2 = np.array([1, 1, 1, 0, 0])  # "machine learning data science"
doc3 = np.array([0, 0, 1, 1, 0])  # "data science statistics"

print(f"doc1 vs doc2: {cosine_similarity(doc1, doc2):.4f}")  # High similarity
print(f"doc1 vs doc3: {cosine_similarity(doc1, doc3):.4f}")  # Low similarity
print(f"doc2 vs doc3: {cosine_similarity(doc2, doc3):.4f}")  # Medium similarity
```

---

## 8. Exercises with Solutions

### Exercise 1
Determine if the following vectors in ℝ³ are linearly independent:
v₁ = [1, 2, 3], v₂ = [4, 5, 6], v₃ = [7, 8, 9].

**Solution:** Form the matrix and compute determinant:

```
A = [[1, 4, 7],
     [2, 5, 8],
     [3, 6, 9]]

det(A) = 1(45-48) - 4(18-24) + 7(12-15)
       = -3 + 24 - 21 = 0

Since det(A) = 0, the vectors are linearly dependent.

Indeed: v₃ = 2v₂ - v₁
```

### Exercise 2
Find a basis for the null space of A = [[1, 2, -1], [2, 4, -2]].

**Solution:** Row reduce:
```
RREF: [[1, 2, -1],
       [0, 0,  0]]

Free variables: x₂ and x₃
x₁ = -2x₂ + x₃

General solution: x = x₂[-2, 1, 0]ᵀ + x₃[1, 0, 1]ᵀ

Basis for Nul(A): {[-2, 1, 0]ᵀ, [1, 0, 1]ᵀ}
dim(Nul(A)) = 2
```

### Exercise 3
Show that the set W = {(x, y, z) : x + y + z = 0} is a subspace of ℝ³.

**Solution:** Check the three conditions:
```
1. Zero: 0 + 0 + 0 = 0 ✓, so 0 = (0,0,0) ∈ W
2. Closed under addition: if u, v ∈ W:
   (u₁+u₂+u₃) + (v₁+v₂+v₃) = 0 + 0 = 0 ✓
3. Closed under scalar mult: if u ∈ W:
   α(u₁+u₂+u₃) = α·0 = 0 ✓

W is a subspace. It's a plane through the origin.
dim(W) = 2 (basis: [1,-1,0]ᵀ and [1,0,-1]ᵀ)
```

### Exercise 4
In the context of ML, explain why having linearly dependent features is problematic for linear regression.

**Solution:**
Linear regression solves: w* = (XᵀX)⁻¹Xᵀy

If columns of X are linearly dependent, then XᵀX is singular (det = 0), meaning (XᵀX)⁻¹ doesn't exist. There is no unique solution for w. Even if features are *nearly* dependent, XᵀX becomes ill-conditioned, making w extremely sensitive to small perturbations in the data. This is why Ridge regression (which adds λI to XᵀX) is used in practice.

---

## Summary

| Concept | Definition | ML Application |
|---------|-----------|----------------|
| **Vector Space** | Set closed under addition & scalar mult. | Feature space ℝⁿ |
| **Subspace** | Subset that is itself a vector space | Column/null space of X |
| **Span** | All linear combinations of a set of vectors | Model expressiveness |
| **Linear Independence** | No vector is a combination of others | Feature redundancy detection |
| **Basis** | Maximal independent set / minimal spanning set | Data representation |
| **Dimension** | Number of vectors in a basis | Intrinsic data complexity |
| **Inner Product** | Generalized dot product | Cosine similarity, kernels |
| **Orthogonality** | Inner product = 0 | Feature decorrelation, PCA |
