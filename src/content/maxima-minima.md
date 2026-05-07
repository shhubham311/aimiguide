# Maxima & Minima

## Why This Matters in AI/ML

Finding maxima and minima is the **central operation** in all of machine learning. Training a model means **minimizing a loss function**. Finding the maximum likelihood estimate means **maximizing a probability function**. Regularization introduces trade-offs between fitting the data and model complexity. Every time you call `model.fit()` or run `optimizer.step()`, you are performing a search for a minimum. This lesson covers the theory behind these operations.

---

## 1. Classification of Extrema

### 1.1 Definitions

- **Local Maximum:** f(c) ≥ f(x) for all x in some open interval around c
- **Local Minimum:** f(c) ≤ f(x) for all x in some open interval around c
- **Global Maximum:** f(c) ≥ f(x) for all x in the domain
- **Global Minimum:** f(c) ≤ f(x) for all x in the domain
- **Absolute/Global Extrema:** The largest/smallest values of f on its entire domain

```
  f(x)
   ^
   |     • Local Max        • Global Max
   |    / \                / \
   |   /   \              /   \
   |  /     \     •      /     \
   | /   •   \   / \    /       \
   |/   Loc   \ /   \  /         \
   •  Min     Local  \/  •Global  \
   |    Min    Max       Min       \
   |                               \
   +----------------------------------> x
```

### 1.2 Fermat's Theorem (Necessary Condition)

If f has a local extremum at c and f'(c) exists, then:

```
f'(c) = 0
```

Points where f'(c) = 0 are called **critical points** (or stationary points).

**Important:** Not every critical point is an extremum! (See inflection points below.)

### 1.3 Types of Critical Points

```
  Local Max          Saddle/Inflection         Local Min
  
    /                 \                      /
   /                   \                    /
  •  f'=0              • f'=0            • f'=0
   \                   /                    \
    \                 /                      \
   f''<0            f''=0                   f''>0
 (concave down)  (need higher-order test) (concave up)
```

---

## 2. The First Derivative Test

### 2.1 Method

1. Find all critical points: solve f'(x) = 0
2. Determine the sign of f' on each side of the critical point
3. Classify:

| f' before c | f' after c | Classification |
|------------|-----------|----------------|
| positive (+) | negative (−) | Local Maximum |
| negative (−) | positive (+) | Local Minimum |
| same sign | same sign | Neither (inflection point) |

### 2.2 Example

```
f(x) = x³ - 3x + 2
f'(x) = 3x² - 3 = 3(x - 1)(x + 1)

Critical points: x = 1 and x = -1

At x = -1:  f' goes from + to −  →  Local Maximum  [f(-1) = 4]
At x =  1:  f' goes from − to +  →  Local Minimum  [f(1) = 0]

  f(x)
   ^
  4|   •
   |  / \
   | /   \
   |/     \
  0+-------•-----> x
   |        \
   |         \
  -2|          •
   |
  -1       1
```

---

## 3. The Second Derivative Test

### 3.1 Method

For a critical point c where f'(c) = 0:

```
If f''(c) > 0  →  Local Minimum  (function is concave up)
If f''(c) < 0  →  Local Maximum  (function is concave down)
If f''(c) = 0  →  Test is inconclusive (use first derivative test)
```

### 3.2 Geometric Interpretation

```
f'' > 0 (concave up):    f'' < 0 (concave down):
     .                    .
    / \                  / \
   /   \                •   \     f''(c) > 0 means 
  /     \              /      \   the curve "smiles" 
 •       •            •        •  (minimum)
(Min)   (Min)       (Max)    (Max)
```

### 3.3 Why It Works (Taylor Expansion)

At a critical point f'(c) = 0, the second-order Taylor expansion is:

```
f(c+h) ≈ f(c) + (1/2)f''(c)h²
```

- If f''(c) > 0: f(c+h) > f(c) for all small h → **minimum**
- If f''(c) < 0: f(c+h) < f(c) for all small h → **maximum**

---

## 4. Multivariate Extension: Critical Points in Higher Dimensions

### 4.1 Gradient = Zero

For f(x₁, x₂, ..., xₙ), a critical point satisfies:

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] = [0, 0, ..., 0]
```

### 4.2 The Hessian Matrix

The Hessian is the matrix of all second partial derivatives:

```
H = [∂²f/∂x₁²    ∂²f/∂x₁∂x₂   ...  ∂²f/∂x₁∂xₙ  ]
    [∂²f/∂x₂∂x₁   ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ  ]
    [  ...           ...         ...    ...        ]
    [∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²   ]
```

### 4.3 Hessian-Based Classification

| Hessian Property | Type of Critical Point |
|-----------------|----------------------|
| All eigenvalues > 0 (positive definite) | **Local Minimum** |
| All eigenvalues < 0 (negative definite) | **Local Maximum** |
| Mixed signs (indefinite) | **Saddle Point** |
| Some eigenvalues = 0 | Inconclusive |

```
  Local Min (positive definite)    Saddle Point (indefinite)
  
       z                                z
       |  •                             |     •
       | / \                            |    / \
       |/   \                           |   /   \
       /     \                          |  /     \
      /       \                         | /       \
     /         \                        |/    •-----\
    +------------> x,y                  +------------> x,y
  Bowl shape (all curvatures positive)  Saddle: min in one dir, max in another
```

```python
import numpy as np

def classify_critical_point(hessian):
    """Classify a critical point using the Hessian eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(hessian)
    print(f"Eigenvalues: {eigenvalues}")
    
    if np.all(eigenvalues > 0):
        return "Local Minimum (Positive Definite)"
    elif np.all(eigenvalues < 0):
        return "Local Maximum (Negative Definite)"
    elif np.all(eigenvalues == 0):
        return "Inconclusive (Zero Hessian)"
    else:
        return "Saddle Point (Indefinite)"

# Example 1: f(x,y) = x² + y²  (clear minimum at origin)
H1 = np.array([[2.0, 0.0], [0.0, 2.0]])
print(f"f = x²+y²: {classify_critical_point(H1)}")

# Example 2: f(x,y) = x² - y²  (saddle point at origin)
H2 = np.array([[2.0, 0.0], [0.0, -2.0]])
print(f"f = x²-y²: {classify_critical_point(H2)}")

# Example 3: f(x,y) = -x² - y²  (clear maximum at origin)
H3 = np.array([[-2.0, 0.0], [0.0, -2.0]])
print(f"f = -x²-y²: {classify_critical_point(H3)}")
```

---

## 5. Global vs Local: The Challenge in ML

### 5.1 Non-Convexity in Deep Learning

Loss landscapes of neural networks are **highly non-convex** with millions of local minima and saddle points:

```
  Loss
   ^
   | /\    /\      /\
   |/  \  /  \    /  \    /\
   |    \/    \  /    \  /  \
   |           \/      \/    \
   |          • Saddle   • Local Min
   |                           \
   |        • Global Min        •
   +------------------------------> Parameters

  In high dimensions, saddle points are far more common than local minima.
  (Dauphin et al., 2014: "Identifying and attacking the saddle point problem")
```

### 5.2 Convex Functions: The Ideal Case

A function f is **convex** if for all x, y and λ ∈ [0,1]:

```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```

For twice-differentiable convex functions: **Hessian is positive semidefinite everywhere**.

**Key result:** For convex functions, every local minimum is a **global minimum**.

```
  Convex Function           Non-Convex Function
  (all local min = global)  (multiple local minima)
  
   ^                          ^
   |  \                      / \    / \
   |   \                    /   \  /   \
   |    \      •Global     /     \/     \  •Local
   |     \    /  Min       /      •Local \    Min
   |      \  /            /        Min    \
   |       \/            /                  \
   +--------->           +------------------->
```

**Common convex loss functions in ML:**
- **MSE** (Mean Squared Error): f = Σ(y - ŷ)² → convex
- **Logistic Loss:** f = -Σ[y ln(σ) + (1-y) ln(1-σ)] → convex
- **Hinge Loss (SVM):** f = max(0, 1-y·wᵀx) → convex
- **Cross-Entropy:** → convex in parameters

```python
import numpy as np

def is_convex_hessian(H):
    """Check if Hessian is positive semidefinite."""
    eigenvalues = np.linalg.eigvalsh(H)
    return np.all(eigenvalues >= -1e-10)

# MSE loss gradient and Hessian
# L(w) = (1/2n) ||Xw - y||²
# ∇L = (1/n) Xᵀ(Xw - y)
# H = (1/n) XᵀX  (always positive semidefinite!)

np.random.seed(42)
X = np.random.randn(100, 3)
H_mse = X.T @ X / X.shape[0]  # Hessian of MSE

print(f"MSE Hessian positive semidefinite: {is_convex_hessian(H_mse)}")
print(f"MSE Hessian eigenvalues: {np.linalg.eigvalsh(H_mse)}")
```

---

## 6. Constrained Extrema: Lagrange Multipliers (Preview)

In many ML problems, we optimize subject to constraints (e.g., regularization with budget).

To find extrema of f(x,y) subject to g(x,y) = c:

```
∇f = λ∇g

where λ is the Lagrange multiplier.
```

**Example in ML:** Maximizing variance subject to ||w|| = 1 leads to PCA (eigenvector problem).

---

## 7. Exercises with Solutions

### Exercise 1
Find all local extrema of f(x) = x⁴ - 4x³ + 6x² + 1.

**Solution:**
```
f'(x) = 4x³ - 12x² + 12x = 4x(x² - 3x + 3)

The quadratic x² - 3x + 3 has discriminant 9 - 12 = -3 < 0 (no real roots).

Only critical point: x = 0

f''(x) = 12x² - 24x + 12 = 12(x² - 2x + 1) = 12(x-1)²

f''(0) = 12 > 0  →  Local Minimum at x = 0, f(0) = 1
```

### Exercise 2
Find and classify the critical points of f(x,y) = x³ + y³ - 3xy.

**Solution:**
```
∂f/∂x = 3x² - 3y = 0  →  y = x²
∂f/∂y = 3y² - 3x = 0  →  x = y²

Substituting: y = (y²)² = y⁴
y⁴ - y = 0  →  y(y³ - 1) = 0
y = 0 or y = 1

Points: (0,0) and (1,1)

Hessian: H = [[6x, -3], [-3, 6y]]

At (0,0): H = [[0,-3],[-3,0]], eigenvalues ±3 → Saddle Point
At (1,1): H = [[6,-3],[-3,6]], eigenvalues 3, 9 → Local Minimum
```

### Exercise 3
Show that f(x) = x³ has no local extrema.

**Solution:**
```
f'(x) = 3x² = 0  →  x = 0 (only critical point)

f''(x) = 6x

f''(0) = 0 → Second derivative test is inconclusive.

First derivative test: f'(x) = 3x² ≥ 0 on both sides of x = 0.
f' does not change sign → (0, 0) is an inflection point, not an extremum.
```

### Exercise 4
A rectangular fence is to be built with 100m of fencing along a river (no fence needed on the river side). What dimensions maximize the area?

**Solution:**
```
Let x = width (perpendicular to river), y = length (parallel to river).
Fencing: 2x + y = 100  →  y = 100 - 2x
Area: A = xy = x(100 - 2x) = 100x - 2x²

A'(x) = 100 - 4x = 0  →  x = 25
A''(x) = -4 < 0  →  Maximum ✓

y = 100 - 2(25) = 50

Maximum area: 25 × 50 = 1250 m²
```

---

## Summary

| Concept | Condition | Classification |
|---------|-----------|---------------|
| **Critical Point** | f'(c) = 0 or f'(c) undefined | Candidate for extremum |
| **1st Derivative Test** | f' changes + → − | Local Maximum |
| **1st Derivative Test** | f' changes − → + | Local Minimum |
| **2nd Derivative Test** | f''(c) > 0 | Local Minimum |
| **2nd Derivative Test** | f''(c) < 0 | Local Maximum |
| **Hessian PD** | All eigenvalues > 0 | Local Minimum (multivariate) |
| **Hessian ND** | All eigenvalues < 0 | Local Maximum (multivariate) |
| **Hessian indefinite** | Mixed eigenvalue signs | Saddle Point |
| **Convexity** | Hessian PSD everywhere | All local min = global min |
