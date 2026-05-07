# Limits, Continuity & Differentiability

## Why This Matters in AI/ML

Limits, continuity, and differentiability form the bedrock of every optimization algorithm used in machine learning. When you train a neural network via **gradient descent**, you are implicitly relying on the derivative (the limit definition) to determine the direction of steepest descent. The **loss function** must be continuous and differentiable for gradient-based methods to work. Understanding these concepts is essential for debugging training instability, designing custom loss functions, and implementing advanced optimizers like L-BFGS.

---

## 1. Limits: The Foundation of Calculus

### 1.1 Intuitive Definition

The **limit** of a function f(x) as x approaches a value *a* is the value that f(x) gets arbitrarily close to as x gets arbitrarily close to *a*.

**Formal (epsilon-delta) Definition:**

We write:

```
lim(x -> a) f(x) = L
```

if for every ε > 0, there exists a δ > 0 such that:

```
|f(x) - L| < ε   whenever   0 < |x - a| < δ
```

### 1.2 Key Limit Laws

```
1. lim [f(x) + g(x)]       = lim f(x) + lim g(x)
2. lim [f(x) · g(x)]       = lim f(x) · lim g(x)
3. lim [f(x) / g(x)]       = lim f(x) / lim g(x)    [if lim g(x) ≠ 0]
4. lim [c · f(x)]          = c · lim f(x)
5. lim [f(x)]^n            = [lim f(x)]^n
```

### 1.3 Important Special Limits

```
lim(x -> 0) sin(x)/x   = 1
lim(x -> 0) (e^x - 1)/x = 1
lim(x -> ∞) (1 + 1/x)^x = e
lim(x -> 0) ln(1+x)/x   = 1
```

### 1.4 Limit Evaluation Techniques

**Technique 1: Direct Substitution**

If f(a) is defined, try substituting directly:
```
lim(x -> 3) (x^2 - 9)/(x - 3)
= lim(x -> 3) (x+3)(x-3)/(x-3)
= lim(x -> 3) (x + 3) = 6
```

**Technique 2: L'Hôpital's Rule** (for 0/0 or ∞/∞ forms):
```
If lim f(x)/g(x) = 0/0 or ∞/∞, then:
lim f(x)/g(x) = lim f'(x)/g'(x)
```

**Technique 3: Squeeze Theorem:**

If g(x) ≤ f(x) ≤ h(x) for all x near a, and lim g(x) = lim h(x) = L, then lim f(x) = L.

```
Visual: Squeeze Theorem

  h(x) ___
       /   \
      / ... \        ... f(x) is squeezed
     /  .-.  \       between g(x) and h(x)
    /  ( f )  \
   /___`-'_____\
  g(x)         

  Both g(x) and h(x) converge to L at x = a,
  so f(x) must also converge to L.
```

---

## 2. Continuity

### 2.1 Definition

A function f is **continuous** at x = a if and only if three conditions hold:

```
1. f(a) is defined
2. lim(x -> a) f(x) exists
3. lim(x -> a) f(x) = f(a)
```

### 2.2 Types of Discontinuities

```
  y |
    |      •
    |     / \        Removable Discontinuity (hole):
    |    /   \       f(a) undefined, but limit exists
    |   /     \
    |  •       •     lim(x->a) f(x) = L ≠ f(a)
    |           
----+---•----------- Jump Discontinuity:
    |   |\          Left and right limits differ
    |   | \         lim(x->a⁻) ≠ lim(x->a⁺)
    |   |  \
    |   |   •       
    |
----+---/----------- Infinite Discontinuity:
    |  /            Function approaches ±∞
    | /             lim(x->a) f(x) = ±∞
    |/
    +---------> x
```

### 2.3 Continuity in ML

**Why continuous loss functions matter:**

- A continuous loss function ensures that small changes in model parameters produce small changes in the loss — enabling gradient descent.
- The **cross-entropy loss** is continuous (and differentiable) over the open interval (0,1), making it ideal for classification.
- Discontinuous loss functions (like 0-1 loss) cannot be optimized directly with gradient methods.

```python
# Comparing continuous vs discontinuous loss
import numpy as np

def zero_one_loss(y_true, y_pred):
    """Discontinuous: 0 if correct, 1 if wrong. Not differentiable!"""
    return np.where(y_true == y_pred, 0, 1)

def cross_entropy_loss(y_true, y_pred):
    """Continuous and differentiable: ideal for gradient descent"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))

# Example: true label [1, 0], prediction varies
y_true = np.array([1.0, 0.0])
predictions = [
    np.array([0.99, 0.01]),  # very confident, correct
    np.array([0.70, 0.30]),  # somewhat correct
    np.array([0.51, 0.49]),  # barely correct
    np.array([0.30, 0.70]),  # wrong
]

for p in predictions:
    ce = cross_entropy_loss(y_true, p)
    z1 = zero_one_loss(y_true, np.array([1.0 if pi >= 0.5 else 0.0 for pi in p]))
    print(f"Pred: {p} | Cross-Entropy: {ce:.4f} | 0-1 Loss: {z1}")
# Cross-entropy gives a smooth gradient signal; 0-1 loss is flat!
```

---

## 3. Differentiability

### 3.1 Definition

A function f is **differentiable** at x = a if the following limit exists:

```
f'(a) = lim(h -> 0) [f(a + h) - f(a)] / h
```

This is called the **derivative** — it gives the instantaneous rate of change (slope of the tangent line).

### 3.2 Geometric Interpretation

```
  y |
    |        /
    |       /  tangent at x=a
    |      /   slope = f'(a)
    |     /
    |    •----------- secant line (average rate)
    |   / \  .
    |  /   \    .
    | /     \      .
    |/       \        .
    +---------•-----------> x
              a    a+h

  As h -> 0, the secant becomes the tangent.
```

### 3.3 Differentiability Implies Continuity

If f is differentiable at a, then f is continuous at a. The converse is NOT true:

```
f(x) = |x| is continuous everywhere but NOT differentiable at x = 0

  y |
    |    /
    |   /
    |  /
    | /
----•---------> x
    |\
    | \
    |  \
    |   \

  The "corner" at x=0 has no unique tangent line.
```

### 3.4 Rules of Differentiation

```
Power Rule:     d/dx [x^n]        = n · x^(n-1)
Product Rule:   d/dx [f·g]        = f'·g + f·g'
Quotient Rule:  d/dx [f/g]        = (f'·g - f·g') / g²
Chain Rule:     d/dx [f(g(x))]    = f'(g(x)) · g'(x)

Derivatives of Common Functions:
d/dx [e^x]    = e^x
d/dx [ln(x)]  = 1/x
d/dx [sin(x)] = cos(x)
d/dx [cos(x)] = -sin(x)
```

### 3.5 Partial Derivatives (Preview for Multivariable Calculus)

For a function of multiple variables f(x₁, x₂, ..., xₙ), the **partial derivative** with respect to xᵢ:

```
∂f/∂xᵢ = lim(h -> 0) [f(x₁,...,xᵢ+h,...,xₙ) - f(x₁,...,xᵢ,...,xₙ)] / h
```

The collection of all partial derivatives forms the **gradient vector**:

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**This is the single most important concept for neural network training.**

```python
import numpy as np

def f(x):
    """Example function: f(x) = x^3 - 2x^2 + x"""
    return x**3 - 2*x**2 + x

def derivative(f, x, h=1e-7):
    """Numerical derivative using the limit definition"""
    return (f(x + h) - f(x)) / h

def analytical_derivative(x):
    """f'(x) = 3x^2 - 4x + 1"""
    return 3*x**2 - 4*x + 1

# Compare numerical vs analytical
x_vals = np.linspace(-1, 3, 9)
print("x     | Numerical  | Analytical | Difference")
print("-" * 50)
for x in x_vals:
    num = derivative(f, x)
    ana = analytical_derivative(x)
    print(f"{x:5.2f} | {num:10.6f} | {ana:10.6f} | {abs(num-ana):.2e}")
```

### 3.6 The Chain Rule in Neural Networks

The chain rule is the mechanism behind **backpropagation**. Consider a simple two-layer network:

```
Input x -> Weight w₁ -> Hidden h = σ(w₁x + b₁) -> Weight w₂ -> Output ŷ = w₂h + b₂
```

To compute ∂Loss/∂w₁ (how the loss changes with w₁):

```
∂Loss/∂w₁ = ∂Loss/∂ŷ · ∂ŷ/∂h · ∂h/∂z · ∂z/∂w₁
```

where z = w₁x + b₁. This is the chain rule applied repeatedly!

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """d/dx σ(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

# Chain rule in action: simple network
x = 2.0   # input
w1 = 0.5  # weight 1
b1 = 0.0  # bias 1
w2 = 0.3  # weight 2
b2 = 0.0  # bias 2
y_true = 1.0  # target

# Forward pass
z = w1 * x + b1
h = sigmoid(z)
y_pred = w2 * h + b2

# Loss (MSE)
loss = 0.5 * (y_pred - y_true)**2

# Backward pass (chain rule)
dL_dy = y_pred - y_true          # ∂Loss/∂ŷ
dy_dw2 = h                       # ∂ŷ/∂w₂
dy_dh = w2                       # ∂ŷ/∂h
dh_dz = sigmoid_derivative(z)    # ∂h/∂z
dz_dw1 = x                       # ∂z/∂w₁

# Chain rule compositions
dL_dw2 = dL_dy * dy_dw2
dL_dw1 = dL_dy * dy_dh * dh_dz * dz_dw1

print(f"Forward pass: z={z:.4f}, h={h:.4f}, y_pred={y_pred:.4f}")
print(f"Loss: {loss:.4f}")
print(f"Gradient w.r.t. w2: {dL_dw2:.6f}")
print(f"Gradient w.r.t. w1: {dL_dw1:.6f}")
```

---

## 4. Key Theorems

### 4.1 Intermediate Value Theorem (IVT)

If f is continuous on [a, b] and k is any number between f(a) and f(b), then there exists c ∈ (a, b) such that f(c) = k.

**ML Application:** The bisection method for root-finding (used in some optimization algorithms) relies on IVT.

### 4.2 Mean Value Theorem (MVT)

If f is continuous on [a, b] and differentiable on (a, b), then there exists c ∈ (a, b) such that:

```
f'(c) = [f(b) - f(a)] / (b - a)
```

```
  y |
    |      B
    |     /|\
    |    / | \    The slope of secant AB equals
    |   /  |  \   the slope of tangent at some
    |  /   |   \  point c between A and B
    | /  • c    \
    |/            \
    A              |
    +--------------+-------> x
    a              b
```

---

## 5. Exercises with Solutions

### Exercise 1
Evaluate `lim(x -> 2) (x³ - 8)/(x - 2)`.

**Solution:** Factor numerator: x³ - 8 = (x-2)(x² + 2x + 4).

```
lim(x -> 2) (x-2)(x² + 2x + 4) / (x - 2)
= lim(x -> 2) (x² + 2x + 4) = 4 + 4 + 4 = 12
```

### Exercise 2
Use L'Hôpital's rule: `lim(x -> 0) (1 - cos x) / x²`.

**Solution:** Both numerator and denominator approach 0.

```
= lim(x -> 0) sin(x) / (2x)    [still 0/0]
= lim(x -> 0) cos(x) / 2       [L'Hôpital again]
= 1/2
```

### Exercise 3
Prove that f(x) = x² is continuous at x = 3 using the ε-δ definition.

**Solution:** We need: |x² - 9| < ε whenever |x - 3| < δ.

Note: |x² - 9| = |x + 3||x - 3|. When |x - 3| < 1, we have 2 < x < 4, so |x + 3| < 7.

Choose δ = min(1, ε/7). Then:

```
|x² - 9| = |x + 3||x - 3| < 7 · (ε/7) = ε  ✓
```

### Exercise 4
Where is f(x) = |x - 1| + |x + 2| not differentiable?

**Solution:** The absolute value function creates "corners" where its argument is zero:
- |x - 1| has a corner at x = 1
- |x + 2| has a corner at x = -2

Therefore, f(x) is not differentiable at **x = 1** and **x = -2**.

---

## Summary Table

| Concept | Definition | Key Property | ML Relevance |
|---------|-----------|--------------|-------------|
| **Limit** | Value f(x) approaches as x → a | Epsilon-delta definition | Derivative definition |
| **Continuity** | lim f(x) = f(a) at every point | No jumps, holes, or asymptotes | Smooth loss functions |
| **Differentiability** | Derivative exists (limit of secant slopes) | Implies continuity | Gradient descent, backpropagation |
| **Partial Derivative** | Derivative w.r.t. one variable, others fixed | Forms the gradient vector | Multi-parameter optimization |
| **Chain Rule** | d/dx f(g(x)) = f'(g(x))·g'(x) | Composition of derivatives | Backpropagation algorithm |
