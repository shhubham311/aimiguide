# Taylor Series

## Why This Matters in AI/ML

Taylor series are everywhere in machine learning. They provide the mathematical justification for **gradient descent** — the first-order Taylor approximation is exactly what GD uses. **Newton's method** relies on the second-order approximation. Taylor expansions underpin **Gaussian processes**, **Laplace approximation** for Bayesian inference, **optimization theory** proofs, and the derivation of the **Fisher information matrix**. Understanding Taylor series gives you insight into why gradient methods work and when they fail.

---

## 1. The Core Idea: Approximating Functions with Polynomials

### 1.1 Motivation

Many functions (e^x, sin(x), ln(x)) are expensive or impossible to compute directly. A **polynomial** — made of simple additions and multiplications — can approximate any smooth function to arbitrary precision.

### 1.2 Taylor's Theorem

If f is infinitely differentiable at x = a, then:

```
f(x) = f(a) + f'(a)(x-a)/1! + f''(a)(x-a)²/2! + f'''(a)(x-a)³/3! + ...
```

In summation notation:

```
f(x) = Σ(n=0 to ∞) [f⁽ⁿ⁾(a) / n!] · (x - a)ⁿ
```

**Variable definitions:**

| Symbol | Meaning |
|--------|---------|
| f(x) | The original function |
| a | The center (expansion point) |
| f⁽ⁿ⁾(a) | The n-th derivative of f evaluated at a |
| n! | n factorial = n × (n-1) × ... × 1 |
| (x - a)ⁿ | The n-th power of the distance from center |

### 1.3 Maclaurin Series (Taylor Series Centered at a = 0)

When a = 0, we get the simpler **Maclaurin series**:

```
f(x) = f(0) + f'(0)x + f''(0)x²/2! + f'''(0)x³/3! + ...
```

### 1.4 Common Maclaurin Series (Must Memorize)

```
1. e^x         = 1 + x + x²/2! + x³/3! + x⁴/4! + ...           (all x)
2. sin(x)      = x - x³/3! + x⁵/5! - x⁷/7! + ...               (all x)
3. cos(x)      = 1 - x²/2! + x⁴/4! - x⁶/6! + ...               (all x)
4. ln(1+x)     = x - x²/2 + x³/3 - x⁴/4 + ...                   (-1 < x ≤ 1)
5. 1/(1-x)     = 1 + x + x² + x³ + x⁴ + ...                     (|x| < 1)
6. (1+x)^k     = 1 + kx + k(k-1)x²/2! + ...   (binomial series) (|x| < 1)
7. arctan(x)   = x - x³/3 + x⁵/5 - x⁷/7 + ...                   (|x| ≤ 1)
```

---

## 2. Intuition: Building the Approximation Step by Step

```
  Approximating f(x) = e^x with Taylor polynomials at a = 0:

  y=e^x
  4 |                    *       e^x (the true function)
    |                  *   *
  3 |    T₃ ------*       *
    |   /       *   *
  2 |  T₂ --*         *       T₃(x) = 1 + x + x²/2 + x³/6
    | /   *                 T₂(x) = 1 + x + x²/2
  1 |T₁ *                   T₁(x) = 1 + x
    |*                          T₀(x) = 1
  0 +----+----+----+----+----> x
   -1    0    1    2    3

  Each additional term captures more of the curve's shape.
```

```python
import numpy as np
import matplotlib.pyplot as plt

def taylor_exp(x, n_terms):
    """Compute the n-term Taylor approximation of e^x at a=0."""
    result = np.zeros_like(x, dtype=float)
    for n in range(n_terms):
        result += x**n / np.math.factorial(n)
    return result

x = np.linspace(-2, 3, 200)

for n in [1, 2, 3, 5, 10]:
    approx = taylor_exp(x, n)
    plt.plot(x, approx, label=f'T{n}')

plt.plot(x, np.exp(x), 'k--', linewidth=2, label='e^x')
plt.ylim(-2, 12)
plt.legend()
plt.title("Taylor Approximations of e^x")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. Remainder and Convergence

### 3.1 Taylor's Remainder Theorem

The error from truncating at n terms is:

```
R_n(x) = f⁽ⁿ⁺¹⁾(c) / (n+1)! · (x - a)^(n+1)
```

for some c between a and x (Lagrange form of the remainder).

### 3.2 Radius of Convergence

A Taylor series converges to f(x) only within a certain radius R:

- For e^x, sin(x), cos(x): R = ∞ (converges everywhere)
- For ln(1+x): R = 1 (converges for |x| < 1, and at x = 1)
- For 1/(1-x): R = 1

### 3.3 Convergence Rate

```
  e^x expansion error at x = 2:
  
  Terms | Approximation | Error       | Error Ratio
  ----- | ------------- | ----------- | -----------
    1   | 1.000         | 6.389       | —
    2   | 3.000         | 4.389       | 0.69
    3   | 5.000         | 2.389       | 0.54
    5   | 7.000         | 0.389       | 0.09
    7   | 7.356         | 0.034       | 0.007
   10   | 7.3886        | 0.0007      | 0.0001
  
  e^2 = 7.38906... — convergence is fast for moderate x.
```

---

## 4. Taylor Series in Machine Learning

### 4.1 First-Order Approximation → Gradient Descent

The gradient descent update rule is derived from the first-order Taylor expansion:

```
f(w + Δw) ≈ f(w) + ∇f(w)ᵀΔw

To minimize, choose Δw = -α∇f(w), giving:
f(w - α∇f(w)) ≈ f(w) - α‖∇f(w)‖²
```

This shows: **moving opposite to the gradient decreases f** (as long as α is small enough).

```
  f(w)
   ^
   |   • f(w - α∇f)  ← lower than f(w) ✓
   |    \
   |  •   \  f(w)    ← current point
   |   \    \
   |    \    \
   |     •    \  f(w + α∇f) ← higher (wrong direction)
   |
   +---------> w

  The first-order Taylor expansion tells us which direction is "downhill."
```

### 4.2 Second-Order Approximation → Newton's Method

Adding the Hessian matrix:

```
f(w + Δw) ≈ f(w) + ∇f(w)ᵀΔw + (1/2)ΔwᵀHΔw

Setting derivative to zero:
∇f(w) + HΔw = 0
=> Δw = -H⁻¹∇f(w)
```

This is the **Newton's method** update — much faster convergence than gradient descent, but requires computing and inverting the Hessian.

```python
import numpy as np

def gradient_descent(grad_f, w, lr=0.1, steps=50):
    """First-order method: uses only the gradient"""
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad_f(w)
        path.append(w.copy())
    return np.array(path)

def newtons_method(grad_f, hess_f, w, steps=10):
    """Second-order method: uses gradient + Hessian"""
    path = [w.copy()]
    for _ in range(steps):
        H = hess_f(w)
        g = grad_f(w)
        w = w - np.linalg.solve(H, g)  # H⁻¹g
        path.append(w.copy())
    return np.array(path)

# Minimize f(w) = w₁² + 5w₂² (elliptical bowl)
def f(w):
    return w[0]**2 + 5*w[1]**2

def grad_f(w):
    return np.array([2*w[0], 10*w[1]])

def hess_f(w):
    return np.array([[2.0, 0.0], [0.0, 10.0]])

w0 = np.array([5.0, 5.0])
gd_path = gradient_descent(grad_f, w0.copy(), lr=0.05, steps=30)
nm_path = newtons_method(grad_f, hess_f, w0.copy(), steps=5)

print("Gradient Descent (30 steps):")
for i, w in enumerate(gd_path[::5]):
    print(f"  Step {i*5:3d}: w = {w}, f(w) = {f(w):.6f}")

print("\nNewton's Method (5 steps):")
for i, w in enumerate(nm_path):
    print(f"  Step {i:3d}: w = {w}, f(w) = {f(w):.6f}")
# Newton converges in 1 step for quadratic functions!
```

### 4.3 Taylor Expansion of the Logistic Function

The sigmoid function σ(z) = 1/(1+e^(-z)) has the Taylor expansion at z = 0:

```
σ(z) = 1/2 + z/4 - z³/48 + z⁵/480 - ...
```

This tells us:
- σ(z) ≈ 1/2 + z/4 for small z (nearly linear near origin)
- The function is approximately linear only in a narrow range around z = 0
- This is why **vanishing gradients** occur for large |z|: higher-order terms vanish

### 4.4 Laplace Approximation (Bayesian ML)

The Laplace approximation uses a second-order Taylor expansion to approximate a posterior distribution:

```
ln p(θ|D) ≈ ln p(θ₀|D) + (1/2)(θ - θ₀)ᵀ H (θ - θ₀)
```

where θ₀ is the MAP estimate and H is the Hessian of the log-posterior. This gives a Gaussian approximation to the posterior — the basis of many Bayesian neural network methods.

---

## 5. Multivariate Taylor Series

For f(x₁, x₂, ..., xₙ), the Taylor expansion around point **a**:

```
First Order:  f(a) + ∇f(a)ᵀ(x - a)

Second Order: f(a) + ∇f(a)ᵀ(x - a) + (1/2)(x-a)ᵀH(a)(x-a)

General:
f(x) = Σ(|α|=0 to ∞) [D^α f(a) / α!] (x-a)^α
```

where α = (α₁, ..., αₙ) is a multi-index.

---

## 6. Practical Considerations

### 6.1 When Taylor Series Fail

- **Non-analytic functions:** Functions like |x|, √(x²) are not infinitely differentiable
- **Large deviations:** The approximation is only good *near* the expansion point
- **Stiff functions:** Functions with very high curvature may need many terms

### 6.2 Choosing the Center

```
  f(x) = 1/x approximated at different centers:

  a=0.5:  f(x) ≈ 2 - 4(x-0.5) + 8(x-0.5)² - ...   (good near x=0.5)
  a=1:    f(x) ≈ 1 - (x-1) + (x-1)² - ...           (good near x=1)
  a=2:    f(x) ≈ 0.5 - 0.25(x-2) + 0.125(x-2)² - ...(good near x=2)

  No single Taylor series works well globally!
```

---

## 7. Exercises with Solutions

### Exercise 1
Find the Maclaurin series for f(x) = e^(2x).

**Solution:** Substitute 2x into the series for e^x:

```
e^(2x) = 1 + 2x + (2x)²/2! + (2x)³/3! + ...
       = 1 + 2x + 4x²/2 + 8x³/6 + ...
       = 1 + 2x + 2x² + 4x³/3 + ...
       = Σ (2x)^n / n!
```

### Exercise 2
Compute the third-order Taylor polynomial of f(x) = sin(x) centered at a = π/2.

**Solution:** Compute derivatives at a = π/2:

```
f(π/2)     = sin(π/2)   = 1
f'(π/2)    = cos(π/2)   = 0
f''(π/2)   = -sin(π/2)  = -1
f'''(π/2)  = -cos(π/2)  = 0

T₃(x) = 1 + 0·(x-π/2) + (-1)/2!·(x-π/2)² + 0·(x-π/2)³/3!
      = 1 - (x - π/2)²/2
```

### Exercise 3
Use the Taylor series for e^x to estimate e^0.5 with an error less than 0.001.

**Solution:** Since e^x = Σ x^n/n!, the remainder after n terms:

```
|R_n(0.5)| = e^c · 0.5^(n+1) / (n+1)!   for some c ∈ (0, 0.5)
           ≤ e^0.5 · 0.5^(n+1) / (n+1)!
           ≤ 2 · 0.5^(n+1) / (n+1)!
```

Try n = 4: 2 · 0.5⁵ / 120 = 2/3840 ≈ 0.00052 < 0.001 ✓

```
e^0.5 ≈ 1 + 0.5 + 0.125 + 0.02083 + 0.002604 = 1.6484
True: e^0.5 ≈ 1.64872 (error ≈ 0.0003)
```

### Exercise 4
Show that the first-order Taylor expansion justifies gradient descent for f(w) = w².

**Solution:** f(w + Δw) ≈ f(w) + f'(w)Δw = w² + 2w·Δw

Setting Δw = -α·2w: f(w - 2αw) ≈ w² - 4αw²

For α > 0, this is always less than w² when w ≠ 0, confirming the update reduces f.

---

## Summary

| Concept | Formula | ML Application |
|---------|---------|----------------|
| **Taylor Series** | f(x) = Σ f⁽ⁿ⁾(a)(x-a)ⁿ/n! | Universal approximation |
| **1st Order** | f(x) ≈ f(a) + f'(a)(x-a) | Gradient descent |
| **2nd Order** | f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2 | Newton's method |
| **Remainder** | Rₙ = f⁽ⁿ⁺¹⁾(c)(x-a)^(n+1)/(n+1)! | Convergence guarantees |
| **Multivariate** | f(x) ≈ f(a) + ∇f(a)ᵀ(x-a) + (x-a)ᵀH(x-a)/2 | Multi-parameter optimization |
