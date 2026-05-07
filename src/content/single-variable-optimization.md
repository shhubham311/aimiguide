# Optimization Involving a Single Variable

## Why This Matters in AI/ML

Single-variable optimization is the simplest form of the optimization problems that drive all of machine learning. Every ML training process — from simple linear regression to billion-parameter language models — is fundamentally about finding parameters that minimize a loss function. While real-world ML involves millions of variables, the intuition and algorithms for single-variable optimization extend directly. Understanding the theory of convergence, step sizes, and trade-offs in 1D prepares you for the multi-dimensional case.

---

## 1. Problem Formulation

### 1.1 The Standard Form

Find x* such that:

```
Minimize:  f(x)
Subject to: x ∈ [a, b]    (optional: bounded domain)
```

Or equivalently, maximize -f(x).

### 1.2 Necessary and Sufficient Conditions (Unconstrained)

```
Necessary:  f'(x*) = 0
Sufficient: f'(x*) = 0 AND f''(x*) > 0
```

For a maximization problem, replace f''(x*) > 0 with f''(x*) < 0.

---

## 2. Analytical Methods

### 2.1 Direct Solution (when possible)

For simple functions, find f'(x) = 0 and solve algebraically.

**Example:** Minimize f(x) = x² - 4x + 7

```
f'(x) = 2x - 4 = 0  →  x* = 2
f''(x) = 2 > 0      →  Minimum confirmed
f(2) = 4 - 8 + 7 = 3
```

```
  f(x)
   ^
  7 |       • (vertex at x=2, f=3)
   |      / \
  6 |     /   \
   |    /     \
  5 |   /       \
   |  /         \
  4 | /           \
  3 |/             \
   +-----•---------•-----> x
   0     2         4
```

### 2.2 Completing the Square

For quadratic functions f(x) = ax² + bx + c:

```
f(x) = a[x² + (b/a)x] + c
     = a[(x + b/2a)² - (b/2a)²] + c
     = a(x + b/2a)² + (c - b²/4a)

Minimum at: x* = -b/(2a)
Value:     f(x*) = c - b²/(4a)
```

This is useful for understanding the **ridge regression** closed-form solution.

---

## 3. Numerical Methods: When Analytical Solutions Don't Exist

### 3.1 Bisection Method

Uses the first derivative: find x* where f'(x) = 0 by bracketing.

```python
import numpy as np

def bisection_min(f_prime, a, b, tol=1e-10, max_iter=100):
    """
    Find minimum of f using bisection on f'.
    Requires: f'(a) < 0 and f'(b) > 0 (minimum bracketed).
    """
    if f_prime(a) * f_prime(b) > 0:
        raise ValueError("f' must change sign in [a, b]")
    
    for i in range(max_iter):
        mid = (a + b) / 2
        if b - a < tol:
            return mid, i + 1
        if f_prime(mid) < 0:
            a = mid  # minimum is to the right
        else:
            b = mid  # minimum is to the left
    return (a + b) / 2, max_iter

# Minimize f(x) = x^3 - 6x^2 + 9x + 1
f_prime = lambda x: 3*x**2 - 12*x + 9
x_min, iters = bisection_min(f_prime, 0.5, 2.5)
print(f"Bisection: minimum at x = {x_min:.8f} (iterations: {iters})")
print(f"f(x_min) = {x_min**3 - 6*x_min**2 + 9*x_min + 1:.8f}")
```

```
  Bisection Method Visualization:

  f'(x) = 3x² - 12x + 9

  f'(x)
    ^
    |  •(0.5)     •(2.5)
    |   \       /
    |    \     /
    |     \   /
    |      \ /
    |       •(1)      ← f'(1) = 0, minimum of f
    |      / \
    |     /   \
    |    /     \
    +--•-------•-------> x
    0         3

  Bisection halves the interval each step: [a,b] → [a, mid] or [mid, b]
```

### 3.2 Newton's Method for Optimization

Uses f'(x) and f''(x):

```
x_{n+1} = x_n - f'(x_n) / f''(x_n)
```

This is equivalent to finding the root of f' using Newton's method.

```python
def newton_optimize(f_prime, f_double_prime, x0, tol=1e-10, max_iter=100):
    """Newton's method for finding a local minimum."""
    x = x0
    for i in range(max_iter):
        fp = f_prime(x)
        fpp = f_double_prime(x)
        if abs(fp) < tol:
            return x, i + 1
        x = x - fp / fpp
    return x, max_iter

# Same function
f_prime = lambda x: 3*x**2 - 12*x + 9
f_double_prime = lambda x: 6*x - 12

x_min, iters = newton_optimize(f_prime, f_double_prime, 0.5)
print(f"Newton: minimum at x = {x_min:.10f} (iterations: {iters})")
```

### 3.3 Gradient Descent (1D)

The workhorse of modern ML:

```
x_{n+1} = x_n - α · f'(x_n)
```

where α is the **learning rate**.

```python
def gradient_descent_1d(f_prime, x0, lr=0.01, tol=1e-10, max_iter=10000):
    """1D gradient descent."""
    x = x0
    path = [x]
    for i in range(max_iter):
        fp = f_prime(x)
        if abs(fp) < tol:
            return x, i + 1, np.array(path)
        x = x - lr * fp
        path.append(x)
    return x, max_iter, np.array(path)

# Minimize f(x) = (x-3)² + 2
f_prime = lambda x: 2*(x - 3)
f = lambda x: (x - 3)**2 + 2

# Try different learning rates
for lr in [0.01, 0.1, 0.5, 1.0, 1.5]:
    try:
        x_min, iters, _ = gradient_descent_1d(f_prime, 0.0, lr=lr, max_iter=100)
        print(f"lr={lr:.2f}: x={x_min:.6f}, f={f(x_min):.6f}, iters={iters}")
    except Exception as e:
        print(f"lr={lr:.2f}: Diverged! ({e})")
```

```
  Effect of Learning Rate on Convergence:

  lr=0.1 (good):              lr=1.5 (too large - oscillation/diverges):
  
  f(x)  |                    f(x)  |
   ^    |                      ^    |  /\/\/\
   |    |\                     |    | /
   |    | \  / \  / \         |    |/      \
   |    |  \/   \/   \___     |    /        \
   |    |              •      |   /          \
   +----+-------> x           +----+-------> x
        x*                            x*

  Rule of thumb: lr < 2/L where L is the Lipschitz constant of f'.
```

### 3.4 Golden Section Search (Derivative-Free)

When derivatives are unavailable:

```python
def golden_section_search(f, a, b, tol=1e-8):
    """Find minimum of unimodal f on [a, b] without derivatives."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio ≈ 1.618
    resphi = 2 - phi             # 1/phi ≈ 0.618
    
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1, f2 = f(x1), f(x2)
    
    iters = 0
    while abs(b - a) > tol:
        iters += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)
    
    return (a + b) / 2, iters

f = lambda x: (x - 3)**2 + 2
x_min, iters = golden_section_search(f, 0, 10)
print(f"Golden Section: minimum at x = {x_min:.8f} (iterations: {iters})")
```

---

## 4. Convergence Analysis

### 4.1 Convergence Rates

```
Method          | Convergence Rate | Information Needed
--------------- | ---------------- | ------------------
Bisection       | Linear (½^n)     | f' sign only
Gradient Descent| Linear           | f' (first derivative)
Newton's Method | Quadratic        | f', f'' (first & second)
Golden Section  | Linear (0.618^n) | f values only
```

**Quadratic convergence** (Newton's) means the number of correct digits roughly doubles each step. **Linear convergence** (GD) means the error decreases by a constant factor each step.

### 4.2 Conditions for Convergence

**Gradient Descent converges** when the learning rate satisfies:

```
0 < α < 2/L
```

where L is the **Lipschitz constant** of f' (i.e., |f''(x)| ≤ L for all x).

For f(x) = (x-3)²: f''(x) = 2, so L = 2, and we need α < 1.

**Newton's Method converges quadratically** when:
- f''(x*) ≠ 0
- Initial guess is "close enough" to x*

---

## 5. Line Search: Connecting 1D to Multi-Dimensional Optimization

### 5.1 What is Line Search?

In multi-dimensional optimization, each gradient descent step requires choosing a step size. **Line search** reduces this to a 1D optimization problem:

```
Given direction d, find α* that minimizes:
  φ(α) = f(x + αd)
```

This is a single-variable problem!

### 5.2 Types of Line Search

1. **Exact Line Search:** Minimize φ(α) precisely (expensive)
2. **Backtracking Line Search (Armijo):** Find α satisfying sufficient decrease condition:

```
f(x + αd) ≤ f(x) + c₁ · α · ∇f(x)ᵀd
```

where c₁ is typically 10⁻⁴.

```python
def backtracking_line_search(f, grad, x, d, alpha_init=1.0, c1=1e-4, rho=0.5):
    """
    Backtracking line search (Armijo condition).
    Find step size alpha such that:
      f(x + alpha*d) <= f(x) + c1*alpha*grad(x).d
    """
    alpha = alpha_init
    fx = f(x)
    directional_deriv = grad(x) @ d  # gradient dot direction
    
    # Direction must be a descent direction
    if directional_deriv >= 0:
        raise ValueError("d is not a descent direction")
    
    iters = 0
    while f(x + alpha * d) > fx + c1 * alpha * directional_deriv:
        alpha *= rho  # reduce step size
        iters += 1
        if alpha < 1e-15:
            break
    return alpha, iters

# Example: minimize f(x) = x₁² + 5x₂² from x = [5, 5]
f = lambda x: x[0]**2 + 5*x[1]**2
grad = lambda x: np.array([2*x[0], 10*x[1]])

x = np.array([5.0, 5.0])
d = -grad(x)  # steepest descent direction

alpha, iters = backtracking_line_search(f, grad, x, d)
print(f"Step size: {alpha:.6f} (found in {iters} backtracking steps)")
x_new = x + alpha * d
print(f"x: {x} -> {x_new}")
print(f"f(x): {f(x):.2f} -> {f(x_new):.6f}")
```

---

## 6. Convexity in Single-Variable Optimization

### 6.1 Why Convexity Matters

For **convex** functions, every local minimum is global, and gradient descent with appropriate step size always converges.

A single-variable function is convex if:

```
f''(x) ≥ 0  for all x
```

Or equivalently: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for all x, y and λ ∈ [0, 1].

### 6.2 Strong Convexity and Convergence Speed

A function is **μ-strongly convex** if:

```
f''(x) ≥ μ > 0  for all x
```

Strong convexity guarantees a **unique global minimum** and linear convergence of gradient descent with rate:

```
f(x_n) - f(x*) ≤ (1 - μ/L)^n · (f(x₀) - f(x*))
```

### 6.3 Non-Convex Challenges

For non-convex functions, gradient descent may converge to:
- Local minima (suboptimal)
- Saddle points (especially in high dimensions)
- May oscillate or diverge

```
  Convex Loss (Linear Regression):     Non-Convex Loss (Neural Net):
  
  Loss   |                             Loss   |
     ^   |                                ^   |
     |  /|                                | /\|  /\
     | / |                               |/  |\/  \
     |/  |  • Global min                 |   •    •Local minima
     +---+---> w                         +---+---> weights

  One global minimum                    Many local minima
  GD always converges to it             GD may get stuck
```

---

## 7. Practical Tips for ML Optimization

### 7.1 Learning Rate Scheduling

```python
# Common learning rate schedules
def constant_lr(epoch, lr=0.01):
    return lr

def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** (epoch // epochs_drop))

def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)

def cosine_decay(epoch, initial_lr=0.1, total_epochs=100):
    import math
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

# Visualize
for epoch in [0, 10, 20, 50, 80, 100]:
    print(f"Epoch {epoch:3d}: Step={step_decay(epoch):.6f}, "
          f"Exp={exponential_decay(epoch):.6f}, "
          f"Cos={cosine_decay(epoch):.6f}")
```

### 7.2 Gradient Clipping

For functions with very large gradients:

```python
def clip_gradient(grad, max_norm=1.0):
    """Clip gradient to prevent explosion."""
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad

# Example: very steep region
grad = np.array([100.0, -200.0])
print(f"Original norm: {np.linalg.norm(grad):.2f}")
print(f"Clipped: {clip_gradient(grad, max_norm=1.0)}")
```

---

## 8. Exercises with Solutions

### Exercise 1
Minimize f(x) = x⁴ - 4x³ + 6x² - 4x + 1 using the analytical method.

**Solution:**
```
f(x) = (x-1)⁴

f'(x) = 4(x-1)³ = 0  →  x* = 1
f''(x) = 12(x-1)²

f''(1) = 0 → Second derivative test inconclusive.
But f'(x) changes from negative to positive at x = 1 (use first derivative test).
→ Global Minimum at x = 1, f(1) = 0
```

### Exercise 2
Implement gradient descent to minimize f(x) = sin(x) + x²/10. Start at x₀ = 3.

```python
import numpy as np

f = lambda x: np.sin(x) + x**2/10
f_prime = lambda x: np.cos(x) + x/5

x = 3.0
lr = 0.1
for i in range(100):
    grad = f_prime(x)
    x = x - lr * grad
    if abs(grad) < 1e-10:
        break

print(f"Minimum at x = {x:.8f}, f(x) = {f(x):.8f} (iterations: {i+1})")
# Expected: x ≈ -1.3767 (the global minimum)
```

### Exercise 3
Show that gradient descent on f(x) = ax² converges only if α < 1/a.

**Solution:**
```
x_{n+1} = x_n - 2aα·x_n = x_n(1 - 2aα)

This converges iff |1 - 2aα| < 1
→ -1 < 1 - 2aα < 1
→ 0 < 2aα < 2
→ 0 < α < 1/a
```

### Exercise 4
Use Newton's method to find √5 (minimize f(x) = x² - 5).

**Solution:**
```
f(x) = x² - 5, f'(x) = 2x, f''(x) = 2

x_{n+1} = x_n - 2x_n/2 = x_n - x_n/2 = x_n/2  ... 

Wait: x_{n+1} = x_n - f'(x_n)/f''(x_n) = x_n - 2x_n/2 = 0. 
That's wrong! Newton for minimization finds f'=0.

f'(x) = 2x = 0 → x = 0. But f''(0) = 2 > 0 so min at x=0.

Actually, to find √5 we should solve f(x) = x² - 5 = 0 using Newton's ROOT finding:
x_{n+1} = x_n - f(x_n)/f'(x_n) = x_n - (x_n² - 5)/(2x_n) = (x_n + 5/x_n)/2

Starting at x₀ = 2:
x₁ = (2 + 5/2)/2 = 2.25
x₂ = (2.25 + 5/2.25)/2 = 2.2361...
√5 = 2.2360679... ✓
```

---

## Summary

| Method | Rate | Derivatives | Pros | Cons |
|--------|------|-------------|------|------|
| **Analytical** | Exact | Need closed-form | Perfect answer | Not always possible |
| **Bisection** | Linear (½) | f' sign only | Always converges | Slow |
| **Gradient Descent** | Linear | f' | Simple, scalable | Sensitive to lr |
| **Newton's Method** | Quadratic | f', f'' | Very fast | Needs good init |
| **Golden Section** | Linear | f only | No derivatives | Slow, unimodal only |
