# Activation Functions: Sigmoid, Tanh, Softmax, ReLU, Leaky ReLU

## Why This Topic Matters

Activation functions are the **non-linear heartbeat of neural networks**. Without them, a deep neural network — no matter how many layers — would collapse into a single linear transformation. They introduce the non-linearity that allows networks to learn complex, real-world patterns like curves, decision boundaries, and hierarchical features.

Choosing the right activation function can mean the difference between a network that converges in 50 epochs and one that never learns at all. The **vanishing gradient problem** with Sigmoid/Tanh led to the ReLU revolution, which in turn motivated Leaky ReLU, ELU, GELU, and Swish. Understanding *why* each function exists and *when* to use it is essential for every deep learning practitioner.

---

## 1. Why Do We Need Activation Functions?

### The Linear Collapse Problem

Consider a neural network without any activation functions:

$$\hat{y} = \mathbf{W}^{(3)}(\mathbf{W}^{(2)}(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)}) + \mathbf{b}^{(3)}$$

Expanding:
$$\hat{y} = \mathbf{W}^{(3)}\mathbf{W}^{(2)}\mathbf{W}^{(1)}\mathbf{x} + \text{combined biases}$$

$$\hat{y} = \mathbf{W}'\mathbf{x} + \mathbf{b}'$$

**No matter how many layers we stack, the result is still a linear function!** We could replace the entire network with a single linear layer.

```
WITHOUT Activation Functions:          WITH Activation Functions:

  Input ──→ W1 ──→ W2 ──→ W3 ──→ ŷ   Input ──→ W1 ──→ ReLU ──→ W2 ──→ ReLU ──→ W3 ──→ ŷ
                                              │                 │
  Result: ŷ = W'x + b' (LINEAR)              │                 │
  ≡ Single linear layer              Result: ŷ = f3(W2·f2(W1·x + b1) + b2) + b3
  ≡ Cannot learn non-linear patterns              (NON-LINEAR — learns curves!)
```

**Activation functions break the linearity**, enabling the network to approximate any function.

---

## 2. Sigmoid (Logistic Sigmoid)

### Formula

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Properties

| Property | Value |
|----------|-------|
| Output range | $(0, 1)$ |
| Derivative | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ |
| Center of output | $0.5$ (not zero-centered) |
| Monotonic? | Yes |

### Shape and Behavior

```
Output σ(x)
  1.0 ┤            ┌──────────────────
      │           /
  0.5 ┤........./..........................
      │        /
  0.0 ┤───────┘
      └──────┼──────┼──────┼──────→ x
           -4      0      4

  For large positive x: σ(x) → 1
  For large negative x: σ(x) → 0
  At x = 0: σ(0) = 0.5
```

### Gradient Behavior

```
Gradient σ'(x)
  0.25 ┤           ╱╲
       │          ╱  ╲
       │         ╱    ╲
  0.0  ┤────────╱      ╲─────────→ x
       └───────┼──────┼──────
              -4      4

  Maximum gradient = 0.25 (at x=0)
  For |x| > 5: gradient ≈ 0 (saturated!)
```

### Vanishing Gradient Problem

In deep networks, the sigmoid gradient at each layer is at most 0.25. With many layers:

$$\text{Gradient at Layer 1} \leq (0.25)^L \rightarrow 0 \text{ as } L \text{ grows}$$

```
Layer 10: gradient ≤ (0.25)^10 = 0.000000954
Layer 20: gradient ≤ (0.25)^20 ≈ 0

→ Weights in early layers barely update!
→ Network learns very slowly or not at all.
```

### Additional Problems

1. **Not zero-centered**: Gradients always have the same sign, causing zigzag updates
2. **Computationally expensive**: Involves exponentials
3. **Output saturation**: Neurons "die" with constant output near 0 or 1

### When to Use Sigmoid

- **Binary classification output layer** (output interpreted as probability)
- **Gating mechanisms** in LSTM/GRU cells
- **Multi-label classification** (each output independently between 0 and 1)

### PyTorch Code

```python
import torch
import torch.nn as nn

# Built-in
sigmoid = nn.Sigmoid()
x = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
print(f"Sigmoid: {sigmoid(x)}")
# Output: tensor([0.0180, 0.1192, 0.5000, 0.8808, 0.9820])

# Manual implementation
def sigmoid_manual(x):
    return 1.0 / (1.0 + torch.exp(-x))

# Derivative
def sigmoid_derivative(x):
    s = sigmoid_manual(x)
    return s * (1 - s)

# In a network
class SigmoidNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),  # ← Can cause vanishing gradients
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
```

---

## 3. Tanh (Hyperbolic Tangent)

### Formula

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1 + e^{-2x}} - 1$$

### Properties

| Property | Value |
|----------|-------|
| Output range | $(-1, 1)$ |
| Derivative | $\tanh'(x) = 1 - \tanh^2(x)$ |
| Center of output | $0$ (zero-centered!) |
| Monotonic? | Yes |

### Shape and Behavior

```
Output tanh(x)
   1.0 ┤           ┌───────────────────
       │          ╱
   0.5 ┤        ╱
       │       ╱
   0.0 ┤......╱......................................
       │    ╱
  -0.5 ┤  ╱
       │ ╱
  -1.0 ┤─────────────────────────────────→ x
       └──────┼──────┼──────┼──────
            -4      0      4
```

### Gradient Behavior

```
Gradient tanh'(x) = 1 - tanh²(x)
  1.0 ┤           ╱╲
      │          ╱  ╲
      │         ╱    ╲
  0.0 ┤────────╱      ╲─────────→ x
      └───────┼──────┼──────
             -4      4

  Maximum gradient = 1.0 (at x=0) ← Better than Sigmoid's 0.25!
  Still saturates for |x| > 3
```

### Advantages Over Sigmoid

| Aspect | Sigmoid | Tanh |
|--------|---------|------|
| Zero-centered | No | Yes |
| Max gradient | 0.25 | 1.0 |
| Gradient strength | Weaker | Stronger |
| Output range | $(0, 1)$ | $(-1, 1)$ |

**Tanh is generally preferred over Sigmoid for hidden layers** because it's zero-centered and has stronger gradients.

### Still Suffers From Vanishing Gradients

For $|x| > 3$, the gradient approaches zero:

$$\tanh'(x) = 1 - \tanh^2(x) \approx 0 \text{ when } |x| > 3$$

### PyTorch Code

```python
tanh = nn.Tanh()
x = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
print(f"Tanh: {tanh(x)}")
# Output: tensor([-0.9993, -0.9640,  0.0000,  0.9640,  0.9993])

# Manual
def tanh_manual(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def tanh_derivative(x):
    t = tanh_manual(x)
    return 1 - t ** 2

# Use in network (better than sigmoid for hidden layers)
class TanhNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),      # ← Still vanishes for large |x|
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
```

---

## 4. Softmax

### Formula

For a vector $\mathbf{z} = [z_1, z_2, \ldots, z_K]$, the softmax function is:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### Properties

| Property | Value |
|----------|-------|
| Output range | $(0, 1)$ per component |
| Sum of outputs | $\sum_i \text{Softmax}(z_i) = 1$ |
| Use | **Multi-class classification** output layer |
| Monotonic? | No (relative to other components) |

### What Softmax Does

Softmax converts a vector of arbitrary real numbers (logits) into a **probability distribution**:

```
Logits (raw scores)         Probabilities (after Softmax)
Class 0:  [2.0, 1.0, 0.1]  ──→  [0.659, 0.242, 0.099]
Class 1:  [5.0, 0.0, 0.0]  ──→  [0.993, 0.003, 0.003]
Class 2:  [0.5, 0.5, 0.5]  ──→  [0.333, 0.333, 0.333]

Properties:
  - All outputs are positive: p_i > 0
  - All outputs sum to 1: Σ p_i = 1
  - Higher logit → higher probability
  - Difference of logits is amplified by exp()
```

### Numerical Stability Trick

Raw softmax can cause overflow with large values. The numerically stable version:

$$\text{Softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{K} e^{z_j - \max(\mathbf{z})}}$$

Subtracting the maximum prevents overflow since all exponents are $\leq 0$.

### Softmax + Cross-Entropy Loss

Softmax is almost always used with **categorical cross-entropy**:

$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\text{Softmax}(z_c))$$

Where $y_c$ is the one-hot encoded true class.

### PyTorch Code

```python
# Softmax for inference
softmax = nn.Softmax(dim=1)
logits = torch.tensor([[2.0, 1.0, 0.1]])
probs = softmax(logits)
print(f"Probabilities: {probs}")
# Output: tensor([[0.6590, 0.2424, 0.0986]])

# Numerically stable manual softmax
def stable_softmax(x):
    x_max = x.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=1, keepdim=True)

# IMPORTANT: For training, use LogSoftmax + NLLLoss
# (more numerically stable than Softmax + CrossEntropyLoss)
log_softmax = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()

# OR equivalently (most common in PyTorch):
criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax + NLLLoss
# → Do NOT apply softmax before CrossEntropyLoss!
```

```python
# Full classification example
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
            # NO softmax here — CrossEntropyLoss handles it
        )
    
    def forward(self, x):
        return self.network(x)

model = Classifier(784, 128, 10)
criterion = nn.CrossEntropyLoss()  # Includes softmax

logits = model(x)  # Raw scores
loss = criterion(logits, targets)  # Applies softmax internally
```

---

## 5. ReLU (Rectified Linear Unit)

### Formula

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### Properties

| Property | Value |
|----------|-------|
| Output range | $[0, \infty)$ |
| Derivative | $\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$ |
| Non-saturating | Yes (for positive region) |
| Computation | Extremely cheap (comparison + max) |

### Shape and Behavior

```
Output ReLU(x)
     │          ╱
   3 ┤        ╱
     │       ╱
   2 ┤      ╱
     │     ╱
   1 ┤    ╱
     │   ╱
   0 ┤──╱────────────────────────→ x
     │
  -1 ┤
     │
  -2 ┤
     └────┼────┼────┼────→
          -2    0    2

  For x > 0: gradient = 1 (constant, no saturation!)
  For x ≤ 0: gradient = 0 (neuron is "dead")
```

### Why ReLU Revolutionized Deep Learning

```
Comparison of Gradient Flow:

Sigmoid Network (4 layers):
Input → σ(z) → σ(z) → σ(z) → σ(z) → Output
Gradients: 0.25 × 0.25 × 0.25 × 0.25 = 0.0039 (vanishes!)

ReLU Network (4 layers):
Input → ReLU → ReLU → ReLU → ReLU → Output
Gradients: 1 × 1 × 1 × 1 = 1 (preserved!)
(So long as neurons stay in the positive region)
```

### The Dying ReLU Problem

When a neuron's weights are updated such that the input is always negative, the gradient is always zero:

$$\text{If } \mathbf{w}^T\mathbf{x} + b \leq 0 \text{ for all training samples: gradient = 0 forever}$$

The neuron "dies" and never recovers.

```
Dying ReLU:

    ╱│  ╱
   ╱ │ ╱
  ╱  │╱
 ─────╱──────────────── ← Active region (gradient = 1)
     │
     │
     │
     └ ← Dead region (gradient = 0, never updates)
```

**Mitigation:** Careful initialization (He initialization), smaller learning rates, or use Leaky ReLU.

### PyTorch Code

```python
relu = nn.ReLU()

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"ReLU: {relu(x)}")
# Output: tensor([0., 0., 0., 1., 3.])

# ReLU network (standard for most deep networks)
class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# He initialization (designed for ReLU)
def init_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)

model = ReLUNetwork(784, [256, 128, 64], 10)
model.apply(init_weights_he)
```

---

## 6. Leaky ReLU

### Formula

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

Where $\alpha$ is a small positive constant (typically $\alpha = 0.01$).

### Shape and Behavior

```
Output LeakyReLU(x, α=0.01)
     │          ╱
   3 ┤        ╱
     │       ╱
   1 ┤      ╱
     │     ╱
   0 ┤────╱──────────────────────→ x
     │  ╱
     │ ╱  (small slope α=0.01)
  -1 ┤╱
     └────┼────┼────┼────→
          -2    0    2

  For x > 0: gradient = 1
  For x ≤ 0: gradient = α = 0.01 (small but non-zero!)
  → No dead neurons!
```

### Why Leaky ReLU?

| Aspect | ReLU | Leaky ReLU |
|--------|------|------------|
| Gradient (x > 0) | 1 | 1 |
| Gradient (x ≤ 0) | 0 | $\alpha$ (e.g., 0.01) |
| Dead neurons? | Yes | No |
| Computation | Max(0, x) | Conditional |
| Performance | Excellent | Slightly better in some cases |

### Parametric ReLU (PReLU)

PReLU makes $\alpha$ a learnable parameter:

$$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha_i x & \text{if } x \leq 0 \end{cases}$$

Where $\alpha_i$ is learned during training (can be different per channel).

### PyTorch Code

```python
# Leaky ReLU with α=0.01 (default)
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"LeakyReLU: {leaky_relu(x)}")
# Output: tensor([-0.0300, -0.0100,  0.0000,  1.0000,  3.0000])

# PReLU (learnable α)
prelu = nn.PReLU(num_parameters=1)  # One α for all channels
print(f"Initial α: {prelu.weight.data}")
# Output: tensor([0.25])  ← default initialization

# In a network
class LeakyReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),  # Standard Leaky ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),           # Learnable Leaky ReLU
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```

---

## 7. Comparison Summary

```
Shape Comparison:

    1.0 ┤ ─ ─ ─ ─ Sigmoid
        │          ╱
    0.5 ┤ ........╱................. Sigmoid
        │   ╱───╱
    0.0 ┤──╱──────────────────────── ReLU (kink at 0)
        │╱  ╲                        
   -0.5 ┤     ╲─────────── Tanh    
        │       ╲                    
   -1.0 ┤────────────────────────────
        └────┼────┼────┼────┼────→ x
             -4   -1    1    4
```

| Function | Range | Zero-centered? | Gradient Issues | Best For |
|----------|-------|---------------|-----------------|----------|
| **Sigmoid** | $(0, 1)$ | No | Vanishing (max 0.25) | Binary output, gates |
| **Tanh** | $(-1, 1)$ | Yes | Vanishing (for \|x\|>3) | Hidden layers (RNNs) |
| **Softmax** | $(0, 1)$, sums to 1 | N/A | N/A (output only) | Multi-class output |
| **ReLU** | $[0, \infty)$ | No | Dying neurons | Hidden layers (default) |
| **Leaky ReLU** | $(-\infty, \infty)$ | No | Minimal | Alternative to ReLU |

### Modern Activations Worth Knowing

- **ELU**: $f(x) = x$ if $x > 0$, else $\alpha(e^x - 1)$. Smooth, zero-centered.
- **GELU**: $f(x) = x \cdot \Phi(x)$ where $\Phi$ is the CDF of $\mathcal{N}(0,1)$. Used in Transformers (GPT, BERT).
- **Swish**: $f(x) = x \cdot \sigma(\beta x)$. Self-gated, smooth. Used in EfficientNet.
- **Mish**: $f(x) = x \cdot \tanh(\ln(1 + e^x))$. Smooth, used in YOLOv4.

---

## 8. Exercises

### Exercise 1: Compute Activations by Hand
Given input $x = 2.0$, compute the output of: (a) Sigmoid, (b) Tanh, (c) ReLU, (d) Leaky ReLU with $\alpha = 0.05$.

**Solution:**
- (a) $\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1.1353} = 0.8808$
- (b) $\tanh(2) = \frac{e^2 - e^{-2}}{e^2 + e^{-2}} = \frac{7.3891 - 0.1353}{7.3891 + 0.1353} = 0.9640$
- (c) $\text{ReLU}(2) = \max(0, 2) = 2.0$
- (d) $\text{LeakyReLU}(2) = 2.0$ (since $2 > 0$)

For $x = -1.5$:
- (a) $\sigma(-1.5) = \frac{1}{1 + e^{1.5}} = 0.1824$
- (b) $\tanh(-1.5) = -0.9051$
- (c) $\text{ReLU}(-1.5) = 0$
- (d) $\text{LeakyReLU}(-1.5) = 0.05 \times (-1.5) = -0.075$

### Exercise 2: Vanishing Gradient Demonstration
Compute the gradient of a 5-layer sigmoid network at layer 1, assuming each layer's pre-activation is $z = 0.5$.

**Solution:**
At each layer: $\sigma'(0.5) = \sigma(0.5)(1 - \sigma(0.5)) = 0.6225 \times 0.3775 = 0.2350$

Gradient at layer 1 through 4 sigmoid layers:
$$(0.2350)^4 = 0.00305$$

**Only 0.3% of the gradient reaches layer 1!** With ReLU and positive activations, it would be $1^4 = 1$.

### Exercise 3: Softmax by Hand
Compute softmax of the vector $[1.0, 2.0, 3.0]$.

**Solution:**
$$e^{1.0} = 2.718, \quad e^{2.0} = 7.389, \quad e^{3.0} = 20.086$$
$$\text{sum} = 2.718 + 7.389 + 20.086 = 30.193$$
$$\text{Softmax} = \left[\frac{2.718}{30.193}, \frac{7.389}{30.193}, \frac{20.086}{30.193}\right] = [0.0900, 0.2447, 0.6652]$$

### Exercise 4: Implement and Compare All Activations

```python
import torch
import torch.nn as nn

# Generate data
torch.manual_seed(42)
X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

# Test each activation in a 3-layer MLP
activations = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.01),
}

results = {}
for name, act in activations.items():
    model = nn.Sequential(
        nn.Linear(20, 64), act,
        nn.Linear(64, 32), act,
        nn.Linear(32, 1), nn.Sigmoid()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(200):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        acc = ((model(X) > 0.5).float() == y).float().mean()
    results[name] = acc.item()
    print(f"{name:12s}: Accuracy = {acc:.4f}, Final Loss = {loss:.4f}")

# Observation: ReLU and LeakyReLU typically converge fastest
# due to non-saturating gradients
```

---

## 9. Summary

| Decision Rule | Recommendation |
|--------------|----------------|
| Hidden layer default | **ReLU** (fast, effective, widely used) |
| If dying neurons occur | **Leaky ReLU** or **ELU** |
| RNN/LSTM gates | **Sigmoid** (for gates) or **Tanh** (for state) |
| Multi-class output | **Softmax** (with CrossEntropyLoss) |
| Binary output | **Sigmoid** (with BCELoss) |
| Transformer models | **GELU** or **Swish** |

The choice of activation function is a critical hyperparameter. Start with ReLU for hidden layers and softmax/sigmoid for outputs. Switch to Leaky ReLU or more modern activations if you encounter training difficulties.
