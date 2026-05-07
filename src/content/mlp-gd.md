# Multi-Layer Perceptron (MLP) & Gradient Descent Algorithms

## Why This Topic Matters

The Multi-Layer Perceptron (MLP) is the **foundational deep learning architecture**. Every advanced architecture — CNNs, RNNs, Transformers — builds upon the principles of the MLP: weighted connections, activation functions, and backpropagation. Understanding the MLP and how it learns via gradient descent is essential before tackling any specialized neural network.

Gradient descent is the **engine that makes neural networks learn**. It is the optimization algorithm used to minimize the loss function by iteratively adjusting the network's weights. Without gradient descent, neural networks would be static mathematical functions incapable of learning from data.

---

## 1. The Perceptron: Where It All Started

### Single Perceptron

The perceptron, proposed by Frank Rosenblatt in 1958, is the simplest neural unit:

```
              ┌───────────┐
 x₁ ──w₁──→ │           │
 x₂ ──w₂──→ │  Σ + b    │──→ f(·) ──→ ŷ
 x₃ ──w₃──→ │           │
              └───────────┘

Where:
  Σ = x₁w₁ + x₂w₂ + x₃w₃ (weighted sum)
  b = bias (threshold adjustment)
  f(·) = activation function (step function for perceptron)
```

### The Mathematical Model

A single perceptron computes:

$$\hat{y} = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^T \mathbf{x} + b)$$

In vector notation:
$$\hat{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$$

Where:
- $\mathbf{x} \in \mathbb{R}^{n}$ is the input vector
- $\mathbf{W} \in \mathbb{R}^{m \times n}$ is the weight matrix
- $\mathbf{b} \in \mathbb{R}^{m}$ is the bias vector
- $f(\cdot)$ is the activation function
- $\hat{y} \in \mathbb{R}^{m}$ is the output

### Limitation of the Single Perceptron

A single perceptron with a step activation can only learn **linearly separable** functions. It cannot solve the XOR problem:

```
XOR Truth Table:
  x₁  x₂  │  Output
  ──── ─── │ ───────
   0   0   │    0
   0   1   │    1
   1   0   │    1
   1   1   │    0

No single line can separate the 1s from the 0s!

    x₂
    1 │   ●       ●
      │
    0 │       ●   ●
      └──────────── x₁
        0      1

  ● = class 0    ○ = class 1
  (The points are not linearly separable)
```

---

## 2. The Multi-Layer Perceptron (MLP)

### Architecture

The MLP solves the XOR problem by adding **hidden layers** with non-linear activation functions:

```
Input    Hidden     Hidden     Output
Layer    Layer 1    Layer 2    Layer
(2)      (4)        (4)        (1)

 x₁ ─────┬──── h₁ ──┬──── h₅ ──┬──── ŷ
         ├──── h₂ ──┤     h₆ ──┤
 x₂ ─────┼──── h₃ ──┤     h₇ ──┤
         └──── h₄ ──┘     h₈ ──┘

  Each connection has a learnable weight.
  Each neuron has an activation function (e.g., ReLU).
  The combination of layers creates a complex decision boundary.
```

### Forward Pass — Step by Step

Given input $\mathbf{x}$, an MLP with $L$ layers computes:

**Layer $\ell$ forward pass:**
$$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$
$$\mathbf{a}^{(\ell)} = f(\mathbf{z}^{(\ell)})$$

Where:
- $\mathbf{z}^{(\ell)}$ is the pre-activation (linear transform) at layer $\ell$
- $\mathbf{a}^{(\ell)}$ is the post-activation (after applying $f$) at layer $\ell$
- $\mathbf{a}^{(0)} = \mathbf{x}$ (input is layer 0)
- $\mathbf{a}^{(L)} = \hat{\mathbf{y}}$ (output of last layer)

### Universal Approximation Theorem

**Theorem (Cybenko 1989, Hornik 1991):** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$.

Formally: For any continuous function $g: \mathbb{R}^n \rightarrow \mathbb{R}$ and $\epsilon > 0$, there exists weights $\mathbf{W}, \mathbf{b}$ such that:

$$|f(\mathbf{x}; \mathbf{W}, \mathbf{b}) - g(\mathbf{x})| < \epsilon \quad \forall \mathbf{x}$$

**Caveat:** This theorem guarantees *existence* but not *learnability*. Deep networks (many layers) are often more parameter-efficient than very wide shallow networks for complex functions.

---

## 3. Loss Functions

### Mean Squared Error (MSE) — Regression

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### Binary Cross-Entropy — Binary Classification

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### Categorical Cross-Entropy — Multi-class Classification

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Where $C$ is the number of classes and $y_{i,c}$ is a one-hot encoded label.

---

## 4. Gradient Descent — The Core Optimization Algorithm

### The Intuition

Imagine you're standing on a mountain in thick fog and need to reach the valley below. You can't see the bottom, but you can feel the slope under your feet. You take a step in the **steepest downhill direction**. Repeat until you reach the bottom.

```
Loss Landscape (1D visualization):

Loss (J)  │
          │    ╲
          │     ╲
          │      ╲ ← gradient points uphill
          │       ╲
          │        ╲
          │         ● ← start (random weights)
          │          ╲
          │           ╲
          │            ╲
          │             ● ← minimum (optimal weights)
          └────────────────── Weight (w)
          
  We move in the OPPOSITE direction of the gradient:
  w_new = w_old - α × gradient
```

### The Gradient Descent Update Rule

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

Where:
- $\theta$ represents all learnable parameters (weights and biases)
- $\alpha$ (or $\eta$) is the **learning rate** — controls step size
- $\nabla_\theta \mathcal{L}$ is the gradient of the loss with respect to parameters

### The Learning Rate ($\alpha$)

The learning rate is arguably the **most important hyperparameter**:

```
Too Small α:                  Optimal α:                Too Large α:
━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━
●→ → → → → ... (slow)        ●→→→● (fast converge)     ●→→→→↗↘↗↘ (diverge)

Loss                          Loss                      Loss
 │  ┌──────┐                 │    ╲                     │ ↗  ╲  ↗
 │  │      │                 │     ╲                    │    ╲↗  ╲↗
 │  │      │                 │      ●                    │  ↗     ╲
 │  │      │                 │                          │↗  (oscillates
 └──┘      └──              └──                         └── and diverges!)
 Time                          Time                       Time
```

### Variants of Gradient Descent

| Variant | Batch Size | Pros | Cons |
|---------|-----------|------|------|
| **Batch GD** | Full dataset (N) | Stable gradient | Very slow; memory-intensive |
| **Stochastic GD (SGD)** | 1 sample | Fast updates; noisy (escapes local minima) | Very noisy; can't vectorize |
| **Mini-batch GD** | 32–512 samples | Best of both worlds | Requires tuning batch size |

```
Data: [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀]

Batch GD:     Use ALL samples → compute gradient → update once
              [x₁, x₂, ..., x₁₀] → ∇L → θ = θ - α∇L

SGD:          Use ONE sample at a time → update 10 times
              [x₁] → ∇L → θ = θ - α∇L
              [x₂] → ∇L → θ = θ - α∇L
              ...

Mini-batch:   Use BATCH of 4 → update 2-3 times
              [x₁, x₂, x₃, x₄] → ∇L → θ = θ - α∇L
              [x₅, x₆, x₇, x₈] → ∇L → θ = θ - α∇L
              [x₉, x₁₀]       → ∇L → θ = θ - α∇L
```

---

## 5. Backpropagation — Computing the Gradient

Backpropagation (Rumelhart, Hinton & Williams, 1986) is the algorithm that **efficiently computes gradients** through the chain rule of calculus.

### The Chain Rule

For a network with loss $\mathcal{L}$ depending on output $\hat{y}$ which depends on weights $w$:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Backpropagation Through an MLP

For a 2-layer network:

**Forward pass:**
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = f(\mathbf{z}^{(1)})$$
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}, \quad \hat{\mathbf{y}} = g(\mathbf{z}^{(2)})$$

**Backward pass (compute gradients):**

$$\boldsymbol{\delta}^{(2)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \odot g'(\mathbf{z}^{(2)})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} ({\mathbf{a}^{(1)}})^T, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}$$

$$\boldsymbol{\delta}^{(1)} = ({\mathbf{W}^{(2)}}^T \boldsymbol{\delta}^{(2)}) \odot f'(\mathbf{z}^{(1)})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$$

```
Forward Pass  ──────────────────────────────────→
x → W¹ → z¹ → f → a¹ → W² → z² → g → ŷ → L

Backward Pass ←─────────────────────────────────
L → ∂L/∂ŷ → δ² → ∂L/∂W² → δ¹ → ∂L/∂W¹
                 ↓              ↓
            Update W²,b²    Update W¹,b¹
```

---

## 6. PyTorch Implementation

### Building an MLP from Scratch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════
# 1. Define the MLP Architecture
# ══════════════════════════════════════════════════════════
class MLP(nn.Module):
    """
    Multi-Layer Perceptron for classification.
    
    Architecture:
        Input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Softmax
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation — CrossEntropyLoss includes Softmax)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ══════════════════════════════════════════════════════════
# 2. Create Data (MNIST-like)
# ══════════════════════════════════════════════════════════
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ══════════════════════════════════════════════════════════
# 3. Initialize Model, Loss, Optimizer
# ══════════════════════════════════════════════════════════
model = MLP(input_dim=784, hidden_dims=[256, 128, 64], output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ══════════════════════════════════════════════════════════
# 4. Training Loop
# ══════════════════════════════════════════════════════════
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        # Flatten 28x28 images to 784-dim vectors
        data = data.view(data.size(0), -1)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass (gradient descent)
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Compute gradients (backpropagation)
        optimizer.step()        # Update weights (gradient descent step)
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch}]  Loss: {epoch_loss:.4f}  Acc: {accuracy:.2f}%")
    return epoch_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    
    with torch.no_grad():  # No gradient computation needed
        for data, target in loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"  → Test Accuracy: {accuracy:.2f}%")
    return accuracy

# ══════════════════════════════════════════════════════════
# 5. Run Training
# ══════════════════════════════════════════════════════════
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("=" * 50)

for epoch in range(1, 21):
    train(model, train_loader, criterion, optimizer, epoch)
    evaluate(model, test_loader, criterion)
    scheduler.step()
```

### Manual Backpropagation (Understanding the Math)

```python
import torch

# A simple 2-layer network, manually computed
torch.manual_seed(42)

# Data
x = torch.randn(1, 3)      # Input: 1 sample, 3 features
y = torch.tensor([1.0])     # Target

# Weights
W1 = torch.randn(3, 4, requires_grad=True)   # Layer 1: 3→4
b1 = torch.randn(4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)   # Layer 2: 4→1
b2 = torch.randn(1, requires_grad=True)

# Forward pass
z1 = x @ W1 + b1           # Linear transform
a1 = torch.relu(z1)         # ReLU activation
z2 = a1 @ W2 + b2          # Linear transform
loss = ((z2 - y) ** 2).mean()  # MSE loss

# Backward pass (PyTorch computes automatically)
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"∂L/∂W1 shape: {W1.grad.shape}")  # Should be [3, 4]
print(f"∂L/∂W2 shape: {W2.grad.shape}")  # Should be [4, 1]

# Manual gradient descent step
lr = 0.01
with torch.no_grad():
    W1 -= lr * W1.grad
    b1 -= lr * b1.grad
    W2 -= lr * W2.grad
    b2 -= lr * b2.grad
```

---

## 7. Common Problems and Solutions

### Vanishing Gradients

In deep networks, gradients can become extremely small during backpropagation:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \cdot \underbrace{\prod_{\ell=2}^{L} \mathbf{W}^{(\ell)} \odot f'(\mathbf{z}^{(\ell)})}_{\text{This product can vanish!}}$$

**Solutions:** ReLU activation, batch normalization, residual connections, careful initialization (Xavier/He).

### Exploding Gradients

Gradients can grow exponentially large:

**Solutions:** Gradient clipping, LSTM/GRU cells (for RNNs), proper initialization.

### Local Minima and Saddle Points

```
Loss Landscape:

     ╱╲           Saddle Point
    ╱  ╲    ●────────●
   ╱    ╲  /          ╲
──╱      ╲/            ╲──
   Local   ╲            ╲
   Minimum  ╲   Global  ╲
             ╲  Minimum  ╲
              ╲          ╱
               ╲        ╱

Solutions: Momentum, Adam optimizer, learning rate scheduling
```

---

## 8. Real-World ML Connections

| Domain | MLP Application | Why MLP? |
|--------|----------------|----------|
| Finance | Credit scoring, fraud detection | Tabular data with mixed features |
| Healthcare | Disease diagnosis from lab results | Interpretable architecture |
| Retail | Customer churn prediction | Non-linear patterns in behavior |
| NLP | Text classification | Bag-of-words → MLP works for simple tasks |
| Tabular ML | House price prediction | Outperforms complex models on small tabular data |

---

## 9. Exercises

### Exercise 1: Implement XOR with an MLP
Build and train an MLP to solve the XOR problem.

**Solution:**
```python
import torch
import torch.nn as nn

# XOR data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# XOR MLP
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

predictions = (model(X) > 0.5).float()
print(f"Predictions:\n{predictions.numpy()}")
# Expected: [[0], [1], [1], [0]]
```

### Exercise 2: Learning Rate Experiment
Using the MLP above, train with learning rates of 0.1, 0.01, 0.001, and 0.0001. Observe which converges fastest and which fails. Plot the loss curves.

**Solution:**
```python
learning_rates = [0.1, 0.01, 0.001, 0.0001]
results = {}

for lr in learning_rates:
    model = MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(5):
        epoch_loss = 0
        for data, target in train_loader:
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    
    results[lr] = losses
    print(f"LR={lr}: Losses = {[f'{l:.4f}' for l in losses]}")

# Observation: lr=0.1 may diverge, lr=0.01 is often best,
# lr=0.001 is stable but slow, lr=0.0001 is very slow
```

### Exercise 3: Gradient Computation
For the computation graph $L = (wx + b - y)^2$, compute $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ by hand, then verify with PyTorch.

**Solution:**
Let $z = wx + b$, then $L = (z - y)^2$.

$$\frac{\partial L}{\partial z} = 2(z - y)$$
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = 2(wx + b - y) \cdot x$$
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = 2(wx + b - y)$$

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)
y = torch.tensor(5.0)

z = w * x + b
L = (z - y) ** 2
L.backward()

print(f"∂L/∂w = {w.grad.item():.4f}")  # 2*(2*3+1-5)*3 = 2*2*3 = 12
print(f"∂L/∂b = {b.grad.item():.4f}")  # 2*(2*3+1-5) = 4
```

### Exercise 4: Batch Size Impact
Train the MNIST MLP with batch sizes of 1, 32, 256, and 1024. Compare convergence speed and final accuracy.

**Solution:** (Conceptual answer)
- **Batch 1 (SGD):** Very noisy loss, slow per-epoch time, may reach slightly better minimum due to noise
- **Batch 32:** Good balance, standard choice, smooth convergence
- **Batch 256:** Faster per-epoch, smoother gradients, may converge to slightly sharper minima
- **Batch 1024:** Fastest per-epoch, may miss narrow minima, can generalize worse
- **Key insight:** Larger batches → smoother gradients but may need more epochs to find good solutions

---

## 10. Summary

| Concept | Formula / Idea |
|---------|---------------|
| Perceptron output | $\hat{y} = f(\mathbf{w}^T\mathbf{x} + b)$ |
| Layer forward pass | $\mathbf{a}^{(\ell)} = f(\mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)})$ |
| Gradient descent | $\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}$ |
| Chain rule (backprop) | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$ |
| Universal approximation | Single hidden layer can approximate any continuous function |

The MLP + backpropagation + gradient descent trio forms the backbone of all modern deep learning. Every advanced architecture is, at its core, a creative variation of these principles.
