# Optimization Algorithms & Types

## Why This Topic Matters

Optimization algorithms are the **engines that drive neural network training**. While the network architecture defines *what* can be learned, the optimizer determines *how efficiently* it's learned. A poorly chosen optimizer can cause training to stall, oscillate, or diverge — even with a perfect architecture.

The evolution from vanilla SGD to Adam represents one of the most important practical advances in deep learning. Modern optimizers adapt learning rates per parameter, handle sparse gradients, and incorporate momentum — all of which dramatically speed up convergence. Understanding these algorithms is essential for debugging training failures and achieving state-of-the-art results.

---

## 1. The Optimization Problem

### Formal Definition

Given a neural network with parameters $\theta$, training data $\{(x_i, y_i)\}_{i=1}^{N}$, and a loss function $\mathcal{L}$, we seek:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f(x_i; \theta), y_i)$$

This is typically solved iteratively:

$$\theta_{t+1} = \theta_t + \Delta \theta_t$$

Where $\Delta \theta_t$ is the **update rule**, and its form defines the optimization algorithm.

### The Loss Landscape

```
Loss Landscape (2D parameter space):

High Loss    ╱╲     ╱╲
             ╱  ╲   ╱  ╲       ╱╲
            ╱    ╲ ╱    ╲     ╱  ╲
           ╱      ╳      ╲   ╱    ╲
          ╱     ╱    ╲     ╲ ╱      ╲
         ╱    ╱        ╲     ╳       ╲
        ╱   ╱   Saddle   ╲  ╱ ╲        ╲
       ╱  ╱      Point    ╲╱   ╲        ╲
      ╱ ╱                      ╲         ╲
     ╱╱                        ╲          ╲
Low Loss ═══════════════════════════════════════

Goal: Navigate from random start to the global (or good local) minimum.

Challenges:
  • Saddle points (flat regions, not true minima)
  • Local minima (deceptive wells)
  • Noisy gradients (from mini-batch sampling)
  • Ill-conditioned curvature (elongated valleys)
```

---

## 2. Gradient Descent Variants

### 2.1 Batch Gradient Descent

Uses the **entire dataset** for each update:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t; \text{all data})$$

```
All N samples ──→ Compute Full Gradient ──→ Update Once

[■■■■■■■■■■■■■■■■■■■■■■■■■] ──→ ∇L_full ──→ θ = θ - α∇L

Pros: Guaranteed convergence (for convex loss)
Cons: One update per epoch, memory-intensive, slow
```

### 2.2 Stochastic Gradient Descent (SGD)

Uses a **single random sample** per update:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}_i(\theta_t)$$

```
Sample 1 ──→ ∇L₁ ──→ Update
Sample 2 ──→ ∇L₂ ──→ Update
Sample 3 ──→ ∇L₃ ──→ Update
...

Pros: Frequent updates, noisy (escapes local minima), low memory
Cons: Very noisy, can't vectorize efficiently
```

### 2.3 Mini-Batch Gradient Descent (Standard)

Uses a **small batch** (typically 32–512) per update:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \frac{1}{B} \sum_{i \in \text{batch}} \mathcal{L}_i(\theta_t)$$

Where $B$ is the batch size.

```
Batch 1: [■■■■■■■■■■■■■■■■■■■■■■■■■] ──→ ∇L ──→ Update
Batch 2: [■■■■■■■■■■■■■■■■■■■■■■■■■] ──→ ∇L ──→ Update
...

Pros: Vectorized (GPU-friendly), stable gradients, frequent updates
Cons: Requires batch size tuning
```

**This is what is used in practice.** When people say "SGD," they usually mean mini-batch SGD.

---

## 3. Momentum

### The Problem with Vanilla SGD

In ill-conditioned loss landscapes (elongated valleys), SGD oscillates across the narrow dimension and makes slow progress along the long dimension:

```
Without Momentum:
               ●→ → → ↗ ↘ ↗ ↘
              ↗         ↘         ↗
            ↗                      ↘●  (slow!)
    ──────────────────────────────────────→ θ₂
    │
    │ Valley: narrow in θ₁, long in θ₂
    ↓
    θ₁

With Momentum:
    ●→→→→→→→→→→→→→→→→→→→→→→→→→→→→→●  (fast!)
    ──────────────────────────────────────→ θ₂
    │         Accumulates velocity in the right direction
    ↓
    θ₁
```

### Momentum Update Rule

Momentum accumulates a **velocity vector** that dampens oscillations:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha \mathbf{v}_t$$

Where:
- $\mathbf{v}_t$ is the velocity (exponentially weighted moving average of gradients)
- $\beta$ is the momentum coefficient (typically $\beta = 0.9$)
- $\alpha$ is the learning rate

**Intuition:** A ball rolling downhill accumulates momentum, helping it push past small bumps and oscillations.

### Nesterov Accelerated Gradient (NAG)

NAG looks ahead by computing the gradient at the *anticipated* position:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_t - \alpha \beta \mathbf{v}_{t-1})$$
$$\theta_{t+1} = \theta_t - \alpha \mathbf{v}_t$$

```
Regular Momentum:                Nesterov:
Current: ●                        Current: ●
         │                                  │
         └─→ Compute grad at ●              └─→ Look ahead: ○
            Then jump past                    Compute grad at ○
                                              Then jump past ○

NAG "anticipates" where momentum will take us, providing a corrective gradient.
```

---

## 4. Adaptive Learning Rate Methods

### 4.1 Adagrad (Adaptive Gradient)

Adagrad adapts the learning rate per parameter based on historical gradient magnitudes:

$$\theta_{t+1,i} = \theta_{t,i} - \frac{\alpha}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}$$

Where:
- $G_t = \sum_{\tau=1}^{t} g_\tau g_\tau^T$ is the accumulated sum of squared gradients
- $g_{t,i}$ is the gradient for parameter $i$ at step $t$
- $\epsilon$ is a small constant for numerical stability ($10^{-8}$)

```
Effect: Parameters with large historical gradients get smaller learning rates.
       Parameters with small historical gradients get larger learning rates.

Parameter A: ████████████████ (many large gradients) → lr decreases
Parameter B: ██               (few small gradients)  → lr increases
```

**Problem:** The learning rate only decreases, never increases. It can become too small and stop learning entirely.

### 4.2 RMSprop (Root Mean Square Propagation)

RMSprop fixes Adagrad's decay problem by using an **exponentially decaying average**:

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

Where $\beta$ is the decay rate (typically $\beta = 0.9$).

```
RMSprop vs Adagrad:

Adagrad:  lr ████████████████████ (monotonically decreasing to ~0)
RMSprop:  lr ████████░░████████░░ (adapts — can increase if gradients shrink)

The "leaky" accumulation allows the learning rate to recover.
```

### 4.3 Adam (Adaptive Moment Estimation)

Adam combines **momentum** (first moment) with **RMSprop** (second moment):

**Step 1: Compute biased first and second moment estimates:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(momentum / first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(RMSprop / second moment)}$$

**Step 2: Bias correction (critical for early steps):**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Step 3: Parameter update:**

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

Where:
- $\beta_1 = 0.9$ (momentum)
- $\beta_2 = 0.999$ (second moment)
- $\epsilon = 10^{-8}$ (numerical stability)
- $\alpha = 0.001$ (default learning rate — often a good starting point)

```
Adam Update Visualization:

Step 1: Accumulate moments
  m_t = 0.9 × m_{t-1} + 0.1 × g_t       (gradient direction)
  v_t = 0.999 × v_{t-1} + 0.001 × g_t²  (gradient magnitude)

Step 2: Bias correction
  ĥ_m_t = m_t / (1 - 0.9^t)     (corrects early-step bias)
  ĥ_v_t = v_t / (1 - 0.999^t)

Step 3: Update
  θ = θ - (lr / √ĥ_v_t) × ĥ_m_t

  ← Larger gradients → Larger v_t → Smaller effective lr
  ← Consistent direction → Larger m_t → Faster movement
```

### 4.4 AdamW (Adam with Decoupled Weight Decay)

Standard Adam + L2 regularization can conflict. AdamW decouples weight decay:

$$\theta_{t+1} = \theta_t - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

Where $\lambda$ is the weight decay coefficient. This is now the **default optimizer for training large language models** (GPT, BERT, LLaMA, etc.).

---

## 5. Optimizer Comparison

```
Convergence Speed vs. Hyperparameter Sensitivity:

Fast convergence
     ↑
     │  Adam ●
     │  AdamW ●
     │
     │     RMSprop ●
     │
     │  Momentum ●
     │       SGD ●
     │
     └──────────────────────────→ Less sensitive to lr
       Sensitive                  Robust

General trend:
  • Adam/AdamW: Fast convergence, works well out of the box
  • SGD + Momentum: Slower but can generalize better (common in CV)
  • RMSprop: Good for RNNs (historically)
  • AdamW: Best for Transformers and large models
```

| Optimizer | Hyperparameters | Convergence | Generalization | Best For |
|-----------|----------------|-------------|----------------|----------|
| SGD | lr | Slow | Excellent | CV (with tuned LR schedule) |
| SGD + Momentum | lr, β | Medium | Excellent | CV, small datasets |
| Adagrad | lr, ε | Medium (decays) | Fair | Sparse data |
| RMSprop | lr, β, ε | Fast | Good | RNNs |
| **Adam** | lr, β₁, β₂, ε | Fast | Good | General-purpose |
| **AdamW** | lr, β₁, β₂, ε, λ | Fast | Very Good | Transformers, NLP |

---

## 6. Learning Rate Scheduling

### Why Schedule?

A high learning rate helps escape saddle points early; a low learning rate refines the solution later.

```
Training Phases:

Loss │
     │╲
     │ ╲       Phase 1: High LR (exploration)
     │  ╲
     │   ╲
     │    ╲╱‾‾‾‾‾╲        Phase 2: Decay (refinement)
     │           ╲
     │            ╲╱‾‾‾‾‾‾‾‾──→ Minimum
     └────────────────────────── Epoch
```

### Common Schedules

#### Step Decay
$$\alpha_t = \alpha_0 \times \gamma^{\lfloor t / T \rfloor}$$

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

#### Cosine Annealing
$$\alpha_t = \frac{\alpha_{\min}}{2} + \frac{\alpha_{\max} - \alpha_{\min}}{2} \left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

#### Warmup + Cosine Decay (Modern Standard)
Linearly increase LR for first $W$ steps, then cosine decay:

```python
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)
    ],
    milestones=[1000]
)
```

#### ReduceLROnPlateau
Reduce LR by factor when metric plateaus:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# Call: scheduler.step(val_loss) after each epoch
```

---

## 7. Gradient Clipping

For RNNs and large models, gradients can explode. Clipping prevents this:

### By Norm
$$\text{If } \|\mathbf{g}\| > c: \quad \mathbf{g} \leftarrow \frac{c}{\|\mathbf{g}\|} \mathbf{g}$$

### By Value
$$\text{If } |g_i| > c: \quad g_i \leftarrow \text{sign}(g_i) \cdot c$$

```python
# Before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# OR
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

## 8. PyTorch Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Data ──
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

# ── Model ──
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

model = Net()

# ══════════════════════════════════════════════
# Try different optimizers — uncomment one:
# ══════════════════════════════════════════════

# 1. Vanilla SGD
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. SGD + Momentum
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 3. Adagrad
# optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# 4. RMSprop
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

# 5. Adam (most popular default)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# 6. AdamW (best for Transformers)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

# ── Learning Rate Scheduler ──
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ── Training ──
for epoch in range(20):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping (important for RNNs, safe for all)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(1)
            correct += pred.eq(target).sum().item()
    
    acc = 100.0 * correct / len(test_data)
    print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | "
          f"Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
```

---

## 9. Real-World ML Connections

| Domain | Preferred Optimizer | Why |
|--------|--------------------|----|
| Image Classification (ResNet) | SGD + Momentum + Cosine LR | Better generalization with tuned schedule |
| Object Detection (YOLO) | SGD + Momentum | Established best practice |
| NLP / Transformers | AdamW + Warmup + Cosine | Handles large sparse gradients |
| GAN Training | Adam (β₁=0.5) | Low momentum stabilizes adversarial training |
| Reinforcement Learning | Adam / RMSprop | Non-stationary objectives |
| Fine-tuning (Transfer) | AdamW (lr=1e-5) | Small LR prevents catastrophic forgetting |

---

## 10. Exercises

### Exercise 1: SGD vs Adam Comparison
Train the same MLP on MNIST with (a) SGD (lr=0.01) and (b) Adam (lr=0.001) for 10 epochs. Compare convergence speed and final accuracy.

**Solution:**
```python
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.ReLU(), nn.Linear(128,10))
    def forward(self, x): return self.net(x)

for name, opt_cls, lr in [("SGD", optim.SGD, 0.01), ("Adam", optim.Adam, 0.001)]:
    model = MLP()
    optimizer = opt_cls(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    model.eval()
    correct = sum((model(data).argmax(1) == target).sum().item() for data, target in train_loader)
    print(f"{name:5s}: Accuracy after 10 epochs = {100*correct/len(train_data):.2f}%")
# Adam typically achieves >95% in 10 epochs while SGD reaches ~85-90%
```

### Exercise 2: Derive Adam Update for One Step
Given: $g_1 = [2, -1]$, $\alpha = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Compute the first Adam update starting from $m_0 = v_0 = [0, 0]$ and $\theta_0 = [1, 1]$.

**Solution:**
$m_1 = 0.9 \times [0, 0] + 0.1 \times [2, -1] = [0.2, -0.1]$
$v_1 = 0.999 \times [0, 0] + 0.001 \times [4, 1] = [0.004, 0.001]$
$\hat{m}_1 = [0.2, -0.1] / (1 - 0.9) = [2, -1]$
$\hat{v}_1 = [0.004, 0.001] / (1 - 0.999) = [4, 1]$

$\theta_1 = [1, 1] - 0.01 \times [2/\sqrt{4}, -1/\sqrt{1}] = [1, 1] - 0.01 \times [1, -1]$
$\theta_1 = [0.99, 1.01]$

### Exercise 3: Learning Rate Schedule Visualization
Plot the learning rate over 100 epochs for StepLR, CosineAnnealing, and ExponentialLR.

**Solution:**
```python
import torch.optim as optim
import matplotlib.pyplot as plt

model = nn.Linear(10, 2)
lr_history = {}

for name, scheduler_fn in [
    ("StepLR", lambda o: optim.lr_scheduler.StepLR(o, step_size=30, gamma=0.5)),
    ("Cosine", lambda o: optim.lr_scheduler.CosineAnnealingLR(o, T_max=100)),
    ("ExpLR", lambda o: optim.lr_scheduler.ExponentialLR(o, gamma=0.98)),
]:
    opt = optim.SGD(model.parameters(), lr=0.1)
    sched = scheduler_fn(opt)
    lrs = []
    for epoch in range(100):
        lrs.append(sched.get_last_lr()[0])
        sched.step()
    lr_history[name] = lrs

for name, lrs in lr_history.items():
    plt.plot(lrs, label=name)
plt.xlabel("Epoch"); plt.ylabel("Learning Rate")
plt.legend(); plt.title("LR Schedules"); plt.show()
```

### Exercise 4: Gradient Clipping Impact
Train an RNN on a sequence task with and without gradient clipping. Observe gradient norms and convergence.

**Solution:**
```python
import torch, torch.nn as nn

# Generate sine wave sequences
sequences = [torch.sin(torch.linspace(i, i+10, 50)).unsqueeze(1) for i in range(100)]
targets = [seq[1:] for seq in sequences]
inputs = [seq[:-1] for seq in sequences]

class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

for clip in [None, 0.5, 1.0]:
    model = SimpleRNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        for inp, tgt in zip(inputs, targets):
            optimizer.zero_grad()
            loss = criterion(model(inp), tgt)
            loss.backward()
            
            if clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
    
    print(f"Clip={clip}: Final loss ≈ {loss.item():.6f}")
```

---

## 11. Summary

| Algorithm | Key Idea | When to Use |
|-----------|----------|-------------|
| **SGD** | Simple gradient step | Baseline, academic benchmarks |
| **SGD + Momentum** | Velocity accumulation | Computer vision (with LR schedule) |
| **Adagrad** | Per-param LR from squared gradients | Sparse data (NLP word embeddings) |
| **RMSprop** | Decaying squared gradient average | RNNs |
| **Adam** | Momentum + RMSprop + bias correction | Default for most tasks |
| **AdamW** | Adam + decoupled weight decay | Transformers, fine-tuning |

**Practical Rules of Thumb:**
1. Start with **Adam** (lr=0.001) for quick prototyping
2. Switch to **SGD + Momentum** + **Cosine LR** for final CV models
3. Use **AdamW** for Transformers and NLP tasks
4. Always monitor training/validation loss curves
5. Use learning rate warmup for large models
6. Apply gradient clipping for RNNs and GANs
