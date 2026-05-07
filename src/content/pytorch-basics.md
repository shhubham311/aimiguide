# Basics of PyTorch & Tensors

## Why This Topic Matters

PyTorch is the **dominant deep learning framework** in both research and industry. Developed by Meta's AI Research Lab (FAIR), it powers everything from academic papers to production systems at Tesla, Uber, and OpenAI. Understanding PyTorch's core abstractions — tensors, autograd, and modules — is the essential first step before building any neural network.

Tensors are the **fundamental data structure** of PyTorch — they generalize scalars, vectors, matrices, and higher-dimensional arrays into a unified type that supports GPU acceleration, automatic differentiation, and efficient computation. Every neural network operation in PyTorch operates on tensors.

---

## 1. What is a Tensor?

### Definition

A tensor is a **multidimensional array** — a generalization of scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions.

| Name | Dimensions | Notation | Example |
|------|-----------|----------|---------|
| Scalar | 0 | $a$ | Temperature: $72.5$ |
| Vector | 1 | $\mathbf{v}$ | RGB pixel: $[255, 128, 0]$ |
| Matrix | 2 | $\mathbf{A}$ | Grayscale image: $28 \times 28$ |
| 3D Tensor | 3 | $\mathcal{T}$ | Color image: $3 \times 224 \times 224$ |
| 4D Tensor | 4 | $\mathcal{B}$ | Batch of images: $B \times 3 \times H \times W$ |
| 5D Tensor | 5 | $\mathcal{V}$ | Video batch: $B \times T \times C \times H \times W$ |

### Tensor Shapes in Deep Learning

```
Common tensor shapes:

Image Classification:
  Single image:  [3, 224, 224]       (channels, height, width)
  Batch:         [32, 3, 224, 224]   (batch, channels, height, width)
                    B   C    H    W

NLP / Text:
  Single token:  [768]              (embedding dimension)
  Sequence:      [128, 768]         (sequence_length, embedding_dim)
  Batch:         [32, 128, 768]     (batch, seq_len, embed_dim)
                    B    S     E

Times Series:
  Features:      [64, 5]            (time_steps, num_features)
  Batch:         [32, 64, 5]        (batch, time_steps, features)
```

---

## 2. Creating Tensors

```python
import torch

# ════════════════════════════════════════════════
# Basic Creation
# ════════════════════════════════════════════════

# From Python data
scalar = torch.tensor(42)                    # 0D tensor (scalar)
vector = torch.tensor([1.0, 2.0, 3.0])       # 1D tensor (vector)
matrix = torch.tensor([[1, 2], [3, 4]])      # 2D tensor (matrix)

print(scalar.shape)   # torch.Size([])
print(vector.shape)   # torch.Size([3])
print(matrix.shape)   # torch.Size([2, 2])

# Special tensors
zeros = torch.zeros(3, 4)          # 3×4 matrix of zeros
ones = torch.ones(2, 3)            # 2×3 matrix of ones
eye = torch.eye(3)                 # 3×3 identity matrix
rand = torch.rand(2, 3)            # 2×3 random [0, 1)
randn = torch.randn(3, 3)          # 3×3 standard normal
arange = torch.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5) # [0.0, 0.25, 0.5, 0.75, 1.0]

# From NumPy (and back)
import numpy as np
np_array = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.from_numpy(np_array)
back_to_np = tensor_from_np.numpy()

# Match another tensor's shape and dtype
x = torch.randn(3, 4)
like_x = torch.zeros_like(x)       # zeros with same shape/dtype as x
rand_like_x = torch.randn_like(x)  # random with same shape/dtype

# ════════════════════════════════════════════════
# Tensor Properties
# ════════════════════════════════════════════════
t = torch.randn(2, 3, 4)
print(f"Shape:    {t.shape}")          # torch.Size([2, 3, 4])
print(f"Dtype:    {t.dtype}")          # torch.float32
print(f"Device:   {t.device}")         # cpu
print(f"Numel:    {t.numel()}")        # 24 (total elements)
print(f"NDim:     {t.ndim}")           # 3

# Datatypes
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
long_tensor = torch.tensor([0, 1, 2], dtype=torch.long)  # For indices
half_tensor = float_tensor.half()                          # float16
```

---

## 3. Tensor Operations

### Basic Math Operations

```python
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Element-wise operations
print(a + b)          # Addition
print(a - b)          # Subtraction
print(a * b)          # Element-wise multiplication (Hadamard)
print(a / b)          # Element-wise division
print(a ** 2)         # Element-wise power
print(torch.sqrt(a))  # Element-wise square root
print(a.abs())        # Absolute value
print(a.clamp(min=2, max=5))  # Clamp values to [2, 5]

# In-place operations (modify the tensor, suffixed with _)
c = a.clone()
c.add_(b)     # c = c + b (in-place)
c.mul_(2)     # c = c * 2 (in-place)

# ════════════════════════════════════════════════
# Matrix Operations
# ════════════════════════════════════════════════

# Matrix multiplication
print(torch.matmul(a, b))    # 2×2 @ 2×2 = 2×2
print(a @ b)                 # Python operator (equivalent)
print(torch.mm(a, b))        # Same for 2D matrices

# For batched matrix multiplication
batch_a = torch.randn(10, 3, 4)  # 10 matrices of shape 3×4
batch_b = torch.randn(10, 4, 5)  # 10 matrices of shape 4×5
result = torch.bmm(batch_a, batch_b)  # 10 matrices of shape 3×5
# OR: result = batch_a @ batch_b  (also works for batched)

# Transpose
print(a.T)                    # 2×2 transpose
print(a.transpose(0, 1))      # Same as .T for 2D

# Matrix inverse and determinant
A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
print(torch.inverse(A))       # Inverse
print(torch.det(A))           # Determinant

# Dot product (1D only)
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
print(torch.dot(v1, v2))      # 1*4 + 2*5 + 3*6 = 32
```

### Reduction Operations

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print(x.sum())          # 21.0 (sum of all elements)
print(x.mean())         # 3.5  (mean of all elements)
print(x.max())          # 6.0
print(x.min())          # 1.0
print(x.prod())         # 720.0
print(x.std())          # Standard deviation
print(x.var())          # Variance

# Along specific dimensions
print(x.sum(dim=0))     # [5.0, 7.0, 9.0] — sum columns (reduce rows)
print(x.sum(dim=1))     # [6.0, 15.0]     — sum rows (reduce columns)
print(x.mean(dim=0))    # [2.5, 3.5, 4.5]
print(x.max(dim=1))     # (values=tensor([3,6]), indices=tensor([2,2]))

# Keep dimensions
print(x.sum(dim=1, keepdim=True))
# tensor([[6.], [15.]])  — keeps 2D shape
```

### Indexing and Slicing

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(x[0])        # First row:    [0, 1, 2, 3]
print(x[0, 1])     # Element:      1
print(x[:, 1])     # Second column: [1, 5, 9]
print(x[0:2, 1:3]) # Submatrix:    [[1, 2], [5, 6]]
print(x[::2, ::2]) # Every other:  [[0, 2], [8, 10]]

# Boolean indexing
mask = x > 5
print(x[mask])  # [6, 7, 8, 9, 10, 11]

# Fancy indexing
indices = torch.tensor([0, 2])
print(x[indices])  # [[0,1,2,3], [8,9,10,11]]
```

### Reshaping

```python
x = torch.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
print(x.reshape(3, 4))     # 3×4
print(x.view(3, 4))        # Same (requires contiguous memory)
print(x.view(-1, 6))       # Auto-infer: 2×6

# Add/remove dimensions
print(x.unsqueeze(0).shape)  # [1, 12]    — add dim at position 0
print(x.unsqueeze(1).shape)  # [12, 1]    — add dim at position 1
print(x.view(1, -1).shape)   # [1, 12]    — add batch dimension

y = torch.randn(1, 3, 4)
print(y.squeeze(0).shape)    # [3, 4]     — remove dim of size 1
print(y.squeeze().shape)     # [3, 4]     — remove all size-1 dims

# Flatten
z = torch.randn(2, 3, 4)
print(z.flatten().shape)        # [24]
print(z.flatten(start_dim=1).shape)  # [2, 12] — keep batch dim

# Permute (rearrange dimensions)
img = torch.randn(3, 224, 224)  # CHW format
img_hwc = img.permute(1, 2, 0)  # HWC format: [224, 224, 3]

# Concatenate and stack
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print(torch.cat([a, b], dim=0).shape)  # [4, 3] — along rows
print(torch.cat([a, b], dim=1).shape)  # [2, 6] — along columns
print(torch.stack([a, b], dim=0).shape) # [2, 2, 3] — new dimension
```

---

## 4. GPU Acceleration

```python
# Check GPU availability
print(torch.cuda.is_available())          # True/False
print(torch.cuda.device_count())           # Number of GPUs
print(torch.cuda.get_device_name(0))       # GPU name

# Move tensors to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Method 1: .to()
x = torch.randn(3, 4).to(device)
y = torch.randn(3, 4, device='cuda')

# Method 2: .cuda() / .cpu()
z = x.cuda()
w = z.cpu()

# Both tensors must be on the same device for operations!
a = torch.randn(3).to('cpu')
b = torch.randn(3).to('cuda')
# a + b  → ERROR! Different devices
result = a.to('cuda') + b  # OK

# Move entire model to GPU
model = MyModel().to(device)
data = data.to(device)
output = model(data)  # Runs on GPU
```

```
Data Flow: CPU ↔ GPU

  ┌──────────┐                    ┌──────────┐
  │   CPU    │   .to('cuda')      │   GPU    │
  │          │ ──────────────────→ │ (VRAM)   │
  │ RAM      │                    │          │
  │          │   .to('cpu')       │ Tensor   │
  │          │ ←────────────────── │ Cores    │
  └──────────┘                    └──────────┘
       │                               │
  Slow (PCIe bus)             Fast (parallel ops)
  ~10-30 GB/s                 ~10+ TFLOPS
```

---

## 5. Autograd: Automatic Differentiation

### The Core Idea

PyTorch's autograd automatically computes gradients for any computation graph. Simply set `requires_grad=True` on tensors and call `.backward()` on the output.

```python
import torch

# Create tensors with gradient tracking
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass: y = w*x + b
y = w * x + b

# Compute gradient: dy/dw, dy/dx, dy/db
y.backward()

print(f"dy/dw = {w.grad}")  # 2.0 (since dy/dw = x = 2)
print(f"dy/dx = {x.grad}")  # 3.0 (since dy/dx = w = 3)
print(f"dy/db = {b.grad}")  # 1.0 (since dy/db = 1)
```

### Computation Graph

```
Computation Graph for y = w*x² + b:

       x (2.0)
        │
        ▼ x²
       ┌───┐
       │ × │  x*x
       └─┬─┘
         │ 2*x = 4.0
         ▼
  w ──→ ⊗ ──→ ⊕ ──→ y = w*x² + b = 3*4 + 1 = 13.0
  (3.0)  │     ▲
         │     │
         │  b (1.0)
         │
  dy/dw = x² = 4.0
  dy/db = 1.0
  dy/dx = 2*w*x = 12.0
```

### Gradient Control

```python
# ════════════════════════════════════════════════
# Common patterns with autograd
# ════════════════════════════════════════════════

# 1. Disable gradient tracking (for inference)
x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    y = x * 2  # No gradient tracked — saves memory and time
    # y.backward()  → ERROR: no gradient

# OR use decorator
@torch.no_grad()
def inference(model, x):
    return model(x)

# 2. Zero gradients (MUST do before each backward pass)
optimizer.zero_grad()  # Resets all .grad to zero

# 3. Detach from computation graph
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.detach()  # z is a new tensor with no grad_fn
# z.backward()   → ERROR

# 4. Gradient accumulation (useful for large batches)
# Simulate batch size of 256 with actual batch of 64:
optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / 4  # Divide by accumulation steps
    loss.backward()                        # Accumulate gradients
    if (i + 1) % 4 == 0:                   # Every 4 mini-batches
        optimizer.step()                   # Update weights
        optimizer.zero_grad()              # Reset gradients
```

### Custom Gradients with `autograd.Function`

```python
class ReLUFunc(torch.autograd.Function):
    """Custom ReLU with manual forward and backward."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0  # Gradient is 0 for negative inputs
        return grad_input

# Usage
x = torch.randn(5, requires_grad=True)
y = ReLUFunc.apply(x)
y.sum().backward()
print(f"Gradients: {x.grad}")
```

---

## 6. Building Neural Networks with `nn.Module`

### The Module Pattern

```python
import torch
import torch.nn as nn

# ════════════════════════════════════════════════
# Method 1: nn.Sequential (simple linear chains)
# ════════════════════════════════════════════════
simple_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

# ════════════════════════════════════════════════
# Method 2: Custom nn.Module (recommended)
# ════════════════════════════════════════════════
class ImageClassifier(nn.Module):
    """Custom neural network for image classification."""
    
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        # Define layers as attributes
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Define the forward pass."""
        x = x.view(x.size(0), -1)       # Flatten: [B, C, H, W] → [B, C*H*W]
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)                  # No activation (CrossEntropy handles it)
        return x

model = ImageClassifier()
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### Common `nn` Layers

```python
# ════════════════════════════════════════════════
# Linear Layers
# ════════════════════════════════════════════════
nn.Linear(in_features, out_features, bias=True)  # Fully connected

# ════════════════════════════════════════════════
# Convolutional Layers
# ════════════════════════════════════════════════
nn.Conv1d(in_channels, out_channels, kernel_size)   # 1D (audio, time series)
nn.Conv2d(in_channels, out_channels, kernel_size)   # 2D (images)
nn.Conv3d(in_channels, out_channels, kernel_size)   # 3D (video, medical)

# ════════════════════════════════════════════════
# Pooling Layers
# ════════════════════════════════════════════════
nn.MaxPool2d(kernel_size=2)         # Max pooling (2×2)
nn.AvgPool2d(kernel_size=2)         # Average pooling
nn.AdaptiveAvgPool2d((1, 1))       # Global average pooling → [B, C, 1, 1]

# ════════════════════════════════════════════════
# Normalization
# ════════════════════════════════════════════════
nn.BatchNorm1d(num_features)        # For 1D data (after Linear)
nn.BatchNorm2d(num_features)        # For 2D data (after Conv2d)
nn.LayerNorm(normalized_shape)      # For sequences/Transformers
nn.GroupNorm(num_groups, num_channels)  # Alternative to BatchNorm

# ════════════════════════════════════════════════
# Recurrent Layers
# ════════════════════════════════════════════════
nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

# ════════════════════════════════════════════════
# Regularization
# ════════════════════════════════════════════════
nn.Dropout(p=0.5)                   # Randomly zero elements during training
nn.AlphaDropout(p=0.5)             # For SELU activation
nn.Dropout2d(p=0.5)                # For convolutional layers

# ════════════════════════════════════════════════
# Activation Functions
# ════════════════════════════════════════════════
nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(), nn.GELU(), nn.Softmax()
nn.Softmax(dim=1), nn.LogSoftmax(dim=1)

# ════════════════════════════════════════════════
# Embedding Layers
# ════════════════════════════════════════════════
nn.Embedding(num_embeddings=10000, embedding_dim=300)  # Word embeddings
nn.Embedding(num_embeddings=50000, embedding_dim=768)  # BERT-style
```

---

## 7. Datasets and DataLoaders

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ════════════════════════════════════════════════
# Custom Dataset
# ════════════════════════════════════════════════
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create dataset
import numpy as np
X = np.random.randn(1000, 20)
y = np.random.randint(0, 5, 1000)
dataset = MyDataset(X, y)

# DataLoader with batching, shuffling, and parallel loading
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,        # Shuffle each epoch
    num_workers=4,        # Parallel data loading
    pin_memory=True       # Faster CPU→GPU transfer
)

# Quick dataset from tensors
tensor_dataset = TensorDataset(
    torch.randn(1000, 20),
    torch.randint(0, 5, (1000,))
)
loader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

# Iterating through a DataLoader
for batch_features, batch_labels in train_loader:
    # batch_features shape: [32, 20]
    # batch_labels shape:   [32]
    predictions = model(batch_features.to(device))
    loss = criterion(predictions, batch_labels.to(device))
    # ... train step ...
```

---

## 8. Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# ════════════════════════════════════════════════
# Full Training Template
# ════════════════════════════════════════════════

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Train/validation split
train_dataset, val_dataset = random_split(full_dataset, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2)

# 3. Model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = MNISTNet().to(device)

# 4. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# 5. Training Loop
best_val_acc = 0.0
patience_counter = 0
max_patience = 7

for epoch in range(50):
    # ── Train ──
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += output.argmax(1).eq(target).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / len(train_dataset)
    
    # ── Validate ──
    model.eval()
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            val_correct += output.argmax(1).eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / len(val_dataset)
    
    # ── Learning rate scheduling ──
    scheduler.step(val_loss)
    
    # ── Early stopping ──
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ★ New best model saved! (Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    if patience_counter >= max_patience:
        print("Early stopping triggered!")
        break

# 6. Load best model and evaluate on test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        test_correct += model(data).argmax(1).eq(target).sum().item()

print(f"\nTest Accuracy: {100.0 * test_correct / len(test_dataset):.2f}%")
```

---

## 9. Saving and Loading Models

```python
# ════════════════════════════════════════════════
# Method 1: Save/Load state_dict (RECOMMENDED)
# ════════════════════════════════════════════════

# Save
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = MNISTNet()  # Create model with same architecture
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# ════════════════════════════════════════════════
# Method 2: Save entire model (less flexible)
# ════════════════════════════════════════════════
torch.save(model, 'full_model.pth')
model = torch.load('full_model.pth')

# ════════════════════════════════════════════════
# Save checkpoint (for resuming training)
# ════════════════════════════════════════════════
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
    'best_accuracy': best_val_acc
}
torch.save(checkpoint, 'checkpoint.pth')

# Resume
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 10. Exercises

### Exercise 1: Tensor Shapes
Create a tensor of shape `[4, 3, 224, 224]` representing a batch of 4 RGB images. Compute the mean and standard deviation across the batch dimension.

**Solution:**
```python
batch = torch.randn(4, 3, 224, 224)
batch_mean = batch.mean(dim=0)     # Shape: [3, 224, 224]
batch_std = batch.std(dim=0)       # Shape: [3, 224, 224]
per_image_mean = batch.mean(dim=[1, 2, 3])  # Shape: [4] — mean per image
```

### Exercise 2: Manual Gradient
Given $f(x, w) = \frac{1}{2}\|wx - 1\|^2$ with $w = 2.0$, $x = 3.0$, compute $\frac{\partial f}{\partial w}$ by hand and verify with PyTorch.

**Solution:**
$f = \frac{1}{2}(2 \times 3 - 1)^2 = \frac{1}{2}(5)^2 = 12.5$
$\frac{\partial f}{\partial w} = (wx - 1) \cdot x = 5 \times 3 = 15$

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
f = 0.5 * (w * x - 1) ** 2
f.backward()
print(f"∂f/∂w = {w.grad}")  # 15.0 ✓
```

### Exercise 3: Build a Custom Dataset
Create a PyTorch Dataset for the Iris dataset (from sklearn), split into train/test, and train a simple MLP.

**Solution:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class IrisDataset(Dataset):
    def __init__(self):
        data = load_iris()
        self.X = torch.FloatTensor(data.data)
        self.y = torch.LongTensor(data.target)
    
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

dataset = IrisDataset()
train_set, test_set = random_split(dataset, [120, 30])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

# Test accuracy
model.eval()
correct = sum(model(test_set[i][0]).argmax(0).item() == test_set[i][1].item()
              for i in range(len(test_set)))
print(f"Test accuracy: {100 * correct / len(test_set):.1f}%")
```

### Exercise 4: Implement BatchNorm from Scratch
Implement a simple forward pass of Batch Normalization.

**Solution:**
```python
def manual_batchnorm(x, gamma, beta, eps=1e-5):
    """
    x: [N, D] input
    gamma: [D] scale parameter
    beta: [D] shift parameter
    """
    mean = x.mean(dim=0)               # [D]
    var = x.var(dim=0, unbiased=False)  # [D]
    x_norm = (x - mean) / torch.sqrt(var + eps)  # Normalize
    return gamma * x_norm + beta               # Scale and shift

# Verify against nn.BatchNorm1d
x = torch.randn(32, 64)
bn = nn.BatchNorm1d(64)
bn_out = bn(x)

gamma = bn.weight
beta = bn.bias
manual_out = manual_batchnorm(x, gamma, beta)
print(f"Difference: {(bn_out - manual_out).abs().max().item():.8f}")  # Should be ~0
```

---

## 11. Summary

| Concept | Key PyTorch API |
|---------|----------------|
| Tensor creation | `torch.tensor()`, `zeros()`, `randn()`, `ones_like()` |
| Tensor ops | `+`, `*`, `@`, `matmul()`, `sum()`, `mean()` |
| Reshaping | `view()`, `reshape()`, `flatten()`, `permute()` |
| GPU | `.to('cuda')`, `.cuda()`, `.cpu()` |
| Autograd | `requires_grad=True`, `.backward()`, `torch.no_grad()` |
| Neural networks | `nn.Module`, `nn.Linear`, `nn.Conv2d`, `nn.Sequential` |
| Data loading | `Dataset`, `DataLoader`, `TensorDataset` |
| Training loop | `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` |
| Saving | `torch.save(state_dict)`, `torch.load(path)` |

PyTorch's design philosophy — **imperative, dynamic, Pythonic** — makes it the framework of choice for researchers and practitioners alike. Mastering tensors and autograd is the foundation for everything that follows.
