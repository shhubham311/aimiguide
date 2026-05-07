# CNN & Important Architectures (VGG16, ResNet, Inception)

## Why This Topic Matters

Convolutional Neural Networks (CNNs) are the backbone of **computer vision** — the field that enables self-driving cars, medical image analysis, facial recognition, and augmented reality. The key insight of CNNs is that **spatial structure matters**: pixels near each other are related, and the same visual patterns (edges, textures, shapes) can appear anywhere in an image.

CNNs revolutionized AI by winning the ImageNet competition in 2012 (AlexNet), reducing the top-5 error rate from 26% to 15% in a single year. The architectures that followed — VGG, GoogLeNet/Inception, ResNet, EfficientNet — represent some of the most impactful engineering achievements in deep learning. Understanding these architectures teaches you the design principles that power virtually all modern vision systems.

---

## 1. Why Not MLPs for Images?

### The Problem: Too Many Parameters, No Spatial Awareness

```
Image: 224 × 224 × 3 = 150,528 pixels

MLP with one hidden layer of 4096 neurons:
  Parameters = (150,528 × 4,096) + 4,096 = 616,960,256 ≈ 617 MILLION!

Problems:
  1. Massive memory and compute requirements
  2. No spatial invariance (learned weights tied to specific pixel positions)
  3. Prone to severe overfitting
  4. Ignores 2D structure of images

CNN Solution:
  Convolutional layer with 64 filters of size 3×3×3:
  Parameters = (3 × 3 × 3) × 64 + 64 = 1,792 ≈ 1.8 THOUSAND!

  Benefits:
  ✓ Shared weights (same filter applied everywhere)
  ✓ Local connectivity (each neuron sees only a small patch)
  ✓ Spatial invariance (pattern detected regardless of position)
  ✓ Translation equivariance (shift input → shift output)
```

---

## 2. Core CNN Operations

### 2.1 Convolution (Conv2D)

A convolution operation slides a filter (kernel) across the input image, computing dot products at each position:

$$\text{Output}(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Input}(i+m, j+n) \cdot \text{Kernel}(m, n) + b$$

Where:
- $\text{Input}$ is the input feature map
- $\text{Kernel}$ is the filter of size $k \times k$
- $b$ is the bias
- $\text{Output}(i,j)$ is the value at position $(i,j)$

```
Example: 5×5 input, 3×3 kernel (no padding, stride=1)

Input:           Kernel:           Output (Feature Map):
[1  1  1  0  0]  [1  0  1]         [4  3  4]
[0  1  1  1  0]  [0  1  0]         [2  4  3]
[0  0  1  1  1]  [1  0  1]         [2  3  4]
[1  1  0  1  0]  
[1  0  1  0  1]  Output size: (5-3)/1 + 1 = 3×3

Computation for output[0,0]:
= 1(1) + 1(0) + 1(1) + 0(0) + 1(1) + 1(0) + 0(1) + 0(0) + 1(1)
= 1 + 0 + 1 + 0 + 1 + 0 + 0 + 0 + 1 = 4
```

### Output Size Formula

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} \right\rfloor + 1$$

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2P - K}{S} \right\rfloor + 1$$

Where:
- $H_{\text{in}}, W_{\text{in}}$ = input height/width
- $K$ = kernel size
- $S$ = stride (step size)
- $P$ = padding (zeros added to borders)

```
Examples (input 32×32):

K=3, S=1, P=0:  (32-3)/1 + 1 = 30×30
K=3, S=1, P=1:  (32+2-3)/1 + 1 = 32×32 (same padding)
K=5, S=2, P=2:  (32+4-5)/2 + 1 = 17×17
K=3, S=2, P=1:  (32+2-3)/2 + 1 = 16×16
```

### 2.2 What Do Convolutional Layers Learn?

```
Layer 1 (Shallow):          Layer 2:               Layer 3 (Deep):
╱╲ ─ ─ ╱╲ ─ ╱╲ ─ ─ ╱╲     ╱╲╱╲ ─ ─ ┌──┐ ─ ╱╲   ┌─────┐  ╱╲  ◯
Edges & Corners              Textures               Parts & Patterns
(detect edges in any        (combinations of       (eye, wheel, car
 direction, at any          edges: stripes,         body, face parts)
 position)                  grids, dots)

Multiple filters per layer detect different features simultaneously:

Input Image (3×H×W)   Conv Layer (64 filters)    Output (64×H×W)
     │                        │                        │
  RGB image           64 different 3×3 kernels     64 feature maps
     │                   ╱╲ ─ ╱╲ ─ ╱╲                  │
     │                   │  │  │  │  │                  │
     └──────────────────→ Conv2D ─────────────────→ 64 channels
                           │
                     Each filter specializes:
                     Filter 1 → horizontal edges
                     Filter 2 → vertical edges
                     Filter 3 → diagonal lines
                     Filter 4 → color blobs
                     ... etc
```

### 2.3 Pooling

Pooling reduces spatial dimensions while retaining important information:

```
Max Pooling (2×2, stride=2):

Input:               Output:
[1  3  2  4]         [3  4]
[5  6  7  8]  ──→    [8  7]
[9  8  7  6]         (takes max of each 2×2 block)
[3  2  1  0]

Input: 32×32 → Output: 16×16 (50% reduction)
Input: 224×224 → After 5 pool layers: 224 → 112 → 56 → 28 → 14 → 7

Average Pooling: Takes mean instead of max
Global Average Pooling: Reduces entire feature map to single number per channel
```

### 2.4 A Complete CNN Architecture

```
Input Image
[224 × 224 × 3]
      │
      ▼
┌─────────────────────┐
│ Conv2D: 32 filters   │   Learn low-level features (edges, corners)
│ Kernel: 3×3          │   Output: [224 × 224 × 32]
│ Padding: 1           │
│ Activation: ReLU     │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ MaxPool: 2×2         │   Spatial reduction
│                     │   Output: [112 × 112 × 32]
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Conv2D: 64 filters   │   Learn mid-level features (textures, patterns)
│ Kernel: 3×3          │   Output: [112 × 112 × 64]
│ Padding: 1           │
│ Activation: ReLU     │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ MaxPool: 2×2         │   Output: [56 × 56 × 64]
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Conv2D: 128 filters  │   Learn high-level features (parts, objects)
│ Kernel: 3×3          │   Output: [56 × 56 × 128]
│ Activation: ReLU     │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ MaxPool: 2×2         │   Output: [28 × 28 × 128]
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Flatten             │   Output: [100,352]
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Fully Connected     │   Output: [512]
│ Activation: ReLU    │
│ Dropout: 0.5        │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Fully Connected     │   Output: [num_classes]
│ Activation: Softmax │
└─────────────────────┘
```

---

## 3. PyTorch CNN Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ════════════════════════════════════════════════
# Basic CNN for CIFAR-10
# ════════════════════════════════════════════════
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 32×32→32×32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16×16→16×16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 8×8→8×8
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)   # Halves spatial dimensions
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # Conv Block 1: [B, 3, 32, 32] → [B, 32, 16, 16]
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        
        # Conv Block 2: [B, 32, 16, 16] → [B, 64, 8, 8]
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        
        # Conv Block 3: [B, 64, 8, 8] → [B, 128, 4, 4]
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        
        # Flatten and classify
        x = x.view(x.size(0), -1)           # [B, 128*4*4] = [B, 2048]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)                       # [B, num_classes]
        return x

# Data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BasicCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += model(images).argmax(1).eq(labels).sum().item()
    print(f"Epoch {epoch+1}: Acc = {100*correct/len(test_dataset):.2f}%")
```

---

## 4. VGG16 Architecture

### Design Philosophy

VGGNet (Simonyan & Zisserman, 2014) proved that **depth** and **small filters** are the key to good performance. The entire network uses only **3×3 convolutions** stacked on top of each other.

### Key Insight: Two 3×3 Convs ≈ One 5×5 Conv

```
Receptive field equivalence:

One 5×5 conv:              Two 3×3 convs:
┌─────────────────────┐    ┌─────────────┐┌─────────────┐
│  5 × 5              │    │  3 × 3      ││  3 × 3      │
│  Receptive field: 5  │    │  RF: 3      ││  RF: 5      │
│  Parameters: 25×C²   │    │  2×9×C²=18C² │              │
│  Non-linearity: 1    │    │  2 ReLU     │              │
└─────────────────────┘    └─────────────┘└─────────────┘

Benefits of two 3×3:
  ✓ Fewer parameters (18 vs 25 per channel pair)
  ✓ Two non-linear activations (more expressive)
  ✓ Less computation overall
```

### VGG16 Architecture

```
VGG16 Architecture (13 Conv + 3 FC layers):

Input: [224 × 224 × 3]
  │
  ├── Block 1: Conv3-64 × 2  →  Pool    → [112 × 112 × 64]
  ├── Block 2: Conv3-128 × 2  →  Pool    → [56 × 56 × 128]
  ├── Block 3: Conv3-256 × 3  →  Pool    → [28 × 28 × 256]
  ├── Block 4: Conv3-512 × 3  →  Pool    → [14 × 14 × 512]
  ├── Block 5: Conv3-512 × 3  →  Pool    → [7 × 7 × 512]
  │
  ├── Flatten → [25088]
  ├── FC-4096 → ReLU → Dropout(0.5)
  ├── FC-4096 → ReLU → Dropout(0.5)
  └── FC-1000 → Softmax

Total Parameters: ~138 million
```

### PyTorch VGG16

```python
import torchvision.models as models

# Load pre-trained VGG16
vgg16 = models.vgg16(pretrained=True)    # Or: weights=models.VGG16_Weights.DEFAULT
print(vgg16)

# Modify for CIFAR-10 (10 classes instead of 1000)
vgg16.classifier[6] = nn.Linear(4096, 10)

# Freeze convolutional layers (for transfer learning)
for param in vgg16.features.parameters():
    param.requires_grad = False

# Custom VGG-style architecture
class VGGStyle(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

---

## 5. ResNet (Residual Networks)

### The Problem: Vanishing Gradients in Deep Networks

```
Training Error vs. Network Depth:

Error │  ↑  (deeper = worse!)
      │  │╲  56-layer
      │  │  ╲
      │  │   ╲         ← When depth increases WITHOUT
      │  │    ╲           residual connections,
      │  │     ╲          error actually gets worse
      │  │  ╱───╲──── 20-layer
      │  │ ╱
      │  │╱
      └──────────────── Depth

Problem: Gradients vanish as they pass through many layers.
Solution: Shortcut/skip connections!
```

### The Residual Block

The key innovation: instead of learning $\mathcal{F}(x) = \mathcal{H}(x)$, learn the **residual** $\mathcal{F}(x) = \mathcal{H}(x) - x$:

$$\mathcal{H}(x) = \mathcal{F}(x) + x$$

If the optimal transformation is close to identity ($\mathcal{H}(x) \approx x$), it's easier to push $\mathcal{F}(x) \approx 0$ than to learn the identity from scratch.

```
Residual Block:

        x ─────────────────────────────┐
        │                              │
        ▼                              │
   ┌─────────┐                         │
   │ Conv2D  │                         │
   │ 3×3     │                         │
   └────┬────┘                         │
        │                              │
        ▼                              │
   ┌─────────┐                         │
   │ BatchNorm│                        │
   └────┬────┘                         │
        │                              │
        ▼                              │
   ┌─────────┐                         │
   │  ReLU   │                         │
   └────┬────┘                         │
        │                              │
        ▼                              │
   ┌─────────┐                         │
   │ Conv2D  │                         │
   │ 3×3     │                         │
   └────┬────┘                         │
        │                              │
        ▼                              │
   ┌─────────┐                         │
   │ BatchNorm│                        │
   └────┬────┘                         │
        │                              │
        ▼                              ▼
        └──────────── ⊕ ← Addition  ──┘
                     │
                     ▼
                ┌─────────┐
                │  ReLU   │
                └────┬────┘
                     │
                     ▼
                  Output

F(x) = Conv → BN → ReLU → Conv → BN
Output = ReLU(F(x) + x)    ← Skip connection (identity shortcut)
```

### Gradient Flow in ResNet

```
Without skip connection:         With skip connection:
L → ∂L/∂x → (W₃)(W₂)(W₁)       L → ∂L/∂x → (W₃)(W₂)(W₁) + 1

  Gradients: W₁W₂W₃              Gradients: W₁W₂W₃ + 1
  Can vanish if W's are small    Always has at least 1 (identity path)
  
  "Highway for gradients!"
```

### ResNet Architecture Family

```
ResNet Variants:

ResNet-18:   [BasicBlock × 2,2,2,2]   = 18 layers
ResNet-34:   [BasicBlock × 3,4,6,3]   = 34 layers
ResNet-50:   [Bottleneck × 3,4,6,3]  = 50 layers
ResNet-101:  [Bottleneck × 3,4,23,3] = 101 layers
ResNet-152:  [Bottleneck × 3,8,36,3] = 152 layers

Bottleneck Block (used in ResNet-50+):
  1×1 Conv (reduce dim) → 3×3 Conv → 1×1 Conv (restore dim)
  [256] → 64 → 64 → [256]
  
  Saves computation: 3×3 conv operates on fewer channels
```

### PyTorch ResNet

```python
import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet-50
resnet = models.resnet50(pretrained=True)

# Transfer learning: modify for your dataset
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Freeze all layers except the final FC
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc.requires_grad_(True)  # Only train the last layer

# Custom Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (match dimensions if needed)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual    # ← The key skip connection!
        return torch.relu(out)
```

---

## 6. Inception (GoogLeNet)

### The Motivation: Multi-Scale Features

Different objects have different sizes. Inception solves this by applying **multiple filter sizes in parallel**:

```
Inception Module:

Input Feature Map
      │
      ├── 1×1 Conv ──────────────────────────────→ Concat
      │                                            │
      ├── 1×1 Conv → 3×3 Conv ────────────────────→ │
      │                                            │
      ├── 1×1 Conv → 5×5 Conv ────────────────────→ │
      │                                            │
      ├── 3×3 MaxPool → 1×1 Conv ─────────────────→ │
      │                                            │
      ▼                                            ▼
  (dimensionality     ┌──────────────────────────────┐
   reduction          │  Concatenated Feature Maps    │
   via 1×1 conv)      │  (filters from all branches)  │
                      └──────────────────────────────┘

Key idea: 1×1 convolutions reduce dimensionality BEFORE expensive 3×3/5×5 convolutions
```

### 1×1 Convolutions: Dimensionality Reduction

```
Without 1×1 reduction:
  256 channels → 3×3 Conv with 256 filters
  Parameters = 256 × 3 × 3 × 256 = 590,496

With 1×1 reduction:
  256 → 1×1 Conv (64 filters) → 3×3 Conv (64 filters) → 1×1 Conv (256)
  = 256×1×1×64 + 64×3×3×64 + 64×1×1×256
  = 16,384 + 36,864 + 16,384 = 69,632
  
Reduction: 590,496 → 69,632 (8.5× fewer parameters!)
```

---

## 7. Architecture Comparison

```
Architecture Timeline & Key Stats:

Year │ Architecture │ Depth │ Params    │ Top-5 Error │ Key Innovation
─────┼─────────────┼───────┼───────────┼─────────────┼──────────────────
2012 │ AlexNet     │   8   │  60M      │  15.3%      │ ReLU, Dropout, GPU
2014 │ VGG16       │  16   │  138M     │  7.3%       │ Small 3×3 filters
2014 │ GoogLeNet   │  22   │  6.8M     │  6.7%       │ Inception modules
2015 │ ResNet-152  │ 152   │  60M      │  3.6%       │ Skip connections
2016 │ Inception-v3│  48   │  24M      │  3.5%       │ Factorized convs
2017 │ ResNeXt    │  101  │  25M      │  3.0%       │ Grouped convolutions
2019 │ EfficientNet│  66   │  5.3M     │  2.9%       │ Compound scaling
2020 │ ViT (huge) │  32   │  632M     │  2.4%       │ Vision Transformer
```

```
Error Rate Progression:

15% ┤ ● AlexNet
    │     ╲
10% ┤      ╲
    │       ╱● VGG16
 8% ┤      ╱╱ ● GoogLeNet
    │    ╱╱
 5% ┤   ╱╱ ● ResNet
    │  ╱╱  ╲
 3% ┤ ╱╱     ● EfficientNet / ViT
    │╱
 2% ┤● Human baseline (~5%)
    └────────────────────────── Year
     2012 2014 2015 2019 2022
```

---

## 8. Transfer Learning in Practice

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ════════════════════════════════════════════════
# Transfer Learning Pipeline
# ════════════════════════════════════════════════

# 1. Data transforms (ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. Load pre-trained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 3. Strategy A: Feature Extraction (freeze all, train only classifier)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 5)  # 5 classes
)

# Strategy B: Fine-tuning (unfreeze later layers)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 4. Train with small learning rate for fine-tuning
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
```

---

## 9. Exercises

### Exercise 1: Calculate Convolution Output Size
Input: $128 \times 128 \times 3$, Conv with 64 filters, kernel $5 \times 5$, stride $2$, padding $2$. What is the output size?

**Solution:**
$$H_{\text{out}} = \left\lfloor \frac{128 + 2(2) - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{127}{2} \right\rfloor + 1 = 63 + 1 = 64$$
Output: $64 \times 64 \times 64$

### Exercise 2: Parameter Counting
Count parameters in: Conv2d(in=3, out=64, kernel=3×3) + Conv2d(in=64, out=128, kernel=3×3) + Linear(128×8×8, 10).

**Solution:**
- Conv1: $3 \times 3 \times 3 \times 64 + 64 = 1,792$
- Conv2: $64 \times 3 \times 3 \times 128 + 128 = 73,856$
- FC: $128 \times 8 \times 8 \times 10 + 10 = 81,930$
- **Total: $1,792 + 73,856 + 81,930 = 157,578$**

### Exercise 3: Implement a Residual Block from Scratch
Build and test a residual block with a shortcut connection.

**Solution:**
```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return torch.relu(self.block(x) + x)

# Test: gradient flows through the shortcut
x = torch.randn(2, 64, 32, 32, requires_grad=True)
block = ResBlock(64)
y = block(x)
y.sum().backward()
print(f"Gradient norm: {x.grad.norm().item():.4f}")  # Should be non-zero!
```

### Exercise 4: Compare VGG16 vs ResNet18 Training
Train both on CIFAR-10 (adapted for 32×32 input) and compare accuracy and training time.

**Solution:**
```python
import torchvision.models as models
import time

# VGG16 adapted for CIFAR-10
vgg = models.vgg16()
vgg.features[0] = nn.Conv2d(3, 64, 3, padding=1)  # Replace 7×7 with 3×3
vgg.classifier[6] = nn.Linear(4096, 10)

# ResNet18 adapted for CIFAR-10
resnet = models.resnet18()
resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
resnet.fc = nn.Linear(512, 10)

# Compare (conceptual):
# VGG16: ~138M params, slower, high memory
# ResNet18: ~11M params, faster, skip connections help convergence
# ResNet18 typically achieves better accuracy with fewer parameters
```

---

## 10. Summary

| Architecture | Key Innovation | Parameters | Use Case |
|-------------|---------------|------------|----------|
| **VGG16** | Stacked 3×3 convolutions | 138M | Feature extraction, baseline |
| **ResNet** | Skip (residual) connections | 25M (ResNet-50) | General purpose, transfer learning |
| **Inception** | Multi-scale parallel filters | 6.8M | Efficient multi-scale feature extraction |
| **EfficientNet** | Compound scaling (depth, width, resolution) | 5.3M | Mobile/embedded, efficient inference |

**Design Principles Learned:**
1. Start with simple architectures, add complexity as needed
2. Batch normalization and residual connections are almost always beneficial
3. Pre-trained models (transfer learning) outperform training from scratch on most tasks
4. The choice of architecture matters less than data quality and training methodology
