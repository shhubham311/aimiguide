# GANs & StyleGAN: Generative Adversarial Networks

## Why GANs Matter

Generative Adversarial Networks (GANs), introduced by Ian Goodfellow in 2014, represent one of the most creative ideas in modern machine learning. Instead of training a single model to generate data, GANs pit **two neural networks against each other** in a competitive game: a Generator that creates fake data and a Discriminator that tries to detect it. Through this adversarial process, the Generator becomes increasingly skilled at producing realistic outputs.

GANs have been used to generate photorealistic faces, art, music, video, and data augmentation for medical imaging. **StyleGAN**, developed by NVIDIA, pushed this further by enabling unprecedented control over generated images.

---

## The GAN Framework

### Core Concept

```
                      ┌─────────────────┐
                      │   Real Data     │
                      │  (training set)  │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐      ┌──────────────────┐
                      │                 │      │                  │
                      │  Discriminator  │◄────►│   Generator      │
                      │  D(x)           │      │   G(z)           │
                      │                 │      │                  │
                      └────────┬────────┘      └────────┬─────────┘
                               │                        │
                               ▼                        │
                        Real or Fake?                   │
                        (binary classification)         │
                                                        │
                      Random Noise z ◄──────────────────┘
                      (latent vector)
                        ┌───────────────┐
                        │ z ~ N(0, I)   │
                        │ e.g., dim=100 │
                        └───────────────┘
```

### The Min-Max Objective

GANs solve the following minimax game:

```
min_G max_D  V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

Where:
  D(x)         = Probability that x is real (scalar in [0, 1])
  G(z)         = Generated data from noise z
  p_data       = Real data distribution
  p_z          = Prior distribution (usually Gaussian)
  E[·]         = Expected value
```

**Intuition:**
- **Discriminator** maximizes V: correctly classifies real as 1, fake as 0
- **Generator** minimizes V: produces fakes that fool the discriminator

### Optimal Solution

At Nash equilibrium: **D\*(x) = 0.5 everywhere**, meaning the discriminator can't distinguish real from fake, and **p_g = p_data**, meaning the generator perfectly matches the real data distribution.

```
Training Progression:

Epoch 1:    D(real)=0.99, D(fake)=0.01  → Discriminator dominates
Epoch 50:   D(real)=0.85, D(fake)=0.30  → Generator improving
Epoch 200:  D(real)=0.52, D(fake)=0.48  → Near Nash equilibrium
Epoch 500:  D(real)=0.50, D(fake)=0.50  → Perfect! Generator wins
```

---

## GAN Architecture Details

### Generator (Transposed Convolution)

```
Noise z (100-dim)
    │
    ▼
┌──────────┐
│ Dense    │ → Reshape to (4, 4, 256)
│ 100→1024 │
└────┬─────┘
     │
     ▼
┌──────────┐
│ ConvT    │ → (8, 8, 128)      Transposed Convolution:
│ 4×4→8×8  │                       ↑ Upscales spatial resolution
└────┬─────┘                       ↓ Reduces channel depth
     │
     ▼
┌──────────┐
│ ConvT    │ → (16, 16, 64)
│ 8×8→16×16│
└────┬─────┘
     │
     ▼
┌──────────┐
│ ConvT    │ → (32, 32, 3)
│16×16→32×32│  Output image (32×32×3)
└────┬─────┘
     │
     ▼
  tanh → [-1, 1]
```

### Discriminator (Standard Convolution)

```
Image (32×32×3)
    │
    ▼
┌──────────┐
│ Conv2D   │ → (16, 16, 64)
│ + LeakyReLU│
└────┬─────┘
     │
     ▼
┌──────────┐
│ Conv2D   │ → (8, 8, 128)
│ + LeakyReLU│
└────┬─────┘
     │
     ▼
┌──────────┐
│ Conv2D   │ → (4, 4, 256)
│ + LeakyReLU│
└────┬─────┘
     │
     ▼
┌──────────┐
│ Flatten  │ → (4096,)
│ + Dense  │ → (1,) → sigmoid
└──────────┘
    │
    ▼
Real/Fake probability
```

---

## Python Implementation: DCGAN

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Hyperparameters
latent_dim = 100
lr = 0.0002
beta1 = 0.5
batch_size = 128
epochs = 50

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                      download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: (256, 7, 7)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: (128, 14, 14)
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (1, 28, 28)
        )

    def forward(self, z):
        return self.net(z.view(-1, latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (1, 28, 28)
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (128, 14, 14)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (256, 7, 7)
            nn.Conv2d(256, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: (1, 1, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # ── Train Discriminator ──
        optimizer_D.zero_grad()

        # Real loss
        output_real = D(real_imgs)
        loss_D_real = criterion(output_real, real_labels)

        # Fake loss
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)
        output_fake = D(fake_imgs.detach())
        loss_D_fake = criterion(output_fake, fake_labels)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()

        # ── Train Generator ──
        optimizer_G.zero_grad()

        output = D(fake_imgs)
        loss_G = criterion(output, real_labels)  # Generator wants D to think fake is real
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss_D: {loss_D:.4f}  Loss_G: {loss_G:.4f}")
```

---

## StyleGAN: Style-Based Generator

StyleGAN (Karras et al., 2019) replaces the traditional latent→image pipeline with a **style injection mechanism** inspired by style transfer.

### Architecture

```
    Latent z (512-dim)
         │
         ▼
    ┌───────────┐
    │ Mapping   │      8 fully-connected layers
    │ Network f │──────► w ∈ W space (learned)
    └───────────┘         (disentangled latent space)
         │
         │   w (style vector)
         │   │
         │   ▼
         │  ┌──────────┐
         │  │   AdaIN  │──────► Modulates each convolution layer
         │  │  y = γ·  │
         │  │    (x-μ) │       γ, β derived from w via A(w)
         │  │    + β   │
         │  └──────────┘
         │
         ▼
    Constant Input (4×4×512)
         │
    Synthesis Network g
    ┌─────────────────────────────────┐
    │  Resolution: 4×4 → 8×8 → ...   │
    │  Each layer:                    │
    │  Conv → AdaIN(style) → Noise    │
    │         ↑                  ↑    │
    │       w vector         per-pixel│
    │                         noise   │
    └─────────────────────────────────┘
         │
         ▼
    High-resolution image (1024×1024)
```

### AdaIN: Adaptive Instance Normalization

```
AdaIN(x_i, y) = y_s,i × (x_i - μ_i) / σ_i + y_b,i

Where:
  x_i       = Feature map at layer i
  y_s,i, y_b,i = Scale (γ) and bias (β) from style vector w
  μ_i, σ_i  = Mean and std of feature map
```

### Style Mixing

StyleGAN enables **style mixing**—using different style vectors w at different layers:

```
w1 (coarse styles:    pose, face shape)   → injected at early layers
w2 (fine styles:      hair color, eyes)   → injected at middle layers
w3 (detail styles:    pores, hair strands)→ injected at late layers

Result: A face with w1's structure, w2's coloring, and w3's details
```

---

## GAN Training Challenges & Solutions

### Mode Collapse
The Generator produces limited variety—all outputs look similar.

**Solution: Wasserstein GAN (WGAN):**
```
Replace BCE loss with Earth Mover's Distance:
Loss_D = E[D(x_fake)] - E[D(x_real)]     (no sigmoid!)
Loss_G = -E[D(x_fake)]

With gradient penalty (WGAN-GP):
GP = λ × E[(||∇_x̂ D(x̂)||_2 - 1)²]
```

### Vanishing Gradients
Generator receives no useful signal when discriminator is too strong.

**Solution: Feature matching loss:**
```
Loss_G = ||E[D_features(x_real)] - E[D_features(G(z))]]||²
```

---

## Real-World Applications

| Application | GAN Variant | Impact |
|---|---|---|
| **Face Generation** | StyleGAN2/3 | ThisPersonDoesNotExist.com |
| **Data Augmentation** | Conditional GAN | Medical imaging (rare disease) |
| **Image Super-Resolution** | SRGAN, ESRGAN | 4× image upscaling |
| **Image-to-Image** | pix2pix, CycleGAN | Sketch→Photo, Day→Night |
| **Video Generation** | DVD-GAN, StyleGAN-V | Deepfake detection research |
| **Drug Discovery** | MolGAN | Novel molecule generation |

---

## Exercises

### Exercise 1: Compute Generator Loss
Given D(G(z)) = 0.3, compute the binary cross-entropy generator loss.

**Solution:**
```
Loss_G = -log(D(G(z))) = -log(0.3) = -(-1.204) = 1.204

The generator wants to maximize D(G(z)), so lower loss is better.
If D(G(z)) = 0.5 (confused): Loss = -log(0.5) = 0.693
If D(G(z)) = 0.99 (fooled):  Loss = -log(0.99) = 0.01
```

### Exercise 2: AdaIN Computation
Given feature x = [2, 4, 6, 8], style scale γ = 1.5, bias β = 0.5, compute AdaIN output.

**Solution:**
```
μ = (2+4+6+8)/4 = 5.0
σ = sqrt(((2-5)² + (4-5)² + (6-5)² + (8-5)²)/4) = sqrt((9+1+1+9)/4) = sqrt(5) ≈ 2.236

AdaIN = 1.5 × (x - 5) / 2.236 + 0.5

= 1.5 × [-3/2.236, -1/2.236, 1/2.236, 3/2.236] + 0.5
= 1.5 × [-1.342, -0.447, 0.447, 1.342] + 0.5
= [-2.013, -0.671, 0.671, 2.013] + 0.5
= [-1.513, -0.171, 1.171, 2.513]
```

### Exercise 3: Design a Conditional GAN
How would you modify a GAN to generate images of specific digits (0-9)?

**Solution:** Use a **Conditional GAN (CGAN)**:
1. Concatenate the digit label to the noise vector: [z, y] → Generator input
2. Concatenate the digit label to the real/fake image: [x, y] → Discriminator input
3. The label can be a one-hot vector (10-dim) appended as additional channels
4. This conditions generation on the desired class

---

## Key Takeaways

1. **GANs learn by competition**—Generator vs. Discriminator in a minimax game
2. **Nash equilibrium** occurs when the discriminator can't distinguish real from fake
3. **StyleGAN** introduces style injection via AdaIN for fine-grained control
4. **WGAN + gradient penalty** addresses training instability and mode collapse
5. **GANs generate high-quality images** but evaluation remains challenging (FID, IS scores)
