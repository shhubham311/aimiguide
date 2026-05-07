# Diffusion Models: DALL-E, Stable Diffusion & The Future of Image Generation

## Why Diffusion Models Matter

In 2020-2022, diffusion models overtook GANs as the state-of-the-art in image generation. They power **DALL-E 2/3, Stable Diffusion, Midjourney, and Imagen**—systems that generate stunning, photorealistic images from text descriptions. Unlike GANs, diffusion models have a solid theoretical foundation in probability, train stably without adversarial dynamics, and produce diverse, high-quality outputs.

Understanding diffusion models is crucial because they now extend beyond images to **video generation (Sora), audio (AudioLDM), 3D asset creation, protein design, and molecular generation**.

---

## The Forward Process: Adding Noise

Diffusion models work by learning to **reverse a gradual noising process**. In the forward process, we slowly corrupt data by adding Gaussian noise over T timesteps:

```
Clean Image x₀  ──►  x₁  ──►  x₂  ──►  ...  ──►  x_T ≈ Pure Noise

     (T=0)       (noise)  (noise)              (T=1000)
     Perfect     slightly  more                 almost
     image       corrupted  corrupted            random
```

### Mathematical Formulation

At each timestep t, we add a small amount of Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)

Where:
  β_t  = Noise schedule (small positive constant at step t)
  N    = Normal (Gaussian) distribution
  I    = Identity matrix
```

The key insight: we can jump directly from x₀ to any x_t in a single step:

```
x_t = √ᾱ_t · x₀ + √(1 - ᾱ_t) · ε

Where:
  ᾱ_t = ∏_{s=1}^{t} (1 - β_s)     (cumulative product of signal retention)
  ε   ~ N(0, I)                      (standard Gaussian noise)
```

### Noise Schedule

```
β_t values typically range from small to large:

Linear schedule:   β₁=0.0001 → β_T=0.02  (T=1000)
Cosine schedule:   ᾱ_t = f(t) using cosine interpolation (smoother degradation)

Signal remaining (ᾱ_t):
  t=0:    ᾱ = 1.000    (100% signal, 0% noise)
  t=250:  ᾱ = 0.580    (58% signal, 42% noise)
  t=500:  ᾱ = 0.185    (18% signal, 82% noise)
  t=750:  ᾱ = 0.020    (2% signal, 98% noise)
  t=1000: ᾱ ≈ 0.000    (≈0% signal, ≈100% noise)
```

---

## The Reverse Process: Denoising

The model learns to **reverse each noising step** by predicting the noise that was added:

```
Pure Noise x_T  ──►  x_{T-1}  ──►  ...  ──►  x₁  ──►  x₀
     (T=1000)   (denoise)     (denoise)     (denoise)    Clean Image
```

The reverse process is parameterized by a neural network:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² · I)

Where μ_θ is predicted by the neural network.
```

### Simplified: Predict Noise Instead of Mean

Instead of predicting the mean μ_θ directly, the model predicts the **noise ε** that was added:

```
Loss = E_{t, x₀, ε} [ || ε - ε_θ(x_t, t) ||² ]

Where:
  t    ~ Uniform(1, T)           (random timestep)
  x₀   ~ q(x₀)                   (real data)
  ε    ~ N(0, I)                  (noise we added)
  x_t  = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε   (noisy input to model)
  ε_θ  = Model prediction of ε
```

This is just **MSE between predicted and actual noise**!

---

## Architecture Overview

### U-Net: The Denoising Network

```
                    U-Net Architecture
                    ─────────────────

    x_t (noisy image) + t (timestep embedding)
              │
    ┌─────────▼─────────┐
    │  Encoder (Down)   │
    │                   │
    │  Conv 64×64  ↓    │     ┌──────────────────────┐
    │  Conv 32×32  ↓    │     │ Timestep Embedding   │
    │  Conv 16×16  ↓    │     │ t → MLP → embeddings │
    │  Conv 8×8   ↓    │     └──────────┬───────────┘
    │                   │                │ (added to each)
    └─────────┬─────────┘                │
              │                          │
    ┌─────────▼─────────┐                │
    │   Bottleneck      │◄───────────────┘
    │   (8×8, 512 ch)   │
    │   + Self-Attention│
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  Decoder (Up)     │
    │                   │
    │  Conv 16×16  ↑  ◄── skip connection
    │  Conv 32×32  ↑  ◄── skip connection
    │  Conv 64×64  ↑  ◄── skip connection
    │  Conv 128×128 ↑   │
    │                   │
    └─────────┬─────────┘
              │
              ▼
        Predicted Noise ε_θ(x_t, t)
```

### Self-Attention Layers

Attention is applied at lower resolutions (16×16, 8×8) to capture global structure:

```
At 16×16 resolution:
  ┌────────────────────────────┐
  │   Spatial Self-Attention   │
  │   Each spatial position    │
  │   attends to every other   │
  │   position                 │
  │                            │
  │   Q, K, V from conv features│
  │   Multi-head (8 heads)     │
  └────────────────────────────┘
```

---

## Stable Diffusion: Latent Diffusion

Generating images in pixel space is extremely expensive (1024×1024×3 = 3M dimensions). Stable Diffusion (Rombach et al., 2022) performs diffusion in a **compressed latent space**:

```
┌─────────────────────────────────────────────────────────┐
│                   Stable Diffusion Pipeline              │
│                                                          │
│  Text Prompt ──► CLIP Text Encoder ──► Text Embedding   │
│                                                │         │
│                                                ▼         │
│                                          ┌──────────┐   │
│  Image ──► VAE Encoder ──► Latent z ──► │Diffusion │──► Denoised
│                                    ▲     │  U-Net   │    Latent z₀
│                                    │     │(cross-   │         │
│                                    │     │ attention│         ▼
│                                    │     │ w/ text) │   VAE Decoder ─► Image
│                                    │     └──────────┘               │
│                                    │                                │
│                              Random Noise                        Output
└─────────────────────────────────────────────────────────┘

Key insight: Latent space is 48×48×4 = 9,216 dims vs 512×512×3 = 786,432 dims
→ 85× reduction in computation!
```

### Cross-Attention for Text Conditioning

The text embedding guides image generation through **cross-attention** in the U-Net:

```
Cross-Attention:
  Q = Spatial features from U-Net (image representation)
  K = Text embedding from CLIP
  V = Text embedding from CLIP

  Attention(Q, K, V) = softmax(QK^T / √d_k) × V

  → Each spatial location attends to relevant text tokens
  → "Red car" → attention weights focused on "red" and "car" tokens
```

---

## DALL-E: Text-Conditioned Generation

DALL-E (Ramesh et al., 2021) combines a discrete VAE with an autoregressive transformer:

```
Text: "A hedgehog wearing a tiny hat"

Step 1: Encode text with CLIP → text tokens [t₁, t₂, ..., tₙ]

Step 2: Generate image tokens autoregressively:
  P(z₁ | t₁..tₙ)
  P(z₂ | z₁, t₁..tₙ)
  P(z₃ | z₁, z₂, t₁..tₙ)
  ...
  P(z_{256} | z₁..z_{255}, t₁..tₙ)   (32×32 = 1024 image tokens)

Step 3: Decode image tokens → 256×256 image via dVAE decoder
```

---

## Classifier-Free Guidance

The key technique for high-quality text-to-image generation:

```
┌────────────────────────────────────────────┐
│   Classifier-Free Guidance                 │
│                                             │
│   ε_guided = ε_uncond + s × (ε_cond - ε_uncond)│
│                                             │
│   Where:                                    │
│   ε_uncond = Noise prediction WITHOUT text  │
│   ε_cond   = Noise prediction WITH text     │
│   s        = Guidance scale (typically 7.5) │
└────────────────────────────────────────────┘

Effect of guidance scale s:
  s = 1.0: Normal (balanced)
  s = 7.5: High quality (default for SD)
  s = 15:  Very specific but less diverse
  s = 0.0: Ignores text prompt entirely

          Quality ↑   Diversity ↑
              ↑
              │        s=1
              │       /
              │      /
              │     /
              │    s=7.5 ← sweet spot
              │   /
              │  /
              │ s=15
              └──────────────────►
```

---

## Python Implementation: Simple Diffusion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to x_0 at timestep t.
        x_0: (B, C, H, W) clean images
        t: (B,) timesteps
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        noisy = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy, noise

    def training_loss(self, model, x_0):
        """Compute simplified diffusion loss."""
        batch_size = x_0.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Create noisy images
        x_t, _ = self.add_noise(x_0, t, noise)

        # Predict noise
        noise_pred = model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, model, shape):
        """Generate samples from pure noise using DDPM."""
        x = torch.randn(shape)  # Start from pure noise

        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, dtype=torch.long, device=x.device)

            # Predict noise
            noise_pred = model(x, t_batch)

            # Compute coefficients
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            # Mean
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred
            )

            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[t])
                x = mean + sigma * noise
            else:
                x = mean

        return x


# Usage
diffusion = SimpleDiffusion(T=1000)

# Training step
loss = diffusion.training_loss(model, clean_images)
loss.backward()

# Sampling
generated = diffusion.sample(model, shape=(4, 3, 64, 64))
```

---

## DDIM: Faster Sampling

Denoising Diffusion Probabilistic Models (DDPM) requires all 1000 steps. **DDIM** (Song et al., 2021) enables sampling in as few as 20-50 steps:

```
DDPM sampling:  1000 steps → ~30 seconds per image
DDIM sampling:    50 steps → ~2 seconds per image
DDIM sampling:    20 steps → ~1 second per image

Tradeoff: Fewer steps = faster but slightly lower quality
```

---

## Exercises

### Exercise 1: Compute Noisy Image
Given x₀ = [1.0, 2.0], ᾱ_t = 0.64, and ε = [0.5, -0.3], compute x_t.

**Solution:**
```
x_t = √ᾱ_t × x₀ + √(1-ᾱ_t) × ε
    = √0.64 × [1.0, 2.0] + √0.36 × [0.5, -0.3]
    = 0.8 × [1.0, 2.0] + 0.6 × [0.5, -0.3]
    = [0.8, 1.6] + [0.3, -0.18]
    = [1.1, 1.42]
```

### Exercise 2: Compute Diffusion Loss
Given true noise ε = [0.2, -0.1, 0.5] and predicted noise ε_θ = [0.15, -0.2, 0.4], compute MSE loss.

**Solution:**
```
MSE = mean((ε - ε_θ)²)
    = mean((0.2-0.15)², (-0.1-(-0.2))², (0.5-0.4)²)
    = mean(0.0025, 0.01, 0.01)
    = 0.0225 / 3
    = 0.0075
```

### Exercise 3: Guidance Scale Effect
Given ε_uncond = [0.1, 0.2, 0.3] and ε_cond = [0.5, 0.6, 0.7], compute guided noise for s=7.5.

**Solution:**
```
ε_guided = ε_uncond + 7.5 × (ε_cond - ε_uncond)
        = [0.1, 0.2, 0.3] + 7.5 × ([0.5, 0.6, 0.7] - [0.1, 0.2, 0.3])
        = [0.1, 0.2, 0.3] + 7.5 × [0.4, 0.4, 0.4]
        = [0.1, 0.2, 0.3] + [3.0, 3.0, 3.0]
        = [3.1, 3.2, 3.3]

Higher s amplifies the difference, making the output more strongly conditioned on text.
```

---

## Key Takeaways

1. **Forward process** gradually adds noise; **reverse process** learns to denoise
2. **The training objective is simple**: predict the noise (MSE loss)
3. **Stable Diffusion** works in latent space for 85× efficiency gain
4. **Cross-attention** bridges text and image representations
5. **Classifier-free guidance** controls quality-diversity tradeoff via scale parameter s
