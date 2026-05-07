# Vision-Language Models: CLIP, BLIP, ViT & Multimodal AI

## Why Vision-Language Models Matter

Real-world AI systems don't operate on text alone—they must **see and understand** simultaneously. Vision-Language Models (VLMs) bridge the gap between visual and textual understanding, enabling applications like image search ("find sunset photos"), visual question answering ("what color is the car?"), automatic image captioning, and autonomous driving perception.

CLIP (Contrastive Language-Image Pre-training) demonstrated that **aligning vision and language in a shared embedding space** unlocks powerful zero-shot capabilities. BLIP added generative captioning, and ViT (Vision Transformer) showed that Transformers work for images too. Together, these form the foundation of modern multimodal AI.

---

## ViT: Vision Transformer

### Core Idea: Images as Sequences

ViT (Dosovitskiy et al., 2020) treats an image like a sequence of tokens—the same way NLP models process text:

```
Text Transformer:  [CLS] The cat sat on the mat [SEP]
                       ↓    ↓   ↓   ↓   ↓   ↓

Vision Transformer: [CLS] P₁ P₂ P₃ ... P₁₉₆ [SEP]
                       ↓   ↓  ↓  ↓      ↓

Where P_i = Patch i (16×16 pixel region)
A 224×224 image → (224/16)² = 196 patches
```

### Architecture

```
Input Image (224×224×3)
         │
         ▼
┌─────────────────────┐
│  Split into Patches  │
│  16×16 → 196 patches │
│  Each: 16×16×3 = 768 │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Linear Projection   │
│  768 → d_model (768) │
│  (patch embedding)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Add Positional      │
│  Embedding           │
│  (learnable, 197 pos)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  prepend [CLS] token │
│  197 tokens total    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Transformer        │
│  Encoder (12 layers) │
│  (same as BERT)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  [CLS] → MLP Head   │
│  → Classification   │
└─────────────────────┘
```

### Patch Embedding

Each 16×16 patch is flattened and projected:

```
Patch i: (16, 16, 3) → flatten → (768,) → Linear → (d_model,)

The linear projection acts like the "token embedding" in NLP.
```

### ViT Implementation

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)        # (B, 768, 14, 14)
        x = x.flatten(2)        # (B, 768, 196)
        x = x.transpose(1, 2)   # (B, 196, 768)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 num_layers=12, num_heads=12, mlp_ratio=4.0, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        # Transformer blocks (same as BERT encoder)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)            # (B, 196, 768)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]               # (B, 768)
        return self.head(cls_output)         # (B, num_classes)
```

---

## CLIP: Contrastive Language-Image Pre-training

### Core Idea: Align Vision and Language

CLIP (Radford et al., 2021) learns a **shared embedding space** where matching image-text pairs are close and mismatched pairs are far:

```
          Image Encoder                    Text Encoder
     ┌──────────────────┐           ┌──────────────────┐
     │  ViT or ResNet   │           │  Transformer     │
     │                  │           │                  │
     │  Image I₁ ──► e₁ │           │  "a photo of     │
     │  Image I₂ ──► e₂ │           │   a cat" ──► t₁  │
     │  Image I₃ ──► e₃ │           │  "a photo of     │
     │                  │           │   a dog" ──► t₂  │
     └────────┬─────────┘           └────────┬─────────┘
              │                              │
              ▼                              ▼
         ┌─────────────────────────────────────┐
         │      Shared Embedding Space         │
         │                                     │
         │  e₁ ←→ t₁  ✓ close (matching)      │
         │  e₁ ←→ t₂  ✗ far (mismatched)       │
         │  e₃ ←→ t₁  ✗ far                    │
         │                                     │
         │  cosine(e₁, t₁) = 0.92             │
         │  cosine(e₁, t₂) = 0.15             │
         └─────────────────────────────────────┘
```

### Contrastive Loss

```
L = -1/N Σᵢ [log(exp(sim(eᵢ, tᵢ)/τ) / Σⱼ exp(sim(eᵢ, tⱼ)/τ))
            + log(exp(sim(tᵢ, eᵢ)/τ) / Σⱼ exp(sim(tᵢ, eⱼ)/τ))]

Where:
  sim(a, b) = (a · b) / (||a|| × ||b||)     (cosine similarity)
  τ         = Temperature parameter (learnable, typically ~0.07)
  N         = Batch size
```

This is **symmetric cross-entropy** over the similarity matrix:

```
                 Text t₁   Text t₂   Text t₃
Image I₁    [   0.92      0.15      0.23   ]  → softmax row → label at t₁
Image I₂    [   0.10      0.88      0.30   ]  → softmax row → label at t₂
Image I₃    [   0.25      0.12      0.85   ]  → softmax row → label at t₃
```

### Zero-Shot Classification with CLIP

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat_photo.jpg")
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits = outputs.logits_per_image  # (1, num_labels)
probs = logits.softmax(dim=-1)

print(f"Probabilities: {probs}")
# [0.94, 0.04, 0.02] → "cat" wins!
```

---

## BLIP: Bootstrapping Language-Image Pre-training

BLIP (Li et al., 2022) combines understanding and generation in a single framework:

```
┌──────────────────────────────────────────────┐
│                  BLIP Framework              │
│                                               │
│  Image ──► Visual Transformer (ViT)          │
│                  │                            │
│                  ▼                            │
│            Image Features                    │
│             /         \                       │
│            ▼           ▼                      │
│     ┌──────────┐ ┌──────────────┐           │
│     │   ITC    │ │   ITM        │           │
│     │(Image-   │ │(Image-Text   │           │
│     │ Text     │ │ Matching)    │           │
│     │Contrast.)│ │              │           │
│     └──────────┘ └──────────────┘           │
│                                               │
│     ┌──────────────────────────┐             │
│     │     Captioner (LM)       │             │
│     │  Image → "A cat sitting  │             │
│     │   on a red sofa"         │             │
│     └──────────────────────────┘             │
│                                               │
│  Web Text ──► Text Encoder + Filter          │
│  (noisy captions are filtered by ITM)        │
└──────────────────────────────────────────────┘
```

### BLIP Components

| Component | Task | Description |
|-----------|------|-------------|
| **ITC** | Image-Text Contrastive | Aligns image/text in shared space |
| **ITM** | Image-Text Matching | Binary: is this text describing this image? |
| **LM** | Language Modeling | Generate captions conditioned on image |
| **CAPF** | Captioning Filter | Filters noisy web captions |

---

## Multimodal Embeddings

### The Shared Embedding Space

```
           CLIP Embedding Space (512-dim)
           ─────────────────────────────

              "dog" ●
                    ●    "puppy" ●
         "cat" ●
                ●          ● "wolf"
                    ●
    "car" ●                            ● "fox"
                    ●
                  ● "truck"

              ● "bicycle"

        Images and text live in the SAME space!
        Nearest neighbors are semantically similar.
```

### Using CLIP Embeddings for Search

```python
import torch
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Get image embeddings
image_features = model.get_image_features(pixel_values=image_tensor)
# (B, 512) — L2 normalized

# Get text embeddings
text_features = model.get_text_features(input_ids=text_token_ids)
# (B, 512) — L2 normalized

# Compute similarity
similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
```

---

## Real-World Applications

| Application | Model | Use Case |
|---|---|---|
| **Image Search** | CLIP | Search photos using natural language |
| **Image Captioning** | BLIP-2, GPT-4V | Generate descriptions of images |
| **Visual QA** | VQA, LLaVA | Answer questions about images |
| **Autonomous Driving** | CLIP + ViT | Scene understanding and reasoning |
| **Content Moderation** | CLIP | Detect NSFW content in images |
| **Medical Imaging** | BiomedCLIP | Zero-shot disease classification |
| **Robotics** | CLIP + RT-2 | Visual grounding for manipulation |

---

## Exercises

### Exercise 1: Compute Patch Count
For a ViT with input image 384×384 and patch size 24, how many patches are created?

**Solution:**
```
num_patches = (384 / 24)² = 16² = 256 patches
Each patch has 24 × 24 × 3 = 1,728 values
After projection: each patch → 768-dim vector
```

### Exercise 2: CLIP Similarity
Given image embedding e = [0.6, 0.8] and text embedding t = [0.3, 0.4], compute cosine similarity.

**Solution:**
```
cosine(e, t) = (e · t) / (||e|| × ||t||)
= (0.6×0.3 + 0.8×0.4) / (√(0.36+0.64) × √(0.09+0.16))
= (0.18 + 0.32) / (1.0 × 0.5)
= 0.50 / 0.50
= 1.0

The vectors are perfectly aligned (t = 0.5 × e)!
```

### Exercise 3: Contrastive Loss
Given similarities sim(e₁, t₁) = 0.9, sim(e₁, t₂) = 0.2, sim(e₁, t₃) = 0.3, and τ = 0.1, compute the loss for image e₁.

**Solution:**
```
scores = [0.9, 0.2, 0.3] / 0.1 = [9.0, 2.0, 3.0]

exp_scores = [e⁹, e², e³] = [8103, 7.39, 20.09]

Loss = -log(exp(9.0) / (exp(9.0) + exp(2.0) + exp(3.0)))
     = -log(8103 / 8130.48)
     = -log(0.9966)
     = 0.0034

Very low loss → the model correctly matches e₁ with t₁!
```

---

## Key Takeaways

1. **ViT** treats images as sequences of patches, enabling Transformer-based vision
2. **CLIP** aligns images and text via contrastive learning in a shared embedding space
3. **BLIP** combines understanding (ITC, ITM) and generation (captioning) for multimodal tasks
4. **Zero-shot transfer** is CLIP's superpower—classify anything without training data
5. **Multimodal embeddings** enable cross-modal search, retrieval, and reasoning
