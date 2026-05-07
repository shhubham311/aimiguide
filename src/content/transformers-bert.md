# Transformers & BERT: The Architecture That Revolutionized NLP

## Why Transformers Matter in AI/ML

Before 2017, NLP was dominated by Recurrent Neural Networks (RNNs) and LSTMs, which processed text sequentially—one word at a time. This created a fundamental bottleneck: **long-range dependencies** were lost, training was slow, and parallelization was nearly impossible. The Transformer architecture, introduced in *"Attention Is All You Need"* (Vaswani et al., 2017), shattered these limitations by replacing recurrence entirely with **self-attention mechanisms**.

Transformers are now the backbone of virtually every modern AI system: GPT, BERT, LLaMA, Stable Diffusion, CLIP, Whisper, and more. Understanding Transformers is not optional—it is **foundational knowledge** for any ML practitioner.

---

## The Core Intuition: Attention

The key insight of the Transformer is: **instead of processing words in order, let every word attend to every other word simultaneously**. This means the model can directly connect a pronoun to its antecedent regardless of distance.

### Scaled Dot-Product Attention

The fundamental operation is:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- **Q (Query)** — What am I looking for? (size: n × d_k)
- **K (Key)** — What do I contain? (size: n × d_k)
- **V (Value)** — What information do I provide? (size: n × d_v)
- **d_k** — Dimension of the key vectors (scaling factor prevents softmax saturation)

```
           ┌─────────────┐
    Q ───► │  MatMul     │──┐
           │  Q × K^T    │  │    ┌─────────────┐
    K ───► └──────┬──────┘  ├───►│   Scale     │──┐
                  │         │     │ ÷ √d_k      │  │
                  │    ┌────┴──┐  └─────────────┘  │
                  │    │      │                    │
                  │    ▼      ▼                    ▼
                  │ ┌──────────────────┐    ┌────────────┐
                  └─│  Softmax (mask)  │    │ MatMul     │──► Output
                    └────────┬─────────┘    │ Attn × V   │
                             │              └────────────┘
                             │                  ▲
                             ▼                  │
                    Attention Weights      V ────┘
                    (each row sums to 1)
```

### Multi-Head Attention

Instead of a single attention function, the Transformer runs **h parallel attention heads**, each learning different patterns:

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
```

Each head has its own learned projection matrices W_i^Q, W_i^K, W_i^V (size d_model × d_k). This lets the model simultaneously focus on:
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Semantic similarity
- Head 3: Positional proximity
- Head h: Coreference resolution

**Why multi-head?** A single attention function would need to average all these patterns together; multiple heads let each specialize.

---

## Positional Encoding

Since Transformers process all tokens in parallel, they have no inherent sense of order. Positional encodings inject positional information:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- **pos** — Position index (0, 1, 2, ...)
- **i** — Dimension index
- **d_model** — Model dimension (e.g., 512)

This creates a unique "fingerprint" for each position that the model can learn to interpret.

---

## The Full Transformer Architecture

```
                    Encoder Side                    Decoder Side
                    ─────────────                    ────────────

Input Embedding ────► [+] ────► Multi-Head ──► [+] ──► FFN        Output Embedding ────► [+] ────► Masked
     +                   ▲        Attention   ▲       │    │          +                         ▲        Multi-Head
Pos. Encoding ──────────┘      ──► [+] ◄─────┘       ▼    ▼     Pos. Encoding ───────────────┘        Attention
                                   ▲            ┌─────────┐                                              │
                                   │            │LayerNorm│                                              ▼
                                   │            └─────────┘                                         [+] ──► FFN ──► [+] ──► Linear ──► Softmax
                                   │                                                                   ▲         ▲
                              Add & Norm                                                           Add&Norm   Add&Norm
                              ┌─────────┐                                                          ┌─────────┐
                              │LayerNorm│                                                          │LayerNorm│
                              └─────────┘                                                          └─────────┘

                              ×N layers (stacked)                                                ×N layers (stacked)
```

### Feed-Forward Network (FFN)

Each Transformer layer also contains a position-wise FFN:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

This is applied identically to each position. It typically expands to 4× the model dimension and then projects back.

### Layer Normalization & Residual Connections

```
output = LayerNorm(x + Sublayer(x))
```

Residual connections allow gradients to flow directly through deep networks, and LayerNorm stabilizes training.

---

## BERT: Bidirectional Encoder Representations from Transformers

BERT (Devlin et al., 2018) takes the **encoder** half of the Transformer and pre-trains it on two tasks:

### 1. Masked Language Modeling (MLM)

Randomly mask 15% of tokens and predict them:

```
Input:  The [MASK] sat on the [MASK].
Label:  The  cat   sat on the  mat
```

The mask token `[MASK]` is replaced with:
- 80% of the time: `[MASK]`
- 10% of the time: a random token
- 10% of the time: the original token (unchanged)

### 2. Next Sentence Prediction (NSP)

Given two sentences, predict whether B follows A:

```
Input:  [CLS] The cat sat on the mat. [SEP] It was very tired. [SEP]
Label:  IsNext ✓
```

### BERT Architecture Variants

| Variant | Layers | Hidden Size | Attention Heads | Parameters |
|---------|--------|-------------|-----------------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### Fine-tuning BERT

```
┌───────────────┐     ┌─────────────────┐     ┌──────────┐
│  Raw Text     │────►│  BERT Encoder   │────►│  [CLS]   │────► Classification Head
│  Tokens       │     │  (Pre-trained)  │     │  Token   │     ┌────────┐
│               │     │                 │     │          │     │  Dense  │────► Label
│  [CLS] w1 w2 │     │  12 Layers of   │     │  Output  │     │  + Soft │
│  w3 ... [SEP] │     │  Self-Attention │     │  (768-d) │     │  max    │
└───────────────┘     └─────────────────┘     └──────────┘     └────────┘
```

---

## Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, num_heads, seq_len, d_k)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Project and reshape: (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        out, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads: (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(out), attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Usage example
d_model, num_heads, d_ff = 512, 8, 2048
encoder = TransformerEncoderLayer(d_model, num_heads, d_ff)

# Simulate batch of 4 sequences, each 10 tokens, 512-dim
x = torch.randn(4, 10, d_model)
output = encoder(x)
print(f"Input shape:  {x.shape}")    # (4, 10, 512)
print(f"Output shape: {output.shape}")  # (4, 10, 512)
```

### Using Hugging Face BERT for Classification

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

texts = ["This movie was absolutely fantastic!", "Terrible experience, would not recommend."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)

print(f"Predictions: {preds}")  # [1 (positive), 0 (negative)]
```

---

## Real-World Applications

| Application | How Transformers Help |
|---|---|
| **Search Engines** | BERT理解查询意图而非纯关键词匹配 |
| **Sentiment Analysis** | 双向上下文捕获否定与讽刺 |
| **Named Entity Recognition** | 完整上下文消歧实体 |
| **Question Answering** | 跨段落推理与证据定位 |
| **Code Understanding** | GitHub Copilot使用GPT变体 |

---

## Exercises

### Exercise 1: Implement Positional Encoding from Scratch
Compute the positional encoding for position=3, dimension=5 (i.e., PE(3,0), PE(3,1), PE(3,2), PE(3,3), PE(3,4)) with d_model=512.

**Solution:**
```
pos = 3, d_model = 512
div_term_i = exp(2i * (-ln(10000) / 512))

PE(3, 0) = sin(3 / 10000^(0/512))    = sin(3 / 1)         = sin(3.0)
PE(3, 1) = cos(3 / 10000^(0/512))    = cos(3 / 1)         = cos(3.0)
PE(3, 2) = sin(3 / 10000^(2/512))    = sin(3 / 1.0091)    ≈ sin(2.973)
PE(3, 3) = cos(3 / 10000^(2/512))    = cos(3 / 1.0091)    ≈ cos(2.973)
PE(3, 4) = sin(3 / 10000^(4/512))    = sin(3 / 1.0183)    ≈ sin(2.946)
```

### Exercise 2: Compute Attention Weights
Given Q = [1, 0], K = [1, 2], V = [3, 4] (single token, d_k=2), compute the attention output.

**Solution:**
```
QK^T = [1, 0] · [1, 2]^T = 1*1 + 0*2 = 1

sqrt(d_k) = sqrt(2) ≈ 1.414

score = 1 / 1.414 ≈ 0.707

softmax([0.707]) = [1.0]  (single element)

Output = 1.0 * [3, 4] = [3, 4]
```

### Exercise 3: BERT Token Analysis
For the sentence "The bank of the river", explain why "bank" gets disambiguated correctly.

**Solution:** BERT's bidirectional attention lets the "[MASK]" for "bank" attend to both "The" (left) and "of the river" (right) simultaneously. The self-attention weights will be highest for "river", pulling the representation toward the geographic meaning of "bank" rather than the financial one.

---

## Key Takeaways

1. **Self-attention** replaces recurrence—every token attends to every other token
2. **Multi-head attention** lets the model learn diverse relationship patterns
3. **Positional encoding** injects order information into the parallel architecture
4. **BERT** uses only the encoder with bidirectional context (MLM + NSP pre-training)
5. **Transformers scale**—the architecture enables models from 110M to 1T+ parameters
