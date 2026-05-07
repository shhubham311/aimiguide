# Large Language Models: GPT, LLaMA, Mistral & Beyond

## Why LLMs Matter

Large Language Models (LLMs) represent one of the most transformative breakthroughs in AI history. These models—GPT-4, LLaMA, Mistral, Claude, Gemini—demonstrate that **scaling up a simple next-token prediction objective** can produce emergent capabilities like reasoning, code generation, multilingual translation, and creative writing. They power ChatGPT, GitHub Copilot, Google Bard, and countless enterprise applications. Understanding how LLMs work, their architecture choices, and their training pipeline is essential for any ML engineer.

---

## The Autoregressive Language Modeling Objective

All decoder-only LLMs are trained on the same fundamental task: **predict the next token**.

```
Given:    "The capital of France is"
Predict:  " Paris"

Loss = -log P("Paris" | "The capital of France is")
```

Formally, the training objective maximizes the log-likelihood:

```
L(θ) = -Σ log P(x_t | x_1, x_2, ..., x_{t-1}; θ)
```

Where:
- **θ** — All model parameters
- **x_t** — Token at position t
- **P(x_t | ...)** — Probability the model assigns to the correct next token

This is computed using **cross-entropy loss**:

```
CrossEntropy = -Σ y_i * log(ŷ_i)
```

---

## GPT Architecture Evolution

### GPT-1 → GPT-2 → GPT-3 → GPT-4

```
┌──────────────────────────────────────────────────────────────┐
│                    GPT Family Scaling                        │
├──────────┬─────────┬──────────┬──────────┬─────────────────┤
│  Model   │ Params  │  Layers  │ Context  │ Training Data   │
├──────────┼─────────┼──────────┼──────────┼─────────────────┤
│  GPT-1   │  117M   │   12     │  512     │  BookCorpus     │
│  GPT-2   │  1.5B   │   48     │  1024    │  WebText        │
│  GPT-3   │  175B   │   96     │  2048    │  300B tokens    │
│  GPT-3.5 │  ~200B  │   ~96    │  4096    │  + RLHF data    │
│  GPT-4   │  ~1.8T  │  ~120    │  128K    │  Multimodal     │
└──────────┴─────────┴──────────┴──────────┴─────────────────┘
```

### GPT Architecture (Decoder-Only Transformer)

```
              Token IDs
                 │
                 ▼
         ┌───────────────┐
         │ Token          │
         │ Embedding      │
         │ + Pos. Encode  │
         └───────┬───────┘
                 │
         ┌───────▼───────┐
         │ Transformer   │     ┌──────────────────────────┐
         │ Decoder Block │     │  ┌─────────────────────┐ │
         │               │     │  │ Masked Self-Attn    │ │
         │  ┌───────────┐│     │  │ (can only see past) │ │
         │  │ Masked    ││     │  └─────────┬───────────┘ │
         │  │ Self-     ││     │            │             │
         │  │ Attention ││     │  ┌─────────▼───────────┐ │
         │  └─────┬─────┘│     │  │ Add & LayerNorm     │ │
         │        │      │     │  └─────────┬───────────┘ │
         │  ┌─────▼─────┐│     │            │             │
         │  │ FFN       ││     │  ┌─────────▼───────────┐ │
         │  │ (SwiGLU)  ││     │  │ Feed-Forward Network│ │
         │  └─────┬─────┘│     │  └─────────┬───────────┘ │
         │        │      │     │            │             │
         │  ┌─────▼─────┐│     │  ┌─────────▼───────────┐ │
         │  │ LayerNorm ││     │  │ Add & LayerNorm     │ │
         │  └───────────┘│     │  └─────────────────────┘ │
         └───────┬───────┘     └──────────────────────────┘
                 │                      × N layers
         ┌───────▼───────┐
         │ LM Head       │
         │ (Linear→Softmax)│
         └───────┬───────┘
                 │
                 ▼
        Next Token Probabilities
```

### Causal (Masked) Attention

The key difference from BERT: GPT uses a **causal mask** to prevent attending to future tokens:

```
     To:  The   cat   sat
From:
The   [  1.0   -∞    -∞  ]    ← "The" can only see itself
cat   [  0.8   1.0   -∞  ]    ← "cat" can see "The" and itself
sat   [  0.3   0.6   1.0 ]    ← "sat" can see everything before it

(Values are softmax-ed attention weights; -∞ → 0 after softmax)
```

---

## LLaMA: Open-Source Efficiency

Meta's LLaMA (Large Language Model Meta AI) proved that **smaller, well-trained models** can match or exceed much larger ones.

### Key Innovations

| Feature | GPT-3 Style | LLaMA Style |
|---------|-------------|-------------|
| Positional Encoding | Learned | RoPE (Rotary) |
| Activation | GELU | SwiGLU |
| Normalization | LayerNorm | RMSNorm |
| Attention | Dense | Grouped-Query Attention (GQA) |

### RMSNorm vs LayerNorm

```
LayerNorm:  x_norm = (x - μ) / σ × γ + β     (computes mean μ and std σ)

RMSNorm:    x_norm = x / RMS(x) × γ            (no mean centering, no bias)
            RMS(x) = sqrt(mean(x²))
```

RMSNorm is ~10% faster and performs comparably.

### RoPE (Rotary Position Embedding)

RoPE encodes position by **rotating** the query and key vectors in a 2D plane:

```
For each pair (x_2i, x_{2i+1}) at position m:

┌               ┐   ┌                    ┐ ┌              ┐
│ x'_2i         │   │ cos(mθ_i) -sin(mθ_i)│ │ x_2i         │
│ x'_{2i+1}     │ = │ sin(mθ_i)  cos(mθ_i)│ │ x_{2i+1}     │
└               ┘   └                    ┘ └              ┘

where θ_i = 10000^(-2i/d_model)
```

This makes relative position information naturally available in the dot product.

---

## Mistral: Efficiency Through Architecture

Mistral 7B outperforms LLaMA-2 13B using:

1. **Sliding Window Attention (SWA)**: Each token attends to at most W=4096 previous tokens instead of the full sequence. This reduces memory from O(n²) to O(n × W).

```
Full Attention:     Token 10 attends to [1,2,3,...,10]    → O(n²)
Sliding Window:     Token 10 attends to [7,8,9,10]       → O(n × W)

     ┌──────────────────────────────┐
     │  [7] [8] [9] [10]           │ ← Window size W=4
     │       [8] [9] [10] [11]     │
     │            ...               │
     │              [97][98][99]    │
     └──────────────────────────────┘
```

2. **GQA (Grouped-Query Attention)**: Shares K and V across multiple query heads, reducing KV-cache size.

---

## Inference: How LLMs Generate Text

### Autoregressive Generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Greedy decoding
output_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output_ids[0]))

# Temperature sampling
output_ids = model.generate(
    input_ids,
    max_new_tokens=20,
    temperature=0.7,      # Higher = more random, Lower = more deterministic
    top_k=50,              # Only sample from top K tokens
    top_p=0.95,            # Nucleus sampling
    do_sample=True
)
```

### Sampling Strategies

```
Logits (raw scores): [2.1, 1.5, 0.3, -0.8, -1.2]

After Temperature (T=0.5): [4.2, 3.0, 0.6, -1.6, -2.4]
    → Sharpens distribution (more focused)

After Temperature (T=2.0): [1.05, 0.75, 0.15, -0.4, -0.6]
    → Flattens distribution (more random)

Top-K (K=3): Keep only top 3 → [2.1, 1.5, 0.3]
Top-P (P=0.9): Keep tokens until cumulative prob ≥ 0.9
```

---

## Tokenization: BPE & SentencePiece

LLMs don't operate on characters or words—they use **subword tokens**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Artificial Intelligence"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print(f"Tokens: {tokens}")        # ['Art', 'ificial', ' Intelligence']
print(f"IDs:    {ids}")            # [16043, 13934, 318]
print(f"Vocab size: {tokenizer.vocab_size}")  # 50257
```

**BPE (Byte Pair Encoding)** merges the most frequent character pairs iteratively until a desired vocabulary size is reached.

---

## Real-World Applications

| Use Case | LLM Role | Example |
|----------|----------|---------|
| **Code Generation** | Autocomplete & generation | GitHub Copilot, Cursor |
| **Customer Support** | Conversational AI | Klarna's AI assistant |
| **Document Analysis** | Summarization & extraction | Notion AI, Adobe |
| **Education** | Tutoring & explanation | Khan Academy's Khanmigo |
| **Research** | Literature review & drafting | Elicit, Semantic Scholar |

---

## Exercises

### Exercise 1: Compute Perplexity
Given a model assigns probabilities [0.3, 0.5, 0.2] to the correct tokens in a 3-token sequence, compute perplexity.

**Solution:**
```
PPL = exp(-1/3 × (log(0.3) + log(0.5) + log(0.2)))
    = exp(-1/3 × (-1.204 + -0.693 + -1.609))
    = exp(-1/3 × (-3.506))
    = exp(1.169)
    ≈ 3.22

Lower perplexity = better model
```

### Exercise 2: Causal Mask Implementation
Implement a 4×4 causal attention mask.

**Solution:**
```python
import torch

def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)  # Upper triangular
    return mask == 0  # True where attention is allowed

# Result:
# [[T, F, F, F],
#  [T, T, F, F],
#  [T, T, T, F],
#  [T, T, T, T]]
```

### Exercise 3: Analyze Tokenization
How many tokens does "I love machine learning!" have in GPT-2 BPE?

**Solution:**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
tokens = tok.tokenize("I love machine learning!")
# ['I', ' love', ' machine', ' learning', '!'] → 5 tokens
```

---

## Key Takeaways

1. **Autoregressive modeling** (next-token prediction) is the universal training objective for LLMs
2. **Causal masking** ensures tokens only attend to the past during training
3. **Scaling laws** show predictable performance gains with more data, parameters, and compute
4. **Efficiency innovations** (RoPE, RMSNorm, GQA, SWA) allow smaller models to punch above their weight
5. **Tokenization** via BPE bridges the gap between characters and words with a fixed vocabulary
