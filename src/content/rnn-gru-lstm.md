# RNN, GRU, LSTM & Attention Basics

## Why This Topic Matters

Recurrent Neural Networks (RNNs) are the foundational architecture for **sequential data** — text, speech, time series, DNA sequences, and stock prices. Unlike feedforward networks that process fixed-size inputs, RNNs maintain an internal **memory state** that allows them to process sequences of arbitrary length.

The evolution from vanilla RNN → LSTM → GRU → Attention → Transformer represents one of the most important arcs in deep learning history. Vanilla RNNs suffer from the **vanishing gradient problem** for long sequences. LSTMs (1997) solved this with gated memory cells. GRUs (2014) simplified LSTMs. And Attention mechanisms (2017) ultimately enabled the Transformer architecture that powers GPT, BERT, and virtually all modern NLP systems.

---

## 1. The Problem: Sequential Data

### Why Can't We Use MLPs or CNNs?

```
Sentences have variable length and sequential dependencies:

"The cat [sat] on the [mat]"     → "sat" depends on "cat", "mat" depends on "the"
"I was born in [France]. I speak [French]"  → "French" depends on "France" (long-range!)

MLP: Fixed input size → Can't handle variable-length sequences
CNN: Fixed receptive field → Can't capture long-range dependencies easily
RNN: Maintains state → Can process any-length sequences with context

Key insight: The meaning of a word depends on ALL previous words in the sequence.
```

---

## 2. Vanilla RNN (Recurrent Neural Network)

### Architecture

The RNN processes input one step at a time, maintaining a hidden state $\mathbf{h}$ that serves as "memory":

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
$$\hat{\mathbf{y}}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

Where:
- $\mathbf{x}_t \in \mathbb{R}^{d}$ is the input at time step $t$
- $\mathbf{h}_t \in \mathbb{R}^{h}$ is the hidden state at time step $t$
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ is the hidden-to-hidden weight matrix
- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$ is the input-to-hidden weight matrix
- $\mathbf{W}_{hy} \in \mathbb{R}^{o \times h}$ is the hidden-to-output weight matrix
- $\mathbf{b}_h, \mathbf{b}_y$ are bias vectors

### Unrolled RNN Diagram

```
Time Step:      t=1          t=2          t=3          t=4

Input:         "The"       "cat"       "sat"       "on"
 x₁            x₂          x₃          x₄
  │             │           │           │
  ▼             ▼           ▼           ▼
┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐
│   │ h₁→│   │ h₂→│   │ h₃→│   │ h₄→│   │
│ R │    │ R │    │ R │    │ R │    │ R │
│ N │    │ N │    │ N │    │ N │    │ N │
│ N │    │ N │    │ N │    │ N │    │ N │
└─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘
  │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼
 ŷ₁        ŷ₂        ŷ₃        ŷ₄        ŷ₅

Output:      "The"      "cat"     "sat"     "on"      "the"

Hidden state hₜ carries information from ALL previous time steps.
Each RNN cell has SHARED weights (W_hh, W_xh are the same at every step).
```

### The Vanishing Gradient Problem in RNNs

During backpropagation through time (BPTT), gradients are multiplied at each time step:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \mathbf{W}_{hh}^T \cdot \text{diag}(\tanh'(\cdot))$$

```
If ‖W_hh‖ < 1 (typical): gradients shrink exponentially

Time step:     t=1   t=2   t=3   ...  t=50  t=100
Gradient:    1.0   0.8   0.64  ...  ~0     ~0     (vanished!)

If ‖W_hh‖ > 1: gradients grow exponentially (exploding gradient)

Time step:     t=1   t=2   t=3   ...  t=50  t=100  
Gradient:    1.0   1.2   1.44  ...  ~∞     ~∞     (exploded!)

Result: Vanilla RNNs can only remember ~10-20 time steps effectively.
```

### PyTorch Vanilla RNN

```python
import torch
import torch.nn as nn

# Single-layer RNN
rnn = nn.RNN(input_size=300, hidden_size=128, batch_first=True)

# Input: [batch_size, seq_len, input_size]
# h_0:   [num_layers, batch_size, hidden_size]
x = torch.randn(32, 50, 300)  # 32 sequences, 50 tokens, 300-dim embeddings
h0 = torch.zeros(1, 32, 128)  # Initial hidden state

output, hidden = rnn(x, h0)
print(f"Output shape: {output.shape}")   # [32, 50, 128]
print(f"Hidden shape: {hidden.shape}")   # [1, 32, 128]

# output[-1] == hidden[-1] (last hidden state)
```

---

## 3. LSTM (Long Short-Term Memory)

### The Core Innovation: Gated Memory

LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem using a **cell state** $\mathbf{c}_t$ with three gates that control information flow:

```
LSTM Cell at time step t:

                    x_t (input)
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌───▼─────┐
    │ Forget  │ │  Input  │ │ Output  │
    │  Gate   │ │  Gate   │ │  Gate   │
    │  (f_t)  │ │ (i_t,   │ │  (o_t)  │
    │         │ │  ĉ_t)   │ │         │
    └────┬────┘ └────┬────┘ └───┬─────┘
         │           │           │
         │    ┌──────▼──────┐    │
         │    │  Candidate  │    │
         │    │  c̃_t        │    │
         │    └──────┬──────┘    │
         │           │           │
         │           ▼           │
         │     i_t * c̃_t        │
         │           │           │
         ▼           │           │
    c_{t-1} ───→ ⊗ ──┤           │
    (previous)  │    │           │
         │      ▼    ▼           │
         │   ⊕ (add)             │
         │    │                   │
         │    ▼ c_t               │
         │    │ (new cell state)  │
         │    │                   │
         │    ├───────────────────┤
         │    │                   ▼
         │    │              tanh(c_t)
         │    │                   │
         │    │                   ▼
         │    │              o_t * tanh(c_t)
         │    │                   │
         │    │                   ▼
         │    │               h_t (output)
         │    │                   │
         │    └──→ c_t ──────────┘  (cell state flows to next step)
         │
         └──→ h_{t-1} (previous hidden state)
```

### LSTM Equations

**Forget Gate** — What information to discard:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input Gate** — What new information to store:
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Cell State Update** — Combine old and new:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output Gate** — What to output:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Where:
- $\sigma$ is the sigmoid function (outputs 0–1, acts as a gate)
- $\odot$ is element-wise (Hadamard) product
- $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ denotes concatenation
- $\mathbf{c}_t$ is the cell state (long-term memory)
- $\mathbf{h}_t$ is the hidden state (short-term memory/output)

### Why LSTMs Solve Vanishing Gradients

```
Cell State Update:
  c_t = f_t * c_{t-1} + i_t * c̃_t

If f_t ≈ 1 (forget gate keeps everything):
  c_t ≈ c_{t-1} + small_update
  dc_t/dc_{t-1} ≈ f_t ≈ 1  (gradient flows perfectly!)

The cell state has an "additive" path (not multiplicative):
  → Gradient can flow unchanged across many time steps
  → LSTMs can remember information for 100+ time steps

Compare with vanilla RNN:
  h_t = tanh(W * h_{t-1} + ...)  → Multiplicative → gradient vanishes
  c_t = f_t * c_{t-1} + ...      → Additive → gradient preserved
```

### PyTorch LSTM

```python
import torch
import torch.nn as nn

# Single-layer LSTM
lstm = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, 
               num_layers=2, dropout=0.2, bidirectional=True)

x = torch.randn(32, 50, 300)  # [batch, seq_len, embed_dim]
h0 = torch.zeros(4, 32, 256)  # [num_layers*2 (bidirectional), batch, hidden]
c0 = torch.zeros(4, 32, 256)  # Cell state

output, (hidden, cell) = lstm(x, (h0, c0))
print(f"Output:  {output.shape}")   # [32, 50, 512] (256*2 for bidirectional)
print(f"Hidden:  {hidden.shape}")   # [4, 32, 256]
print(f"Cell:    {cell.shape}")     # [4, 32, 256]
```

---

## 4. GRU (Gated Recurrent Unit)

### Motivation

GRUs (Cho et al., 2014) simplify LSTMs by merging the cell state and hidden state, and combining the forget and input gates into a single **update gate**. They achieve similar performance to LSTMs with fewer parameters.

### GRU Equations

**Update Gate** — Controls how much past information to keep:
$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Reset Gate** — Controls how much past information to forget:
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Candidate Hidden State:**
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Hidden State Update:**
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### GRU Diagram

```
GRU Cell at time step t:

          x_t (input)
           │
     ┌─────┼─────┐
     │     │     │
  ┌──▼──┐ ┌▼──┐ ┌▼───┐
  │     │ │   │ │    │
  │ z_t │ │r_t│ │    │
  │Update│ │Reset  │
  │Gate │ │Gate   │
  └──┬──┘ └┬──┘ └┬───┘
     │     │     │
     │     ▼     │
     │  r_t * h_{t-1} + x_t
     │     │     │
     │     ▼     │
     │  ┌──────┐ │
     │  │ tanh │ │
     │  │ h̃_t  │ │
     │  └──┬───┘ │
     │     │     │
     │     │     ▼
     │     │  (1-z_t) * h_{t-1}
     │     │     │
     ▼     ▼     ▼
      z_t * h̃_t + (1-z_t) * h_{t-1}
               │
               ▼
              h_t (output → next step)
```

### LSTM vs GRU Comparison

```
LSTM Cell:                    GRU Cell:
┌───────────────────┐        ┌───────────────┐
│  3 Gates          │        │  2 Gates      │
│  (f, i, o)        │        │  (z, r)       │
│  + Cell state c_t  │        │  No cell state│
│  More parameters  │        │  Fewer params │
│  More expressive  │        │  Faster train │
└──────────────────┘        └───────────────┘

                  LSTM              GRU
Parameters:       4 × (h + h × d)  3 × (h + h × d)
Gates:           3 (+ candidate)   2 (+ candidate)
Cell state:      Yes               No (merged with h)
Speed:           Slower            Faster
Memory:          More              Less
Performance:     Slightly better   Competitive
```

### PyTorch GRU

```python
import torch.nn as nn

gru = nn.GRU(input_size=300, hidden_size=256, num_layers=2, 
             batch_first=True, bidirectional=True, dropout=0.2)

x = torch.randn(32, 50, 300)
h0 = torch.zeros(4, 32, 256)  # [num_layers*2, batch, hidden]

output, hidden = gru(x, h0)
print(f"Output: {output.shape}")  # [32, 50, 512]
print(f"Hidden: {hidden.shape}")  # [4, 32, 256]
```

---

## 5. Sequence-to-Sequence (Seq2Seq) Models

### Encoder-Decoder Architecture

Seq2Seq models (Sutskever et al., 2014) use one RNN to **encode** the input sequence and another to **decode** the output sequence:

```
Seq2Seq for Translation (English → French):

ENCODER                              DECODER
                              h_t ──→
"The cat sat"              ┌──────────┐
                         │  Decoder  │──→ "Le"
 x₁→[LSTM]→h₁            │  (LSTM)  │
 x₂→[LSTM]→h₂            │          │
 x₃→[LSTM]→h₃──context──→│          │──→ "chat"
                         │          │
                         │          │──→ "assis"
                         │          │
                         │          │──→ <EOS>
                         └──────────┘

The context vector (last hidden state) must encode ALL input information.
Problem: Fixed-size bottleneck for long sequences.
```

### With Attention

Attention allows the decoder to "look back" at all encoder hidden states, not just the last one:

```
Seq2Seq with Attention:

ENCODER                          DECODER
h₁  h₂  h₃  h₄  h₅
 │   │   │   │   │
 ↓   ↓   ↓   ↓   ↓
[⊕]─[⊕]─[⊕]─[⊕]─[⊕]     ┌──────────┐
 │   │   │   │   │       │  Decoder  │
 ↑   ↑   ↑   ↑   ↑       │          │──→ ŷ₁
 │   │   │   │   │       │          │
 │←──╱╲──────────│       │          │──→ ŷ₂
 │  Attention    │       │          │
 │  Weights      │       │          │
 │  (α₁,α₂,...) │       │          │──→ ŷ₃
 │               │       │          │
 └───────────────┘       └──────────┘

At each decoding step, attention computes:
  1. Score: e_i = score(h_{dec}, h_{enc,i})
  2. Weights: α_i = softmax(e_i)
  3. Context: c = Σ α_i × h_{enc,i}
  4. Output: ŷ_t = Decoder(h_{dec}, c)
```

---

## 6. Attention Mechanism

### Scaled Dot-Product Attention (Bahdanau / Transformer)

The attention mechanism computes a weighted combination of values based on query-key compatibility:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

Where:
- $\mathbf{Q}$ (Query) = "What am I looking for?" $[\text{seq}_q \times d_k]$
- $\mathbf{K}$ (Key) = "What do I contain?" $[\text{seq}_k \times d_k]$
- $\mathbf{V}$ (Value) = "What information do I provide?" $[\text{seq}_k \times d_v]$
- $d_k$ is the key dimension (scaling factor prevents large dot products)

```
Attention Computation:

Q = [q₁, q₂, q₃]   (queries: 3 items looking for info)
K = [k₁, k₂, k₃, k₄] (keys: 4 items with labels)
V = [v₁, v₂, v₃, v₄] (values: 4 items with content)

Step 1: Compute scores (Q · K^T)
         k₁    k₂    k₃    k₄
   q₁ [ 2.1   0.5  -1.0   3.2 ]
   q₂ [ 1.0   2.5   0.8  -0.3 ]
   q₃ [-0.5   1.8   2.2   1.0 ]

Step 2: Scale by 1/√d_k, then softmax (rows sum to 1)
   q₁ [ 0.15  0.05  0.01  0.79 ]  ← q₁ focuses heavily on v₄
   q₂ [ 0.10  0.45  0.35  0.10 ]  ← q₂ focuses on v₂ and v₃
   q₃ [ 0.02  0.25  0.50  0.23 ]  ← q₃ focuses on v₃

Step 3: Weighted sum of values
   output₁ = 0.15*v₁ + 0.05*v₂ + 0.01*v₃ + 0.79*v₄
   output₂ = 0.10*v₁ + 0.45*v₂ + 0.35*v₃ + 0.10*v₄
   output₃ = 0.02*v₁ + 0.25*v₂ + 0.50*v₃ + 0.23*v₄
```

### Multi-Head Attention

Instead of a single attention function, use multiple parallel "heads" each attending to different aspects:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

```
Multi-Head Attention:

        Q        K        V
        │        │        │
    ┌───┼───┐┌───┼───┐┌───┼───┐
    │   │   ││   │   ││   │   │
    ▼   ▼   ▼▼   ▼   ▼▼   ▼   ▼
  Head₁ (learn different projections)
    │
  Attention(Q₁, K₁, V₁)
    │
    ├─── Head₂: Attention(Q₂, K₂, V₂)
    ├─── Head₃: Attention(Q₃, K₃, V₃)
    └─── Head₈: Attention(Q₈, K₈, V₈)
    │
    ▼
  Concatenate all heads
    │
    ▼
  Linear (W^O)
    │
    ▼
  Final Output

Benefit: Different heads learn different relationships
  Head 1: syntactic dependencies
  Head 2: semantic relationships
  Head 3: positional patterns
  etc.
```

### Self-Attention

When Q, K, V all come from the **same** sequence:

```
Self-Attention on "The cat sat on the mat":

      The    cat    sat    on     the    mat
The  [0.3   0.1   0.05  0.05  0.4   0.1 ]  ← "The" attends to "the" (article)
cat  [0.05  0.4   0.2   0.05  0.05  0.25]  ← "cat" attends to itself + "mat" (subject-object)
sat  [0.1   0.5   0.2   0.05  0.05  0.1 ]  ← "sat" attends to "cat" (verb-subject)
on   [0.1   0.05  0.3   0.2   0.05  0.3 ]  ← "on" attends to "sat" + "mat" (prep)
mat  [0.05  0.2   0.1   0.1   0.05  0.5 ]  ← "mat" attends to itself + "cat"

Each word can directly attend to every other word — no distance limitation!
This is why Transformers replaced RNNs for NLP.
```

---

## 7. Complete RNN/LSTM Application: Text Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ════════════════════════════════════════════════
# Sentiment Analysis with LSTM + Attention
# ════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    """Self-attention over LSTM hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lstm_output, mask=None):
        # lstm_output: [B, S, H]
        scores = self.attention(lstm_output).squeeze(-1)  # [B, S]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = torch.softmax(scores, dim=-1)  # [B, S]
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # [B, H]
        return context, weights

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(lstm_output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, lengths=None):
        # x: [B, S] token indices
        embedded = self.embedding(x)           # [B, S, E]
        
        # Pack padded sequence for efficiency
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention pooling
        context, attn_weights = self.attention(lstm_out)
        
        # Classify
        logits = self.classifier(context)
        return logits, attn_weights

# Training example
vocab_size = 50000
embed_dim = 300
hidden_dim = 256
num_classes = 2  # Positive / Negative

model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Dummy training step
x = torch.randint(0, vocab_size, (32, 100))  # Batch of 32, sequence length 100
lengths = torch.randint(10, 100, (32,))       # Variable lengths
lengths, _ = torch.sort(lengths, descending=True)

logits, attn_weights = model(x, lengths)
loss = criterion(logits, torch.randint(0, 2, (32,)))
print(f"Loss: {loss.item():.4f}")
print(f"Attention weights shape: {attn_weights.shape}")  # [32, 100]
```

---

## 8. Real-World ML Connections

| Application | Architecture | Why |
|-------------|-------------|-----|
| Machine Translation | Seq2Seq + Attention (Transformer) | Maps variable-length input to variable-length output |
| Speech Recognition | BiLSTM + CTC | Bidirectional context for audio frames |
| Sentiment Analysis | LSTM/GRU + Attention | Captures long-range dependencies in reviews |
| Time Series Forecasting | LSTM/GRU | Sequential patterns in financial/temporal data |
| Named Entity Recognition | BiLSTM-CRF | Bidirectional context for token classification |
| Text Generation | LSTM/GRU Language Model | Autoregressive next-token prediction |
| Chatbots | Transformer (GPT, BERT) | Attention handles long contexts efficiently |
| Video Understanding | ConvLSTM / 3D-CNN + LSTM | Spatiotemporal feature learning |

### When to Use What

```
Decision Tree:

Sequential data?
  │
  ├─ Short sequences (< 20 steps)
  │    → Vanilla RNN or 1D-CNN (simpler, faster)
  │
  ├─ Medium sequences (20–200 steps)
  │    → LSTM or GRU (handle long dependencies)
  │    → GRU if you need speed/fewer params
  │    → LSTM if you need maximum expressiveness
  │
  ├─ Long sequences (> 200 steps)
  │    → Transformer (attention handles arbitrary distances)
  │    → Or: LSTM + Attention (good compromise)
  │
  └─ Need parallel training?
       → Transformer (no sequential dependency)
       → RNNs must process step by step (slow)
```

---

## 9. Exercises

### Exercise 1: RNN Hidden State Computation
Given: $\mathbf{W}_{hh} = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.3 \end{bmatrix}$, $\mathbf{W}_{xh} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.1 \end{bmatrix}$, $\mathbf{b} = [0, 0]$, $\mathbf{h}_0 = [0, 0]$. Compute $\mathbf{h}_1$ for input $\mathbf{x}_1 = [1, 0]$.

**Solution:**
$$\mathbf{h}_1 = \tanh(\mathbf{W}_{hh}\mathbf{h}_0 + \mathbf{W}_{xh}\mathbf{x}_1 + \mathbf{b})$$
$$= \tanh\left(\begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.3 \end{bmatrix}\begin{bmatrix} 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix}\right)$$
$$= \tanh\left(\begin{bmatrix} 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.3 \end{bmatrix}\right) = \tanh([0.1, 0.3]) = [0.0997, 0.2913]$$

### Exercise 2: LSTM Gate Computation
Given: $\mathbf{c}_{t-1} = [1.0, -0.5]$, $\mathbf{h}_{t-1} = [0.5, 0.3]$, $\mathbf{x}_t = [0.2, -0.1]$. Compute $\mathbf{c}_t$ assuming: $\mathbf{f}_t = [0.8, 0.9]$, $\mathbf{i}_t = [0.3, 0.4]$, $\tilde{\mathbf{c}}_t = [0.5, -0.2]$.

**Solution:**
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$= [0.8 \times 1.0, 0.9 \times (-0.5)] + [0.3 \times 0.5, 0.4 \times (-0.2)]$$
$$= [0.8, -0.45] + [0.15, -0.08]$$
$$= [0.95, -0.53]$$

### Exercise 3: Implement Self-Attention from Scratch

**Solution:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None):
        B, S, D = x.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(context)

# Test
sa = SelfAttention(embed_dim=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, embed_dim=512
output = sa(x)
print(f"Output shape: {output.shape}")  # [2, 10, 512]
```

### Exercise 4: Sentiment Classification
Build a text classification model using GRU on the IMDB dataset (or a synthetic dataset).

**Solution:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Synthetic IMDB-like data
class SentimentData(Dataset):
    def __init__(self, num_samples=1000, vocab_size=10000, max_len=200):
        self.x = torch.randint(1, vocab_size, (num_samples, max_len))
        self.y = torch.randint(0, 2, (num_samples,))
        # Set padding (0) at end of each sequence
        lengths = torch.randint(50, max_len, (num_samples,))
        for i in range(num_samples):
            self.x[i, lengths[i]:] = 0
    
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

dataset = SentimentData(num_samples=500, vocab_size=10000, max_len=100)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.embed(x)                              # [B, S, E]
        _, hidden = self.gru(x)                         # hidden: [2, B, H]
        hidden = hidden.transpose(0, 1).reshape(-1, hidden.shape[1] * hidden.shape[2] * 2)
        # Wait — let's fix the concatenation
        # Bidirectional: forward hidden and backward hidden
        return self.fc(torch.cat([hidden[-1], hidden[-2]], dim=-1))

model = GRUClassifier(10000)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

## 10. Summary

| Architecture | Key Mechanism | Best For | Key Limitation |
|-------------|---------------|----------|----------------|
| **Vanilla RNN** | Hidden state recurrence | Short sequences | Vanishing gradients (>10 steps) |
| **LSTM** | Forget/Input/Output gates + cell state | Medium-long sequences | Slower, more parameters |
| **GRU** | Update/Reset gates (simplified LSTM) | Medium sequences | Less expressive than LSTM |
| **Seq2Seq** | Encoder → context → decoder | Translation, summarization | Fixed bottleneck |
| **Attention** | Query-Key-Value weighted sum | All sequence tasks | O(n²) memory for self-attention |
| **Transformer** | Multi-head self-attention | Long sequences, NLP | Computationally expensive |

**The Evolution:**
```
RNN (1986) → LSTM (1997) → GRU (2014) → Seq2Seq + Attention (2014-15) → Transformer (2017) → GPT/BERT (2018+) → LLMs (2020+)

Each step addressed a specific limitation:
  RNN:     "Let's process sequences"
  LSTM:    "Let's remember long-term dependencies"  
  GRU:     "Let's do that more efficiently"
  Attn:    "Let's look at all encoder states, not just the last one"
  Transf:  "Let's remove recurrence entirely — parallel training!"
```
