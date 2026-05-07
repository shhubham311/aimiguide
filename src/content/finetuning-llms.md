# Fine-Tuning LLMs & Evaluations: From Pre-Training to Production

## Why Fine-Tuning Matters

Pre-trained LLMs are powerful generalists, but they often fall short on domain-specific tasks. A medical assistant must accurately answer clinical questions; a legal AI must cite case law precisely; a coding assistant must follow enterprise style guides. **Fine-tuning** adapts a pre-trained model to specific domains, tasks, and behavioral constraints—transforming a general-purpose LLM into a specialized expert.

This lesson covers the full fine-tuning pipeline: parameter-efficient methods, RLHF alignment, evaluation frameworks, and practical deployment considerations.

---

## The Fine-Tuning Landscape

```
                        Compute Cost
                        Low ◄──────────────► High
                    ┌──────────────────────────┐
     Low Params ◄  │  LoRA / QLoRA             │
     High Params   │  Adapter / Prefix Tuning  │
                    │  Full Fine-Tuning         │
                    │  RLHF / DPO               │
                    └──────────────────────────┘
```

---

## 1. Full Fine-Tuning

The simplest approach: update all parameters of the pre-trained model.

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=100,
    save_strategy="epoch",
    fp16=True,  # Mixed precision for memory efficiency
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

**Problem**: Fine-tuning a 7B parameter model requires ~28GB GPU memory (4 bytes × 7B). This is impractical for most developers.

---

## 2. LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) freezes the pre-trained weights and injects small trainable **rank decomposition matrices**:

```
Original:  h = W_0 × x           (W_0 is frozen: d × d)

LoRA:      h = W_0 × x + ΔW × x
              = W_0 × x + (B × A) × x

Where:
  A ∈ R^(r × d)    (initialized with Kaiming uniform)
  B ∈ R^(d × r)    (initialized with zeros)
  r << d           (rank, typically 4-64)

                   ┌──────┐
              x ───►│  A   │──┐
                   │ r×d  │  │    ┌──────┐
                   └──────┘  ├───►│  B   │───► ΔW·x  (trainable)
                             │    │ d×r  │
         ────────────────────┘    └──────┘

         x ─────────────► W_0 ──────────────► W_0·x  (frozen)

         Total: h = W_0·x + B·A·x
```

**Memory savings**: For d=4096, r=16:
- Full fine-tuning: 4096 × 4096 = 16.7M params per matrix
- LoRA: 4096 × 16 + 16 × 4096 = 131K params (128× reduction!)

### LoRA Implementation

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        # Freeze original weights
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out


# Apply LoRA to a model
def apply_lora(model, target_modules=["q_proj", "v_proj"], rank=8):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent = model
                for attr in name.split('.'):
                    parent = getattr(parent, attr) if attr != name.split('.')[-1] else parent
                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank)
                # Set on parent (simplified)
    return model
```

### Using PEFT Library

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 13,107,200 || all params: 6,738,415,616 || trainable%: 0.19%
```

---

## 3. QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) adds **4-bit quantization** to LoRA:

```
┌─────────────────────────────────────────────────┐
│ Base Model (Frozen, 4-bit NF4 quantization)     │
│ ┌─────────────────────────────────────────────┐ │
│ │ W_0 stored in 4-bit NormalFloat             │ │
│ │ Dequantized → 16-bit during forward pass    │ │
│ └─────────────────────────────────────────────┘ │
│                                                  │
│ LoRA Adapters (Trainable, 16-bit BF16)          │
│ ┌──────┐   ┌──────┐                             │
│ │  A   │──►│  B   │ (rank-16 matrices)          │
│ └──────┘   └──────┘                             │
└─────────────────────────────────────────────────┘
```

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",  # Automatic GPU placement
)
```

This enables fine-tuning a 7B model on a **single 24GB GPU**!

---

## 4. RLHF: Reinforcement Learning from Human Feedback

RLHF is the technique that made ChatGPT behave like a helpful assistant rather than a raw text predictor.

### Step-by-Step Pipeline

```
Step 1: Supervised Fine-Tuning (SFT)
┌────────────────────────────┐
│ Prompt: "Explain quantum..."│
│ Human-written response:    │
│ "Quantum computing uses..." │
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  SFT Model (fine-tuned)    │
└────────────┬───────────────┘
             │
Step 2: Train Reward Model  │
             ▼
┌────────────────────────────────────┐
│ Prompt + Response A → Human rates │
│ Prompt + Response B → Human rates │
│                                    │
│ Train a model to predict ratings  │
│ → Reward Model R(prompt, response)│
└────────────┬───────────────────────┘
             │
Step 3: PPO Optimization      │
             ▼
┌────────────────────────────────────┐
│ SFT Model generates response       │
│ Reward Model scores it             │
│ PPO updates SFT Model to maximize  │
│ reward + KL penalty (don't stray   │
│ too far from SFT model)            │
└────────────────────────────────────┘
```

### PPO Objective

```
maximize E[R(s,a) - β × KL(π_θ || π_ref)]

Where:
  R(s,a)     = Reward model score
  β          = KL penalty coefficient (prevents reward hacking)
  π_θ        = Current policy (model being trained)
  π_ref      = Reference policy (SFT model)
  KL         = Kullback-Leibler divergence
```

### DPO: Direct Preference Optimization

DPO simplifies RLHF by eliminating the reward model:

```
L_DPO = -E[log σ(β × (log π_θ(y_w|x) - log π_ref(y_w|x)
                       - log π_θ(y_l|x) + log π_ref(y_l|x)))]

Where:
  y_w = Preferred (winning) response
  y_l = Dispreferred (losing) response
  σ   = Sigmoid function
```

---

## Evaluation Frameworks

### Traditional NLP Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Perplexity** | PPL = exp(-1/N × Σ log P(x_i)) | Language model quality |
| **BLEU** | BLEU = BP × exp(Σ w_n × log p_n) | Translation quality |
| **ROUGE** | ROUGE-N = overlap_n / count_n | Summarization quality |
| **Accuracy** | correct / total | Classification tasks |

### LLM-Specific Evaluation

```python
# Using the lm-evaluation-harness library
# pip install lm-eval

import lm_eval
from lm_eval.models.huggingface import HFLM

model = HFLM.from_pretrained("your-finetuned-model")

results = lm_eval.simple_evaluate(
    model=model,
    tasks=["hellaswag", "arc_challenge", "mmlu", "truthfulqa"],
    num_fewshot=5,
    batch_size=8,
)

print(results["results"])
```

### Key Benchmarks

```
┌──────────────────┬──────────────────────────────────────┐
│ Benchmark        │ What It Tests                        │
├──────────────────┼──────────────────────────────────────┤
│ MMLU             │ 57 subjects, multiple choice         │
│ HumanEval        │ Python code generation               │
│ GSM8K            │ Math word problems                   │
│ TruthfulQA       │ Factual accuracy vs common myths     │
│ MT-Bench         │ Multi-turn conversation quality      │
│ AlpacaEval       │ Human-preference alignment           │
└──────────────────┴──────────────────────────────────────┘
```

### Custom Evaluation with LLM-as-Judge

```python
def llm_judge(prompt, response, criteria="helpfulness"):
    judge_prompt = f"""
    Rate the following response on a scale of 1-5 for {criteria}.

    Prompt: {prompt}
    Response: {response}

    Score (1-5) and brief explanation:
    """
    # Send to a stronger model (e.g., GPT-4) as judge
    return score, explanation
```

---

## Practical Fine-Tuning Pipeline

```python
# Complete example using TRL + PEFT + BitsAndBytes
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 2. Format instructions
def format_instruction(sample):
    return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"

# 3. Train with QLoRA
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./lora-results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
    ),
    formatting_func=format_instruction,
)

trainer.train()

# 4. Save adapter
trainer.save_model("./my-lora-adapter")
```

---

## Exercises

### Exercise 1: LoRA Parameter Count
For a model with d_model=4096, applying LoRA with rank r=8 to 4 attention matrices (q, k, v, o) per layer across 32 layers, how many trainable parameters are there?

**Solution:**
```
Per matrix: A = r × d = 8 × 4096 = 32,768
            B = d × r = 4096 × 8 = 32,768
            Total per matrix = 65,536

Per layer: 4 matrices × 65,536 = 262,144

Total: 32 layers × 262,144 = 8,388,608 parameters ≈ 8.4M

As % of 7B model: 8.4M / 7B ≈ 0.12%
```

### Exercise 2: Compute KL Divergence
Given π_ref = [0.4, 0.3, 0.3] and π_θ = [0.5, 0.3, 0.2], compute KL(π_θ || π_ref).

**Solution:**
```
KL = Σ π_θ(i) × log(π_θ(i) / π_ref(i))

= 0.5 × log(0.5/0.4) + 0.3 × log(0.3/0.3) + 0.2 × log(0.2/0.3)
= 0.5 × log(1.25) + 0.3 × log(1.0) + 0.2 × log(0.667)
= 0.5 × 0.2231 + 0 + 0.2 × (-0.4055)
= 0.1116 - 0.0811
= 0.0305
```

### Exercise 3: Design a Fine-Tuning Strategy
You need to fine-tune a 70B model for medical Q&A on a single A100 (80GB). What approach would you use?

**Solution:** Use QLoRA with:
- 4-bit NF4 quantization (~35GB for weights)
- LoRA rank r=16 targeting attention matrices
- Gradient checkpointing to reduce activation memory
- Batch size 1 with gradient accumulation steps=16
- This fits comfortably within 80GB while maintaining quality

---

## Key Takeaways

1. **LoRA/QLoRA** make fine-tuning accessible by reducing trainable parameters to <1% of the model
2. **RLHF** aligns models with human preferences through reward modeling + PPO optimization
3. **DPO** provides a simpler alternative to RLHF by directly optimizing preferences
4. **Evaluation** requires both automated benchmarks and human/AI judge assessments
5. **Always compare** fine-tuned vs. base model to ensure genuine improvement
