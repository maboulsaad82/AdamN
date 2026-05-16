# AdamN Optimizer

> The Next Evolution of Gradient Optimization.  
> AdamN optimizer with nested momentum and exact bias correction. Designed for optimal stability in AMP, bf16, and fp16 training.

---

## About AdamN

AdamN is a deep learning optimizer designed to improve early-training stability and accelerate convergence. Instead of relying on a single momentum estimate as in Adam/AdamW, AdamN uses a nested momentum structure:

```text
g_t → v_t → a_t
```

The gradient is first smoothed into `v_t`, then the momentum itself is smoothed again into `a_t`. This gives AdamN a more stable and mature update direction, especially during the first training steps where many deep learning pipelines usually require warmup.

A key part of AdamN is its **exact bias correction** for the nested momentum. Since applying an Exponential Moving Average (EMA) over another EMA introduces compounded cold-start bias, the usual Adam-style correction is not enough. AdamN corrects both smoothing stages together, helping the optimizer start more reliably without relying on hand-tuned warmup schedules.

AdamN keeps the practical spirit of AdamW: it uses an adaptive denominator, supports decoupled weight decay, and can be used as a drop-in optimizer in PyTorch training loops. The goal is not to make optimization more complicated, but to provide a smoother and more stable update direction that can reach useful performance faster in vision, language modeling, and fine-tuning workloads.

Across the reported CNN and ViT experiments, AdamN consistently accelerated early-to-mid training progress, reaching the `40%–80%` accuracy milestones sooner than AdamW and other baselines in many regimes.
---

## Core Idea

Adam/AdamW uses one first-moment estimate:

```text
g_t → m_t
```

AdamN uses nested momentum:

```text
g_t → v_t → a_t
```

where:

```text
g_t  = current gradient
v_t  = first momentum estimate
a_t  = nested momentum estimate
s_t  = adaptive denominator estimate
```

The update direction is based on the bias-corrected nested momentum `a_t`, while the denominator follows the Adam-style adaptive scaling.

---

## Mathematical Form

```text
v_t = β1 v_{t-1} + (1 - β1) g_t
```

```text
a_t = β2 a_{t-1} + (1 - β2) v_t
```

```text
s_t = β3 s_{t-1} + (1 - β3) g_t^2
```

Then AdamN applies exact bias correction to the nested momentum and updates the parameters using:

```text
θ_t = θ_{t-1} - η · â_t / (sqrt(ŝ_t) + ε)
```

where:

```text
â_t = exact bias-corrected nested momentum
ŝ_t = bias-corrected adaptive denominator
η   = learning rate
ε   = numerical stabilizer
```

---

## Why AdamN?

AdamN was designed to address an important practical issue in deep learning training: the early update direction can be noisy, unstable, or mis-scaled, especially at the beginning of training.

AdamN helps by:

- smoothing the gradient direction twice through nested momentum,
- applying exact bias correction for the double-EMA structure,
- reducing the need for hand-tuned warmup schedules,
- keeping AdamW-style adaptive scaling and decoupled weight decay,
- providing a more stable update direction during early training.

In simple terms, AdamN tries to make the optimizer remember the learning direction more carefully before taking each step.

---

## Installation

```bash
pip install "https://github.com/maboulsaad82/AdmaN/raw/main/adamn_optimizer-0.1.0-cp312-cp312-linux_x86_64.whl"
```

For Google Colab or Jupyter Notebook:

```python
!pip install "https://github.com/maboulsaad82/AdmaN/raw/main/adamn_optimizer-0.1.0-cp312-cp312-linux_x86_64.whl"
```

> Note: this wheel is built for Python 3.12 on Linux x86_64.

You can check your Python version with:

```python
!python --version
```

---

## Quick Start

```python
import torch
import torch.nn as nn
from adamn_optimizer import AdamN
```

```python
model = nn.Linear(10, 1)

optimizer = AdamN(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.1, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
    decoupled_wd=True,
    bias_correction="exact",
)
```

```python
x = torch.randn(32, 10)
y = torch.randn(32, 1)

criterion = nn.MSELoss()

optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()

print("Loss:", loss.item())
```

---

## Example Training Loop

```python
import torch
import torch.nn as nn
from adamn_optimizer import AdamN

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

criterion = nn.CrossEntropyLoss()

optimizer = AdamN(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.1, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
    decoupled_wd=True,
    bias_correction="exact",
)

for epoch in range(10):
    model.train()

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")
```

---

## Recommended Defaults

```python
optimizer = AdamN(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.1, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
    decoupled_wd=True,
    bias_correction="exact",
)
```

Recommended starting values:

```text
β1 = 0.9
β2 = 0.1
β3 = 0.999
eps = 1e-8
weight_decay = 1e-4
bias_correction = "exact"
```

For different training regimes and task-specific hyperparameters, please refer to **Table 5** in the AdamN paper.

---
## AdamN Hyperparameters Across Training Regimes

The following table summarizes the AdamN settings used across different training regimes. For the complete experimental setup, please refer to **Table 5** in the AdamN paper.

| Hyperparameter | CNN Full Training | CNN Transfer Learning | ViT Full Training | ViT Transfer Learning | NLP |
|---|---:|---:|---:|---:|---:|
| Learning rate, `η` | `1e-3` | `1e-4` | `1e-3` | `1e-4` | `1e-3` |
| Betas, `(β1, β2, β3)` | `(0.9, 0.1, 0.999)` | `(0.9, 0.1, 0.999)` | `(0.9, 0.1, 0.999)` | `(0.9, 0.1, 0.999)` | `(0.4, 0.1, 0.999)` |
| Epsilon, `ε` | `1e-8` | `1e-8` | `1e-8` | `1e-8` | `1e-8` |
| Weight decay, `λ` | `1e-4` | `1e-5` | `5e-3` | `1e-4` | `1e-4` |
| Batch size | `128` | `128` | `256` | `256` | `-` |


**Notes**

- `η`: base learning rate.
- `β1`: momentum of the gradient estimate `g_t`.
- `β2`: momentum of the nested direction estimate `v_t`.
- `β3`: denominator memory.
- `ε`: numerical stability term.
- `λ`: decoupled weight decay.
- AdamN was used without warmup and with cosine scheduling in the reported experiments.

  ---

## AdamN vs AdamW

```text
AdamW:
    g_t → m_t
    single momentum estimate + adaptive denominator

AdamN:
    g_t → v_t → a_t
    nested momentum estimate + exact double-EMA bias correction + adaptive denominator
```

AdamN uses one additional optimizer state tensor compared with AdamW. This increases optimizer-state memory, but in return it provides a smoother and more stable update direction.

---

## Publication

The AdamN work has been published in the journal article:

**AdamN: Accelerating Deep Learning Training via Nested Momentum and Exact Bias Handling**

Published in **Electronics**.

---

## Patent Status

A provisional patent application related to AdamN has been filed with the United States Patent and Trademark Office (USPTO) under the title:

**Nested Double-Smoothing Optimizer For Training Neural Networks**

Application No. **63/942,313**.

---

## Citation

```bibtex
@article{aboulsaad2026adamn,
  title={AdamN: Accelerating Deep Learning Training via Nested Momentum and Exact Bias Handling},
  author={Aboulsaad, Mohamed and Shaout, Adnan},
  journal={Electronics},
  year={2026}
}
```

---

## License

Please refer to the repository license file for usage terms.

