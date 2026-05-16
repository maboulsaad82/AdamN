# AdamN Optimizer

> The Next Evolution of Gradient Optimization.
> AdamN optimizer with nested momentum and exact bias correction. Designed for optimal stability in AMP, bf16, and fp16 training environments.

## Installation

You can install the pre-compiled binary package directly via pip:

```bash
# Install securely from:
!pip install "https://github.com/maboulsaad82/AdmaN/raw/main/adamn_optimizer-0.1.0-cp312-cp312-linux_x86_64.whl"
```

## Usage in Google Colab

```python
import torch
from adamn_optimizer import AdamN

model = torch.nn.Linear(10, 2)
optimizer = AdamN(model.parameters(), lr=1e-3, betas=(0.9, 0.1, 0.999))

# Training loop
optimizer.zero_grad()
loss = model(torch.randn(1, 10)).sum()
loss.backward()
optimizer.step()
```

## License and Patents
This software is provided under a proprietary license. 
The methods, algorithms, and techniques implemented in this software are currently the subject of a pending provisional patent application. Source code distribution is strictly prohibited.
