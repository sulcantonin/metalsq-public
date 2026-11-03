"""10-shot sinusoid regression with Meta-LSQ (the benchmark of the paper, Sec. 4.1).

    python examples/sinusoid.py            # Sobolev variant (values + derivatives)
    python examples/sinusoid.py --values   # value-only variant (two scalars)
"""
import sys
import time

import torch

from metalsq import MetaLSQ, tasks

sobolev = "--values" not in sys.argv
model = MetaLSQ(in_dim=1, n_features=200, sobolev=sobolev, seed=0)

t0 = time.time()
model.meta_fit(tasks.sinusoid(K=10, K_q=10), steps=2000, batch_size=4, lr=5e-3,
               seed=1000, log_every=500)
print(f"meta-training took {time.time() - t0:.1f} s -> {model}")

# 200 held-out tasks, 50 query points each.
test = tasks.sinusoid(K=10, K_q=50)(200, torch.Generator().manual_seed(9999))
with torch.no_grad():
    mse = torch.stack([(model.adapt(t.x, t.y, t.dy)(t.xq) - t.yq).pow(2).mean() for t in test])
print(f"test MSE: mean {mse.mean():.4f}, median {mse.median():.2e}")

# The adapted model has closed-form derivatives of every order.
t = test[0]
f = model.adapt(t.x, t.y, t.dy)
x = torch.linspace(-5, 5, 5, dtype=torch.float64).unsqueeze(-1)
with torch.no_grad():
    print("u   ", f(x).numpy().round(3))
    print("u'  ", f.derivative(x, (1,)).numpy().round(3))
    print("u'''", f.derivative(x, (3,)).numpy().round(3))
