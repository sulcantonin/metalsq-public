"""Operator rows and a label-free meta-objective: a family of 1-D Helmholtz problems

    -u'' + k^2 u = f  on [0, 2 pi],  u(0) = u(2 pi) = 0,  k ~ U[1, 10],

with f a random sum of three sinusoids. Each task is solved by one collocation
solve: the PDE rows are built from the closed-form second derivative of the basis.
The bandwidth and ridge are meta-learned from the equation residual at held-out
collocation points, so no solution is ever seen in training. The exact solution is
used only to report the error at the end (cf. App. F of the paper).

    python examples/helmholtz.py
"""
import math

import torch

from metalsq import MetaLSQ, Task

L = 2 * math.pi
X = torch.linspace(0.05, L - 0.05, 120, dtype=torch.float64).unsqueeze(-1)
X_FIT, X_VAL = X[::2], X[1::2]                     # inner-solve and held-out collocation
X_BC = torch.tensor([[0.0], [L]], dtype=torch.float64)
BC_WEIGHT = 10.0


def sample(n_tasks, gen):
    out = []
    for _ in range(n_tasks):
        k = float(torch.rand(1, generator=gen)) * 9 + 1
        amp = torch.randn(3, generator=gen, dtype=torch.float64)
        freq = (torch.rand(3, generator=gen, dtype=torch.float64) * 3 + 1).round()
        phase = torch.rand(3, generator=gen, dtype=torch.float64) * L
        out.append(Task(x=X_BC, y=torch.zeros(2, dtype=torch.float64),
                        extra=dict(k=k, amp=amp, freq=freq, phase=phase)))
    return out


def forcing(t, x):
    e = t.extra
    return (e["amp"] * torch.sin(x * e["freq"] + e["phase"])).sum(-1)


def exact(t, x):
    e, k = t.extra, t.extra["k"]
    part = lambda z: (e["amp"] * torch.sin(z * e["freq"] + e["phase"]) / (k ** 2 + e["freq"] ** 2)).sum(-1)
    ends = part(X_BC)
    x1 = x[:, 0]
    return (part(x) - ends[0] * torch.sinh(k * (L - x1)) / math.sinh(k * L)
            - ends[1] * torch.sinh(k * x1) / math.sinh(k * L))


def pde_rows(t, xc):
    """Rows of -u'' + k^2 u = f at collocation points xc, from closed-form derivatives."""
    def rows(basis, W):
        A = -basis.derivative(xc, W, (2,)) + t.extra["k"] ** 2 * basis.features(xc, W)
        return [(A, forcing(t, xc))]
    return rows


def solve(model, t, xc):
    """One collocation solve: weighted boundary rows plus PDE rows at xc."""
    def rows(basis, W):
        return [(BC_WEIGHT * basis.features(t.x, W), torch.zeros(2, dtype=torch.float64)),
                *pde_rows(t, xc)(basis, W)]
    return model.adapt(t.x[:0], t.y[:0], rows=rows)


def residual_loss(model, tasks):
    loss = 0.0
    for t in tasks:
        f = solve(model, t, X_FIT)
        r = -f.derivative(X_VAL, (2,)) + t.extra["k"] ** 2 * f(X_VAL) - forcing(t, X_VAL)
        loss = loss + r.pow(2).mean()
    return loss / len(tasks)


model = MetaLSQ(in_dim=1, n_features=200, sobolev=False, sigma=2.0, mu=1e-6, seed=0)
model.meta_fit(sample, steps=1000, batch_size=4, lr=5e-2, loss=residual_loss, log_every=250)

test = sample(50, torch.Generator().manual_seed(9999))
xs = torch.linspace(0, L, 200, dtype=torch.float64).unsqueeze(-1)
with torch.no_grad():
    err = []
    for t in test:
        f = solve(model, t, X)
        err.append((f(xs) - exact(t, xs)).pow(2).mean().sqrt())
print(f"{model}\nmean RMS error against the exact solution: {torch.stack(err).mean():.2e}")
