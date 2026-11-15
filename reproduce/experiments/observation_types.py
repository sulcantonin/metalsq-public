"""Minimal ablation: mixing observation types in one closed-form solve, on a family the
sine basis does not contain.

Tasks solve the Airy-type equation u'' = k x u on [-4, 2], with k ~ U[0.5, 2]:
u(x) = a Ai(s x) + b Bi(s x), s = k^(1/3), a, b ~ U[-1, 1]. The functions oscillate on
the left and grow or decay exponentially on the right, so the frozen sine basis only
approximates them. Per task k is known (the equation), a and b are not. Over the basis
each kind of linear observation is a row of the same system:

    value      u(x_i)                       sin(W x_i + b) / sqrt(N)
    slope      u'(x_i)                      W cos(W x_i + b) / sqrt(N)
    integral   int_{-4}^{x_i} u             (x_i + 4) sinc(W (x_i + 4) / 2)
                                            * sin(W (x_i - 4) / 2 + b) / sqrt(N)
    equation   u''(z_j) - k z_j u(z_j) = 0  -(W^2 + k z_j) sin(W z_j + b) / sqrt(N)

Reference values come from scipy's Airy functions (values and slopes) and fine-grid
quadrature (integrals). Each variant meta-learns the bandwidth, the ridge and one
weight per extra row type through the solve, on the query MSE of the values, and is
scored by the RMS error on a 200-point grid relative to the task's spread, on 200
held-out tasks per seed, 3 seeds.

    python3 experiments/observation_types.py --save results/observation_types.json
"""
import json
import math
import sys

import numpy as np
import torch
from scipy.special import airy

torch.set_default_dtype(torch.float64)
N, X0, X1, N_EQ = 200, -4.0, 2.0, 16
GRID = torch.linspace(X0, X1, 200)
Z_EQ = torch.linspace(X0, X1, N_EQ)
FINE = np.linspace(X0, X1, 6001)

VARIANTS = {                                     # (number of values, extra row types)
    "4 values": (4, ()),
    "4 values + slopes": (4, ("slope",)),
    "4 values + integrals": (4, ("integral",)),
    "4 values + equation": (4, ("equation",)),
    "2 values": (2, ()),
    "2 values + equation": (2, ("equation",)),
    "4 values + slopes + integrals + equation": (4, ("slope", "integral", "equation")),
}


class AiryTask:
    def __init__(self, k, a, b):
        self.k, self.a, self.b, self.s = k, a, b, k ** (1 / 3)
        u = self._u(FINE)
        cum = np.concatenate([[0.0], np.cumsum((u[1:] + u[:-1]) / 2 * np.diff(FINE))])
        self._cum = cum

    def _u(self, x):
        ai, aip, bi, bip = airy(self.s * np.asarray(x))
        return self.a * ai + self.b * bi

    def value(self, x):
        return torch.tensor(self._u(x.numpy()))

    def slope(self, x):
        ai, aip, bi, bip = airy(self.s * x.numpy())
        return torch.tensor(self.s * (self.a * aip + self.b * bip))

    def integral(self, x):
        return torch.tensor(np.interp(x.numpy(), FINE, self._cum))


def sample_tasks(n, gen):
    u = lambda lo, hi: (lo + (hi - lo) * torch.rand(n, generator=gen)).tolist()
    return [AiryTask(k, a, b) for k, a, b in zip(u(0.5, 2.0), u(-1, 1), u(-1, 1))]


def rows(kind, W, b, x, k):
    z = x.unsqueeze(-1) * W + b
    if kind == "value":
        return torch.sin(z) / math.sqrt(N)
    if kind == "slope":
        return W * torch.cos(z) / math.sqrt(N)
    if kind == "integral":
        h = W * (x.unsqueeze(-1) - X0) / 2
        sinc = torch.where(h.abs() < 1e-8, torch.ones_like(h), torch.sin(h) / h)
        return ((x.unsqueeze(-1) - X0) * sinc * torch.sin(W * (x.unsqueeze(-1) + X0) / 2 + b)
                / math.sqrt(N))
    if kind == "equation":
        return -(W ** 2 + k * x.unsqueeze(-1)) * torch.sin(z) / math.sqrt(N)
    raise ValueError(kind)


class Model(torch.nn.Module):
    def __init__(self, kinds, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_hat = torch.randn(N, generator=g)
        self.b = torch.rand(N, generator=g) * 2 * math.pi
        self.kinds = kinds
        self.log_sigma = torch.nn.Parameter(torch.tensor(0.0))
        self.log_mu = torch.nn.Parameter(torch.tensor(math.log(1e-4)))
        self.log_w = torch.nn.Parameter(torch.zeros(len(kinds)))

    def fit(self, task, x):
        """One ridge solve over all the observation rows of one task."""
        W = self.log_sigma.exp() * self.W_hat
        A, t = [rows("value", W, self.b, x, task.k)], [task.value(x)]
        for j, kind in enumerate(self.kinds):
            sw = self.log_w[j].exp().sqrt()
            if kind == "equation":
                A.append(sw * rows(kind, W, self.b, Z_EQ, task.k)); t.append(torch.zeros(N_EQ))
            else:
                A.append(sw * rows(kind, W, self.b, x, task.k))
                t.append(sw * (task.slope(x) if kind == "slope" else task.integral(x)))
        A, t = torch.cat(A), torch.cat(t)
        K = A @ A.T + self.log_mu.exp() * torch.eye(len(t))
        beta = A.T @ torch.linalg.solve(K, t)
        return lambda xx: rows("value", W, self.b, xx, task.k) @ beta


def support(n_values, gen):
    return X0 + (X1 - X0) * torch.rand(n_values, generator=gen)


def meta_train(n_values, kinds, seed, steps=1500, batch=8, lr=2e-2):
    model = Model(kinds, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 1000)
    for _ in range(steps):
        tasks = sample_tasks(batch, gen)
        loss = 0.0
        for task in tasks:
            x, xq = support(n_values, gen), support(50, gen)
            loss = loss + (model.fit(task, x)(xq) - task.value(xq)).pow(2).mean()
        opt.zero_grad(); (loss / batch).backward(); opt.step()
    return model


def evaluate(model, n_values, seed, n_test=200):
    gen = torch.Generator().manual_seed(seed + 9999)
    errs = []
    with torch.no_grad():
        for task in sample_tasks(n_test, gen):
            x = support(n_values, gen)
            u = task.value(GRID)
            errs.append(float(((model.fit(task, x)(GRID) - u).pow(2).mean() / u.var()).sqrt()))
    return torch.tensor(errs)


def main(save=None, seeds=(0, 1, 2)):
    out = {}
    for name, (n_values, kinds) in VARIANTS.items():
        per_seed = []
        for seed in seeds:
            model = meta_train(n_values, kinds, seed)
            e = evaluate(model, n_values, seed)
            per_seed.append({"median": float(e.median()), "mean": float(e.mean()),
                             "sigma": model.log_sigma.exp().item(), "mu": model.log_mu.exp().item(),
                             "w": model.log_w.exp().tolist()})
        med = sorted(p["median"] for p in per_seed)
        out[name] = {"n_values": n_values, "rows": list(kinds), "per_seed": per_seed,
                     "median_of_seed_medians": med[len(med) // 2]}
        print(f"{name:<44s} median relative RMS error per seed: "
              + ", ".join(f"{p['median']:.3g}" for p in per_seed), flush=True)
    if save:
        with open(save, "w") as f:
            json.dump({"family": "u'' = k x u on [-4, 2] (Airy)", "variants": out}, f, indent=1)
        print(f"[saved] {save}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None)
