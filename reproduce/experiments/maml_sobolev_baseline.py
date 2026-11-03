"""Information-matched baseline: MLP-MAML that ALSO sees the support gradients.

Meta-LSQ-Sobolev adapts from (x, y, dy/dx) while the MLP-MAML baseline of
``maml_sinusoid.py`` adapts from (x, y) only, so the headline sinusoid
comparison is not information-matched.  This script closes that gap: the MLP's
INNER loss becomes the Sobolev loss of Czarnecki et al.,

    L^s(theta) = mean_i (f(x_i) - y_i)^2 + lam * mean_i (df/dx(x_i) - g_i)^2,

with df/dx obtained by autograd, so the inner step differentiates through a
backward pass and the outer step differentiates through that again (double
backpropagation).  That cost asymmetry -- one extra stacked row block for
Meta-LSQ vs a doubled graph for the MLP -- is exactly the paper's claim, and it
is measured here on the meta-learning benchmark rather than only on the
per-task GMM fits.

A smooth activation is used (SiLU by default) because a ReLU network's
input-gradient is piecewise constant, which makes derivative matching
degenerate; ``--act relu`` reproduces the standard baseline's architecture for
reference.  lam = 0 is the control (same architecture, values only), so the
lam > 0 rows isolate the effect of the extra information rather than of the
activation change.

Run:  OMP_NUM_THREADS=1 python3 experiments/maml_sobolev_baseline.py \
          --save results/maml_sobolev_baseline.json
"""

import argparse
import json
import math
import time

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from experiments.maml_sinusoid import sample_tasks

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

ACTS = {"relu": torch.relu, "silu": torch.nn.functional.silu, "tanh": torch.tanh}


class SmoothMLP(nn.Module):
    """Same shape as ``maml_sinusoid.MAMLNet`` but with a selectable activation
    and a functional forward that keeps the graph for double backpropagation."""

    def __init__(self, hidden=40, depth=3, act="silu", in_dim=1, out_dim=1):
        super().__init__()
        self.depth = depth
        self.act = ACTS[act]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, out_dim))

    def forward(self, x, params):
        h = x
        for i in range(self.depth - 1):
            h = self.act(torch.nn.functional.linear(h, params[f"w{i+1}"], params[f"b{i+1}"]))
        return torch.nn.functional.linear(h, params[f"w{self.depth}"],
                                          params[f"b{self.depth}"]).squeeze(-1)

    def init_params(self):
        return {**{f"w{i+1}": l.weight for i, l in enumerate(self.layers)},
                **{f"b{i+1}": l.bias for i, l in enumerate(self.layers)}}


def _sobolev_support_loss(net, params, xs, ys, dys, lam, create_graph):
    """Value MSE + lam * input-gradient MSE on the support set.

    ``xs`` must require grad for the derivative term; the resulting loss is a
    function of ``params`` through a backward pass, so taking its parameter
    gradient with create_graph=True is the double-backpropagation the paper
    contrasts against the stacked Sobolev solve."""
    if lam <= 0.0:
        return (net(xs, params) - ys).pow(2).mean()
    xg = xs.detach().requires_grad_(True)
    pred = net(xg, params)
    # create_graph here so d(pred)/dx stays differentiable in params.
    (dpred,) = torch.autograd.grad(pred.sum(), xg, create_graph=True)
    val = (pred - ys).pow(2).mean()
    grad = (dpred.squeeze(-1) - dys).pow(2).mean()
    return val + lam * grad


def _one_step(net, opt, gen, lam, meta_bs, K, K_q, inner_lr, inner_steps):
    xs, ys, dys, xq, yq = sample_tasks(meta_bs, K, K_q, gen)
    opt.zero_grad()
    meta_loss = 0.0
    for t in range(meta_bs):
        params = net.init_params()
        for _ in range(inner_steps):
            loss_s = _sobolev_support_loss(net, params, xs[t], ys[t], dys[t],
                                           lam, create_graph=True)
            grads = torch.autograd.grad(loss_s, list(params.values()),
                                        create_graph=True)
            params = {k: v - inner_lr * g
                      for (k, v), g in zip(params.items(), grads)}
        meta_loss = meta_loss + (net(xq[t], params) - yq[t]).pow(2).mean()
    meta_loss = meta_loss / meta_bs
    meta_loss.backward()
    opt.step()
    return meta_loss


def train(n_iters, lam, act, seed, meta_bs=4, K=10, K_q=10, inner_lr=1e-2,
          inner_steps=1, outer_lr=1e-3, hidden=40, depth=3, log_every=200,
          warmup=25):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    net = SmoothMLP(hidden=hidden, depth=depth, act=act).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=outer_lr)
    # Warm-up on a throwaway net/optimiser: the first autograd calls in a
    # process pay one-off lazy-init costs that would otherwise be charged to
    # whichever configuration happens to run first.
    wnet = SmoothMLP(hidden=hidden, depth=depth, act=act).to(device)
    wopt = torch.optim.Adam(wnet.parameters(), lr=outer_lr)
    wgen = torch.Generator().manual_seed(12345)
    for _ in range(warmup):
        _one_step(wnet, wopt, wgen, lam, meta_bs, K, K_q, inner_lr, inner_steps)
    curve = []
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, dys, xq, yq = sample_tasks(meta_bs, K, K_q, gen)
        opt.zero_grad()
        meta_loss = 0.0
        for t in range(meta_bs):
            params = net.init_params()
            for _ in range(inner_steps):
                loss_s = _sobolev_support_loss(net, params, xs[t], ys[t], dys[t],
                                               lam, create_graph=True)
                grads = torch.autograd.grad(loss_s, list(params.values()),
                                            create_graph=True)
                params = {k: v - inner_lr * g
                          for (k, v), g in zip(params.items(), grads)}
            meta_loss = meta_loss + (net(xq[t], params) - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            curve.append((time.time() - t0, it, float(meta_loss.item())))
    return net, time.time() - t0, curve


def evaluate(net, lam, seed, n_test=200, K=10, K_q=50, inner_lr=1e-2, inner_steps=1):
    """Adaptation at test time uses the same support information as training."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)
    losses = []
    t0 = time.time()
    for t in range(n_test):
        params = net.init_params()
        for _ in range(inner_steps):
            loss_s = _sobolev_support_loss(net, params, xs[t], ys[t], dys[t],
                                           lam, create_graph=False)
            grads = torch.autograd.grad(loss_s, list(params.values()),
                                        create_graph=True)
            params = {k: v - inner_lr * g for (k, v), g in zip(params.items(), grads)}
        with torch.no_grad():
            losses.append((net(xq[t], params) - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean()), (time.time() - t0) / n_test * 1e3


def _stats(v):
    """Mean and *sample* std (n-1), matching experiments/run_campaign.py so the
    error bars in the paper are all the same estimator."""
    t = torch.tensor(v, dtype=torch.float64)
    std = float(t.std(unbiased=True)) if t.numel() > 1 else 0.0
    return {"mean": float(t.mean()), "std": std, "seeds": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--act", default="silu", choices=sorted(ACTS))
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    out = {"config": {"iters": a.iters, "seeds": a.seeds, "lams": a.lams,
                      "act": a.act, "n_test": a.n_test, "hidden": 40, "depth": 3,
                      "inner_steps": 1, "inner_lr": 1e-2, "outer_lr": 1e-3,
                      "K": 10, "K_q_train": 10, "K_q_test": 50},
           "per_lam": []}
    for lam in a.lams:
        mses, times, infs = [], [], []
        for s in a.seeds:
            net, tt, _ = train(a.iters, lam, a.act, s)
            mse, inf = evaluate(net, lam, s, n_test=a.n_test)
            mses.append(mse); times.append(tt); infs.append(inf)
            print(f"[lam={lam:g} act={a.act} seed={s}] test MSE {mse:.4f}  "
                  f"train {tt:.1f}s  inf {inf:.2f} ms/task", flush=True)
        rec = {"lam": lam, "act": a.act, "test_mse": _stats(mses),
               "train_s": _stats(times), "inf_ms": _stats(infs)}
        out["per_lam"].append(rec)
        print(f"== lam={lam:g}: MSE {rec['test_mse']['mean']:.4f} "
              f"+/- {rec['test_mse']['std']:.4f}, train "
              f"{rec['train_s']['mean']:.1f}s", flush=True)

    base = next((r for r in out["per_lam"] if r["lam"] == 0.0), None)
    if base is not None:
        for r in out["per_lam"]:
            if r["lam"] > 0:
                r["train_s_over_valueonly"] = (r["train_s"]["mean"]
                                               / base["train_s"]["mean"])
    if a.save:
        path = a.save if _os.path.isabs(a.save) else _os.path.join(_REPO_ROOT, a.save)
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
