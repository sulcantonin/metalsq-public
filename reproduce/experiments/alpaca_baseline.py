"""Experiment #2.1: ALPaCA baseline (Harrison et al. 2018) vs Meta-LSQ-Sobolev.

ALPaCA = closed-form Bayesian linear regression in a LEARNED feature space.
A small MLP encoder phi_theta(x) maps x -> R^p (its last hidden layer is the
feature map). Per task, we solve ridge/posterior in that feature space:

    beta = (Phi_s^T Phi_s + lambda I)^{-1} Phi_s^T y_s,   y_q = Phi_q beta

The encoder weights theta AND the prior precision lambda are META-LEARNED
end-to-end by Adam on the query value MSE (differentiable through the solve,
exactly like Meta-LSQ's outer loop). Same task family as maml_sinusoid.

KEY CONTRAST:
  - ALPaCA learns the BASIS (encoder, ~1.8k params) -> it can fit the family
    by warping features.
  - Meta-LSQ-Sobolev FREEZES a random Fourier basis and learns 3 scalars
    (sigma, mu, grad_w).

We report in-class (sinusoid) test MSE, OOC (tanh, meta-trained on sinusoids)
test MSE, and parameter counts, honestly. ALPaCA may match/beat in-class since
it has a learned basis.

Self-contained; imports maml_sinusoid READ-ONLY for the task generator and the
Meta-LSQ-Sobolev reference numbers.
"""

import sys, os, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch
import torch.nn as nn

from experiments.maml_sinusoid import (
    sample_tasks,
    train_fastlsq_meta,
    eval_fastlsq_meta,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")


# ---------------- OOC (tanh) task generator ----------------

def sample_tanh_tasks(n_tasks, K, K_q, gen):
    """Out-of-class family: y = a * tanh(s * (x - c)).

    a ~ U[0.1, 5] (matches sinusoid amplitude range), s ~ U[0.5, 2] (slope),
    c ~ U[-2, 2] (offset), x in [-5, 5]. Returns the same tuple shape as
    sample_tasks (dy unused here but kept for a consistent signature)."""
    a = torch.rand(n_tasks, 1, generator=gen) * 4.9 + 0.1     # [0.1, 5]
    s = torch.rand(n_tasks, 1, generator=gen) * 1.5 + 0.5     # [0.5, 2]
    c = torch.rand(n_tasks, 1, generator=gen) * 4.0 - 2.0     # [-2, 2]
    xs = torch.rand(n_tasks, K, generator=gen) * 10 - 5
    xq = torch.rand(n_tasks, K_q, generator=gen) * 10 - 5
    ys = a * torch.tanh(s * (xs - c))
    yq = a * torch.tanh(s * (xq - c))
    dys = a * s * (1 - torch.tanh(s * (xs - c)) ** 2)
    return xs.unsqueeze(-1), ys, dys, xq.unsqueeze(-1), yq


# ---------------- ALPaCA encoder + closed-form solve ----------------

class ALPaCAEncoder(nn.Module):
    """phi_theta: x in R^1 -> features in R^p. Two hidden layers (40 units),
    last hidden layer (dim p) is the learned feature map. A bias column is
    appended at solve time (so the basis can represent a per-task offset)."""

    def __init__(self, hidden=40, p=40, in_dim=1):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, p)
        self.p = p

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        h = torch.tanh(self.l2(h))          # (..., p) feature map
        return h


def alpaca_solve(Phi_s, ys, lam):
    """Ridge / posterior-mean solve in feature space.

    Phi_s: (K, p+1) support features (bias column appended).
    ys:    (K,)  support targets.
    lam:   scalar prior precision (lambda I regularizer).
    Returns beta (p+1,)."""
    p1 = Phi_s.shape[1]
    A = Phi_s.T @ Phi_s + lam * torch.eye(p1, device=Phi_s.device, dtype=Phi_s.dtype)
    b = Phi_s.T @ ys
    beta = torch.linalg.solve(A, b)
    return beta


def append_bias(Phi):
    ones = torch.ones(Phi.shape[0], 1, device=Phi.device, dtype=Phi.dtype)
    return torch.cat([Phi, ones], dim=1)


def alpaca_forward_task(enc, xs, ys, xq, lam):
    Phi_s = append_bias(enc(xs))            # (K, p+1)
    Phi_q = append_bias(enc(xq))            # (K_q, p+1)
    beta = alpaca_solve(Phi_s, ys, lam)
    return Phi_q @ beta


# ---------------- ALPaCA meta-training ----------------

def train_alpaca(n_iters=8000, meta_bs=4, K=10, K_q=10, hidden=40, p=40,
                 outer_lr=1e-3, seed=0, log_every=500):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    enc = ALPaCAEncoder(hidden=hidden, p=p).to(device)
    log_lam = torch.tensor(math.log(1e-2), requires_grad=True)   # learned prior precision
    params = list(enc.parameters()) + [log_lam]
    opt = torch.optim.Adam(params, lr=outer_lr)

    curve = []
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, _, xq, yq = sample_tasks(meta_bs, K, K_q, gen, noise=0.0)
        opt.zero_grad()
        lam = log_lam.exp()
        meta_loss = 0.0
        for t in range(meta_bs):
            qpred = alpaca_forward_task(enc, xs[t], ys[t], xq[t], lam)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            curve.append((time.time() - t0, it, float(meta_loss.item())))
    train_time = time.time() - t0

    n_params = sum(q.numel() for q in enc.parameters()) + 1   # +1 for log_lam
    meta = {
        "enc": enc,
        "lam": float(log_lam.exp().item()),
        "p": p,
        "n_params": n_params,
    }
    return meta, train_time, curve


def eval_alpaca(meta, sampler, n_test=200, K=10, K_q=50, seed=0):
    """sampler(n,K,Kq,gen) -> (xs,ys,dys,xq,yq). Closed-form per task, no_grad."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sampler(n_test, K, K_q, gen)
    enc = meta["enc"]
    lam = torch.tensor(meta["lam"])
    losses = []
    t0 = time.time()
    with torch.no_grad():
        for t in range(n_test):
            qpred = alpaca_forward_task(enc, xs[t], ys[t], xq[t], lam)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    eval_time = time.time() - t0
    return float(torch.tensor(losses).mean()), eval_time / n_test


# wrappers giving sample_tasks the (n,K,Kq,gen) signature used by eval_alpaca
def _sin_sampler(n, K, Kq, gen):
    return sample_tasks(n, K, Kq, gen, noise=0.0)

def _tanh_sampler(n, K, Kq, gen):
    return sample_tanh_tasks(n, K, Kq, gen)


# ---------------- driver ----------------

def main():
    SEEDS = [0, 1, 2]
    N_ITERS = 8000
    N_TEST = 200
    K, K_q = 10, 50

    alpaca_in, alpaca_ooc, alpaca_params, alpaca_lam, alpaca_train_s = [], [], [], [], []
    lsq_in, lsq_ooc, lsq_params = [], [], []

    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)

        # --- ALPaCA: learned-basis closed-form meta-learner ---
        t0 = time.time()
        meta, t_train, curve = train_alpaca(n_iters=N_ITERS, seed=seed)
        mse_in, _ = eval_alpaca(meta, _sin_sampler, n_test=N_TEST, K=K, K_q=K_q, seed=seed)
        mse_ooc, _ = eval_alpaca(meta, _tanh_sampler, n_test=N_TEST, K=K, K_q=K_q, seed=seed)
        alpaca_in.append(mse_in)
        alpaca_ooc.append(mse_ooc)
        alpaca_params.append(meta["n_params"])
        alpaca_lam.append(meta["lam"])
        alpaca_train_s.append(t_train)
        print(f"  ALPaCA   in={mse_in:.4f} ooc={mse_ooc:.4f} "
              f"lam={meta['lam']:.2e} params={meta['n_params']} "
              f"train={t_train:.1f}s (final_meta_loss={curve[-1][2]:.4f})", flush=True)

        # --- Meta-LSQ-Sobolev reference: frozen basis, 3 scalars ---
        fls_meta, t_fls, _ = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=seed)
        l_in, _ = eval_fastlsq_meta(fls_meta, n_test=N_TEST, K=K, K_q=K_q, seed=seed)
        # OOC for Meta-LSQ: re-implement eval on tanh tasks (read-only inner solve)
        l_ooc = _eval_lsq_sobolev_ooc(fls_meta, n_test=N_TEST, K=K, K_q=K_q, seed=seed)
        lsq_in.append(l_in)
        lsq_ooc.append(l_ooc)
        lsq_params.append(3)  # sigma, mu, grad_w
        print(f"  Meta-LSQ in={l_in:.4f} ooc={l_ooc:.4f} "
              f"sigma={fls_meta['sigma']:.3f} mu={fls_meta['mu']:.2e} "
              f"gw={fls_meta['grad_w']:.2f} (3 scalars)", flush=True)

    def stats(v):
        t = torch.tensor(v, dtype=torch.float64)
        return float(t.mean()), float(t.std(unbiased=len(v) > 1))

    res = {
        "config": {
            "seeds": SEEDS, "n_iters_alpaca": N_ITERS, "n_iters_lsq": 2000,
            "n_test": N_TEST, "K": K, "K_q": K_q,
            "alpaca_arch": "MLP enc tanh, 2 hidden(40), p=40, +bias col, learned lam",
            "ooc_family": "y = a*tanh(s*(x-c)), a~U[.1,5] s~U[.5,2] c~U[-2,2]",
        },
        "alpaca": {
            "in_class_mse": stats(alpaca_in),
            "ooc_tanh_mse": stats(alpaca_ooc),
            "n_params": alpaca_params[0],
            "lam_mean": float(torch.tensor(alpaca_lam).mean()),
            "train_s": stats(alpaca_train_s),
            "raw_in": alpaca_in, "raw_ooc": alpaca_ooc,
        },
        "meta_lsq_sobolev": {
            "in_class_mse": stats(lsq_in),
            "ooc_tanh_mse": stats(lsq_ooc),
            "n_params": 3,
            "raw_in": lsq_in, "raw_ooc": lsq_ooc,
        },
    }

    os.makedirs(_os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    out = _os.path.join(_REPO_ROOT, "results", "alpaca_baseline.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n========== SUMMARY (mean +/- std over seeds) ==========")
    ai_m, ai_s = res["alpaca"]["in_class_mse"]
    ao_m, ao_s = res["alpaca"]["ooc_tanh_mse"]
    li_m, li_s = res["meta_lsq_sobolev"]["in_class_mse"]
    lo_m, lo_s = res["meta_lsq_sobolev"]["ooc_tanh_mse"]
    print(f"ALPaCA   ({alpaca_params[0]} params): in={ai_m:.4f}+/-{ai_s:.4f}  ooc(tanh)={ao_m:.4f}+/-{ao_s:.4f}")
    print(f"Meta-LSQ (3 params)   : in={li_m:.4f}+/-{li_s:.4f}  ooc(tanh)={lo_m:.4f}+/-{lo_s:.4f}")
    print(f"[saved] {out}")
    return res


def _eval_lsq_sobolev_ooc(meta, n_test=200, K=10, K_q=50, seed=0):
    """Evaluate a meta-trained Meta-LSQ-Sobolev model on OOC tanh tasks.
    Reuses the frozen-basis inner solve (imported read-only)."""
    from experiments.maml_sinusoid import fastlsq_inner_solve, fastlsq_predict
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tanh_tasks(n_test, K, K_q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"]); mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    losses = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                                          sobolev=meta["sobolev"], grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


if __name__ == "__main__":
    main()
