"""Out-of-class evaluation: meta-train on sinusoids, evaluate on tanh / triangle.

Quantifies the cost of basis-task mismatch. Each method is meta-trained ONCE on
the canonical sinusoid family, then evaluated on three test families:

  In-class:  y = a sin(x + phi)                          (matches the basis exactly)
  OOC-A:     y = a tanh(b * (x - phi))                   (smooth but not in span)
  OOC-B:     y = a * triangle((x - phi) / T)             (non-smooth, harder)

Reports a method x family grid of test MSE plus the per-method degradation ratio
(OOC / in-class). The in-class advantage of Meta-LSQ should shrink and/or flip
when the task family is mismatched -- this is the honest framing.
"""

import os
import sys
import math
import time
import torch

# Allow `python experiments/out_of_class.py` to find the `experiments` package
# without setting PYTHONPATH (repo root = parent of this file's dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.maml_sinusoid import (
    train_mlp_maml,
    eval_mlp_maml,
    train_fastlsq_meta,
    eval_fastlsq_meta,
    make_basis,
    fastlsq_inner_solve,
    fastlsq_predict,
    sample_tasks as sample_sinusoid_tasks,
    MAMLNet,
    maml_inner_step,
)

torch.set_default_dtype(torch.float32)
SEED = 0


# ---------------- alternate task families ----------------

def sample_tanh_tasks(n_tasks, K, K_q, gen):
    """y = a * tanh(b * (x - phi))."""
    a = torch.rand(n_tasks, 1, generator=gen) * 4.9 + 0.1
    b = torch.rand(n_tasks, 1, generator=gen) * 1.5 + 0.5  # [0.5, 2.0]
    phi = torch.rand(n_tasks, 1, generator=gen) * 4 - 2
    xs = torch.rand(n_tasks, K, generator=gen) * 10 - 5
    xq = torch.rand(n_tasks, K_q, generator=gen) * 10 - 5
    ys = a * torch.tanh(b * (xs - phi))
    yq = a * torch.tanh(b * (xq - phi))
    # dy/dx = a * b * sech^2 = a * b * (1 - tanh^2)
    th = torch.tanh(b * (xs - phi))
    dys = a * b * (1 - th * th)
    return xs.unsqueeze(-1), ys, dys, xq.unsqueeze(-1), yq


def _triangle(z):
    """Triangle wave with period 1, amplitude 1. tri(0)=1, tri(0.5)=-1."""
    # z mod 1 in [0,1); map to triangle
    u = z - torch.floor(z)
    return torch.where(u < 0.5, 1.0 - 4.0 * u, 4.0 * u - 3.0)


def _triangle_deriv(z):
    u = z - torch.floor(z)
    return torch.where(u < 0.5, -4.0 * torch.ones_like(u), 4.0 * torch.ones_like(u))


def sample_triangle_tasks(n_tasks, K, K_q, gen):
    """y = a * triangle((x - phi) / T), T ~ U[1, 4]."""
    a = torch.rand(n_tasks, 1, generator=gen) * 4.9 + 0.1
    T = torch.rand(n_tasks, 1, generator=gen) * 3 + 1
    phi = torch.rand(n_tasks, 1, generator=gen) * 4 - 2
    xs = torch.rand(n_tasks, K, generator=gen) * 10 - 5
    xq = torch.rand(n_tasks, K_q, generator=gen) * 10 - 5
    ys = a * _triangle((xs - phi) / T)
    yq = a * _triangle((xq - phi) / T)
    dys = (a / T) * _triangle_deriv((xs - phi) / T)
    return xs.unsqueeze(-1), ys, dys, xq.unsqueeze(-1), yq


# ---------------- generic evaluators ----------------

def _stats(losses):
    """Per-task MSE summary. The OOC distribution is heavy-tailed (rare
    extrapolation blowups), so the mean alone is misleading; we report the
    median as the typical-task summary alongside it."""
    t = torch.tensor(losses)
    return {"mean": float(t.mean()), "median": float(t.median())}


def eval_meta_lsq_on_family(meta, sampler, n_test=200, K=10, K_q=50, seed=SEED,
                            mu_floor=None):
    """Evaluate a meta-learned Meta-LSQ solve on a task family.

    mu_floor: if given, the ridge is floored at this value, mu <- max(mu, mu_floor).
    The meta-learned mu is tiny (optimal for the noiseless in-class regime where
    the basis spans the target); flooring it regularises the closed-form solve
    against extrapolation blowups on out-of-distribution families.
    """
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sampler(n_test, K, K_q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu_val = meta["mu"] if mu_floor is None else max(meta["mu"], mu_floor)
    mu = torch.tensor(mu_val)
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    losses = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=meta["sobolev"], grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return _stats(losses)


def eval_mlp_on_family(net, sampler, n_test=200, K=10, K_q=50,
                       inner_lr=0.01, inner_steps=1, seed=SEED):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, _, xq, yq = sampler(n_test, K, K_q, gen)
    losses = []
    for t in range(n_test):
        params = net.init_params()
        for _ in range(inner_steps):
            # first_order=True: same adapted parameters, no unused second-order
            # graph (see eval_mlp_maml in maml_sinusoid.py).
            params = maml_inner_step(net, params, xs[t], ys[t], inner_lr,
                                     first_order=True)
        with torch.no_grad():
            qpred = net(xq[t], params)
        losses.append((qpred - yq[t]).pow(2).mean().item())
    return _stats(losses)


# ---------------- driver ----------------

def main(save=None):
    print("Meta-training on sinusoid family...")
    t0 = time.time()
    mlp_net, t_mlp, _ = train_mlp_maml(n_iters=8000, log_every=2000)
    print(f"  MLP-MAML  ({t_mlp:.1f}s)")
    fl_meta, t_fl, _ = train_fastlsq_meta(sobolev=False, n_iters=2000, log_every=500)
    print(f"  FastLSQ-Meta  ({t_fl:.1f}s)")
    fls_meta, t_fls, _ = train_fastlsq_meta(sobolev=True, n_iters=2000, log_every=500)
    print(f"  FastLSQ-Sobolev-Meta  ({t_fls:.1f}s)")
    print(f"Total meta-training: {time.time()-t0:.1f}s\n")

    families = [
        ("in-class (sin)", sample_sinusoid_tasks),
        ("OOC-A (tanh)", sample_tanh_tasks),
        ("OOC-B (triangle)", sample_triangle_tasks),
    ]

    MU_FLOOR = 1e-2   # deployment-time ridge floor for the robust variant

    out = {}
    for fam_name, fam_sampler in families:
        out[fam_name] = {
            "mlp":                 eval_mlp_on_family(mlp_net, fam_sampler),
            "meta_lsq":            eval_meta_lsq_on_family(fl_meta, fam_sampler),
            "meta_lsq_sob":        eval_meta_lsq_on_family(fls_meta, fam_sampler),
            "meta_lsq_sob_robust": eval_meta_lsq_on_family(fls_meta, fam_sampler,
                                                           mu_floor=MU_FLOOR),
        }

    cols = [("mlp", "MLP-MAML"), ("meta_lsq_sob", "M-LSQ-Sob"),
            ("meta_lsq_sob_robust", f"M-LSQ-Sob(mu>={MU_FLOOR:g})")]
    print(f"{'family':<18} " + " ".join(f"{lab+' mean/med':>26}" for _, lab in cols))
    for fam_name, _ in families:
        d = out[fam_name]
        cells = " ".join(f"{d[k]['mean']:>12.3f}/{d[k]['median']:<12.4f}" for k, _ in cols)
        print(f"{fam_name:<18} {cells}")

    print(f"\nMLP / Meta-LSQ-Sob(robust) per family  (>1 = Meta-LSQ wins, on mean and median):")
    for fam_name, _ in families:
        d = out[fam_name]
        rm = d["mlp"]["mean"] / d["meta_lsq_sob_robust"]["mean"]
        rmd = d["mlp"]["median"] / d["meta_lsq_sob_robust"]["median"]
        print(f"  {fam_name:<18} mean {rm:>8.2f}x   median {rmd:>8.2f}x")

    if save:
        import json
        with open(save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {save}")


if __name__ == "__main__":
    import sys
    _save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    main(save=_save)
