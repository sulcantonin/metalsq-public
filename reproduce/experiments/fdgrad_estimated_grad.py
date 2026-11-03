"""Meta-LSQ-Sobolev with ESTIMATED (finite-difference) support gradients.

Addresses the 'analytic-gradient-friendly' scope concern: instead of feeding the
inner solve the analytic support gradient dy/dx, we ESTIMATE it by a central
finite difference from (noisy) function-value queries:

    dy_fd(x) = ( y(x+h) - y(x-h) ) / (2 h)

where every value query (the two FD probes AND the support value y_s) carries i.i.d.
Gaussian noise of std nu. The central FD has error
    bias  = O(h^2)   (Taylor truncation, here -a sin(x+phi) * h^2 / 6)
    noise = O(nu / h) (two independent noise draws divided by 2h)
so there is a bias-variance sweet spot in h.

We meta-learn Meta-LSQ-Sobolev (learnable sigma, mu, and the Sobolev row weight w via
log_gw with the torch.sqrt path, so w is genuinely trainable). Support gradient TARGETS
are corrupted to dy_fd; support VALUES y_s carry the same noise nu; the QUERY targets are
CLEAN (so test MSE measures error vs the true function).

Sweeps (3 seeds each):
  (A) h in {0.001, 0.01, 0.1, 0.5} at nu=0.1
  (B) nu in {0, 0.05, 0.1, 0.25} at h=0.1
For each cell: meta-learned w (=grad_w), clean-query test MSE, and the empirical FD-error
variance/bias of the estimated gradient. Baseline: value-only Meta-LSQ (w fixed = 0).

HYPOTHESIS: w stays high when FD gradients are accurate (small h, low nu) and drops when
they are noisy/biased (large nu; h too large -> bias; h too small at nu>0 -> variance),
mirroring the Bayesian-optimal w = sigma_y^2 / sigma_g^2 with sigma_g^2 the FD-error variance.
"""

import sys, os, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch

from experiments.maml_sinusoid import make_basis, fastlsq_predict

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")


# --------------------------------------------------------------------------------------
# Task sampler that returns the sinusoid PARAMETERS so we can query the generator at
# arbitrary points (x +- h) for the finite-difference probes. We re-implement here rather
# than reuse sample_tasks() because FD needs generator access at probe locations.
# --------------------------------------------------------------------------------------

def sample_tasks_fd(n_tasks, K, K_q, gen, h, nu):
    """Return (xs, ys_noisy, dy_fd, xq, yq_clean).

    ys_noisy : support values with noise std nu.
    dy_fd    : central FD gradient from noisy probes y(x+h), y(x-h).
    yq_clean : clean query targets (true function).
    All shapes match maml_sinusoid: xs (n,K,1), ys/dy (n,K), xq (n,K_q,1), yq (n,K_q).
    """
    a = torch.rand(n_tasks, 1, generator=gen) * 4.9 + 0.1     # [0.1, 5]
    phi = torch.rand(n_tasks, 1, generator=gen) * math.pi     # [0, pi]
    xs = torch.rand(n_tasks, K, generator=gen) * 10 - 5
    xq = torch.rand(n_tasks, K_q, generator=gen) * 10 - 5

    # true values
    ys_true = a * torch.sin(xs + phi)
    yq = a * torch.sin(xq + phi)                              # CLEAN query targets

    # noisy support values (carry noise nu)
    ys = ys_true + nu * torch.randn(ys_true.shape, generator=gen)

    # FD probes, each an INDEPENDENT noisy value query
    yp = a * torch.sin(xs + h + phi) + nu * torch.randn(xs.shape, generator=gen)
    ym = a * torch.sin(xs - h + phi) + nu * torch.randn(xs.shape, generator=gen)
    dy_fd = (yp - ym) / (2.0 * h)

    # analytic gradient (for diagnostics only: bias/variance of the estimate)
    dy_true = a * torch.cos(xs + phi)

    return (xs.unsqueeze(-1), ys, dy_fd, xq.unsqueeze(-1), yq, dy_true)


# --------------------------------------------------------------------------------------
# Inner solve (reimplemented locally so we control the torch.sqrt(grad_w) path and can set
# grad_w to a learnable tensor). Identical math to maml_sinusoid.fastlsq_inner_solve.
# --------------------------------------------------------------------------------------

def inner_solve(xs, ys, dys, Wb, bb, sigma, mu, grad_w, sobolev=True):
    W = sigma * Wb                                   # (1, N)
    N = W.shape[1]
    Z = xs @ W + bb                                  # (K, N)
    H = torch.sin(Z) / math.sqrt(N)
    if sobolev:
        G = (W * torch.cos(Z)) / math.sqrt(N)        # d/dx of sin feature
        sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype, device=H.device))
        A = torch.cat([H, sqrt_w * G], dim=0)
        t = torch.cat([ys, sqrt_w * dys], dim=0)
    else:
        A = H
        t = ys
    AtA = A.T @ A + mu * torch.eye(N)
    Atb = A.T @ t
    beta = torch.linalg.solve(AtA, Atb)
    return W, beta


# --------------------------------------------------------------------------------------
# Meta-train: learnable log_sigma, log_mu, and (if sobolev) log_gw.
# --------------------------------------------------------------------------------------

def meta_train(h, nu, sobolev, grad_w_init, n_iters, seed,
               meta_bs=4, K=10, K_q=10, N=200, lr=5e-3):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)

    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(grad_w_init), requires_grad=True)
    params = [log_sigma, log_mu] + ([log_gw] if sobolev else [])
    opt = torch.optim.Adam(params, lr=lr)

    for it in range(n_iters):
        xs, ys, dy_fd, xq, yq, _ = sample_tasks_fd(meta_bs, K, K_q, gen, h, nu)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        gw = log_gw.exp() if sobolev else 0.0
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = inner_solve(xs[t], ys[t], dy_fd[t], Wb, bb, sigma, mu, gw,
                                  sobolev=sobolev)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()

    return {
        "Wb": Wb, "bb": bb,
        "sigma": float(log_sigma.exp().item()),
        "mu": float(log_mu.exp().item()),
        "grad_w": float(log_gw.exp().item()) if sobolev else 0.0,
        "sobolev": sobolev,
    }


def meta_eval(meta, h, nu, seed, n_test=200, K=10, K_q=50, N=200):
    """Clean-query test MSE. Support still uses FD gradients + noisy values (the only
    information available at adaptation time)."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dy_fd, xq, yq, _ = sample_tasks_fd(n_test, K, K_q, gen, h, nu)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"]
    losses = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = inner_solve(xs[t], ys[t], dy_fd[t], Wb, bb, sigma, mu, gw,
                                  sobolev=meta["sobolev"])
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


def fd_error_stats(h, nu, seed, n_tasks=400, K=10):
    """Empirical bias^2 and variance of the FD gradient estimate vs analytic gradient,
    over a large pool of tasks/points. Pure diagnostic, no training."""
    gen = torch.Generator().manual_seed(seed + 4242)
    _, _, dy_fd, _, _, dy_true = sample_tasks_fd(n_tasks, K, 1, gen, h, nu)
    err = (dy_fd - dy_true).reshape(-1)
    bias = float(err.mean().item())
    var = float(err.var(unbiased=True).item())
    mse = float(err.pow(2).mean().item())
    return {"fd_bias": bias, "fd_var": var, "fd_mse": mse}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def run_cell(h, nu, seeds, n_iters, grad_w_init):
    ws, mse_sob, mse_val = [], [], []
    for s in seeds:
        meta_s = meta_train(h, nu, sobolev=True, grad_w_init=grad_w_init,
                            n_iters=n_iters, seed=s)
        ws.append(meta_s["grad_w"])
        mse_sob.append(meta_eval(meta_s, h, nu, seed=s))
        # value-only baseline (w fixed = 0): meta-learn sigma, mu only
        meta_v = meta_train(h, nu, sobolev=False, grad_w_init=grad_w_init,
                           n_iters=n_iters, seed=s)
        mse_val.append(meta_eval(meta_v, h, nu, seed=s))
    fd = fd_error_stats(h, nu, seed=seeds[0])
    wt = torch.tensor(ws); ms = torch.tensor(mse_sob); mv = torch.tensor(mse_val)
    return {
        "h": h, "nu": nu,
        "w_mean": float(wt.mean()), "w_std": float(wt.std(unbiased=False)),
        "w_all": ws,
        "mse_sob_mean": float(ms.mean()), "mse_sob_std": float(ms.std(unbiased=False)),
        "mse_sob_all": mse_sob,
        "mse_val_mean": float(mv.mean()), "mse_val_std": float(mv.std(unbiased=False)),
        "mse_val_all": mse_val,
        **fd,
    }


def main():
    seeds = [0, 1, 2]
    n_iters = 1500
    grad_w_init = 10.0
    K, K_q, N = 10, 10, 200

    t0 = time.time()

    # Sweep A: vary h at nu=0.1
    sweep_A = []
    for h in [0.001, 0.01, 0.1, 0.5]:
        sweep_A.append(run_cell(h, nu=0.1, seeds=seeds, n_iters=n_iters,
                                grad_w_init=grad_w_init))
        print(f"[A] h={h:<6} done  w={sweep_A[-1]['w_mean']:.3f}  "
              f"mse_sob={sweep_A[-1]['mse_sob_mean']:.4f}  "
              f"mse_val={sweep_A[-1]['mse_val_mean']:.4f}  "
              f"fd_mse={sweep_A[-1]['fd_mse']:.4g}  ({time.time()-t0:.0f}s)")

    # Sweep B: vary nu at h=0.1
    sweep_B = []
    for nu in [0.0, 0.05, 0.1, 0.25]:
        sweep_B.append(run_cell(0.1, nu=nu, seeds=seeds, n_iters=n_iters,
                                grad_w_init=grad_w_init))
        print(f"[B] nu={nu:<6} done  w={sweep_B[-1]['w_mean']:.3f}  "
              f"mse_sob={sweep_B[-1]['mse_sob_mean']:.4f}  "
              f"mse_val={sweep_B[-1]['mse_val_mean']:.4f}  "
              f"fd_mse={sweep_B[-1]['fd_mse']:.4g}  ({time.time()-t0:.0f}s)")

    out = {
        "config": {
            "seeds": seeds, "n_iters": n_iters, "grad_w_init": grad_w_init,
            "K": K, "K_q_train": K_q, "K_q_test": 50, "N": N,
            "meta_bs": 4, "lr": 5e-3, "n_test": 200,
            "sweep_A": {"h": [0.001, 0.01, 0.1, 0.5], "nu": 0.1},
            "sweep_B": {"nu": [0.0, 0.05, 0.1, 0.25], "h": 0.1},
            "description": "Meta-LSQ-Sobolev with ESTIMATED (central finite-difference) "
                           "support gradients; w (grad_w) is learnable via log_gw / "
                           "torch.sqrt path. Baseline mse_val = value-only (w=0).",
        },
        "sweep_A": sweep_A,
        "sweep_B": sweep_B,
        "total_time_s": time.time() - t0,
    }

    os.makedirs(_os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    path = _os.path.join(_REPO_ROOT, "results", "fdgrad_estimated_grad.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {path}   total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
