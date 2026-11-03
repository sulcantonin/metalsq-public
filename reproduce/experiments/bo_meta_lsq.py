"""Bayesian optimisation with a Meta-LSQ surrogate (value-only RFF ridge).

Surrogate produces a closed-form Gaussian posterior over value (the same solve
also yields a gradient posterior, which vanilla EI does not use but which would
unlock gradient-aware acquisition).
We compare:
  - random search
  - Meta-LSQ surrogate (value-only) + classic Expected Improvement
  - sklearn GaussianProcessRegressor + classic Expected Improvement

Test problems: Branin (d=2), Hartmann-6 (d=6).

TODO: head-to-head against BoTorch / TuRBO / gradient-aware acquisitions.
      Requires installing botorch; deferred.
"""

import math
import time
import torch
import numpy as np
from scipy.stats import norm

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

torch.set_default_dtype(torch.float64)
SEED = 0


# ---------------- test functions ----------------

def branin(x):
    """x: (..., 2) in [-5, 10] x [0, 15]. Global min ~0.398."""
    x1, x2 = x[..., 0], x[..., 1]
    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s


def branin_min():
    return 0.397887


BRANIN_BOUNDS = torch.tensor([[-5.0, 10.0], [0.0, 15.0]])


def hartmann6(x):
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    # x: (..., 6) in [0,1]^6
    x_ = x.unsqueeze(-2)  # (..., 1, 6)
    inner = (A * (x_ - P) ** 2).sum(dim=-1)  # (..., 4)
    return -(alpha * torch.exp(-inner)).sum(dim=-1)


def hartmann6_min():
    return -3.32237


HARTMANN_BOUNDS = torch.stack([torch.zeros(6), torch.ones(6)], dim=-1)


# ---------------- Meta-LSQ-Sobolev surrogate ----------------

def make_basis(d, N, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(d, N, generator=g) * sigma).double()
    b = (torch.rand(1, N, generator=g) * 2 * math.pi).double()
    return W, b


def fit_surrogate(X, y, W, b, mu, use_grad=False, grad=None, grad_w=1.0):
    """Fit beta. If use_grad, stack value+grad rows.
    X: (M, d)  y: (M,)  grad: (M, d) or None
    Returns beta (N,) and the Cholesky factor of K = AtA+muI for posterior var.
    """
    N = W.shape[1]
    z = X @ W + b
    H = torch.sin(z) / math.sqrt(N)
    if use_grad and grad is not None:
        # Build per-coordinate gradient design matrices
        M, d = X.shape
        Gs = []
        for k in range(d):
            Gk = (W[k:k+1, :] * torch.cos(z)) / math.sqrt(N)
            Gs.append(Gk)
        G_stack = torch.cat(Gs, dim=0)
        dy_stack = grad.T.reshape(-1)
        # torch.sqrt to keep grad_w differentiable (math.sqrt detaches a tensor);
        # the BO surrogate here is value-only (use_grad=False) so this branch is
        # latent, but we keep it consistent with the rest of the codebase.
        sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype, device=H.device))
        A = torch.cat([H, sqrt_w * G_stack], dim=0)
        t = torch.cat([y, sqrt_w * dy_stack], dim=0)
    else:
        A = H
        t = y
    K = A.T @ A + mu * torch.eye(N).double()
    L = torch.linalg.cholesky(K)
    beta = torch.cholesky_solve((A.T @ t).unsqueeze(-1), L).squeeze(-1)
    return beta, L, A


def predict(x_test, W, b, beta):
    z = x_test @ W + b
    return torch.sin(z) / math.sqrt(W.shape[1]) @ beta


def predict_var(x_test, W, b, L, mu):
    """Posterior variance under sigma_noise=1, prior sigma_beta^2=1/mu."""
    z = x_test @ W + b
    h = torch.sin(z) / math.sqrt(W.shape[1])
    # var(x) = mu * h K^{-1} h^T where K = AtA + mu I
    # via solve with L
    v = torch.cholesky_solve(h.T, L)  # (N, M)
    var = mu * (h * v.T).sum(dim=-1)
    return var.clamp(min=1e-12)


# ---------------- acquisition ----------------

def expected_improvement(mu_pred, var_pred, y_best, xi=0.0):
    """Standard EI; minimisation."""
    sigma = torch.sqrt(var_pred)
    improvement = y_best - mu_pred - xi
    z = improvement / sigma
    z_np = z.numpy()
    ei = improvement.numpy() * norm.cdf(z_np) + sigma.numpy() * norm.pdf(z_np)
    return torch.tensor(ei).clamp(min=0)


def sample_candidates(bounds, n, gen):
    d = bounds.shape[0]
    u = torch.rand(n, d, generator=gen).double()
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + u * (hi - lo)


# ---------------- BO loops ----------------

def run_random(objective, bounds, n_iters, gen):
    Xs = sample_candidates(bounds, n_iters, gen)
    ys = objective(Xs)
    best = []
    cur_best = float("inf")
    for y in ys:
        cur_best = min(cur_best, float(y))
        best.append(cur_best)
    return best


def run_meta_lsq_bo(objective, bounds, n_iters, n_init=5, N=200, sigma=1.0,
                    mu=1e-4, use_grad=False, n_candidates=2000, gen=None,
                    basis_seed=SEED):
    d = bounds.shape[0]
    Xs = [sample_candidates(bounds, n_init, gen)]
    ys = [objective(Xs[0])]
    X = Xs[0]
    y = ys[0]
    best = [float(y.min())]
    # Vary the frozen RFF draw per trial so the reported std reflects basis
    # variance (a major variance source for an RFF surrogate), not just the
    # candidate-sampling noise.
    W, b = make_basis(d, N, sigma, seed=basis_seed)
    for it in range(n_init, n_iters):
        # Re-fit surrogate (we only have values; grad unavailable for black-box)
        beta, L, A = fit_surrogate(X, y, W, b, mu, use_grad=False)
        cand = sample_candidates(bounds, n_candidates, gen)
        mu_pred = predict(cand, W, b, beta)
        var_pred = predict_var(cand, W, b, L, mu)
        ei = expected_improvement(mu_pred, var_pred, float(y.min()))
        idx = int(torch.argmax(ei))
        x_new = cand[idx:idx+1]
        y_new = objective(x_new)
        X = torch.cat([X, x_new], dim=0)
        y = torch.cat([y, y_new], dim=0)
        best.append(float(y.min()))
    # Pad to length n_iters
    while len(best) < n_iters:
        best.append(best[-1])
    return best


def run_gp_bo(objective, bounds, n_iters, n_init=5, n_candidates=2000, gen=None):
    if not HAVE_SKLEARN:
        return None
    d = bounds.shape[0]
    X = sample_candidates(bounds, n_init, gen)
    y = objective(X)
    best = [float(y.min())]
    for it in range(n_init, n_iters):
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-4)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                       random_state=0, alpha=1e-6)
        gp.fit(X.numpy(), y.numpy())
        cand = sample_candidates(bounds, n_candidates, gen)
        mu_pred, sigma_pred = gp.predict(cand.numpy(), return_std=True)
        z = (float(y.min()) - mu_pred) / (sigma_pred + 1e-12)
        ei = (float(y.min()) - mu_pred) * norm.cdf(z) + sigma_pred * norm.pdf(z)
        ei = np.clip(ei, 0, None)
        idx = int(np.argmax(ei))
        x_new = cand[idx:idx+1]
        y_new = objective(x_new)
        X = torch.cat([X, x_new], dim=0)
        y = torch.cat([y, y_new], dim=0)
        best.append(float(y.min()))
    while len(best) < n_iters:
        best.append(best[-1])
    return best


# ---------------- driver ----------------

def regret(curve, true_min):
    return [c - true_min for c in curve]


def run_problem(name, objective, bounds, true_min, n_iters=40, n_seeds=5):
    print(f"\n========== {name} (d={bounds.shape[0]}, {n_iters} iters, {n_seeds} seeds) ==========")
    out = {"random": [], "meta_lsq": [], "gp": []}
    times = {"random": 0.0, "meta_lsq": 0.0, "gp": 0.0}
    for seed in range(n_seeds):
        gen = torch.Generator().manual_seed(seed)
        t0 = time.time()
        r = run_random(objective, bounds, n_iters, gen)
        times["random"] += time.time() - t0
        out["random"].append(r)

        gen = torch.Generator().manual_seed(seed)
        t0 = time.time()
        m = run_meta_lsq_bo(objective, bounds, n_iters, sigma=1.0, gen=gen, basis_seed=seed)
        times["meta_lsq"] += time.time() - t0
        out["meta_lsq"].append(m)

        if HAVE_SKLEARN:
            gen = torch.Generator().manual_seed(seed)
            t0 = time.time()
            g = run_gp_bo(objective, bounds, n_iters, gen=gen)
            times["gp"] += time.time() - t0
            out["gp"].append(g)

    print(f"{'method':<12} {'final regret (mean)':>22} {'final regret (std)':>22} {'time (s)':>10}")
    for k in ("random", "meta_lsq", "gp"):
        if not out[k]:
            continue
        regrets = np.array(out[k])[:, -1] - true_min
        print(f"{k:<12} {regrets.mean():>22.4e} {regrets.std():>22.4e} {times[k]:>10.1f}")

    # Convergence at a few checkpoints
    cp = [4, 9, 19, 29, n_iters - 1]
    cp = [c for c in cp if c < n_iters]
    for k in ("random", "meta_lsq", "gp"):
        if not out[k]:
            continue
        print(f"  {k}: ", end="")
        for c in cp:
            mean_r = np.mean([curve[c] - true_min for curve in out[k]])
            print(f" @{c+1}={mean_r:.3e}", end="")
        print()

    return out, times


def main(save=None):
    pr_branin = run_problem("Branin",     branin,    BRANIN_BOUNDS,   branin_min(),    n_iters=30, n_seeds=5)
    pr_hart   = run_problem("Hartmann-6", hartmann6, HARTMANN_BOUNDS, hartmann6_min(), n_iters=40, n_seeds=5)

    if save:
        import json
        problems = {"branin": (pr_branin, branin_min()),
                    "hartmann6": (pr_hart, hartmann6_min())}
        npz = {}
        summ = {}
        for pname, ((out, times), tmin) in problems.items():
            summ[pname] = {"true_min": tmin}
            for k in ("random", "meta_lsq", "gp"):
                if not out[k]:
                    continue
                arr = np.array(out[k], dtype=float)   # (n_seeds, n_iters)
                npz[f"{pname}_{k}"] = arr
                final_regret = arr[:, -1] - tmin
                summ[pname][k] = {"final_regret_mean": float(final_regret.mean()),
                                  "final_regret_std": float(final_regret.std()),
                                  "time_s": times[k]}
        np.savez(save + ".npz", **npz)
        with open(save + ".json", "w") as f:
            json.dump(summ, f, indent=2)
        print(f"\n[saved] {save}.npz and {save}.json")


if __name__ == "__main__":
    import sys
    _save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    main(save=_save)
