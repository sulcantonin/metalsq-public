"""Meta-LSQ-Sobolev with ESTIMATED (finite-difference) support gradients.

Addresses the 'analytic-gradient-friendly' scope concern: instead of feeding the
inner Sobolev solve the analytic dy/dx = a cos(x+phi), we ESTIMATE the support
gradient by finite differences from (noisy) function values:

    dy_fd(x) = ( y_noisy(x+h) - y_noisy(x-h) ) / (2 h)

where every value query (the support value y_s AND the two FD probes) carries
i.i.d. Gaussian noise of std nu. The generator y = a sin(x+phi) is known, so we
can evaluate it at x +/- h. FD error decomposes as

    bias  ~ (h^2 / 6) * y'''(x)        (Taylor truncation, O(h^2))
    noise ~ N(0, nu^2 / (2 h^2))       (two independent noisy probes / (2h))

so the FD-gradient error variance is roughly  sigma_g^2 ~ nu^2/(2 h^2) + bias^2.
The Sobolev weight w = log_gw.exp() is meta-learnable (torch.sqrt(grad_w) so the
gradient flows). Theory: optimal w = sigma_y^2 / sigma_g^2, so w should be HIGH
when FD gradients are accurate (small h, low nu) and DROP when they are
noisy/biased (large nu; h too large -> bias, h too small -> noise blow-up).

Sweeps:
  (A) h in {0.001, 0.01, 0.1, 0.5} at nu = 0.1
  (B) nu in {0, 0.05, 0.1, 0.25} at h = 0.1
For each cell: meta-learned (sigma, mu, w) and CLEAN-query test MSE (3 seeds).
Baseline: value-only Meta-LSQ (w=0, i.e. no Sobolev rows) at each nu.

Self-contained: re-implements the meta-train/eval loops (read-only import of the
basis + predict helpers) so support gradient targets can be swapped for dy_fd
without touching any existing file.
"""

import sys, os, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

# read-only imports of frozen helpers
from experiments.maml_sinusoid import make_basis, fastlsq_predict


# ---------------- task generator with FD support gradients ----------------

def sample_tasks_fd(n_tasks, K, K_q, gen, h, nu):
    """Sinusoid tasks where the SUPPORT gradient is a finite-difference estimate.

    Returns (x_s, y_s_noisy, dy_fd, x_q, y_q_clean).

    - y_s carries value noise std nu (same noise model as the analytic case).
    - dy_fd = (y(x+h)-y(x-h))/(2h) where BOTH probes carry independent noise nu.
      The generator is known, so the probes are real evaluations of a sin(.).
    - x_q / y_q are CLEAN: test MSE is measured against the true function.
    """
    a = torch.rand(n_tasks, 1, generator=gen) * 4.9 + 0.1   # [0.1, 5]
    phi = torch.rand(n_tasks, 1, generator=gen) * math.pi   # [0, pi]
    xs = torch.rand(n_tasks, K, generator=gen) * 10 - 5
    xq = torch.rand(n_tasks, K_q, generator=gen) * 10 - 5

    ys_clean = a * torch.sin(xs + phi)
    yq = a * torch.sin(xq + phi)                              # clean query targets

    # noisy support values
    ys = ys_clean + nu * torch.randn(ys_clean.shape, generator=gen)

    # finite-difference gradient from two independent noisy probes of the known gen
    yp = a * torch.sin(xs + h + phi) + nu * torch.randn(xs.shape, generator=gen)
    ym = a * torch.sin(xs - h + phi) + nu * torch.randn(xs.shape, generator=gen)
    dy_fd = (yp - ym) / (2.0 * h)

    return xs.unsqueeze(-1), ys, dy_fd, xq.unsqueeze(-1), yq


# ---------------- inner solve (Sobolev, learnable w via torch.sqrt) ----------------

def inner_solve(xs, ys, dys, Wb, bb, sigma, mu, sobolev, grad_w):
    """Closed-form ridge solve; identical math to the frozen fastlsq_inner_solve.
    torch.sqrt(grad_w) keeps w learnable (gradient flows through the Sobolev rows)."""
    W = sigma * Wb                                  # (1, N)
    Z = xs @ W + bb                                 # (K, N)
    H = torch.sin(Z) / math.sqrt(W.shape[1])
    if sobolev:
        G = (W * torch.cos(Z)) / math.sqrt(W.shape[1])
        sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype, device=H.device))
        A = torch.cat([H, sqrt_w * G], dim=0)
        t = torch.cat([ys, sqrt_w * dys], dim=0)
    else:
        A = H; t = ys
    AtA = A.T @ A + mu * torch.eye(W.shape[1])
    Atb = A.T @ t
    beta = torch.linalg.solve(AtA, Atb)
    return W, beta


# ---------------- meta-train / eval ----------------

def train_meta(h, nu, sobolev, n_iters=1500, meta_bs=4, K=10, K_q=10, N=200,
               lr=5e-3, seed=0, grad_w0=1.0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(grad_w0), requires_grad=True)
    params = [log_sigma, log_mu] + ([log_gw] if sobolev else [])
    opt = torch.optim.Adam(params, lr=lr)

    for it in range(n_iters):
        xs, ys, dys, xq, yq = sample_tasks_fd(meta_bs, K, K_q, gen, h=h, nu=nu)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        gw = log_gw.exp() if sobolev else 1.0
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = inner_solve(xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                                  sobolev=sobolev, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()

    return {"Wb": Wb, "bb": bb,
            "sigma": float(log_sigma.exp().item()),
            "mu": float(log_mu.exp().item()),
            "grad_w": float(log_gw.exp().item()) if sobolev else 0.0,
            "sobolev": sobolev}


def eval_meta(meta, h, nu, n_test=200, K=10, K_q=50, seed=0):
    """Adapt from FD-gradient support (matched h, nu), evaluate on CLEAN query."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks_fd(n_test, K, K_q, gen, h=h, nu=nu)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"]); mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    losses = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = inner_solve(xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                                  sobolev=meta["sobolev"], grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


# ---------------- driver ----------------

def fd_error_stats(h, nu, n=20000, seed=123):
    """Empirical mean FD-gradient error variance over the task family,
    to cross-check the w = sigma_y^2 / sigma_g^2 story with real numbers."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 1, generator=g) * 4.9 + 0.1
    phi = torch.rand(n, 1, generator=g) * math.pi
    x = torch.rand(n, 1, generator=g) * 10 - 5
    true_d = a * torch.cos(x + phi)
    yp = a * torch.sin(x + h + phi) + nu * torch.randn(x.shape, generator=g)
    ym = a * torch.sin(x - h + phi) + nu * torch.randn(x.shape, generator=g)
    dy_fd = (yp - ym) / (2.0 * h)
    err = dy_fd - true_d
    return float(err.var().item()), float((err.mean()).abs().item())


def run(seeds=(0, 1, 2), save=None):
    H_LIST = [0.001, 0.01, 0.1, 0.5]
    NU_LIST = [0.0, 0.05, 0.1, 0.25]
    NU_FIXED = 0.1
    H_FIXED = 0.1

    def avg(fn):
        vals = [fn(s) for s in seeds]
        return float(sum(vals) / len(vals)), vals

    results = {"seeds": list(seeds), "sweep_h": [], "sweep_nu": []}
    t0 = time.time()

    # ---- Sweep A: vary h at nu = NU_FIXED ----
    print(f"=== Sweep A: vary h at nu={NU_FIXED} ===")
    for h in H_LIST:
        sig_var, sig_bias = fd_error_stats(h, NU_FIXED)
        w_mean, w_all = avg(lambda s: train_meta(h, NU_FIXED, sobolev=True, seed=s)["grad_w"])
        mse_mean, mse_all = avg(lambda s: eval_meta(
            train_meta(h, NU_FIXED, sobolev=True, seed=s), h, NU_FIXED, seed=s))
        # implied optimal w from empirical FD error variance: sigma_y^2 / sigma_g^2
        w_implied = (NU_FIXED**2) / sig_var if sig_var > 0 else float('inf')
        row = {"h": h, "nu": NU_FIXED, "w_mean": w_mean, "w_all": w_all,
               "mse_mean": mse_mean, "mse_all": mse_all,
               "fd_err_var": sig_var, "fd_bias": sig_bias, "w_implied": w_implied}
        results["sweep_h"].append(row)
        print(f"  h={h:<6} w={w_mean:8.3f}  MSE={mse_mean:.4f}  "
              f"fd_var={sig_var:.4f} bias={sig_bias:.4f} w_implied={w_implied:7.3f}")

    # ---- Sweep B: vary nu at h = H_FIXED ----
    print(f"\n=== Sweep B: vary nu at h={H_FIXED} ===")
    for nu in NU_LIST:
        sig_var, sig_bias = fd_error_stats(H_FIXED, nu)
        w_mean, w_all = avg(lambda s: train_meta(H_FIXED, nu, sobolev=True, seed=s)["grad_w"])
        mse_mean, mse_all = avg(lambda s: eval_meta(
            train_meta(H_FIXED, nu, sobolev=True, seed=s), H_FIXED, nu, seed=s))
        # value-only baseline (w=0) at same nu
        mse0_mean, mse0_all = avg(lambda s: eval_meta(
            train_meta(H_FIXED, nu, sobolev=False, seed=s), H_FIXED, nu, seed=s))
        w_implied = (nu**2) / sig_var if sig_var > 0 else float('inf')
        row = {"nu": nu, "h": H_FIXED, "w_mean": w_mean, "w_all": w_all,
               "mse_mean": mse_mean, "mse_all": mse_all,
               "mse_valonly_mean": mse0_mean, "mse_valonly_all": mse0_all,
               "fd_err_var": sig_var, "fd_bias": sig_bias, "w_implied": w_implied}
        results["sweep_nu"].append(row)
        print(f"  nu={nu:<5} w={w_mean:8.3f}  MSE(sob)={mse_mean:.4f}  "
              f"MSE(val-only)={mse0_mean:.4f}  fd_var={sig_var:.4f} "
              f"w_implied={w_implied:7.3f}")

    results["total_time_s"] = time.time() - t0
    print(f"\nTotal time: {results['total_time_s']:.1f}s")

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        with open(save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {save}")
    return results


if __name__ == "__main__":
    run(save=_os.path.join(_REPO_ROOT, "results", "fd_gradient_sweep.json"))
