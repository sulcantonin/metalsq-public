"""Finite-difference (ESTIMATED) gradient Sobolev Meta-LSQ sweep.

Addresses the 'analytic-gradient-friendly' scope concern: instead of feeding the
inner Sobolev solve EXACT support gradients dy/dx = a cos(x+phi), we estimate them
by central finite differences on (possibly noisy) function values queried from the
known generator:

    dy_fd(x) = ( y_noisy(x+h) - y_noisy(x-h) ) / (2h)

where every value query (the support y_s itself AND both FD probes) carries i.i.d.
Gaussian value noise of std nu. The query/test targets stay CLEAN, so reported MSE
is error against the true function.

FD error decomposition:
    bias  = O(h^2)   (Taylor truncation of central difference)
    noise = O(nu/h)  (two independent noisy probes divided by 2h)
=> a bias-variance sweet spot in h. The FD-error variance plays the role of
sigma_g^2 in the optimal Sobolev weight w* = sigma_y^2 / sigma_g^2, so we expect
the meta-learned w to be high when FD gradients are accurate (small bias + small
noise) and to collapse toward the value-only regime (w->0) when they are not.

Sweeps:
  (A) h in {0.001, 0.01, 0.1, 0.5} at nu=0.1
  (B) nu in {0, 0.05, 0.1, 0.25}  at h=0.1
Baseline: value-only Meta-LSQ (no Sobolev rows, w=0) under the same nu.

Self-contained; imports the reference module READ-ONLY. Never modifies any file.
"""

import sys
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import os, math, time, json
import torch

from experiments.maml_sinusoid import (
    make_basis,
    fastlsq_inner_solve,
    fastlsq_predict,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")
RESULTS_PATH = _os.path.join(_REPO_ROOT, "results", "fdgrad_sobolev_sweep.json")


# ---------------- FD-gradient task generator ----------------

def sample_tasks_fd(n_tasks, K, K_q, gen, h, nu):
    """Sinusoid tasks with FINITE-DIFFERENCE, possibly noisy support gradients.

    Returns (x_s, y_s, dy_fd, x_q, y_q) where:
      - y_s carries value noise std nu.
      - dy_fd = (y_noisy(x+h) - y_noisy(x-h)) / (2h), each probe an INDEPENDENT
        noisy query of the known sinusoid generator (so probe noise is fresh, not
        the same draw as y_s -- this is what makes FD noise ~ nu/h).
      - x_q, y_q are CLEAN (query measures error vs true function).
    """
    a = torch.rand(n_tasks, 1, generator=gen) * 4.9 + 0.1   # [0.1, 5]
    phi = torch.rand(n_tasks, 1, generator=gen) * math.pi   # [0, pi]
    xs = torch.rand(n_tasks, K, generator=gen) * 10 - 5
    xq = torch.rand(n_tasks, K_q, generator=gen) * 10 - 5

    def f(x):  # true sinusoid value at x
        return a * torch.sin(x + phi)

    # support values (noisy)
    ys = f(xs)
    if nu > 0.0:
        ys = ys + nu * torch.randn(ys.shape, generator=gen)

    # central FD gradient from two independent noisy probes
    yp = f(xs + h)
    ym = f(xs - h)
    if nu > 0.0:
        yp = yp + nu * torch.randn(yp.shape, generator=gen)
        ym = ym + nu * torch.randn(ym.shape, generator=gen)
    dy_fd = (yp - ym) / (2.0 * h)

    # clean query
    yq = f(xq)
    return xs.unsqueeze(-1), ys, dy_fd, xq.unsqueeze(-1), yq


# ---------------- meta-train loop (reimplemented, learnable w) ----------------

def train_meta(n_iters, seed, h, nu, sobolev, grad_w0=10.0,
               meta_bs=4, K=10, K_q=10, N=200, lr=5e-3):
    """Meta-train Meta-LSQ on FD-gradient tasks.

    sobolev=True  -> Sobolev rows on FD gradients, learnable w via log_gw.
    sobolev=False -> value-only baseline (w=0).
    Returns dict with learned (sigma, mu, grad_w).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)

    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(grad_w0), requires_grad=True)
    params = [log_sigma, log_mu] + ([log_gw] if sobolev else [])
    opt = torch.optim.Adam(params, lr=lr)

    t0 = time.time()
    for it in range(n_iters):
        xs, ys, dys, xq, yq = sample_tasks_fd(meta_bs, K, K_q, gen, h=h, nu=nu)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        # torch.sqrt path inside fastlsq_inner_solve keeps gw learnable
        gw = log_gw.exp() if sobolev else 1.0
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=sobolev, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()
    train_s = time.time() - t0

    return {
        "Wb": Wb, "bb": bb,
        "sigma": float(log_sigma.exp().item()),
        "mu": float(log_mu.exp().item()),
        "grad_w": float(log_gw.exp().item()) if sobolev else 0.0,
        "sobolev": sobolev,
        "train_s": train_s,
    }


def eval_meta(meta, seed, h, nu, n_test=200, K=10, K_q=50):
    """Clean-query test MSE. Support carries the SAME nu / FD-gradient corruption
    as training; query is clean."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks_fd(n_test, K, K_q, gen, h=h, nu=nu)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    losses = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=meta["sobolev"], grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


# ---------------- diagnostic: empirical FD-gradient error ----------------

def fd_grad_error_stats(seed, h, nu, n=4000, K=10):
    """Empirical bias^2, variance and MSE of dy_fd vs the TRUE gradient
    a cos(x+phi), to corroborate the bias-variance story behind w."""
    gen = torch.Generator().manual_seed(seed + 7777)
    a = torch.rand(n, 1, generator=gen) * 4.9 + 0.1
    phi = torch.rand(n, 1, generator=gen) * math.pi
    xs = torch.rand(n, K, generator=gen) * 10 - 5

    def f(x):
        return a * torch.sin(x + phi)

    dtrue = a * torch.cos(xs + phi)
    yp = f(xs + h) + (nu * torch.randn(n, K, generator=gen) if nu > 0 else 0.0)
    ym = f(xs - h) + (nu * torch.randn(n, K, generator=gen) if nu > 0 else 0.0)
    dy_fd = (yp - ym) / (2.0 * h)
    err = dy_fd - dtrue
    # noiseless central-difference bias (deterministic part)
    ypc = f(xs + h); ymc = f(xs - h)
    bias = (ypc - ymc) / (2.0 * h) - dtrue
    return {
        "mse": float(err.pow(2).mean()),
        "bias2": float(bias.pow(2).mean()),
        "var": float((err - bias).pow(2).mean()),
    }


# ---------------- driver ----------------

def run_cell(seeds, h, nu, n_iters):
    """Average over seeds for one (h, nu). Returns sobolev + value-only results."""
    w_list, mse_sob, mse_val = [], [], []
    for s in seeds:
        m_sob = train_meta(n_iters=n_iters, seed=s, h=h, nu=nu, sobolev=True)
        w_list.append(m_sob["grad_w"])
        mse_sob.append(eval_meta(m_sob, seed=s, h=h, nu=nu))

        m_val = train_meta(n_iters=n_iters, seed=s, h=h, nu=nu, sobolev=False)
        mse_val.append(eval_meta(m_val, seed=s, h=h, nu=nu))

    def ms(xs):
        t = torch.tensor(xs)
        return float(t.mean()), float(t.std(unbiased=False))

    w_m, w_s = ms(w_list)
    sob_m, sob_s = ms(mse_sob)
    val_m, val_s = ms(mse_val)
    fd = fd_grad_error_stats(seed=seeds[0], h=h, nu=nu)
    return {
        "h": h, "nu": nu,
        "w_mean": w_m, "w_std": w_s,
        "mse_sob_mean": sob_m, "mse_sob_std": sob_s,
        "mse_val_mean": val_m, "mse_val_std": val_s,
        "w_list": w_list,
        "fd_grad_mse": fd["mse"], "fd_grad_bias2": fd["bias2"], "fd_grad_var": fd["var"],
    }


def init_sensitivity_probe(n_iters_long=4000):
    """Diagnostic: is the learned w a true optimum or an optimization stall?

    The main sweep inits log_gw at log(10). The probe re-fits the worst-FD cell
    (h=0.001, nu=0.1) and a good-FD cell (h=0.5, nu=0.1) from a SMALL init w0=0.01
    and with a longer budget, to see where the meta-objective actually wants w.
    Returns a list of records. Cheap (4 fits, single seed)."""
    recs = []
    cfgs = [
        ("h0.001_w0.01_1500", 0.001, 0.1, 0.01, 1500),
        ("h0.001_w10_4000",   0.001, 0.1, 10.0, n_iters_long),
        ("h0.001_w0.01_4000", 0.001, 0.1, 0.01, n_iters_long),
        ("h0.5_w0.01_4000",   0.5,   0.1, 0.01, n_iters_long),
    ]
    for tag, h, nu, w0, ni in cfgs:
        m = train_meta(n_iters=ni, seed=0, h=h, nu=nu, sobolev=True, grad_w0=w0)
        mse = eval_meta(m, seed=0, h=h, nu=nu)
        recs.append({"tag": tag, "h": h, "nu": nu, "w0": w0, "n_iters": ni,
                     "learned_w": m["grad_w"], "sob_mse": mse})
        print(f"[probe] {tag:<20} w0={w0:<5} iters={ni:<5} -> w={m['grad_w']:.4f}  sob_mse={mse:.4f}")
    return recs


def main():
    seeds = [0, 1, 2]
    n_iters = 1500
    t_start = time.time()

    # (A) sweep h at nu=0.1
    h_grid = [0.001, 0.01, 0.1, 0.5]
    sweep_h = []
    for h in h_grid:
        r = run_cell(seeds, h=h, nu=0.1, n_iters=n_iters)
        sweep_h.append(r)
        print(f"[A] h={h:<6} nu=0.1  w={r['w_mean']:7.3f}+-{r['w_std']:5.3f}  "
              f"sob_mse={r['mse_sob_mean']:.4f}  val_mse={r['mse_val_mean']:.4f}  "
              f"fd_mse={r['fd_grad_mse']:.3f} (bias2={r['fd_grad_bias2']:.3f} var={r['fd_grad_var']:.3f})")

    # (B) sweep nu at h=0.1
    nu_grid = [0.0, 0.05, 0.1, 0.25]
    sweep_nu = []
    for nu in nu_grid:
        r = run_cell(seeds, h=0.1, nu=nu, n_iters=n_iters)
        sweep_nu.append(r)
        print(f"[B] h=0.1   nu={nu:<5} w={r['w_mean']:7.3f}+-{r['w_std']:5.3f}  "
              f"sob_mse={r['mse_sob_mean']:.4f}  val_mse={r['mse_val_mean']:.4f}  "
              f"fd_mse={r['fd_grad_mse']:.3f} (bias2={r['fd_grad_bias2']:.3f} var={r['fd_grad_var']:.3f})")

    print("\n--- init-sensitivity probe (is w stuck or optimal?) ---")
    probe = init_sensitivity_probe()

    total_s = time.time() - t_start
    out = {
        "init_probe": probe,
        "config": {
            "seeds": seeds, "n_iters": n_iters, "K": 10, "K_q_train": 10,
            "K_q_test": 50, "N": 200, "meta_bs": 4, "n_test": 200,
            "grad_w0": 10.0, "lr": 5e-3,
            "sweep_A": {"h": h_grid, "nu": 0.1},
            "sweep_B": {"nu": nu_grid, "h": 0.1},
        },
        "sweep_h": sweep_h,
        "sweep_nu": sweep_nu,
        "total_s": total_s,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {RESULTS_PATH}   total {total_s:.1f}s")


if __name__ == "__main__":
    main()
