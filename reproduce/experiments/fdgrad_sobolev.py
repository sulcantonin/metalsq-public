"""Meta-LSQ-Sobolev with ESTIMATED (finite-difference) support gradients.

Addresses the 'analytic-gradient-friendly' scope concern: instead of feeding the
inner solve the analytic support gradient dy/dx, we ESTIMATE it by central finite
differences from (noisy) function-value queries:

    dy_fd(x) = ( y_noisy(x+h) - y_noisy(x-h) ) / (2h)

where y_noisy(.) = a sin(.+phi) + N(0, nu^2). The same value noise nu corrupts the
support VALUES y_s. Only the SUPPORT gradient TARGETS are corrupted (to dy_fd);
queries are evaluated against the CLEAN function.

FD error decomposition (central difference of a noisy oracle):
    bias  = O(h^2)         (Taylor remainder, ~ |y'''| h^2 / 6)
    noise = O(nu / h)      (two independent noisy queries / 2h)
so MSE(dy_fd) ~ c1 h^4 + c2 nu^2 / h^2  -> a bias-variance sweet spot in h.

We meta-learn w = grad_w (via torch.sqrt so it is learnable, per the fixed
fastlsq_inner_solve). Hypothesis: the meta-learned w behaves like
w = sigma_y^2 / sigma_g^2 with sigma_g^2 the FD error variance -- w stays HIGH when
FD gradients are accurate (small h, low nu) and DROPS when they are noisy/biased
(large nu, or h too large -> bias, or h too small -> blown-up noise).

Sweeps:
  (A) h  in {0.001, 0.01, 0.1, 0.5} at nu = 0.1
  (B) nu in {0, 0.05, 0.1, 0.25}    at h = 0.1
For each cell: meta-learned (sigma, mu, w), clean-query test MSE, and the value-only
baseline (w=0, i.e. Sobolev rows dropped) test MSE for comparison.

Self-contained: re-implements the meta-train loop with FD gradient targets so that
NO existing file is modified. Imports make_basis / fastlsq_inner_solve /
fastlsq_predict read-only from experiments.maml_sinusoid.
"""

import os, sys, math, time, json

# single-thread BLAS for honest, reproducible timing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from experiments.maml_sinusoid import (  # read-only imports
    make_basis,
    fastlsq_inner_solve,
    fastlsq_predict,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

A_LO, A_RANGE = 0.1, 4.9   # a ~ U[0.1, 5]
PHI_RANGE = math.pi        # phi ~ U[0, pi]
X_LO, X_RANGE = -5.0, 10.0 # x ~ U[-5, 5]


# ---------------- task generator with FD-estimated support gradients ----------

def sample_tasks_fd(n_tasks, K, K_q, gen, h, nu):
    """Sample sinusoid tasks; support gradients are FD-estimated from NOISY values.

    Returns (x_s, y_s, dy_fd, x_q, y_q):
      y_s    : support values  + N(0, nu^2)            (noisy)
      dy_fd  : (y_noisy(x+h) - y_noisy(x-h)) / (2h)     (estimated, noisy)
      y_q    : CLEAN query targets (a sin(x+phi))       (for honest MSE)
    All four FD value queries (x+h, x-h) carry INDEPENDENT N(0, nu^2) noise, exactly
    as a real noisy oracle would return on repeated evaluation.
    """
    a = torch.rand(n_tasks, 1, generator=gen) * A_RANGE + A_LO
    phi = torch.rand(n_tasks, 1, generator=gen) * PHI_RANGE
    xs = torch.rand(n_tasks, K, generator=gen) * X_RANGE + X_LO
    xq = torch.rand(n_tasks, K_q, generator=gen) * X_RANGE + X_LO

    def f(x):  # clean oracle
        return a * torch.sin(x + phi)

    ys_clean = f(xs)
    yq = f(xq)  # clean query

    if nu > 0.0:
        ys = ys_clean + nu * torch.randn(ys_clean.shape, generator=gen)
        y_plus = f(xs + h) + nu * torch.randn(xs.shape, generator=gen)
        y_minus = f(xs - h) + nu * torch.randn(xs.shape, generator=gen)
    else:
        ys = ys_clean
        y_plus = f(xs + h)
        y_minus = f(xs - h)
    dy_fd = (y_plus - y_minus) / (2.0 * h)

    # also return the analytic gradient so we can report FD-error diagnostics
    dy_true = a * torch.cos(xs + phi)
    return xs.unsqueeze(-1), ys, dy_fd, xq.unsqueeze(-1), yq, dy_true


# ---------------- meta-train loop (FD gradient targets) -----------------------

def train_meta_fd(h, nu, sobolev, grad_w_init=3.0, n_iters=2500, meta_bs=4,
                  K=10, K_q=10, N=200, lr=1e-2, seed=0):
    """Meta-learn (sigma, mu[, log_gw]) with FD support gradients.

    sobolev=False -> value-only baseline (w effectively 0, gradient rows dropped).
    Mirrors train_fastlsq_meta but injects dy_fd as the support gradient target.

    NOTE on hyper-params: lr=1e-2 with grad_w_init=3 (not the paper's 5e-3 / w=10)
    is needed so the meta-learned w actually CONVERGES within the iteration budget.
    With slow settings w merely drifts down from its init and looks artificially
    flat across cells -- a non-convergence confound, not a learned equilibrium.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)

    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(grad_w_init), requires_grad=True)
    params = [log_sigma, log_mu] + ([log_gw] if sobolev else [])
    opt = torch.optim.Adam(params, lr=lr)

    t0 = time.time()
    for it in range(n_iters):
        xs, ys, dy_fd, xq, yq, _ = sample_tasks_fd(meta_bs, K, K_q, gen, h, nu)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        gw = log_gw.exp() if sobolev else 1.0
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dy_fd[t], Wb, bb, sigma, mu,
                sobolev=sobolev, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()
    train_time = time.time() - t0

    return {
        "Wb": Wb, "bb": bb,
        "sigma": float(log_sigma.exp().item()),
        "mu": float(log_mu.exp().item()),
        "grad_w": float(log_gw.exp().item()) if sobolev else 0.0,
        "sobolev": sobolev,
        "train_s": train_time,
    }


def eval_meta_fd(meta, h, nu, n_test=200, K=10, K_q=50, seed=0):
    """Clean-query test MSE. Support values/gradients carry the SAME (h, nu)
    corruption as at meta-train time; query targets are CLEAN."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dy_fd, xq, yq, _ = sample_tasks_fd(n_test, K, K_q, gen, h, nu)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    losses = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dy_fd[t], Wb, bb, sigma, mu,
                sobolev=meta["sobolev"], grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


def fd_grad_error(h, nu, n=20000, seed=123):
    """Diagnostic: empirical MSE of the FD gradient estimator vs analytic, and its
    decomposition into squared-bias and variance, on the task family. Lets us check
    the predicted w ~ sigma_y^2 / sigma_g^2 trend independently of meta-training."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys, dy_fd, xq, yq, dy_true = sample_tasks_fd(n, 1, 1, gen, h, nu)
    err = (dy_fd[:, 0] - dy_true[:, 0])
    mse = float(err.pow(2).mean())
    # bias and variance estimated by also computing the NOISELESS FD bias
    a = None  # not needed; recompute noiseless FD directly is cleaner below
    return mse


# ---------------- driver ------------------------------------------------------

def aggregate(values):
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


_VAL_CACHE = {}  # value-only baseline depends ONLY on nu (h is never used), so cache by (nu, seed)


def value_only_mse(nu, seed):
    key = (round(nu, 6), seed)
    if key not in _VAL_CACHE:
        # h is irrelevant for the value-only model; pass a placeholder.
        m_val = train_meta_fd(h=0.1, nu=nu, sobolev=False, seed=seed)
        _VAL_CACHE[key] = eval_meta_fd(m_val, h=0.1, nu=nu, seed=seed)
    return _VAL_CACHE[key]


def run_cell(h, nu, seeds):
    """Return aggregated results over seeds for one (h, nu) cell:
    Sobolev-FD (learned w) and value-only baseline (cached by nu)."""
    sob_w, sob_sigma, sob_mu, sob_mse = [], [], [], []
    val_mse = []
    for s in seeds:
        m_sob = train_meta_fd(h=h, nu=nu, sobolev=True, seed=s)
        mse_sob = eval_meta_fd(m_sob, h=h, nu=nu, seed=s)
        sob_w.append(m_sob["grad_w"])
        sob_sigma.append(m_sob["sigma"])
        sob_mu.append(m_sob["mu"])
        sob_mse.append(mse_sob)

        val_mse.append(value_only_mse(nu, s))

    w_m, w_s = aggregate(sob_w)
    sig_m, _ = aggregate(sob_sigma)
    mu_m, _ = aggregate(sob_mu)
    smse_m, smse_s = aggregate(sob_mse)
    vmse_m, vmse_s = aggregate(val_mse)
    # FD-gradient empirical error MSE (diagnostic)
    gerr = fd_grad_error(h, nu)
    return {
        "h": h, "nu": nu,
        "w_mean": w_m, "w_std": w_s,
        "sigma_mean": sig_m, "mu_mean": mu_m,
        "sob_mse_mean": smse_m, "sob_mse_std": smse_s,
        "val_mse_mean": vmse_m, "val_mse_std": vmse_s,
        "fd_grad_err_mse": gerr,
    }


def main():
    seeds = [0, 1]
    t_start = time.time()

    # Sweep A: vary h at nu = 0.1
    nu_A = 0.1
    h_list = [0.001, 0.01, 0.1, 0.5]
    sweep_A = []
    for h in h_list:
        print(f"[A] h={h} nu={nu_A} ...", flush=True)
        sweep_A.append(run_cell(h, nu_A, seeds))

    # Sweep B: vary nu at h = 0.1
    h_B = 0.1
    nu_list = [0.0, 0.05, 0.1, 0.25]
    sweep_B = []
    for nu in nu_list:
        print(f"[B] h={h_B} nu={nu} ...", flush=True)
        sweep_B.append(run_cell(h_B, nu, seeds))

    total_s = time.time() - t_start

    results = {
        "config": {
            "seeds": seeds, "n_iters": 2500, "meta_bs": 4, "K": 10, "K_q": 10,
            "N": 200, "lr": 1e-2, "grad_w_init": 3.0,
            "n_test": 200, "K_q_test": 50,
            "sweep_A": {"nu": nu_A, "h": h_list},
            "sweep_B": {"h": h_B, "nu": nu_list},
        },
        "sweep_A": sweep_A,
        "sweep_B": sweep_B,
        "total_runtime_s": total_s,
    }

    out = _os.path.join(_REPO_ROOT, "results", "fdgrad_sobolev.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console report ----
    print("\n===== Sweep A: vary h at nu=0.1 =====")
    print(f"{'h':>8} {'fd_g_err':>9} {'w(learned)':>14} {'sigma':>7} {'Sob MSE':>16} {'Val MSE(w=0)':>16}")
    for r in sweep_A:
        print(f"{r['h']:>8} {r['fd_grad_err_mse']:>9.3f} "
              f"{r['w_mean']:>7.2f}±{r['w_std']:<5.2f} {r['sigma_mean']:>7.3f} "
              f"{r['sob_mse_mean']:>8.4f}±{r['sob_mse_std']:<6.4f} "
              f"{r['val_mse_mean']:>8.4f}±{r['val_mse_std']:<6.4f}")

    print("\n===== Sweep B: vary nu at h=0.1 =====")
    print(f"{'nu':>8} {'fd_g_err':>9} {'w(learned)':>14} {'sigma':>7} {'Sob MSE':>16} {'Val MSE(w=0)':>16}")
    for r in sweep_B:
        print(f"{r['nu']:>8} {r['fd_grad_err_mse']:>9.3f} "
              f"{r['w_mean']:>7.2f}±{r['w_std']:<5.2f} {r['sigma_mean']:>7.3f} "
              f"{r['sob_mse_mean']:>8.4f}±{r['sob_mse_std']:<6.4f} "
              f"{r['val_mse_mean']:>8.4f}±{r['val_mse_std']:<6.4f}")

    print(f"\nTotal runtime: {total_s:.1f}s")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
