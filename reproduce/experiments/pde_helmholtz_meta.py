"""Meta-LSQ over a parametric 1-D Helmholtz family.

Problem:  -u''(x) + k^2 u(x) = f(x) on [0, 2 pi], u(0) = u(2 pi) = 0.

Tasks vary k ~ U[1, 10] and f a smooth random forcing (sampled as a low-frequency
sum of sinusoids). Because f is a sum of sines, each task's solution is closed form
and errors are scored against it exactly (absolute RMS on a uniform grid). An earlier
version scored against a 257-point finite-difference reference with linear
interpolation; that reference is itself ~2e-5 RMS off the exact solution, which
swamped the comparison below.

A second correction (November 2026): the amortised evaluation used to solve on 80
collocation points with UNWEIGHTED boundary rows, while meta-training and the per-task
baseline weight the two boundary rows by BC_WEIGHT = 10 and the baseline solves on all
120 points. That train/test mismatch alone made the amortised bandwidth look 27x worse
than per-task selection. Every solve now goes through `fit`, so training, evaluation
and the baseline share one boundary weight and one collocation set.

`--svd` additionally scores both methods with FastLSQ's rank-revealing least-squares
solve and no ridge (fastlsq>=0.7.1, `solve_lstsq(method="svd")`, singular values below
1e-12 relative dropped). Training always uses the differentiable ridge solve.

Two comparisons:
  (1) Meta-LSQ over (sigma, mu):  amortise basis bandwidth across tasks.
  (2) Per-task auto-bandwidth:    pick sigma by training-residual proxy per task.

Headline question: does meta-amortised bandwidth match (or beat) per-task picking
at lower total compute?

The PINN-MAML comparison is left as a TODO -- we'd need a PINN-MAML reference
implementation, which is out of scope for this file.
"""

import math
import time
import torch

torch.set_default_dtype(torch.float64)
SEED = 0
BC_WEIGHT = 10.0
L = 2 * math.pi
X_ALL = torch.linspace(0.05, L - 0.05, 120).reshape(-1, 1)   # collocation
X_BC = torch.tensor([[0.0], [L]])


# ---------------- task generator ----------------

class HelmholtzTask:
    """One Helmholtz task: k, f, and the closed-form solution u."""
    def __init__(self, k, f_amp, f_freq, f_phase):
        self.k = float(k)
        self.f_amp = f_amp.clone()
        self.f_freq = f_freq.clone()
        self.f_phase = f_phase.clone()

    def f(self, x):
        # x is (M,) or (M, 1)
        x_ = x.reshape(-1)
        out = torch.zeros_like(x_)
        for amp, freq, ph in zip(self.f_amp, self.f_freq, self.f_phase):
            out = out + amp * torch.sin(freq * x_ + ph)
        return out.reshape_as(x)

    def _particular(self, x):
        # -u'' + k^2 u = a sin(w x + p)  is solved by  a sin(w x + p) / (k^2 + w^2).
        out = torch.zeros_like(x)
        for amp, freq, ph in zip(self.f_amp, self.f_freq, self.f_phase):
            out = out + amp * torch.sin(freq * x + ph) / (self.k ** 2 + freq ** 2)
        return out

    def exact_at(self, x):
        # Particular part plus the homogeneous sinh correction that enforces
        # u(0) = u(2 pi) = 0. sinh(k L) <= sinh(20 pi) ~ 1e27, safe in float64.
        x_ = x.reshape(-1)
        L, k = 2 * math.pi, self.k
        ends = self._particular(torch.tensor([0.0, L]))
        shL = math.sinh(k * L)
        v = (self._particular(x_) - ends[0] * torch.sinh(k * (L - x_)) / shL
             - ends[1] * torch.sinh(k * x_) / shL)
        return v.reshape_as(x)


def sample_task(gen):
    k = float(torch.rand(1, generator=gen).item() * 9 + 1)
    n_modes = 3
    f_amp = torch.randn(n_modes, generator=gen) * 1.0
    f_freq = (torch.rand(n_modes, generator=gen) * 3 + 1).round()  # int freqs 1..3
    f_phase = torch.rand(n_modes, generator=gen) * 2 * math.pi
    return HelmholtzTask(k, f_amp, f_freq, f_phase)


# ---------------- FastLSQ collocation solve for one task ----------------

def make_basis(N, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(1, N, generator=g) * sigma).double()
    b = (torch.rand(1, N, generator=g) * 2 * math.pi).double()
    return W, b


def assemble_helmholtz_system(task, x_col, x_bc, W, b, k):
    """Build [PDE_op | BC_op] residual rows.
    PDE row at x:    (k^2 + W^2) sin(Wx + b) beta / sqrt(N) = f(x)
                     (u = sin(Wx+b) gives -u'' = W^2 u, so -u'' + k^2 u = (W^2+k^2) u)
    BC row at 0,2pi: sin(Wx + b) beta / sqrt(N) = 0
    """
    N = W.shape[1]
    op = ((W * W + k * k) / math.sqrt(N)) * torch.sin(x_col @ W + b)
    rhs_pde = task.f(x_col).reshape(-1)
    H_bc = torch.sin(x_bc @ W + b) / math.sqrt(N)
    rhs_bc = torch.zeros(x_bc.shape[0])
    return torch.cat([op, H_bc], dim=0), torch.cat([rhs_pde, rhs_bc], dim=0)


def fit(task, W, b, x_fit, mu, solver="ridge"):
    """Solve for beta on collocation x_fit plus the two boundary rows (weighted by
    BC_WEIGHT). "ridge": (A^T A + mu I)^{-1} A^T t, differentiable in W and mu.
    "svd": FastLSQ's rank-revealing solve with no ridge (evaluation only)."""
    A, rhs = assemble_helmholtz_system(task, x_fit, X_BC, W, b, task.k)
    rw = torch.ones(A.shape[0])
    rw[-2:] = BC_WEIGHT
    A, rhs = A * rw[:, None], rhs * rw
    if solver == "svd":
        from fastlsq import solve_lstsq
        return solve_lstsq(A, rhs, mu=0.0, method="svd")
    N = W.shape[1]
    return torch.linalg.solve(A.T @ A + mu * torch.eye(N).double(), A.T @ rhs)


def heldout_residual(task, W, b, beta, x_val):
    A, rhs = assemble_helmholtz_system(task, x_val, torch.zeros(0, 1).double(), W, b, task.k)
    return ((A @ beta - rhs) ** 2).mean()


def predict(x_test, W, b, beta):
    z = x_test @ W + b
    return torch.sin(z) / math.sqrt(W.shape[1]) @ beta


N_EVAL = 200   # uniform points on [0, 2 pi], endpoints included


def task_error(task, W, b, beta, n_eval=N_EVAL):
    # Absolute RMS error against the closed-form solution.
    x = torch.linspace(0.0, 2 * math.pi, n_eval).reshape(-1, 1)
    u_pred = predict(x, W, b, beta)
    u_ref = task.exact_at(x).reshape(-1)
    return float(((u_pred - u_ref) ** 2).mean().sqrt())


# ---------------- per-task auto-bandwidth baseline ----------------

def per_task_auto(task, scales=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0), N=200, mu=1e-6,
                  basis_seed=0, solver="ridge"):
    """Pick sigma per task by a HELD-OUT collocation residual: fit beta on a train
    set of collocation points (+BC), then score the PDE residual on a disjoint
    held-out set. Selecting on the *training* residual is monotone in sigma (higher
    frequency always lowers the in-sample residual) and degenerately saturates at
    the largest scale; the held-out criterion mirrors what the amortised meta-loop
    optimises, so the comparison isolates per-task vs amortised, not the criterion."""
    Wb, bb = make_basis(N, sigma=1.0, seed=basis_seed)
    x_tr, x_val = X_ALL[::2], X_ALL[1::2]
    best = None
    for s in scales:
        W = s * Wb
        resid = float(heldout_residual(task, W, bb, fit(task, W, bb, x_tr, mu, solver), x_val))
        if best is None or resid < best[0]:
            best = (resid, s)
    s = best[1]
    W = s * Wb
    return s, W, bb, fit(task, W, bb, X_ALL, mu, solver)   # final solve on all collocation


# ---------------- Meta-LSQ over Helmholtz family ----------------

def train_meta(n_iters=200, meta_bs=4, N=200, basis_seed=0, lr=5e-2, seed=SEED):
    """Meta-learn (sigma, mu) over the Helmholtz family with a LABEL-FREE outer
    objective: fit beta on a TRAIN set of collocation points (+BC), then minimise
    the PDE residual on a DISJOINT held-out collocation set. No reference
    solution enters the meta-objective -- the governing operator alone
    supervises the bandwidth, made possible because the PDE residual is closed-form
    in the sinusoidal basis (cf. the integro-differential family, App.~\\ref{app:integro})."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    log_sigma = torch.tensor(math.log(2.0), requires_grad=True)
    log_mu = torch.tensor(math.log(1e-6), requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu], lr=lr)

    # Use the SAME basis draw across iterations (frozen W shape, scaled by sigma).
    Wb_unscaled, bb = make_basis(N, sigma=1.0, seed=basis_seed)
    x_tr = X_ALL[::2]            # collocation for the inner solve
    x_val = X_ALL[1::2]          # disjoint held-out collocation for the meta residual

    for it in range(n_iters):
        tasks = [sample_task(gen) for _ in range(meta_bs)]
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        loss = 0.0
        for task in tasks:
            W = sigma * Wb_unscaled
            beta = fit(task, W, bb, x_tr, mu)       # inner solve on TRAIN collocation + BC
            # LABEL-FREE meta-loss: PDE residual on held-out collocation (no u_ref).
            loss = loss + heldout_residual(task, W, bb, beta, x_val)
        loss = loss / meta_bs
        loss.backward()
        opt.step()
        if it % 20 == 0:
            print(f"  iter {it:4d}  label-free meta-residual {float(loss.item()):.3e}  "
                  f"sigma={float(sigma):.3f}  mu={float(mu):.2e}")
    return float(log_sigma.exp().item()), float(log_mu.exp().item()), Wb_unscaled, bb


# ---------------- evaluation ----------------

N_ITERS = 1000   # 200 steps left the ridge still falling and one seed's sigma unconverged


def run_seed(seed, test_tasks, n_iters=N_ITERS, svd=False):
    """`seed` redraws the basis and the training-task stream; the test tasks are fixed.
    Per-task selection is scored twice: with its default ridge (1e-6) and with the
    ridge Meta-LSQ learned, so the bandwidth's share of the gap is visible."""
    t0 = time.perf_counter()
    sigma_star, mu_star, Wb, bb = train_meta(n_iters=n_iters, basis_seed=seed, seed=seed)
    t_meta_train = time.perf_counter() - t0

    def amortised(solver):
        errs, t0 = [], time.perf_counter()
        for task in test_tasks:
            W = sigma_star * Wb
            errs.append(task_error(task, W, bb, fit(task, W, bb, X_ALL, mu_star, solver)))
        return sum(errs) / len(errs), time.perf_counter() - t0

    def per_task(mu, solver):
        errs, scales, t0 = [], [], time.perf_counter()
        for task in test_tasks:
            s, W, b, beta = per_task_auto(task, basis_seed=seed, mu=mu, solver=solver)
            errs.append(task_error(task, W, b, beta))
            scales.append(s)
        return sum(errs) / len(errs), time.perf_counter() - t0, scales

    meta, t_meta_eval = amortised("ridge")
    auto, t_auto_eval, scales = per_task(1e-6, "ridge")
    auto_matched, _, _ = per_task(mu_star, "ridge")
    row = {"seed": seed, "sigma_star": sigma_star, "mu_star": mu_star,
           "mean_meta_L2": meta, "mean_auto_L2": auto, "mean_auto_matched_mu_L2": auto_matched,
           "t_meta_train": t_meta_train, "t_meta_eval": t_meta_eval, "t_auto_eval": t_auto_eval,
           "auto_sigma_min": min(scales), "auto_sigma_max": max(scales)}
    if svd:
        row["mean_meta_svd_L2"] = amortised("svd")[0]
        row["mean_auto_svd_L2"] = per_task(0.0, "svd")[0]
    return row


def main(seeds=(0, 1, 2, 3, 4), svd=False, save=None):
    import contextlib, io, statistics
    gen = torch.Generator().manual_seed(SEED + 9999)
    test_tasks = [sample_task(gen) for _ in range(50)]
    rows = []
    for seed in seeds:
        with contextlib.redirect_stdout(io.StringIO()):     # silence the training log
            row = run_seed(seed, test_tasks, svd=svd)
        print(f"seed {seed}: sigma*={row['sigma_star']:.2f} mu*={row['mu_star']:.1e} | "
              f"amortised {row['mean_meta_L2']:.2e} | per-task {row['mean_auto_L2']:.2e} "
              f"(same ridge {row['mean_auto_matched_mu_L2']:.2e}) | eval time ratio "
              f"{row['t_auto_eval'] / row['t_meta_eval']:.1f}"
              + (f" | svd {row['mean_meta_svd_L2']:.2e} vs {row['mean_auto_svd_L2']:.2e}"
                 if svd else ""), flush=True)
        rows.append(row)
    med = lambda k: statistics.median(r[k] for r in rows)
    out = {"_note": "Nov 2026: amortised evaluation now uses the same boundary weight and "
                    "collocation as training and the per-task baseline (it used unweighted "
                    "boundary rows and 80 points, which made it look 27x worse); "
                    f"{N_ITERS} meta-training steps; 5 seeds.",
           "error_metric": f"absolute RMS error vs closed-form exact solution at {N_EVAL} "
                           "uniform points on [0, 2pi], averaged over tasks",
           "n_test": len(test_tasks), "n_iters": N_ITERS, "seeds": list(seeds),
           "median_meta_L2": med("mean_meta_L2"), "median_auto_L2": med("mean_auto_L2"),
           "median_auto_matched_mu_L2": med("mean_auto_matched_mu_L2"),
           "median_time_ratio": statistics.median(r["t_auto_eval"] / r["t_meta_eval"] for r in rows),
           "per_seed": rows}
    if svd:
        import fastlsq
        out["fastlsq_version"] = fastlsq.__version__
        out["median_meta_svd_L2"] = med("mean_meta_svd_L2")
        out["median_auto_svd_L2"] = med("mean_auto_svd_L2")
    if save:
        import json
        with open(save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {save}")
    return out

    # TODO: head-to-head against PINN-MAML on the same Helmholtz family.
    # Requires importing a PINN-MAML implementation; deferred to follow-up work.


if __name__ == "__main__":
    import sys
    argv = sys.argv
    _save = argv[argv.index("--save") + 1] if "--save" in argv else None
    main(svd="--svd" in argv, save=_save)
