"""Meta-LSQ on a parametric integro-differential family (FastLSQ v0.3.0).

FastLSQ v0.3.0 adds closed-form INTEGRAL and integro-differential operators: the
cyclic identity that makes derivatives of sin features closed-form runs backwards
(integration = differentiation of negative order, with a stable sinc form for the
running/Volterra integral). So an integro-differential operator assembles into one
(M, N) linear-least-squares design matrix with NO autograd and NO quadrature --
exactly the property Meta-LSQ exploits for values, gradients, and PDE operators.

This extends the operator-agnostic story from {value, derivative, differential} to
{integral, integro-differential}. We demonstrate two things on the Volterra family

        u'(x) + lambda * \int_0^x u(s) ds = f(x),   u(0) = 0,   x in [0, 1],

with manufactured solution u*(x) = a sin(w x) (so f is known analytically):

  (A) FORWARD, amortised bandwidth.  Meta-learn a single (sigma, mu) across the
      family by minimising the held-out COLLOCATION RESIDUAL (label-free -- no
      reference solutions, unlike the Helmholtz appendix), and compare to a
      per-task bandwidth sweep. One closed-form solve per task at test time.

  (B) INVERSE, memory-kernel recovery.  Given noisy observations of u and the known
      forcing f, recover the unknown memory strength lambda by AdamW through the
      differentiable closed-form solve (IFT) -- lambda is a meta-parameter, just
      like (sigma, mu).

Single-thread CPU, float64.  Run:  python experiments/integro_meta.py --save results/integro.json
"""
import os
import sys
import json
import math
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FASTLSQ = os.environ.get(
    "FASTLSQ_PATH", os.path.join(os.path.dirname(os.path.dirname(_HERE)), "FastLSQ"))
if _FASTLSQ not in sys.path:
    sys.path.insert(0, _FASTLSQ)
import torch
from fastlsq import SinusoidalBasis, Op, IntegralOperator, solve_lstsq

torch.set_default_dtype(torch.float64)
SEED = 0
L = 1.0
W_IC = 100.0


# ---------------- task family ----------------
def manufactured(a, w, lam):
    """u*(x) = a sin(w x);  f = a w cos(w x) + lam * a (1 - cos(w x))/w."""
    u_star = lambda x: a * torch.sin(w * x)
    f_rhs = lambda x: a * w * torch.cos(w * x) + lam * a * (1.0 - torch.cos(w * x)) / w
    return u_star, f_rhs


def sample_family(n, gen):
    a = torch.rand(n, generator=gen) * 1.5 + 0.5     # [0.5, 2]
    w = torch.rand(n, generator=gen) * 3.0 + 1.0     # [1, 4]
    lam = torch.rand(n, generator=gen) * 4.5 + 0.5   # [0.5, 5]
    return list(zip(a.tolist(), w.tolist(), lam.tolist()))


def make_base(N, seed):
    g = torch.Generator().manual_seed(seed)
    Wb = torch.randn(1, N, generator=g)
    b = torch.rand(1, N, generator=g) * 2 * math.pi
    return Wb, b


def assemble(basis, lam, x_col, x_ic, f_rhs):
    """Integro-differential design + IC row (closed form, no autograd/quadrature)."""
    op = Op.partial(0, 1, d=1) + lam * IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    A = torch.cat([op.apply(basis, x_col), W_IC * basis.evaluate(x_ic)])
    b = torch.cat([f_rhs(x_col), W_IC * torch.zeros(1, 1)])
    return A, b


def solve_task(task, sigma, mu, Wb, bb, x_col, x_ic, n_test=400):
    a, w, lam = task
    basis = SinusoidalBasis((sigma * Wb), bb, normalize=True)
    u_star, f_rhs = manufactured(a, w, lam)
    A, b = assemble(basis, lam, x_col, x_ic, f_rhs)
    beta = solve_lstsq(A, b, mu=mu).squeeze(-1)
    xt = torch.linspace(0, L, n_test).reshape(-1, 1)
    u_pred = basis.evaluate(xt) @ beta
    u_true = u_star(xt).squeeze(-1)
    l2 = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-15)).item()
    return l2, beta, basis


def held_out_residual(task, sigma, mu, Wb, bb, x_tr, x_val, x_ic):
    """Label-free meta-objective: fit on x_tr collocation, score PDE+IC residual on
    a disjoint x_val (no reference solution used)."""
    a, w, lam = task
    basis = SinusoidalBasis((sigma * Wb), bb, normalize=True)
    _, f_rhs = manufactured(a, w, lam)
    A, b = assemble(basis, lam, x_tr, x_ic, f_rhs)
    beta = solve_lstsq(A, b, mu=mu)
    op = Op.partial(0, 1, d=1) + lam * IntegralOperator.volterra(dim=0, lower=0.0, d=1)
    r = op.apply(basis, x_val) @ beta - f_rhs(x_val)
    return (r ** 2).mean()


# ---------------- (A) forward: amortised vs per-task bandwidth ----------------
def forward_experiment(N=400, n_train_tasks=16, n_test_tasks=50, n_col=200, seed=SEED):
    print("\n==== (A) forward: parametric Volterra integro-differential family ====")
    Wb, bb = make_base(N, seed)
    g = torch.Generator().manual_seed(seed + 1)
    x_col = torch.linspace(0.02, L - 0.02, n_col).reshape(-1, 1)
    x_tr = x_col[::2]; x_val = x_col[1::2]
    x_ic = torch.zeros(1, 1)
    train_tasks = sample_family(n_train_tasks, g)
    test_tasks = sample_family(n_test_tasks, torch.Generator().manual_seed(seed + 999))

    # meta-amortise (sigma, mu) by label-free held-out collocation residual
    log_s = torch.tensor(math.log(5.0), requires_grad=True)
    log_m = torch.tensor(math.log(1e-8), requires_grad=True)
    opt = torch.optim.Adam([log_s, log_m], lr=0.05)
    t0 = time.time()
    for it in range(150):
        opt.zero_grad()
        loss = sum(held_out_residual(t, log_s.exp(), log_m.exp(), Wb, bb, x_tr, x_val, x_ic)
                   for t in train_tasks) / len(train_tasks)
        loss.backward(); opt.step()
    t_meta = time.time() - t0
    sigma_star, mu_star = float(log_s.exp()), float(log_m.exp())
    print(f"  amortised sigma*={sigma_star:.3f}  mu*={mu_star:.2e}  (meta-train {t_meta:.1f}s, label-free)")

    # evaluate amortised vs per-task bandwidth on held-out tasks
    scales = (1.0, 2.0, 3.0, 5.0, 8.0)
    amort_l2, amort_t, auto_l2, auto_t = [], 0.0, [], 0.0
    with torch.no_grad():
        for t in test_tasks:
            ta = time.time()
            l2, _, _ = solve_task(t, sigma_star, mu_star, Wb, bb, x_col, x_ic)
            amort_t += time.time() - ta; amort_l2.append(l2)
            tb = time.time()
            best = None
            for s in scales:
                r = held_out_residual(t, s, mu_star, Wb, bb, x_tr, x_val, x_ic).item()
                if best is None or r < best[0]:
                    best = (r, s)
            l2b, _, _ = solve_task(t, best[1], mu_star, Wb, bb, x_col, x_ic)
            auto_t += time.time() - tb; auto_l2.append(l2b)
    amort_mean = float(torch.tensor(amort_l2).mean())
    auto_mean = float(torch.tensor(auto_l2).mean())
    print(f"  amortised: mean L2 {amort_mean:.3e}  ({amort_t/len(test_tasks)*1e3:.2f} ms/task)")
    print(f"  per-task : mean L2 {auto_mean:.3e}  ({auto_t/len(test_tasks)*1e3:.2f} ms/task)")
    return {"sigma_star": sigma_star, "mu_star": mu_star, "t_meta_train": t_meta,
            "amort_mean_L2": amort_mean, "auto_mean_L2": auto_mean,
            "amort_ms_per_task": amort_t / len(test_tasks) * 1e3,
            "auto_ms_per_task": auto_t / len(test_tasks) * 1e3,
            "n_test": len(test_tasks), "N": N}


# ---------------- (B) inverse: recover the memory strength lambda ----------------
def inverse_experiment(N=400, sigma=5.0, mu=1e-8, seed=SEED,
                       lams=(0.5, 1.0, 2.0, 4.0), noises=(0.0, 0.02, 0.05), n_data=60):
    print("\n==== (B) inverse: recover memory-kernel strength lambda ====")
    Wb, bb = make_base(N, seed)
    basis = SinusoidalBasis((sigma * Wb), bb, normalize=True)
    x_ic = torch.zeros(1, 1)
    x_col = torch.linspace(0.02, L - 0.02, 200).reshape(-1, 1)
    a, w = 1.0, 3.0
    rows = []
    for lam_true in lams:
        for noise in noises:
            g = torch.Generator().manual_seed(seed + int(1000 * lam_true) + int(1e4 * noise))
            u_star, f_rhs = manufactured(a, w, lam_true)
            x_data = torch.rand(n_data, 1, generator=g) * L
            u_obs = u_star(x_data).squeeze(-1) + noise * torch.randn(n_data, generator=g)
            lam = torch.tensor(1.0, requires_grad=True)   # init away from truth
            opt = torch.optim.Adam([lam], lr=0.1)
            for it in range(200):
                opt.zero_grad()
                op = Op.partial(0, 1, d=1) + lam * IntegralOperator.volterra(dim=0, lower=0.0, d=1)
                A = torch.cat([op.apply(basis, x_col), W_IC * basis.evaluate(x_ic)])
                b = torch.cat([f_rhs(x_col), W_IC * torch.zeros(1, 1)])
                beta = solve_lstsq(A, b, mu=mu)
                u_fit = (basis.evaluate(x_data) @ beta).squeeze(-1)
                loss = (u_fit - u_obs).pow(2).mean()
                loss.backward(); opt.step()
            lam_hat = float(lam.detach())
            rel = abs(lam_hat - lam_true) / lam_true
            rows.append({"lam_true": lam_true, "noise": noise, "lam_hat": lam_hat, "rel_err": rel})
            print(f"  lambda*={lam_true:.2f} noise={noise:.2f}  ->  lambda_hat={lam_hat:.3f}  (rel err {rel:.1%})")
    return rows


def main(save=None):
    t0 = time.time()
    fwd = forward_experiment()
    inv = inverse_experiment()
    out = {"forward": fwd, "inverse": inv, "fastlsq_version": "0.3.0"}
    print(f"\nintegro-differential campaign done in {time.time()-t0:.0f}s")
    if save:
        with open(save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {save}")
    return out


if __name__ == "__main__":
    import sys as _s
    _save = _s.argv[_s.argv.index("--save") + 1] if "--save" in _s.argv else None
    main(save=_save)
