"""Structurally-matched GP baselines with derivative observations.

The paper's closest mathematical comparator is gradient-enhanced kriging with an
RBF kernel (experiments/review_experiments.py, part A), which reaches 0.086 on
the 10-shot sinusoid benchmark. That kernel cannot express periodicity, so the
comparison is against a structurally mismatched GP. This script adds the
matched comparators:

  rbf        the paper's kernel, re-run here so every row shares one protocol
  periodic   MacKay's exp(-2 sin^2(pi r / p) / l^2), period selected per task
  sm1        one spectral-mixture component, exp(-2 pi^2 v r^2) cos(2 pi m r);
             as v -> 0 its sample paths are exactly A sin(w x + phi), i.e. the
             task family itself

All three get derivative rows (the same support gradients Meta-LSQ-Sobolev
sees), per-task hyperparameters by marginal likelihood over a grid, and an
analytically profiled signal variance -- so the amplitude a ~ U[0.1,5] costs
them nothing. The RBF row in the paper has a fixed unit signal variance and a
fixed jitter; `--paper-rbf` reproduces that weaker setting for reference.

Reported: test MSE over the same 200 held-out tasks / 5 seeds as Table 1, and
per-task wall-clock, which is the axis the amortisation argument runs on.

Run:  OMP_NUM_THREADS=1 python3 experiments/periodic_gek.py --save results/periodic_gek.json
"""
import os, sys, json, math, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

torch.set_default_dtype(torch.float32)
from experiments.maml_sinusoid import sample_tasks

SEEDS = (0, 1, 2, 3, 4)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(_REPO_ROOT, "results", "periodic_gek.json")


def _ms(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return {"mean": float(t.mean()),
            "std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0}


# ---------------- kernels: return k, k', k'' as functions of r = x - x' ----------------
# Each returns a triple of tensors shaped like r. Params are 1-D tensors of grid
# values; r is broadcast against a leading grid axis by the caller.

def k_rbf(r, l):
    """exp(-r^2 / 2 l^2)."""
    k = torch.exp(-r ** 2 / (2 * l ** 2))
    d1 = -(r / l ** 2) * k
    d2 = (r ** 2 / l ** 4 - 1.0 / l ** 2) * k
    return k, d1, d2


def k_periodic(r, l, p):
    """MacKay periodic kernel exp(-2 sin^2(pi r / p) / l^2)."""
    u = math.pi * r / p
    k = torch.exp(-2.0 * torch.sin(u) ** 2 / l ** 2)
    # k = exp(g), g = -2 sin^2(u)/l^2
    g1 = -(2.0 * math.pi / (p * l ** 2)) * torch.sin(2 * u)
    g2 = -(4.0 * math.pi ** 2 / (p ** 2 * l ** 2)) * torch.cos(2 * u)
    d1 = g1 * k
    d2 = (g2 + g1 ** 2) * k
    return k, d1, d2


def k_sm1(r, v, m):
    """One spectral-mixture component: exp(-2 pi^2 v r^2) cos(2 pi m r)."""
    al = 2.0 * math.pi ** 2 * v
    be = 2.0 * math.pi * m
    E = torch.exp(-al * r ** 2)
    C = torch.cos(be * r)
    S = torch.sin(be * r)
    k = E * C
    d1 = E * (-2 * al * r * C - be * S)
    d2 = E * ((4 * al ** 2 * r ** 2 - 2 * al - be ** 2) * C + 4 * al * be * r * S)
    return k, d1, d2


# ---------------- gradient-enhanced Gram ----------------

def gek_gram(k, d1, d2):
    """Assemble [[Kff, Kfg], [Kgf, Kgg]] from k(r), k'(r), k''(r) with r = x_i - x_j.

    cov(f_i, f'_j)  = d/dx_j k(x_i - x_j) = -k'(r)
    cov(f'_i, f'_j) = -k''(r)
    """
    Kff = k
    Kfg = -d1
    Kgg = -d2
    top = torch.cat([Kff, Kfg], dim=-1)
    bot = torch.cat([Kfg.transpose(-1, -2), Kgg], dim=-1)
    return torch.cat([top, bot], dim=-2)


def profile_marglik(G, t, lam):
    """Profile log marginal likelihood with signal variance integrated out.

    For kernel s^2 * (G + lam I) the ML-optimal s^2 is t' (G+lam I)^-1 t / n, and
    the profiled objective is -(n/2) log s^2_hat - (1/2) log|G + lam I|. Grid
    points whose Cholesky fails are returned as -inf.
    """
    n = G.shape[-1]
    A = G + lam * torch.eye(n, dtype=G.dtype).expand_as(G)
    L, info = torch.linalg.cholesky_ex(A)
    ok = info == 0
    Lsafe = torch.where(ok[..., None, None], L, torch.eye(n, dtype=G.dtype).expand_as(L))
    a = torch.cholesky_solve(t.expand(*Lsafe.shape[:-2], n).unsqueeze(-1), Lsafe).squeeze(-1)
    quad = (t * a).sum(-1).clamp_min(1e-300)
    logdet = 2.0 * torch.log(torch.diagonal(Lsafe, dim1=-2, dim2=-1)).sum(-1)
    score = -(n / 2.0) * torch.log(quad / n) - 0.5 * logdet
    return torch.where(ok & torch.isfinite(score), score, torch.full_like(score, -float("inf")))


def gek_predict(kfun, params, xs, ys, dys, xq, lam):
    """Posterior mean at xq for one task under one hyperparameter setting."""
    R = xs[:, None] - xs[None, :]
    G = gek_gram(*kfun(R, *params))
    n = G.shape[-1]
    A = G + lam * torch.eye(n, dtype=G.dtype)
    t = torch.cat([ys, dys])
    alpha = torch.cholesky_solve(t.unsqueeze(-1), torch.linalg.cholesky(A)).squeeze(-1)
    Rq = xq[:, None] - xs[None, :]
    kq, d1q, _ = kfun(Rq, *params)
    kstar = torch.cat([kq, -d1q], dim=-1)
    return kstar @ alpha


# ---------------- grids ----------------

def _log(a, b, n):
    return torch.logspace(math.log10(a), math.log10(b), n, dtype=torch.float64)


def build_grid(name):
    """Return (kfun, list_of_param_tuples, label)."""
    lams = [1e-8, 1e-6, 1e-4]
    if name == "rbf":
        ls = _log(0.1, 10.0, 32)
        return k_rbf, [(l,) for l in ls], lams
    if name == "periodic":
        ls = _log(0.2, 5.0, 14)
        ps = _log(1.0, 20.0, 44)
        return k_periodic, [(l, p) for l in ls for p in ps], lams
    if name == "sm1":
        vs = _log(1e-7, 1.0, 14)
        ms = torch.linspace(0.01, 0.6, 60, dtype=torch.float64)
        return k_sm1, [(v, m) for v in vs for m in ms], lams
    raise ValueError(name)


def run_kernel(name, n_test=200, K=10, K_q=50, seed=0):
    """Per-task ML hyperparameter selection over the grid, then posterior mean."""
    kfun, grid, lams = build_grid(name)
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)

    # Stack the whole grid once; the design depends on the task only through xs.
    mses = []
    t0 = time.time()
    for ti in range(n_test):
        xst = xs[ti].squeeze(-1).double()
        xqt = xq[ti].squeeze(-1).double()
        t = torch.cat([ys[ti].double(), dys[ti].double()])
        R = (xst[:, None] - xst[None, :])[None]          # (1,K,K)
        Gs = []
        for params in grid:
            pt = tuple(p.double() if torch.is_tensor(p) else torch.tensor(float(p)) for p in params)
            Gs.append(gek_gram(*kfun(R[0], *pt)))
        Gs = torch.stack(Gs)                             # (Gsz, 2K, 2K)
        best = (-float("inf"), None, None)
        for lam in lams:
            sc = profile_marglik(Gs, t, lam)
            j = int(torch.argmax(sc))
            if float(sc[j]) > best[0]:
                best = (float(sc[j]), grid[j], lam)
        _, params, lam = best
        pt = tuple(p.double() if torch.is_tensor(p) else torch.tensor(float(p)) for p in params)
        pred = gek_predict(kfun, pt, xst, ys[ti].double(), dys[ti].double(), xqt, lam)
        mses.append(float(((pred - yq[ti].double()) ** 2).mean()))
    dt_ms = (time.time() - t0) / n_test * 1e3
    return mses, dt_ms


# ---------------- paper's RBF setting, for reference ----------------

def run_paper_rbf(n_test=200, K=10, K_q=50, seed=0):
    """The exact setting behind the 0.086 in Table 1: unit signal variance,
    fixed jitter 1e-4, length-scale grid (0.3 .. 3.0), non-profiled marglik."""
    from experiments.review_experiments import gek_marglik, gek_solve_predict
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)
    ls = (0.3, 0.5, 0.8, 1.2, 2.0, 3.0)
    mses = []
    t0 = time.time()
    for ti in range(n_test):
        xst = xs[ti].squeeze(-1); xqt = xq[ti].squeeze(-1)
        best = None
        for l in ls:
            ml = gek_marglik(xst, ys[ti], dys[ti], l, 1e-4)
            if best is None or ml > best[0]:
                best = (ml, l)
        pred = gek_solve_predict(xst, ys[ti], dys[ti], xqt, best[1])
        mses.append(float(((pred - yq[ti]) ** 2).mean()))
    return mses, (time.time() - t0) / n_test * 1e3


# ---------------- derivative self-check ----------------

def check_derivatives(tol=1e-5):
    """Finite-difference the analytic k', k'' for every kernel. Guards the Gram."""
    r = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    r = r[r.abs() > 1e-3]
    h = 1e-5
    cases = [("rbf", k_rbf, (torch.tensor(0.8),)),
             ("periodic", k_periodic, (torch.tensor(1.3), torch.tensor(6.0))),
             ("sm1", k_sm1, (torch.tensor(0.05), torch.tensor(0.159)))]
    worst = {}
    for nm, f, p in cases:
        k0, d1, d2 = f(r, *p)
        kp = f(r + h, *p)[0]; km = f(r - h, *p)[0]
        fd1 = (kp - km) / (2 * h)
        fd2 = (kp - 2 * k0 + km) / h ** 2
        e1 = float((d1 - fd1).abs().max())
        e2 = float((d2 - fd2).abs().max())
        worst[nm] = {"d1": e1, "d2": e2}
        status = "ok" if (e1 < tol and e2 < 1e-3) else "FAIL"
        print(f"  {nm:9s} |k'-fd| {e1:.2e}  |k''-fd| {e2:.2e}   {status}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--kernels", default="rbf,periodic,sm1")
    ap.add_argument("--paper-rbf", action="store_true",
                    help="also reproduce the weaker RBF setting used in Table 1")
    args = ap.parse_args()

    print("==== derivative self-check (analytic vs finite difference) ====")
    checks = check_derivatives()

    out = {"n_test": args.n_test, "seeds": list(SEEDS[:args.seeds]), "checks": checks,
           "kernels": {}}
    seeds = SEEDS[:args.seeds]

    if args.paper_rbf:
        per_seed, times = [], []
        for s in seeds:
            m, dt = run_paper_rbf(n_test=args.n_test, seed=s)
            per_seed.append(sum(m) / len(m)); times.append(dt)
            print(f"  paper-rbf seed {s}: mean MSE {per_seed[-1]:.4f}  ({dt:.1f} ms/task)")
        out["kernels"]["rbf_paper_setting"] = {
            "test_mse": _ms(per_seed), "ms_per_task": sum(times) / len(times)}

    for name in args.kernels.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n==== {name} GEK (derivative rows, per-task ML, profiled variance) ====")
        per_seed, medians, times = [], [], []
        for s in seeds:
            m, dt = run_kernel(name, n_test=args.n_test, seed=s)
            mt = torch.tensor(m, dtype=torch.float64)
            per_seed.append(float(mt.mean())); medians.append(float(mt.median()))
            times.append(dt)
            print(f"  seed {s}: mean MSE {per_seed[-1]:.4g}  median {medians[-1]:.4g}"
                  f"  ({dt:.1f} ms/task)")
        out["kernels"][name] = {"test_mse": _ms(per_seed),
                                "test_mse_median": _ms(medians),
                                "ms_per_task": sum(times) / len(times)}
        print(f"  -> {name}: {out['kernels'][name]['test_mse']['mean']:.4g} "
              f"+/- {out['kernels'][name]['test_mse']['std']:.4g}")

    if args.save:
        p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
