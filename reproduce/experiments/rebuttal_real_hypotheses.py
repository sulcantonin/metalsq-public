"""Rebuttal-stage tests of what the paper and the author responses claim about
real data, plus the shared-hyperparameter GP control reviewer dX2X asked for.

Everything reuses the protocol of experiments/real_calibration.py verbatim:
UCI Air Quality, one ISO week per task, chronological 60/40 split, K=10 support
and 50 query points, standardisation on meta-train weeks only, and the same
seeded support/query draws for every method within a seed.

Hypotheses tested
  H1  (regime)   The advantage over per-task polynomial fitting grows with the
                 target's nonlinearity. The paper ran the least and most
                 nonlinear of the sensor/reference pairs; here all four pairs
                 with enough weeks are run (NMHC has only 8 weeks and is skipped).
  H2  (FD rows)  Finite-differencing the observed support values yields
                 derivative rows that add no information (paper, Sec. real data;
                 response to vgVV, point 2). Test: Meta-LSQ-Sobolev with 3-point
                 local slopes from the K support values and a learnable w,
                 against the value-only fit on identical draws.
  H3  (ridge)    The learned ridge on noisy real data is noise-calibrated but not
                 tail-optimal: a deployment floor still helps (responses, ridge
                 recipe). Test: evaluate the trained value-only model at
                 mu in {mu*, 1e-3, 1e-2, 1e-1, 1} on held-out weeks.
  H4  (dX2X)     An exact GP whose hyperparameters are shared across tasks and
                 meta-learned by query loss (no per-task marginal likelihood)
                 matches Meta-LSQ: value-only RBF-GP on the real data, and
                 gradient-enhanced RBF-GP (GEK) on the 10-shot sinusoid benchmark
                 against the same 200 test tasks per seed as Table 1.

Run (each target is an independent process; timings are not comparable across
concurrently running processes and are not reported as such):
  OMP_NUM_THREADS=1 python3 experiments/rebuttal_real_hypotheses.py --part real --target CO   --save results/rebuttal_real_CO.json
  OMP_NUM_THREADS=1 python3 experiments/rebuttal_real_hypotheses.py --part sin               --save results/rebuttal_gek_shared.json
"""

import os, sys, math, json, time, argparse

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import (
    make_basis, fastlsq_inner_solve, fastlsq_predict, sample_tasks,
)
from experiments.real_calibration import (
    load_tasks, split_tasks, normaliser, draw, batch, nonlinearity,
    train_metalsq, train_maml, eval_maml, eval_global_linear, K, K_Q,
)
from experiments.periodic_gek import k_rbf, gek_gram

torch.set_default_dtype(torch.float32)
SEEDS = (0, 1, 2, 3, 4)

# Every sensor/reference pair in the dataset with enough weeks to be a task
# family. Nonlinearity gain (deg-5 R^2 minus linear R^2 on meta-train weeks)
# is recomputed at run time; the screen gives CO +0.005, NO2 +0.021,
# C6H6 +0.031, NOx +0.271.
TARGETS = {
    "CO":   ("PT08.S1(CO)",   "CO(GT)"),
    "NO2":  ("PT08.S4(NO2)",  "NO2(GT)"),
    "C6H6": ("PT08.S2(NMHC)", "C6H6(GT)"),
    "NOx":  ("PT08.S3(NOx)",  "NOx(GT)"),
}
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0)


# ---------------- statistics ----------------

def summarise(per_seed_lists):
    """per_seed_lists: list over seeds of per-task MSE lists."""
    means = [float(np.mean(v)) for v in per_seed_lists]
    meds = [float(np.median(v)) for v in per_seed_lists]
    pooled = np.concatenate([np.asarray(v) for v in per_seed_lists])
    sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return {"mean": float(np.mean(means)), "mean_std": sd(means),
            "median": float(np.mean(meds)), "median_std": sd(meds),
            "pooled_p90": float(np.percentile(pooled, 90)),
            "pooled_max": float(pooled.max()),
            "per_seed_mean": means, "per_seed_median": meds}


# ---------------- H2: finite-difference derivative rows ----------------

def fd_gradients(xs, ys):
    """Slope at each support point from the support VALUES alone: least-squares
    slope through the point and its two nearest neighbours in x (a central
    difference on an irregular grid). Degenerate stencils (coincident x) fall
    back to the global support slope."""
    x = xs.squeeze(-1).numpy().astype(np.float64)
    y = ys.numpy().astype(np.float64)
    n = x.size
    order = np.argsort(x)
    xo, yo = x[order], y[order]
    span = lambda a: float(a.max() - a.min())
    g_global = np.polyfit(xo, yo, 1)[0] if span(xo) > 1e-9 else 0.0
    g = np.empty(n)
    for r in range(n):
        lo = max(0, min(r - 1, n - 3))
        xx, yy = xo[lo:lo + 3], yo[lo:lo + 3]
        g[order[r]] = np.polyfit(xx, yy, 1)[0] if span(xx) > 1e-9 else g_global
    return torch.tensor(g, dtype=torch.float32)


def train_metalsq_fd(tasks, norm, n_iters=2000, N=200, lr=5e-3, bs=4, seed=0, w0=1.0):
    """Meta-LSQ-Sobolev with finite-difference support slopes; (sigma, mu, w) learned."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)   # same task draws as value-only
    Wb, bb = make_basis(N=N, seed=seed)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(w0), requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu, log_gw], lr=lr)
    t0 = time.time()
    for _ in range(n_iters):
        opt.zero_grad()
        loss = 0.0
        for xs_, ys_, xq_, yq_ in batch(tasks, norm, gen, bs):
            dys_ = fd_gradients(xs_, ys_)
            W, beta = fastlsq_inner_solve(xs_, ys_, dys_, Wb, bb, log_sigma.exp(),
                                          log_mu.exp(), sobolev=True, grad_w=log_gw.exp())
            loss = loss + (fastlsq_predict(xq_, W, bb, beta) - yq_).pow(2).mean()
        (loss / bs).backward()
        opt.step()
    return {"Wb": Wb, "bb": bb, "sigma": float(log_sigma.exp()),
            "mu": float(log_mu.exp()), "w": float(log_gw.exp()),
            "train_s": time.time() - t0}


def eval_metalsq_generic(m, tasks, norm, seed=0, mu=None, sobolev=False):
    """Held-out per-task query MSE; identical draws to every other method (seed+9999)."""
    gen = torch.Generator().manual_seed(seed + 9999)
    mu_use = m["mu"] if mu is None else mu
    out = []
    with torch.no_grad():
        for t in tasks:
            xs_, ys_, xq_, yq_ = draw(t, norm, gen)
            dys_ = fd_gradients(xs_, ys_) if sobolev else ys_
            W, beta = fastlsq_inner_solve(xs_, ys_, dys_, m["Wb"], m["bb"],
                                          torch.tensor(m["sigma"]), torch.tensor(mu_use),
                                          sobolev=sobolev, grad_w=m.get("w", 1.0))
            out.append(float((fastlsq_predict(xq_, W, m["bb"], beta) - yq_).pow(2).mean()))
    return out


# ---------------- classical baselines (per-task polynomial) ----------------

def eval_pertask_poly(tasks, norm, seed=0, deg=1):
    gen = torch.Generator().manual_seed(seed + 9999)
    out = []
    for t in tasks:
        xs_, ys_, xq_, yq_ = draw(t, norm, gen)
        c = np.polyfit(xs_.squeeze(-1).numpy(), ys_.numpy(), deg)
        out.append(float(((np.polyval(c, xq_.squeeze(-1).numpy()) - yq_.numpy()) ** 2).mean()))
    return out


# ---------------- H4: exact GP with shared, meta-learned hyperparameters ----------------

def _chol(A):
    """Cholesky with escalating jitter; the learned noise can get very small."""
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype)
    for jit in (0.0, 1e-8, 1e-6, 1e-4):
        L, info = torch.linalg.cholesky_ex(A + jit * I)
        if int(info) == 0:
            return L
    return torch.linalg.cholesky(A + 1e-2 * I)


def gp_rbf_mean(xs, ys, xq, l, lam):
    """Exact GP posterior mean, RBF kernel with unit signal variance, noise lam.
    Value observations only. float64 throughout."""
    xs = xs.squeeze(-1).double(); xq = xq.squeeze(-1).double(); ys = ys.double()
    R = xs[:, None] - xs[None, :]
    Kss = torch.exp(-R ** 2 / (2 * l ** 2)) + lam * torch.eye(xs.numel(), dtype=torch.float64)
    alpha = torch.cholesky_solve(ys.unsqueeze(-1), _chol(Kss)).squeeze(-1)
    Rq = xq[:, None] - xs[None, :]
    return torch.exp(-Rq ** 2 / (2 * l ** 2)) @ alpha


def gek_rbf_mean(xs, ys, dys, xq, l, lam_y, lam_g):
    """Exact gradient-enhanced kriging posterior mean (values and derivatives),
    RBF kernel, separate value/derivative noise. Same Gram conventions as
    experiments/periodic_gek.py."""
    xs = xs.squeeze(-1).double(); xq = xq.squeeze(-1).double()
    n = xs.numel()
    R = xs[:, None] - xs[None, :]
    G = gek_gram(*k_rbf(R, l))
    noise = torch.cat([lam_y.expand(n), lam_g.expand(n)])
    A = G + torch.diag(noise)
    t = torch.cat([ys.double(), dys.double()])
    alpha = torch.cholesky_solve(t.unsqueeze(-1), _chol(A)).squeeze(-1)
    Rq = xq[:, None] - xs[None, :]
    kq, d1q, _ = k_rbf(Rq, l)
    return torch.cat([kq, -d1q], dim=-1) @ alpha


def train_gp_shared_real(tasks, norm, n_iters=2000, lr=5e-3, bs=4, seed=0):
    """Shared (length-scale, noise) meta-learned by query MSE; same draws as Meta-LSQ."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    log_l = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    log_lam = torch.tensor(math.log(1e-2), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_l, log_lam], lr=lr)
    t0 = time.time()
    for _ in range(n_iters):
        opt.zero_grad()
        loss = 0.0
        for xs_, ys_, xq_, yq_ in batch(tasks, norm, gen, bs):
            pred = gp_rbf_mean(xs_, ys_, xq_, log_l.exp(), log_lam.exp())
            loss = loss + (pred - yq_.double()).pow(2).mean()
        (loss / bs).backward()
        opt.step()
    return {"l": float(log_l.exp()), "lam": float(log_lam.exp()), "train_s": time.time() - t0}


def eval_gp_shared_real(m, tasks, norm, seed=0):
    gen = torch.Generator().manual_seed(seed + 9999)
    out = []
    l, lam = torch.tensor(m["l"], dtype=torch.float64), torch.tensor(m["lam"], dtype=torch.float64)
    with torch.no_grad():
        for t in tasks:
            xs_, ys_, xq_, yq_ = draw(t, norm, gen)
            pred = gp_rbf_mean(xs_, ys_, xq_, l, lam)
            out.append(float((pred - yq_.double()).pow(2).mean()))
    return out


def train_gek_shared_sin(sobolev, n_iters=2000, bs=4, Kt=10, Kq=10, lr=5e-3, seed=0,
                         l0=1.0, lam0=1e-2):
    """Sinusoid benchmark: shared (l, lam_y[, lam_g]) meta-learned by query MSE
    through the exact GP/GEK posterior mean. Same task stream as Meta-LSQ
    (generator seed+1000, batch 4, K=10, K_q=10). ``l0``/``lam0`` initialise the
    length-scale and both noise terms."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    log_l = torch.tensor(math.log(l0), dtype=torch.float64, requires_grad=True)
    log_ly = torch.tensor(math.log(lam0), dtype=torch.float64, requires_grad=True)
    log_lg = torch.tensor(math.log(lam0), dtype=torch.float64, requires_grad=True)
    params = [log_l, log_ly] + ([log_lg] if sobolev else [])
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    for _ in range(n_iters):
        xs, ys, dys, xq, yq = sample_tasks(bs, Kt, Kq, gen)
        opt.zero_grad()
        loss = 0.0
        for t in range(bs):
            if sobolev:
                pred = gek_rbf_mean(xs[t], ys[t], dys[t], xq[t], log_l.exp(), log_ly.exp(), log_lg.exp())
            else:
                pred = gp_rbf_mean(xs[t], ys[t], xq[t], log_l.exp(), log_ly.exp())
            loss = loss + (pred - yq[t].double()).pow(2).mean()
        (loss / bs).backward()
        opt.step()
    return {"l": float(log_l.exp()), "lam_y": float(log_ly.exp()),
            "lam_g": float(log_lg.exp()) if sobolev else None,
            "sobolev": sobolev, "train_s": time.time() - t0}


def eval_gek_shared_sin(m, n_test=200, Kt=10, Kq=50, seed=0):
    """Same 200 held-out tasks per seed as eval_fastlsq_meta (generator seed+9999)."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, Kt, Kq, gen)
    l = torch.tensor(m["l"], dtype=torch.float64)
    ly = torch.tensor(m["lam_y"], dtype=torch.float64)
    lg = torch.tensor(m["lam_g"] if m["sobolev"] else 0.0, dtype=torch.float64)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for t in range(n_test):
            if m["sobolev"]:
                pred = gek_rbf_mean(xs[t], ys[t], dys[t], xq[t], l, ly, lg)
            else:
                pred = gp_rbf_mean(xs[t], ys[t], xq[t], l, ly)
            out.append(float((pred - yq[t].double()).pow(2).mean()))
    return out, (time.time() - t0) / n_test * 1e3


# ---------------- drivers ----------------

def run_real(target, seeds, args):
    sensor, ref = TARGETS[target]
    tasks, weeks = load_tasks(sensor, ref)
    tr, te = split_tasks(tasks)
    norm = normaliser(tr)
    nl = nonlinearity(tr)
    print(f"### {target}: {sensor} -> {ref}; weeks {len(tasks)} (train {len(tr)}, test {len(te)}); "
          f"nonlinearity gain {nl['gain']:+.3f}", flush=True)

    per = {}                     # method -> list over seeds of per-task lists
    add = lambda k, v: per.setdefault(k, []).append(v)
    learned = {"metalsq_sigma": [], "metalsq_mu": [], "fd_sigma": [], "fd_mu": [], "fd_w": [],
               "gp_l": [], "gp_lam": []}
    train_s = {"metalsq": [], "fd": [], "gp": [], "maml": []}

    for s in seeds:
        m = train_metalsq(tr, norm, n_iters=args.iters_lsq, seed=s)
        learned["metalsq_sigma"].append(m["sigma"]); learned["metalsq_mu"].append(m["mu"])
        train_s["metalsq"].append(m["train_s"])
        add("Meta-LSQ (w=0)", eval_metalsq_generic(m, te, norm, seed=s))
        for mu in RIDGE_GRID:                                   # H3
            add(f"Meta-LSQ (w=0, mu>={mu:g})", eval_metalsq_generic(m, te, norm, seed=s, mu=max(m["mu"], mu)))

        mf = train_metalsq_fd(tr, norm, n_iters=args.iters_lsq, seed=s)       # H2
        learned["fd_sigma"].append(mf["sigma"]); learned["fd_mu"].append(mf["mu"]); learned["fd_w"].append(mf["w"])
        train_s["fd"].append(mf["train_s"])
        add("Meta-LSQ FD-Sobolev", eval_metalsq_generic(mf, te, norm, seed=s, sobolev=True))
        add("Meta-LSQ FD-Sobolev (mu>=0.01)", eval_metalsq_generic(mf, te, norm, seed=s, mu=max(mf["mu"], 1e-2), sobolev=True))

        g = train_gp_shared_real(tr, norm, n_iters=args.iters_lsq, seed=s)    # H4 (real)
        learned["gp_l"].append(g["l"]); learned["gp_lam"].append(g["lam"]); train_s["gp"].append(g["train_s"])
        add("GP-RBF, shared meta-learned hp", eval_gp_shared_real(g, te, norm, seed=s))

        if args.iters_maml > 0:
            net, t_ = train_maml(tr, norm, n_iters=args.iters_maml, seed=s)
            train_s["maml"].append(t_)
            add("MLP-MAML", eval_maml(net, te, norm, seed=s))

        add("per-task linear", eval_pertask_poly(te, norm, seed=s, deg=1))
        add("per-task quadratic", eval_pertask_poly(te, norm, seed=s, deg=2))
        add("global linear", eval_global_linear(tr, te, norm, seed=s))
        print(f"  seed {s}: sigma {m['sigma']:.3f} mu {m['mu']:.2e} | FD w {mf['w']:.3f} mu {mf['mu']:.2e} "
              f"| GP l {g['l']:.3f} lam {g['lam']:.2e}", flush=True)

    methods = {k: summarise(v) for k, v in per.items()}
    print(f"\n=== {target}: held-out weeks, normalised query MSE ({len(seeds)} seeds) ===")
    print(f"{'method':<34}{'mean':>16}{'median':>16}{'p90':>9}{'max':>10}")
    for k, v in methods.items():
        print(f"{k:<34}{v['mean']:>9.3f} +/-{v['mean_std']:<6.3f}{v['median']:>9.3f} +/-{v['median_std']:<6.3f}"
              f"{v['pooled_p90']:>9.3f}{v['pooled_max']:>10.3f}")
    return {"target": target, "sensor": sensor, "ref": ref, "n_tasks": len(tasks),
            "n_train": len(tr), "n_test": len(te), "K": K, "K_q": K_Q, "seeds": list(seeds),
            "nonlinearity": nl, "methods": methods, "learned": learned,
            "train_s": {k: float(np.mean(v)) for k, v in train_s.items() if v},
            "per_task": per}


def run_sin(seeds, args):
    out = {}
    for name, sob in (("GEK-RBF shared hp (values+derivatives)", True),
                      ("GP-RBF shared hp (values only)", False)):
        per, ms, learned = [], [], []
        for s in seeds:
            m = train_gek_shared_sin(sob, n_iters=args.iters_lsq, seed=s)
            v, dt = eval_gek_shared_sin(m, seed=s)
            per.append(v); ms.append(dt); learned.append(m)
            print(f"  {name} seed {s}: mean {np.mean(v):.4f} median {np.median(v):.2e} "
                  f"| l {m['l']:.3f} lam_y {m['lam_y']:.2e} lam_g {m['lam_g']} | {dt:.2f} ms/task, "
                  f"train {m['train_s']:.1f}s", flush=True)
        out[name] = {**summarise(per), "ms_per_task": float(np.mean(ms)),
                     "train_s": float(np.mean([m["train_s"] for m in learned])),
                     "learned": learned, "per_task": per}
        print(f"=> {name}: {out[name]['mean']:.4f} +/- {out[name]['mean_std']:.4f} "
              f"(median {out[name]['median']:.2e})")
    return {"n_test": 200, "K": 10, "K_q": 50, "seeds": list(seeds), "methods": out,
            "reference": "Meta-LSQ-Sobolev 0.019+/-0.016, Meta-LSQ w=0 0.060+/-0.043 (Table 1, same seeds/test tasks)"}


def run_relfloor(target, seeds, args):
    """H3 follow-up: a deployment floor RELATIVE to the learned ridge, mu = c * mu*,
    against the absolute floor 1e-2. Retrains the value-only model (deterministic
    per seed, so it is the same model as in run_real)."""
    sensor, ref = TARGETS[target]
    tasks, _ = load_tasks(sensor, ref)
    tr, te = split_tasks(tasks)
    norm = normaliser(tr)
    per, mus = {}, []
    add = lambda k, v: per.setdefault(k, []).append(v)
    for s in seeds:
        m = train_metalsq(tr, norm, n_iters=args.iters_lsq, seed=s)
        mus.append(m["mu"])
        for c in (1, 3, 10, 30, 100):
            add(f"c={c}", eval_metalsq_generic(m, te, norm, seed=s, mu=c * m["mu"]))
        add("abs 1e-2", eval_metalsq_generic(m, te, norm, seed=s, mu=max(m["mu"], 1e-2)))
    methods = {k: summarise(v) for k, v in per.items()}
    print(f"=== {target}: mu = c * mu* (mu* = {np.mean(mus):.2e}) ===")
    for k, v in methods.items():
        print(f"  {k:<9} mean {v['mean']:.4f}+/-{v['mean_std']:.4f}  median {v['median']:.4f}  "
              f"p90 {v['pooled_p90']:.3f}  max {v['pooled_max']:.3f}")
    return {"target": target, "seeds": list(seeds), "mu_star": mus, "methods": methods}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("real", "sin", "relfloor"), required=True)
    ap.add_argument("--target", choices=tuple(TARGETS), default="CO")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters-lsq", type=int, default=2000)
    ap.add_argument("--iters-maml", type=int, default=8000)
    ap.add_argument("--save", required=True)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    t0 = time.time()
    if args.part == "real":
        blob = run_real(args.target, seeds, args)
    elif args.part == "relfloor":
        blob = run_relfloor(args.target, seeds, args)
    else:
        blob = run_sin(seeds, args)
    blob["wall_s_total"] = time.time() - t0
    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"\n[saved] {p}  ({blob['wall_s_total']:.0f}s)")


if __name__ == "__main__":
    main()
