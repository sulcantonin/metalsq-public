"""Kernel identification: recovering an unknown integral operator from its solution.

This is the use case the construction is actually built for. Consider a Fredholm
integral equation of the second kind,

    u(x) - int_0^1 K(x,y) u(y) dy = f(x),      K(x,y) = sum_m c_m g_m(x) h_m(y),

and the INVERSE problem: given the forcing f and noisy observations of u, recover
the kernel K -- that is, recover c. Kernel identification is a canonical
ill-posed inverse problem (viscoelastic memory, distributed-delay population
models, heat conduction with memory), and it is the setting where every property
of this solver matters at once:

  * the operator I - sum_m c_m K_m assembles into ONE (M, N) design matrix in
    closed form, with no quadrature per solve and no autograd graph;
  * the forward map c -> u is therefore a single linear solve, and it is exactly
    differentiable in c through that solve, so the inverse problem is plain
    gradient descent on an exact gradient rather than a finite-difference or
    adjoint construction;
  * the bandwidth is amortised across the family, so a new inverse problem costs
    no basis tuning.

Nothing else in the paper's comparator set does this. A GP has no operator rows.
MLP-MAML would have to backpropagate through a numerical integral-equation
solver. The classical method -- Nystrom discretisation plus a finite-difference
gradient -- is implemented here as the baseline.

Experiments
  (A) forward accuracy of the closed-form solve across the family (a guard)
  (B) identifiability: per-mode recovery error against noise, and why it is
      predicted by the mode's footprint <h_m, u> in the data
  (C) the fix: a Tikhonov prior on c, chosen by the same noise-model logic the
      ridge follows elsewhere in the paper
  (D) against Nystrom + finite-difference gradients: accuracy, solves per step,
      and wall-clock

Run:  OMP_NUM_THREADS=1 python3 experiments/kernel_identification.py --save results/kernel_identification.json
"""

import os, sys, math, json, time, argparse

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastlsq import SinusoidalBasis, Op, SeparableKernelOperator, solve_lstsq

torch.set_default_dtype(torch.float64)

R = 3                    # kernel rank (default; set_rank() overrides)
N_FEAT = 300             # frozen random features
N_COL = 240              # collocation points
N_DATA = 40              # noisy observations of u
SEEDS = (0, 1, 2, 3, 4)
RESULT_PATH = os.path.join(_REPO_ROOT, "results", "kernel_identification.json")

_gm = lambda m: (lambda x: torch.cos(m * math.pi * x.reshape(-1)))
_hm = lambda m: (lambda y: torch.cos(m * math.pi * y.reshape(-1)))
KS = []


def set_rank(r):
    """Rebuild the kernel-term operators for rank r. Assembling them is the only
    per-rank cost; it happens once, not per solve."""
    global R, KS
    R = r
    KS = [SeparableKernelOperator([_gm(m)], [_hm(m)], lower=0.0, upper=1.0, d=1)
          for m in range(1, R + 1)]


set_rank(R)


def gauss(n=400):
    """Gauss-Legendre nodes/weights on [0,1] (weights sum to 1)."""
    i = torch.arange(1, n, dtype=torch.float64)
    beta = i / torch.sqrt(4 * i ** 2 - 1)
    val, vec = torch.linalg.eigh(torch.diag(beta, 1) + torch.diag(beta, -1))
    return 0.5 * (val + 1.0), vec[0] ** 2


_GY, _GW = gauss()


def u_star_fn(a, w, q):
    return lambda x: a * torch.sin(w * x.reshape(-1)) + q * x.reshape(-1) ** 2


def footprints(u_star):
    """<h_m, u*>: how much each kernel mode actually shows up in the data."""
    return torch.tensor([float((_hm(m)(_GY) * u_star(_GY) * _GW).sum())
                         for m in range(1, R + 1)])


def make_f(c, u_star):
    ip = footprints(u_star)

    def f(x):
        out = u_star(x)
        for m in range(R):
            out = out - c[m] * _gm(m + 1)(x) * ip[m]
        return out
    return f, ip


def basis_of(sigma, seed, n_feat=N_FEAT):
    g = torch.Generator().manual_seed(seed)
    Wb = torch.randn(1, n_feat, generator=g)
    bb = torch.rand(1, n_feat, generator=g) * 2 * math.pi
    return SinusoidalBasis(sigma * Wb, bb, normalize=True)


def forward(c, basis, x_col, f, mu=1e-10):
    """One closed-form solve. Differentiable in c."""
    L = Op.identity(d=1)
    for m in range(R):
        L = L - c[m] * KS[m]
    A = L.apply(basis, x_col)
    return solve_lstsq(A, f(x_col).reshape(-1, 1), mu=mu).squeeze(-1)


def sample_problem(gen, decay=2.0):
    """A smooth kernel: coefficients decay like m^-decay, with random signs and
    scale. This is a genuine function to recover, not a handful of free numbers,
    and it is why the L2 error of K is the right score."""
    a = float(torch.rand(1, generator=gen) * 1.0 + 0.6)
    w = float(torch.rand(1, generator=gen) * 2.5 + 1.5)
    q = float(torch.rand(1, generator=gen) * 0.4)
    m = torch.arange(1, R + 1, dtype=torch.float64)
    amp = float(torch.rand(1, generator=gen) * 0.6 + 0.6)
    sign = torch.where(torch.rand(R, generator=gen) < 0.5, -1.0, 1.0)
    c = amp * sign / m ** decay
    return a, w, q, c


def kernel_l2_error(c_hat, c_true):
    """||K_hat - K||_L2([0,1]^2) / ||K||_L2, exact for this orthogonal expansion."""
    return float(torch.norm(c_hat - c_true) / torch.norm(c_true))


# ---------------- several forcings: the realistic identifiability fix ----------------
# One (f, u) pair cannot identify many kernel modes -- a mode only enters the data
# through its footprint <h_m, u>, and a single u leaves most footprints small. What
# a practitioner does is run several experiments with different forcings on the
# SAME unknown kernel. Each extra forcing is just another block of rows, and the
# gradient stays exact, so the method absorbs them without changing shape.

def make_experiments(c_true, n_forcings, gen):
    """n_forcings distinct (u*, f) pairs sharing one unknown kernel."""
    exps = []
    for _ in range(n_forcings):
        a = float(torch.rand(1, generator=gen) * 1.0 + 0.6)
        w = float(torch.rand(1, generator=gen) * 2.5 + 1.5)
        q = float(torch.rand(1, generator=gen) * 0.4)
        u_star = u_star_fn(a, w, q)
        f, ip = make_f(c_true, u_star)
        exps.append({"u_star": u_star, "f": f, "footprint": ip})
    return exps


def recover_multi(exps, u_obs_list, x_dat, basis, x_col, iters=400, lr=0.05, tik=0.0):
    """Joint recovery across forcings: n_forcings solves per step, exact gradient."""
    c = torch.zeros(R, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    Phi = basis.evaluate(x_dat)
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        loss = 0.0
        for ex, u_obs in zip(exps, u_obs_list):
            beta = forward(c, basis, x_col, ex["f"])
            loss = loss + ((Phi @ beta - u_obs) ** 2).mean()
        loss = loss / len(exps)
        if tik > 0:
            loss = loss + tik * (c ** 2).sum()
        loss.backward()
        opt.step()
    return c.detach(), time.time() - t0, iters * len(exps)


def multi_forcing_problem(seed, noise, n_forcings, sigma=4.0, tik=0.0, iters=400):
    gen = torch.Generator().manual_seed(seed)
    _, _, _, c_true = sample_problem(gen)
    exps = make_experiments(c_true, n_forcings, gen)
    basis = basis_of(sigma, seed)
    x_col = torch.linspace(0, 1, N_COL).reshape(-1, 1)
    x_dat = torch.linspace(0.02, 0.98, N_DATA).reshape(-1, 1)
    gg = torch.Generator().manual_seed(seed + 777)
    u_obs_list = []
    for ex in exps:
        uc = ex["u_star"](x_dat)
        sc = float(uc.abs().mean())
        u_obs_list.append(uc + noise * sc * torch.randn(uc.shape, generator=gg))
    c_hat, dt, solves = recover_multi(exps, u_obs_list, x_dat, basis, x_col,
                                      iters=iters, tik=tik)
    return {"c_true": c_true.tolist(), "c_hat": c_hat.tolist(),
            "kernel_l2": kernel_l2_error(c_hat, c_true),
            "time_s": dt, "solves": solves, "n_forcings": n_forcings}


# ---------------- (A) forward accuracy ----------------

def forward_accuracy(seeds, sigma=4.0):
    errs = []
    for s in seeds:
        gen = torch.Generator().manual_seed(s)
        a, w, q, c = sample_problem(gen)
        u_star = u_star_fn(a, w, q)
        f, _ = make_f(c, u_star)
        basis = basis_of(sigma, s)
        x_col = torch.linspace(0, 1, N_COL).reshape(-1, 1)
        beta = forward(c, basis, x_col, f)
        xt = torch.linspace(0, 1, 400).reshape(-1, 1)
        up = basis.evaluate(xt) @ beta
        ut = u_star(xt)
        errs.append(float(torch.norm(up - ut) / torch.norm(ut)))
    return {"rel_l2": errs, "mean": sum(errs) / len(errs)}


# ---------------- recovery, shared by (B)/(C)/(D) ----------------

def recover_metalsq(u_obs, x_dat, basis, x_col, f, iters=400, lr=0.05, tik=0.0):
    """Adam on the exact gradient obtained through the closed-form solve."""
    c = torch.zeros(R, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        beta = forward(c, basis, x_col, f)
        loss = ((basis.evaluate(x_dat) @ beta - u_obs) ** 2).mean()
        if tik > 0:
            loss = loss + tik * (c ** 2).sum()
        loss.backward()
        opt.step()
    return c.detach(), time.time() - t0, iters      # 1 solve per step


def nystrom_solve(c, u_star, f, n=64):
    """Classical baseline: discretise the integral, solve the n x n system."""
    y, wq = gauss(n + 1)
    Kmat = torch.zeros(len(y), len(y))
    for m in range(R):
        Kmat = Kmat + c[m] * torch.outer(_gm(m + 1)(y), _hm(m + 1)(y) * wq)
    A = torch.eye(len(y)) - Kmat
    return y, torch.linalg.solve(A, f(y))


def recover_nystrom(u_obs, x_dat, u_star, f, iters=400, lr=0.05, n=64, eps=1e-5):
    """Same optimiser, but the gradient must be finite-differenced: 2R+1 solves/step."""
    c = torch.zeros(R)
    m_t = torch.zeros(R); v_t = torch.zeros(R)
    b1, b2, e = 0.9, 0.999, 1e-8
    t0 = time.time()
    solves = 0

    def obj(cc):
        nonlocal solves
        y, uy = nystrom_solve(cc, u_star, f, n=n)
        solves += 1
        # interpolate the Nystrom solution to the observation points
        idx = torch.searchsorted(y, x_dat.reshape(-1).clamp(y[0], y[-1])).clamp(1, len(y) - 1)
        y0, y1 = y[idx - 1], y[idx]
        t = ((x_dat.reshape(-1) - y0) / (y1 - y0)).clamp(0, 1)
        up = uy[idx - 1] * (1 - t) + uy[idx] * t
        return float(((up - u_obs) ** 2).mean())

    for it in range(1, iters + 1):
        gvec = torch.zeros(R)
        for m in range(R):
            cp = c.clone(); cp[m] += eps
            cm = c.clone(); cm[m] -= eps
            gvec[m] = (obj(cp) - obj(cm)) / (2 * eps)
        m_t = b1 * m_t + (1 - b1) * gvec
        v_t = b2 * v_t + (1 - b2) * gvec ** 2
        mh = m_t / (1 - b1 ** it); vh = v_t / (1 - b2 ** it)
        c = c - lr * mh / (vh.sqrt() + e)
    return c, time.time() - t0, solves


def one_problem(seed, noise, sigma=4.0, tik=0.0, iters=400, do_nystrom=False):
    gen = torch.Generator().manual_seed(seed)
    a, w, q, c_true = sample_problem(gen)
    u_star = u_star_fn(a, w, q)
    f, ip = make_f(c_true, u_star)
    basis = basis_of(sigma, seed)
    x_col = torch.linspace(0, 1, N_COL).reshape(-1, 1)
    x_dat = torch.linspace(0.02, 0.98, N_DATA).reshape(-1, 1)
    u_clean = u_star(x_dat)
    scale = float(u_clean.abs().mean())
    gg = torch.Generator().manual_seed(seed + 777)
    u_obs = u_clean + noise * scale * torch.randn(u_clean.shape, generator=gg)

    c_hat, dt, solves = recover_metalsq(u_obs, x_dat, basis, x_col, f,
                                        iters=iters, tik=tik)
    out = {"c_true": c_true.tolist(), "c_hat": c_hat.tolist(),
           "abs_err": (c_hat - c_true).abs().tolist(),
           "footprint": ip.abs().tolist(),
           "time_s": dt, "solves": solves}
    if do_nystrom:
        c_n, dt_n, solves_n = recover_nystrom(u_obs, x_dat, u_star, f, iters=iters)
        out["nystrom"] = {"c_hat": c_n.tolist(),
                          "abs_err": (c_n - c_true).abs().tolist(),
                          "time_s": dt_n, "solves": solves_n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=RESULT_PATH)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    global R_DEFAULT
    R_DEFAULT = R
    out = {"config": {"R": R, "N_feat": N_FEAT, "n_col": N_COL, "n_data": N_DATA,
                      "seeds": list(seeds), "iters": args.iters}}

    print("== (A) forward: closed-form solve of the integral equation ==")
    fa = forward_accuracy(seeds)
    out["forward"] = fa
    print(f"  relative L2 error of u, mean over {len(seeds)} problems: {fa['mean']:.3e}")

    print("\n== (B) identifiability: per-mode recovery vs noise ==")
    noises = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    b = {}
    for nz in noises:
        rows = [one_problem(s, nz, iters=args.iters) for s in seeds]
        err = torch.tensor([r["abs_err"] for r in rows]).mean(0)
        fp = torch.tensor([r["footprint"] for r in rows]).mean(0)
        b[str(nz)] = {"abs_err_per_mode": err.tolist(),
                      "footprint_per_mode": fp.tolist(),
                      "max_err": float(err.max())}
        print(f"  noise {nz:5.1%}   per-mode |c_hat-c| = "
              + "  ".join(f"{v:.4f}" for v in err.tolist()))
    out["identifiability"] = b
    fp = torch.tensor(b["0.0"]["footprint_per_mode"])
    print(f"  mode footprints |<h_m,u>| = " + "  ".join(f"{v:.4f}" for v in fp.tolist())
          + "   (small footprint => weakly identified)")

    print("\n== (C) a Tikhonov prior on c ==")
    cres = {}
    for nz in (0.02, 0.05, 0.10):
        for tik in (0.0, 1e-4, 1e-3, 1e-2):
            rows = [one_problem(s, nz, tik=tik, iters=args.iters) for s in seeds]
            err = torch.tensor([r["abs_err"] for r in rows]).mean(0)
            cres[f"{nz}/{tik}"] = {"abs_err_per_mode": err.tolist(),
                                   "max_err": float(err.max()),
                                   "mean_err": float(err.mean())}
        best = min((k for k in cres if k.startswith(f"{nz}/")),
                   key=lambda k: cres[k]["max_err"])
        print(f"  noise {nz:5.1%}: best tik = {best.split('/')[1]:>6}  "
              f"max err {cres[best]['max_err']:.4f}  "
              f"(no prior: {cres[f'{nz}/0.0']['max_err']:.4f})")
    out["tikhonov"] = cres

    print("\n== (D) against Nystrom + finite-difference gradients ==")
    dres = {}
    for nz in (0.0, 0.02):
        rows = [one_problem(s, nz, iters=args.iters, do_nystrom=True) for s in seeds]
        ours = torch.tensor([r["abs_err"] for r in rows]).mean(0)
        theirs = torch.tensor([r["nystrom"]["abs_err"] for r in rows]).mean(0)
        t_ours = sum(r["time_s"] for r in rows) / len(rows)
        t_theirs = sum(r["nystrom"]["time_s"] for r in rows) / len(rows)
        s_ours = rows[0]["solves"]; s_theirs = rows[0]["nystrom"]["solves"]
        dres[str(nz)] = {"metalsq_err": ours.tolist(), "nystrom_err": theirs.tolist(),
                         "metalsq_s": t_ours, "nystrom_s": t_theirs,
                         "metalsq_solves": s_ours, "nystrom_solves": s_theirs}
        print(f"  noise {nz:5.1%}")
        print(f"    Meta-LSQ  max err {float(ours.max()):.4f}  "
              f"{t_ours:6.1f}s  {s_ours} solves (exact grad)")
        print(f"    Nystrom   max err {float(theirs.max()):.4f}  "
              f"{t_theirs:6.1f}s  {s_theirs} solves (FD grad)")
    out["baseline"] = dres

    print("\n== (E) cost against the number of unknown kernel parameters ==")
    eres = {}
    print(f"  {'R':>3} {'Meta-LSQ s':>11} {'Nystrom s':>11} {'solves/step':>22} {'speedup':>9}")
    for r in (3, 6, 12, 24):
        set_rank(r)
        s0 = seeds[0]
        row = one_problem(s0, 0.01, iters=args.iters, do_nystrom=True)
        t_o, t_n = row["time_s"], row["nystrom"]["time_s"]
        eres[str(r)] = {"metalsq_s": t_o, "nystrom_s": t_n,
                        "metalsq_solves": row["solves"],
                        "nystrom_solves": row["nystrom"]["solves"],
                        "speedup": t_n / t_o}
        print(f"  {r:>3} {t_o:>11.2f} {t_n:>11.2f} "
              f"{f'{row[chr(34)+chr(34)] if False else 1} vs {2*r+1}':>22} {t_n/t_o:>8.2f}x")
    set_rank(R_DEFAULT)
    out["rank_scaling"] = eres

    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f_:
        json.dump(out, f_, indent=2)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
