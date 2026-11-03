"""Generalised cross-validation as the data-driven ridge, replacing the hand-set floor.

App. D compares the hand-set floor mu >= 1e-2 against a per-task leave-out CV over a
5-point ladder, which matches on typical performance but misses the tail. GCV is the
other classical choice and needs no ladder and no held-out split: for the stacked
design A it reads mu straight off the singular values,

    GCV(mu) = n * RSS(mu) / tr(I - S(mu))^2,     S(mu) = A (A'A + mu I)^-1 A'

which in the SVD A = U diag(s) V' is, with c = U'y and rho_i = mu / (s_i^2 + mu),

    RSS(mu)        = sum_i rho_i^2 c_i^2 + (||y||^2 - ||c||^2)
    tr(I - S(mu))  = sum_i rho_i + (n - r)

so the whole mu-curve costs one SVD of a (K(d+1) x N) matrix plus O(grid * r).

Note the regime: A here is 20 x 200, i.e. underdetermined, so RSS -> 0 AND
tr(I-S) -> 0 as mu -> 0. The ratio is still well defined -- rho_i ~ mu/s_i^2 makes
mu cancel -- and its limit sum_i c_i^2/s_i^4 / (sum_i 1/s_i^2)^2 is dominated by the
smallest singular values. That is precisely the behaviour we want: GCV tolerates a
near-zero ridge on a well-conditioned design and raises it on an ill-conditioned one.

Policies compared, on the same families and seeds as autoridge_cv.py:
  tuned  meta-learned mu* as-is
  floor  max(mu*, 1e-2)              -- the hand-set constant
  auto   leave-out CV over a ladder  -- App. D's existing alternative
  gcv    argmin_mu GCV(mu), continuous grid, no held-out points

Run:  OMP_NUM_THREADS=1 python3 experiments/gcv_ridge.py --save results/gcv_ridge.json
      (add --only-validate to recompute just the gcv_guard block of that JSON)
"""

import os, sys, math, json, time, argparse

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import (
    train_fastlsq_meta, fastlsq_inner_solve, fastlsq_predict,
    sample_tasks as sample_sinusoid_tasks,
)
from experiments.out_of_class import sample_tanh_tasks, sample_triangle_tasks
from experiments.autoridge_cv import _mu_grid, auto_mu_for_task, aggregate

torch.set_default_dtype(torch.float32)

SEEDS = [0, 1, 2, 3, 4]
N_TEST = 200
K = 10
K_Q = 50
N_HELDOUT = 3
N_ITERS = 2000
MU_FLOOR = 1e-2
RESULT_PATH = os.path.join(_REPO_ROOT, "results", "gcv_ridge.json")

# continuous grid; no ladder, no magic constant
GCV_GRID = torch.logspace(-8, 2, 121, dtype=torch.float64)


def build_design(xs_t, ys_t, dys_t, Wb, bb, sigma, sobolev, gw):
    """The exact stacked system fastlsq_inner_solve forms, as (A, t)."""
    W = sigma * Wb
    Z = xs_t @ W + bb
    N = W.shape[1]
    H = torch.sin(Z) / math.sqrt(N)
    if not sobolev:
        return H, ys_t
    G = (W * torch.cos(Z)) / math.sqrt(N)
    sqrt_w = torch.sqrt(torch.as_tensor(gw, dtype=H.dtype))
    return torch.cat([H, sqrt_w * G], 0), torch.cat([ys_t, sqrt_w * dys_t], 0)


def gcv_argmin(A, t, grid=GCV_GRID):
    """GCV over the grid for a design A and target t; returns (argmin index, curve).

    The single GCV implementation: used both by the ridge policy and by its guard
    in validate_gcv, so the guard tests exactly the code that produces the results.
    """
    A = A.double(); t = t.double()
    n = A.shape[0]
    U, s, _ = torch.linalg.svd(A, full_matrices=False)
    c = U.T @ t                                   # (r,)
    r = s.shape[0]
    tail = float((t @ t) - (c @ c))               # energy outside range(U); ~0 when r == n
    tail = max(tail, 0.0)

    s2 = (s ** 2)[None, :]                        # (1,r)
    mu = grid[:, None]                            # (G,1)
    rho = mu / (s2 + mu)                          # (G,r)
    rss = (rho ** 2 * (c ** 2)[None, :]).sum(1) + tail
    tr = rho.sum(1) + (n - r)
    gcv = n * rss / tr.clamp_min(1e-300) ** 2
    j = int(torch.argmin(torch.where(torch.isfinite(gcv), gcv, torch.full_like(gcv, float("inf")))))
    return j, gcv


def gcv_mu_for_task(xs_t, ys_t, dys_t, Wb, bb, sigma, sobolev, gw, grid=GCV_GRID):
    """Minimise the GCV criterion over the grid; returns (mu, gcv_curve_min)."""
    A, t = build_design(xs_t, ys_t, dys_t, Wb, bb, sigma, sobolev, gw)
    j, gcv = gcv_argmin(A, t, grid)
    return float(grid[j]), float(gcv[j])


def validate_gcv(reps=10, verbose=True):
    """Guard: does this GCV code select a sensible ridge where GCV is known to work?

    The negative result below (GCV cannot replace the floor) is only meaningful if
    the implementation is right, so we check it on an ill-conditioned overdetermined
    problem with a genuine interior optimum: n=100, p=40, singular values decaying
    over three decades, Gaussian noise. Scored by ACHIEVED test error at the selected
    mu rather than by argmin location, which is uninformative when the curve is flat.
    """
    dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        g = torch.Generator().manual_seed(7)
        rows = []
        for noise in (1.0, 0.3, 0.1, 0.03):
            acc = []
            for _ in range(reps):
                n, p = 100, 40
                U, _ = torch.linalg.qr(torch.randn(n, p, generator=g))
                V, _ = torch.linalg.qr(torch.randn(p, p, generator=g))
                s = torch.logspace(0, -3, p)
                A = U @ torch.diag(s) @ V.T
                beta = torch.randn(p, generator=g)
                t = A @ beta + noise * torch.randn(n, generator=g)
                Ute, _ = torch.linalg.qr(torch.randn(400, p, generator=g))
                Ate = Ute @ torch.diag(s) @ V.T
                tte = Ate @ beta

                # Same GCV code as the ridge policy (n > p here, so the residual
                # energy outside range(A) and the (n - r) trace term both matter).
                j_gcv, _ = gcv_argmin(A, t)
                I = torch.eye(p)
                errs = torch.tensor([
                    float(((Ate @ torch.linalg.solve(A.T @ A + mu * I, A.T @ t) - tte) ** 2).mean())
                    for mu in GCV_GRID])
                acc.append((float(errs[j_gcv]),
                            float(errs.min()), float(errs[0])))
            m = torch.tensor(acc)
            e_g, e_o, e_0 = float(m[:, 0].mean()), float(m[:, 1].mean()), float(m[:, 2].mean())
            # Fraction of the attainable gain (no-ridge -> oracle) that GCV captures.
            # Scale-free, so it does not need a threshold tuned per noise level the way
            # a raw err@GCV/err@oracle ratio does.
            recov = (e_0 - e_g) / (e_0 - e_o) if e_0 > e_o else float("nan")
            rows.append({"noise": noise, "err_gcv": e_g, "err_oracle": e_o,
                         "err_no_ridge": e_0, "gcv_over_oracle": float((m[:, 0] / m[:, 1]).mean()),
                         "gain_recovered": recov})
            if verbose:
                print(f"  noise={noise:<5} err@GCV {e_g:.4g}  err@oracle {e_o:.4g}  "
                      f"err@mu->0 {e_0:.4g}   recovers {recov * 100:.0f}% of the attainable gain")
        ok = all(r["gain_recovered"] > 0.6 for r in rows)
        if verbose:
            print(f"  -> implementation {'OK' if ok else 'SUSPECT'}: GCV recovers >60% of the "
                  f"no-ridge-to-oracle gain at every noise level")
        return {"rows": rows, "ok": bool(ok)}
    finally:
        torch.set_default_dtype(dt)


def eval_policy(meta, sampler, policy, seed):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sampler(N_TEST, K, K_Q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    sobolev = meta["sobolev"]
    gw = meta["grad_w"] if sobolev else 1.0
    mu_star = meta["mu"]
    mu_grid = _mu_grid(mu_star)

    pgen = torch.Generator().manual_seed(seed + 4321)
    perm = torch.randperm(K, generator=pgen)

    losses, chosen = [], []
    with torch.no_grad():
        for t in range(N_TEST):
            if policy == "tuned":
                mu_use = mu_star
            elif policy == "floor":
                mu_use = max(mu_star, MU_FLOOR)
            elif policy == "auto":
                mu_use = auto_mu_for_task(xs[t], ys[t], dys[t], Wb, bb, sigma,
                                          sobolev, gw, mu_grid, N_HELDOUT, perm)
            elif policy == "gcv":
                mu_use, _ = gcv_mu_for_task(xs[t], ys[t], dys[t], Wb, bb,
                                            sigma, sobolev, gw)
            else:
                raise ValueError(policy)
            chosen.append(float(mu_use))
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], Wb, bb, sigma,
                                          torch.tensor(float(mu_use)),
                                          sobolev=sobolev, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return losses, chosen


def tail_stats(per_seed_losses):
    """Tail summary matching Table 5: the tail is where leave-out CV failed."""
    pooled = torch.tensor([v for sl in per_seed_losses for v in sl], dtype=torch.float64)
    srt = torch.sort(pooled).values
    n = srt.numel()
    top1 = srt[int(0.99 * n):]
    return {
        "mean": float(pooled.mean()), "median": float(pooled.median()),
        "p90": float(srt[int(0.90 * n)]), "p99": float(srt[int(0.99 * n)]),
        "max": float(srt[-1]),
        "top1pct_share": float(top1.sum() / pooled.sum()) if float(pooled.sum()) > 0 else 0.0,
        "n": int(n),
    }


def main():
    global N_TEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=RESULT_PATH)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--only-validate", action="store_true",
                    help="recompute only the gcv_guard block of an existing --save JSON")
    args = ap.parse_args()
    N_TEST = args.n_test
    seeds = SEEDS[:args.seeds]
    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)

    if args.only_validate:
        # Leaves every other field (and its key order) as written by the full run.
        print("==== GCV implementation guard (ill-conditioned, interior optimum) ====")
        with open(p) as f:
            out = json.load(f)
        out["gcv_guard"] = validate_gcv()
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {p}   (gcv_guard only)")
        return

    t_start = time.time()
    print("==== GCV implementation guard (ill-conditioned, interior optimum) ====")
    guard = validate_gcv()

    families = [("in-class (sin)", sample_sinusoid_tasks),
                ("OOC tanh", sample_tanh_tasks),
                ("OOC triangle", sample_triangle_tasks)]
    policies = ["tuned", "floor", "auto", "gcv"]

    metas, mu_stars = {}, []
    for seed in seeds:
        meta, t_fit, _ = train_fastlsq_meta(sobolev=True, n_iters=N_ITERS,
                                            seed=seed, log_every=10 ** 9)
        metas[seed] = meta
        mu_stars.append(meta["mu"])
        print(f"[seed {seed}] sigma={meta['sigma']:.3f} mu*={meta['mu']:.2e} "
              f"gw={meta['grad_w']:.3f}  ({t_fit:.1f}s)")

    results, mu_hist = {}, {}
    for fam_name, sampler in families:
        results[fam_name] = {}
        mu_hist[fam_name] = {}
        for policy in policies:
            per_seed, mus_all = [], []
            for seed in seeds:
                losses, chosen = eval_policy(metas[seed], sampler, policy, seed)
                per_seed.append(losses)
                mus_all.extend(chosen)
            results[fam_name][policy] = {**aggregate(per_seed), **tail_stats(per_seed)}
            if policy in ("auto", "gcv"):
                h = {}
                for m in mus_all:
                    kk = f"{m:.0e}"
                    h[kk] = h.get(kk, 0) + 1
                mu_hist[fam_name][policy] = dict(sorted(h.items(),
                                                        key=lambda kv: -kv[1])[:8])

    elapsed = time.time() - t_start

    print("\n===== family x policy: mean / median query MSE =====")
    lab = {"tuned": "tuned mu*", "floor": "floor 1e-2", "auto": "leave-out CV", "gcv": "GCV"}
    print(f"{'family':<16}" + "".join(f"{lab[p]:>24}" for p in policies))
    for fam_name, _ in families:
        row = f"{fam_name:<16}"
        for p in policies:
            s = results[fam_name][p]
            row += f"{s['mean']:>11.4g} /{s['median']:>11.5g}"
        print(row)

    print("\n===== tail (pooled per-task): p99 / max / share of mean in worst 1% =====")
    for fam_name, _ in families:
        print(f"  {fam_name}")
        for p in policies:
            s = results[fam_name][p]
            print(f"    {lab[p]:<14} p99 {s['p99']:>11.4g}  max {s['max']:>11.4g}"
                  f"  top1% {s['top1pct_share'] * 100:>5.1f}%")

    print("\nselected-mu histograms (top values):")
    for fam_name, _ in families:
        for p in ("auto", "gcv"):
            print(f"  {fam_name:<16} {p:<5} {mu_hist[fam_name][p]}")

    out = {"config": {"seeds": seeds, "n_test": N_TEST, "K": K, "K_q": K_Q,
                      "n_heldout": N_HELDOUT, "n_iters": N_ITERS,
                      "mu_floor": MU_FLOOR,
                      "gcv_grid": [float(GCV_GRID[0]), float(GCV_GRID[-1]),
                                   int(GCV_GRID.numel())]},
           "mu_star_per_seed": mu_stars, "results": results,
           "mu_histogram": mu_hist, "gcv_guard": guard, "elapsed_s": elapsed}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {p}   ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
