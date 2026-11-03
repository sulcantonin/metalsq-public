"""Anisotropy study: what the extra bandwidth parameters actually buy.

The one-paragraph result in the paper is that on an anisotropic 2-D sinusoid
family a diagonal bandwidth beats a scalar one ~4x and a full lower-triangular
one beats the diagonal a further ~1.7x. The second number has no explanation:
that family draws omega_x and omega_y independently, so its frequency second
moment E[omega omega^T] is diagonal -- the family is axis-aligned, and a
rotation should buy nothing. This script asks what is really going on.

Reading the solve as kernel ridge fixes the prediction. Features are
sin(W_j^T x + b_j) with W = M \hat W and \hat W ~ N(0, I), so W_j ~ N(0, MM^T)
and the induced kernel is Gaussian with inverse-length-scale matrix
Sigma = MM^T. Matching a family whose frequencies have second moment C wants
Sigma ~ C. Hence:

  scalar M = sigma I      Sigma isotropic          -- cannot express any anisotropy
  diagonal M = diag(s)    Sigma axis-aligned       -- expresses axis-aligned C
  full M = L (lower tri)  Sigma any PSD            -- expresses rotated C

Four experiments, all on y = a sin(omega^T x + phi) with omega = R u,
u_k ~ U[c_k(1-d), c_k(1+d)], and c logarithmically spread over a ratio r with
geometric mean 1 (so overall roughness is fixed as r varies):

  E1 ratio sweep     R = I, r in {1,2,4,8}. Does the diagonal gain track r,
                     and vanish at r = 1?  (r = 16 is excluded: the bandwidth is
                     still climbing at 4000 iters there, and since scalar has
                     fewer parameters to converge than diag/full the comparison
                     would be confounded by optimisation, not expressiveness.)
  E2 rotation sweep  r = 8, theta in {0, 22.5, 45} deg in the (0,1) plane. Prediction:
                     the diagonal's advantage collapses as theta -> 45 deg while
                     the full L keeps it. This is the falsifiable one.
  E3 feature count   R = I, r = 8, N in {150,300,1200}. If the unexplained
                     full-over-diagonal gain is really the full L correcting the
                     realised random draw (whose sample covariance deviates from
                     I by O(1/sqrt(N))), that gain must shrink as N grows.
  E4 recovery        Does the learned Sigma recover the family's true frequency
                     structure -- the scale ratio for diag, the principal angle
                     for full L?

Two methodological points, both forced by pilot runs. Ratios are reported
PAIRED across seeds: absolute MSEs have overlapping error bars while the
per-seed ratios are tight. And the per-task summary is the MEDIAN, not the mean,
for the reason section 4.4 already gives -- under the near-zero learned ridge a
few test tasks extrapolate enormously, and a single such task moves a 200-task
mean by orders of magnitude. A pilot at lr = 2e-2 (chosen to speed convergence)
made this pathology routine and produced uninterpretable numbers, e.g. a 543x
"gain" at r = 1 where the correct answer is 1x; lr is therefore kept at the
5e-3 the rest of the paper uses.

Run:  OMP_NUM_THREADS=1 python3 experiments/anisotropy_study.py --save results/anisotropy_study.json
"""

import os, sys, math, json, time, argparse

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.anisotropic_bandwidth import (
    solve_task, predict, W_from_scalar, W_from_diag, W_from_full,
)

torch.set_default_dtype(torch.float32)

SEEDS = (0, 1, 2, 3, 4)
RESULT_PATH = os.path.join(_REPO_ROOT, "results", "anisotropy_study.json")
DELTA = 0.25          # per-coordinate frequency spread, omega_k in c_k(1 +/- DELTA)
PARAMS = ("scalar", "diag", "full")


# ---------------- family ----------------

def scales(d, r):
    """Per-coordinate frequency scales spanning a ratio r, geometric mean 1."""
    if d == 1 or r == 1:
        return torch.ones(d)
    return torch.logspace(-0.5, 0.5, d, base=float(r))


def rotation(d, theta):
    """Rotation by theta (radians) in the (0,1) plane, identity elsewhere."""
    R = torch.eye(d)
    if d >= 2 and theta != 0.0:
        c, s = math.cos(theta), math.sin(theta)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
    return R


def sample_tasks(n, K, K_q, gen, d, c, R):
    """y = a sin(omega^T x + phi), omega = R u, u_k ~ U[c_k(1-D), c_k(1+D)]."""
    a = torch.rand(n, 1, generator=gen) * 4.9 + 0.1
    phi = torch.rand(n, 1, generator=gen) * math.pi
    u = c[None, :] * (1.0 + DELTA * (2 * torch.rand(n, d, generator=gen) - 1))
    omega = u @ R.T                                             # (n,d)

    X_s = torch.rand(n, K, d, generator=gen) * 4 - 2
    X_q = torch.rand(n, K_q, d, generator=gen) * 4 - 2
    arg_s = torch.einsum('nkd,nd->nk', X_s, omega) + phi
    arg_q = torch.einsum('nkd,nd->nk', X_q, omega) + phi
    val_s = a * torch.sin(arg_s)
    val_q = a * torch.sin(arg_q)
    grad_s = (a * torch.cos(arg_s))[..., None] * omega[:, None, :]   # (n,K,d)
    return X_s, val_s, grad_s, X_q, val_q


# ---------------- batched Sobolev solve ----------------
# Same system as anisotropic_bandwidth.solve_task, batched over tasks: this
# study trains ~100 models, and the per-task Python loop dominates. Verified
# against the unbatched reference in _selfcheck() below.

def solve_batch(X, y, grad, W, b, mu, grad_w):
    """X:(B,K,d) y:(B,K) grad:(B,K,d) W:(d,N) b:(1,N) -> beta:(B,N).

    Solved in the dual. The design is M x N with M = K(1+d) rows against N
    features and M << N here (45 vs 300), so the primal normal equations are an
    N x N system of rank at most M. The identity

        (A'A + mu I_N)^-1 A'  =  A' (AA' + mu I_M)^-1

    gives the same beta from an M x M solve -- ~300x fewer flops at these sizes,
    which is what makes this ~100-model study affordable. Same underdetermined
    structure that makes GCV degenerate in experiments/gcv_ridge.py.
    """
    B, K, d = X.shape
    N = W.shape[1]
    z = X @ W + b                                     # (B,K,N)
    rt = math.sqrt(N)
    H = torch.sin(z) / rt
    cosz = torch.cos(z) / rt
    G = torch.cat([W[k].view(1, 1, N) * cosz for k in range(d)], dim=1)   # (B,K*d,N)
    sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype, device=H.device))
    A = torch.cat([H, sqrt_w * G], dim=1)                                 # (B,M,N)
    t = torch.cat([y, sqrt_w * grad.transpose(1, 2).reshape(B, K * d)], dim=1)
    M = A.shape[1]
    AAt = A @ A.transpose(1, 2) + mu * torch.eye(M, dtype=A.dtype)         # (B,M,M)
    return (A.transpose(1, 2) @ torch.linalg.solve(AAt, t.unsqueeze(-1))).squeeze(-1)


def predict_batch(X_q, W, b, beta):
    """X_q:(B,Kq,d) beta:(B,N) -> (B,Kq)."""
    z = X_q @ W + b
    return torch.einsum('bqn,bn->bq', torch.sin(z) / math.sqrt(W.shape[1]), beta)


def _selfcheck(tol=1e-9, verbose=True):
    """Batched dual solve must equal the primal reference behind the paper's numbers.

    Run in float64: the two forms are algebraically identical, so any real
    difference is a bug. In float32 they agree only to ~1e-3, because A'A + mu I
    at mu = 1e-4 is badly conditioned and the primal squares that conditioning --
    which is the point of solving in the dual, not a discrepancy to explain away.
    """
    dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        g = torch.Generator().manual_seed(0)
        d, N, K, B = 2, 60, 8, 3
        Wb = torch.randn(d, N, generator=g); bb = torch.rand(1, N, generator=g) * 6.28
        X, y, gr, _, _ = sample_tasks(B, K, 4, g, d, scales(d, 8), rotation(d, 0.3))
        mu, gw = torch.tensor(1e-4), torch.tensor(2.0)
        got = solve_batch(X, y, gr, Wb, bb, mu, gw)
        err = 0.0
        for t in range(B):
            ref = solve_task(X[t], y[t], gr[t], Wb, bb, mu, gw)
            err = max(err, float((ref - got[t]).abs().max() / ref.abs().max().clamp_min(1e-12)))
        if verbose:
            print(f"  dual-vs-primal solve (float64): max rel err {err:.2e}  "
                  f"{'ok' if err < tol else 'FAIL'}")
        return {"max_rel_err": err, "ok": bool(err < tol)}
    finally:
        torch.set_default_dtype(dt)


# ---------------- training ----------------

def make_W_fn(param, d, Wb):
    if param == "scalar":
        p = torch.tensor(0.0, requires_grad=True)
        return [p], (lambda: W_from_scalar(p, Wb))
    if param == "diag":
        p = torch.zeros(d, requires_grad=True)
        return [p], (lambda: W_from_diag(p, Wb))
    if param == "full":
        p = torch.zeros(d * (d + 1) // 2, requires_grad=True)
        return [p], (lambda: W_from_full(p, Wb))
    raise ValueError(param)


def train(param, d, c, R, N=300, iters=4000, lr=5e-3, meta_bs=4, K=15, K_q=20, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    gb = torch.Generator().manual_seed(seed + 7)
    Wb = torch.randn(d, N, generator=gb)
    b = torch.rand(1, N, generator=gb) * 2 * math.pi

    ps, W_fn = make_W_fn(param, d, Wb)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam(ps + [log_mu, log_gw], lr=lr)

    for _ in range(iters):
        X_s, v_s, g_s, X_q, v_q = sample_tasks(meta_bs, K, K_q, gen, d, c, R)
        opt.zero_grad()
        W = W_fn(); mu = log_mu.exp(); gw = log_gw.exp()
        beta = solve_batch(X_s, v_s, g_s, W, b, mu, gw)
        loss = ((predict_batch(X_q, W, b, beta) - v_q) ** 2).mean()
        loss.backward()
        opt.step()

    return {"W_fn": W_fn, "b": b, "Wb": Wb, "d": d,
            "mu": float(log_mu.exp()), "gw": float(log_gw.exp()),
            "n_learnable": sum(p.numel() for p in ps) + 2}


def evaluate(meta, d, c, R, n_test=200, K=15, K_q=50, seed=0):
    gen = torch.Generator().manual_seed(seed + 9999)
    X_s, v_s, g_s, X_q, v_q = sample_tasks(n_test, K, K_q, gen, d, c, R)
    with torch.no_grad():
        W = meta["W_fn"]()
        mu = torch.tensor(meta["mu"]); gw = torch.tensor(meta["gw"])
        beta = solve_batch(X_s, v_s, g_s, W, meta["b"], mu, gw)
        per_task = ((predict_batch(X_q, W, meta["b"], beta) - v_q) ** 2).mean(dim=1)
    return {"mean": float(per_task.mean()), "median": float(per_task.median())}


def sigma_structure(meta):
    """Sigma = M M^T from the learned parameterisation, via W = M \hat W.

    Recovered as M = W \hat W^+ so it works for every parameterisation; returns
    the eigenvalue ratio (anisotropy the model learned) and the principal-axis
    angle in the (0,1) plane, in degrees.
    """
    with torch.no_grad():
        W = meta["W_fn"](); Wb = meta["Wb"]
        M = W @ torch.linalg.pinv(Wb)
        S = M @ M.T
        ev, evec = torch.linalg.eigh(S)
        ratio = float((ev[-1] / ev[0].clamp_min(1e-12)).sqrt())   # scale ratio, not variance
        v = evec[:, -1]
        ang = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
        if ang > 90.0:
            ang -= 180.0
        return {"scale_ratio": ratio, "principal_angle_deg": ang}


# ---------------- paired reporting ----------------

def paired(a, b):
    """Per-seed ratio a/b: mean, std, and how often a > b."""
    r = [x / y for x, y in zip(a, b)]
    t = torch.tensor(r)
    return {"mean": float(t.mean()),
            "std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0,
            "wins": int(sum(1 for x in r if x > 1)), "n": len(r), "per_seed": r}


def run_cell(d, c, R, N, iters, seeds, params=PARAMS):
    """Train+eval every parameterisation over seeds. Returns {param: {...}}."""
    out = {}
    for p in params:
        means, meds, structs = [], [], []
        for s in seeds:
            m = train(p, d, c, R, N=N, iters=iters, seed=s)
            e = evaluate(m, d, c, R, seed=s)
            means.append(e["mean"]); meds.append(e["median"])
            structs.append(sigma_structure(m))
        t = torch.tensor(meds)
        out[p] = {"mse_mean": float(torch.tensor(means).mean()),
                  "mse_median": float(t.mean()),
                  "mse_std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0,
                  "mse_per_seed": meds,
                  "mse_mean_per_seed": means,
                  "scale_ratio": float(torch.tensor([x["scale_ratio"] for x in structs]).mean()),
                  "scale_ratio_per_seed": [x["scale_ratio"] for x in structs],
                  # Per-seed angles are kept because the mean of an angle is not
                  # meaningful: for the diagonal parameterisation Sigma is exactly
                  # diagonal, so every seed reports 0 or 90 depending on which
                  # coordinate carries the larger scale, and averaging those gives
                  # a spurious intermediate value.
                  "principal_angle_per_seed": [x["principal_angle_deg"] for x in structs],
                  "principal_angle_deg": float(
                      torch.tensor([x["principal_angle_deg"] for x in structs]).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=RESULT_PATH)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--only", default="e1,e2,e3")
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    only = set(x.strip() for x in args.only.split(","))
    t0 = time.time()
    print("== solver guard ==")
    guard = _selfcheck()
    out = {"guard": guard, "config": {"seeds": list(seeds), "iters": args.iters, "delta": DELTA,
                      "K": 15, "K_q": 50, "n_test": 200}}

    # ---- E1: anisotropy-ratio sweep, axis aligned ----
    if "e1" in only:
        print("== E1: anisotropy ratio sweep (R = I, d = 2) ==")
        print(f"{'r':>5} {'scalar':>10} {'diag':>10} {'full':>10} "
              f"{'sc/diag (paired)':>22} {'diag/full (paired)':>22} {'learned s2/s1':>14}")
        e1 = {}
        for r in (1, 2, 4, 8):
            c = scales(2, r)
            cell = run_cell(2, c, rotation(2, 0.0), 300, args.iters, seeds)
            sd = paired(cell["scalar"]["mse_per_seed"], cell["diag"]["mse_per_seed"])
            df = paired(cell["diag"]["mse_per_seed"], cell["full"]["mse_per_seed"])
            e1[str(r)] = {"cells": cell, "scalar_over_diag": sd, "diag_over_full": df,
                          "true_ratio": float(c[-1] / c[0])}
            print(f"{r:>5} {cell['scalar']['mse_median']:>10.3f} {cell['diag']['mse_median']:>10.3f} "
                  f"{cell['full']['mse_median']:>10.3f} "
                  f"{sd['mean']:>9.2f}x +/-{sd['std']:<5.2f} {sd['wins']}/{sd['n']} "
                  f"{df['mean']:>9.2f}x +/-{df['std']:<5.2f} {df['wins']}/{df['n']} "
                  f"{cell['diag']['scale_ratio']:>14.2f}")
        out["e1_ratio_sweep"] = e1

    # ---- E2: rotation sweep ----
    if "e2" in only:
        print("\n== E2: rotation sweep (r = 8, d = 2) ==")
        print(f"{'theta':>6} {'scalar':>10} {'diag':>10} {'full':>10} "
              f"{'sc/diag (paired)':>22} {'sc/full (paired)':>22} {'full angle':>11}")
        e2 = {}
        c = scales(2, 8)
        for deg in (0, 22.5, 45):
            R = rotation(2, math.radians(deg))
            cell = run_cell(2, c, R, 300, args.iters, seeds)
            sd = paired(cell["scalar"]["mse_per_seed"], cell["diag"]["mse_per_seed"])
            sf = paired(cell["scalar"]["mse_per_seed"], cell["full"]["mse_per_seed"])
            e2[str(deg)] = {"cells": cell, "scalar_over_diag": sd, "scalar_over_full": sf}
            print(f"{deg:>6} {cell['scalar']['mse_median']:>10.3f} {cell['diag']['mse_median']:>10.3f} "
                  f"{cell['full']['mse_median']:>10.3f} "
                  f"{sd['mean']:>9.2f}x +/-{sd['std']:<5.2f} {sd['wins']}/{sd['n']} "
                  f"{sf['mean']:>9.2f}x +/-{sf['std']:<5.2f} {sf['wins']}/{sf['n']} "
                  f"{cell['full']['principal_angle_deg']:>10.1f}d")
        out["e2_rotation_sweep"] = e2

    # ---- E3: does the full-over-diag gain shrink with N? ----
    if "e3" in only:
        print("\n== E3: feature count (R = I, r = 8) -- is full-L correcting the draw? ==")
        print(f"{'N':>6} {'diag':>10} {'full':>10} {'diag/full (paired)':>24} "
              f"{'draw anisotropy':>16}")
        e3 = {}
        c = scales(2, 8)
        for N in (150, 300, 1200):
            cell = run_cell(2, c, rotation(2, 0.0), N, args.iters, seeds,
                            params=("diag", "full"))
            df = paired(cell["diag"]["mse_per_seed"], cell["full"]["mse_per_seed"])
            # anisotropy of the realised frozen draw itself, averaged over seeds
            draws = []
            for s in seeds:
                gb = torch.Generator().manual_seed(s + 7)
                Wb = torch.randn(2, N, generator=gb)
                ev = torch.linalg.eigvalsh(Wb @ Wb.T / N)
                draws.append(float((ev[-1] / ev[0]).sqrt()))
            da = float(torch.tensor(draws).mean())
            e3[str(N)] = {"cells": cell, "diag_over_full": df, "draw_anisotropy": da}
            print(f"{N:>6} {cell['diag']['mse_median']:>10.3f} {cell['full']['mse_median']:>10.3f} "
                  f"{df['mean']:>11.2f}x +/-{df['std']:<5.2f} {df['wins']}/{df['n']} "
                  f"{da:>16.3f}")
        out["e3_feature_count"] = e3

    out["elapsed_s"] = time.time() - t0
    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {p}   ({out['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
