"""N-scaling + numerical-stability sweep for Meta-LSQ-Sobolev on the sinusoid benchmark.

Addresses: "how far does it scale / is it numerically stable?"

We sweep the random-feature count N in {50,100,200,400,800,1500} and, over 3 seeds,
record for Meta-LSQ-Sobolev (train_fastlsq_meta(sobolev=True, N=...)):
  - test MSE  (mean +/- std over seeds)           -> is N a width-like knob (MSE flat)?
  - learned (sigma, mu, grad_w)                    -> are meta-params stable across N?
  - per-outer-step wall-clock = train_time/n_iters -> assembly O(M*N*d) + solve O(N^3)
  - NUMERICAL STABILITY at the learned (sigma,mu) on a few eval tasks:
        * condition number of  AtA = A^T A + mu I   (torch.linalg.cond, == s_max/s_min)
        * smallest eigenvalue of AtA                 (lam_min, should ~ mu)
    in float32 -> does small mu cause ill-conditioning at large N?

Self-contained. Imports the reference implementation read-only. Single-thread, float32.
"""

import sys, os, json, math, time
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch
torch.set_default_dtype(torch.float32)

from experiments.maml_sinusoid import (
    sample_tasks,
    make_basis,
    train_fastlsq_meta,
    eval_fastlsq_meta,
    fastlsq_inner_solve,
)

OUT = _os.path.join(_REPO_ROOT, "results", "nscale_stability_sweep.json")

N_LIST = [50, 100, 200, 400, 800, 1500]
SEEDS = [0, 1, 2]
N_ITERS = 1500          # keep total runtime < ~6 min across all N and seeds
K = 10
K_Q = 10


def assemble_AtA(meta, n_cond_tasks=5, seed=4242):
    """Reassemble AtA = A^T A + mu I exactly as fastlsq_inner_solve does, at the
    LEARNED (sigma, mu, grad_w), on a few held-out eval tasks; return per-task
    condition number and smallest eigenvalue (float32, matching the solve path)."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys, dys, xq, yq = sample_tasks(n_cond_tasks, K, K_Q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    N = Wb.shape[1]
    conds, eigmins = [], []
    with torch.no_grad():
        for t in range(n_cond_tasks):
            # Rebuild A exactly like the solver (sobolev -> stacked H and grad rows).
            W = sigma * Wb
            Z = xs[t] @ W + bb
            H = torch.sin(Z) / math.sqrt(N)
            if meta["sobolev"]:
                G = (W * torch.cos(Z)) / math.sqrt(N)
                sqrt_w = torch.sqrt(torch.as_tensor(gw, dtype=H.dtype))
                A = torch.cat([H, sqrt_w * G], dim=0)
            else:
                A = H
            AtA = A.T @ A + mu * torch.eye(N)
            # condition number (= s_max/s_min of the SPD matrix) and smallest eigenvalue
            cond = float(torch.linalg.cond(AtA).item())
            svd = torch.linalg.svdvals(AtA)
            eigmin = float(svd[-1].item())   # AtA SPD => singular values == eigenvalues
            conds.append(cond)
            eigmins.append(eigmin)
    return conds, eigmins


def run_one(N, seed):
    t0 = time.time()
    meta, train_time, _curve = train_fastlsq_meta(
        sobolev=True, n_iters=N_ITERS, N=N, K=K, K_q=K_Q, seed=seed, log_every=10_000)
    mse, _inf = eval_fastlsq_meta(meta, seed=seed)
    conds, eigmins = assemble_AtA(meta)
    per_step_ms = 1000.0 * train_time / N_ITERS
    return {
        "mse": float(mse),
        "sigma": float(meta["sigma"]),
        "mu": float(meta["mu"]),
        "grad_w": float(meta["grad_w"]),
        "per_step_ms": per_step_ms,
        "cond_mean": float(sum(conds) / len(conds)),
        "cond_max": float(max(conds)),
        "eigmin_mean": float(sum(eigmins) / len(eigmins)),
        "eigmin_min": float(min(eigmins)),
        "wall_s": time.time() - t0,
    }


def agg(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False)) if len(vals) > 1 else 0.0


def main():
    results = {}
    grand_t0 = time.time()
    for N in N_LIST:
        per_seed = [run_one(N, s) for s in SEEDS]
        keys = ["mse", "sigma", "mu", "grad_w", "per_step_ms",
                "cond_mean", "cond_max", "eigmin_mean", "eigmin_min"]
        entry = {"N": N, "raw": {k: [r[k] for r in per_seed] for k in keys}}
        for k in keys:
            m, s = agg([r[k] for r in per_seed])
            entry[k + "_mean"] = m
            entry[k + "_std"] = s
        results[str(N)] = entry
        print(f"N={N:5d}  MSE={entry['mse_mean']:.4f}+/-{entry['mse_std']:.4f}  "
              f"sigma={entry['sigma_mean']:.3f}  mu={entry['mu_mean']:.2e}  "
              f"gw={entry['grad_w_mean']:.2f}  step={entry['per_step_ms_mean']:.2f}ms  "
              f"cond={entry['cond_mean_mean']:.2e}  eigmin={entry['eigmin_mean_mean']:.2e}",
              flush=True)
    results["_meta"] = {
        "n_iters": N_ITERS, "seeds": SEEDS, "K": K, "K_q": K_Q,
        "N_list": N_LIST, "total_wall_s": time.time() - grand_t0,
        "dtype": "float32",
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT}  (total {results['_meta']['total_wall_s']:.1f}s)")


if __name__ == "__main__":
    main()
