"""N-scaling + numerical stability sweep for Meta-LSQ-Sobolev on the sinusoid benchmark.

Self-contained experiment (read-only import of experiments.maml_sinusoid).
Sweeps feature count N in {50,100,200,400,800,1500} for FastLSQ-Sobolev-Meta.

For each N, over 3 seeds we record:
  * test MSE mean +/- std
  * learned (sigma, mu, grad_w) -> stability of meta-learned params across N
  * per-outer-step wall-clock (train_time / n_iters) -> assembly O(M N d) + solve O(N^3)
  * NUMERICAL STABILITY at the learned (sigma, mu):
        - condition number of (A^T A + mu I) (torch.linalg.cond) on a few tasks
        - smallest eigenvalue of (A^T A + mu I)
    where A is the Sobolev stacked design [H; sqrt(gw)*G].

Run: OMP_NUM_THREADS=1 python experiments/nscale_numerics.py
"""

import os, sys, json, math, time

import torch

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from experiments.maml_sinusoid import (
    sample_tasks, train_fastlsq_meta, eval_fastlsq_meta, fastlsq_inner_solve,
)

torch.set_default_dtype(torch.float32)

N_LIST = [50, 100, 200, 400, 800, 1500]
SEEDS = [0, 1, 2]
# n_iters reduced from 2000 -> 800: only 3 scalars (sigma, mu, w) are meta-learned
# and they converge well within a few hundred Adam steps, so 800 is ample while
# keeping the O(N^3)-solve sweep at large N inside the wall-clock budget.
N_ITERS = 800
K = 10
N_COND_TASKS = 8   # tasks used to probe conditioning of (A^T A + mu I)


def assemble_AtA(xs, ys, dys, Wb, bb, sigma, mu, grad_w):
    """Rebuild the exact (A^T A + mu I) matrix used inside fastlsq_inner_solve
    (Sobolev path) so we can probe its conditioning / smallest eigenvalue."""
    N = Wb.shape[1]
    W = sigma * Wb                                  # (1, N)
    Z = xs @ W + bb                                 # (K, N)
    H = torch.sin(Z) / math.sqrt(N)
    G = (W * torch.cos(Z)) / math.sqrt(N)
    sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype))
    A = torch.cat([H, sqrt_w * G], dim=0)           # (2K, N)
    AtA = A.T @ A + mu * torch.eye(N)
    return AtA


def numerical_probe(meta, seed):
    """Condition number + smallest eigenvalue of (A^T A + mu I) at learned hp,
    averaged over N_COND_TASKS tasks. Returns (cond_mean, cond_max, eigmin_mean,
    eigmin_min). float32 throughout to expose float32 ill-conditioning."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(N_COND_TASKS, K, 50, gen, noise=0.0)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    conds, eigmins = [], []
    with torch.no_grad():
        for t in range(N_COND_TASKS):
            AtA = assemble_AtA(xs[t], ys[t], dys[t], Wb, bb, sigma, mu, gw)
            # symmetric PSD -> use eigvalsh; cond = lambda_max / lambda_min
            ev = torch.linalg.eigvalsh(AtA)         # ascending
            lam_min = float(ev[0].item())
            lam_max = float(ev[-1].item())
            cond = lam_max / lam_min if lam_min > 0 else float("inf")
            conds.append(cond)
            eigmins.append(lam_min)
    conds_t = torch.tensor(conds)
    eig_t = torch.tensor(eigmins)
    return (float(conds_t.mean()), float(conds_t.max()),
            float(eig_t.mean()), float(eig_t.min()))


def run():
    results = {}
    t_start = time.time()
    for N in N_LIST:
        per_seed = {"mse": [], "sigma": [], "mu": [], "grad_w": [],
                    "per_step_ms": [], "cond_mean": [], "cond_max": [],
                    "eigmin_mean": [], "eigmin_min": []}
        for seed in SEEDS:
            meta, train_time, curve = train_fastlsq_meta(
                sobolev=True, n_iters=N_ITERS, N=N, seed=seed, log_every=10**9)
            mse, _ = eval_fastlsq_meta(meta, seed=seed)
            cm, cx, em_mean, em_min = numerical_probe(meta, seed)
            per_seed["mse"].append(mse)
            per_seed["sigma"].append(meta["sigma"])
            per_seed["mu"].append(meta["mu"])
            per_seed["grad_w"].append(meta["grad_w"])
            per_seed["per_step_ms"].append(1000.0 * train_time / N_ITERS)
            per_seed["cond_mean"].append(cm)
            per_seed["cond_max"].append(cx)
            per_seed["eigmin_mean"].append(em_mean)
            per_seed["eigmin_min"].append(em_min)
            print(f"[N={N:5d} seed={seed}] mse={mse:.5f} "
                  f"sigma={meta['sigma']:.3f} mu={meta['mu']:.2e} "
                  f"gw={meta['grad_w']:.2f} step={1000*train_time/N_ITERS:.2f}ms "
                  f"cond={cm:.2e} eigmin={em_mean:.2e}", flush=True)

        def ms(key):
            v = torch.tensor(per_seed[key])
            return float(v.mean()), float(v.std(unbiased=False))

        agg = {}
        for key in per_seed:
            m, s = ms(key)
            agg[key + "_mean"] = m
            agg[key + "_std"] = s
        agg["raw"] = per_seed
        results[str(N)] = agg

    total = time.time() - t_start
    results["_meta"] = {"n_iters": N_ITERS, "seeds": SEEDS, "K": K,
                        "N_list": N_LIST, "n_cond_tasks": N_COND_TASKS,
                        "total_wall_s": total, "dtype": "float32",
                        "method": "FastLSQ-Sobolev-Meta"}

    out_dir = _os.path.join(_REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nscale_numerics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}   total {total:.1f}s")

    # pretty table
    print("\n==== N-scaling + numerics (FastLSQ-Sobolev-Meta, 3 seeds) ====")
    hdr = (f"{'N':>5} {'test MSE':>18} {'sigma':>14} {'mu':>12} {'gw':>12} "
           f"{'step ms':>10} {'cond(AtA+muI)':>16} {'eigmin':>12}")
    print(hdr)
    for N in N_LIST:
        a = results[str(N)]
        print(f"{N:>5} "
              f"{a['mse_mean']:.5f}+/-{a['mse_std']:.5f}  "
              f"{a['sigma_mean']:>6.3f}+/-{a['sigma_std']:.3f} "
              f"{a['mu_mean']:.2e} "
              f"{a['grad_w_mean']:>7.2f}+/-{a['grad_w_std']:.2f} "
              f"{a['per_step_ms_mean']:>9.2f} "
              f"{a['cond_mean_mean']:.3e} "
              f"{a['eigmin_mean_mean']:.2e}")
    return results


if __name__ == "__main__":
    run()
