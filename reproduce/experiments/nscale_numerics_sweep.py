"""N-scaling + numerical stability sweep for Meta-LSQ-Sobolev on the sinusoid benchmark.

Sweeps feature count N in {50,100,200,400,800,1500}. For each N over 3 seeds records:
  - test MSE (mean +/- std)
  - learned (sigma, mu, w) and their cross-N stability
  - per-outer-step wall-clock (training loop time / n_iters)
  - NUMERICAL STABILITY at the learned (sigma, mu):
        cond(A^T A + mu I), smallest eigenvalue lambda_min, largest singular value,
        measured on a few held-out tasks (Sobolev-stacked A so it matches the inner solve).

Read-only import of the benchmark module; writes results/nscale_numerics_sweep.json.
"""

import os, sys, time, json, math

# single-thread for honest per-step timing
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_default_dtype(torch.float32)

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from experiments.maml_sinusoid import (
    train_fastlsq_meta, eval_fastlsq_meta, make_basis, sample_tasks,
)

REPO = _REPO_ROOT
RESULTS_DIR = os.path.join(REPO, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_LIST = [50, 100, 200, 400, 800, 1500]
SEEDS = [0, 1, 2]
N_ITERS = 1500          # <=2000 budget; Sobolev meta has only 3 scalars, converges fast
N_COND_TASKS = 8        # tasks used to measure conditioning


def assemble_sobolev_A(xs, dys_unused, Wb, bb, sigma, mu, grad_w):
    """Rebuild the exact Sobolev design matrix A and normal matrix M = A^T A + mu I
    used inside fastlsq_inner_solve, for conditioning diagnostics. Returns M (N,N)."""
    W = sigma * Wb                                   # (1, N)
    Z = xs @ W + bb                                  # (K, N)
    Nf = W.shape[1]
    H = torch.sin(Z) / math.sqrt(Nf)
    G = (W * torch.cos(Z)) / math.sqrt(Nf)
    sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype))
    A = torch.cat([H, sqrt_w * G], dim=0)            # (2K, N)
    M = A.T @ A + mu * torch.eye(Nf)                 # (N, N)
    return M


def conditioning_stats(meta, n_tasks=N_COND_TASKS, K=10, seed=12345):
    """Compute cond / lambda_min / sigma_max of M = A^T A + mu I on a few tasks,
    in float32 (the dtype actually used by the solver)."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys, dys, xq, yq = sample_tasks(n_tasks, K, K, gen, noise=0.0)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"], dtype=torch.float32)
    mu = torch.tensor(meta["mu"], dtype=torch.float32)
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    conds, lmins, smaxs = [], [], []
    with torch.no_grad():
        for t in range(n_tasks):
            M = assemble_sobolev_A(xs[t], dys[t], Wb, bb, sigma, mu, gw)
            # M is SPD: singular values == eigenvalues. Use svdvals for stable cond.
            sv = torch.linalg.svdvals(M)             # descending
            smax = float(sv[0])
            smin = float(sv[-1])
            conds.append(smax / max(smin, 1e-30))
            lmins.append(smin)
            smaxs.append(smax)
    return {
        "cond_mean": float(torch.tensor(conds).mean()),
        "cond_max": float(max(conds)),
        "lambda_min_mean": float(torch.tensor(lmins).mean()),
        "lambda_min_min": float(min(lmins)),
        "sigma_max_mean": float(torch.tensor(smaxs).mean()),
    }


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


def run():
    t_start = time.time()
    rows = []
    for N in N_LIST:
        per_seed = {"mse": [], "sigma": [], "mu": [], "w": [], "per_step_ms": [],
                    "cond_mean": [], "cond_max": [], "lambda_min_mean": [],
                    "lambda_min_min": [], "sigma_max_mean": [], "train_s": []}
        for seed in SEEDS:
            meta, train_time, curve = train_fastlsq_meta(
                sobolev=True, n_iters=N_ITERS, N=N, seed=seed, log_every=10**9)
            mse, _inf = eval_fastlsq_meta(meta, seed=seed)
            cs = conditioning_stats(meta)
            per_seed["mse"].append(mse)
            per_seed["sigma"].append(meta["sigma"])
            per_seed["mu"].append(meta["mu"])
            per_seed["w"].append(meta["grad_w"])
            per_seed["per_step_ms"].append(1000.0 * train_time / N_ITERS)
            per_seed["train_s"].append(train_time)
            for k in ("cond_mean", "cond_max", "lambda_min_mean",
                      "lambda_min_min", "sigma_max_mean"):
                per_seed[k].append(cs[k])
            print(f"[N={N:5d} seed={seed}] mse={mse:.4f} sigma={meta['sigma']:.3f} "
                  f"mu={meta['mu']:.2e} w={meta['grad_w']:.2f} "
                  f"per_step={1000*train_time/N_ITERS:.2f}ms "
                  f"cond={cs['cond_mean']:.2e} lmin={cs['lambda_min_mean']:.2e}",
                  flush=True)

        mse_m, mse_s = mean_std(per_seed["mse"])
        sig_m, sig_s = mean_std(per_seed["sigma"])
        mu_m, mu_s = mean_std(per_seed["mu"])
        w_m, w_s = mean_std(per_seed["w"])
        ps_m, ps_s = mean_std(per_seed["per_step_ms"])
        cond_m, _ = mean_std(per_seed["cond_mean"])
        cond_max = float(max(per_seed["cond_max"]))
        lmin_m, _ = mean_std(per_seed["lambda_min_mean"])
        lmin_min = float(min(per_seed["lambda_min_min"]))
        smax_m, _ = mean_std(per_seed["sigma_max_mean"])
        row = {
            "N": N,
            "mse_mean": mse_m, "mse_std": mse_s,
            "sigma_mean": sig_m, "sigma_std": sig_s,
            "mu_mean": mu_m, "mu_std": mu_s,
            "w_mean": w_m, "w_std": w_s,
            "per_step_ms_mean": ps_m, "per_step_ms_std": ps_s,
            "cond_mean": cond_m, "cond_max": cond_max,
            "lambda_min_mean": lmin_m, "lambda_min_min": lmin_min,
            "sigma_max_mean": smax_m,
            "raw": per_seed,
        }
        rows.append(row)
        print(f"==> N={N}: MSE={mse_m:.4f}+/-{mse_s:.4f} "
              f"sigma={sig_m:.3f}+/-{sig_s:.3f} mu={mu_m:.2e} w={w_m:.2f} "
              f"per_step={ps_m:.2f}ms cond={cond_m:.2e} lmin={lmin_m:.2e}\n", flush=True)

    out = {
        "config": {
            "N_list": N_LIST, "seeds": SEEDS, "n_iters": N_ITERS,
            "method": "Meta-LSQ-Sobolev", "dtype": "float32",
            "n_cond_tasks": N_COND_TASKS,
            "total_wall_s": time.time() - t_start,
        },
        "rows": rows,
    }
    out_path = os.path.join(RESULTS_DIR, "nscale_numerics_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}  total {out['config']['total_wall_s']:.1f}s")
    return out


if __name__ == "__main__":
    run()
