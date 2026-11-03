"""Experiment #2.4: data-driven ridge floor via per-task leave-out CV.

Meta-train Meta-LSQ-Sobolev ONCE on the sinusoid family. Then on each test
family (in-class sin, OOC tanh, OOC triangle), compare three ridge policies:

  (a) tuned mu_star (brittle)   : use the meta-learned mu as-is.
  (b) hand-set floor mu=1e-2    : mu <- max(mu_star, 1e-2).
  (c) auto per-task mu (CV)     : for each task, hold out a few of the K=10
        support points, fit beta on the rest at each mu in a grid
        {mu_star, 1e-3, 1e-2, 1e-1, 1.0}, pick the mu with the smallest
        held-out support VALUE residual, refit on all support, eval query MSE.

The question: does auto-mu match the hand-set floor's robustness WITHOUT manual
tuning? Reports a 3x3 table (family x policy), mean AND median test MSE, over
several seeds with ~200 tasks/family. Negative results reported honestly.

Run:  OMP_NUM_THREADS=1 python3 experiments/autoridge_cv.py
"""

import os
import sys
import math
import json
import time

import torch

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from experiments.maml_sinusoid import (
    train_fastlsq_meta,
    fastlsq_inner_solve,
    fastlsq_predict,
    sample_tasks as sample_sinusoid_tasks,
)
from experiments.out_of_class import (
    sample_tanh_tasks,
    sample_triangle_tasks,
)

torch.set_default_dtype(torch.float32)

# ---------------- config ----------------
SEEDS = [0, 1, 2, 3, 4]          # 5 seeds
N_TEST = 200                      # tasks per family per seed
K = 10                            # support points
K_Q = 50                          # query points
N_HELDOUT = 3                     # support points held out per CV fold
N_ITERS = 2000
MU_FLOOR = 1e-2                   # hand-set floor (policy b)
RESULT_PATH = _os.path.join(_REPO_ROOT, "results", "autoridge_cv.json")


def _mu_grid(mu_star):
    """CV grid: the meta-learned mu plus a coarse decade ladder."""
    grid = [float(mu_star), 1e-3, 1e-2, 1e-1, 1.0]
    # de-dup while keeping order (mu_star may coincide with a ladder value)
    seen = set()
    out = []
    for m in grid:
        key = round(math.log10(max(m, 1e-30)), 6)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def auto_mu_for_task(xs_t, ys_t, dys_t, Wb, bb, sigma, sobolev, gw,
                     mu_grid, n_heldout, perm):
    """Pick mu by single leave-out fold on the support VALUE residual.

    Hold out the last ``n_heldout`` support points (after a fixed permutation),
    fit beta on the remaining support at each mu, score the held-out value MSE,
    and return the mu with the smallest held-out value residual.

    Crucially the held-out scoring uses ONLY the value rows (sin features) so
    that the criterion is the same quantity we ultimately care about (query
    value MSE), independent of the Sobolev gradient weighting.
    """
    Kpts = xs_t.shape[0]
    idx_tr = perm[:Kpts - n_heldout]
    idx_ho = perm[Kpts - n_heldout:]

    xs_tr, ys_tr, dys_tr = xs_t[idx_tr], ys_t[idx_tr], dys_t[idx_tr]
    xs_ho, ys_ho = xs_t[idx_ho], ys_t[idx_ho]

    best_mu = mu_grid[0]
    best_err = float("inf")
    for mu in mu_grid:
        mu_t = torch.tensor(float(mu))
        W, beta = fastlsq_inner_solve(
            xs_tr, ys_tr, dys_tr, Wb, bb, sigma, mu_t,
            sobolev=sobolev, grad_w=gw)
        pred_ho = fastlsq_predict(xs_ho, W, bb, beta)
        err = (pred_ho - ys_ho).pow(2).mean().item()
        if err < best_err:
            best_err = err
            best_mu = float(mu)
    return best_mu


def eval_policy(meta, sampler, policy, seed):
    """Evaluate one ridge policy on one family for one seed.

    policy in {"tuned", "floor", "auto"}. Returns list of per-task query MSEs.
    """
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sampler(N_TEST, K, K_Q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    sobolev = meta["sobolev"]
    gw = meta["grad_w"] if sobolev else 1.0
    mu_star = meta["mu"]
    mu_grid = _mu_grid(mu_star)

    # fixed CV permutation per seed (deterministic, not data-dependent)
    pgen = torch.Generator().manual_seed(seed + 4321)
    perm = torch.randperm(K, generator=pgen)

    losses = []
    chosen_mus = []
    with torch.no_grad():
        for t in range(N_TEST):
            if policy == "tuned":
                mu_use = mu_star
            elif policy == "floor":
                mu_use = max(mu_star, MU_FLOOR)
            elif policy == "auto":
                mu_use = auto_mu_for_task(
                    xs[t], ys[t], dys[t], Wb, bb, sigma,
                    sobolev, gw, mu_grid, N_HELDOUT, perm)
            else:
                raise ValueError(policy)
            chosen_mus.append(float(mu_use))
            mu_t = torch.tensor(float(mu_use))
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu_t,
                sobolev=sobolev, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return losses, chosen_mus


def aggregate(per_seed_losses):
    """per_seed_losses: list over seeds of lists of per-task MSEs.

    We pool the per-task MSEs across seeds, then report mean and median of the
    pooled per-task distribution (matches Table 3 framing: per-task summary)."""
    pooled = torch.tensor([v for seed_l in per_seed_losses for v in seed_l])
    return {
        "mean": float(pooled.mean()),
        "median": float(pooled.median()),
        "n": int(pooled.numel()),
    }


def main():
    t_start = time.time()
    families = [
        ("in-class (sin)", sample_sinusoid_tasks),
        ("OOC tanh", sample_tanh_tasks),
        ("OOC triangle", sample_triangle_tasks),
    ]
    policies = ["tuned", "floor", "auto"]

    # Meta-train Sobolev model once per seed (the meta-learned mu_star differs
    # per seed). All policies for that seed share the same trained model.
    metas = {}
    mu_stars = []
    for seed in SEEDS:
        meta, t_fit, _ = train_fastlsq_meta(
            sobolev=True, n_iters=N_ITERS, seed=seed, log_every=1000)
        metas[seed] = meta
        mu_stars.append(meta["mu"])
        print(f"[seed {seed}] trained sobolev meta in {t_fit:.1f}s  "
              f"sigma={meta['sigma']:.3f} mu_star={meta['mu']:.2e} "
              f"gw={meta['grad_w']:.3f}")

    # collect losses
    results = {}          # fam -> policy -> stats
    auto_mu_hist = {}     # fam -> {mu_value: count}
    for fam_name, sampler in families:
        results[fam_name] = {}
        for policy in policies:
            per_seed = []
            mus_all = []
            for seed in SEEDS:
                losses, chosen = eval_policy(metas[seed], sampler, policy, seed)
                per_seed.append(losses)
                if policy == "auto":
                    mus_all.extend(chosen)
            results[fam_name][policy] = aggregate(per_seed)
            if policy == "auto":
                hist = {}
                for m in mus_all:
                    k = f"{m:.0e}"
                    hist[k] = hist.get(k, 0) + 1
                auto_mu_hist[fam_name] = hist

    elapsed = time.time() - t_start

    # ---------------- print 3x3 table ----------------
    print("\n========== 3x3 table: family x policy  (mean / median query MSE) ==========")
    pol_labels = {"tuned": "(a) tuned mu*", "floor": "(b) floor 1e-2", "auto": "(c) auto CV"}
    header = f"{'family':<18}" + "".join(f"{pol_labels[p]:>26}" for p in policies)
    print(header)
    for fam_name, _ in families:
        row = f"{fam_name:<18}"
        for p in policies:
            s = results[fam_name][p]
            row += f"{s['mean']:>12.4f} /{s['median']:>11.5f}"
        print(row)

    print("\nAuto-mu selection histogram (pooled over seeds & tasks):")
    for fam_name, _ in families:
        print(f"  {fam_name:<18} {auto_mu_hist[fam_name]}")

    print(f"\nmu_star per seed: {[f'{m:.2e}' for m in mu_stars]}")
    print(f"Total wallclock: {elapsed:.1f}s")

    out = {
        "config": {
            "seeds": SEEDS, "n_test": N_TEST, "K": K, "K_q": K_Q,
            "n_heldout": N_HELDOUT, "n_iters": N_ITERS, "mu_floor": MU_FLOOR,
            "mu_grid_ladder": [1e-3, 1e-2, 1e-1, 1.0],
        },
        "mu_star_per_seed": mu_stars,
        "results": results,
        "auto_mu_histogram": auto_mu_hist,
        "elapsed_s": elapsed,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {RESULT_PATH}")


if __name__ == "__main__":
    main()
