"""Experiment #2.4: OOC tail analysis -- does the ridge truncate the blowup tail?

Meta-train Meta-LSQ-Sobolev on the sinusoid family, then evaluate per-task query
MSE on the OOC-B triangle family for several ridge values mu. The meta-learned mu
is tiny (~1e-5), which is optimal in-class but produces rare extrapolation blowups
out-of-class. We test whether raising mu at EVAL TIME (reusing fastlsq_inner_solve
with the chosen mu) truncates the heavy upper tail (p99/max collapse) at only a
modest cost to the median (typical) task.

Distribution reported per mu, over 200 tasks x 3 seeds (pooled per-task MSEs):
  median, p90, p99, max, and the fraction of the MEAN contributed by the top-1%
  of tasks (= "tail mass": how concentrated the error is in rare blowups).

Optional: prediction error vs distance-to-nearest-support point, to support the
"extrapolation" reading (query points far from any support blow up under tiny mu).

Read-only import of experiments.maml_sinusoid + experiments.out_of_class.
"""

import os
import sys
import json
import math
import numpy as np
import torch

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from experiments.maml_sinusoid import (
    train_fastlsq_meta,
    fastlsq_inner_solve,
    fastlsq_predict,
)
from experiments.out_of_class import sample_triangle_tasks

torch.set_default_dtype(torch.float32)

SEEDS = [0, 1, 2]
N_TEST = 200
K = 10
K_Q = 50
N_ITERS = 2000
N_BASIS = 200
# Fixed ridge ladder; mu_star (meta-learned, ~1e-5) is added per seed.
FIXED_MUS = [1e-3, 1e-2, 1e-1]


def per_task_eval(meta, mu_val, n_test=N_TEST, K=K, K_q=K_Q, seed=0,
                  return_extrap=False):
    """Per-task query MSE on the triangle family using a chosen eval-time mu.

    Reuses fastlsq_inner_solve with the meta-learned (sigma, grad_w, basis) but
    overrides the ridge to mu_val. Returns a numpy array of per-task MSEs.
    If return_extrap, also returns per-QUERY-POINT (sq error, dist-to-nearest-
    support) pairs pooled across tasks for the extrapolation analysis.
    """
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_triangle_tasks(n_test, K, K_q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(float(mu_val))
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    losses = []
    sq_errs = []   # per query point
    dists = []     # per query point: distance to nearest support x
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=meta["sobolev"], grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            err = (qpred - yq[t])
            losses.append(err.pow(2).mean().item())
            if return_extrap:
                # distance from each query x to nearest support x
                xqv = xq[t].squeeze(-1)              # (K_q,)
                xsv = xs[t].squeeze(-1)              # (K,)
                d = (xqv[:, None] - xsv[None, :]).abs().min(dim=1).values  # (K_q,)
                sq_errs.append(err.pow(2).numpy())
                dists.append(d.numpy())
    losses = np.asarray(losses, dtype=np.float64)
    if return_extrap:
        return losses, np.concatenate(sq_errs), np.concatenate(dists)
    return losses


def dist_summary(losses):
    """Distribution summary of per-task MSEs (heavy-tailed)."""
    l = np.asarray(losses, dtype=np.float64)
    n = l.size
    mean = float(l.mean())
    # top-1% of tasks by MSE
    k = max(1, int(math.ceil(0.01 * n)))
    top = np.sort(l)[-k:]
    top_frac = float(top.sum() / l.sum()) if l.sum() > 0 else 0.0
    return {
        "n": int(n),
        "mean": mean,
        "median": float(np.median(l)),
        "p90": float(np.percentile(l, 90)),
        "p99": float(np.percentile(l, 99)),
        "max": float(l.max()),
        "top1pct_mean_frac": top_frac,
    }


def extrap_curve(sq_errs, dists, edges):
    """Mean squared error binned by distance-to-nearest-support."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dists >= lo) & (dists < hi)
        if m.sum() == 0:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0,
                        "mean_sq_err": None})
        else:
            out.append({"lo": float(lo), "hi": float(hi),
                        "n": int(m.sum()),
                        "mean_sq_err": float(sq_errs[m].mean())})
    return out


def main():
    # Per-seed pooled per-task MSEs, keyed by mu label.
    pooled = {}          # mu_label -> list of arrays (one per seed)
    mu_star_vals = []
    sigma_vals = []
    gw_vals = []

    # extrapolation pooling: per (mu_label) accumulate (sq_err, dist)
    extrap_pool = {}

    # We will determine mu_star per seed; pool under the label "mu_star".
    mu_labels = ["mu_star"] + [f"{m:g}" for m in FIXED_MUS]
    for lab in mu_labels:
        pooled[lab] = []
        extrap_pool[lab] = {"sq": [], "d": []}

    for seed in SEEDS:
        print(f"[seed {seed}] meta-training Meta-LSQ-Sobolev ...", flush=True)
        meta, t_train, _ = train_fastlsq_meta(
            sobolev=True, n_iters=N_ITERS, N=N_BASIS, seed=seed, log_every=10**9)
        mu_star = float(meta["mu"])
        mu_star_vals.append(mu_star)
        sigma_vals.append(float(meta["sigma"]))
        gw_vals.append(float(meta["grad_w"]))
        print(f"  trained in {t_train:.1f}s  sigma={meta['sigma']:.3f} "
              f"mu_star={mu_star:.2e} gw={meta['grad_w']:.2f}", flush=True)

        # ridge ladder for this seed: meta-learned + fixed
        ladder = [("mu_star", mu_star)] + [(f"{m:g}", m) for m in FIXED_MUS]
        for lab, mu_val in ladder:
            do_extrap = True  # cheap; collect for every mu
            losses, sq, d = per_task_eval(
                meta, mu_val, seed=seed, return_extrap=True)
            pooled[lab].append(losses)
            extrap_pool[lab]["sq"].append(sq)
            extrap_pool[lab]["d"].append(d)
            s = dist_summary(losses)
            print(f"    mu={lab:>7} ({mu_val:.2e}):  med={s['median']:.4g}  "
                  f"p90={s['p90']:.4g}  p99={s['p99']:.4g}  max={s['max']:.4g}  "
                  f"top1%frac={s['top1pct_mean_frac']:.3f}", flush=True)

    # ---- aggregate across seeds (pool all per-task MSEs) ----
    edges = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 100.0])
    table = {}
    extrap_curves = {}
    for lab in mu_labels:
        allmse = np.concatenate(pooled[lab])
        table[lab] = dist_summary(allmse)
        sq = np.concatenate(extrap_pool[lab]["sq"])
        d = np.concatenate(extrap_pool[lab]["d"])
        extrap_curves[lab] = extrap_curve(sq, d, edges)

    # mu value mapping (mu_star is per-seed; report mean)
    mu_value_map = {"mu_star": float(np.mean(mu_star_vals))}
    for m in FIXED_MUS:
        mu_value_map[f"{m:g}"] = float(m)

    results = {
        "config": {
            "seeds": SEEDS,
            "n_test": N_TEST, "K": K, "K_q": K_Q,
            "n_iters": N_ITERS, "N_basis": N_BASIS,
            "family": "OOC-B triangle (sample_triangle_tasks)",
            "method": "Meta-LSQ-Sobolev meta-trained on sinusoids",
            "fixed_mus": FIXED_MUS,
        },
        "meta_learned": {
            "mu_star_per_seed": mu_star_vals,
            "mu_star_mean": float(np.mean(mu_star_vals)),
            "sigma_mean": float(np.mean(sigma_vals)),
            "grad_w_mean": float(np.mean(gw_vals)),
        },
        "mu_value_map": mu_value_map,
        "mu_labels": mu_labels,
        "quantile_table_pooled": table,
        "extrap_curve_dist_edges": edges.tolist(),
        "extrap_curves": extrap_curves,
    }

    out_path = _os.path.join(_REPO_ROOT, "results", "ooc_tail_ridge.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- pretty print ----
    print("\n===== Pooled quantile table (per-task query MSE, 200x3 tasks) =====")
    print(f"meta-learned mu_star mean = {np.mean(mu_star_vals):.2e} "
          f"(per-seed {['%.2e' % v for v in mu_star_vals]})")
    hdr = f"{'mu':>10} {'mu_val':>10} {'median':>10} {'p90':>10} {'p99':>10} {'max':>12} {'mean':>10} {'top1%frac':>10}"
    print(hdr)
    for lab in mu_labels:
        s = table[lab]
        print(f"{lab:>10} {mu_value_map[lab]:>10.2e} {s['median']:>10.4g} "
              f"{s['p90']:>10.4g} {s['p99']:>10.4g} {s['max']:>12.4g} "
              f"{s['mean']:>10.4g} {s['top1pct_mean_frac']:>10.3f}")

    print("\n===== Extrapolation: mean sq err vs dist-to-nearest-support =====")
    print(f"{'dist bin':>14} " + " ".join(f"{lab:>12}" for lab in mu_labels))
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        cells = []
        for lab in mu_labels:
            v = extrap_curves[lab][i]["mean_sq_err"]
            cells.append(f"{v:>12.4g}" if v is not None else f"{'--':>12}")
        binlab = f"[{lo:g},{hi:g})"
        print(f"{binlab:>14} " + " ".join(cells))

    print(f"\n[saved] {out_path}")
    return results


if __name__ == "__main__":
    main()
