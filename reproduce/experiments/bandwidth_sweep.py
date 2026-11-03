"""What the meta-learned bandwidth is actually choosing.

The whole method turns on one scalar: sigma sets the spread of the frozen
frequency draw W_j ~ N(0, sigma^2 I), so it decides which band of frequencies the
basis can represent. This script measures the resulting test error across a grid
of FIXED sigma, holding the other two scalars at their meta-learned values, and
records where the meta-learner actually lands.

The picture it supports is the geometric one: too narrow a bandwidth and the
basis has no mass at the frequency the task family uses, so the fit underfits;
too wide and the basis spends its N features on frequencies the family never
visits, so K = 10 support points can no longer pin the coefficients down. The
meta-learner finds the trough between the two from three scalars and no grid
search.

Run:  OMP_NUM_THREADS=1 python3 experiments/bandwidth_sweep.py --save results/bandwidth_sweep.json
"""

import os, sys, json, math, argparse

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import (
    train_fastlsq_meta, fastlsq_inner_solve, fastlsq_predict, sample_tasks,
)

torch.set_default_dtype(torch.float32)

SEEDS = (0, 1, 2)
RESULT_PATH = os.path.join(_REPO_ROOT, "results", "bandwidth_sweep.json")
# The task family is y = a sin(x + phi): a single angular frequency, omega = 1.
FAMILY_OMEGA = 1.0


def eval_at_sigma(meta, sigma, n_test=200, K=10, K_q=50, seed=0):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"]
    out = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], Wb, bb,
                                          torch.tensor(float(sigma)), mu,
                                          sobolev=True, grad_w=gw)
            out.append(float((fastlsq_predict(xq[t], W, bb, beta) - yq[t]).pow(2).mean()))
    v = torch.tensor(out)
    return {"mean": float(v.mean()), "median": float(v.median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=RESULT_PATH)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=200)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]

    grid = [float(x) for x in torch.logspace(math.log10(0.05), math.log10(8.0), 25)]
    per_seed, learned = {}, []
    for s in seeds:
        meta, _, _ = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=s,
                                        log_every=10 ** 9)
        learned.append(meta["sigma"])
        per_seed[str(s)] = [eval_at_sigma(meta, g, n_test=args.n_test, seed=s)
                            for g in grid]
        print(f"[seed {s}] learned sigma = {meta['sigma']:.3f}  "
              f"mu = {meta['mu']:.2e}  w = {meta['grad_w']:.2f}")

    curve = []
    for i, g in enumerate(grid):
        med = [per_seed[str(s)][i]["median"] for s in seeds]
        mean = [per_seed[str(s)][i]["mean"] for s in seeds]
        curve.append({"sigma": g,
                      "median": float(torch.tensor(med).mean()),
                      "mean": float(torch.tensor(mean).mean())})

    sig_star = float(torch.tensor(learned).mean())
    best = min(curve, key=lambda c: c["median"])
    print(f"\nmeta-learned sigma* = {sig_star:.3f} +/- "
          f"{float(torch.tensor(learned).std(unbiased=True)):.3f}")
    print(f"grid-optimal sigma  = {best['sigma']:.3f}  (median MSE {best['median']:.3g})")
    print(f"ratio sigma*/grid-optimal = {sig_star / best['sigma']:.2f}x")
    print("\n  sigma      median MSE")
    for c in curve:
        mark = "  <- meta-learned" if abs(c["sigma"] - sig_star) < 0.12 else ""
        print(f"  {c['sigma']:7.3f}  {c['median']:12.4g}{mark}")

    out = {"grid": grid, "curve": curve, "per_seed": per_seed,
           "learned_sigma": learned, "learned_sigma_mean": sig_star,
           "grid_optimal_sigma": best["sigma"], "family_omega": FAMILY_OMEGA,
           "config": {"seeds": list(seeds), "n_test": args.n_test,
                      "K": 10, "K_q": 50, "n_iters": 2000}}
    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
