"""Shared-hyperparameter exact GP control under Meta-LSQ's own protocol
(reviewer dX2X), at 2000 and 6000 meta-training steps.

Four meta-learners, each trained and tested on identical draws within a seed:

  metalsq_sob    Meta-LSQ-Sobolev: train_fastlsq_meta(sobolev=True). N=200
                 Gaussian random features (basis seed s); Adam learns
                 (log sigma, log mu, log w) from sigma=1, mu=1e-4, w=1.
  metalsq_val    Meta-LSQ w=0: train_fastlsq_meta(sobolev=False); Adam learns
                 (log sigma, log mu) from the same initialisation.
  gek_shared     Exact gradient-enhanced RBF GP (GEK) posterior mean, unit signal
                 variance, value and derivative observations with separate noise.
                 One length-scale l, value noise lam_y and derivative noise lam_g
                 shared by all tasks; Adam learns (log l, log lam_y, log lam_g)
                 from l0=1, lam_y=lam_g=1e-4 (train_gek_shared_sin, float64,
                 Cholesky with escalating jitter 0, 1e-8, 1e-6, 1e-4, then 1e-2).
  gp_val_shared  Exact RBF GP on values only; Adam learns (log l, log lam) from
                 l0=1, lam0=1e-4.

Meta-training (identical for all four, Table 1 protocol): per seed s,
torch.manual_seed(s); task generator torch.Generator().manual_seed(s + 1000);
each step draws a batch of 4 noiseless sinusoid tasks y = a sin(x + phi),
a ~ U[0.1, 5], phi ~ U[0, pi], x ~ U[-5, 5], with K=10 support points (values
and derivatives) and K_q=10 query points; the loss is the batch-mean query MSE
of the closed-form predictor (posterior mean / ridge solve) and is minimised by
Adam with lr 5e-3 for T in {2000, 6000} steps. The 6000-step run continues the
same task stream (its first 2000 steps are the 2000-step run).

Test: hyperparameters frozen; the same 200 held-out tasks per seed as Table 1
(generator s + 9999), K=10, K_q=50; per-task query MSE on values. Seeds 0-4.
"mean" is the mean over seeds of the per-seed mean test MSE, "mean_std" the
sample std (ddof=1) over seeds of those per-seed means (summarise() in
rebuttal_real_hypotheses.py).

Parts
  --part mse     (default) all four methods x {2000, 6000} steps x 5 seeds ->
                 results/rebuttal_gek_shared_matched.json. train_s there is NOT a
                 timing measurement (runs may share the machine); use --part timing.
  --part timing  re-times all eight (method, steps) trainings sequentially in one
                 process (one untimed 20-step warm-up per method first) ->
                 results/rebuttal_gek_shared_timing.json, {"<method>_<steps>":
                 {"train_s": [per seed], "mse": [per seed]}}. train_s is the
                 wall time of the training loop only. Run on an idle machine.

An existing output is copied to results/superseded/<name> before it is
overwritten (an older superseded copy is kept).

Run:  OMP_NUM_THREADS=1 python3 experiments/rebuttal_gek_shared_matched.py --part mse
      OMP_NUM_THREADS=1 python3 experiments/rebuttal_gek_shared_matched.py --part timing
"""
import os, sys, json, time, shutil, argparse

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import (
    sample_tasks, train_fastlsq_meta, fastlsq_inner_solve, fastlsq_predict,
)
from experiments.rebuttal_real_hypotheses import train_gek_shared_sin, eval_gek_shared_sin, summarise

torch.set_default_dtype(torch.float32)
SEEDS = (0, 1, 2, 3, 4)
BUDGETS = (2000, 6000)
LR, L0, LAM0 = 5e-3, 1.0, 1e-4
# order of the existing timing file
ORDER = (("metalsq_sob", 2000), ("gek_shared", 2000), ("metalsq_sob", 6000), ("gek_shared", 6000),
         ("metalsq_val", 2000), ("gp_val_shared", 2000), ("metalsq_val", 6000), ("gp_val_shared", 6000))
NAMES = {
    ("gek_shared", 2000): "GEK shared, matched protocol (2000 steps, l0=1, lam0=1e-4)",
    ("gp_val_shared", 2000): "GP shared values-only, matched protocol",
    ("gek_shared", 6000): "GEK shared, matched protocol (6000 steps, l0=1, lam0=1e-4)",
    ("gp_val_shared", 6000): "GP shared values-only, matched protocol (6000 steps)",
    ("metalsq_sob", 2000): "Meta-LSQ-Sobolev (2000 steps)",
    ("metalsq_sob", 6000): "Meta-LSQ-Sobolev (6000 steps)",
    ("metalsq_val", 2000): "Meta-LSQ w=0 (2000 steps)",
    ("metalsq_val", 6000): "Meta-LSQ w=0 (6000 steps)",
}


def eval_metalsq_per_task(meta, n_test=200, K=10, K_q=50, seed=0):
    """eval_fastlsq_meta, returning the per-task query MSEs and ms/task."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)
    sigma = torch.tensor(meta["sigma"]); mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    out = []
    t0 = time.time()
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], meta["Wb"], meta["bb"], sigma, mu,
                                          sobolev=meta["sobolev"], grad_w=gw)
            out.append(float((fastlsq_predict(xq[t], W, meta["bb"], beta) - yq[t]).pow(2).mean()))
    return out, (time.time() - t0) / n_test * 1e3


def train(method, steps, seed):
    """Returns (learned-hyperparameter dict incl. train_s, per-task test MSEs, ms/task)."""
    if method in ("metalsq_sob", "metalsq_val"):
        sob = method == "metalsq_sob"
        meta, train_s, _ = train_fastlsq_meta(n_iters=steps, sobolev=sob, lr=LR, seed=seed,
                                              log_every=10 ** 9)
        v, ms = eval_metalsq_per_task(meta, seed=seed)
        return {"sigma": meta["sigma"], "mu": meta["mu"], "w": meta["grad_w"],
                "sobolev": sob, "train_s": train_s}, v, ms
    sob = method == "gek_shared"
    m = train_gek_shared_sin(sob, n_iters=steps, lr=LR, seed=seed, l0=L0, lam0=LAM0)
    v, ms = eval_gek_shared_sin(m, seed=seed)
    return m, v, ms


def _backup(p):
    if os.path.exists(p):
        d = os.path.join(_REPO_ROOT, "results", "superseded")
        os.makedirs(d, exist_ok=True)
        q = os.path.join(d, os.path.basename(p))
        if not os.path.exists(q):
            shutil.copy2(p, q)
            print(f"[backup] {p} -> {q}")


def run_mse(seeds):
    out = {}
    for method, steps in ORDER:
        per, ms, learned = [], [], []
        for s in seeds:
            m, v, dt = train(method, steps, s)
            per.append(v); ms.append(dt); learned.append(m)
            print(f"  {method}_{steps} seed {s}: mean {np.mean(v):.4f} median {np.median(v):.2e} | "
                  + " ".join(f"{k} {m[k]:.3g}" for k in ("sigma", "mu", "w", "l", "lam_y", "lam_g")
                             if m.get(k) is not None), flush=True)
        name = NAMES[(method, steps)]
        out[name] = {"id": f"{method}_{steps}", "steps": steps, **summarise(per), "learned": learned,
                     "train_s": float(np.mean([m["train_s"] for m in learned])),
                     "ms_per_task": float(np.mean(ms))}
        print(f"=> {name}: {out[name]['mean']:.4f} +/- {out[name]['mean_std']:.4f} "
              f"(median {out[name]['median']:.2e})", flush=True)
    out["_protocol"] = ("experiments/rebuttal_gek_shared_matched.py --part mse: Adam lr 5e-3, GP/GEK init "
                        "l0=1 lam0=1e-4, Meta-LSQ init sigma=1 mu=1e-4 w=1; training stream gen seed+1000, "
                        "batch 4, K=10, K_q=10; test 200 tasks gen seed+9999, K=10, K_q=50; seeds "
                        f"{list(seeds)}. mean_std = sample std over seeds of per-seed means. train_s here "
                        "is not a timing measurement (shared machine); see rebuttal_gek_shared_timing.json.")
    return out


def run_timing(seeds):
    for method in ("metalsq_sob", "gek_shared", "metalsq_val", "gp_val_shared"):   # untimed warm-up
        train(method, 20, 0)
    out = {}
    for method, steps in ORDER:
        key = f"{method}_{steps}"
        out[key] = {"train_s": [], "mse": []}
        for s in seeds:
            m, v, _ = train(method, steps, s)
            out[key]["train_s"].append(m["train_s"]); out[key]["mse"].append(float(np.mean(v)))
            print(f"  {key} seed {s}: train {m['train_s']:.2f}s  mse {np.mean(v):.4f}", flush=True)
        print(f"=> {key}: train {np.mean(out[key]['train_s']):.2f}s  mse {np.mean(out[key]['mse']):.4f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("mse", "timing"), default="mse")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    save = args.save or ("results/rebuttal_gek_shared_matched.json" if args.part == "mse"
                         else "results/rebuttal_gek_shared_timing.json")
    t0 = time.time()
    blob = run_mse(seeds) if args.part == "mse" else run_timing(seeds)
    p = save if os.path.isabs(save) else os.path.join(_REPO_ROOT, save)
    _backup(p)
    with open(p, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"\n[saved] {p}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
