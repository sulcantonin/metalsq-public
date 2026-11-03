"""Diagnostic for the shared-hyperparameter exact-GP control (reviewer dX2X).

Adam-trained shared hyperparameters for an exact RBF GEK gave 0.045 on the
10-shot sinusoid benchmark against 0.019 for Meta-LSQ-Sobolev, although the
paper says Meta-LSQ is the N -> infinity limit of exactly that model. This
script separates three explanations:

  (a) implied     Evaluate the exact GEK at the hyperparameters Meta-LSQ itself
                  learned, l = 1/sigma*, lam_y = mu*, lam_g = mu*/w*, per seed on
                  the same 200 test tasks. If this matches 0.019 the limit
                  statement holds and the gap is the optimiser's.
  (b) grid        Optimiser-free shared selection: pick (l, lam_y, lam_g) by mean
                  query MSE on 200 held-out VALIDATION tasks (generator seed+2000,
                  disjoint from the seed+9999 test tasks), then test. This is the
                  best a shared exact GEK can do on this benchmark up to grid
                  resolution.
  (c) adam-long   Adam from a closer initialisation with more steps.

Cholesky jitter fall-backs are counted: if the exact kernel needs jitter where
the 200-feature ridge system does not, the finite basis is doing numerical work
the exact kernel cannot.

Run:  OMP_NUM_THREADS=1 python3 experiments/rebuttal_gek_shared_diag.py --save results/rebuttal_gek_shared_diag.json
"""
import os, sys, math, json, time, argparse

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import sample_tasks, train_fastlsq_meta, eval_fastlsq_meta
from experiments.periodic_gek import k_rbf, gek_gram
from experiments.rebuttal_real_hypotheses import train_gek_shared_sin, eval_gek_shared_sin, summarise

SEEDS = (0, 1, 2, 3, 4)
JIT = {"calls": 0, "jitter": 0, "fail": 0}


def chol_batched(A):
    """Batched Cholesky with escalating jitter; counts how often jitter was needed."""
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype)
    JIT["calls"] += A.shape[0]
    L, info = torch.linalg.cholesky_ex(A)
    bad = info != 0
    jit = 1e-10
    while bool(bad.any()) and jit <= 1e-2:
        JIT["jitter"] += int(bad.sum()) if jit == 1e-10 else 0
        L2, info2 = torch.linalg.cholesky_ex(A + jit * I)
        L = torch.where(bad[:, None, None], L2, L)
        bad = bad & (info2 != 0)
        jit *= 100
    JIT["fail"] += int(bad.sum())
    return L, bad


def gek_batch_mse(xs, ys, dys, xq, yq, l, lam_y, lam_g, sobolev=True):
    """Per-task query MSE of the exact RBF GP / GEK posterior mean, batched over tasks."""
    T, K = xs.shape
    R = xs[:, :, None] - xs[:, None, :]
    k, d1, d2 = k_rbf(R, l)
    if sobolev:
        G = gek_gram(k, d1, d2)
        noise = torch.cat([torch.full((K,), lam_y, dtype=torch.float64),
                           torch.full((K,), lam_g, dtype=torch.float64)])
        t = torch.cat([ys, dys], dim=-1)
    else:
        G = k
        noise = torch.full((K,), lam_y, dtype=torch.float64)
        t = ys
    A = G + torch.diag(noise)
    L, bad = chol_batched(A)
    alpha = torch.cholesky_solve(t.unsqueeze(-1), L)
    Rq = xq[:, :, None] - xs[:, None, :]
    kq, d1q, _ = k_rbf(Rq, l)
    kstar = torch.cat([kq, -d1q], dim=-1) if sobolev else kq
    pred = (kstar @ alpha).squeeze(-1)
    mse = ((pred - yq) ** 2).mean(-1)
    mse = torch.where(bad, torch.full_like(mse, float("inf")), mse)
    return mse


def tasks64(n, K, Kq, seed_offset, seed):
    gen = torch.Generator().manual_seed(seed + seed_offset)
    xs, ys, dys, xq, yq = sample_tasks(n, K, Kq, gen)
    return (xs.squeeze(-1).double(), ys.double(), dys.double(), xq.squeeze(-1).double(), yq.double())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    out = {"seeds": list(seeds), "implied": {}, "grid": {}, "adam_long": {}, "metalsq": {}}

    # ---------- (a) exact GEK / GP at the hyperparameters Meta-LSQ learned ----------
    print("== (a) exact kernel at Meta-LSQ's learned hyperparameters ==")
    rows = {"sob": [], "val": []}
    ms_lsq = {"sob": [], "val": []}
    learned = {"sob": [], "val": []}
    for s in seeds:
        for key, sob in (("sob", True), ("val", False)):
            meta, _, _ = train_fastlsq_meta(n_iters=2000, sobolev=sob, seed=s, log_every=10 ** 9)
            mse_lsq, _ = eval_fastlsq_meta(meta, seed=s)
            ms_lsq[key].append(mse_lsq)
            l = 1.0 / meta["sigma"]; lam_y = meta["mu"]
            lam_g = meta["mu"] / meta["grad_w"] if sob else None
            learned[key].append({"sigma": meta["sigma"], "mu": meta["mu"], "w": meta["grad_w"],
                                 "l": l, "lam_y": lam_y, "lam_g": lam_g})
            xs, ys, dys, xq, yq = tasks64(200, 10, 50, 9999, s)
            JIT.update(calls=0, jitter=0, fail=0)
            mse = gek_batch_mse(xs, ys, dys, xq, yq, torch.tensor(l, dtype=torch.float64),
                                lam_y, lam_g if sob else 0.0, sobolev=sob)
            fin = torch.isfinite(mse)
            rows[key].append({"mean": float(mse[fin].mean()), "median": float(mse[fin].median()),
                              "n_fail": int((~fin).sum()), "jitter": JIT["jitter"]})
            print(f"  seed {s} {key}: Meta-LSQ {mse_lsq:.4f} (sigma {meta['sigma']:.3f}, mu {meta['mu']:.1e}, "
                  f"w {meta['grad_w']}) -> exact kernel at implied hp: mean {rows[key][-1]['mean']:.4f} "
                  f"median {rows[key][-1]['median']:.2e}  [jitter {JIT['jitter']}, fail {rows[key][-1]['n_fail']}]",
                  flush=True)
    for key in ("sob", "val"):
        out["implied"][key] = {"per_seed": rows[key],
                               "mean": float(np.mean([r["mean"] for r in rows[key]])),
                               "mean_std": float(np.std([r["mean"] for r in rows[key]], ddof=1))}
        out["metalsq"][key] = {"per_seed_mse": ms_lsq[key], "mean": float(np.mean(ms_lsq[key])),
                               "mean_std": float(np.std(ms_lsq[key], ddof=1)), "learned": learned[key]}
        print(f"  => {key}: Meta-LSQ {out['metalsq'][key]['mean']:.4f}+/-{out['metalsq'][key]['mean_std']:.4f}   "
              f"exact at implied hp {out['implied'][key]['mean']:.4f}+/-{out['implied'][key]['mean_std']:.4f}")

    # ---------- (b) grid-selected shared hyperparameters on validation tasks ----------
    print("\n== (b) shared hyperparameters by grid on 200 validation tasks (seed+2000), tested on seed+9999 ==")
    ls = torch.logspace(math.log10(0.3), math.log10(6.0), 16, dtype=torch.float64)
    lams = [10.0 ** e for e in range(-9, -1)]          # 1e-9 .. 1e-2
    for key, sob in (("sob", True), ("val", False)):
        per_seed = []
        t0 = time.time()
        for s in seeds:
            xv, yv, dv, xqv, yqv = tasks64(200, 10, 10, 2000, s)   # K_q=10 as in meta-training
            best = (float("inf"), None)
            for l in ls:
                for ly in lams:
                    for lg in (lams if sob else [None]):
                        m = gek_batch_mse(xv, yv, dv, xqv, yqv, l, ly, lg if sob else 0.0, sobolev=sob)
                        fin = torch.isfinite(m)
                        if fin.all():
                            v = float(m.mean())
                            if v < best[0]:
                                best = (v, (float(l), ly, lg))
            l, ly, lg = best[1]
            xs, ys, dys, xq, yq = tasks64(200, 10, 50, 9999, s)
            JIT.update(calls=0, jitter=0, fail=0)
            m = gek_batch_mse(xs, ys, dys, xq, yq, torch.tensor(l, dtype=torch.float64), ly, lg if sob else 0.0, sobolev=sob)
            fin = torch.isfinite(m)
            per_seed.append({"l": l, "lam_y": ly, "lam_g": lg, "val_mse": best[0],
                             "test_mean": float(m[fin].mean()), "test_median": float(m[fin].median()),
                             "n_fail": int((~fin).sum())})
            print(f"  seed {s} {key}: best l {l:.3f} lam_y {ly:.0e} lam_g {lg}  val {best[0]:.4f}  "
                  f"-> test mean {per_seed[-1]['test_mean']:.4f} median {per_seed[-1]['test_median']:.2e}", flush=True)
        means = [r["test_mean"] for r in per_seed]
        out["grid"][key] = {"per_seed": per_seed, "mean": float(np.mean(means)),
                            "mean_std": float(np.std(means, ddof=1)), "grid_s": time.time() - t0}
        print(f"  => {key}: grid-selected exact kernel {out['grid'][key]['mean']:.4f}+/-{out['grid'][key]['mean_std']:.4f}")

    # ---------- (c) Adam from a closer initialisation, more steps ----------
    print("\n== (c) Adam, 6000 steps, lr 1e-2, init l=1.5 lam=1e-4 ==")
    for key, sob in (("sob", True), ("val", False)):
        per, learned_c = [], []
        for s in seeds:
            m = train_gek_shared_sin(sob, n_iters=6000, lr=1e-2, seed=s, l0=1.5, lam0=1e-4)
            v, _ = eval_gek_shared_sin(m, seed=s)
            per.append(v); learned_c.append(m)
            print(f"  seed {s} {key}: mean {np.mean(v):.4f} median {np.median(v):.2e} | l {m['l']:.3f} "
                  f"lam_y {m['lam_y']:.1e} lam_g {m['lam_g']}", flush=True)
        out["adam_long"][key] = {**summarise(per), "learned": learned_c}
        print(f"  => {key}: {out['adam_long'][key]['mean']:.4f}+/-{out['adam_long'][key]['mean_std']:.4f}")

    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
