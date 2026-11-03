"""Dimensionality scan: FastLSQ-Sobolev on GMM log-density for d in {4,8,16,32,64}.

Addresses the curse-of-dimensionality elephant. The paper currently flags d=32
collapse as an appendix-TODO; we move it into the main body by measuring it.

Output: a table of value/grad/Hessian rRMSE vs d. Adds an MLP column for d <= 16
as a reality check.
"""

import os
import sys
import time
import math
import torch

# Allow `python experiments/dimensionality_scan.py` to find the `experiments`
# package without setting PYTHONPATH (repo root = parent of this file's dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.derivative_oracle_test import GMM, MLP, mlp_eval, fastlsq_eval, relrmse
from experiments.sobolev_test import fit_fastlsq_sobolev, fit_mlp_sobolev

torch.set_default_dtype(torch.float32)
SEED = 0


def run_one(d, n_train=2000, n_test=400, N=1500, eval_mlp=True):
    print(f"\n========= d = {d} =========")
    gmm = GMM(d, K=3, seed=SEED)
    X_tr = gmm.sample(n_train, seed=SEED)
    y_tr = gmm.log_p(X_tr)
    dy_tr = gmm.grad_log_p(X_tr)
    X_val = gmm.sample(n_test, seed=SEED + 2)   # held-out set for bandwidth selection
    y_val = gmm.log_p(X_val)
    X_te = gmm.sample(n_test, seed=SEED + 1)
    y_te = gmm.log_p(X_te)
    g_te = gmm.grad_log_p(X_te)
    H_te = gmm.hess_log_p(X_te)

    out = {"d": d}

    # FastLSQ-Sobolev
    t = time.time()
    basis, beta, scale = fit_fastlsq_sobolev(gmm, X_tr, y_tr, dy_tr, N=N,
                                              use_grad=True, grad_weight=1.0,
                                              X_val=X_val, y_val=y_val)
    t_fl = time.time() - t
    try:
        fl_v, fl_g, fl_H = fastlsq_eval(basis, beta, X_te)
        out["fl"] = (relrmse(fl_v, y_te), relrmse(fl_g, g_te), relrmse(fl_H, H_te), t_fl, scale)
    except Exception as e:
        print(f"  FastLSQ eval failed at d={d}: {e}")
        out["fl"] = (float("nan"), float("nan"), float("nan"), t_fl, scale)

    # MLP-Sobolev (skip when too expensive)
    if eval_mlp:
        t = time.time()
        net = fit_mlp_sobolev(d, X_tr, y_tr, dy_tr, grad_weight=1.0)
        t_mlp = time.time() - t
        try:
            mlp_v, mlp_g, mlp_H = mlp_eval(net, X_te)
            out["mlp"] = (relrmse(mlp_v, y_te), relrmse(mlp_g, g_te), relrmse(mlp_H, H_te), t_mlp)
        except Exception as e:
            print(f"  MLP eval failed at d={d}: {e}")
            out["mlp"] = (float("nan"), float("nan"), float("nan"), t_mlp)
    else:
        out["mlp"] = None

    fv, fg, fH, tf, sc = out["fl"]
    print(f"  FastLSQ-Sobolev:  val={fv:.2e}  grad={fg:.2e}  hess={fH:.2e}  fit={tf:.1f}s  sigma={sc}")
    if out["mlp"]:
        mv, mg, mH, tm = out["mlp"]
        print(f"  MLP-Sobolev:      val={mv:.2e}  grad={mg:.2e}  hess={mH:.2e}  fit={tm:.1f}s")
    return out


def main(save=None):
    results = []
    # MLP is expensive in higher d (double-backprop), so cap it at d<=16
    for d in (4, 8, 16, 32, 64):
        results.append(run_one(d, eval_mlp=(d <= 16)))

    print("\n=== Summary: FastLSQ-Sobolev on GMM log-density ===")
    print(f"{'d':>3} | {'val rRMSE':>10} {'grad rRMSE':>11} {'hess rRMSE':>11} | sigma | fit s")
    for r in results:
        v, g, h, t, s = r["fl"]
        print(f"{r['d']:>3} | {v:>10.2e} {g:>11.2e} {h:>11.2e} | {s} | {t:.1f}")

    print("\n=== MLP-Sobolev (where evaluated) ===")
    print(f"{'d':>3} | {'val rRMSE':>10} {'grad rRMSE':>11} {'hess rRMSE':>11} | fit s")
    for r in results:
        if r["mlp"] is None:
            print(f"{r['d']:>3} |   (skipped, too expensive)")
            continue
        v, g, h, t = r["mlp"]
        print(f"{r['d']:>3} | {v:>10.2e} {g:>11.2e} {h:>11.2e} | {t:.1f}")

    if save:
        import json
        packed = []
        for r in results:
            fv, fg, fh, ft, fs = r["fl"]
            entry = {"d": r["d"],
                     "fl": {"val": fv, "grad": fg, "hess": fh, "fit_s": ft, "sigma": fs}}
            if r["mlp"] is not None:
                mv, mg, mh, mt = r["mlp"]
                entry["mlp"] = {"val": mv, "grad": mg, "hess": mh, "fit_s": mt}
            packed.append(entry)
        with open(save, "w") as f:
            json.dump(packed, f, indent=2)
        print(f"\n[saved] {save}")


if __name__ == "__main__":
    import sys
    _save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    main(save=_save)
