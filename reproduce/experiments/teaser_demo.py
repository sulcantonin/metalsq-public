"""Capture one representative task fit for the paper's teaser figure.

Meta-trains MLP-MAML and Meta-LSQ-Sobolev on the sinusoid family exactly as the
headline experiment does, then records --- for a single held-out task --- the
support points, the analytic support gradients, the true function, and each
method's prediction on a dense grid.  Nothing here is a new result; it is the
headline experiment rendered visible.

Like every other script here it is plotting-free: it writes JSON that
paper/*/make_figures.py reads.

Run:  OMP_NUM_THREADS=1 python3 experiments/teaser_demo.py --save results/teaser.json
"""

import argparse
import json

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import torch

from experiments.maml_sinusoid import (
    train_mlp_maml, train_fastlsq_meta,
    fastlsq_inner_solve, fastlsq_predict,
    maml_inner_step, sample_tasks,
)

torch.set_default_dtype(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=None,
                    help="held-out task index; default picks the MEDIAN task (see below)")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    print("meta-training MLP-MAML (8000 iters)...", flush=True)
    mlp, _, _ = train_mlp_maml(n_iters=8000, seed=a.seed, log_every=4000)
    print("meta-training Meta-LSQ-Sobolev (2000 iters)...", flush=True)
    meta, _, _ = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=a.seed, log_every=1000)
    print("meta-training Meta-LSQ value-only, w=0 (2000 iters)...", flush=True)
    meta_v, _, _ = train_fastlsq_meta(sobolev=False, n_iters=2000, seed=a.seed,
                                      log_every=1000)

    # Same held-out stream the evaluators use, so the shown task is a genuine
    # test task rather than a hand-picked one.
    gen = torch.Generator().manual_seed(a.seed + 9999)
    n_test = 200
    xs, ys, dys, xq, yq = sample_tasks(n_test, 10, 50, gen)

    # Score every held-out task, then pick the one to show by a stated rule
    # rather than by eye. Two nuisance effects have to be handled together:
    #   * absolute MSE scales with the task amplitude a ~ U[0.1, 5], so ranking
    #     on it drifts toward small-amplitude tasks -> rank on RELATIVE error
    #     (query MSE over query variance), which is scale-free;
    #   * a task can still be typical in relative error while being atypical in
    #     amplitude, which would make the picture unrepresentative of the family
    #     -> restrict to tasks whose amplitude lies in the family's interquartile
    #     range before taking the median.
    # The result is the median-difficulty task at a typical amplitude.
    amp_lo, amp_hi = 0.1 + 0.25 * 4.9, 0.1 + 0.75 * 4.9     # IQR of a ~ U[0.1, 5]
    per_task, per_task_rel, amp = [], [], []
    with torch.no_grad():
        for i in range(n_test):
            W_i, b_i = fastlsq_inner_solve(
                xs[i], ys[i], dys[i], meta["Wb"], meta["bb"],
                torch.tensor(meta["sigma"]), torch.tensor(meta["mu"]),
                sobolev=True, grad_w=meta["grad_w"])
            mse_i = float((fastlsq_predict(xq[i], W_i, meta["bb"], b_i)
                           - yq[i]).pow(2).mean())
            per_task.append(mse_i)
            per_task_rel.append(mse_i / float(yq[i].var()))
            # y = a sin(x + phi) = (a cos phi) sin x + (a sin phi) cos x, so the
            # amplitude is the norm of the two-term least-squares fit.
            B = torch.cat([torch.sin(xs[i]), torch.cos(xs[i])], dim=-1)
            c = torch.linalg.lstsq(B, ys[i].unsqueeze(-1)).solution
            amp.append(float(c.norm()))
    pool = [i for i in range(n_test) if amp_lo <= amp[i] <= amp_hi]
    order = sorted(pool, key=lambda i: per_task_rel[i])
    t = a.task if a.task is not None else order[len(order) // 2]
    rank = order.index(t) if t in order else -1
    pt = torch.tensor(per_task)
    print(f"per-task MSE over {n_test} held-out tasks: "
          f"mean {float(pt.mean()):.4f}, median {float(pt.median()):.2e}")
    print(f"amplitude pool: {len(pool)}/{n_test} tasks with "
          f"a in [{amp_lo:.2f}, {amp_hi:.2f}]")
    print(f"showing task {t} (rank {rank}/{len(order)} by relative error within "
          f"the pool, MSE {per_task[t]:.2e}, rel {per_task_rel[t]:.2e}, "
          f"amplitude {amp[t]:.2f})")

    grid = torch.linspace(-5.0, 5.0, 400).unsqueeze(-1)

    # The task parameters are recoverable from the support values; we re-derive
    # the true curve by least-squares on (a cos phi, a sin phi), which is exact
    # because y = a sin(x + phi) = (a cos phi) sin x + (a sin phi) cos x.
    A = torch.cat([torch.sin(xs[t]), torch.cos(xs[t])], dim=-1)     # (K, 2)
    coef = torch.linalg.lstsq(A, ys[t].unsqueeze(-1)).solution      # (2, 1)
    true_grid = (torch.cat([torch.sin(grid), torch.cos(grid)], dim=-1) @ coef).squeeze(-1)

    with torch.no_grad():
        W, beta = fastlsq_inner_solve(
            xs[t], ys[t], dys[t], meta["Wb"], meta["bb"],
            torch.tensor(meta["sigma"]), torch.tensor(meta["mu"]),
            sobolev=True, grad_w=meta["grad_w"])
        lsq_grid = fastlsq_predict(grid, W, meta["bb"], beta)

    # the same task fit from the same 10 points but WITHOUT their slopes
    with torch.no_grad():
        Wv, beta_v = fastlsq_inner_solve(
            xs[t], ys[t], dys[t], meta_v["Wb"], meta_v["bb"],
            torch.tensor(meta_v["sigma"]), torch.tensor(meta_v["mu"]),
            sobolev=False)
        lsq_val_grid = fastlsq_predict(grid, Wv, meta_v["bb"], beta_v)
        q_val = fastlsq_predict(xq[t], Wv, meta_v["bb"], beta_v)

    params = mlp.init_params()
    params = maml_inner_step(mlp, params, xs[t], ys[t], 1e-2, first_order=True)
    with torch.no_grad():
        mlp_grid = mlp(grid, params)

    def mse(pred_at_q):
        return float((pred_at_q - yq[t]).pow(2).mean())

    with torch.no_grad():
        q_lsq = fastlsq_predict(xq[t], W, meta["bb"], beta)
    params_q = mlp.init_params()
    params_q = maml_inner_step(mlp, params_q, xs[t], ys[t], 1e-2, first_order=True)
    with torch.no_grad():
        q_mlp = mlp(xq[t], params_q)

    out = {
        "config": {"seed": a.seed, "task": t, "task_rank": rank, "n_test": n_test,
                   "selection": "median relative query error among tasks whose amplitude lies in the family interquartile range",
                   
                   "K": 10, "K_q": 50,
                   "sigma": meta["sigma"], "mu": meta["mu"], "grad_w": meta["grad_w"]},
        "per_task_mse_meta_lsq": {"mean": float(pt.mean()), "median": float(pt.median())},
        "shown_task_amplitude": amp[t],
        "shown_task_rel_err": per_task_rel[t],
        "grid": grid.squeeze(-1).tolist(),
        "true": true_grid.tolist(),
        "meta_lsq_sobolev": lsq_grid.tolist(),
        "meta_lsq_valueonly": lsq_val_grid.tolist(),
        "mlp_maml": mlp_grid.tolist(),
        "support_x": xs[t].squeeze(-1).tolist(),
        "support_y": ys[t].tolist(),
        "support_dy": dys[t].tolist(),
        "task_mse": {"meta_lsq_sobolev": mse(q_lsq),
                     "meta_lsq_valueonly": mse(q_val),
                     "mlp_maml": mse(q_mlp)},
        "basis": {"sigma_star": float(meta["sigma"]),
                  "Wb": meta["Wb"].squeeze(0).tolist()[:64],
                  "bb": meta["bb"].squeeze(0).tolist()[:64]},
    }
    print(f"task {t}: Meta-LSQ-Sobolev MSE {out['task_mse']['meta_lsq_sobolev']:.5f}, "
          f"MLP-MAML MSE {out['task_mse']['mlp_maml']:.5f}")

    if a.save:
        path = a.save if _os.path.isabs(a.save) else _os.path.join(_REPO_ROOT, a.save)
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
