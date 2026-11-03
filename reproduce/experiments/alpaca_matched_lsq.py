"""Value-only Meta-LSQ at ALPaCA's budget: the control behind the ALPaCA comparison.

alpaca_baseline.py trains ALPaCA for 8000 steps on seeds 0-2; the value-only Meta-LSQ
figure it was compared with came from 2000 steps on seeds 0-4, where the ridge is still
falling. Here value-only Meta-LSQ is trained for 8000 steps on the same three seeds
and evaluated on the same 200 test tasks. ``--float64`` runs training and evaluation
in double precision: in float32 the primal ridge solve (200x200, rank <= 10) breaks
down once mu drops below about 1e-7, which caps how small the learned ridge can get.

    python3 experiments/alpaca_matched_lsq.py [--float64] --save results/alpaca_matched_lsq_f64.json
"""
import sys, os, json, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.maml_sinusoid import train_fastlsq_meta, sample_tasks, fastlsq_inner_solve, fastlsq_predict
if "--float64" in sys.argv:           # after the import: maml_sinusoid sets float32 on import
    torch.set_default_dtype(torch.float64)

SEEDS, N_ITERS, N_TEST, K, K_Q = (0, 1, 2), 8000, 200, 10, 50


def per_task_mse(meta, seed):
    gen = torch.Generator().manual_seed(seed + 9999)          # eval_fastlsq_meta's test tasks
    xs, ys, dys, xq, yq = sample_tasks(N_TEST, K, K_Q, gen)
    sigma, mu = torch.tensor(meta["sigma"]), torch.tensor(meta["mu"])
    out = []
    with torch.no_grad():
        for t in range(N_TEST):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], meta["Wb"], meta["bb"], sigma, mu)
            out.append(float((fastlsq_predict(xq[t], W, meta["bb"], beta) - yq[t]).pow(2).mean()))
    return out


rows = []
for seed in SEEDS:
    meta = train_fastlsq_meta(sobolev=False, n_iters=N_ITERS, seed=seed, log_every=10**6)[0]
    e = per_task_mse(meta, seed)
    rows.append({"seed": seed, "sigma": meta["sigma"], "mu": meta["mu"],
                 "mean_mse": statistics.mean(e), "median_mse": statistics.median(e)})
    print(f"seed {seed}: sigma {meta['sigma']:.3f} mu {meta['mu']:.2e} "
          f"mean {rows[-1]['mean_mse']:.4f} median {rows[-1]['median_mse']:.2e}", flush=True)
out = {"dtype": str(torch.get_default_dtype()), "n_iters": N_ITERS, "seeds": list(SEEDS),
       "mean_mse": statistics.mean(r["mean_mse"] for r in rows), "per_seed": rows}
print(f"mean over seeds: {out['mean_mse']:.4f}")
if "--save" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--save") + 1], "w"), indent=2)
