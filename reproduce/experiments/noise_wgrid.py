# Seed-0 diagnostic behind App. wmodel: train Meta-LSQ-Sobolev (values-only noise), then score
# the test tasks over a grid of w. Usage: python3 experiments/noise_wgrid.py SEED ITERS OUT.json
import sys, math, torch, io, contextlib, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maml_sinusoid as M
seed = int(sys.argv[1]); iters = int(sys.argv[2])
out = {}
for nz in (0.0, 0.5):
    with contextlib.redirect_stdout(io.StringIO()):
        meta = M.train_fastlsq_meta(sobolev=True, n_iters=iters, seed=seed, noise=nz, grad_noise=0.0, log_every=10**6)[0]
    row = {"sigma": meta["sigma"], "mu": meta["mu"], "w_learned": meta["grad_w"], "grid": {}}
    for w in (meta["grad_w"], 0.3, 1, 3, 10, 30, 100, 1000):
        m = dict(meta); m["grad_w"] = float(w)
        mse = M.eval_fastlsq_meta(m, seed=seed, noise=nz, grad_noise=0.0)[0]
        row["grid"][f"{w:.3g}"] = mse
    out[str(nz)] = row
    print(f"seed {seed} iters {iters} noise {nz}: sigma {meta['sigma']:.3f} mu {meta['mu']:.2e} w {meta['grad_w']:.2f} | " +
          " ".join(f"w={k}:{v:.4f}" for k, v in row["grid"].items()), flush=True)
json.dump(out, open(f"{sys.argv[3]}", "w"), indent=1)
