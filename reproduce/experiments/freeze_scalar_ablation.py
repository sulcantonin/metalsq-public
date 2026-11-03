"""FREEZE-scalar ablation for Meta-LSQ-Sobolev on the sinusoid benchmark.

Goal: show that each of the 3 meta-learned log-scalars (sigma, mu, w) genuinely
contributes. We reimplement the train_fastlsq_meta loop (read-only import of the
inner solve / predict / task generator) so that each log-parameter can be either a
frozen Python-constant or an Adam-optimised tensor.

Five configs (3 seeds, 200 test tasks, K=10 / K_q=50):
  (1) ALL learned   (sigma, mu, w)         -- the default Meta-LSQ-Sobolev
  (2) freeze w=1    (learn sigma, mu)
  (3) freeze mu=1e-4 (learn sigma, w)
  (4) freeze sigma=1.0 (learn mu, w)
  (5) freeze ALL at init (sigma=1, mu=1e-4, w=1) -> no meta-learning

The gap of each frozen config from (1) measures that scalar's contribution.

Also: init-sensitivity. Rerun config (1) from 2 different initial log-parameter
values and check convergence to similar (sigma, mu, w).

Single-thread, float32. Run with OMP_NUM_THREADS=1.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import math
import time
import json
import torch

from experiments.maml_sinusoid import (
    sample_tasks,
    make_basis,
    fastlsq_inner_solve,
    fastlsq_predict,
    eval_fastlsq_meta,
)

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# Reimplemented training loop with selective freezing.
#
# Each of the three log-scalars is described by an entry in `init`/`learn`.
# A learned scalar becomes a requires_grad tensor handed to Adam; a frozen
# scalar stays a plain (detached) tensor whose value never changes. This is a
# faithful copy of train_fastlsq_meta (sobolev=True path, torch.sqrt(grad_w)),
# just with the parameter list built conditionally.
# ---------------------------------------------------------------------------

def train_freeze(n_iters=2000, meta_bs=4, K=10, K_q=10, N=200,
                 init_sigma=1.0, init_mu=1e-4, init_w=1.0,
                 learn_sigma=True, learn_mu=True, learn_w=True,
                 lr=5e-3, seed=0, log_every=100, noise=0.0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)

    # Build log-parameters. Learned ones get requires_grad and go to Adam.
    log_sigma = torch.tensor(math.log(init_sigma), requires_grad=learn_sigma)
    log_mu = torch.tensor(math.log(init_mu), requires_grad=learn_mu)
    log_gw = torch.tensor(math.log(init_w), requires_grad=learn_w)

    opt_params = []
    if learn_sigma:
        opt_params.append(log_sigma)
    if learn_mu:
        opt_params.append(log_mu)
    if learn_w:
        opt_params.append(log_gw)
    opt = torch.optim.Adam(opt_params, lr=lr) if opt_params else None

    curve = []
    traj = []  # (iter, sigma, mu, w) for init-sensitivity tracking
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, dys, xq, yq = sample_tasks(meta_bs, K, K_q, gen, noise=noise)
        if opt is not None:
            opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        gw = log_gw.exp()  # always sobolev=True here
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=True, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        if opt is not None:
            meta_loss.backward()
            opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            curve.append((time.time() - t0, it, float(meta_loss.item())))
            traj.append((it,
                         float(log_sigma.exp().item()),
                         float(log_mu.exp().item()),
                         float(log_gw.exp().item())))
    train_time = time.time() - t0

    meta = {"Wb": Wb, "bb": bb,
            "sigma": float(log_sigma.exp().item()),
            "mu": float(log_mu.exp().item()),
            "grad_w": float(log_gw.exp().item()),
            "sobolev": True}
    return meta, train_time, curve, traj


def run_config(name, learn_sigma, learn_mu, learn_w, seeds,
               init_sigma=1.0, init_mu=1e-4, init_w=1.0,
               n_iters=2000):
    mses, sigmas, mus, ws, ttimes = [], [], [], [], []
    for s in seeds:
        meta, ttime, _, _ = train_freeze(
            n_iters=n_iters, seed=s,
            init_sigma=init_sigma, init_mu=init_mu, init_w=init_w,
            learn_sigma=learn_sigma, learn_mu=learn_mu, learn_w=learn_w)
        mse, _ = eval_fastlsq_meta(meta, n_test=200, K=10, K_q=50, seed=s)
        mses.append(mse)
        sigmas.append(meta["sigma"])
        mus.append(meta["mu"])
        ws.append(meta["grad_w"])
        ttimes.append(ttime)
    out = {
        "name": name,
        "learn": {"sigma": learn_sigma, "mu": learn_mu, "w": learn_w},
        "init": {"sigma": init_sigma, "mu": init_mu, "w": init_w},
        "seeds": list(seeds),
        "mse_per_seed": mses,
        "mse_mean": float(sum(mses) / len(mses)),
        "mse_std": float(torch.tensor(mses).std(unbiased=False).item()),
        "sigma_per_seed": sigmas,
        "mu_per_seed": mus,
        "w_per_seed": ws,
        "sigma_mean": float(sum(sigmas) / len(sigmas)),
        "mu_mean": float(sum(mus) / len(mus)),
        "w_mean": float(sum(ws) / len(ws)),
        "train_s_mean": float(sum(ttimes) / len(ttimes)),
    }
    return out


def main():
    seeds = [0, 1, 2]
    n_iters = 2000
    results = {"meta": {"seeds": seeds, "n_iters": n_iters,
                        "n_test": 200, "K": 10, "K_q": 50,
                        "init_default": {"sigma": 1.0, "mu": 1e-4, "w": 1.0}}}

    configs = [
        ("all_learned",   True,  True,  True),
        ("freeze_w",      True,  True,  False),
        ("freeze_mu",     True,  False, True),
        ("freeze_sigma",  False, True,  True),
        ("freeze_all",    False, False, False),
    ]

    print("=== FREEZE-scalar ablation (Meta-LSQ-Sobolev, sinusoid) ===")
    print(f"seeds={seeds}, n_iters={n_iters}, 200 test tasks\n")
    cfg_out = {}
    for name, ls, lm, lw in configs:
        t0 = time.time()
        out = run_config(name, ls, lm, lw, seeds, n_iters=n_iters)
        cfg_out[name] = out
        print(f"{name:<14} MSE={out['mse_mean']:.4f} +/- {out['mse_std']:.4f}  "
              f"sigma={out['sigma_mean']:.3f} mu={out['mu_mean']:.2e} "
              f"w={out['w_mean']:.3f}  ({time.time()-t0:.1f}s)")
    results["configs"] = cfg_out

    base = cfg_out["all_learned"]["mse_mean"]
    print("\n--- Gap from all-learned (degradation when frozen) ---")
    gaps = {}
    for name in ["freeze_w", "freeze_mu", "freeze_sigma", "freeze_all"]:
        gap = cfg_out[name]["mse_mean"] - base
        gaps[name] = float(gap)
        print(f"  {name:<14} delta_MSE = {gap:+.4f}  ({cfg_out[name]['mse_mean']:.4f} vs {base:.4f})")
    results["gaps_from_all_learned"] = gaps

    # Which single scalar matters most: largest degradation among the three
    # single-freeze configs.
    single = {"freeze_w": gaps["freeze_w"],
              "freeze_mu": gaps["freeze_mu"],
              "freeze_sigma": gaps["freeze_sigma"]}
    worst = max(single, key=single.get)
    scalar_name = {"freeze_w": "w", "freeze_mu": "mu", "freeze_sigma": "sigma"}[worst]
    results["most_important_scalar"] = scalar_name
    print(f"\nMost important scalar (largest single-freeze degradation): "
          f"{scalar_name} ({worst}, delta={single[worst]:+.4f})")

    # ---- Init-sensitivity: config (1) from 2 different initial log-params ----
    # We test convergence to a similar (sigma, mu, w) from 3 inits (default +
    # two perturbations) with 3000 iters / 2 seeds. The default-init basin (from
    # config 1) sits at sigma~0.6, mu~1e-5; a fair test perturbs around it. We
    # also probe two *extreme* inits (1 seed, 2000 iters) to honestly expose the
    # large-sigma conditioning pathology.
    init_iters = 3000
    init_seeds = [0, 1]
    print(f"\n--- Init-sensitivity (config 1, all learned, {init_iters} iters, "
          f"seeds={init_seeds}) ---")
    init_sets = [
        ("init_default", {"init_sigma": 1.0, "init_mu": 1e-4, "init_w": 1.0}),
        ("init_low",     {"init_sigma": 0.5, "init_mu": 1e-3, "init_w": 0.3}),
        ("init_high",    {"init_sigma": 2.0, "init_mu": 1e-5, "init_w": 3.0}),
    ]
    init_out = {}
    for label, kw in init_sets:
        per = {"sigma": [], "mu": [], "w": [], "mse": [], "traj_seed0": None}
        for s in init_seeds:
            meta, _, _, traj = train_freeze(
                n_iters=init_iters, seed=s,
                learn_sigma=True, learn_mu=True, learn_w=True, **kw)
            mse, _ = eval_fastlsq_meta(meta, n_test=200, K=10, K_q=50, seed=s)
            per["sigma"].append(meta["sigma"])
            per["mu"].append(meta["mu"])
            per["w"].append(meta["grad_w"])
            per["mse"].append(mse)
            if s == init_seeds[0]:
                per["traj_seed0"] = traj
        per["sigma_mean"] = float(sum(per["sigma"]) / len(per["sigma"]))
        per["mu_mean"] = float(sum(per["mu"]) / len(per["mu"]))
        per["w_mean"] = float(sum(per["w"]) / len(per["w"]))
        per["mse_mean"] = float(sum(per["mse"]) / len(per["mse"]))
        per["init"] = kw
        init_out[label] = per
        print(f"  {label:13s}: init(sigma={kw['init_sigma']}, mu={kw['init_mu']:.0e}, "
              f"w={kw['init_w']}) -> "
              f"converged sigma={per['sigma_mean']:.3f} mu={per['mu_mean']:.2e} "
              f"w={per['w_mean']:.3f}  MSE={per['mse_mean']:.4f}")
    results["init_sensitivity"] = init_out
    results["init_sensitivity_meta"] = {"n_iters": init_iters, "seeds": init_seeds}

    # ---- Extreme-init pathology (honest caveat): 1 seed, 2000 iters ----
    print("\n--- Extreme-init probe (1 seed, 2000 iters) ---")
    extreme_sets = [
        ("extreme_lowsig", {"init_sigma": 0.3, "init_mu": 1e-2, "init_w": 0.1}),
        ("extreme_highsig", {"init_sigma": 3.0, "init_mu": 1e-6, "init_w": 10.0}),
    ]
    extreme_out = {}
    for label, kw in extreme_sets:
        meta, _, _, _ = train_freeze(
            n_iters=2000, seed=0,
            learn_sigma=True, learn_mu=True, learn_w=True, **kw)
        mse, _ = eval_fastlsq_meta(meta, n_test=200, K=10, K_q=50, seed=0)
        extreme_out[label] = {"init": kw, "sigma": meta["sigma"],
                              "mu": meta["mu"], "w": meta["grad_w"], "mse": mse}
        print(f"  {label:16s}: init(sigma={kw['init_sigma']}, mu={kw['init_mu']:.0e}, "
              f"w={kw['init_w']}) -> sigma={meta['sigma']:.3f} mu={meta['mu']:.2e} "
              f"w={meta['grad_w']:.3f}  MSE={mse:.4f}")
    results["extreme_init_probe"] = extreme_out

    os.makedirs(_os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    outpath = _os.path.join(_REPO_ROOT, "results", "freeze_scalar_ablation.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {outpath}")
    return results


if __name__ == "__main__":
    main()
