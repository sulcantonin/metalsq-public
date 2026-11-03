"""Review-response experiments:
  (A) GP-with-derivatives (gradient-enhanced kriging) baseline on the sinusoid
      benchmark -- the closest mathematical comparator (kernel ridge with
      derivative observations), per-task RBF length-scale by marginal likelihood.
      One seed (test tasks of seed 0); its std is over the 200 test tasks.
  (B) Derivative-aware meta-evaluation: query VALUE, GRADIENT, and HESSIAN MSE
      for value-only (w=0) vs Sobolev (w meta-learned) meta-learners, seeds 0-4
      -- ties the Sobolev benefit to the meta-learning task, not just per-task
      GMM fitting.
  (C) Gradient-supervision-weight sweep: for each fixed w in {0,0.1,0.3,1,3,10},
      meta-train (sigma, mu) with w FROZEN at that value (2000 Adam steps, the
      Table-1 task stream), then report query value+grad MSE, seeds 0-2.
  (D) Feature-count sweep: N in {50,100,200}, Sobolev (w meta-learned), test
      MSE, seeds 0-2. (N=400/800 single-thread solves did not finish in budget.)

Every evaluation uses the Table-1 protocol: 200 held-out tasks per seed
(generator seed+9999), K=10 support, K_q=50 query. Unless stated otherwise,
"mean +/- std" is over seeds of the per-seed mean over the 200 test tasks
(sample std, ddof=1).

Single-thread CPU.  OMP_NUM_THREADS=1 python3 experiments/review_experiments.py --save results/review.json
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
torch.set_default_dtype(torch.float32)
from experiments.maml_sinusoid import (train_fastlsq_meta, eval_fastlsq_meta, make_basis,
                                        sample_tasks, fastlsq_inner_solve, fastlsq_predict)

SEEDS = (0, 1, 2, 3, 4)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _ms(xs, per_seed=False):
    t = torch.tensor(xs, dtype=torch.float64)
    out = {"mean": float(t.mean()), "std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0}
    if per_seed:
        out["per_seed"] = [float(x) for x in xs]
    return out


# ---------- RFF model derivative predictions (cyclic identity) ----------
def rff_predict(xq, W, bb, beta, order=0):
    """order 0=value, 1=gradient, 2=Hessian (d=1) of u_N at xq."""
    N = W.shape[1]
    Z = xq @ W + bb
    if order == 0:
        H = torch.sin(Z) / math.sqrt(N)
    elif order == 1:
        H = (W * torch.cos(Z)) / math.sqrt(N)
    else:
        H = (-(W ** 2) * torch.sin(Z)) / math.sqrt(N)
    return H @ beta


# ---------- (A) gradient-enhanced kriging (GP with derivatives) ----------
def gek_solve_predict(xs, ys, dys, xq, l, sig_n2=1e-4):
    """1-D RBF gradient-enhanced kriging. xs,ys,dys: (K,); xq: (Kq,). Returns query value pred."""
    xs = xs.double(); ys = ys.double(); dys = dys.double(); xq = xq.double()
    K = xs.shape[0]
    R = xs[:, None] - xs[None, :]                       # (K,K)
    k = torch.exp(-R ** 2 / (2 * l ** 2))
    Kff = k
    Kfg = k * (R / l ** 2)                              # cov(f_i, f'_j)
    Kgg = k * (1.0 / l ** 2 - R ** 2 / l ** 4)
    top = torch.cat([Kff, Kfg], dim=1)
    bot = torch.cat([Kfg.T, Kgg], dim=1)
    G = torch.cat([top, bot], dim=0) + sig_n2 * torch.eye(2 * K).double()
    t = torch.cat([ys, dys])
    alpha = torch.linalg.solve(G, t)
    Rq = xq[:, None] - xs[None, :]                      # (Kq,K)
    kq = torch.exp(-Rq ** 2 / (2 * l ** 2))
    Kqf = kq
    Kqg = kq * (Rq / l ** 2)                            # cov(f(xq), f'(xs))
    kstar = torch.cat([Kqf, Kqg], dim=1)               # (Kq, 2K)
    return (kstar @ alpha).float()


def gek_marglik(xs, ys, dys, l, sig_n2):
    xs = xs.double(); K = xs.shape[0]
    R = xs[:, None] - xs[None, :]
    k = torch.exp(-R ** 2 / (2 * l ** 2))
    G = torch.cat([torch.cat([k, k * (R / l ** 2)], 1),
                   torch.cat([(k * (R / l ** 2)).T, k * (1 / l ** 2 - R ** 2 / l ** 4)], 1)], 0)
    G = G + sig_n2 * torch.eye(2 * K).double()
    t = torch.cat([ys.double(), dys.double()])
    L = torch.linalg.cholesky(G)
    a = torch.cholesky_solve(t.unsqueeze(-1), L).squeeze(-1)
    return float(-0.5 * (t @ a) - torch.log(torch.diag(L)).sum())


def gp_derivatives_baseline(n_test=200, K=10, K_q=50, seed=0):
    print("\n==== (A) GP-with-derivatives (gradient-enhanced kriging) ====")
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)
    ls = (0.3, 0.5, 0.8, 1.2, 2.0, 3.0)
    mses = []; t0 = time.time()
    for t in range(n_test):
        xst = xs[t].squeeze(-1); xqt = xq[t].squeeze(-1)
        best = None
        for l in ls:
            ml = gek_marglik(xst, ys[t], dys[t], l, 1e-4)
            if best is None or ml > best[0]:
                best = (ml, l)
        pred = gek_solve_predict(xst, ys[t], dys[t], xqt, best[1])
        mses.append(((pred - yq[t]) ** 2).mean().item())
    dt = (time.time() - t0) / n_test * 1e3
    out = {"test_mse": _ms(mses), "ms_per_task": dt}
    print(f"  GP-with-derivatives: test MSE {out['test_mse']['mean']:.4f}  ({dt:.1f} ms/task)")
    return out


# ---------- (B) derivative-aware meta-evaluation ----------
def eval_derivative_metrics(meta, n_test=200, K=10, K_q=50, seed=0):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)
    # true query derivatives are analytic once (a, phi) of each task is known; the generator
    # does not expose them, so they are recovered from the first (noiseless) support point below.
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"]); mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    vmse, gmse, hmse = [], [], []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                                          sobolev=meta["sobolev"], grad_w=gw)
            # y=a sin(x+phi), dy=a cos(x+phi) => a=sqrt(y^2+dy^2), phi from atan2 (noiseless).
            x0 = xs[t][0, 0]; y0 = ys[t][0]; dy0 = dys[t][0]
            a = torch.sqrt(y0 ** 2 + dy0 ** 2)
            phi = torch.atan2(y0, dy0) - x0                # since y=a sin(x+phi), dy=a cos(x+phi)
            xqt = xq[t].squeeze(-1)
            gt = a * torch.cos(xqt + phi)
            ht = -a * torch.sin(xqt + phi)
            vp = rff_predict(xq[t], W, bb, beta, 0)
            gp = rff_predict(xq[t], W, bb, beta, 1)
            hp = rff_predict(xq[t], W, bb, beta, 2)
            # value error against the stored query targets, exactly the Table-1 MSE
            vmse.append(((vp - yq[t]) ** 2).mean().item())
            gmse.append(((gp - gt) ** 2).mean().item())
            hmse.append(((hp - ht) ** 2).mean().item())
    return {"val": _ms(vmse), "grad": _ms(gmse), "hess": _ms(hmse)}


def derivative_metrics(seeds=SEEDS):
    print("\n==== (B) derivative-aware meta-evaluation (value-only vs Sobolev) ====")
    agg = {"vanilla": {"val": [], "grad": [], "hess": []},
           "sobolev": {"val": [], "grad": [], "hess": []}}
    learned = {"vanilla": [], "sobolev": []}
    for s in seeds:
        fl = train_fastlsq_meta(sobolev=False, n_iters=2000, seed=s, log_every=100000)[0]
        fls = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=s, log_every=100000)[0]
        for tag, meta in (("vanilla", fl), ("sobolev", fls)):
            r = eval_derivative_metrics(meta, seed=s)
            for k in ("val", "grad", "hess"):
                agg[tag][k].append(r[k]["mean"])
            learned[tag].append({"sigma": meta["sigma"], "mu": meta["mu"], "w": meta["grad_w"]})
        print(f"  seed {s}: vanilla val={agg['vanilla']['val'][-1]:.4f} | sobolev val={agg['sobolev']['val'][-1]:.4f} "
              f"(w={fls['grad_w']:.2f})", flush=True)
    out = {tag: {**{k: _ms(agg[tag][k], per_seed=True) for k in ("val", "grad", "hess")},
                 "learned": learned[tag]} for tag in agg}
    out["seeds"] = list(seeds)
    for tag in ("vanilla", "sobolev"):
        o = out[tag]
        print(f"  {tag:8s} val={o['val']['mean']:.4f} grad={o['grad']['mean']:.3f} hess={o['hess']['mean']:.3f}")
    return out


# ---------- (C) gradient-weight sweep (fixed w) ----------
def train_fastlsq_meta_fixed_w(w, n_iters=2000, meta_bs=4, K=10, K_q=10, N=200, lr=5e-3, seed=0):
    """train_fastlsq_meta with the derivative weight FROZEN at ``w``: Adam updates
    only (log sigma, log mu). Same basis, initialisation, task stream (generator
    seed+1000) and optimiser as Table 1. w=0 drops the derivative rows, which is
    exactly the value-only meta-learner."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu], lr=lr)
    sobolev = w > 0
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, dys, xq, yq = sample_tasks(meta_bs, K, K_q, gen)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                                          sobolev=sobolev, grad_w=float(w))
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()
    return {"Wb": Wb, "bb": bb, "sigma": float(log_sigma.exp().item()),
            "mu": float(log_mu.exp().item()), "grad_w": float(w) if sobolev else None,
            "sobolev": sobolev}, time.time() - t0


def w_sweep(ws=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0), seeds=(0, 1, 2)):
    print("\n==== (C) gradient-supervision-weight sweep (w frozen during meta-training) ====")
    rows = []
    for w in ws:
        vmse, gmse, hp = [], [], []
        for s in seeds:
            meta, _ = train_fastlsq_meta_fixed_w(w, n_iters=2000, seed=s)
            r = eval_derivative_metrics(meta, seed=s)
            vmse.append(r["val"]["mean"]); gmse.append(r["grad"]["mean"])
            hp.append({"sigma": meta["sigma"], "mu": meta["mu"]})
        rows.append({"w": w, "val_mse": _ms(vmse, per_seed=True), "grad_mse": _ms(gmse, per_seed=True),
                     "learned": hp, "seeds": list(seeds)})
        print(f"  w={w:5.1f}: val={rows[-1]['val_mse']['mean']:.4f}+/-{rows[-1]['val_mse']['std']:.4f}  "
              f"grad={rows[-1]['grad_mse']['mean']:.4f}+/-{rows[-1]['grad_mse']['std']:.4f}  "
              f"sigma={np.mean([h['sigma'] for h in hp]):.3f} mu={np.mean([h['mu'] for h in hp]):.1e}",
              flush=True)
    return rows


# ---------- (D) feature-count sweep ----------
def n_sweep(Ns=(50, 100, 200), seeds=(0, 1, 2)):
    print("\n==== (D) feature-count (N) sweep, Sobolev ====")
    rows = []
    for N in Ns:
        mses, hp = [], []
        for s in seeds:
            fls = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=s, N=N, log_every=100000)[0]
            mses.append(eval_fastlsq_meta(fls, seed=s)[0])
            hp.append({"sigma": fls["sigma"], "mu": fls["mu"], "w": fls["grad_w"]})
        rows.append({"N": N, "test_mse": _ms(mses, per_seed=True), "learned": hp, "seeds": list(seeds)})
        print(f"  N={N:4d}: test MSE {rows[-1]['test_mse']['mean']:.4f} +/- {rows[-1]['test_mse']['std']:.4f}",
              flush=True)
    return rows


def main(save=None):
    t0 = time.time()
    out = {"gp_derivatives": gp_derivatives_baseline(),
           "derivative_metrics": derivative_metrics(),
           "w_sweep": w_sweep(),
           "n_sweep": n_sweep()}
    out["_note"] = ("Written by experiments/review_experiments.py (post grad_w-gradient fix). "
                    "std = sample std over seeds of per-seed means (gp_derivatives: over its 200 test "
                    "tasks, seed 0 only). w_sweep: w frozen during meta-training, seeds 0-2; "
                    "n_sweep: seeds 0-2; derivative_metrics: seeds 0-4. "
                    f"Total wall time {time.time()-t0:.0f}s on a shared machine.")
    print(f"\nreview experiments done in {time.time()-t0:.0f}s")
    if save:
        with open(save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {save}")
    return out


if __name__ == "__main__":
    _save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    main(save=_save)
