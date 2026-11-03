"""Experiment #2.2: value+grad meta-objective ablation for Meta-LSQ-Sobolev.

Question
--------
The standard Meta-LSQ outer loop minimises QUERY VALUE MSE only; the Sobolev
(derivative-matching) rows enter only the *inner* closed-form solve. So the
query-Hessian improvement we see in the Sobolev variant might be merely
*incidental* to value fitting. Here we test whether optimising a
derivative-aware OUTER objective improves the gradient / Hessian fit further.

Setup
-----
We copy the structure of train_fastlsq_meta(sobolev=True) but change the outer
loss to

    outer = value_query_MSE + alpha * grad_query_MSE

where the predicted query gradient is

    g_hat(xq) = (W * cos(Z)) / sqrt(N)  @  beta        (Z = xq @ W + bb)

and the TRUE query gradient is a*cos(xq+phi). For a sinusoid task the amplitude
and phase are recoverable from ANY support point because
    a   = sqrt(y^2 + dy^2)   (since y=a sin, dy=a cos)
    phi = atan2(y, dy) - x
so the meta-loop never needs ground-truth (a, phi) handed in -- it reconstructs
them from the support (x_s, y_s, dy_s) it already has. We average the recovered
(a, phi) over the K support points for robustness.

We sweep alpha in {0 (value-only default), 0.3, 1.0} and report query VALUE,
GRADIENT and HESSIAN MSE over 200 test tasks, 3 seeds.

Hessian here = second derivative wrt the 1-D input x:
    true:  f''(xq) = -a sin(xq+phi) = -y_q
    pred:  (-(W^2) * sin(Z)) / sqrt(N) @ beta

Run:  OMP_NUM_THREADS=1 python3 experiments/metaobj_valgrad_ablation.py
"""

import sys, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch
torch.set_default_dtype(torch.float32)

from experiments.maml_sinusoid import sample_tasks, make_basis, fastlsq_inner_solve

device = torch.device("cpu")
N = 200
K = 10
K_q = 50
META_BS = 4
N_ITERS = 2000
N_TEST = 200
LR = 5e-3


# ---- recover (a, phi) from support points: a=sqrt(y^2+dy^2), phi=atan2(y,dy)-x ----
def recover_a_phi(xs, ys, dys):
    """xs: (K,1), ys/dys: (K,). Return scalar (a, phi) averaged over support pts.

    phi averaged on the unit circle to avoid wrap-around artefacts."""
    x = xs.squeeze(-1)                       # (K,)
    a = torch.sqrt(ys.pow(2) + dys.pow(2))   # (K,)  amplitude per point
    phi = torch.atan2(ys, dys) - x           # (K,)  phase per point (mod 2pi)
    a_mean = a.mean()
    # circular mean of phase
    phi_mean = torch.atan2(torch.sin(phi).mean(), torch.cos(phi).mean())
    return a_mean, phi_mean


# ---- closed-form predictions of value / grad / hessian at query points ----
def predict_vgh(xq, W, bb, beta):
    """Return (f, f', f'') at query points xq (Kq,1). W:(1,N), bb:(1,N), beta:(N,)."""
    Z = xq @ W + bb                          # (Kq, N)
    sqrtN = math.sqrt(W.shape[1])
    f = (torch.sin(Z) / sqrtN) @ beta                       # value
    g = ((W * torch.cos(Z)) / sqrtN) @ beta                 # 1st deriv
    h = ((-(W.pow(2)) * torch.sin(Z)) / sqrtN) @ beta       # 2nd deriv
    return f, g, h


def train_metaobj(alpha, seed):
    """Meta-train sobolev FastLSQ with outer loss = val_MSE + alpha*grad_MSE."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(1.0), requires_grad=True)
    params = [log_sigma, log_mu, log_gw]
    opt = torch.optim.Adam(params, lr=LR)

    t0 = time.time()
    for it in range(N_ITERS):
        xs, ys, dys, xq, yq = sample_tasks(META_BS, K, K_q, gen, noise=0.0)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        gw = log_gw.exp()
        meta_loss = 0.0
        for t in range(META_BS):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=True, grad_w=gw)
            f, g, _ = predict_vgh(xq[t], W, bb, beta)
            val_mse = (f - yq[t]).pow(2).mean()
            loss_t = val_mse
            if alpha > 0.0:
                a_hat, phi_hat = recover_a_phi(xs[t], ys[t], dys[t])
                g_true = a_hat * torch.cos(xq[t].squeeze(-1) + phi_hat)  # (Kq,)
                grad_mse = (g - g_true).pow(2).mean()
                loss_t = loss_t + alpha * grad_mse
            meta_loss = meta_loss + loss_t
        meta_loss = meta_loss / META_BS
        meta_loss.backward()
        opt.step()
    train_time = time.time() - t0

    return {"Wb": Wb, "bb": bb,
            "sigma": float(log_sigma.exp().item()),
            "mu": float(log_mu.exp().item()),
            "grad_w": float(log_gw.exp().item())}, train_time


def evaluate(meta, seed):
    """Return (val_mse, grad_mse, hess_mse) over N_TEST tasks (clean query targets)."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(N_TEST, K, K_q, gen, noise=0.0)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"]
    vlist, glist, hlist = [], [], []
    with torch.no_grad():
        for t in range(N_TEST):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=True, grad_w=gw)
            f, g, h = predict_vgh(xq[t], W, bb, beta)
            # true a, phi recovered from support (exact in noiseless case)
            a_hat, phi_hat = recover_a_phi(xs[t], ys[t], dys[t])
            xqf = xq[t].squeeze(-1)
            f_true = a_hat * torch.sin(xqf + phi_hat)      # == yq[t]
            g_true = a_hat * torch.cos(xqf + phi_hat)
            h_true = -a_hat * torch.sin(xqf + phi_hat)     # == -yq[t]
            vlist.append((f - yq[t]).pow(2).mean().item())
            glist.append((g - g_true).pow(2).mean().item())
            hlist.append((h - h_true).pow(2).mean().item())
    return (float(torch.tensor(vlist).mean()),
            float(torch.tensor(glist).mean()),
            float(torch.tensor(hlist).mean()))


def main():
    alphas = [0.0, 0.3, 1.0]
    seeds = [0, 1, 2]
    results = {}
    for alpha in alphas:
        per_seed = {"val": [], "grad": [], "hess": [],
                    "sigma": [], "mu": [], "grad_w": [], "train_s": []}
        for seed in seeds:
            meta, t_train = train_metaobj(alpha, seed)
            v, g, h = evaluate(meta, seed)
            per_seed["val"].append(v)
            per_seed["grad"].append(g)
            per_seed["hess"].append(h)
            per_seed["sigma"].append(meta["sigma"])
            per_seed["mu"].append(meta["mu"])
            per_seed["grad_w"].append(meta["grad_w"])
            per_seed["train_s"].append(t_train)
            print(f"alpha={alpha:>4} seed={seed}  "
                  f"val={v:.5f} grad={g:.5f} hess={h:.5f}  "
                  f"sigma={meta['sigma']:.3f} mu={meta['mu']:.2e} gw={meta['grad_w']:.3f} "
                  f"({t_train:.1f}s)")

        def ms(key):
            tt = torch.tensor(per_seed[key])
            return float(tt.mean()), float(tt.std(unbiased=False))

        results[str(alpha)] = {
            "val_mean": ms("val")[0],   "val_std": ms("val")[1],
            "grad_mean": ms("grad")[0], "grad_std": ms("grad")[1],
            "hess_mean": ms("hess")[0], "hess_std": ms("hess")[1],
            "sigma_mean": ms("sigma")[0],
            "mu_mean": ms("mu")[0],
            "grad_w_mean": ms("grad_w")[0],
            "train_s_mean": ms("train_s")[0],
            "per_seed": per_seed,
        }

    print("\n========= value+grad meta-objective ablation (3 seeds, 200 tasks) =========")
    print(f"{'alpha':>6} {'val MSE':>20} {'grad MSE':>20} {'hess MSE':>22} {'grad_w':>8}")
    for alpha in alphas:
        r = results[str(alpha)]
        print(f"{alpha:>6} "
              f"{r['val_mean']:>10.5f}+/-{r['val_std']:<7.5f} "
              f"{r['grad_mean']:>10.5f}+/-{r['grad_std']:<7.5f} "
              f"{r['hess_mean']:>10.4f}+/-{r['hess_std']:<8.4f} "
              f"{r['grad_w_mean']:>8.2f}")

    # relative changes vs alpha=0 baseline
    base = results["0.0"]
    print("\n--- relative change vs alpha=0 (value-only default) ---")
    for alpha in [0.3, 1.0]:
        r = results[str(alpha)]
        dv = 100 * (r["val_mean"] - base["val_mean"]) / base["val_mean"]
        dg = 100 * (r["grad_mean"] - base["grad_mean"]) / base["grad_mean"]
        dh = 100 * (r["hess_mean"] - base["hess_mean"]) / base["hess_mean"]
        print(f"alpha={alpha}:  val {dv:+.1f}%   grad {dg:+.1f}%   hess {dh:+.1f}%")

    with open(_os.path.join(_REPO_ROOT, "results", "metaobj_valgrad_ablation.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    print("\n[saved] results/metaobj_valgrad_ablation.json")


if __name__ == "__main__":
    main()
