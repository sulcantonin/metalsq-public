"""Experiment #4: gradient-noise robustness -- validating the w = sigma_y^2 / sigma_g^2
model from Appendix C.

We meta-train Meta-LSQ-Sobolev on sinusoid tasks where ONLY the support GRADIENT
targets dy are corrupted by additive Gaussian noise of std eps_g. The support
VALUE targets y stay CLEAN, and the query (test) set stays CLEAN so the reported
MSE measures the TRUE error against the noiseless target function.

The existing train_fastlsq_meta(noise=...) corrupts BOTH y and dy, so we cannot
use it. We replicate its structure here (learn log_sigma, log_mu, log_gw via Adam
through a Sobolev inner solve, query value MSE loss), but sample tasks with the
clean sampler and add noise to dys ONLY just before the inner solve.

IMPORTANT autograd note: the existing fastlsq_inner_solve computes the gradient
row weight as math.sqrt(grad_w). When grad_w is a tensor, math.sqrt() returns a
plain Python float and DETACHES it from the autograd graph -- so log_gw never
receives a gradient and grad_w stays frozen at its init (w=1.0). This is a latent
bug in the shipped train_fastlsq_meta path (its saved grad_w is exactly 1.0). To
genuinely test whether the meta-learner drives w down, we use a LOCAL copy of the
inner solve, fastlsq_inner_solve_diff, that is identical except it uses
torch.sqrt(grad_w) so the gradient flows. We do NOT modify the shipped module.

HYPOTHESIS (Appendix C): the optimal gradient weight w in the Sobolev solve is
w = sigma_y^2 / sigma_g^2, where sigma_g^2 is the variance of gradient-target
noise. As eps_g rises, sigma_g^2 rises, so the meta-LEARNED w = exp(log_gw)
should DECREASE (the meta-learner should down-weight the noisy gradient rows).

We record, per eps_g, averaged over 3 seeds:
  (a) learned gradient weight  w = exp(log_gw)
  (b) clean query test MSE.
"""

import os, sys, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch
torch.set_default_dtype(torch.float32)

from experiments.maml_sinusoid import (
    sample_tasks, make_basis, fastlsq_inner_solve, fastlsq_predict,
)


def fastlsq_inner_solve_diff(xs, ys, dys, Wb, bb, sigma, mu, grad_w):
    """Differentiable Sobolev inner solve, IDENTICAL in math to the shipped
    fastlsq_inner_solve(sobolev=True) but using torch.sqrt(grad_w) so that
    autograd can flow into log_gw (the shipped version uses math.sqrt which
    detaches a tensor grad_w to a float). Returns (W, beta)."""
    N = Wb.shape[1]
    W = sigma * Wb                                  # (1, N)
    Z = xs @ W + bb                                 # (K, N)
    H = torch.sin(Z) / math.sqrt(N)
    G = (W * torch.cos(Z)) / math.sqrt(N)           # d=1, (K, N)
    sw = torch.sqrt(grad_w)                         # differentiable in grad_w
    A = torch.cat([H, sw * G], dim=0)
    t = torch.cat([ys, sw * dys], dim=0)
    AtA = A.T @ A + mu * torch.eye(N)
    Atb = A.T @ t
    beta = torch.linalg.solve(AtA, Atb)
    return W, beta

EPS_G_LIST = [0.0, 0.1, 0.25, 0.5]
SEEDS = [0, 1, 2]
N_ITERS = 2000
META_BS = 4
K = 10
K_Q = 10
N = 200
LR = 5e-3
N_TEST = 200
K_TEST = 10
KQ_TEST = 50


def train_gradnoise_meta(eps_g, seed, n_iters=N_ITERS, meta_bs=META_BS,
                         K=K, K_q=K_Q, N=N, lr=LR):
    """Replica of train_fastlsq_meta (sobolev=True) but gradient targets dy are
    corrupted by N(0, eps_g^2) noise BEFORE the inner solve; y and query stay clean.

    Returns dict with learned sigma, mu, grad_w."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    # separate generator so the noise stream is reproducible and independent
    noise_gen = torch.Generator().manual_seed(seed + 7000)
    Wb, bb = make_basis(N=N, seed=seed, dist="normal", nu=3)

    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(1.0), requires_grad=True)  # grad_w init = 1
    params = [log_sigma, log_mu, log_gw]
    opt = torch.optim.Adam(params, lr=lr)

    for it in range(n_iters):
        # CLEAN tasks (noise=0): y_s, dy_s, y_q are all true-function values
        xs, ys, dys, xq, yq = sample_tasks(meta_bs, K, K_q, gen, noise=0.0)
        # corrupt ONLY the support gradient targets
        if eps_g > 0.0:
            dys = dys + eps_g * torch.randn(dys.shape, generator=noise_gen)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        gw = log_gw.exp()
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = fastlsq_inner_solve_diff(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()  # CLEAN query loss
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()

    return {
        "Wb": Wb, "bb": bb,
        "sigma": float(log_sigma.exp().item()),
        "mu": float(log_mu.exp().item()),
        "grad_w": float(log_gw.exp().item()),
        "sobolev": True,
    }


def eval_clean(meta, eps_g, seed, n_test=N_TEST, K=K_TEST, K_q=KQ_TEST):
    """Evaluate on CLEAN query targets. The support GRADIENT is again corrupted by
    the SAME eps_g noise (so this measures deployment under matched noisy-gradient
    conditions); query MSE is against the true function."""
    gen = torch.Generator().manual_seed(seed + 9999)
    noise_gen = torch.Generator().manual_seed(seed + 13000)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen, noise=0.0)
    if eps_g > 0.0:
        dys = dys + eps_g * torch.randn(dys.shape, generator=noise_gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"]
    losses = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve_diff(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu, grad_w=torch.tensor(gw))
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


def main():
    t0 = time.time()
    results = {"eps_g_list": EPS_G_LIST, "seeds": SEEDS,
               "config": {"n_iters": N_ITERS, "meta_bs": META_BS, "K": K,
                          "K_q": K_Q, "N": N, "lr": LR, "n_test": N_TEST,
                          "K_test": K_TEST, "Kq_test": KQ_TEST},
               "per_eps": []}

    for eps_g in EPS_G_LIST:
        w_seeds, mse_seeds, sigma_seeds, mu_seeds = [], [], [], []
        for seed in SEEDS:
            meta = train_gradnoise_meta(eps_g, seed)
            mse = eval_clean(meta, eps_g, seed)
            w_seeds.append(meta["grad_w"])
            mse_seeds.append(mse)
            sigma_seeds.append(meta["sigma"])
            mu_seeds.append(meta["mu"])
            print(f"eps_g={eps_g:.2f} seed={seed} "
                  f"w={meta['grad_w']:.4f} sigma={meta['sigma']:.3f} "
                  f"mu={meta['mu']:.2e} cleanMSE={mse:.4f}", flush=True)

        wt = torch.tensor(w_seeds)
        mt = torch.tensor(mse_seeds)
        entry = {
            "eps_g": eps_g,
            "w_seeds": w_seeds,
            "w_mean": float(wt.mean()), "w_std": float(wt.std(unbiased=False)),
            "mse_seeds": mse_seeds,
            "mse_mean": float(mt.mean()), "mse_std": float(mt.std(unbiased=False)),
            "sigma_seeds": sigma_seeds, "mu_seeds": mu_seeds,
        }
        results["per_eps"].append(entry)
        print(f"==> eps_g={eps_g:.2f}  w_mean={entry['w_mean']:.4f}+-{entry['w_std']:.4f}  "
              f"MSE_mean={entry['mse_mean']:.4f}+-{entry['mse_std']:.4f}", flush=True)

    # Hypothesis check: is w monotonically decreasing in eps_g?
    w_means = [e["w_mean"] for e in results["per_eps"]]
    monotone_dec = all(w_means[i] > w_means[i + 1] for i in range(len(w_means) - 1))
    w0 = w_means[0]
    w_last = w_means[-1]
    results["hypothesis"] = {
        "w_means": w_means,
        "monotone_decreasing": bool(monotone_dec),
        "w_drop_factor_first_to_last": float(w0 / w_last) if w_last > 0 else None,
        "statement": ("w decreases monotonically as eps_g rises"
                      if monotone_dec else
                      "w does NOT decrease monotonically with eps_g"),
    }
    # Appendix-C closed form: w_pred(eps_g) = sigma_y^2 / sigma_g^2.
    # With sigma_g = eps_g, and treating the eps_g=0 learned w as a reference for
    # sigma_y^2 (w0 ~ sigma_y^2 / sigma_g0^2 -> finite because mu/ridge regularizes),
    # we report the ratio w(eps)/w(0) vs the 1/eps_g^2 prediction qualitatively.
    results["total_time_s"] = time.time() - t0

    os.makedirs(_os.path.join(_REPO_ROOT, 'results'), exist_ok=True)
    out = _os.path.join(_REPO_ROOT, 'results', 'grad_noise_robustness.json')
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out}")
    print(f"[hypothesis] {results['hypothesis']['statement']}")
    print(f"[w_means by eps_g] {dict(zip(EPS_G_LIST, [round(w,4) for w in w_means]))}")
    print(f"[total time] {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
