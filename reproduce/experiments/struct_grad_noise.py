"""Experiment #2.3/#3.4: STRUCTURED gradient noise & PARTIAL gradients.

Robustness of the optimal-gradient-weight model w = sigma_y^2 / sigma_g^2
(Appendix C) BEYOND the i.i.d.-Gaussian gradient-noise case tested in
grad_noise_robustness.py. We meta-train Meta-LSQ-Sobolev on sinusoid tasks but
corrupt ONLY the SUPPORT gradient targets dy before the inner solve; the support
VALUES y and the QUERY set stay CLEAN, so the reported MSE is the true error
against the noiseless target function.

Two structured corruptions are tested (3 seeds each):

(A) MAGNITUDE-PROPORTIONAL noise (heteroscedastic, NON-i.i.d.):
      dy <- dy + eps * |dy| * randn,   eps in {0, 0.1, 0.25, 0.5}.
    The gradient-target noise std now scales with the local gradient magnitude
    |dy| (large where the sinusoid is steep, ~0 near turning points). This is a
    structured departure from the constant-std additive noise of the baseline.
    HYPOTHESIS: the meta-learned w should STILL DECREASE with eps -- the w-model
    interpretation (down-weight noisier gradient rows) survives non-i.i.d. noise.

(B) PARTIAL gradients (missing-but-clean rows):
      supply gradient rows at only a fraction p of the K=10 support points
      (drop the other rows from the stacked Sobolev system), p in {0,.25,.5,1}.
      The supplied gradients are CLEAN (no noise). p=0 => value-only solve.
    HYPOTHESIS: with few but TRUSTWORTHY gradients w should stay HIGH (no reason
    to down-weight clean rows), and clean-query MSE should degrade GRACEFULLY as
    p drops (less Sobolev information, but no misinformation).

We reuse the shipped sample_tasks/make_basis/fastlsq_predict read-only. The
shipped fastlsq_inner_solve(sobolev=True) is now autograd-correct (torch.sqrt),
but we keep a LOCAL differentiable inner solve so we can (i) flow grad into
log_gw and (ii) drop a subset of gradient rows for the partial-gradient case.
We never modify any existing file.
"""

import os, sys, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch
torch.set_default_dtype(torch.float32)

from experiments.maml_sinusoid import (
    sample_tasks, make_basis, fastlsq_predict,
)


def fastlsq_inner_solve_diff(xs, ys, dys, Wb, bb, sigma, mu, grad_w, grad_mask=None):
    """Differentiable Sobolev inner solve, IDENTICAL in math to the shipped
    fastlsq_inner_solve(sobolev=True). torch.sqrt(grad_w) keeps autograd flowing
    into log_gw. ``grad_mask`` (bool, shape (K,)) optionally selects which support
    points contribute a gradient row; rows where mask is False are DROPPED from
    the stacked system (partial-gradient case). mask=None keeps all K rows.
    Returns (W, beta)."""
    N = Wb.shape[1]
    W = sigma * Wb                                  # (1, N)
    Z = xs @ W + bb                                 # (K, N)
    H = torch.sin(Z) / math.sqrt(N)                 # value rows (K, N)
    G = (W * torch.cos(Z)) / math.sqrt(N)           # gradient rows (K, N), d=1
    if grad_mask is not None:
        G = G[grad_mask]
        dys_g = dys[grad_mask]
    else:
        dys_g = dys
    sw = torch.sqrt(grad_w)                          # differentiable in grad_w
    if G.shape[0] > 0:
        A = torch.cat([H, sw * G], dim=0)
        t = torch.cat([ys, sw * dys_g], dim=0)
    else:                                            # p=0: value-only solve
        A = H
        t = ys
    AtA = A.T @ A + mu * torch.eye(N)
    Atb = A.T @ t
    beta = torch.linalg.solve(AtA, Atb)
    return W, beta


# ---------------- config ----------------
EPS_LIST = [0.0, 0.1, 0.25, 0.5]      # magnitude-proportional noise levels
P_LIST = [0.0, 0.25, 0.5, 1.0]        # fraction of support points giving a gradient
SEEDS = [0, 1, 2]
N_ITERS = 1200
META_BS = 4
K = 10
K_Q = 10
N = 200
LR = 5e-3
N_TEST = 150
K_TEST = 10
KQ_TEST = 50


def _make_mask(p, K, gen):
    """Bool mask of length K with floor(p*K) True entries (random subset)."""
    n_keep = int(math.floor(p * K + 1e-9))
    mask = torch.zeros(K, dtype=torch.bool)
    if n_keep > 0:
        idx = torch.randperm(K, generator=gen)[:n_keep]
        mask[idx] = True
    return mask


# ---------------- (A) magnitude-proportional noise ----------------

def train_magprop(eps, seed):
    """Meta-train Sobolev with dy <- dy + eps*|dy|*randn (support gradients only)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    noise_gen = torch.Generator().manual_seed(seed + 7000)
    Wb, bb = make_basis(N=N, seed=seed, dist="normal", nu=3)

    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(1.0), requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu, log_gw], lr=LR)

    for it in range(N_ITERS):
        xs, ys, dys, xq, yq = sample_tasks(META_BS, K, K_Q, gen, noise=0.0)
        if eps > 0.0:
            dys = dys + eps * dys.abs() * torch.randn(dys.shape, generator=noise_gen)
        opt.zero_grad()
        sigma, mu, gw = log_sigma.exp(), log_mu.exp(), log_gw.exp()
        meta_loss = 0.0
        for t in range(META_BS):
            W, beta = fastlsq_inner_solve_diff(xs[t], ys[t], dys[t], Wb, bb, sigma, mu, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        (meta_loss / META_BS).backward()
        opt.step()

    return {"Wb": Wb, "bb": bb, "sigma": float(log_sigma.exp()),
            "mu": float(log_mu.exp()), "grad_w": float(log_gw.exp())}


def eval_magprop(meta, eps, seed):
    """Clean-query MSE; support gradients again corrupted by matched eps*|dy| noise."""
    gen = torch.Generator().manual_seed(seed + 9999)
    noise_gen = torch.Generator().manual_seed(seed + 13000)
    xs, ys, dys, xq, yq = sample_tasks(N_TEST, K_TEST, KQ_TEST, gen, noise=0.0)
    if eps > 0.0:
        dys = dys + eps * dys.abs() * torch.randn(dys.shape, generator=noise_gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma, mu = torch.tensor(meta["sigma"]), torch.tensor(meta["mu"])
    gw = torch.tensor(meta["grad_w"])
    losses = []
    with torch.no_grad():
        for t in range(N_TEST):
            W, beta = fastlsq_inner_solve_diff(xs[t], ys[t], dys[t], Wb, bb, sigma, mu, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


# ---------------- (B) partial (clean) gradients ----------------

def train_partial(p, seed):
    """Meta-train Sobolev where only a fraction p of the K support points contribute
    a (clean) gradient row; the rest are dropped from the stacked system."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    mask_gen = torch.Generator().manual_seed(seed + 5000)
    Wb, bb = make_basis(N=N, seed=seed, dist="normal", nu=3)

    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(1.0), requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu, log_gw], lr=LR)

    for it in range(N_ITERS):
        xs, ys, dys, xq, yq = sample_tasks(META_BS, K, K_Q, gen, noise=0.0)
        opt.zero_grad()
        sigma, mu, gw = log_sigma.exp(), log_mu.exp(), log_gw.exp()
        meta_loss = 0.0
        for t in range(META_BS):
            mask = _make_mask(p, K, mask_gen)        # fresh random subset per task
            W, beta = fastlsq_inner_solve_diff(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu, grad_w=gw, grad_mask=mask)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        (meta_loss / META_BS).backward()
        opt.step()

    return {"Wb": Wb, "bb": bb, "sigma": float(log_sigma.exp()),
            "mu": float(log_mu.exp()), "grad_w": float(log_gw.exp())}


def eval_partial(meta, p, seed):
    """Clean-query MSE; at test a matched fraction p of support points give clean gradients."""
    gen = torch.Generator().manual_seed(seed + 9999)
    mask_gen = torch.Generator().manual_seed(seed + 17000)
    xs, ys, dys, xq, yq = sample_tasks(N_TEST, K_TEST, KQ_TEST, gen, noise=0.0)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma, mu = torch.tensor(meta["sigma"]), torch.tensor(meta["mu"])
    gw = torch.tensor(meta["grad_w"])
    losses = []
    with torch.no_grad():
        for t in range(N_TEST):
            mask = _make_mask(p, K_TEST, mask_gen)
            W, beta = fastlsq_inner_solve_diff(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu, grad_w=gw, grad_mask=mask)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    return float(torch.tensor(losses).mean())


def _agg(levels, train_fn, eval_fn, level_key):
    out = []
    for lvl in levels:
        w_s, mse_s, sig_s, mu_s = [], [], [], []
        for seed in SEEDS:
            meta = train_fn(lvl, seed)
            mse = eval_fn(meta, lvl, seed)
            w_s.append(meta["grad_w"]); mse_s.append(mse)
            sig_s.append(meta["sigma"]); mu_s.append(meta["mu"])
            print(f"  {level_key}={lvl:.2f} seed={seed} w={meta['grad_w']:.4f} "
                  f"sigma={meta['sigma']:.3f} mu={meta['mu']:.2e} cleanMSE={mse:.4f}", flush=True)
        wt, mt = torch.tensor(w_s), torch.tensor(mse_s)
        entry = {level_key: lvl, "w_seeds": w_s,
                 "w_mean": float(wt.mean()), "w_std": float(wt.std(unbiased=False)),
                 "mse_seeds": mse_s,
                 "mse_mean": float(mt.mean()), "mse_std": float(mt.std(unbiased=False)),
                 "sigma_seeds": sig_s, "mu_seeds": mu_s}
        out.append(entry)
        print(f"==> {level_key}={lvl:.2f}  w={entry['w_mean']:.4f}+-{entry['w_std']:.4f}  "
              f"MSE={entry['mse_mean']:.4f}+-{entry['mse_std']:.4f}", flush=True)
    return out


def main():
    t0 = time.time()
    cfg = {"n_iters": N_ITERS, "meta_bs": META_BS, "K": K, "K_q": K_Q, "N": N,
           "lr": LR, "n_test": N_TEST, "K_test": K_TEST, "Kq_test": KQ_TEST}
    results = {"seeds": SEEDS, "config": cfg}

    print("=== (A) magnitude-proportional gradient noise:  dy += eps*|dy|*randn ===", flush=True)
    results["magprop"] = {"eps_list": EPS_LIST,
                          "per_level": _agg(EPS_LIST, train_magprop, eval_magprop, "eps")}

    print("\n=== (B) partial (clean) gradients: fraction p of K support pts give a grad row ===", flush=True)
    results["partial"] = {"p_list": P_LIST,
                          "per_level": _agg(P_LIST, train_partial, eval_partial, "p")}

    # hypothesis checks
    wA = [e["w_mean"] for e in results["magprop"]["per_level"]]
    mA = [e["mse_mean"] for e in results["magprop"]["per_level"]]
    A_mono_dec_w = all(wA[i] > wA[i + 1] for i in range(len(wA) - 1))
    results["magprop"]["w_means"] = wA
    results["magprop"]["mse_means"] = mA
    results["magprop"]["w_monotone_decreasing"] = bool(A_mono_dec_w)
    results["magprop"]["w_drop_factor"] = float(wA[0] / wA[-1]) if wA[-1] > 0 else None

    wB = [e["w_mean"] for e in results["partial"]["per_level"]]
    mB = [e["mse_mean"] for e in results["partial"]["per_level"]]
    # for p>0 levels: does w stay high (>1, i.e. gradients trusted)?
    wB_pos = [w for p, w in zip(P_LIST, wB) if p > 0.0]
    results["partial"]["w_means"] = wB
    results["partial"]["mse_means"] = mB
    results["partial"]["w_stays_high_for_p_gt_0"] = bool(all(w > 1.0 for w in wB_pos))
    results["partial"]["mse_monotone_improving_with_p"] = bool(
        all(mB[i] >= mB[i + 1] for i in range(len(mB) - 1)))

    results["total_time_s"] = time.time() - t0

    os.makedirs(_os.path.join(_REPO_ROOT, 'results'), exist_ok=True)
    out = _os.path.join(_REPO_ROOT, 'results', 'struct_grad_noise.json')
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out}")
    print(f"[A] w by eps: {dict(zip(EPS_LIST, [round(w,3) for w in wA]))}  "
          f"monotone_dec={results['magprop']['w_monotone_decreasing']}")
    print(f"[A] MSE by eps: {dict(zip(EPS_LIST, [round(m,4) for m in mA]))}")
    print(f"[B] w by p:   {dict(zip(P_LIST, [round(w,3) for w in wB]))}  "
          f"stays_high={results['partial']['w_stays_high_for_p_gt_0']}")
    print(f"[B] MSE by p: {dict(zip(P_LIST, [round(m,4) for m in mB]))}  "
          f"improves_with_p={results['partial']['mse_monotone_improving_with_p']}")
    print(f"[total time] {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
