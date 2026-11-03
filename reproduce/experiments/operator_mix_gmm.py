"""Experiment #3.1: meta-learn WHICH Sobolev operators to supervise (GMM log-density).

We fit the GMM log-density with a sinusoidal random-feature basis sin(W x + b)/sqrt(N)
and a single stacked ridge least-squares solve, varying WHICH operator rows we stack:

    value row:     phi      = sin(Z) / sqrt(N)               target y   = gmm.log_p
    gradient-k:    d_k phi  = W_k cos(Z) / sqrt(N)           target dy  = gmm.grad_log_p
    Laplacian:     Lap phi  = -(sum_k W_k^2) sin(Z)/sqrt(N)  target lap = tr(gmm.hess_log_p)

These are exactly the FastLSQ cyclic-identity rows (see fastlsq/basis.py:
evaluate / gradient / laplacian).  The Laplacian target is the trace of the
analytic Hessian (sum of the diagonal of gmm.hess_log_p), i.e. 2nd-order supervision.

Three configs are compared, all selecting the bandwidth sigma (and, implicitly,
keeping the operator weights at their config values) on a held-out VALIDATION set:

    value-only            : stack [phi]                       vs [y]
    value+grad            : stack [phi ; w_g grad]            vs [y ; w_g dy]
    value+grad+Laplacian  : stack [phi ; w_g grad ; w_l lap]  vs [y ; w_g dy ; w_l lap_t]

w_g, w_l are searched on a tiny grid (incl. {0,1}) on the validation set.
We then report rRMSE of value / grad / FULL Hessian on a held-out TEST set, at
d in {4, 8}, N=1500, averaged over 3 seeds.

Question: does adding the 2nd-order (Laplacian) row reduce Hessian (and value)
error BEYOND value+grad?  Report the table and state honestly whether it helps.

Float64 throughout (GMM convention).  Single thread.
"""

import sys, os, time, math, json
import torch

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# Resolve FastLSQ (env or sibling) exactly as the existing modules do.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FASTLSQ = os.environ.get(
    "FASTLSQ_PATH", os.path.join(os.path.dirname(os.path.dirname(_HERE)), "FastLSQ"))
if _FASTLSQ not in sys.path:
    sys.path.insert(0, _FASTLSQ)

from fastlsq.basis import SinusoidalBasis
from fastlsq.linalg import solve_lstsq
# Read-only reuse of the GMM target (analytic value/grad/Hessian) from the existing module.
from experiments.derivative_oracle_test import GMM, relrmse

torch.set_default_dtype(torch.float64)

# Bandwidth grid (matches the existing GMM fitters in sobolev_test.py / derivative_oracle_test.py).
SCALES = (0.1, 0.25, 0.5, 1.0, 2.0)
# Operator-row weight grid for the supervised rows. Per the task spec this is the
# small {0,1} on/off grid (plus a stronger 2.0 weight to let 2nd-order rows assert
# themselves if they help). 0 means the row is dropped from the stack.
W_GRID = (1.0, 2.0)
MU = 1e-8
N = 1500
N_BLOCKS = 3
N_TRAIN = 2500
N_TEST = 300
N_VAL = 300
SEEDS = (0, 1, 2)
DIMS = (4, 8)


def build_basis(d, scale, seed):
    """Three-block isotropic sinusoidal basis at a given bandwidth scale."""
    torch.manual_seed(seed)
    per_block = N // N_BLOCKS
    Ws, bs = [], []
    for _ in range(N_BLOCKS):
        W = torch.randn(d, per_block) * scale
        b = torch.rand(1, per_block) * 2 * math.pi
        Ws.append(W); bs.append(b)
    W = torch.cat(Ws, dim=1)
    b = torch.cat(bs, dim=1)
    return SinusoidalBasis(W, b, normalize=True)


def fit_stacked(basis, X_tr, y_tr, dy_tr, lap_tr, w_g, w_l):
    """One stacked ridge LSQ for [value ; w_g grad ; w_l laplacian] vs targets.

    w_g == 0 -> no gradient rows; w_l == 0 -> no Laplacian row.
    Returns beta (N,).
    """
    cache = basis.cache(X_tr)
    H = basis.evaluate(X_tr, cache=cache)                 # (M, N)
    A_rows = [H]
    t_rows = [y_tr]
    if w_g > 0:
        G = basis.gradient(X_tr, cache=cache)             # (M, d, N)
        G_stack = G.reshape(-1, G.shape[-1])              # (M*d, N)
        dy_stack = dy_tr.reshape(-1)                       # (M*d,)
        A_rows.append(w_g * G_stack)
        t_rows.append(w_g * dy_stack)
    if w_l > 0:
        L = basis.laplacian(X_tr, cache=cache)            # (M, N)
        A_rows.append(w_l * L)
        t_rows.append(w_l * lap_tr)
    A = torch.cat(A_rows, dim=0)
    t = torch.cat(t_rows, dim=0)
    beta = solve_lstsq(A, t.unsqueeze(-1), mu=MU).squeeze(-1)
    return beta


def full_hessian(basis, beta, X):
    """Assemble the full predicted Hessian (M, d, d) via the cyclic identity.

    Diagonal: -W_k^2 sin(Z) * beta ; off-diagonal: -W_i W_j sin(Z) * beta.
    """
    cache = basis.cache(X)
    M, d = X.shape
    Hp = torch.zeros(M, d, d)
    diag = torch.einsum("idn,n->id", basis.hessian_diag(X, cache=cache), beta)
    for i in range(d):
        Hp[:, i, i] = diag[:, i]
    for i in range(d):
        for j in range(i + 1, d):
            alpha = [0] * d
            alpha[i] = 1; alpha[j] = 1
            v = basis.derivative(X, tuple(alpha), cache=cache) @ beta
            Hp[:, i, j] = v
            Hp[:, j, i] = v
    return Hp


def predict_all(basis, beta, X):
    cache = basis.cache(X)
    val = basis.evaluate(X, cache=cache) @ beta
    grad = torch.einsum("idn,n->id", basis.gradient(X, cache=cache), beta)
    Hp = full_hessian(basis, beta, X)
    return val, grad, Hp


def val_score(basis, beta, X_val, y_val, dy_val, lap_val, use_grad, use_lap):
    """Validation objective used for sigma / weight selection.

    Sum of the relevant supervised-residual MSEs, each normalized by target power
    so value/grad/Laplacian are comparable.  Only the rows the config actually
    supervises contribute (so configs are scored on what they claim to fit).
    """
    cache = basis.cache(X_val)
    vp = basis.evaluate(X_val, cache=cache) @ beta
    s = (vp - y_val).pow(2).mean().item() / (y_val.pow(2).mean().item() + 1e-12)
    if use_grad:
        gp = torch.einsum("idn,n->id", basis.gradient(X_val, cache=cache), beta)
        s += (gp - dy_val).pow(2).mean().item() / (dy_val.pow(2).mean().item() + 1e-12)
    if use_lap:
        lp = basis.laplacian(X_val, cache=cache) @ beta
        s += (lp - lap_val).pow(2).mean().item() / (lap_val.pow(2).mean().item() + 1e-12)
    return s


def select_and_eval(config, d, seed):
    """For one config, select (sigma, w_g, w_l) on validation, eval on test."""
    gmm = GMM(d, K=3, seed=seed)
    X_tr = gmm.sample(N_TRAIN, seed=seed)
    y_tr = gmm.log_p(X_tr)
    dy_tr = gmm.grad_log_p(X_tr)
    lap_tr = torch.diagonal(gmm.hess_log_p(X_tr), dim1=1, dim2=2).sum(-1)  # trace = Laplacian target

    X_val = gmm.sample(N_VAL, seed=seed + 100)
    y_val = gmm.log_p(X_val)
    dy_val = gmm.grad_log_p(X_val)
    lap_val = torch.diagonal(gmm.hess_log_p(X_val), dim1=1, dim2=2).sum(-1)

    X_te = gmm.sample(N_TEST, seed=seed + 200)
    y_te = gmm.log_p(X_te)
    g_te = gmm.grad_log_p(X_te)
    H_te = gmm.hess_log_p(X_te)

    # Candidate weight pairs per config.
    if config == "value":
        wg_opts, wl_opts = (0.0,), (0.0,)
        use_grad, use_lap = False, False
    elif config == "value+grad":
        wg_opts, wl_opts = tuple(w for w in W_GRID if w > 0), (0.0,)
        use_grad, use_lap = True, False
    elif config == "value+grad+lap":
        wg_opts = tuple(w for w in W_GRID if w > 0)
        wl_opts = tuple(w for w in W_GRID if w > 0)
        use_grad, use_lap = True, True
    else:
        raise ValueError(config)

    best = None  # (score, sigma, w_g, w_l, basis, beta)
    for scale in SCALES:
        basis = build_basis(d, scale, seed)
        for w_g in wg_opts:
            for w_l in wl_opts:
                beta = fit_stacked(basis, X_tr, y_tr, dy_tr, lap_tr, w_g, w_l)
                sc = val_score(basis, beta, X_val, y_val, dy_val, lap_val,
                               use_grad, use_lap)
                if best is None or sc < best[0]:
                    best = (sc, scale, w_g, w_l, basis, beta)

    _, scale, w_g, w_l, basis, beta = best
    vp, gp, Hp = predict_all(basis, beta, X_te)
    return {
        "val_rrmse": relrmse(vp, y_te),
        "grad_rrmse": relrmse(gp, g_te),
        "hess_rrmse": relrmse(Hp, H_te),
        "sigma": scale, "w_g": w_g, "w_l": w_l,
    }


def mean_std(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(v)


def main():
    configs = ["value", "value+grad", "value+grad+lap"]
    out = {"meta": {"N": N, "n_blocks": N_BLOCKS, "n_train": N_TRAIN,
                    "n_val": N_VAL, "n_test": N_TEST, "seeds": list(SEEDS),
                    "scales": list(SCALES), "w_grid": list(W_GRID), "mu": MU},
           "results": {}}
    t0 = time.time()
    for d in DIMS:
        out["results"][f"d{d}"] = {}
        for config in configs:
            tc = time.time()
            v_list, g_list, h_list, sig_list, wg_list, wl_list = [], [], [], [], [], []
            for seed in SEEDS:
                r = select_and_eval(config, d, seed)
                v_list.append(r["val_rrmse"]); g_list.append(r["grad_rrmse"])
                h_list.append(r["hess_rrmse"]); sig_list.append(r["sigma"])
                wg_list.append(r["w_g"]); wl_list.append(r["w_l"])
            vm, vs = mean_std(v_list); gm, gs = mean_std(g_list); hm, hs = mean_std(h_list)
            out["results"][f"d{d}"][config] = {
                "val_mean": vm, "val_std": vs,
                "grad_mean": gm, "grad_std": gs,
                "hess_mean": hm, "hess_std": hs,
                "val_per_seed": v_list, "grad_per_seed": g_list, "hess_per_seed": h_list,
                "sigma_per_seed": sig_list, "w_g_per_seed": wg_list, "w_l_per_seed": wl_list,
            }
            print(f"d={d:>2} {config:<16} "
                  f"val={vm:.3e}+/-{vs:.1e}  grad={gm:.3e}+/-{gs:.1e}  "
                  f"hess={hm:.3e}+/-{hs:.1e}  (sig={sig_list}, wg={wg_list}, wl={wl_list}) "
                  f"[{time.time()-tc:.0f}s]", flush=True)
    out["meta"]["wall_clock_s"] = time.time() - t0

    os.makedirs(_os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    path = _os.path.join(_REPO_ROOT, "results", "operator_mix_gmm.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {path}   ({out['meta']['wall_clock_s']:.1f}s)")
    return out


if __name__ == "__main__":
    main()
