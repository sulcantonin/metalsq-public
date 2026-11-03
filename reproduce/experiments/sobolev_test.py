"""Sobolev fitting: match values AND gradients during training.

For FastLSQ:  stack [H ; G_1 ; ... ; G_d] vs [y ; dy_1 ; ... ; dy_d]; one LSQ.
For MLP:      add gradient-MSE term to the loss; double-backprop SGD.
"""

import sys, os, time, math
import torch
import torch.nn as nn

# Resolve the FastLSQ dependency portably: $FASTLSQ_PATH if set, else a sibling
# ../FastLSQ checkout next to this repo. (See requirements.txt.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_FASTLSQ = os.environ.get(
    "FASTLSQ_PATH", os.path.join(os.path.dirname(os.path.dirname(_HERE)), "FastLSQ"))
for _p in (_FASTLSQ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from fastlsq.basis import SinusoidalBasis
from fastlsq.linalg import solve_lstsq
from derivative_oracle_test import GMM, MLP, mlp_eval, fastlsq_eval, relrmse

torch.set_default_dtype(torch.float32)  # override the float64 set inside derivative_oracle_test
SEED = 0


def fit_fastlsq_sobolev(gmm, X_tr, y_tr, dy_tr, N=2000, n_blocks=3,
                        scales=(0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 3.0), mu=1e-8,
                        use_grad=True, grad_weight=1.0, X_val=None, y_val=None,
                        seed=SEED):
    """Fit value + gradient targets with one stacked least-squares solve.

    Bandwidth ``scale`` is selected on a genuine held-out validation set
    (X_val, y_val) when supplied; with N>=M features the in-sample training
    residual is near-zero for every scale and would simply pick the most
    overfit bandwidth, so out-of-sample selection is necessary."""
    best = None
    for scale in scales:
        torch.manual_seed(seed)
        per_block = N // n_blocks
        Ws, bs = [], []
        for _ in range(n_blocks):
            W = torch.randn(gmm.d, per_block) * scale
            b = torch.rand(1, per_block) * 2 * math.pi
            Ws.append(W); bs.append(b)
        W = torch.cat(Ws, dim=1); b = torch.cat(bs, dim=1)
        basis = SinusoidalBasis(W, b, normalize=True)

        cache = basis.cache(X_tr)
        H = basis.evaluate(X_tr, cache=cache)                 # (M, N)
        if use_grad:
            G = basis.gradient(X_tr, cache=cache)             # (M, d, N)
            # Stack value rows and gradient rows. Weight gradient rows by sqrt(grad_weight).
            d = gmm.d
            G_stack = G.reshape(-1, G.shape[-1])              # (M*d, N)
            dy_stack = dy_tr.reshape(-1)                       # (M*d,)
            A = torch.cat([H, math.sqrt(grad_weight) * G_stack], dim=0)
            t = torch.cat([y_tr, math.sqrt(grad_weight) * dy_stack], dim=0)
        else:
            A = H; t = y_tr

        beta = solve_lstsq(A, t.unsqueeze(-1), mu=mu).squeeze(-1)
        # Model selection on a held-out validation residual (falls back to the
        # in-sample residual only if no validation set is supplied).
        if X_val is not None:
            H_val = basis.evaluate(X_val)
            score = (H_val @ beta - y_val).pow(2).mean().item()
        else:
            score = (H @ beta - y_tr).pow(2).mean().item()
        if best is None or score < best[0]:
            best = (score, basis, beta, scale)
    _, basis, beta, scale = best
    return basis, beta, scale


def fit_mlp_sobolev(d, X_tr, y_tr, dy_tr, hidden=96, depth=3, epochs=2500,
                    lr=1e-3, grad_weight=1.0):
    torch.manual_seed(SEED)
    net = MLP(d, hidden=hidden, depth=depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = min(2048, X_tr.shape[0])
    for ep in range(epochs):
        idx = torch.randperm(X_tr.shape[0])[:bs]
        x = X_tr[idx].detach().requires_grad_(True)
        v = net(x)
        # value loss
        loss_v = (v - y_tr[idx]).pow(2).mean()
        # gradient loss (requires double-backprop)
        if grad_weight > 0:
            g = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
            loss_g = (g - dy_tr[idx]).pow(2).mean()
            loss = loss_v + grad_weight * loss_g
        else:
            loss = loss_v
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def run(d=8, n_train=2000, n_test=400, N=1500, grad_weight=1.0):
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

    rows = []

    for label, gw in (("vanilla (val only)", 0.0), ("Sobolev (val+grad)", grad_weight)):
        # FastLSQ
        t = time.time()
        basis, beta, scale = fit_fastlsq_sobolev(
            gmm, X_tr, y_tr, dy_tr, N=N, use_grad=(gw > 0), grad_weight=max(gw, 1e-8),
            X_val=X_val, y_val=y_val)
        t_fl = time.time() - t
        fl_v, fl_g, fl_H = fastlsq_eval(basis, beta, X_te)
        rows.append((f"FastLSQ {label}", t_fl,
                     relrmse(fl_v, y_te), relrmse(fl_g, g_te), relrmse(fl_H, H_te), scale))

        # MLP
        t = time.time()
        net = fit_mlp_sobolev(d, X_tr, y_tr, dy_tr, grad_weight=gw)
        t_mlp = time.time() - t
        mlp_v, mlp_g, mlp_H = mlp_eval(net, X_te)
        rows.append((f"MLP     {label}", t_mlp,
                     relrmse(mlp_v, y_te), relrmse(mlp_g, g_te), relrmse(mlp_H, H_te), None))

    print(f"{'method':<32} {'fit s':>7} | {'val':>9} {'grad':>9} {'hess':>9} | scale")
    for name, t, v, g, H, s in rows:
        s_str = f"{s}" if s is not None else "-"
        print(f"{name:<32} {t:>7.2f} | {v:>9.2e} {g:>9.2e} {H:>9.2e} | {s_str}")

    # Sobolev-vs-vanilla wall-clock ratios for the paper's quantified-cost claim.
    # rows are ordered [FastLSQ vanilla, MLP vanilla, FastLSQ Sob, MLP Sob]
    try:
        t_fl_v   = rows[0][1]; t_mlp_v  = rows[1][1]
        t_fl_s   = rows[2][1]; t_mlp_s  = rows[3][1]
        print(f"  Wall-clock ratio (Sobolev / vanilla):  "
              f"FastLSQ {t_fl_s/t_fl_v:.2f}x   MLP {t_mlp_s/t_mlp_v:.2f}x")
    except Exception:
        pass

    return rows


def multiseed_fastlsq(d, seeds=(0, 1, 2, 3, 4), n_train=2000, n_test=400, N=1500,
                      grad_weight=1.0):
    """The FastLSQ half of `run` over several seeds. A seed redraws the mixture, the
    train/validation/test samples and the features. With one seed the d=4 ranking
    was set by a single test point far from the training data (|x| = 6, nearest
    training point 1.86 away against a median of 0.40), so per-seed means are
    reported alongside medians."""
    rows = []
    for sd in seeds:
        gmm = GMM(d, K=3, seed=sd)
        X_tr = gmm.sample(n_train, seed=sd)
        y_tr, dy_tr = gmm.log_p(X_tr), gmm.grad_log_p(X_tr)
        X_val = gmm.sample(n_test, seed=sd + 2); y_val = gmm.log_p(X_val)
        X_te = gmm.sample(n_test, seed=sd + 1)
        y_te, g_te, H_te = gmm.log_p(X_te), gmm.grad_log_p(X_te), gmm.hess_log_p(X_te)
        row = {"seed": sd}
        for key, gw in (("fastlsq_vanilla", 0.0), ("fastlsq_sobolev", grad_weight)):
            basis, beta, scale = fit_fastlsq_sobolev(
                gmm, X_tr, y_tr, dy_tr, N=N, use_grad=(gw > 0), grad_weight=max(gw, 1e-8),
                X_val=X_val, y_val=y_val, seed=sd)
            v, g, H = fastlsq_eval(basis, beta, X_te)
            row[key] = {"val": relrmse(v, y_te), "grad": relrmse(g, g_te),
                        "hess": relrmse(H, H_te), "sigma": scale}
        print(f"d={d} seed {sd}: " + " | ".join(
            f"{k[8:]} val {r['val']:.2e} grad {r['grad']:.2e} hess {r['hess']:.2e} sigma {r['sigma']}"
            for k, r in row.items() if k != "seed"), flush=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    import sys
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    if "--multiseed" in sys.argv:
        import json
        out = {"d4": multiseed_fastlsq(4), "d8": multiseed_fastlsq(8)}
        if save:
            with open(save, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[saved] {save}")
        sys.exit(0)
    rows4 = run(d=4)
    rows8 = run(d=8)
    if save:
        import json
        keys = ["fastlsq_vanilla", "mlp_vanilla", "fastlsq_sobolev", "mlp_sobolev"]
        def pack(rows):
            out = {}
            for k, r in zip(keys, rows):
                _name, t, v, g, H, s = r
                out[k] = {"fit_s": t, "val": v, "grad": g, "hess": H, "sigma": s}
            return out
        with open(save, "w") as f:
            json.dump({"d4": pack(rows4), "d8": pack(rows8)}, f, indent=2)
        print(f"\n[saved] {save}")
