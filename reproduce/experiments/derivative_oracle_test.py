"""Sanity experiment: FastLSQ vs MLP as a derivative oracle in d=8/16/32.

Target: log-density of a Gaussian mixture (analytic value/grad/Hessian).
Question: at matched value error, how do gradient and Hessian errors compare?
"""

import sys, os, time, math
import torch
import torch.nn as nn
import numpy as np

# Resolve the FastLSQ dependency portably: $FASTLSQ_PATH if set, else a sibling
# ../FastLSQ checkout next to this repo. (See requirements.txt.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_FASTLSQ = os.environ.get(
    "FASTLSQ_PATH", os.path.join(os.path.dirname(os.path.dirname(_HERE)), "FastLSQ"))
if _FASTLSQ not in sys.path:
    sys.path.insert(0, _FASTLSQ)
from fastlsq.basis import SinusoidalBasis
from fastlsq.linalg import solve_lstsq

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")
SEED = 0


# ---------- target: GMM log-density with analytic derivatives ----------

class GMM:
    """Isotropic Gaussian mixture with unit covariance."""
    def __init__(self, d, K=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.d = d
        self.K = K
        # Means on a small sphere so f stays interesting but bounded.
        mus = torch.randn(K, d, generator=g)
        mus = mus / mus.norm(dim=1, keepdim=True) * 1.5
        self.mus = mus
        self.log_pis = torch.full((K,), -math.log(K))

    def _log_components(self, x):
        # x: (M, d) -> (M, K)
        diff = x.unsqueeze(1) - self.mus.unsqueeze(0)         # (M, K, d)
        log_n = -0.5 * (diff * diff).sum(-1) - 0.5 * self.d * math.log(2 * math.pi)
        return log_n + self.log_pis.unsqueeze(0)

    def log_p(self, x):
        return torch.logsumexp(self._log_components(x), dim=1)

    def responsibilities(self, x):
        lc = self._log_components(x)
        return torch.softmax(lc, dim=1)                        # (M, K)

    def grad_log_p(self, x):
        r = self.responsibilities(x)
        diff = self.mus.unsqueeze(0) - x.unsqueeze(1)          # (M, K, d) = (mu - x)
        return (r.unsqueeze(-1) * diff).sum(dim=1)             # (M, d)

    def hess_log_p(self, x):
        r = self.responsibilities(x)                            # (M, K)
        diff = self.mus.unsqueeze(0) - x.unsqueeze(1)           # (M, K, d)
        # E_r[(mu-x)(mu-x)^T]
        outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)         # (M, K, d, d)
        cov_term = (r.unsqueeze(-1).unsqueeze(-1) * outer).sum(dim=1)  # (M, d, d)
        I = torch.eye(self.d).expand_as(cov_term)
        g = self.grad_log_p(x)                                  # (M, d)
        gg = g.unsqueeze(-1) * g.unsqueeze(-2)
        return cov_term - I - gg

    def sample(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        comp = torch.randint(0, self.K, (n,), generator=g)
        eps = torch.randn(n, self.d, generator=g)
        return self.mus[comp] + eps


# ---------- FastLSQ fit ----------

def fit_fastlsq(gmm, X_tr, y_tr, N=1500, n_blocks=3,
                scales=(0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 3.0),
                mu=1e-8, X_val=None, y_val=None):
    """Fit y_tr ~ sum beta_j sin(W_j^T x + b_j) via ridge LSQ.

    Bandwidth selected on a held-out validation set (X_val, y_val) when supplied;
    with N>=M features the in-sample residual is near-zero for every scale, so
    out-of-sample selection is needed to avoid picking the most overfit bandwidth."""
    best = None
    for scale in scales:
        torch.manual_seed(SEED)
        per_block = N // n_blocks
        Ws, bs = [], []
        for _ in range(n_blocks):
            W = torch.randn(gmm.d, per_block) * scale
            b = torch.rand(1, per_block) * 2 * math.pi
            Ws.append(W); bs.append(b)
        W = torch.cat(Ws, dim=1)
        b = torch.cat(bs, dim=1)
        basis = SinusoidalBasis(W, b, normalize=True)
        H = basis.evaluate(X_tr)                # (M, N)
        beta = solve_lstsq(H, y_tr.unsqueeze(-1), mu=mu).squeeze(-1)
        if X_val is not None:
            score = (basis.evaluate(X_val) @ beta - y_val).pow(2).mean().item()
        else:
            score = (H @ beta - y_tr).pow(2).mean().item()
        if best is None or score < best[0]:
            best = (score, basis, beta, scale)
    _, basis, beta, scale = best
    return basis, beta, scale


def fastlsq_eval(basis, beta, X):
    cache = basis.cache(X)
    val = basis.evaluate(X, cache=cache) @ beta
    grad = torch.einsum("idn,n->id", basis.gradient(X, cache=cache), beta)
    # Full Hessian: assemble from per-pair second derivatives.
    M, d = X.shape
    H = torch.zeros(M, d, d)
    # Diagonal: -W_k^2 sin(Z) * beta
    diag = torch.einsum("idn,n->id", basis.hessian_diag(X, cache=cache), beta)
    for i in range(d):
        H[:, i, i] = diag[:, i]
    # Off-diagonal via derivative(): D^{e_i + e_j} sin = -W_i W_j sin(Z)
    for i in range(d):
        for j in range(i + 1, d):
            alpha = [0] * d
            alpha[i] = 1; alpha[j] = 1
            phi_ij = basis.derivative(X, tuple(alpha), cache=cache)
            v = phi_ij @ beta
            H[:, i, j] = v
            H[:, j, i] = v
    return val, grad, H


# ---------- MLP baseline ----------

class MLP(nn.Module):
    def __init__(self, d, hidden=128, depth=3, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(d, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(d, X_tr, y_tr, hidden=128, depth=3, epochs=4000, lr=1e-3):
    torch.manual_seed(SEED)
    net = MLP(d, hidden=hidden, depth=depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = min(2048, X_tr.shape[0])
    losses = []
    for ep in range(epochs):
        idx = torch.randperm(X_tr.shape[0])[:bs]
        pred = net(X_tr[idx])
        loss = (pred - y_tr[idx]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 500 == 0:
            losses.append(loss.item())
    return net


def mlp_eval(net, X):
    X = X.detach().requires_grad_(True)
    val = net(X)
    grad = torch.autograd.grad(val.sum(), X, create_graph=True)[0]
    M, d = X.shape
    H = torch.zeros(M, d, d)
    for i in range(d):
        gi = torch.autograd.grad(grad[:, i].sum(), X, retain_graph=True, create_graph=False)[0]
        H[:, i, :] = gi
    H = 0.5 * (H + H.transpose(1, 2))
    return val.detach(), grad.detach(), H.detach()


# ---------- driver ----------

def rmse(a, b):
    return float((a - b).pow(2).mean().sqrt())


def relrmse(a, b):
    return rmse(a, b) / float(b.pow(2).mean().sqrt() + 1e-12)


def run_one(d, n_train=4000, n_test=500, N=1500):
    print(f"\n=== d = {d} ===")
    gmm = GMM(d, K=3, seed=SEED)
    X_tr = gmm.sample(n_train, seed=SEED)
    y_tr = gmm.log_p(X_tr)
    X_val = gmm.sample(n_test, seed=SEED + 2)   # held-out set for bandwidth selection
    y_val = gmm.log_p(X_val)
    X_te = gmm.sample(n_test, seed=SEED + 1)
    y_te = gmm.log_p(X_te)
    g_te = gmm.grad_log_p(X_te)
    H_te = gmm.hess_log_p(X_te)

    # FastLSQ
    t = time.time()
    basis, beta, scale = fit_fastlsq(gmm, X_tr, y_tr, N=N, X_val=X_val, y_val=y_val)
    t_fl_fit = time.time() - t
    t = time.time()
    fl_val, fl_grad, fl_H = fastlsq_eval(basis, beta, X_te)
    t_fl_eval = time.time() - t

    # MLP
    t = time.time()
    net = fit_mlp(d, X_tr, y_tr)
    t_mlp_fit = time.time() - t
    t = time.time()
    mlp_val, mlp_grad, mlp_H = mlp_eval(net, X_te)
    t_mlp_eval = time.time() - t

    print(f"  scale chosen: {scale}")
    print(f"  fit time:  FastLSQ {t_fl_fit:.2f}s   MLP {t_mlp_fit:.2f}s")
    print(f"  eval time: FastLSQ {t_fl_eval:.2f}s   MLP {t_mlp_eval:.2f}s")
    print(f"  value RMSE     FastLSQ={relrmse(fl_val, y_te):.3e}   MLP={relrmse(mlp_val, y_te):.3e}")
    print(f"  grad  RMSE     FastLSQ={relrmse(fl_grad, g_te):.3e}   MLP={relrmse(mlp_grad, g_te):.3e}")
    print(f"  Hess  RMSE     FastLSQ={relrmse(fl_H, H_te):.3e}   MLP={relrmse(mlp_H, H_te):.3e}")

    return {
        "d": d,
        "fl": (relrmse(fl_val, y_te), relrmse(fl_grad, g_te), relrmse(fl_H, H_te)),
        "mlp": (relrmse(mlp_val, y_te), relrmse(mlp_grad, g_te), relrmse(mlp_H, H_te)),
    }


if __name__ == "__main__":
    import sys
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    results = []
    for d in (2, 4, 8):
        results.append(run_one(d, N=2000))
    print("\n=== Summary (relative RMSE; ratios show derivative degradation vs value) ===")
    hdr = f"{'d':>3} | {'val':>9} {'grad':>9} {'hess':>9} {'g/v':>6} {'h/v':>6}"
    print("FastLSQ:" + " " * 4 + hdr)
    for r in results:
        v, g, h = r['fl']
        print(f"        {r['d']:>3} | {v:>9.2e} {g:>9.2e} {h:>9.2e} {g/v:>6.1f} {h/v:>6.1f}")
    print("MLP:    " + " " * 4 + hdr)
    for r in results:
        v, g, h = r['mlp']
        print(f"        {r['d']:>3} | {v:>9.2e} {g:>9.2e} {h:>9.2e} {g/v:>6.1f} {h/v:>6.1f}")

    if save:
        import json
        packed = []
        for r in results:
            fv, fg, fh = r['fl']
            mv, mg, mh = r['mlp']
            packed.append({"d": r["d"],
                           "fl": {"val": fv, "grad": fg, "hess": fh},
                           "mlp": {"val": mv, "grad": mg, "hess": mh}})
        with open(save, "w") as f:
            json.dump(packed, f, indent=2)
        print(f"\n[saved] {save}")
