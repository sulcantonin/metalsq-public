"""MAML on sinusoid regression: MLP-MAML vs FastLSQ-Meta vs FastLSQ-Sobolev-Meta.

Task family:  y = a sin(x + phi),  a ~ U[0.1, 5], phi ~ U[0, pi], x in [-5, 5].
K-shot adaptation: fit a model from K (x, y) pairs (and optionally dy/dx),
                   evaluate on K_q query points.

Three methods:
  MLP-MAML:               classic MAML, 1 inner SGD step on a small MLP.
  FastLSQ-Meta:           closed-form ridge solve on RFFs, meta-learn (sigma, mu).
  FastLSQ-Sobolev-Meta:   same, but inner solve also matches dy/dx.

Compares: final 5-shot test MSE, total meta-training time, per-task inference time.
"""

import math, time
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)
SEED = 0
device = torch.device("cpu")


# ---------------- task generator ----------------

def sample_tasks(n_tasks, K, K_q, gen, noise=0.0, grad_noise=None):
    """Return (x_s, y_s, dy_s, x_q, y_q) for n_tasks sinusoid tasks.

    ``noise``: std of Gaussian observation noise added to the SUPPORT values y_s
    (and support gradients dy_s) only; the query targets stay clean so the
    reported MSE measures error against the true function. noise=0 is the
    original noiseless benchmark.

    ``grad_noise``: std of the noise on the support gradients dy_s. None (default)
    means "same as ``noise``", i.e. the original behaviour, bit-for-bit. Setting
    grad_noise=0.0 with noise>0 gives value-only noise: the gradient noise draw is
    still consumed from ``gen`` (so tasks and y_s noise are identical to the
    grad_noise=None stream) but is not added, leaving dy_s clean."""
    a = torch.rand(n_tasks, 1, generator=gen) * 4.9 + 0.1   # [0.1, 5]
    phi = torch.rand(n_tasks, 1, generator=gen) * math.pi   # [0, pi]
    xs = torch.rand(n_tasks, K, generator=gen) * 10 - 5
    xq = torch.rand(n_tasks, K_q, generator=gen) * 10 - 5
    ys = a * torch.sin(xs + phi)
    yq = a * torch.sin(xq + phi)
    dys = a * torch.cos(xs + phi)
    if grad_noise is None:
        if noise > 0.0:
            ys = ys + noise * torch.randn(ys.shape, generator=gen)
            dys = dys + noise * torch.randn(dys.shape, generator=gen)
    elif noise > 0.0 or grad_noise > 0.0:
        ys = ys + noise * torch.randn(ys.shape, generator=gen)
        eps_g = torch.randn(dys.shape, generator=gen)
        if grad_noise > 0.0:
            dys = dys + grad_noise * eps_g
    return xs.unsqueeze(-1), ys, dys, xq.unsqueeze(-1), yq


# ---------------- MLP-MAML ----------------

class MAMLNet(nn.Module):
    """MLP with configurable depth and hidden width. depth = number of Linear layers
    (so depth=3 gives input -> hidden -> hidden -> output, two hidden layers)."""
    def __init__(self, hidden=40, depth=3, in_dim=1, out_dim=1):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, out_dim))

    def forward(self, x, params=None):
        if params is None:
            h = x
            for i, layer in enumerate(self.layers[:-1]):
                h = torch.relu(layer(h))
            return self.layers[-1](h).squeeze(-1)
        h = x
        for i in range(self.depth - 1):
            h = torch.relu(torch.nn.functional.linear(h, params[f"w{i+1}"], params[f"b{i+1}"]))
        return torch.nn.functional.linear(h, params[f"w{self.depth}"], params[f"b{self.depth}"]).squeeze(-1)

    def init_params(self):
        return {
            **{f"w{i+1}": layer.weight for i, layer in enumerate(self.layers)},
            **{f"b{i+1}": layer.bias for i, layer in enumerate(self.layers)},
        }


def maml_inner_step(net, params, xs, ys, lr, first_order=False):
    pred = net(xs, params)
    loss = (pred - ys).pow(2).mean()
    grads = torch.autograd.grad(loss, list(params.values()), create_graph=not first_order)
    return {k: v - lr * g for (k, v), g in zip(params.items(), grads)}


def train_mlp_maml(n_iters=8000, meta_bs=4, K=10, K_q=10, inner_lr=0.01,
                   inner_steps=1, outer_lr=1e-3, hidden=40, depth=3,
                   first_order=False, seed=SEED, log_every=500, clip_grad=None,
                   noise=0.0, grad_noise=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    net = MAMLNet(hidden=hidden, depth=depth).to(device)
    meta_opt = torch.optim.Adam(net.parameters(), lr=outer_lr)
    curve = []  # (wallclock, iter, meta_loss)
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, _, xq, yq = sample_tasks(meta_bs, K, K_q, gen, noise=noise, grad_noise=grad_noise)
        meta_opt.zero_grad()
        meta_loss = 0.0
        for t in range(meta_bs):
            params = net.init_params()
            for _ in range(inner_steps):
                params = maml_inner_step(net, params, xs[t], ys[t], inner_lr, first_order=first_order)
            qpred = net(xq[t], params)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad)
        meta_opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            curve.append((time.time() - t0, it, float(meta_loss.item())))
    train_time = time.time() - t0
    return net, train_time, curve


def eval_mlp_maml(net, n_test=200, K=10, K_q=50, inner_lr=0.01, inner_steps=1, seed=SEED,
                  noise=0.0, grad_noise=None):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, _, xq, yq = sample_tasks(n_test, K, K_q, gen, noise=noise, grad_noise=grad_noise)
    losses = []
    t0 = time.time()
    for t in range(n_test):
        params = net.init_params()
        for _ in range(inner_steps):
            # first_order=True at eval: the adapted parameters are numerically
            # identical, but we do not need the outer-gradient graph here, and
            # building it would inflate the measured per-task inference time.
            params = maml_inner_step(net, params, xs[t], ys[t], inner_lr,
                                     first_order=True)
        with torch.no_grad():
            qpred = net(xq[t], params)
        losses.append((qpred - yq[t]).pow(2).mean().item())
    eval_time = time.time() - t0
    return float(torch.tensor(losses).mean()), eval_time / n_test


# ---------------- FastLSQ-Meta ----------------

def make_basis(N=200, seed=SEED, dist="normal", nu=3):
    """Frozen base frequencies (later scaled by sigma) and phases.

    ``dist`` selects the frequency distribution p(omega): 'normal' (Gaussian RFF
    -> RBF kernel, the default); 'studentt' (Student-t(nu) -> Matern-like, heavier
    tails / rougher sample paths for non-smooth targets); 'laplace'. All use sin
    features, so the cyclic derivative identity (hence the Sobolev rows) is
    unchanged -- only the spectrum changes."""
    g = torch.Generator().manual_seed(seed)
    if dist == "normal":
        Wb = torch.randn(1, N, generator=g)     # base frequencies, will be scaled by sigma
    elif dist == "studentt":
        Z = torch.randn(1, N, generator=g)
        V = torch.randn(int(nu), N, generator=g).pow(2).sum(0, keepdim=True)  # chi^2_nu
        Wb = Z / torch.sqrt(V / nu)
    elif dist == "laplace":
        u = torch.rand(1, N, generator=g) - 0.5
        Wb = -torch.sign(u) * torch.log1p(-2.0 * u.abs())
    else:
        raise ValueError(f"unknown freq dist {dist!r}")
    bb = torch.rand(1, N, generator=g) * 2 * math.pi
    return Wb, bb


def fastlsq_inner_solve(xs, ys, dys, Wb, bb, sigma, mu, sobolev=False, grad_w=1.0):
    """Closed-form ridge solve. Returns beta (N,)."""
    W = sigma * Wb                              # (1, N)
    Z = xs @ W + bb                             # (K, N)
    H = torch.sin(Z) / math.sqrt(W.shape[1])
    if sobolev:
        G = (W * torch.cos(Z)) / math.sqrt(W.shape[1])  # d=1, shape (K, N)
        # torch.sqrt (not math.sqrt) so the gradient flows into a learnable grad_w:
        # math.sqrt() coerces a requires_grad tensor to a float and detaches it, which
        # silently froze the meta-learned w at its init. torch.as_tensor accepts both a
        # Python float (eval path) and a tensor (train path: gw = log_gw.exp()).
        sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype, device=H.device))
        A = torch.cat([H, sqrt_w * G], dim=0)
        t = torch.cat([ys, sqrt_w * dys], dim=0)
    else:
        A = H; t = ys
    AtA = A.T @ A + mu * torch.eye(W.shape[1])
    Atb = A.T @ t
    beta = torch.linalg.solve(AtA, Atb)
    return W, beta


def fastlsq_predict(xq, W, bb, beta):
    Z = xq @ W + bb
    H = torch.sin(Z) / math.sqrt(W.shape[1])
    return H @ beta


def train_fastlsq_meta(n_iters=2000, meta_bs=4, K=10, K_q=10, N=200,
                       sobolev=False, grad_w=1.0, lr=5e-3, seed=SEED, log_every=100,
                       noise=0.0, dist="normal", nu=3, mu_floor=0.0, grad_noise=None):
    """``mu_floor``: lower-clamp the ridge used in the inner solve, and report the
    clamped value. Default 0.0 leaves the original path untouched. With a floor the
    outer loop cannot drive mu into the ill-conditioned regime that makes long
    training runs diverge (see the 50 s cell of the wall-clock table).
    ``grad_noise``: see sample_tasks; None (default) keeps the original y+dy noise."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed, dist=dist, nu=nu)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(math.log(grad_w), requires_grad=True)
    params = [log_sigma, log_mu] + ([log_gw] if sobolev else [])
    opt = torch.optim.Adam(params, lr=lr)

    curve = []
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, dys, xq, yq = sample_tasks(meta_bs, K, K_q, gen, noise=noise, grad_noise=grad_noise)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        if mu_floor > 0.0:
            mu = mu.clamp_min(mu_floor)
        gw = log_gw.exp() if sobolev else 1.0
        meta_loss = 0.0
        for t in range(meta_bs):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=sobolev, grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq[t]).pow(2).mean()
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            curve.append((time.time() - t0, it, float(meta_loss.item())))
    train_time = time.time() - t0

    return {"Wb": Wb, "bb": bb,
            "sigma": float(log_sigma.exp().item()),
            "mu": max(float(log_mu.exp().item()), float(mu_floor)),
            "grad_w": float(log_gw.exp().item()) if sobolev else None,
            "sobolev": sobolev}, train_time, curve


def eval_fastlsq_meta(meta, n_test=200, K=10, K_q=50, seed=SEED, noise=0.0, grad_noise=None):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen, noise=noise, grad_noise=grad_noise)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    mu = torch.tensor(meta["mu"])
    gw = meta["grad_w"] if meta["sobolev"] else 1.0
    losses = []
    t0 = time.time()
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(
                xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                sobolev=meta["sobolev"], grad_w=gw)
            qpred = fastlsq_predict(xq[t], W, bb, beta)
            losses.append((qpred - yq[t]).pow(2).mean().item())
    eval_time = time.time() - t0
    return float(torch.tensor(losses).mean()), eval_time / n_test


# ---------------- driver ----------------

def main(save=None):
    le = 50 if save else 500   # finer convergence curve when capturing; coarse enough
    #                            that per-iter .item() sync doesn't distort wall-clock
    print("Training MLP-MAML (8000 iters)...")
    mlp_net, t_mlp, curve_mlp = train_mlp_maml(n_iters=8000, log_every=le)
    mse_mlp, inf_mlp = eval_mlp_maml(mlp_net)

    print("Training FastLSQ-Meta (2000 iters)...")
    fl_meta, t_fl, curve_fl = train_fastlsq_meta(sobolev=False, n_iters=2000, log_every=le)
    mse_fl, inf_fl = eval_fastlsq_meta(fl_meta)

    print("Training FastLSQ-Sobolev-Meta (2000 iters)...")
    fls_meta, t_fls, curve_fls = train_fastlsq_meta(sobolev=True, n_iters=2000, log_every=le)
    mse_fls, inf_fls = eval_fastlsq_meta(fls_meta)

    print("\n=========== Results: 10-shot sinusoid (50-pt query, 200 test tasks) ===========")
    print(f"{'method':<28} {'train s':>9} {'inf ms':>8} {'test MSE':>12}  learned hp")
    print(f"{'MLP-MAML (8k iters)':<28} {t_mlp:>9.2f} {inf_mlp*1000:>8.3f} {mse_mlp:>12.4f}  -")
    print(f"{'FastLSQ-Meta (2k)':<28} {t_fl:>9.2f} {inf_fl*1000:>8.3f} {mse_fl:>12.4f}  "
          f"sigma={fl_meta['sigma']:.3f} mu={fl_meta['mu']:.2e}")
    print(f"{'FastLSQ-Sobolev-Meta (2k)':<28} {t_fls:>9.2f} {inf_fls*1000:>8.3f} {mse_fls:>12.4f}  "
          f"sigma={fls_meta['sigma']:.3f} mu={fls_meta['mu']:.2e} gw={fls_meta['grad_w']:.2f}")

    print("\n--- Convergence curve (wallclock_s, iter, meta_loss) ---")
    for name, c in [("MLP", curve_mlp), ("FastLSQ", curve_fl), ("FastLSQ-Sobolev", curve_fls)]:
        print(f"\n{name}:")
        # show ~6 points
        idx = [int(i * (len(c)-1) / 5) for i in range(6)]
        for i in idx:
            t, it, l = c[i]
            print(f"  t={t:6.2f}s  iter={it:5d}  loss={l:.4f}")

    print(f"\nSpeedups vs MLP-MAML at convergence:")
    print(f"  FastLSQ-Meta:          train {t_mlp/t_fl:.1f}×    inf {inf_mlp/inf_fl:.1f}×    final-MSE ratio {mse_mlp/mse_fl:.1f}×")
    print(f"  FastLSQ-Sobolev-Meta:  train {t_mlp/t_fls:.1f}×    inf {inf_mlp/inf_fls:.1f}×    final-MSE ratio {mse_mlp/mse_fls:.1f}×")

    if save:
        import json
        import numpy as np
        np.savez(save + ".npz",
                 curve_mlp=np.array(curve_mlp, dtype=float),
                 curve_fl=np.array(curve_fl, dtype=float),
                 curve_fls=np.array(curve_fls, dtype=float))
        scal = {
            "mlp":          {"train_s": t_mlp,  "inf_ms": inf_mlp * 1000, "test_mse": mse_mlp},
            "meta_lsq":     {"train_s": t_fl,   "inf_ms": inf_fl * 1000,  "test_mse": mse_fl,
                             "sigma": fl_meta["sigma"], "mu": fl_meta["mu"], "N": 200},
            "meta_lsq_sob": {"train_s": t_fls,  "inf_ms": inf_fls * 1000, "test_mse": mse_fls,
                             "sigma": fls_meta["sigma"], "mu": fls_meta["mu"],
                             "grad_w": fls_meta["grad_w"], "N": 200},
        }
        with open(save + ".json", "w") as f:
            json.dump(scal, f, indent=2)
        print(f"\n[saved] {save}.npz and {save}.json")


if __name__ == "__main__":
    import sys
    _save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    main(save=_save)
