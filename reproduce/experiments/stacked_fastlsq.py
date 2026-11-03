"""Stacked FastLSQ: shared (W, b) basis across L residual sinusoidal blocks.

Tests the proposed deep extension of FastLSQ. Each layer is a residual block
with a shared frozen basis (W, b) and learnable per-layer readout B_l and
bandwidth sigma_l. Trained end-to-end with backprop. The per-layer Jacobian is
closed-form via the cyclic identity; we verify it matches autograd and then
compose Jacobians by chain rule to get d(output)/dx analytically.

Architecture:
  h_0 = x in R^d
  z_l = sigma_l * (h_{l-1} @ W) + b              # W (d, N), b (N,) shared, frozen
  h_l = h_{l-1} + (1/sqrt(N)) * sin(z_l) @ B_l^T  # B_l (d, N) per layer
  y   = w_out @ h_L + bias_out                    # scalar output

Per-layer Jacobian:
  J_l(h) = I + (sigma_l / sqrt(N)) * B_l @ diag(cos(z_l(h))) @ W^T   in R^{d x d}

End-to-end gradient: w_out^T @ J_L(h_{L-1}) @ J_{L-1}(h_{L-2}) @ ... @ J_1(h_0).

Comparisons on the GMM log-density target (d=4) against:
  (a) single-layer FastLSQ-Sobolev    (Table 1 of paper)
  (b) MLP-Sobolev                     (Table 1 of paper)
"""

import math
import time
import torch
import torch.nn as nn

from experiments.derivative_oracle_test import GMM, relrmse

torch.set_default_dtype(torch.float64)
SEED = 0


class StackedFastLSQ(nn.Module):
    def __init__(self, d, N, L, sigma_init=1.0, b_scale=0.5, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d, N, generator=g).double()
        b = (torch.rand(N, generator=g) * 2 * math.pi).double()
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        Bs = [nn.Parameter((torch.randn(d, N, generator=g).double() * (b_scale / math.sqrt(N))))
              for _ in range(L)]
        self.B_list = nn.ParameterList(Bs)
        self.log_sigma = nn.Parameter(torch.full((L,), math.log(sigma_init), dtype=torch.float64))
        self.w_out = nn.Parameter(torch.randn(d, generator=g).double() * 0.1)
        self.bias_out = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.L = L
        self.N = N
        self.d = d

    def forward(self, x):
        h = x
        for l in range(self.L):
            sigma = self.log_sigma[l].exp()
            z = sigma * (h @ self.W) + self.b
            h = h + torch.sin(z) @ self.B_list[l].T / math.sqrt(self.N)
        return h @ self.w_out + self.bias_out

    def forward_with_jacobian(self, x):
        """Returns (y, dy/dx) with the Jacobian computed in closed form."""
        M, d = x.shape[0], self.d
        h = x
        # Running input-Jacobian d h_l / d x: starts as I.
        J = torch.eye(d, dtype=x.dtype).unsqueeze(0).expand(M, d, d).contiguous()
        for l in range(self.L):
            sigma = self.log_sigma[l].exp()
            z = sigma * (h @ self.W) + self.b              # (M, N)
            cos_z = torch.cos(z)                            # (M, N)
            scale = sigma / math.sqrt(self.N)
            # J_l(h) = I + scale * B_l @ diag(cos_z[m]) @ W^T,  per-row.
            # B_scaled[m, i, n] = B_l[i, n] * cos_z[m, n]
            B_scaled = self.B_list[l].unsqueeze(0) * cos_z.unsqueeze(1)  # (M, d, N)
            J_l = torch.eye(d, dtype=x.dtype).unsqueeze(0) + scale * (B_scaled @ self.W.T)
            J = J_l @ J
            h = h + torch.sin(z) @ self.B_list[l].T / math.sqrt(self.N)
        y = h @ self.w_out + self.bias_out
        # d y / d x = w_out^T @ J  per batch row -> (M, d)
        grad = (J.transpose(1, 2) @ self.w_out.unsqueeze(-1)).squeeze(-1)
        return y, grad


def verify_jacobian(net, x, atol=1e-7):
    """Sanity check: analytic Jacobian must match autograd."""
    x_ag = x.detach().clone().requires_grad_(True)
    y_auto = net(x_ag)
    g_auto = torch.autograd.grad(y_auto.sum(), x_ag)[0]
    y_anal, g_anal = net.forward_with_jacobian(x)
    val_diff = (y_auto - y_anal).abs().max().item()
    grad_diff = (g_auto - g_anal).abs().max().item()
    print(f"  value max-diff   = {val_diff:.2e}")
    print(f"  gradient max-diff = {grad_diff:.2e}")
    return grad_diff < atol


def train_value(net, X_tr, y_tr, epochs=3000, lr=1e-3, bs=2048, log_every=500):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        idx = torch.randperm(X_tr.shape[0])[:bs]
        pred = net(X_tr[idx])
        loss = (pred - y_tr[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % log_every == 0:
            losses.append(loss.item())
    return losses


def train_sobolev(net, X_tr, y_tr, g_tr, epochs=3000, lr=1e-3, bs=2048,
                  grad_weight=1.0, log_every=500):
    """Train with values + analytic-Jacobian gradient supervision.
    The analytic Jacobian path means NO double-backprop here -- each gradient
    target is computed via forward_with_jacobian, which is itself differentiable.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        idx = torch.randperm(X_tr.shape[0])[:bs]
        pred_v, pred_g = net.forward_with_jacobian(X_tr[idx])
        loss_v = (pred_v - y_tr[idx]).pow(2).mean()
        loss_g = (pred_g - g_tr[idx]).pow(2).mean()
        loss = loss_v + grad_weight * loss_g
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % log_every == 0:
            losses.append((loss_v.item(), loss_g.item()))
    return losses


def evaluate(net, X_te, y_te, g_te):
    """Value rRMSE and grad rRMSE on test points (no_grad: pure inference)."""
    with torch.no_grad():
        val, grad = net.forward_with_jacobian(X_te)
    return relrmse(val, y_te), relrmse(grad, g_te)


def n_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def main():
    d = 4
    gmm = GMM(d, K=3, seed=SEED)
    X_tr = gmm.sample(2000, seed=SEED)
    y_tr = gmm.log_p(X_tr)
    g_tr = gmm.grad_log_p(X_tr)
    X_te = gmm.sample(400, seed=SEED + 1)
    y_te = gmm.log_p(X_te)
    g_te = gmm.grad_log_p(X_te)

    print(f"\n=== Sanity: analytic Jacobian vs autograd (d={d}, random net) ===")
    for L in (1, 2, 3, 4):
        net_check = StackedFastLSQ(d=d, N=200, L=L, seed=SEED + L)
        print(f"L={L}:")
        ok = verify_jacobian(net_check, X_te[:64])
        if not ok:
            print(f"  WARNING: analytic Jacobian does NOT match autograd at L={L}")

    print(f"\n=== Value-only training (Adam, 3000 epochs) ===")
    configs = [
        ("L=1 N=500", dict(N=500, L=1)),
        ("L=2 N=500", dict(N=500, L=2)),
        ("L=3 N=500", dict(N=500, L=3)),
        ("L=4 N=500", dict(N=500, L=4)),
        ("L=2 N=200", dict(N=200, L=2)),
        ("L=4 N=200", dict(N=200, L=4)),
    ]
    val_rows = []
    for name, cfg in configs:
        torch.manual_seed(SEED)
        net = StackedFastLSQ(d=d, **cfg, seed=SEED)
        t0 = time.time()
        train_value(net, X_tr, y_tr, epochs=3000, lr=1e-3)
        t_train = time.time() - t0
        v, g = evaluate(net, X_te, y_te, g_te)
        p = n_params(net)
        val_rows.append((name, p, t_train, v, g))
        print(f"  {name:<10}  params={p:>6}  train={t_train:5.1f}s  val={v:.3e}  grad={g:.3e}")

    print(f"\n=== Sobolev training (value + analytic-Jacobian grad) ===")
    sob_rows = []
    for name, cfg in configs:
        torch.manual_seed(SEED)
        net = StackedFastLSQ(d=d, **cfg, seed=SEED)
        t0 = time.time()
        train_sobolev(net, X_tr, y_tr, g_tr, epochs=3000, lr=1e-3, grad_weight=1.0)
        t_train = time.time() - t0
        v, g = evaluate(net, X_te, y_te, g_te)
        p = n_params(net)
        sob_rows.append((name, p, t_train, v, g))
        print(f"  {name:<10}  params={p:>6}  train={t_train:5.1f}s  val={v:.3e}  grad={g:.3e}")

    print(f"\n=== Baselines from Table 1 (d=4) ===")
    print(f"  FastLSQ-Sobolev (N=1500, 1 layer):  val=6.17e-02  grad=3.91e-01  fit=0.2s")
    print(f"  MLP-Sobolev     (3x96 SiLU):        val=9.95e-03  grad=4.58e-02  fit=11.5s")

    print(f"\n=== Summary: value-only training ===")
    print(f"  {'config':<10} {'params':>7} {'train s':>8} {'val rRMSE':>11} {'grad rRMSE':>11}")
    for name, p, t, v, g in val_rows:
        print(f"  {name:<10} {p:>7} {t:>8.1f} {v:>11.3e} {g:>11.3e}")

    print(f"\n=== Summary: Sobolev training (val + grad targets) ===")
    print(f"  {'config':<10} {'params':>7} {'train s':>8} {'val rRMSE':>11} {'grad rRMSE':>11}")
    for name, p, t, v, g in sob_rows:
        print(f"  {name:<10} {p:>7} {t:>8.1f} {v:>11.3e} {g:>11.3e}")


if __name__ == "__main__":
    main()
