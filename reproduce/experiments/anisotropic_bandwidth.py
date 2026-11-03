"""Scalar vs diagonal vs full-L bandwidth for anisotropic task families.

Task: y = a sin(omega_x * x1 + omega_y * x2 + phi)
       a ~ U[0.1, 5], phi ~ U[0, pi]
       omega_x ~ U[0.5, 1.5], omega_y ~ U[3, 5]   (deliberately mismatched scales)

Three meta-models, all closed-form Sobolev solves:
  - scalar sigma         (1 learnable scalar  for W)
  - diagonal D           (d=2 learnable scalars for W)
  - full lower-tri L     (d(d+1)/2 = 3 learnable for W)

Reports test MSE on 200 held-out anisotropic tasks. Expected: diagonal closes
most of the gap to full-L; scalar is significantly worse.
"""

import math
import time
import torch

torch.set_default_dtype(torch.float32)
SEED = 0
D = 2


def sample_aniso_tasks(n_tasks, K, K_q, gen):
    a = torch.rand(n_tasks, 1, 1, generator=gen) * 4.9 + 0.1     # (B,1,1)
    phi = torch.rand(n_tasks, 1, 1, generator=gen) * math.pi
    omega_x = torch.rand(n_tasks, 1, 1, generator=gen) * 1.0 + 0.5
    omega_y = torch.rand(n_tasks, 1, 1, generator=gen) * 2.0 + 3.0
    xs = torch.rand(n_tasks, K, 1, generator=gen) * 4 - 2
    ys_in = torch.rand(n_tasks, K, 1, generator=gen) * 4 - 2
    X_s = torch.cat([xs, ys_in], dim=-1)                          # (B, K, 2)
    xq = torch.rand(n_tasks, K_q, 1, generator=gen) * 4 - 2
    yq_in = torch.rand(n_tasks, K_q, 1, generator=gen) * 4 - 2
    X_q = torch.cat([xq, yq_in], dim=-1)
    arg_s = omega_x * X_s[..., :1] + omega_y * X_s[..., 1:2] + phi
    arg_q = omega_x * X_q[..., :1] + omega_y * X_q[..., 1:2] + phi
    val_s = (a * torch.sin(arg_s)).squeeze(-1)                     # (B, K)
    val_q = (a * torch.sin(arg_q)).squeeze(-1)
    # gradients dy/dx_k
    dval_x = (a * omega_x * torch.cos(arg_s)).squeeze(-1)          # (B, K)
    dval_y = (a * omega_y * torch.cos(arg_s)).squeeze(-1)
    grad_s = torch.stack([dval_x, dval_y], dim=-1)                  # (B, K, 2)
    return X_s, val_s, grad_s, X_q, val_q


# ---------------- closed-form Sobolev solve ----------------

def assemble_AB(X, W, b):
    """X: (M, d), W: (d, N), b: (1, N).
    Returns H: (M, N), Gk: list of (M, N), one per dim k.
    """
    z = X @ W + b
    N = W.shape[1]
    H = torch.sin(z) / math.sqrt(N)
    Gs = [(W[k:k+1, :] * torch.cos(z)) / math.sqrt(N) for k in range(W.shape[0])]
    return H, Gs


def solve_task(X, y, grad, W, b, mu, grad_w):
    H, Gs = assemble_AB(X, W, b)
    G_stack = torch.cat(Gs, dim=0)
    dy_stack = grad.T.reshape(-1)
    # torch.sqrt (not math.sqrt) so the gradient flows into the learnable grad_w
    # (math.sqrt detaches a requires_grad tensor; see maml_sinusoid.fastlsq_inner_solve).
    sqrt_w = torch.sqrt(torch.as_tensor(grad_w, dtype=H.dtype, device=H.device))
    A = torch.cat([H, sqrt_w * G_stack], dim=0)
    t = torch.cat([y, sqrt_w * dy_stack], dim=0)
    K = A.T @ A + mu * torch.eye(W.shape[1])
    beta = torch.linalg.solve(K, A.T @ t)
    return beta


def predict(X_q, W, b, beta):
    z = X_q @ W + b
    return torch.sin(z) / math.sqrt(W.shape[1]) @ beta


# ---------------- bandwidth parameterisations ----------------

def W_from_scalar(log_sigma, Wb):
    return log_sigma.exp() * Wb


def W_from_diag(log_diag, Wb):
    # log_diag: (d,)  Wb: (d, N)
    return log_diag.exp().unsqueeze(-1) * Wb


def W_from_full(L_flat, Wb):
    # L_flat: d*(d+1)/2 elements; build lower-tri L
    d = Wb.shape[0]
    L = torch.zeros(d, d)
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                L[i, j] = L_flat[idx].exp()  # diag positive
            else:
                L[i, j] = L_flat[idx]
            idx += 1
    return L @ Wb


# ---------------- training ----------------

def train_meta(parameterisation, n_iters=1500, lr=5e-3, N=300, meta_bs=4,
               K=15, K_q=20, log_every=200, seed=SEED):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    g_basis = torch.Generator().manual_seed(seed + 7)
    Wb = torch.randn(D, N, generator=g_basis)
    b = torch.rand(1, N, generator=g_basis) * 2 * math.pi

    if parameterisation == "scalar":
        params = [torch.tensor(0.0, requires_grad=True)]
        W_fn = lambda: W_from_scalar(params[0], Wb)
    elif parameterisation == "diag":
        params = [torch.zeros(D, requires_grad=True)]
        W_fn = lambda: W_from_diag(params[0], Wb)
    elif parameterisation == "full":
        params = [torch.zeros(D * (D + 1) // 2, requires_grad=True)]
        # init: identity-like for the L matrix => log of diag entries = 0
        with torch.no_grad():
            params[0][0] = 0.0  # L[0,0] = 1
            params[0][1] = 0.0  # L[1,0] = 0
            params[0][2] = 0.0  # L[1,1] = 1
        W_fn = lambda: W_from_full(params[0], Wb)
    else:
        raise ValueError(parameterisation)

    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(0.0, requires_grad=True)
    all_params = params + [log_mu, log_gw]
    opt = torch.optim.Adam(all_params, lr=lr)

    for it in range(n_iters):
        X_s, val_s, grad_s, X_q, val_q = sample_aniso_tasks(meta_bs, K, K_q, gen)
        opt.zero_grad()
        W = W_fn()
        mu = log_mu.exp()
        gw = log_gw.exp()
        loss = 0.0
        for t in range(meta_bs):
            beta = solve_task(X_s[t], val_s[t], grad_s[t], W, b, mu, gw)
            qpred = predict(X_q[t], W, b, beta)
            loss = loss + ((qpred - val_q[t]) ** 2).mean()
        loss = loss / meta_bs
        loss.backward()
        opt.step()
        if it % log_every == 0:
            print(f"  iter {it:5d}  meta-loss {float(loss.item()):.4e}")

    return {"params": params, "Wb": Wb, "b": b, "mu": float(log_mu.exp().item()),
            "gw": float(log_gw.exp().item()), "W_fn": W_fn,
            "parameterisation": parameterisation,
            "n_learnable": sum(p.numel() for p in all_params)}


def eval_meta(meta, n_test=200, K=15, K_q=50, seed=SEED):
    gen = torch.Generator().manual_seed(seed + 9999)
    X_s, val_s, grad_s, X_q, val_q = sample_aniso_tasks(n_test, K, K_q, gen)
    losses = []
    with torch.no_grad():
        W = meta["W_fn"]()
        for t in range(n_test):
            beta = solve_task(X_s[t], val_s[t], grad_s[t], W, meta["b"],
                              torch.tensor(meta["mu"]), torch.tensor(meta["gw"]))
            qpred = predict(X_q[t], W, meta["b"], beta)
            losses.append(((qpred - val_q[t]) ** 2).mean().item())
    return float(torch.tensor(losses).mean())


def main(save=None):
    results = []
    for p in ("scalar", "diag", "full"):
        print(f"\n--- Training {p} ---")
        t0 = time.time()
        meta = train_meta(p)
        t_train = time.time() - t0
        mse = eval_meta(meta)
        print(f"  train: {t_train:.1f}s   test MSE: {mse:.4e}   "
              f"n_learnable: {meta['n_learnable']}")
        results.append((p, t_train, mse, meta["n_learnable"]))

    print(f"\n=== Summary ===")
    print(f"{'param':<10} {'train s':>10} {'test MSE':>12} {'n learn':>10}")
    for p, t, m, n in results:
        print(f"{p:<10} {t:>10.1f} {m:>12.4e} {n:>10}")

    if save:
        import json
        out = {p: {"train_s": t, "test_mse": m, "n_learnable": n} for p, t, m, n in results}
        with open(save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {save}")


if __name__ == "__main__":
    import sys
    _save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    main(save=_save)
