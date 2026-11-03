"""Few-shot N-way K-shot CLASSIFICATION via closed-form one-vs-rest ridge in a
frozen RFF basis (Meta-LSQ-Classify) vs a small MLP-MAML classifier.

This broadens Meta-LSQ from regression to classification (R2D2 / MetaOptNet
territory): instead of fitting y = f(x), we one-hot encode N-way labels and
solve a single ridge least-squares system in closed form on the support set,
then predict the query class by argmax of the linear read-out. Only two scalars
(sigma, mu) are meta-learned; the RFF basis (W, b) is frozen.

DATA  -- synthetic N-way K-shot Gaussian blobs in d=2:
  Each task draws N_way class centers ~ U[-3, 3]^2; each class samples
  K support + Kq query points ~ center + 0.4 * N(0, I_2).

CLOSED-FORM CLASSIFIER (per task):
  Phi_s = sin(W^T x + b)/sqrt(M)   features for the NK support points (NK x M),
          with W ~ N(0, sigma^2 I_2)  (sigma scales a frozen base spectrum).
  Y_s   = one-hot support labels (NK x N_way).
  Beta  = (Phi_s^T Phi_s + mu I)^{-1} Phi_s^T Y_s     (M x N_way).
  Predict argmax_c (Phi_q Beta)_c.

META-TRAINING: Adam minimises query MSE to one-hot through the differentiable
solve, across sampled tasks. (CE-on-query is also tried as an ablation.)

BASELINE: MAMLNet adapted to N-way logits, cross-entropy inner loss, 1-5 inner
SGD steps.

We compare query ACCURACY (3 seeds, 200 test tasks) and learnable-param counts.
Self-contained; imports Meta-LSQ helpers read-only.
"""
import os, sys, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F

# import the frozen basis + MLP read-only from the canonical module
from experiments.maml_sinusoid import MAMLNet, make_basis  # noqa: E402

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")


# ------------------------- data: N-way K-shot Gaussian blobs -------------------------

def sample_class_tasks(n_tasks, N_way, K, Kq, gen, d=2, spread=0.4, center_range=3.0):
    """Return (xs, ys, xq, yq) for n_tasks N-way K-shot blob tasks.

    xs: (n_tasks, N_way*K,  d)   support inputs
    ys: (n_tasks, N_way*K)       support integer labels in [0, N_way)
    xq: (n_tasks, N_way*Kq, d)   query inputs
    yq: (n_tasks, N_way*Kq)      query integer labels
    Points are grouped by class (class 0 first K, etc.) but the closed-form
    solve and CE losses are permutation-invariant in row order, so no shuffle
    is needed.
    """
    # centers: (n_tasks, N_way, d) ~ U[-center_range, center_range]
    centers = (torch.rand(n_tasks, N_way, d, generator=gen) * 2 - 1) * center_range

    def draw(n_per):
        # noise: (n_tasks, N_way, n_per, d)
        eps = torch.randn(n_tasks, N_way, n_per, d, generator=gen) * spread
        pts = centers.unsqueeze(2) + eps                       # (n_tasks, N_way, n_per, d)
        x = pts.reshape(n_tasks, N_way * n_per, d)
        lab = torch.arange(N_way).repeat_interleave(n_per)      # (N_way*n_per,)
        y = lab.unsqueeze(0).expand(n_tasks, -1).contiguous()   # (n_tasks, N_way*n_per)
        return x, y

    xs, ys = draw(K)
    xq, yq = draw(Kq)
    return xs, ys, xq, yq


# ------------------------- Meta-LSQ closed-form classifier -------------------------

def make_basis_d(M, d, seed):
    """Frozen base RFF frequencies for d-dim input.

    Returns Wb: (d, M) base frequencies ~ N(0, I) (later scaled by sigma) and
    bb: (1, M) phases ~ U[0, 2pi). For d=1 this reduces to make_basis() from the
    canonical module (which returns a (1, M) row); we reuse that for the phases'
    RNG convention but build the d-row frequency matrix here.
    """
    g = torch.Generator().manual_seed(seed)
    Wb = torch.randn(d, M, generator=g)          # (d, M)
    bb = torch.rand(1, M, generator=g) * 2 * math.pi
    return Wb, bb


def rff_features(x, W, bb):
    """x: (n, d), W: (d, M), bb: (1, M) -> Phi: (n, M)."""
    Z = x @ W + bb
    return torch.sin(Z) / math.sqrt(W.shape[1])


def ridge_classify_solve(xs, ys_onehot, Wb, bb, sigma, mu):
    """Closed-form one-vs-rest ridge in RFF space.

    xs: (NK, d), ys_onehot: (NK, N_way). Returns (W, Beta) with Beta (M, N_way).
    """
    W = sigma * Wb                                # (d, M)
    Phi = rff_features(xs, W, bb)                 # (NK, M)
    M = W.shape[1]
    A = Phi.T @ Phi + mu * torch.eye(M)           # (M, M)
    B = Phi.T @ ys_onehot                         # (M, N_way)
    Beta = torch.linalg.solve(A, B)               # (M, N_way)
    return W, Beta


def ridge_classify_predict(xq, W, bb, Beta):
    Phi_q = rff_features(xq, W, bb)               # (Nq, M)
    return Phi_q @ Beta                           # (Nq, N_way) scores


def train_meta_lsq_classify(n_iters, N_way, K, Kq, M, seed,
                            meta_bs=4, lr=5e-3, loss_type="mse",
                            d=2, spread=0.4, center_range=3.0, log_every=200):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis_d(M, d, seed)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-3), requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu], lr=lr)

    eye = torch.eye(N_way)
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, xq, yq = sample_class_tasks(meta_bs, N_way, K, Kq, gen,
                                            d=d, spread=spread, center_range=center_range)
        opt.zero_grad()
        sigma = log_sigma.exp()
        mu = log_mu.exp()
        meta_loss = 0.0
        for t in range(meta_bs):
            ys_oh = eye[ys[t]]                       # (NK, N_way)
            W, Beta = ridge_classify_solve(xs[t], ys_oh, Wb, bb, sigma, mu)
            scores = ridge_classify_predict(xq[t], W, bb, Beta)   # (Nq, N_way)
            if loss_type == "mse":
                yq_oh = eye[yq[t]]
                meta_loss = meta_loss + (scores - yq_oh).pow(2).mean()
            else:  # cross-entropy on query scores
                meta_loss = meta_loss + F.cross_entropy(scores, yq[t])
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        opt.step()
    train_time = time.time() - t0
    return {"Wb": Wb, "bb": bb,
            "sigma": float(log_sigma.exp().item()),
            "mu": float(log_mu.exp().item()),
            "N_way": N_way, "M": M, "d": d}, train_time


def eval_meta_lsq_classify(meta, n_test, N_way, K, Kq, seed,
                           spread=0.4, center_range=3.0):
    gen = torch.Generator().manual_seed(seed + 9999)
    d = meta["d"]
    xs, ys, xq, yq = sample_class_tasks(n_test, N_way, K, Kq, gen,
                                        d=d, spread=spread, center_range=center_range)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"]); mu = torch.tensor(meta["mu"])
    eye = torch.eye(N_way)
    accs = []
    t0 = time.time()
    with torch.no_grad():
        for t in range(n_test):
            ys_oh = eye[ys[t]]
            W, Beta = ridge_classify_solve(xs[t], ys_oh, Wb, bb, sigma, mu)
            scores = ridge_classify_predict(xq[t], W, bb, Beta)
            pred = scores.argmax(dim=1)
            accs.append((pred == yq[t]).float().mean().item())
    inf_ms = (time.time() - t0) / n_test * 1000
    return float(torch.tensor(accs).mean()), float(torch.tensor(accs).std()), inf_ms


# ------------------------- MLP-MAML classifier baseline -------------------------

def maml_cls_inner_step(net, params, xs, ys, lr, first_order=False):
    logits = net(xs, params)              # (NK, N_way)
    loss = F.cross_entropy(logits, ys)
    grads = torch.autograd.grad(loss, list(params.values()), create_graph=not first_order)
    return {k: v - lr * g for (k, v), g in zip(params.items(), grads)}


def train_mlp_maml_classify(n_iters, N_way, K, Kq, seed, hidden=40, depth=3,
                            inner_lr=0.05, inner_steps=3, outer_lr=1e-3,
                            meta_bs=4, d=2, spread=0.4, center_range=3.0,
                            first_order=False):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    net = MAMLNet(hidden=hidden, depth=depth, in_dim=d, out_dim=N_way).to(device)
    meta_opt = torch.optim.Adam(net.parameters(), lr=outer_lr)
    t0 = time.time()
    for it in range(n_iters):
        xs, ys, xq, yq = sample_class_tasks(meta_bs, N_way, K, Kq, gen,
                                            d=d, spread=spread, center_range=center_range)
        meta_opt.zero_grad()
        meta_loss = 0.0
        for t in range(meta_bs):
            params = net.init_params()
            for _ in range(inner_steps):
                params = maml_cls_inner_step(net, params, xs[t], ys[t], inner_lr,
                                             first_order=first_order)
            qlogits = net(xq[t], params)
            meta_loss = meta_loss + F.cross_entropy(qlogits, yq[t])
        meta_loss = meta_loss / meta_bs
        meta_loss.backward()
        meta_opt.step()
    train_time = time.time() - t0
    n_params = sum(p.numel() for p in net.parameters())
    return net, train_time, n_params


def eval_mlp_maml_classify(net, n_test, N_way, K, Kq, seed, inner_lr=0.05,
                           inner_steps=3, d=2, spread=0.4, center_range=3.0):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, xq, yq = sample_class_tasks(n_test, N_way, K, Kq, gen,
                                        d=d, spread=spread, center_range=center_range)
    accs = []
    t0 = time.time()
    for t in range(n_test):
        params = net.init_params()
        for _ in range(inner_steps):
            params = maml_cls_inner_step(net, params, xs[t], ys[t], inner_lr)
        qlogits = net(xq[t], params)
        pred = qlogits.argmax(dim=1)
        accs.append((pred == yq[t]).float().mean().item())
    inf_ms = (time.time() - t0) / n_test * 1000
    return float(torch.tensor(accs).mean()), float(torch.tensor(accs).std()), inf_ms


# Note on MAMLNet output: forward() ends with .squeeze(-1). For out_dim=N_way>1
# the last dim (N_way) is never size 1, so squeeze(-1) is a no-op and logits keep
# shape (NK, N_way). (Verified at runtime.)


# ------------------------- driver -------------------------

def run_setting(N_way, K, Kq, M, n_iters_lsq, n_iters_mlp, n_test, seeds,
                inner_steps):
    lsq_acc, mlp_acc = [], []
    lsq_inf, mlp_inf = [], []
    lsq_train, mlp_train = [], []
    sigmas, mus = [], []
    n_params_mlp = None
    lsq_ce_acc = []
    for s in seeds:
        meta, tt = train_meta_lsq_classify(n_iters_lsq, N_way, K, Kq, M, seed=s,
                                           loss_type="mse")
        a, _, inf = eval_meta_lsq_classify(meta, n_test, N_way, K, Kq, seed=s)
        lsq_acc.append(a); lsq_inf.append(inf); lsq_train.append(tt)
        sigmas.append(meta["sigma"]); mus.append(meta["mu"])

        # CE-objective ablation for the closed-form classifier
        meta_ce, _ = train_meta_lsq_classify(n_iters_lsq, N_way, K, Kq, M, seed=s,
                                             loss_type="ce")
        a_ce, _, _ = eval_meta_lsq_classify(meta_ce, n_test, N_way, K, Kq, seed=s)
        lsq_ce_acc.append(a_ce)

        net, tt_m, npm = train_mlp_maml_classify(n_iters_mlp, N_way, K, Kq, seed=s,
                                                 inner_steps=inner_steps)
        n_params_mlp = npm
        am, _, infm = eval_mlp_maml_classify(net, n_test, N_way, K, Kq, seed=s,
                                             inner_steps=inner_steps)
        mlp_acc.append(am); mlp_inf.append(infm); mlp_train.append(tt_m)

    def ms(v):
        t = torch.tensor(v)
        return float(t.mean()), float(t.std())

    out = {
        "N_way": N_way, "K": K, "Kq": Kq, "M": M, "n_test": n_test,
        "seeds": seeds, "inner_steps": inner_steps,
        "lsq_acc_mean": ms(lsq_acc)[0], "lsq_acc_std": ms(lsq_acc)[1],
        "lsq_acc_per_seed": lsq_acc,
        "lsq_ce_acc_mean": ms(lsq_ce_acc)[0], "lsq_ce_acc_per_seed": lsq_ce_acc,
        "mlp_acc_mean": ms(mlp_acc)[0], "mlp_acc_std": ms(mlp_acc)[1],
        "mlp_acc_per_seed": mlp_acc,
        "lsq_inf_ms": ms(lsq_inf)[0], "mlp_inf_ms": ms(mlp_inf)[0],
        "lsq_train_s": ms(lsq_train)[0], "mlp_train_s": ms(mlp_train)[0],
        "sigma_mean": ms(sigmas)[0], "mu_mean": ms(mus)[0],
        "sigmas": sigmas, "mus": mus,
        "lsq_n_learnable": 2,
        "mlp_n_learnable": n_params_mlp,
        "chance_acc": 1.0 / N_way,
    }
    return out


def main():
    seeds = [0, 1, 2]
    n_test = 200
    M = 200
    results = {}

    print("=== 5-way 5-shot ===")
    results["5way_5shot"] = run_setting(
        N_way=5, K=5, Kq=15, M=M,
        n_iters_lsq=1500, n_iters_mlp=2000, n_test=n_test, seeds=seeds,
        inner_steps=5)
    r = results["5way_5shot"]
    print(f"  Meta-LSQ acc {r['lsq_acc_mean']:.4f} +/- {r['lsq_acc_std']:.4f}  "
          f"(CE obj {r['lsq_ce_acc_mean']:.4f})")
    print(f"  MLP-MAML acc {r['mlp_acc_mean']:.4f} +/- {r['mlp_acc_std']:.4f}  "
          f"({r['mlp_n_learnable']} params)")
    print(f"  sigma {r['sigma_mean']:.3f}  mu {r['mu_mean']:.2e}  chance {r['chance_acc']:.3f}")

    print("=== 5-way 1-shot ===")
    results["5way_1shot"] = run_setting(
        N_way=5, K=1, Kq=15, M=M,
        n_iters_lsq=1500, n_iters_mlp=2000, n_test=n_test, seeds=seeds,
        inner_steps=5)
    r = results["5way_1shot"]
    print(f"  Meta-LSQ acc {r['lsq_acc_mean']:.4f} +/- {r['lsq_acc_std']:.4f}  "
          f"(CE obj {r['lsq_ce_acc_mean']:.4f})")
    print(f"  MLP-MAML acc {r['mlp_acc_mean']:.4f} +/- {r['mlp_acc_std']:.4f}")

    os.makedirs(_os.path.join(_REPO_ROOT, 'results'), exist_ok=True)
    out_path = _os.path.join(_REPO_ROOT, 'results', 'closedform_classify.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
