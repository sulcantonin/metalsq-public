"""When does the unfloored ridge actually break, and does the floor prevent it?

The equal-wall-clock table once reported a 50 s cell in which 2 of 3 seeds
diverged. Re-running it on different hardware, all three seeds are fine: at a
fixed wall-clock budget the ITERATION COUNT is machine-dependent, and 50 s on a
faster machine buys more outer steps than on a slower one. Reporting "diverged"
against a wall-clock budget therefore describes the machine as much as the
method.

This script replaces that anecdote with the underlying curve. Meta-LSQ-Sobolev
is trained far past convergence while we record, at checkpoints, the learned
ridge mu, the condition number of the normal equations it implies, and held-out
test MSE -- with the ridge floor off and on. The prediction from
Sec. "Where Does It Break?" is specific: with no floor, nothing stops Adam
shrinking an already-negligible mu, so cond(A'A + mu I) grows without bound and
the solve eventually loses the fit; with a floor, mu stops falling and the run
is flat forever.

Iteration count, not seconds, is the honest x-axis for this claim.

Run:  OMP_NUM_THREADS=1 python3 experiments/ridge_divergence.py --save results/ridge_divergence.json
"""

import os, sys, math, json, time, argparse

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import (
    make_basis, sample_tasks, fastlsq_inner_solve, fastlsq_predict,
)

torch.set_default_dtype(torch.float32)

SEEDS = (0, 1, 2)
CHECKPOINTS = (2000, 5000, 10000, 20000, 40000, 80000)
RESULT_PATH = os.path.join(_REPO_ROOT, "results", "ridge_divergence.json")


def evaluate(Wb, bb, sigma, mu, gw, n_test=200, K=10, K_q=50, seed=0):
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sample_tasks(n_test, K, K_q, gen)
    out = []
    with torch.no_grad():
        for t in range(n_test):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], Wb, bb,
                                          torch.tensor(sigma), torch.tensor(mu),
                                          sobolev=True, grad_w=gw)
            out.append(float((fastlsq_predict(xq[t], W, bb, beta) - yq[t]).pow(2).mean()))
    t = torch.tensor(out)
    return {"mean": float(t.mean()), "median": float(t.median()),
            "max": float(t.max())}


def condition_number(Wb, bb, sigma, mu, gw, K=10, seed=0):
    """cond(A'A + mu I) on one representative task."""
    gen = torch.Generator().manual_seed(seed + 4242)
    xs, ys, dys, _, _ = sample_tasks(1, K, 1, gen)
    W = torch.tensor(sigma) * Wb
    Z = xs[0] @ W + bb
    N = W.shape[1]
    H = torch.sin(Z) / math.sqrt(N)
    G = (W * torch.cos(Z)) / math.sqrt(N)
    A = torch.cat([H, math.sqrt(gw) * G], 0)
    M = (A.T @ A + mu * torch.eye(N)).double()
    ev = torch.linalg.eigvalsh(M)
    return float(ev[-1] / ev[0].clamp_min(1e-300))


def run(mu_floor, seed, n_iters, checkpoints, N=200, lr=5e-3, meta_bs=4, K=10, K_q=10):
    """One long training run, evaluated at checkpoints without restarting."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu, log_gw], lr=lr)

    marks, t0 = [], time.time()
    for it in range(1, n_iters + 1):
        xs, ys, dys, xq, yq = sample_tasks(meta_bs, K, K_q, gen)
        opt.zero_grad()
        sigma, mu, gw = log_sigma.exp(), log_mu.exp(), log_gw.exp()
        if mu_floor > 0.0:
            mu = mu.clamp_min(mu_floor)
        loss = 0.0
        for t in range(meta_bs):
            W, beta = fastlsq_inner_solve(xs[t], ys[t], dys[t], Wb, bb, sigma, mu,
                                          sobolev=True, grad_w=gw)
            loss = loss + (fastlsq_predict(xq[t], W, bb, beta) - yq[t]).pow(2).mean()
        (loss / meta_bs).backward()
        opt.step()

        if it in checkpoints:
            s = float(log_sigma.exp()); g = float(log_gw.exp())
            m = max(float(log_mu.exp()), mu_floor)
            e = evaluate(Wb, bb, s, m, g, seed=seed)
            marks.append({"iters": it, "sigma": s, "mu": m, "grad_w": g,
                          "cond": condition_number(Wb, bb, s, m, g, seed=seed),
                          **e, "elapsed_s": time.time() - t0})
            print(f"    it {it:6d}  mu {m:.2e}  cond {marks[-1]['cond']:.1e}  "
                  f"mse {e['mean']:.4g} (max task {e['max']:.3g})")
    return marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=RESULT_PATH)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=80000)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    cps = tuple(c for c in CHECKPOINTS if c <= args.max_iters)

    out = {"checkpoints": list(cps), "seeds": list(seeds), "runs": {}}
    for tag, floor in (("no floor", 0.0), ("floor 1e-2", 1e-2)):
        out["runs"][tag] = {}
        for s in seeds:
            print(f"\n== {tag}, seed {s} ==")
            out["runs"][tag][str(s)] = run(floor, s, args.max_iters, cps)

    print("\n=== mean test MSE by iteration count ===")
    hdr = "".join(f"{c:>12}" for c in cps)
    print(f"{'policy':<14}{hdr}")
    for tag in out["runs"]:
        row = f"{tag:<14}"
        for i, c in enumerate(cps):
            vals = [out["runs"][tag][str(s)][i]["mean"] for s in seeds]
            row += f"{sum(vals) / len(vals):>12.4g}"
        print(row)
    print(f"\n=== learned mu ===")
    print(f"{'policy':<14}{hdr}")
    for tag in out["runs"]:
        row = f"{tag:<14}"
        for i, c in enumerate(cps):
            vals = [out["runs"][tag][str(s)][i]["mu"] for s in seeds]
            row += f"{sum(vals) / len(vals):>12.2e}"
        print(row)
    print(f"\n=== cond(A'A + mu I) ===")
    print(f"{'policy':<14}{hdr}")
    for tag in out["runs"]:
        row = f"{tag:<14}"
        for i, c in enumerate(cps):
            vals = [out["runs"][tag][str(s)][i]["cond"] for s in seeds]
            row += f"{sum(vals) / len(vals):>12.2e}"
        print(row)
    n_div = sum(1 for tag in out["runs"] for s in seeds
                for m in out["runs"][tag][str(s)] if m["mean"] > 1.0)
    print(f"\nrun-checkpoints with mean MSE > 1.0: {n_div}")

    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
