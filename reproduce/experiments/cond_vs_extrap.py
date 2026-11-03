"""Experiment #5: numerical conditioning vs genuine extrapolation.

The paper attributes out-of-class (OOC) prediction blowups to model
extrapolation under a near-zero ridge. This script rules out (or quantifies)
NUMERICAL CONDITIONING as a separate cause.

Meta-train Meta-LSQ-Sobolev on the canonical sinusoid family, then for two
task families (in-class sin, OOC triangle) and two ridges (the meta-learned
mu_star ~ 1e-5, and mu=1e-2):

  (1) Per task, compute
        cond(A^T A + mu I)      (= cond(A)^2 essentially, the normal-equation
                                   matrix that fastlsq_inner_solve forms)
        blowup = max|u_hat over query| / max|y_true over query|
      and report Pearson correlation between log(cond) and log(blowup).

  (2) Solve the SAME per-task systems two ways at small mu:
        (i)  normal equations:  beta = (A^T A + mu I)^{-1} A^T t
        (ii) stabilised Tikhonov via torch.linalg.lstsq on the augmented system
             A_aug = [A ; sqrt(mu) I],  t_aug = [t ; 0]
             (this never squares cond(A); solves min ||A beta - t||^2 + mu||beta||^2
              exactly, the same objective, but with cond(A) not cond(A)^2).
      Report whether the stabilised solver reduces blowups and changes test MSE.

CONCLUSION: if the stabilised solver removes the blowups, they are a conditioning
artifact; if both solvers blow up identically and only a larger ridge tames them,
they are genuine extrapolation.

Self-contained; imports read-only from experiments.maml_sinusoid and
experiments.out_of_class. float32 (sinusoid/meta regime).
"""

import os
import sys
import math
import json

import torch

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from experiments.maml_sinusoid import (
    train_fastlsq_meta,
    sample_tasks as sample_sinusoid_tasks,
)
from experiments.out_of_class import sample_triangle_tasks

torch.set_default_dtype(torch.float32)

N_SEEDS = 3
N_ITERS = 2000
N_TEST = 150          # tasks per family
K = 10
K_Q = 50
N_FEAT = 200
MU_BIG = 1e-2


# ---------------------------------------------------------------------------
# Build the Sobolev design matrix A and target t for one task, exactly as
# fastlsq_inner_solve does, but exposing A and t so we can probe conditioning
# and try alternate solvers.
# ---------------------------------------------------------------------------
def build_design(xs, ys, dys, Wb, bb, sigma, gw):
    """Return A (rows, N), t (rows,), W (1, N) for the Sobolev solve.

    Mirrors fastlsq_inner_solve(..., sobolev=True): value rows H and
    gradient rows sqrt(gw)*G stacked; targets ys and sqrt(gw)*dys stacked.
    """
    W = sigma * Wb                                    # (1, N)
    Z = xs @ W + bb                                   # (K, N)
    Nf = W.shape[1]
    H = torch.sin(Z) / math.sqrt(Nf)                  # (K, N)
    G = (W * torch.cos(Z)) / math.sqrt(Nf)            # (K, N)
    A = torch.cat([H, math.sqrt(gw) * G], dim=0)      # (2K, N)
    t = torch.cat([ys, math.sqrt(gw) * dys], dim=0)   # (2K,)
    return A, t, W


def predict(xq, W, bb, beta):
    Z = xq @ W + bb
    H = torch.sin(Z) / math.sqrt(W.shape[1])
    return H @ beta


def solve_normal(A, t, mu, N):
    """beta = (A^T A + mu I)^{-1} A^T t  (what the codebase does)."""
    AtA = A.T @ A + mu * torch.eye(N)
    Atb = A.T @ t
    return torch.linalg.solve(AtA, Atb)


def solve_tikhonov_stacked(A, t, mu, N):
    """Stabilised solve: lstsq on [A; sqrt(mu) I] vs [t; 0].

    Solves the same ridge objective min ||A b - t||^2 + mu ||b||^2 but the
    effective condition number is cond([A; sqrt(mu) I]) ~ cond(A), not cond(A)^2.
    """
    A_aug = torch.cat([A, math.sqrt(mu) * torch.eye(N)], dim=0)   # (rows+N, N)
    t_aug = torch.cat([t, torch.zeros(N)], dim=0)                 # (rows+N,)
    sol = torch.linalg.lstsq(A_aug, t_aug.unsqueeze(-1), driver="gelsd")
    return sol.solution.squeeze(-1)


def cond_normal_matrix(A, mu, N):
    """cond(A^T A + mu I) via singular values of A: (s_max^2+mu)/(s_min^2+mu)."""
    s = torch.linalg.svdvals(A)                       # length min(rows,N) = rows (rows<N)
    # A is (2K, N) with 2K < N, so A^T A + mu I has N eigenvalues:
    #   s_i^2 + mu for the 2K computed singular values, and mu (multiplicity N-2K).
    smax2 = s[0] ** 2
    smin2 = torch.tensor(0.0)   # the null-space directions contribute eigenvalue mu
    eig_max = smax2 + mu
    eig_min = smin2 + mu        # = mu (smallest eigenvalue of A^TA+muI)
    return float(eig_max / eig_min)


def pearson_loglog(cond_list, blow_list):
    c = torch.log(torch.tensor(cond_list, dtype=torch.float64))
    b = torch.log(torch.tensor(blow_list, dtype=torch.float64))
    # guard against zero variance
    if c.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    cc = c - c.mean()
    bb_ = b - b.mean()
    return float((cc @ bb_) / (cc.norm() * bb_.norm()))


def run_family(meta, sampler, seed, mu_star, mu_big):
    """For one family + seed, gather per-task cond, blowup, and per-task MSE for
    both solvers at mu_star and at mu_big."""
    gen = torch.Generator().manual_seed(seed + 9999)
    xs, ys, dys, xq, yq = sampler(N_TEST, K, K_Q, gen)
    Wb, bb = meta["Wb"], meta["bb"]
    sigma = torch.tensor(meta["sigma"])
    gw = meta["grad_w"]
    Nf = Wb.shape[1]

    rec = {
        "mu_star": mu_star, "mu_big": mu_big,
        # per-task, at mu_star
        "cond_star": [], "blow_star_normal": [], "blow_star_stack": [],
        "mse_star_normal": [], "mse_star_stack": [],
        # per-task, at mu_big
        "cond_big": [], "blow_big_normal": [],
        "mse_big_normal": [],
    }

    with torch.no_grad():
        for t in range(N_TEST):
            A, tt, W = build_design(xs[t], ys[t], dys[t], Wb, bb, sigma, gw)
            ytrue = yq[t]
            ymax = float(ytrue.abs().max()) + 1e-12

            # ----- mu_star -----
            c_star = cond_normal_matrix(A, mu_star, Nf)
            beta_n = solve_normal(A, tt, mu_star, Nf)
            beta_s = solve_tikhonov_stacked(A, tt, mu_star, Nf)
            pred_n = predict(xq[t], W, bb, beta_n)
            pred_s = predict(xq[t], W, bb, beta_s)
            blow_n = float(pred_n.abs().max()) / ymax
            blow_s = float(pred_s.abs().max()) / ymax
            mse_n = float((pred_n - ytrue).pow(2).mean())
            mse_s = float((pred_s - ytrue).pow(2).mean())

            rec["cond_star"].append(c_star)
            rec["blow_star_normal"].append(blow_n)
            rec["blow_star_stack"].append(blow_s)
            rec["mse_star_normal"].append(mse_n)
            rec["mse_star_stack"].append(mse_s)

            # ----- mu_big -----
            c_big = cond_normal_matrix(A, mu_big, Nf)
            beta_nb = solve_normal(A, tt, mu_big, Nf)
            pred_nb = predict(xq[t], W, bb, beta_nb)
            blow_nb = float(pred_nb.abs().max()) / ymax
            mse_nb = float((pred_nb - ytrue).pow(2).mean())

            rec["cond_big"].append(c_big)
            rec["blow_big_normal"].append(blow_nb)
            rec["mse_big_normal"].append(mse_nb)

    return rec


def summarize(rec):
    def med(x):
        return float(torch.tensor(x).median())
    def mean(x):
        return float(torch.tensor(x).mean())
    def mx(x):
        return float(torch.tensor(x).max())
    out = {
        "mu_star": rec["mu_star"], "mu_big": rec["mu_big"],
        "cond_star_median": med(rec["cond_star"]),
        "cond_star_max": mx(rec["cond_star"]),
        "cond_big_median": med(rec["cond_big"]),
        "cond_big_max": mx(rec["cond_big"]),
        # blowup magnitudes
        "blow_star_normal_median": med(rec["blow_star_normal"]),
        "blow_star_normal_max": mx(rec["blow_star_normal"]),
        "blow_star_stack_median": med(rec["blow_star_stack"]),
        "blow_star_stack_max": mx(rec["blow_star_stack"]),
        "blow_big_normal_median": med(rec["blow_big_normal"]),
        "blow_big_normal_max": mx(rec["blow_big_normal"]),
        # test MSE
        "mse_star_normal_mean": mean(rec["mse_star_normal"]),
        "mse_star_normal_median": med(rec["mse_star_normal"]),
        "mse_star_stack_mean": mean(rec["mse_star_stack"]),
        "mse_star_stack_median": med(rec["mse_star_stack"]),
        "mse_big_normal_mean": mean(rec["mse_big_normal"]),
        "mse_big_normal_median": med(rec["mse_big_normal"]),
        # how close are the two solvers? max abs diff in blowup and rel MSE diff
        "blow_normal_vs_stack_maxabsdiff": mx(
            [abs(a - b) for a, b in zip(rec["blow_star_normal"], rec["blow_star_stack"])]),
        # log-log correlation cond vs blowup, pooled over both mu in this family
        "corr_logcond_logblow_star": pearson_loglog(rec["cond_star"], rec["blow_star_normal"]),
        "corr_logcond_logblow_pooled": pearson_loglog(
            rec["cond_star"] + rec["cond_big"],
            rec["blow_star_normal"] + rec["blow_big_normal"]),
    }
    return out


def main():
    fams = [("in_class_sin", sample_sinusoid_tasks),
            ("ooc_triangle", sample_triangle_tasks)]

    all_seeds = []
    mu_stars = []
    for seed in range(N_SEEDS):
        meta, _t, _c = train_fastlsq_meta(
            sobolev=True, n_iters=N_ITERS, N=N_FEAT, seed=seed, log_every=10**9)
        mu_star = meta["mu"]
        mu_stars.append(mu_star)
        seed_out = {"seed": seed, "mu_star": mu_star, "sigma": meta["sigma"],
                    "grad_w": meta["grad_w"], "families": {}}
        # raw per-task records, pooled across families for global correlation
        pooled_cond, pooled_blow = [], []
        for fam_name, sampler in fams:
            rec = run_family(meta, sampler, seed, mu_star, MU_BIG)
            seed_out["families"][fam_name] = summarize(rec)
            pooled_cond += rec["cond_star"] + rec["cond_big"]
            pooled_blow += rec["blow_star_normal"] + rec["blow_big_normal"]
        seed_out["corr_logcond_logblow_global"] = pearson_loglog(pooled_cond, pooled_blow)
        all_seeds.append(seed_out)
        print(f"[seed {seed}] mu_star={mu_star:.3e} sigma={meta['sigma']:.3f} "
              f"gw={meta['grad_w']:.3f} global_corr={seed_out['corr_logcond_logblow_global']:.3f}")

    # ---- aggregate across seeds (mean +/- std of the per-seed summaries) ----
    def agg(field_path):
        # field_path is (fam, key) or ("global", key)
        vals = []
        for s in all_seeds:
            if field_path[0] == "global":
                vals.append(s[field_path[1]])
            else:
                vals.append(s["families"][field_path[0]][field_path[1]])
        tv = torch.tensor(vals, dtype=torch.float64)
        return {"mean": float(tv.mean()), "std": float(tv.std(unbiased=False)),
                "vals": [float(v) for v in vals]}

    keys = [
        "cond_star_median", "cond_star_max", "cond_big_median",
        "blow_star_normal_median", "blow_star_normal_max",
        "blow_star_stack_median", "blow_star_stack_max",
        "blow_big_normal_median", "blow_big_normal_max",
        "mse_star_normal_mean", "mse_star_normal_median",
        "mse_star_stack_mean", "mse_star_stack_median",
        "mse_big_normal_mean", "mse_big_normal_median",
        "blow_normal_vs_stack_maxabsdiff",
        "corr_logcond_logblow_star", "corr_logcond_logblow_pooled",
    ]
    aggregate = {"global_corr": agg(("global", "corr_logcond_logblow_global")),
                 "mu_star_mean": float(torch.tensor(mu_stars).mean()),
                 "mu_star_vals": mu_stars,
                 "families": {}}
    for fam_name, _ in fams:
        aggregate["families"][fam_name] = {k: agg((fam_name, k)) for k in keys}

    result = {
        "config": {"n_seeds": N_SEEDS, "n_iters": N_ITERS, "n_test": N_TEST,
                   "K": K, "K_q": K_Q, "N_feat": N_FEAT, "mu_big": MU_BIG,
                   "sobolev": True},
        "per_seed": all_seeds,
        "aggregate": aggregate,
    }

    os.makedirs(_os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    outpath = _os.path.join(_REPO_ROOT, "results", "cond_vs_extrap.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    # ---- console report ----
    print("\n================ SUMMARY (mean over 3 seeds) ================")
    print(f"mu_star (mean) = {aggregate['mu_star_mean']:.3e}   mu_big = {MU_BIG}")
    print(f"global corr[log cond, log blowup] = "
          f"{aggregate['global_corr']['mean']:.3f} +/- {aggregate['global_corr']['std']:.3f}")
    for fam_name, _ in fams:
        a = aggregate["families"][fam_name]
        print(f"\n--- {fam_name} ---")
        print(f"  cond(A^TA+muI):  mu_star median {a['cond_star_median']['mean']:.2e} "
              f"(max {a['cond_star_max']['mean']:.2e}) | "
              f"mu_big median {a['cond_big_median']['mean']:.2e}")
        print(f"  blowup mu_star:  normal median {a['blow_star_normal_median']['mean']:.2e} "
              f"(max {a['blow_star_normal_max']['mean']:.2e}) | "
              f"stacked median {a['blow_star_stack_median']['mean']:.2e} "
              f"(max {a['blow_star_stack_max']['mean']:.2e})")
        print(f"  blowup mu_big:   normal median {a['blow_big_normal_median']['mean']:.2e} "
              f"(max {a['blow_big_normal_max']['mean']:.2e})")
        print(f"  test MSE mu_star: normal mean {a['mse_star_normal_mean']['mean']:.3e} "
              f"| stacked mean {a['mse_star_stack_mean']['mean']:.3e} "
              f"| mu_big mean {a['mse_big_normal_mean']['mean']:.3e}")
        print(f"  normal-vs-stacked blowup max|diff| = "
              f"{a['blow_normal_vs_stack_maxabsdiff']['mean']:.2e}")
        print(f"  corr[log cond,log blow] (mu_star only) = "
              f"{a['corr_logcond_logblow_star']['mean']:.3f}")
    print(f"\n[saved] {outpath}")
    return result


if __name__ == "__main__":
    main()
