"""Revision campaign: multi-seed error bars + a new complementary experiment suite.

Adds, on top of the original experiments:
  * multi-seed (5x) re-runs of the headline MAML, out-of-class, and anisotropic
    results -> mean +/- std (the previous numbers were single-seed point estimates);
  * observation-noise robustness sweep, which DERIVES the deployment ridge floor
    from a noise model (the meta-learned mu* should track the noise-to-signal ratio);
  * posterior calibration: interval coverage / NLL of the closed-form Gaussian
    posterior under the unit-noise convention vs an empirical-Bayes noise scale;
  * effective-rank / identifiability study replacing the (numerically false)
    "rank ceiling -> min(M(d+1),N)" Observation with a true singular-value statement;
  * Matern / heavy-tailed RFF on the non-smooth (triangle) out-of-class family,
    the one genuine limitation, via a one-line change to the frequency draw;
  * a fair-budget frontier for the strong MLP-MAML baseline (10k/30k/60k iters).

Single-thread, serial.  Run:  python experiments/run_campaign.py
"""
import os
import sys
import json
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
torch.set_default_dtype(torch.float32)

from experiments.maml_sinusoid import (
    train_mlp_maml, eval_mlp_maml, train_fastlsq_meta, eval_fastlsq_meta,
    make_basis, sample_tasks)
from experiments.maml_strong_baseline import run as run_strong
from experiments.out_of_class import (
    sample_sinusoid_tasks, sample_tanh_tasks, sample_triangle_tasks,
    eval_meta_lsq_on_family, eval_mlp_on_family)
from experiments.anisotropic_bandwidth import train_meta as train_aniso, eval_meta as eval_aniso

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = (0, 1, 2, 3, 4)


def _ms(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    std = float(t.std(unbiased=True)) if t.numel() > 1 else 0.0
    return {"mean": float(t.mean()), "std": std, "seeds": [float(x) for x in xs]}


def save(name, obj):
    p = os.path.join(RESULTS, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {p}")


# --------------------------------------------------------------- 1. multi-seed MAML
def multiseed_maml(seeds=SEEDS):
    print("\n==== multi-seed MAML (Table 2) ====")
    keys = ["standard", "vanilla", "sobolev"]
    mse = {k: [] for k in keys}; tr = {k: [] for k in keys}; inf = {k: [] for k in keys}
    for s in seeds:
        le = 50 if s == 0 else 100000
        net, t_mlp, c_mlp = train_mlp_maml(n_iters=8000, seed=s, log_every=le, clip_grad=None)
        m_mlp, i_mlp = eval_mlp_maml(net, seed=s)
        flm, t_fl, c_fl = train_fastlsq_meta(sobolev=False, n_iters=2000, seed=s, log_every=le)
        m_fl, i_fl = eval_fastlsq_meta(flm, seed=s)
        fls, t_fls, c_fls = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=s, log_every=le)
        m_fls, i_fls = eval_fastlsq_meta(fls, seed=s)
        mse["standard"].append(m_mlp); tr["standard"].append(t_mlp); inf["standard"].append(i_mlp * 1000)
        mse["vanilla"].append(m_fl);   tr["vanilla"].append(t_fl);   inf["vanilla"].append(i_fl * 1000)
        mse["sobolev"].append(m_fls);  tr["sobolev"].append(t_fls);  inf["sobolev"].append(i_fls * 1000)
        print(f"  seed {s}: mlp={m_mlp:.4f} vanilla={m_fl:.4f} sobolev={m_fls:.4f}")
        if s == 0:
            np.savez(os.path.join(RESULTS, "maml_sinusoid.npz"),
                     curve_mlp=np.array(c_mlp, float), curve_fl=np.array(c_fl, float),
                     curve_fls=np.array(c_fls, float))
            # keep a single-seed json too for back-compat / quick reference
            save("maml_sinusoid.json", {
                "mlp": {"train_s": t_mlp, "inf_ms": i_mlp * 1000, "test_mse": m_mlp},
                "meta_lsq": {"train_s": t_fl, "inf_ms": i_fl * 1000, "test_mse": m_fl,
                             "sigma": flm["sigma"], "mu": flm["mu"], "N": 200},
                "meta_lsq_sob": {"train_s": t_fls, "inf_ms": i_fls * 1000, "test_mse": m_fls,
                                 "sigma": fls["sigma"], "mu": fls["mu"], "grad_w": fls["grad_w"], "N": 200}})
    out = {k: {"test_mse": _ms(mse[k]), "train_s": _ms(tr[k]), "inf_ms": _ms(inf[k])} for k in keys}
    # per-seed ratios vs sobolev
    out["ratio_standard_over_sobolev"] = _ms([a / b for a, b in zip(mse["standard"], mse["sobolev"])])
    save("maml_multiseed.json", out)
    return out


# --------------------------------------------------------------- 2. strong-baseline frontier
def strong_frontier(iters=(10000, 30000, 60000)):
    print("\n==== strong MLP-MAML fair-budget frontier ====")
    rows = {}
    for n in iters:
        le = 25 if n == max(iters) else max(1, n // 8)
        r = run_strong(n_iters=n, hidden=64, depth=4, inner_steps=5, first_order=True, log_every=le)
        rows[str(n)] = {"test_mse": r["test_mse"], "train_s": r["train_s"], "n_params": r["n_params"]}
        print(f"  {n} iters: test_mse={r['test_mse']:.4f}  ({r['train_s']:.0f}s)")
        if n == max(iters):
            np.savez(os.path.join(RESULTS, "maml_strong.npz"), curve=np.array(r["curve"], float))
            save("maml_strong.json", {"train_s": r["train_s"], "test_mse": r["test_mse"],
                                      "inf_ms": r["inf_ms"], "n_params": r["n_params"], "n_iters": n})
    save("strong_frontier.json", rows)
    return rows


# --------------------------------------------------------------- 3. multi-seed OOC
def multiseed_ooc(seeds=SEEDS, mu_floor=1e-2):
    print("\n==== multi-seed out-of-class (Table 3) ====")
    fams = [("in-class (sin)", sample_sinusoid_tasks), ("OOC-A (tanh)", sample_tanh_tasks),
            ("OOC-B (triangle)", sample_triangle_tasks)]
    methods = ["mlp", "meta_lsq_sob", "meta_lsq_sob_robust"]
    acc = {f: {m: {"mean": [], "median": []} for m in methods} for f, _ in fams}
    for s in seeds:
        net, _, _ = train_mlp_maml(n_iters=8000, seed=s, log_every=100000, clip_grad=None)
        fls, _, _ = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=s, log_every=100000)
        for fname, fsamp in fams:
            r_mlp = eval_mlp_on_family(net, fsamp, seed=s)
            r_sob = eval_meta_lsq_on_family(fls, fsamp, seed=s)
            r_rob = eval_meta_lsq_on_family(fls, fsamp, seed=s, mu_floor=mu_floor)
            for m, r in [("mlp", r_mlp), ("meta_lsq_sob", r_sob), ("meta_lsq_sob_robust", r_rob)]:
                acc[fname][m]["mean"].append(r["mean"]); acc[fname][m]["median"].append(r["median"])
        print(f"  seed {s} done")
    out = {f: {m: {"mean": _ms(acc[f][m]["mean"]), "median": _ms(acc[f][m]["median"])} for m in methods}
           for f, _ in fams}
    save("ooc_multiseed.json", out)
    return out


# --------------------------------------------------------------- 4. multi-seed aniso
def multiseed_aniso(seeds=SEEDS):
    print("\n==== multi-seed anisotropic bandwidth (Table 5) ====")
    params = ["scalar", "diag", "full"]
    acc = {p: [] for p in params}; nl = {}
    for s in seeds:
        for p in params:
            meta = train_aniso(p, seed=s, log_every=100000)
            mse = eval_aniso(meta, seed=s)
            acc[p].append(mse); nl[p] = meta["n_learnable"]
        print(f"  seed {s}: " + " ".join(f"{p}={acc[p][-1]:.3f}" for p in params))
    out = {p: {"test_mse": _ms(acc[p]), "n_learnable": nl[p]} for p in params}
    save("aniso_multiseed.json", out)
    return out


# --------------------------------------------------------------- 5. noise robustness
def noise_sweep(noises=(0.0, 0.1, 0.25, 0.5), seeds=(0, 1, 2), grad_noise=0.0, n_iters=8000,
                mlp_from=None):
    """``grad_noise``: std of the noise on the support DERIVATIVES. 0.0 (default)
    = noise on the support values only, derivatives and query clean (the protocol
    the paper describes). None reproduces the original sweep, which corrupted the
    support derivatives with the same std as the values. The y-noise draws and the
    tasks are identical in both modes (see sample_tasks).

    ``n_iters``: Meta-LSQ meta-training steps. 2000 (the benchmark budget) leaves w far
    from converged: it starts at 1 and climbs more slowly under value noise, so at 2000
    steps it looked as if w fell with noise. At 8000 steps it rises, as the kernel
    reading predicts (results/superseded/noise_sweep.2000steps.json has the old run).
    ``mlp_from``: path of an earlier noise_sweep.json whose MLP-MAML column is reused
    (MLP-MAML does not depend on n_iters; it is 8000 steps either way)."""
    print("\n==== observation-noise robustness (derives the ridge floor) ====")
    mode = "values_and_derivatives" if grad_noise is None else (
        "values_only" if grad_noise == 0.0 else "values_and_derivatives_separate_std")
    rows = []
    for nz in noises:
        mus = []; sigmas = []; gws = []; mse_sob = []; mse_mlp = []
        for s in seeds:
            fls = train_fastlsq_meta(sobolev=True, n_iters=n_iters, seed=s, noise=nz, grad_noise=grad_noise,
                                     log_every=100000)[0]
            mus.append(fls["mu"]); sigmas.append(fls["sigma"]); gws.append(fls["grad_w"])
            mse_sob.append(eval_fastlsq_meta(fls, seed=s, noise=nz, grad_noise=grad_noise)[0])
            if mlp_from is None:
                net = train_mlp_maml(n_iters=8000, seed=s, noise=nz, grad_noise=grad_noise,
                                     log_every=100000, clip_grad=None)[0]
                mse_mlp.append(eval_mlp_maml(net, seed=s, noise=nz, grad_noise=grad_noise)[0])
        if mlp_from is not None:
            prev = {r["noise"]: r for r in json.load(open(mlp_from))}[nz]
            assert prev["noise_mode"] == mode and len(prev["mlp_mse"]["seeds"]) == len(seeds)
            mse_mlp = prev["mlp_mse"]["seeds"]
        rows.append({"noise": nz, "noise_mode": mode, "grad_noise": grad_noise, "n_iters": n_iters,
                     "mu_star": _ms(mus), "sigma": _ms(sigmas), "grad_w": _ms(gws),
                     "meta_lsq_sob_mse": _ms(mse_sob), "mlp_mse": _ms(mse_mlp)})
        print(f"  noise={nz} ({mode}): mu*={np.mean(mus):.2e}  w={np.mean(gws):.2f}  sigma={np.mean(sigmas):.3f}  "
              f"sob_mse={np.mean(mse_sob):.4f}  mlp_mse={np.mean(mse_mlp):.4f}")
    save("noise_sweep.json", rows)
    return rows


# --------------------------------------------------------------- 6. posterior calibration
def posterior_calibration(noises=(0.0, 0.1, 0.25), seeds=(0, 1, 2), N=200, n_test=300):
    print("\n==== posterior calibration (coverage of the closed-form Gaussian posterior) ====")
    levels = {"50": 0.6745, "68": 1.0, "90": 1.6449, "95": 1.9600}
    convs = ["unit", "bo_code", "emp_bayes"]
    out = []
    for nz in noises:
        cov = {c: {L: [] for L in levels} for c in convs}; nll = {c: [] for c in convs}
        for s in seeds:
            meta = train_fastlsq_meta(sobolev=False, n_iters=2000, seed=s, noise=nz, log_every=100000)[0]
            sigma = torch.tensor(meta["sigma"]); mu = float(meta["mu"]); Wb = meta["Wb"]; bb = meta["bb"]
            W = sigma * Wb
            gen = torch.Generator().manual_seed(s + 9999)
            xs, ys, dys, xq, yq = sample_tasks(n_test, 10, 50, gen, noise=nz)
            zs = {c: [] for c in convs}
            with torch.no_grad():
                for t in range(n_test):
                    Z = xs[t] @ W + bb; H = torch.sin(Z) / math.sqrt(N)
                    K = H.T @ H + mu * torch.eye(N)
                    L = torch.linalg.cholesky(K)
                    beta = torch.cholesky_solve((H.T @ ys[t]).unsqueeze(-1), L).squeeze(-1)
                    Zq = xq[t] @ W + bb; Hq = torch.sin(Zq) / math.sqrt(N)
                    mean = Hq @ beta
                    KiHqT = torch.cholesky_solve(Hq.T, L)             # (N, Mq)
                    quad = (Hq * KiHqT.T).sum(1).clamp_min(0)         # h^T K^-1 h
                    resid_var = ((H @ beta - ys[t]) ** 2).mean().clamp_min(1e-12)
                    scales = {"unit": 1.0, "bo_code": mu, "emp_bayes": float(resid_var)}
                    for c in convs:
                        sd = (scales[c] * quad + 1e-12).sqrt()
                        z = (yq[t] - mean) / sd
                        zs[c].append(z)
            for c in convs:
                z = torch.cat(zs[c])           # standardized residuals (y - mean)/sd
                for L, zthr in levels.items():
                    cov[c][L].append(float((z.abs() <= zthr).float().mean()))
                # Gaussian NLL with the standardized residual z and its sd implicit:
                # 0.5*z^2 + 0.5*log(2*pi) + log(sd); but sd is folded into z, so we
                # report the calibration-invariant 0.5*z^2 mean and the |z| std instead.
                nll[c].append(float((0.5 * z ** 2 + 0.5 * math.log(2 * math.pi)).mean()))
        entry = {"noise": nz}
        for c in convs:
            entry[c] = {f"cov{L}": _ms(cov[c][L]) for L in levels}
            entry[c]["half_chi2"] = _ms(nll[c])
        out.append(entry)
        print(f"  noise={nz}: unit cov90={np.mean(cov['unit']['90']):.2f}  "
              f"bo cov90={np.mean(cov['bo_code']['90']):.2f}  eb cov90={np.mean(cov['emp_bayes']['90']):.2f}")
    save("calibration.json", out)
    return out


# --------------------------------------------------------------- 7. effective rank
def effective_rank(sigmas=(0.25, 0.56, 1.0, 2.0, 5.0), M=150, N=200, seed=0):
    """SUPERSEDED by effective_rank_k10 and no longer in the campaign step list.

    Kept only for provenance. It is numerically invalid: it runs in the module-wide
    float32, solves the normal equations at mu=1e-8 (cond ~1e8, beyond float32), so
    its recover_err_* are round-off; and it measures rank on M=150 points instead of
    the K=10 support set. Writes to results/superseded/ so it cannot clobber the
    corrected results/eff_rank.json."""
    print("\n==== [SUPERSEDED] float32 M=150 effective-rank study ====")
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(M, 1, generator=g) * 10 - 5)
    Wb, bb = make_basis(N=N, seed=seed)
    # identifiability target: a known in-span beta_true, M_id small so M<N
    M_id = 30
    x_id = (torch.rand(M_id, 1, generator=g) * 10 - 5)
    beta_true = torch.randn(N, generator=g) / math.sqrt(N)
    rows = []
    for sg in sigmas:
        W = sg * Wb
        Z = x @ W + bb; H = torch.sin(Z) / math.sqrt(N); G = (W * torch.cos(Z)) / math.sqrt(N)
        sv_H = torch.linalg.svdvals(H); sv_S = torch.linalg.svdvals(torch.cat([H, G], 0))
        er = lambda sv, tol: int((sv > tol * sv[0]).sum())
        # identifiability recovery on M_id<N
        Zi = x_id @ W + bb; Hi = torch.sin(Zi) / math.sqrt(N); Gi = (W * torch.cos(Zi)) / math.sqrt(N)
        y_id = Hi @ beta_true; g_id = Gi @ beta_true
        mu = 1e-8
        b_val = torch.linalg.solve(Hi.T @ Hi + mu * torch.eye(N), Hi.T @ y_id)
        A = torch.cat([Hi, Gi], 0); t = torch.cat([y_id, g_id], 0)
        b_sob = torch.linalg.solve(A.T @ A + mu * torch.eye(N), A.T @ t)
        rows.append({"sigma": sg,
                     "eff_rank_H_1e6": er(sv_H, 1e-6), "eff_rank_stacked_1e6": er(sv_S, 1e-6),
                     "eff_rank_H_1e10": er(sv_H, 1e-10), "eff_rank_stacked_1e10": er(sv_S, 1e-10),
                     "recover_err_valonly": float((b_val - beta_true).norm()),
                     "recover_err_sobolev": float((b_sob - beta_true).norm())})
        print(f"  sigma={sg}: effrank H={er(sv_H,1e-6)} stacked={er(sv_S,1e-6)} (M={M},N={N}); "
              f"recover val={rows[-1]['recover_err_valonly']:.3f} sob={rows[-1]['recover_err_sobolev']:.3f}")
    save(os.path.join("superseded", "eff_rank_float32_M150_rerun.json"),
         {"M": M, "N": N, "M_id": M_id, "rows": rows})
    return rows


def _ridge_lstsq(A, t, mu):
    """Stable ridge: argmin ||A b - t||^2 + mu ||b||^2 as an ordinary least-squares
    problem on the augmented, full-column-rank system [A; sqrt(mu) I] b = [t; 0]."""
    n = A.shape[1]
    Aa = torch.cat([A, math.sqrt(mu) * torch.eye(n, dtype=A.dtype)], 0)
    ta = torch.cat([t, torch.zeros(n, dtype=t.dtype)], 0)
    return torch.linalg.lstsq(Aa, ta.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)


def _ridge_normal(A, t, mu):
    n = A.shape[1]
    return torch.linalg.solve(A.T @ A + mu * torch.eye(n, dtype=A.dtype), A.T @ t)


def effective_rank_k10(sigmas=(0.25, 0.4, 0.56, 0.65, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0), K=10, N=200,
                       draws=200, n_test=200, ws=(1.0, 10.9), mus=(1e-8, 1e-5), tols=(1e-6, 1e-10),
                       M_old=150, M_id_old=30):
    """Effective rank / identifiability at the K=10 support set, in float64.

    Replaces the float32 M=150 study (effective_rank). Per sigma, `draws` independent
    draws, each with a fresh frozen basis (make_basis seed 10000+d) and fresh support /
    held-out points and target (generator seed 20000+d); the same draws are reused
    across sigma (common random numbers). For each draw:
      * singular values of H (K x N) and of A_w = [H; sqrt(w) G] (2K x N);
        effective rank = #{s_i > tol * s_1}; ridge degrees of freedom
        df(mu) = sum s_i^2 / (s_i^2 + mu); restricted condition s_1 / s_r (r = K, 2K);
      * function-space identifiability: in-span target beta_true ~ N(0, I/N),
        observe f = H beta_true (and f' = G beta_true) at the K support points, fit
        value-only and stacked ridge at each mu (lstsq on [A; sqrt(mu) I]), report the
        held-out RMSE of f and f' on n_test points in [-5, 5] (absolute and relative
        to the RMS of the true f / f'), and ||beta_hat - beta_true||.
    The lstsq solve is cross-checked against a float64 normal-equations solve; the
    agreement is asserted where the normal matrix is well conditioned (mu = 1e-5).
    Also recomputes the old M=150 single-draw rank measurement (same float32 draws as
    effective_rank, arithmetic in float64) for continuity."""
    print("\n==== effective-rank / identifiability at K=10, float64 (replaces the M=150 float32 study) ====")
    # ---- continuity: the old M=150 recipe, draws taken exactly as before (float32 RNG) ----
    g = torch.Generator().manual_seed(0)
    x_old = (torch.rand(M_old, 1, generator=g) * 10 - 5)
    Wb_old, bb_old = make_basis(N=N, seed=0)
    x_id_old = (torch.rand(M_id_old, 1, generator=g) * 10 - 5)
    beta_true_old = torch.randn(N, generator=g) / math.sqrt(N)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dt = torch.float64
        er = lambda sv, tol: int((sv > tol * sv[0]).sum())
        feats = lambda x, W, bb: (torch.sin(x @ W + bb) / math.sqrt(N), (W * torch.cos(x @ W + bb)) / math.sqrt(N))
        rms = lambda v: float(v.pow(2).mean().sqrt())

        cont = []
        for sg in sigmas:
            row = {"sigma": sg}
            for prec, tdt in (("float64", torch.float64), ("float32", torch.float32)):
                W = sg * Wb_old.to(tdt); bb = bb_old.to(tdt)
                H, G = feats(x_old.to(tdt), W, bb)
                sv_H = torch.linalg.svdvals(H); sv_S = torch.linalg.svdvals(torch.cat([H, G], 0))
                for tol in tols:
                    row[f"{prec}_eff_rank_H_{tol:.0e}"] = er(sv_H, tol)
                    row[f"{prec}_eff_rank_stacked_w1_{tol:.0e}"] = er(sv_S, tol)
            W = sg * Wb_old.to(dt); bb = bb_old.to(dt)
            Hi, Gi = feats(x_id_old.to(dt), W, bb); bt = beta_true_old.to(dt)
            b_val = _ridge_lstsq(Hi, Hi @ bt, 1e-8)
            b_sob = _ridge_lstsq(torch.cat([Hi, Gi], 0), torch.cat([Hi @ bt, Gi @ bt], 0), 1e-8)
            row["float64_recover_err_valonly_M30_mu1e-8"] = float((b_val - bt).norm())
            row["float64_recover_err_stacked_w1_M30_mu1e-8"] = float((b_sob - bt).norm())
            row["norm_beta_true"] = float(bt.norm())
            cont.append(row)
            print(f"  [M=150 continuity] sigma={sg}: f64 effrank H={row['float64_eff_rank_H_1e-06']} "
                  f"stacked={row['float64_eff_rank_stacked_w1_1e-06']} (1e-10: {row['float64_eff_rank_H_1e-10']}/"
                  f"{row['float64_eff_rank_stacked_w1_1e-10']}; f32 1e-10: {row['float32_eff_rank_H_1e-10']}/"
                  f"{row['float32_eff_rank_stacked_w1_1e-10']}); f64 recover val={row['float64_recover_err_valonly_M30_mu1e-8']:.3f} "
                  f"stacked={row['float64_recover_err_stacked_w1_M30_mu1e-8']:.3f}")

        # ---- main study: K=10 support set, many draws ----
        fits = ["valonly"] + [f"stacked_w{w:g}" for w in ws]
        max_rel = {f"{mu:.0e}": 0.0 for mu in mus}
        out_rows = []; detail = []
        for sg in sigmas:
            acc = {}
            add = lambda k, v: acc.setdefault(k, []).append(v)
            for d in range(draws):
                Wb, bb = make_basis(N=N, seed=10000 + d)
                assert Wb.dtype == dt
                gd = torch.Generator().manual_seed(20000 + d)
                x = torch.rand(K, 1, generator=gd) * 10 - 5
                xt = torch.rand(n_test, 1, generator=gd) * 10 - 5
                bt = torch.randn(N, generator=gd) / math.sqrt(N)
                W = sg * Wb
                H, G = feats(x, W, bb); Ht, Gt = feats(xt, W, bb)
                f_s, df_s = H @ bt, G @ bt
                f_t, df_t = Ht @ bt, Gt @ bt
                systems = {"valonly": (H, f_s)}
                for w in ws:
                    sw = math.sqrt(w)
                    systems[f"stacked_w{w:g}"] = (torch.cat([H, sw * G], 0), torch.cat([f_s, sw * df_s], 0))
                for name, (A, t) in systems.items():
                    sv = torch.linalg.svdvals(A); r = A.shape[0]
                    for tol in tols:
                        add(f"eff_rank_{name}_{tol:.0e}", er(sv, tol))
                    add(f"log10_cond_restricted_{name}", float(torch.log10(sv[0] / sv[r - 1])))
                    if name != "valonly":
                        add(f"log10_cond_topK_{name}", float(torch.log10(sv[0] / sv[K - 1])))
                    add(f"sv_rel_{name}", (sv / sv[0]).tolist())
                    for mu in mus:
                        mk = f"{mu:.0e}"
                        add(f"df_{name}_mu{mk}", float((sv ** 2 / (sv ** 2 + mu)).sum()))
                        b = _ridge_lstsq(A, t, mu)
                        b_ne = _ridge_normal(A, t, mu)
                        rel = float((b_ne - b).norm() / b.norm())
                        max_rel[mk] = max(max_rel[mk], rel)
                        if mu >= 1e-5:
                            assert rel < 1e-6, (sg, d, name, mu, rel)
                        add(f"rmse_val_{name}_mu{mk}", rms(Ht @ b - f_t))
                        add(f"rmse_der_{name}_mu{mk}", rms(Gt @ b - df_t))
                        add(f"relrmse_val_{name}_mu{mk}", rms(Ht @ b - f_t) / rms(f_t))
                        add(f"relrmse_der_{name}_mu{mk}", rms(Gt @ b - df_t) / rms(df_t))
                        add(f"beta_err_{name}_mu{mk}", float((b - bt).norm()))
                add("norm_beta_true", float(bt.norm()))
                add("rms_f_true", rms(f_t)); add("rms_df_true", rms(df_t))

            def st(k):
                v = torch.tensor(acc[k], dtype=dt)
                return {"mean": float(v.mean()), "std": float(v.std(unbiased=True)), "median": float(v.median())}
            row = {"sigma": sg}
            for k in acc:
                if k.startswith("sv_rel_"):
                    sv = torch.tensor(acc[k], dtype=dt)
                    row[f"median_{k}"] = torch.log10(sv).median(0).values.tolist()
                else:
                    row[k] = st(k)
            detail.append(row)
            wm = f"stacked_w{ws[-1]:g}"
            out_rows.append({
                "sigma": sg,
                # flat summary (means over draws) for downstream consumers; stacked = largest w
                "eff_rank_H_1e6": row["eff_rank_valonly_1e-06"]["mean"],
                "eff_rank_stacked_1e6": row[f"eff_rank_{wm}_1e-06"]["mean"],
                "eff_rank_H_1e10": row["eff_rank_valonly_1e-10"]["mean"],
                "eff_rank_stacked_1e10": row[f"eff_rank_{wm}_1e-10"]["mean"],
                "recover_err_valonly": row["beta_err_valonly_mu1e-05"]["mean"],
                "recover_err_sobolev": row[f"beta_err_{wm}_mu1e-05"]["mean"],
                "rmse_val_valonly": row["rmse_val_valonly_mu1e-05"]["mean"],
                "rmse_val_sobolev": row[f"rmse_val_{wm}_mu1e-05"]["mean"],
                "rmse_der_valonly": row["rmse_der_valonly_mu1e-05"]["mean"],
                "rmse_der_sobolev": row[f"rmse_der_{wm}_mu1e-05"]["mean"]})
            print(f"  sigma={sg}: effrank(1e-6) H={row['eff_rank_valonly_1e-06']['mean']:.2f} "
                  f"stacked w1={row['eff_rank_stacked_w1_1e-06']['mean']:.2f} {wm}={row[f'eff_rank_{wm}_1e-06']['mean']:.2f} | "
                  f"(1e-10) H={row['eff_rank_valonly_1e-10']['mean']:.2f} {wm}={row[f'eff_rank_{wm}_1e-10']['mean']:.2f} | "
                  f"relRMSE val: valonly={row['relrmse_val_valonly_mu1e-05']['median']:.3f} "
                  f"{wm}={row[f'relrmse_val_{wm}_mu1e-05']['median']:.3f} (median, mu=1e-5)")
    finally:
        torch.set_default_dtype(prev_dtype)

    save("eff_rank.json", {
        "_note": ("Replaces the float32 M=150 study (previous file: results/superseded/eff_rank.json). "
                  "That study ran in float32 and solved the normal equations at mu=1e-8 (cond ~1e8), so its "
                  "coefficient-recovery errors were round-off (in float64 both are ~||beta_true||), and it "
                  "measured rank on M=150 points instead of the K=10 support set. This file: float64, "
                  "stable lstsq ridge, K=10, many independent draws, identifiability in function space."),
        "_schema": {
            "config": "settings of this study",
            "rows": ("flat per-sigma summary, MEANS over draws at K=10: eff_rank_H_* = value block H, "
                     "eff_rank_stacked_* = [H; sqrt(w) G] with w=%g; recover_err_* = ||beta_hat-beta_true|| "
                     "and rmse_{val,der}_* = held-out RMSE of f / f', all at mu=1e-5; '_sobolev' = stacked w=%g" % (ws[-1], ws[-1])),
            "detail": ("per sigma, per fit in {valonly, stacked_w<w>}: {mean,std,median} over draws of "
                       "eff_rank_<fit>_<tol>, df_<fit>_mu<mu> (ridge degrees of freedom sum s^2/(s^2+mu)), "
                       "log10_cond_restricted_<fit> (log10 s_1/s_r, r = #rows), log10_cond_topK_<fit> "
                       "(log10 s_1/s_K, stacked only), rmse_{val,der}_<fit>_mu<mu>, relrmse_{val,der}_<fit>_mu<mu> "
                       "(RMSE / RMS of the true f or f' on the held-out points), beta_err_<fit>_mu<mu>; "
                       "median_sv_rel_<fit> = per-index median of log10(s_i/s_1)"),
            "continuity_M150": ("old recipe (seed 0, M=150 rank points, M_id=30 recovery points, stacked "
                                "unweighted [H; G]) recomputed with the original float32 draws: ranks in "
                                "float64 and float32, recovery in float64 via lstsq at mu=1e-8"),
            "solver_check": "max over all fits/draws of ||b_normal_eq - b_lstsq|| / ||b_lstsq|| per mu",
        },
        "config": {"dtype": "float64", "solver": "torch.linalg.lstsq(driver='gelsd') on [A; sqrt(mu) I]",
                   "solver_check": "float64 normal equations (torch.linalg.solve), asserted rel diff < 1e-6 at mu >= 1e-5",
                   "K": K, "N": N, "d": 1, "x_range": [-5, 5], "draws": draws, "n_test": n_test,
                   "sigmas": list(sigmas), "tolerances": list(tols), "w_values": list(ws), "mu_values": list(mus),
                   "beta_true": "N(0, I/N), in span", "basis": "make_basis(N, seed=10000+d), W = sigma*Wb",
                   "points": "torch.Generator seed 20000+d: K support, n_test held-out, then beta_true",
                   "common_random_numbers_across_sigma": True},
        "rows": out_rows,
        "detail": detail,
        "continuity_M150": {"M": M_old, "M_id": M_id_old, "N": N, "seed": 0, "rows": cont},
        "solver_check": {"max_rel_diff_normal_vs_lstsq": max_rel},
    })
    return out_rows


# --------------------------------------------------------------- 8. Matern / heavy-tail RFF OOC
def matern_ooc(seeds=(0, 1, 2), mu_floor=1e-2):
    print("\n==== Matern / heavy-tailed RFF on out-of-class (attacks the triangle limit) ====")
    configs = [("normal", 0), ("studentt", 2), ("studentt", 5), ("laplace", 0)]
    fams = [("OOC-A (tanh)", sample_tanh_tasks), ("OOC-B (triangle)", sample_triangle_tasks),
            ("in-class (sin)", sample_sinusoid_tasks)]
    out = {}
    for dist, nu in configs:
        tag = dist if dist != "studentt" else f"studentt_nu{nu}"
        acc = {f: {"mean": [], "median": []} for f, _ in fams}
        for s in seeds:
            fls = train_fastlsq_meta(sobolev=True, n_iters=2000, seed=s, dist=dist, nu=nu, log_every=100000)[0]
            for fname, fsamp in fams:
                r = eval_meta_lsq_on_family(fls, fsamp, seed=s, mu_floor=mu_floor)
                acc[fname]["mean"].append(r["mean"]); acc[fname]["median"].append(r["median"])
        out[tag] = {f: {"mean": _ms(acc[f]["mean"]), "median": _ms(acc[f]["median"])} for f, _ in fams}
        print(f"  {tag}: triangle mean={np.mean(acc['OOC-B (triangle)']['mean']):.2f} "
              f"median={np.mean(acc['OOC-B (triangle)']['median']):.3f}; "
              f"tanh mean={np.mean(acc['OOC-A (tanh)']['mean']):.3f}")
    save("matern_ooc.json", out)
    return out


if __name__ == "__main__":
    import sys as _s
    only = _s.argv[1:] if len(_s.argv) > 1 else None
    t0 = time.time()
    steps = [("maml", multiseed_maml), ("strong", strong_frontier), ("ooc", multiseed_ooc),
             ("aniso", multiseed_aniso), ("noise", noise_sweep), ("calib", posterior_calibration),
             ("effrank", effective_rank_k10), ("matern", matern_ooc)]
    for name, fn in steps:
        if only and name not in only:
            continue
        ts = time.time()
        fn()
        print(f"[{name}] done in {time.time()-ts:.0f}s (total {time.time()-t0:.0f}s)\n")
    print(f"CAMPAIGN COMPLETE in {time.time()-t0:.0f}s")
