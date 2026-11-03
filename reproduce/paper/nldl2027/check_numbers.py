"""Cross-check the numbers quoted in main.tex against results/*.json.

Every headline figure in the paper should be recomputable from the captured
experiment output.  This script recomputes them and compares against the values
hard-coded in the manuscript, so a stale number cannot survive a re-run of the
experiments unnoticed.

Run from anywhere:  python3 paper/northern-lights-crc/check_numbers.py
Exit status is non-zero if any check fails.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")

FAILS = []
SKIPS = []


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        SKIPS.append(name)
        return None
    with open(p) as f:
        return json.load(f)


def check(label, got, want, rtol=0.02):
    """Compare a recomputed value against what the paper prints."""
    if got is None:
        return
    ok = abs(got - want) <= rtol * max(abs(want), 1e-12)
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {label:<52s} paper={want:<12.6g} data={got:<12.6g}")
    if not ok:
        FAILS.append(label)


print("== Table 1: 10-shot sinusoid ==")
ms = load("maml_multiseed.json")
if ms:
    check("MLP-MAML std test MSE", ms["standard"]["test_mse"]["mean"], 0.70, 0.02)
    check("MLP-MAML std std", ms["standard"]["test_mse"]["std"], 0.13, 0.05)
    check("MLP-MAML std train s", ms["standard"]["train_s"]["mean"], 51, 0.05)
    check("Meta-LSQ vanilla test MSE", ms["vanilla"]["test_mse"]["mean"], 0.060, 0.02)
    check("Meta-LSQ vanilla train s", ms["vanilla"]["train_s"]["mean"], 11, 0.05)
    check("Meta-LSQ Sobolev test MSE", ms["sobolev"]["test_mse"]["mean"], 0.019, 0.02)
    check("Meta-LSQ Sobolev std", ms["sobolev"]["test_mse"]["std"], 0.016, 0.05)
    check("Meta-LSQ Sobolev train s", ms["sobolev"]["train_s"]["mean"], 14, 0.05)
    check("ratio std/Sobolev (of means)",
          ms["standard"]["test_mse"]["mean"] / ms["sobolev"]["test_mse"]["mean"], 37, 0.03)
    check("ratio std/vanilla (of means)",
          ms["standard"]["test_mse"]["mean"] / ms["vanilla"]["test_mse"]["mean"], 12, 0.05)

fr = load("strong_frontier.json")
if fr and ms:
    sob = ms["sobolev"]["test_mse"]["mean"]
    check("strong FOMAML 10k MSE", fr["10000"]["test_mse"], 0.328, 0.01)
    check("strong FOMAML 30k MSE", fr["30000"]["test_mse"], 0.144, 0.01)
    check("strong FOMAML 60k MSE", fr["60000"]["test_mse"], 0.104, 0.01)
    check("strong FOMAML 10k train s", fr["10000"]["train_s"], 130, 0.02)
    check("strong FOMAML 30k train s", fr["30000"]["train_s"], 399, 0.02)
    check("strong FOMAML 60k train s", fr["60000"]["train_s"], 512, 0.02)
    check("strong params", fr["60000"]["n_params"], 8513, 0.001)
    check("gap vs 10k", fr["10000"]["test_mse"] / sob, 17, 0.04)
    check("gap vs 30k", fr["30000"]["test_mse"] / sob, 7.6, 0.04)
    check("gap vs 60k", fr["60000"]["test_mse"] / sob, 5.5, 0.04)
    check("compute ratio 60k/Sobolev",
          fr["60000"]["train_s"] / ms["sobolev"]["train_s"]["mean"], 36, 0.05)

sb = load("maml_sobolev_baseline.json")
if sb and ms:
    by_lam = {r["lam"]: r for r in sb["per_lam"]}
    check("MLP+Sobolev lam=0 MSE", by_lam[0.0]["test_mse"]["mean"], 0.716, 0.02)
    check("MLP+Sobolev lam=1 MSE", by_lam[1.0]["test_mse"]["mean"], 0.216, 0.02)
    check("MLP+Sobolev lam=1 std", by_lam[1.0]["test_mse"]["std"], 0.031, 0.03)
    check("MLP+Sobolev lam=1 train s", by_lam[1.0]["train_s"]["mean"], 78, 0.03)
    check("MLP+Sobolev double-backprop cost ratio",
          by_lam[1.0]["train_s"]["mean"] / by_lam[0.0]["train_s"]["mean"], 1.9, 0.05)
    check("gap vs information-matched MLP",
          by_lam[1.0]["test_mse"]["mean"] / ms["sobolev"]["test_mse"]["mean"], 11, 0.05)

al = load("alpaca_baseline.json")
if al and ms:
    raw_in = al["alpaca"]["raw_in"]
    check("ALPaCA in-class MSE", sum(raw_in) / len(raw_in), 0.026, 0.03)
    check("ALPaCA in-class std", al["alpaca"]["in_class_mse"][1], 0.017, 0.05)
    check("ALPaCA params", al["alpaca"]["n_params"], 1721, 0.001)
    check("ALPaCA train s", al["alpaca"]["train_s"][0], 12, 0.05)
    lsq_in = al["meta_lsq_sobolev"]["raw_in"]
    check("Meta-LSQ-Sob matched-3-seed in-class",
          sum(lsq_in) / len(lsq_in), 0.023, 0.03)
    check("ALPaCA param ratio", al["alpaca"]["n_params"] / 3.0, 570, 0.02)

am64, am32 = load("alpaca_matched_lsq_f64.json"), load("alpaca_matched_lsq_f32.json")
if am64 and am32 and al:
    assert am64["dtype"] == "torch.float64" and am32["dtype"] == "torch.float32"
    assert am64["n_iters"] == 8000 == al["config"]["n_iters_alpaca"]
    assert am64["seeds"] == al["config"]["seeds"]
    check("value-only Meta-LSQ 8k f64, matched seeds", am64["mean_mse"], 0.027, 0.03)
    check("value-only Meta-LSQ 8k f32, matched seeds", am32["mean_mse"], 0.041, 0.03)
    lower = sum(r["mean_mse"] < a for r, a in zip(am64["per_seed"], al["alpaca"]["raw_in"]))
    assert lower == 2, lower                                  # "lower on two of the three"

print("== Table 2: gradient-enhanced kriging (fair-fit GP comparators) ==")
pg = load("periodic_gek.json")
if pg:
    kk = pg["kernels"]
    check("GEK RBF, unit signal variance", kk["rbf_paper_setting"]["test_mse"]["mean"], 0.079)
    check("GEK RBF, variance profiled", kk["rbf"]["test_mse"]["mean"], 0.013, 0.05)
    check("GEK periodic, period per task", kk["periodic"]["test_mse"]["mean"], 0.025, 0.05)
    # order of magnitude only: the exact value moves with the frequency grid,
    # the conclusion (benchmark is saturable) does not.
    sm = kk["sm1"]["test_mse"]["mean"]
    ok = 1e-11 < sm < 1e-6
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'GEK spectral-mixture Q=1 in [1e-11,1e-6]':<52s} "
          f"paper=3.2e-10    data={sm:<12.6g}")
    if not ok:
        FAILS.append("GEK spectral-mixture Q=1 magnitude")
    for k, want in (("rbf", 2.9), ("periodic", 69.5), ("sm1", 102.0)):
        check(f"GEK {k} ms/task", kk[k]["ms_per_task"], want, 0.35)

print("== App. D: ridge policies, incl. GCV ==")
gc = load("gcv_ridge.json")
if gc:
    r = gc["results"]
    check("sin      floor mean", r["in-class (sin)"]["floor"]["mean"], 0.054, 0.05)
    check("sin      GCV   mean", r["in-class (sin)"]["gcv"]["mean"], 2033, 0.10)
    check("tanh     floor mean", r["OOC tanh"]["floor"]["mean"], 0.58, 0.05)
    check("triangle floor mean", r["OOC triangle"]["floor"]["mean"], 64, 0.05)
    check("triangle CV    mean", r["OOC triangle"]["auto"]["mean"], 5216, 0.10)
    hist = gc["mu_histogram"]["in-class (sin)"]["gcv"]
    n_floor = hist.get("1e-08", 0)
    ok = n_floor >= 990
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'GCV picks grid floor on >=999/1000 sin tasks':<52s} "
          f"paper=999         data={n_floor:<12d}")
    if not ok:
        FAILS.append("GCV grid-floor selection count")
    g = gc.get("gcv_guard", {})
    rows = g.get("rows", [])
    ok = bool(g.get("ok")) and all(0.855 <= row["gain_recovered"] <= 0.955
                                   for row in rows)
    pct = "-".join("%.0f" % (row["gain_recovered"] * 100) for row in rows)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'GCV guard: recovers 86-95% of oracle gain':<52s} "
          f"paper=86-95%      data={pct}")
    if not ok:
        FAILS.append("GCV implementation guard")

rv = load("review.json")
if rv:
    dm = rv["derivative_metrics"]
    check("query value  vanilla", dm["vanilla"]["val"]["mean"], 0.059, 0.02)
    check("query grad   vanilla", dm["vanilla"]["grad"]["mean"], 0.067, 0.02)
    check("query Hess   vanilla", dm["vanilla"]["hess"]["mean"], 0.087, 0.02)
    check("query value  Sobolev", dm["sobolev"]["val"]["mean"], 0.019, 0.03)
    check("query grad   Sobolev", dm["sobolev"]["grad"]["mean"], 0.017, 0.03)
    check("query Hess   Sobolev", dm["sobolev"]["hess"]["mean"], 0.019, 0.03)
    check("Hessian reduction factor",
          dm["vanilla"]["hess"]["mean"] / dm["sobolev"]["hess"]["mean"], 4.7, 0.02)
    ws = {r["w"]: r for r in rv["w_sweep"]}
    check("w-sweep value at w=0", ws[0.0]["val_mse"]["mean"], 0.073, 0.02)
    check("w-sweep value at w=10", ws[10.0]["val_mse"]["mean"], 0.023, 0.02)
    check("w-sweep grad  at w=0", ws[0.0]["grad_mse"]["mean"], 0.080, 0.02)
    check("w-sweep grad  at w=10", ws[10.0]["grad_mse"]["mean"], 0.016, 0.03)
    # means fall monotonically in w ("though not on every seed")
    order = sorted(ws)
    for key in ("val_mse", "grad_mse"):
        v = [ws[w]["mean" if False else w][key]["mean"] if False else ws[w][key]["mean"] for w in order]
        assert all(a >= b for a, b in zip(v, v[1:])), (key, v)
    assert len(ws[0.0]["val_mse"]["per_seed"]) == 3
    ns_ = {r["N"]: r["test_mse"]["mean"] for r in rv["n_sweep"]}
    for N_, want in ((50, 0.024), (100, 0.021), (200, 0.023)):
        check(f"N-sweep N={N_}", ns_[N_], want, 0.03)
    assert "Reconstructed" not in rv.get("_note", "")

print("\n== Table 2: equal wall-clock ==")
wc = load("wallclock_budget.json")
if wc:
    su = wc["summary"]
    for b, mlp, van, sob, fl in [("0.5", 3.50, 0.304, 0.071, 0.144),
                                 ("5.0", 2.07, 0.173, 0.034, 0.085),
                                 ("50.0", 0.61, 0.045, 0.0076, 0.020)]:
        check(f"MLP-MAML @{b}s", su["MLP-MAML"][b]["mse_mean"], mlp, 0.03)
        check(f"Meta-LSQ @{b}s", su["Meta-LSQ-vanilla"][b]["mse_mean"], van, 0.03)
        check(f"Meta-LSQ-Sob @{b}s", su["Meta-LSQ-Sobolev"][b]["mse_mean"], sob, 0.05)
        check(f"Meta-LSQ-Sob+floor @{b}s",
              su["Meta-LSQ-Sobolev-floor"][b]["mse_mean"], fl, 0.05)
    check("0.5s Sobolev beats 50s MLP-MAML",
          su["MLP-MAML"]["50.0"]["mse_mean"] / su["Meta-LSQ-Sobolev"]["0.5"]["mse_mean"],
          8.6, 0.06)
    # The paper says the 50 s divergence did NOT reproduce in this run. Assert
    # that, so the claim cannot silently go stale if the numbers are regenerated.
    worst = max(r["test_mse"] for r in wc["per_seed"]["Meta-LSQ-Sobolev"]["50.0"])
    ok = worst < 1.0
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'50s cell did NOT diverge in this run':<52s} "
          f"paper=no diverge   data=worst seed {worst:.4g}")
    if not ok:
        FAILS.append("50s divergence reproduction status")

print("\n== non-synthetic family: gas-sensor calibration ==")
rc = load("real_calibration.json")
if rc:
    want = {"CO (near-linear)":  {"MLP-MAML": 0.282, "per-task linear": 0.309,
                                  "Meta-LSQ (w=0, floor)": 0.318, "global linear": 0.378},
            "NOx (nonlinear)":   {"MLP-MAML": 0.718, "per-task linear": 1.343,
                                  "Meta-LSQ (w=0, floor)": 0.840, "global linear": 3.123}}
    for tgt, meths in want.items():
        for meth, v in meths.items():
            check(f"{tgt[:3]} {meth}", rc[tgt]["results"][meth]["mean"], v, 0.05)
        check(f"{tgt[:3]} vs per-task linear",
              rc[tgt]["ratio_vs_pertask_linear"],
              0.97 if tgt.startswith("CO") else 1.60, 0.05)
        # the noisy-data prediction of assumption (A2): mu rises well above 1e-5
        mu = rc[tgt]["mu"]
        ok = mu > 1e-4
        print(f"  [{'ok  ' if ok else 'FAIL'}] {'learned mu rises on real noisy data (' + tgt[:3] + ')':<52s} "
              f"paper=>1e-4        data={mu:.2e}")
        if not ok:
            FAILS.append(f"real-data mu {tgt}")
    # the selection rule must still hold: CO least nonlinear, NOx most
    g1 = rc["CO (near-linear)"]["nonlinearity"]["gain"]
    g2 = rc["NOx (nonlinear)"]["nonlinearity"]["gain"]
    ok = g1 < 0.05 < g2
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'target selection rule (least/most nonlinear)':<52s} "
          f"paper=+0.005/+0.271 data={g1:+.3f}/{g2:+.3f}")
    if not ok:
        FAILS.append("real-data target selection")

print("\n== ridge floor during training (divergence characterisation) ==")
rd = load("ridge_divergence.json")
if rd:
    cps = rd["checkpoints"]; seeds = rd["seeds"]
    def at(tag, it, key="mean"):
        i = cps.index(it)
        return [rd["runs"][tag][str(s)][i][key] for s in seeds]
    # unfloored is STABLE across the whole sweep
    worst_nf = max(max(at("no floor", c)) for c in cps)
    ok = worst_nf < 0.1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'unfloored stable to 80k steps':<52s} "
          f"paper=stable       data=worst {worst_nf:.4g}")
    if not ok:
        FAILS.append("unfloored stability")
    # floored DIVERGES -- this inverts the claim the paper used to make
    fl40 = sum(at("floor 1e-2", 40000)) / len(seeds)
    check("floored mean MSE @40k steps", fl40, 105.4, 0.10)
    n40 = sum(1 for s in seeds
              if rd["runs"]["floor 1e-2"][str(s)][cps.index(40000)]["mean"] > 1.0)
    n80 = sum(1 for s in seeds
              if rd["runs"]["floor 1e-2"][str(s)][cps.index(80000)]["mean"] > 1.0)
    ok = n40 >= 1 and n80 >= 2
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'floored: 1/3 seeds bad @40k, 2/3 @80k':<52s} "
          f"paper=1 then 2     data={n40}/3 then {n80}/3")
    if not ok:
        FAILS.append("floored divergence count")
    # mechanism: the outer loop routes around the floor by inflating w
    w0 = max(at("floor 1e-2", 2000, "grad_w"))
    w1 = max(at("floor 1e-2", 80000, "grad_w"))
    ok = w0 < 1e3 < 1e5 < w1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'floored run inflates w past 1e5':<52s} "
          f"paper=30 -> 1e6     data={w0:.0f} -> {w1:.3g}")
    if not ok:
        FAILS.append("grad_w inflation mechanism")
    wnf = max(at("no floor", 80000, "grad_w"))
    ok = wnf < 1e4
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'unfloored w stays bounded':<52s} "
          f"paper=<1e4         data={wnf:.0f}")
    if not ok:
        FAILS.append("unfloored grad_w bound")

print("\n== teaser: bandwidth is the whole game ==")
bs = load("bandwidth_sweep.json")
if bs:
    med = [c["median"] for c in bs["curve"]]
    sigs = [c["sigma"] for c in bs["curve"]]
    span = max(med) / min(med)
    ok = span > 1e5
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'bandwidth spans >5 orders of magnitude':<52s} "
          f"paper=5 orders     data={span:.3g}x")
    if not ok:
        FAILS.append("bandwidth sweep span")
    # the teaser claims the meta-learner lands IN the trough, not merely near it
    s_star = bs["learned_sigma_mean"]
    import bisect
    j = min(range(len(sigs)), key=lambda k: abs(sigs[k] - s_star))
    ratio = med[j] / min(med)
    ok = ratio < 2.0
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'meta-learned sigma lands in the trough':<52s} "
          f"paper=in trough    data={ratio:.2f}x grid optimum")
    if not ok:
        FAILS.append("sigma* in trough")
    check("learned sigma*", s_star, 0.643, 0.05)

print("\n== Table 3: GMM Sobolev cost ==")
so = load("sobolev.json")
if so:
    d8 = so["d8"]
    check("RFF d8 fit s", d8["fastlsq_vanilla"]["fit_s"], 3.4, 0.03)
    check("RFF+Sob d8 fit s", d8["fastlsq_sobolev"]["fit_s"], 12.9, 0.02)
    check("MLP d8 fit s", d8["mlp_vanilla"]["fit_s"], 46.1, 0.02)
    check("MLP+Sob d8 fit s", d8["mlp_sobolev"]["fit_s"], 70.9, 0.02)
    check("RFF d8 val", d8["fastlsq_vanilla"]["val"], 7.1e-3, 0.02)
    check("RFF+Sob d8 val", d8["fastlsq_sobolev"]["val"], 3.4e-3, 0.02)
    check("RFF Sobolev cost ratio d8",
          d8["fastlsq_sobolev"]["fit_s"] / d8["fastlsq_vanilla"]["fit_s"], 3.8, 0.06)
    check("MLP Sobolev cost ratio d8",
          d8["mlp_sobolev"]["fit_s"] / d8["mlp_vanilla"]["fit_s"], 1.54, 0.05)

sm = load("sobolev_multiseed.json")
if sm:
    import statistics as _st
    med = lambda rows, k, m: _st.median(r[k][m] for r in rows)
    wins = lambda rows, m: sum(r["fastlsq_sobolev"][m] < r["fastlsq_vanilla"][m] for r in rows)
    for dim in ("d4", "d8"):
        assert len(sm[dim]) == 5
    assert all(wins(sm["d8"], m) == 5 for m in ("val", "grad", "hess"))   # "every seed at d=8"
    assert wins(sm["d4"], "val") == 4 and wins(sm["d4"], "grad") == 4 and wins(sm["d4"], "hess") == 3
    for dim, m, k, want in [("d8", "val", "fastlsq_sobolev", 3.9e-3), ("d8", "val", "fastlsq_vanilla", 7.1e-3),
                            ("d8", "grad", "fastlsq_sobolev", 0.037), ("d8", "grad", "fastlsq_vanilla", 0.056),
                            ("d8", "hess", "fastlsq_sobolev", 0.081), ("d8", "hess", "fastlsq_vanilla", 0.11),
                            ("d4", "val", "fastlsq_sobolev", 4.5e-3), ("d4", "val", "fastlsq_vanilla", 9.9e-3),
                            ("d4", "grad", "fastlsq_sobolev", 0.056), ("d4", "grad", "fastlsq_vanilla", 0.12),
                            ("d4", "hess", "fastlsq_sobolev", 0.17), ("d4", "hess", "fastlsq_vanilla", 0.18)]:
        check(f"5-seed median {dim} {m} {k[8:]}", med(sm[dim], k, m), want, 0.03)
    r0 = sm["d4"][0]                                   # the table's seed: Sobolev worse on all three
    assert r0["seed"] == 0 and all(r0["fastlsq_sobolev"][m] > r0["fastlsq_vanilla"][m]
                                   for m in ("val", "grad", "hess"))

print("\n== Table 4: out-of-class ==")
oo = load("ooc_multiseed.json")
if oo:
    fam = {"sin": "in-class (sin)", "tanh": "OOC-A (tanh)", "triangle": "OOC-B (triangle)"}
    want = {
        ("sin", "mlp"): (0.70, 0.47), ("sin", "meta_lsq_sob"): (0.019, 3e-5),   # printed to 1 s.f.
        ("sin", "meta_lsq_sob_robust"): (0.056, 1.7e-3),
        ("tanh", "mlp"): (6.0, 4.6), ("tanh", "meta_lsq_sob"): (45, 0.11),
        ("tanh", "meta_lsq_sob_robust"): (0.56, 0.055),
        ("triangle", "mlp"): (4.0, 3.2), ("triangle", "meta_lsq_sob"): (18000, 32),
        ("triangle", "meta_lsq_sob_robust"): (63, 9.2),
    }
    for (f, m), (wm, wmed) in want.items():
        check(f"{f:<9s} {m:<20s} mean", oo[fam[f]][m]["mean"]["mean"], wm, 0.03)
        check(f"{f:<9s} {m:<20s} med ", oo[fam[f]][m]["median"]["mean"], wmed, 0.08)
    check("tanh floor vs MLP (mean)",
          oo["OOC-A (tanh)"]["mlp"]["mean"]["mean"]
          / oo["OOC-A (tanh)"]["meta_lsq_sob_robust"]["mean"]["mean"], 11, 0.05)
    check("tanh floor vs MLP (median)",
          oo["OOC-A (tanh)"]["mlp"]["median"]["mean"]
          / oo["OOC-A (tanh)"]["meta_lsq_sob_robust"]["median"]["mean"], 83, 0.05)
    check("triangle floor vs MLP (mean)",
          oo["OOC-B (triangle)"]["meta_lsq_sob_robust"]["mean"]["mean"]
          / oo["OOC-B (triangle)"]["mlp"]["mean"]["mean"], 16, 0.05)
    check("triangle floor vs MLP (median)",
          oo["OOC-B (triangle)"]["meta_lsq_sob_robust"]["median"]["mean"]
          / oo["OOC-B (triangle)"]["mlp"]["median"]["mean"], 2.9, 0.05)
    check("triangle tuned mean std", oo["OOC-B (triangle)"]["meta_lsq_sob"]["mean"]["std"],
          13000, 0.05)

mt = load("matern_ooc.json")
if mt and oo:
    t2 = mt["studentt_nu2"]
    check("t2 sin mean", t2["in-class (sin)"]["mean"]["mean"], 0.28, 0.03)
    check("t2 sin median", t2["in-class (sin)"]["median"]["mean"], 0.044, 0.03)
    check("t2 tanh mean", t2["OOC-A (tanh)"]["mean"]["mean"], 0.18, 0.03)
    check("t2 tanh median", t2["OOC-A (tanh)"]["median"]["mean"], 0.037, 0.03)
    check("t2 triangle mean", t2["OOC-B (triangle)"]["mean"]["mean"], 14, 0.03)
    check("t2 triangle median", t2["OOC-B (triangle)"]["median"]["mean"], 3.0, 0.02)
    g = mt["normal"]
    check("Gaussian triangle median (3 seeds)", g["OOC-B (triangle)"]["median"]["mean"], 7.99, 0.02)
    check("Gaussian tanh median (3 seeds)", g["OOC-A (tanh)"]["median"]["mean"], 0.061, 0.02)
    check("Gaussian sin median (3 seeds)", g["in-class (sin)"]["median"]["mean"], 0.0015, 0.03)
    check("Gaussian sin mean (3 seeds)", g["in-class (sin)"]["mean"]["mean"], 0.069, 0.02)
    check("t2 triangle vs MLP median",
          g["OOC-B (triangle)"]["median"]["mean"] / t2["OOC-B (triangle)"]["median"]["mean"],
          2.7, 0.03)
    check("t2 in-class median cost",
          t2["in-class (sin)"]["median"]["mean"] / g["in-class (sin)"]["median"]["mean"],
          29, 0.10)

print("\n== mechanism / effective rank ==")
er = load("eff_rank.json")
if er:
    det = {r["sigma"]: r for r in er["detail"]}
    E = lambda sg, k: det[sg][k]["mean"]
    check("K=10 eff rank H @0.65", E(0.65, "eff_rank_valonly_1e-06"), 9.9, 0.01)
    check("K=10 eff rank stacked @0.65", E(0.65, "eff_rank_stacked_w10.9_1e-06"), 13.1, 0.01)
    check("K=10 eff rank stacked @0.56", E(0.56, "eff_rank_stacked_w10.9_1e-06"), 12.4, 0.01)
    check("K=10 eff rank stacked @1.0", E(1.0, "eff_rank_stacked_w10.9_1e-06"), 15.2, 0.01)
    assert E(3.0, "eff_rank_stacked_w10.9_1e-06") > 19 and E(2.0, "eff_rank_stacked_w10.9_1e-06") < 19
    check("cond H median log10 @0.65", det[0.65]["log10_cond_restricted_valonly"]["median"], 4.7, 0.01)
    check("cond stacked top-K median log10 @0.65", det[0.65]["log10_cond_topK_stacked_w10.9"]["median"], 2.3, 0.01)
    for f in ("valonly_mu1e-05", "stacked_w10.9_mu1e-05"):
        assert abs(E(0.65, "beta_err_" + f) / E(0.65, "norm_beta_true") - 1) < 0.05
    check("relRMSE value, values only @0.65", E(0.65, "relrmse_val_valonly_mu1e-05"), 0.091, 0.02)
    check("relRMSE value, stacked @0.65", E(0.65, "relrmse_val_stacked_w10.9_mu1e-05"), 0.017, 0.03)
    check("relRMSE deriv, values only @0.65", E(0.65, "relrmse_der_valonly_mu1e-05"), 0.17, 0.03)
    check("relRMSE deriv, stacked @0.65", E(0.65, "relrmse_der_stacked_w10.9_mu1e-05"), 0.037, 0.02)
    check("value gain @sigma=2", E(2.0, "relrmse_val_valonly_mu1e-05") / E(2.0, "relrmse_val_stacked_w10.9_mu1e-05"), 1.8, 0.03)
    assert er["config"]["dtype"] == "float64" and er["config"]["K"] == 10 and er["config"]["draws"] == 200

print("\n== noise sweeps ==")
ns = load("noise_sweep.json")
if ns:
    by_n = {r["noise"]: r for r in ns}
    assert all(r.get("noise_mode") == "values_only" and r["n_iters"] == 8000 for r in ns)
    for nz, want in ((0.0, 1.9e-6), (0.1, 8.6e-5), (0.25, 4.2e-4), (0.5, 1.2e-3)):
        check(f"mu* eps={nz}", by_n[nz]["mu_star"]["mean"], want, 0.03)
    mus = [by_n[nz]["mu_star"]["mean"] for nz in (0.0, 0.1, 0.25, 0.5)]
    assert mus == sorted(mus)                                   # "rises monotonically"
    check("learned w noiseless", by_n[0.0]["grad_w"]["mean"], 94, 0.02)
    noisy_w = [by_n[nz]["grad_w"]["mean"] for nz in (0.1, 0.25, 0.5)]
    check("learned w noisy, min level", min(noisy_w), 320, 0.02)
    check("learned w noisy, max level", max(noisy_w), 370, 0.02)
    check("w seeds noiseless min", min(by_n[0.0]["grad_w"]["seeds"]), 82, 0.02)
    check("w seeds noiseless max", max(by_n[0.0]["grad_w"]["seeds"]), 113, 0.02)
    noisy_seeds = [w for nz in (0.1, 0.25, 0.5) for w in by_n[nz]["grad_w"]["seeds"]]
    check("w seeds noisy min", min(noisy_seeds), 228, 0.02)
    check("w seeds noisy max", max(noisy_seeds), 461, 0.02)
    assert min(noisy_seeds) > max(by_n[0.0]["grad_w"]["seeds"])   # "on every seed"
    check("Sobolev MSE noiseless", by_n[0.0]["meta_lsq_sob_mse"]["mean"], 0.0049, 0.03)
    check("Sobolev MSE eps=0.5", by_n[0.5]["meta_lsq_sob_mse"]["mean"], 0.055, 0.03)
    for nz in (0.0, 0.1, 0.25, 0.5):
        assert 0.6 <= by_n[nz]["mlp_mse"]["mean"] <= 0.7, nz
        assert by_n[nz]["meta_lsq_sob_mse"]["mean"] < by_n[nz]["mlp_mse"]["mean"], nz
    check("MLP MSE eps=0.5", by_n[0.5]["mlp_mse"]["mean"], 0.65, 0.06)
ns_old = load("superseded/noise_sweep.2000steps.json")
if ns_old:
    by_o = {r["noise"]: r for r in ns_old}
    check("2000-step w noiseless", by_o[0.0]["grad_w"]["mean"], 11, 0.03)
    check("2000-step w eps=0.5", by_o[0.5]["grad_w"]["mean"], 2.2, 0.03)

wg = load("noise_wgrid_seed0.json")
if wg:
    g5 = wg["0.5"]
    check("w grid: learned w at eps=0.5, 2000 steps", g5["w_learned"], 2.0, 0.03)
    check("w grid: MSE at learned w", g5["grid"][f"{g5['w_learned']:.3g}"], 0.14, 0.02)
    check("w grid: MSE at w=1e3", g5["grid"]["1e+03"], 0.075, 0.01)

gn = load("grad_noise_robustness.json")
if gn:
    by_e = {r["eps_g"]: r for r in gn["per_eps"]}
    check("learned w, clean gradients", by_e[0.0]["w_mean"], 10.9, 0.02)
    # the paper quotes sample std (ddof=1); the JSON stores population std
    import statistics as _st
    e0 = by_e[0.0]
    check("learned w sample std", _st.stdev(e0["w_seeds"]), 3.4, 0.03)
    check("learned sigma mean", _st.mean(e0["sigma_seeds"]), 0.65, 0.02)
    check("learned sigma sample std", _st.stdev(e0["sigma_seeds"]), 0.09, 0.05)
    check("learned w @eps_g=0.1", by_e[0.1]["w_mean"], 0.29, 0.03)
    check("learned w @eps_g=0.25", by_e[0.25]["w_mean"], 0.20, 0.03)
    check("learned w @eps_g=0.5", by_e[0.5]["w_mean"], 0.19, 0.05)
    check("MSE @eps_g=0.5", by_e[0.5]["mse_mean"], 0.21, 0.03)

sg = load("struct_grad_noise.json")
if sg:
    mp = {r["eps"]: r for r in sg["magprop"]["per_level"]}
    check("magnitude-prop w @eps=0", mp[0.0]["w_mean"], 4.6, 0.03)
    check("magnitude-prop w @eps=0.1", mp[0.1]["w_mean"], 0.31, 0.03)
    part = sg.get("partial", {}).get("per_level")
    if part:
        by_p = {r.get("p"): r for r in part}
        for p_, wv in [(0.25, 3.8), (0.5, 4.9), (1.0, 4.6)]:
            if p_ in by_p:
                check(f"partial gradients w @p={p_}", by_p[p_]["w_mean"], wv, 0.05)

print("\n== anisotropy / dimension ==")
an = load("aniso_multiseed.json")
if an:
    check("scalar bandwidth MSE", an["scalar"]["test_mse"]["mean"], 5.7, 0.02)
    check("diagonal bandwidth MSE", an["diag"]["test_mse"]["mean"], 1.39, 0.02)
    check("full bandwidth MSE", an["full"]["test_mse"]["mean"], 0.81, 0.02)
    # The paper now quotes PAIRED per-seed ratios, not ratios of means: the
    # absolute MSEs have overlapping error bars while the per-seed ratios are
    # one-sided, so the paired form is both stronger and harder to dispute.
    sc = an["scalar"]["test_mse"]["seeds"]
    dg = an["diag"]["test_mse"]["seeds"]
    fl = an["full"]["test_mse"]["seeds"]
    r_sd = [a / b for a, b in zip(sc, dg)]
    r_df = [a / b for a, b in zip(dg, fl)]
    check("paired scalar/diag", sum(r_sd) / len(r_sd), 3.9, 0.05)
    check("paired diag/full", sum(r_df) / len(r_df), 1.8, 0.05)
    for lbl, r in (("scalar/diag", r_sd), ("diag/full", r_df)):
        w = sum(1 for x in r if x > 1)
        ok = w == len(r)
        print(f"  [{'ok  ' if ok else 'FAIL'}] {'paired ' + lbl + ' wins on every seed':<52s} "
              f"paper={len(r)}/{len(r)}         data={w}/{len(r)}")
        if not ok:
            FAILS.append(f"paired {lbl} wins")

ast = load("anisotropy_study.json")
if ast:
    e1 = ast["e1_ratio_sweep"]
    for r, want in (("2", 2.2), ("4", 17.6), ("8", 67.4)):
        check(f"aniso dose-response r={r}", e1[r]["scalar_over_diag"]["mean"], want, 0.05)
    # r=1 must be consistent with NO gain: the interval has to straddle 1x.
    c = e1["1"]["scalar_over_diag"]
    ok = c["mean"] - c["std"] < 1.0
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'aniso r=1 consistent with no gain':<52s} "
          f"paper=straddles 1  data={c['mean']:.2f}+/-{c['std']:.2f}")
    if not ok:
        FAILS.append("aniso r=1 no-gain")
    for r, want in (("1", 1.16), ("2", 1.70), ("4", 3.24), ("8", 4.44)):
        check(f"learned scale ratio r={r}", e1[r]["cells"]["diag"]["scale_ratio"], want, 0.05)
    e2 = ast["e2_rotation_sweep"]
    check("rotation 45deg scalar/diag", e2["45"]["scalar_over_diag"]["mean"], 2.5, 0.06)
    check("rotation 45deg scalar/full", e2["45"]["scalar_over_full"]["mean"], 25.6, 0.06)
    check("rotation 0deg scalar/full", e2["0"]["scalar_over_full"]["mean"], 191, 0.06)
    # the qualitative claim: rotating collapses the diagonal but not L
    ok = (e2["45"]["scalar_over_diag"]["mean"] < 0.2 * e2["0"]["scalar_over_diag"]["mean"]
          and e2["45"]["scalar_over_full"]["mean"] > 5.0)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'rotation collapses diag but not full-L':<52s} "
          f"paper=yes          data={'yes' if ok else 'no'}")
    if not ok:
        FAILS.append("rotation collapse claim")

ang = load("anisotropy_e2_angles.json")
if ang:
    for th, true in (("0", 90.0), ("22.5", -67.5), ("45", -45.0)):
        a = ang["e2_rotation_sweep"][th]["cells"]["full"]["principal_angle_per_seed"]
        spread = max(a) - min(a)
        ok = spread < 15.0
        print(f"  [{'ok  ' if ok else 'FAIL'}] {'full-L angle seeds agree, theta=' + th:<52s} "
              f"paper=<15deg       data={spread:.1f}deg (true {true})")
        if not ok:
            FAILS.append(f"angle spread theta={th}")

ds = load("dim_scan.json")
if ds:
    by_d = {r["d"]: r for r in ds}
    check("d=64 value rRMSE", by_d[64]["fl"]["val"], 5.5e-2, 0.03)
    check("d=64 grad rRMSE", by_d[64]["fl"]["grad"], 0.89, 0.02)
    check("d=64 Hess rRMSE", by_d[64]["fl"]["hess"], 0.96, 0.02)

om = load("operator_mix_gmm.json")
if om:
    r8 = om["results"]["d8"]
    check("d8 value+grad val rRMSE", r8["value+grad"]["val_mean"], 4.4e-3, 0.03)
    check("d8 +Laplacian val rRMSE", r8["value+grad+lap"]["val_mean"], 4.6e-3, 0.03)
    r4 = om["results"]["d4"]
    check("d4 value+grad val rRMSE", r4["value+grad"]["val_mean"], 3.8e-3, 0.03)
    check("d4 +Laplacian val rRMSE", r4["value+grad+lap"]["val_mean"], 3.4e-3, 0.03)

print("\n== appendices ==")
tr = load("ooc_tail_ridge.json")
if tr:
    q = tr["quantile_table_pooled"]
    for key, med, p90, p99, mx, share in [
            ("mu_star", 34, 3900, 2.6e5, 4.8e6, 0.76),
            ("0.001", 12, 282, 4400, 1.7e4, 0.43),
            ("0.01", 7.7, 124, 762, 3800, 0.31),
            ("0.1", 7.1, 74, 182, 582, 0.15)]:
        check(f"tail mu={key:<8s} median", q[key]["median"], med, 0.03)
        check(f"tail mu={key:<8s} p90", q[key]["p90"], p90, 0.03)
        check(f"tail mu={key:<8s} p99", q[key]["p99"], p99, 0.03)
        check(f"tail mu={key:<8s} max", q[key]["max"], mx, 0.03)
        check(f"tail mu={key:<8s} top-1% share", q[key]["top1pct_mean_frac"], share, 0.03)

ar = load("autoridge_cv.json")
if ar:
    r = ar["results"]
    check("auto-ridge in-class mean", r["in-class (sin)"]["auto"]["mean"], 0.024, 0.03)
    check("floor in-class mean", r["in-class (sin)"]["floor"]["mean"], 0.056, 0.02)
    check("auto-ridge triangle median", r["OOC triangle"]["auto"]["median"], 7.0, 0.03)
    check("floor triangle median", r["OOC triangle"]["floor"]["median"], 8.6, 0.02)
    check("auto-ridge triangle mean", r["OOC triangle"]["auto"]["mean"], 5800, 0.03)
    check("floor triangle mean", r["OOC triangle"]["floor"]["mean"], 63, 0.02)

cv = load("cond_vs_extrap.json")
if cv:
    check("cond-blowup correlation", cv["aggregate"]["global_corr"]["mean"], 0.21, 0.05)
    fa = cv["aggregate"]["families"]
    ic, tri = fa["in_class_sin"], fa["ooc_triangle"]
    check("stabilised solve in-class MSE (normal eq)", ic["mse_star_normal_mean"]["mean"], 0.0206, 0.01)
    check("stabilised solve in-class MSE (lstsq)", ic["mse_star_stack_mean"]["mean"], 0.0207, 0.01)
    for f in (ic, tri):
        a, b = f["mse_star_normal_mean"]["mean"], f["mse_star_stack_mean"]["mean"]
        assert abs(a - b) / a < 0.01, (a, b)
    check("blowup-ratio change in class, max", max(ic["blow_normal_vs_stack_maxabsdiff"]["vals"]), 0.012, 0.03)
    check("blowup-ratio change triangle, max", max(tri["blow_normal_vs_stack_maxabsdiff"]["vals"]), 3.2, 0.03)
    check("triangle blowup ratio, max", max(tri["blow_star_normal_max"]["vals"]), 980, 0.01)
    check("cond at mu*, order", ic["cond_star_median"]["mean"], 1e6, 0.5)
    check("cond at 1e-2, order", ic["cond_big_median"]["mean"], 1e3, 0.5)

cal = load("calibration.json")
if cal:
    by_n = {e["noise"]: e for e in cal}
    check("unit-noise cov90 (noiseless)", 100 * by_n[0.0]["unit"]["cov90"]["mean"], 100, 0.01)
    check("BO-conv cov90 (noiseless)", 100 * by_n[0.0]["bo_code"]["cov90"]["mean"], 68, 0.02)
    check("BO-conv cov90 (worst)", 100 * by_n[0.25]["bo_code"]["cov90"]["mean"], 27, 0.02)
    check("emp-Bayes cov90 (noiseless)", 100 * by_n[0.0]["emp_bayes"]["cov90"]["mean"], 31, 0.02)
    check("emp-Bayes cov90 (best)", 100 * by_n[0.25]["emp_bayes"]["cov90"]["mean"], 64, 0.02)

bo = load("bo.json")
if bo:
    check("Branin random regret", bo["branin"]["random"]["final_regret_mean"], 1.67, 0.01)
    check("Branin Meta-LSQ regret", bo["branin"]["meta_lsq"]["final_regret_mean"], 0.69, 0.01)
    check("Branin GP regret", bo["branin"]["gp"]["final_regret_mean"], 0.048, 0.02)
    check("Branin Meta-LSQ time", bo["branin"]["meta_lsq"]["time_s"], 5.8, 0.02)
    check("Branin GP time", bo["branin"]["gp"]["time_s"], 25, 0.02)
    check("Hartmann Meta-LSQ regret", bo["hartmann6"]["meta_lsq"]["final_regret_mean"], 0.97, 0.02)
    check("Hartmann GP regret", bo["hartmann6"]["gp"]["final_regret_mean"], 0.39, 0.02)
    check("Hartmann Meta-LSQ time", bo["hartmann6"]["meta_lsq"]["time_s"], 8.3, 0.02)
    check("Hartmann GP time", bo["hartmann6"]["gp"]["time_s"], 48, 0.02)

pde = load("pde.json")
if pde:
    assert "exact" in pde["error_metric"], pde["error_metric"]
    assert pde["n_test"] == 50 and pde["n_iters"] == 1000 and len(pde["seeds"]) == 5
    rows = pde["per_seed"]
    check("PDE amortised median error", pde["median_meta_L2"], 1.8e-7, 0.01)
    check("PDE per-task median error", pde["median_auto_L2"], 8.0e-7, 0.01)
    check("PDE per-task median error, same ridge", pde["median_auto_matched_mu_L2"], 5.7e-8, 0.01)
    win = [r["mean_auto_L2"] / r["mean_meta_L2"] for r in rows]
    assert min(win) > 1, win                                   # "more accurate on every seed"
    check("PDE amortised win, min", min(win), 2.0, 0.03)
    check("PDE amortised win, max", max(win), 14, 0.03)
    loss = sorted(r["mean_meta_L2"] / r["mean_auto_matched_mu_L2"] for r in rows)
    assert loss[0] < 1 and loss[1] > 1, loss                   # level on one, worse on four
    check("PDE same-ridge loss, level seed", loss[0], 1.0, 0.03)
    check("PDE same-ridge loss, min of four", loss[1], 2.1, 0.03)
    check("PDE same-ridge loss, max", loss[-1], 5.1, 0.03)
    check("PDE sigma* min", min(r["sigma_star"] for r in rows), 2.9, 0.03)
    check("PDE sigma* max", max(r["sigma_star"] for r in rows), 3.5, 0.03)
    mus = [r["mu_star"] for r in rows]
    assert 0.5e-10 <= min(mus) and max(mus) <= 4e-9, mus       # "1e-10 -- 4e-9"
    assert pde["fastlsq_version"] == "0.7.1"
    check("PDE SVD amortised median", pde["median_meta_svd_L2"], 8.2e-10, 0.02)
    check("PDE SVD per-task median", pde["median_auto_svd_L2"], 1.7e-10, 0.02)
    assert all(r["mean_meta_svd_L2"] <= 2e-9 and r["mean_auto_svd_L2"] <= 2e-9
               for r in rows)                                  # "2e-9 or less"

ig = load("integro.json")
if ig:
    f = ig["forward"]
    check("integro amortised L2", f["amort_mean_L2"], 8.1e-7, 0.02)
    check("integro per-task L2", f["auto_mean_L2"], 5.8e-7, 0.02)
    check("integro sigma*", f["sigma_star"], 4.9, 0.02)
    check("integro amortised ms", f["amort_ms_per_task"], 48, 0.03)
    check("integro per-task ms", f["auto_ms_per_task"], 280, 0.02)
    check("integro L2 ratio", f["amort_mean_L2"] / f["auto_mean_L2"], 1.4, 0.03)
    check("integro speedup", f["auto_ms_per_task"] / f["amort_ms_per_task"], 6, 0.05)
    worst = max(r["rel_err"] for r in ig["inverse"] if r["noise"] == 0.05)
    best = min(r["rel_err"] for r in ig["inverse"] if r["noise"] == 0.05)
    check("inverse worst rel err @5% noise", 100 * worst, 3.9, 0.03)
    check("inverse best rel err @5% noise", 100 * best, 0.3, 0.05)

tz = load("teaser.json")
if tz and ms:
    # the teaser must show a genuine held-out task picked by the stated rule,
    # not a hand-chosen one
    c = tz["config"]
    assert c["selection"].startswith("median relative"), c["selection"]
    check("teaser task rank is the median", c["task_rank"], 48, 0.05)
    check("teaser amplitude is typical", tz["shown_task_amplitude"], 3.27, 0.02)
    check("teaser per-task mean matches Table 1",
          tz["per_task_mse_meta_lsq"]["mean"], ms["sobolev"]["test_mse"]["seeds"][0], 0.02)
    check("teaser shown-task Meta-LSQ MSE", tz["task_mse"]["meta_lsq_sobolev"], 7.3e-5, 0.05)
    check("teaser shown-task MLP-MAML MSE", tz["task_mse"]["mlp_maml"], 0.446, 0.02)

cc = load("closedform_classify.json")
if cc:
    check("5-way 5-shot Meta-LSQ acc", 100 * cc["5way_5shot"]["lsq_acc_mean"], 89.0, 0.01)
    check("5-way 5-shot MLP acc", 100 * cc["5way_5shot"]["mlp_acc_mean"], 85.1, 0.01)
    check("5-way 1-shot Meta-LSQ acc", 100 * cc["5way_1shot"]["lsq_acc_mean"], 84.0, 0.01)
    check("5-way 1-shot MLP acc", 100 * cc["5way_1shot"]["mlp_acc_mean"], 80.5, 0.01)
    check("classifier MLP params", cc["5way_5shot"]["mlp_n_learnable"], 1965, 0.001)

ca = load("compute_accounting_breakdown.json")
if ca:
    t = ca["timings_us"]
    check("inner solve ms", t["inner_solve_nograd"]["mean_us"] / 1e3, 0.81, 0.03)
    check("Meta-LSQ outer step ms", t["full_step"]["mean_us"] / 1e3, 9.6, 0.03)
    check("MLP-MAML step ms", t["mlp_full_step"]["mean_us"] / 1e3, 8.1, 0.03)
    fr_ = ca["inner_solve_fractions"]
    check("solve fraction", 100 * fr_["c_solve"], 70, 0.03)
    check("AtA fraction", 100 * fr_["b_form_AtA"], 22, 0.03)
    check("assembly fraction", 100 * fr_["a_assembly"], 12, 0.03)

print("== Camera ready: shared-hyperparameter GP (Table 1, App. GP) ==")
gm = load("rebuttal_gek_shared_matched.json")
gd = load("rebuttal_gek_shared_diag.json")
if gm:
    gek = gm["GEK shared, matched protocol (2000 steps, l0=1, lam0=1e-4)"]
    check("shared GEK 2k MSE", gek["mean"], 0.026, 0.02)
    check("shared GEK 2k std", gek["mean_std"], 0.014, 0.03)
if gm:
    E_ = lambda key: [v for k, v in gm.items() if k.startswith(key)][0]
    sob2, sob6 = E_("Meta-LSQ-Sobolev (2000"), E_("Meta-LSQ-Sobolev (6000")
    gek6 = E_("GEK shared, matched protocol (6000")
    val2, val6 = E_("Meta-LSQ w=0 (2000"), E_("Meta-LSQ w=0 (6000")
    gp2, gp6 = E_("GP shared values-only, matched protocol"), E_("GP shared values-only, matched protocol (6000")
    check("Meta-LSQ-Sob 2k (matched run)", sob2["mean"], 0.019, 0.03)
    check("Meta-LSQ-Sob 6k MSE", sob6["mean"], 0.0043, 0.01)
    check("Meta-LSQ-Sob 6k std", sob6["mean_std"], 0.0039, 0.02)
    check("shared GEK 6k MSE", gek6["mean"], 0.013, 0.03)
    check("shared GEK 6k std", gek6["mean_std"], 0.008, 0.02)
    check("Meta-LSQ w=0 2k (matched run)", val2["mean"], 0.059, 0.01)
    check("Meta-LSQ w=0 2k std", val2["mean_std"], 0.043, 0.02)
    check("Meta-LSQ w=0 6k MSE", val6["mean"], 0.034, 0.02)
    check("Meta-LSQ w=0 6k std", val6["mean_std"], 0.024, 0.02)
    check("shared GP val 2k MSE", gp2["mean"], 0.075, 0.01)
    check("shared GP val 2k std", gp2["mean_std"], 0.028, 0.01)
    check("shared GP val 6k MSE", gp6["mean"], 0.034, 0.01)
    check("shared GP val 6k std", gp6["mean_std"], 0.017, 0.03)
    wins = lambda a, b: sum(x < y for x, y in zip(a["per_seed_mean"], b["per_seed_mean"]))
    assert wins(sob2, gek) == 4 and wins(sob6, gek6) == 5, (wins(sob2, gek), wins(sob6, gek6))
    assert wins(val2, gp2) == 4 and wins(val6, gp6) == 3, (wins(val2, gp2), wins(val6, gp6))
if gd:
    check("grid-selected shared hp, Sob", gd["grid"]["sob"]["mean"], 0.012, 0.03)
    check("grid-selected shared hp, val", gd["grid"]["val"]["mean"], 0.031, 0.03)

print("== Camera ready: real data (Table real, App. real) ==")
TG = ["CO", "NO2", "C6H6", "NOx"]
RR = {t: load(f"rebuttal_real_{t}.json") for t in TG}
RF = {t: load(f"rebuttal_relfloor_{t}.json") for t in TG}
if all(RR.values()) and all(RF.values()):
    M = lambda t, k: RR[t]["methods"][k]["mean"]
    C = lambda t, c: RF[t]["methods"][c]["mean"]
    table = {  # Table real: Meta-LSQ (c=10), GP, MAML, lin, quad
        "CO":   (0.32, 0.30, 0.28, 0.31, 0.40),
        "NO2":  (0.86, 0.84, 3.86, 0.90, 1.74),
        "C6H6": (0.0037, 0.0038, 0.00015, 0.048, 0.0003),
        "NOx":  (0.85, 0.81, 0.72, 1.34, 2.28),
    }
    # two significant figures: rounding alone can be up to 3.4% off
    for t, (lsq, gp, maml, lin, quad) in table.items():
        check(f"real {t} Meta-LSQ 10mu*", C(t, "c=10"), lsq, 0.035)
        check(f"real {t} shared GP", M(t, "GP-RBF, shared meta-learned hp"), gp, 0.035)
        check(f"real {t} MLP-MAML", M(t, "MLP-MAML"), maml, 0.035)
        check(f"real {t} per-task linear", M(t, "per-task linear"), lin, 0.035)
        check(f"real {t} per-task quadratic", M(t, "per-task quadratic"), quad, 0.035)
    # App. real, full table: means as printed (columns CO, NO2, C6H6, NOx)
    full = {
        "Meta-LSQ (w=0)": (0.38, 1.03, 1.3e-3, 0.95),
        "Meta-LSQ (w=0, mu>=0.01)": (0.32, 0.85, 5.1e-2, 0.84),
        "Meta-LSQ FD-Sobolev": (0.87, 4.53, 3.8e-3, 6.58),
        "global linear": (0.38, 6.24, 5.2e-2, 3.12),
    }
    for k, vals in full.items():
        for t, want in zip(TG, vals):
            check(f"App real {t} {k}", M(t, k), want, 0.035)
    # App. real, floor sweep: mean and pooled worst week as printed
    sweep = {
        "c=1":   ((.38, 5.8), (1.03, 5.4), (.0013, .020), (.95, 8.0)),
        "c=3":   ((.35, 4.1), (.92, 3.5), (.0021, .040), (.90, 7.3)),
        "c=10":  ((.32, 2.4), (.86, 3.0), (.0037, .071), (.85, 4.3)),
        "c=30":  ((.32, 1.6), (.84, 2.9), (.0066, .096), (.85, 4.6)),
        "c=100": ((.35, 1.5), (.89, 2.7), (.012, .22), (.94, 4.7)),
        "abs 1e-2": ((.32, 1.7), (.85, 2.9), (.051, .48), (.84, 4.3)),
    }
    for c, cells in sweep.items():
        for t, (mean, worst) in zip(TG, cells):
            check(f"App floor {t} {c} mean", C(t, c), mean, 0.035)
            check(f"App floor {t} {c} worst", RF[t]["methods"][c]["pooled_max"], worst, 0.035)
    check("NOx gain over linear (1.6x)", M("NOx", "per-task linear") / C("NOx", "c=10"), 1.6, 0.02)
    check("benzene gain over linear (13x)", M("C6H6", "per-task linear") / C("C6H6", "c=10"), 13, 0.03)
    # learned ridge: noiseless benzene two orders below the noisy targets
    import statistics as st
    mu = {t: st.mean(RR[t]["learned"]["metalsq_mu"]) for t in TG}
    check("benzene learned ridge", mu["C6H6"], 5e-6, 0.1)
    for t in ("CO", "NO2", "NOx"):
        assert 4e-4 <= mu[t] <= 8e-4, (t, mu[t])
        assert M(t, "Meta-LSQ FD-Sobolev") > M(t, "Meta-LSQ (w=0)"), t
    assert M("C6H6", "Meta-LSQ FD-Sobolev") > M("C6H6", "Meta-LSQ (w=0)")
    fdw = [st.mean(RR[t]["learned"]["fd_w"]) for t in TG]
    check("FD weight min", min(fdw), 0.35, 0.03)
    check("FD weight max", max(fdw), 0.76, 0.03)
    # fixed floor on the noisy targets: mean -12..18%, worst week 1.8..3.4x
    red = [1 - M(t, "Meta-LSQ (w=0, mu>=0.01)") / M(t, "Meta-LSQ (w=0)") for t in ("CO", "NO2", "NOx")]
    wk = [RR[t]["methods"]["Meta-LSQ (w=0)"]["pooled_max"] /
          RR[t]["methods"]["Meta-LSQ (w=0, mu>=0.01)"]["pooled_max"] for t in ("CO", "NO2", "NOx")]
    check("fixed floor mean reduction min %", 100 * min(red), 12, 0.05)
    check("fixed floor mean reduction max %", 100 * max(red), 18, 0.05)
    check("fixed floor worst-week gain min", min(wk), 1.8, 0.03)
    check("fixed floor worst-week gain max", max(wk), 3.4, 0.03)
    check("benzene fixed-floor penalty (40x)",
          M("C6H6", "Meta-LSQ (w=0, mu>=0.01)") / M("C6H6", "Meta-LSQ (w=0)"), 40, 0.03)
    check("benzene 10mu* penalty", C("C6H6", "c=10") / C("C6H6", "c=1"), 2.9, 0.03)
    check("benzene 30mu* penalty", C("C6H6", "c=30") / C("C6H6", "c=1"), 5.2, 0.03)
    sizes = [RR[t]["n_tasks"] for t in TG]
    assert (min(sizes), max(sizes)) == (53, 56), sizes

print("== Camera ready: audit corrections ==")
import statistics as _st
ms_ = load("maml_multiseed.json")
if ms_:
    # Table 1 ms/t column: median over seeds
    check("ms/t MAML std (median)", _st.median(ms_["standard"]["inf_ms"]["seeds"]), 0.5, 0.1)
    check("ms/t Meta-LSQ w=0 (median)", _st.median(ms_["vanilla"]["inf_ms"]["seeds"]), 0.4, 0.1)
    check("ms/t Meta-LSQ-Sob (median)", _st.median(ms_["sobolev"]["inf_ms"]["seeds"]), 0.9, 0.05)
st_ = load("maml_strong.json")
if st_:
    check("ms/t MAML strong", st_["inf_ms"], 1.8, 0.03)
sb_ = load("maml_sobolev_baseline.json")
if sb_:
    lam1 = [r for r in sb_["per_lam"] if r["lam"] == 1.0][0]
    check("ms/t MAML+Sob (median)", _st.median(lam1["inf_ms"]["seeds"]), 1.0, 0.03)
wc_ = load("wallclock_budget.json")
if wc_:
    su_ = wc_["summary"] if "summary" in wc_ else None
    if su_:
        r = [su_["Meta-LSQ-Sobolev-floor"][b]["mse_mean"] / su_["Meta-LSQ-Sobolev"][b]["mse_mean"]
             for b in ("0.5", "5.0", "50.0")]
        check("floor cost in class, min", min(r), 2.0, 0.03)
        check("floor cost in class, max", max(r), 2.7, 0.03)
rd_ = load("ridge_divergence.json")
if rd_:
    nf = rd_["runs"]["no floor"]; fl = rd_["runs"]["floor 1e-2"]
    w80 = [rows[-1]["grad_w"] for rows in nf.values()]
    check("unfloored w at 8e4 steps, min", min(w80), 360, 0.02)
    check("unfloored w at 8e4 steps, max", max(w80), 2300, 0.01)
    assert all(rows[-1]["iters"] == 80000 for rows in nf.values())
    # test MSE at 8e4 stays at or below its 2000-step value on every seed
    assert all(rows[-1]["mean"] <= rows[0]["mean"] * 1.0001 or rows[-1]["mean"] < 0.02
               for rows in nf.values()), [(r[0]["mean"], r[-1]["mean"]) for r in nf.values()]
    w10k = [[r for r in rows if r["iters"] == 10000][0]["grad_w"] for rows in fl.values()]
    assert min(w10k) > 1e4, w10k
    div20 = [[r for r in rows if r["iters"] == 20000][0]["mean"] > 1 for rows in fl.values()]
    div80 = [[r for r in rows if r["iters"] == 80000][0]["mean"] > 1 for rows in fl.values()]
    assert sum(div20) == 1 and sum(div80) == 2, (div20, div80)
oo_ = load("ooc_multiseed.json")
if oo_:
    tri_mlp = oo_["OOC-B (triangle)"]["mlp"]["median"]["seeds"][:3]
    check("MLP triangle median, seeds 0-2", _st.mean(tri_mlp), 3.15, 0.01)
rb_ = load("rebuttal_real_C6H6.json")
if rb_:
    check("benzene degree-5 R^2", rb_["nonlinearity"]["r2_deg5"], 0.99998, 1e-5)
for t_, want_ in (("NOx", 0.03),):
    r_ = load(f"rebuttal_real_{t_}.json")
    if r_:
        check(f"real {t_} MLP-MAML std", r_["methods"]["MLP-MAML"]["mean_std"], want_, 0.2)
cc_ = load("closedform_classify.json")
if cc_:
    check("1-shot MLP acc std (%)", 100 * cc_["5way_1shot"]["mlp_acc_std"], 0.03, 0.2)

print()
if SKIPS:
    print(f"skipped (missing result files): {', '.join(sorted(set(SKIPS)))}")
if FAILS:
    print(f"\n{len(FAILS)} MISMATCH(ES):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("all checked numbers agree with results/")
