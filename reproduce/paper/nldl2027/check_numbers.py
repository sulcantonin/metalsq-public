"""Cross-check the numbers quoted in main.tex against results/*.json.

Every headline figure in the paper should be recomputable from the captured
experiment output.  This script recomputes them and compares against the values
hard-coded in the manuscript, so a stale number cannot survive a re-run of the
experiments unnoticed.

Three kinds of check:
  check_r(label, got, "0.019")  the paper prints this number; `got` has to round
                                to exactly that string (half a unit of its last
                                printed digit).  Use it for every plain number.
  check(label, got, want, rtol) a loose comparison, kept for statements such as
                                "about four times" or "near 11", and for counts.
  claim(label, ok)              a statement without a number ("on every seed",
                                "bold marks the lowest").

Run from anywhere:  python3 paper/northern-lights-crc/check_numbers.py
Exit status is non-zero if any check fails.
"""

import json
import os
import statistics as st
import sys

_st = st

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


def check_r(label, got, printed):
    """Exact-rounding check.  `printed` is the number as main.tex prints it, as a
    string: "0.019", "4.4", "0.70", "3.2e-10", "13".  Passes only if `got` rounds
    to that string at its printed precision, i.e. lies within half a unit of the
    last printed digit (plus a tiny float slack).  Trailing zeros count ("0.70" is
    two decimals), and so do the trailing zeros of an integer: a number the paper
    prints as 3900 but rounds to two figures is written "3.9e3" here."""
    if got is None:
        return
    s = printed.strip().lower()
    mant, _, exp = s.partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    unit = 10.0 ** ((int(exp) if exp else 0) - decimals)
    want = float(s)
    ok = abs(got - want) <= 0.5 * unit * (1 + 1e-9)
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {label:<52s} paper={printed:<12s} data={got:<12.6g}")
    if not ok:
        FAILS.append(label)


def claim(label, ok, shown=""):
    """A statement of the paper that is not a printed number."""
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label:<52s} {shown}")
    if not ok:
        FAILS.append(label)


print("== Table 1: 10-shot sinusoid ==")
ms = load("maml_multiseed.json")
if ms:
    check_r("MLP-MAML std test MSE", ms["standard"]["test_mse"]["mean"], "0.70")
    check_r("MLP-MAML std std", ms["standard"]["test_mse"]["std"], "0.13")
    check_r("MLP-MAML std train s", ms["standard"]["train_s"]["mean"], "51")
    check_r("Meta-LSQ vanilla test MSE", ms["vanilla"]["test_mse"]["mean"], "0.060")
    check_r("Meta-LSQ vanilla std", ms["vanilla"]["test_mse"]["std"], "0.043")
    check_r("Meta-LSQ vanilla train s", ms["vanilla"]["train_s"]["mean"], "11")
    check_r("Meta-LSQ Sobolev test MSE", ms["sobolev"]["test_mse"]["mean"], "0.019")
    check_r("Meta-LSQ Sobolev std", ms["sobolev"]["test_mse"]["std"], "0.016")
    check_r("Meta-LSQ Sobolev train s", ms["sobolev"]["train_s"]["mean"], "14")
    check_r("ratio std/vanilla (of means)",
            ms["standard"]["test_mse"]["mean"] / ms["vanilla"]["test_mse"]["mean"], "12")
    # the +/- of Table 1 are sample stds over 5 seeds (App. details, "sample std")
    for k_ in ("standard", "vanilla", "sobolev"):
        s_ = ms[k_]["test_mse"]["seeds"]
        claim(f"Table 1 {k_}: 5 seeds, +/- is the sample std",
              len(s_) == 5 and abs(st.stdev(s_) - ms[k_]["test_mse"]["std"]) < 1e-9)
    # ms/t column: median over seeds
    check_r("ms/t MAML std (median)", st.median(ms["standard"]["inf_ms"]["seeds"]), "0.5")
    check_r("ms/t Meta-LSQ w=0 (median)", st.median(ms["vanilla"]["inf_ms"]["seeds"]), "0.4")
    check_r("ms/t Meta-LSQ-Sob (median)", st.median(ms["sobolev"]["inf_ms"]["seeds"]), "0.9")

fr = load("strong_frontier.json")
if fr and ms:
    sob = ms["sobolev"]["test_mse"]["mean"]
    check_r("strong FOMAML 10k MSE", fr["10000"]["test_mse"], "0.328")
    check_r("strong FOMAML 30k MSE", fr["30000"]["test_mse"], "0.144")
    check_r("strong FOMAML 60k MSE", fr["60000"]["test_mse"], "0.104")
    check_r("strong FOMAML 10k train s", fr["10000"]["train_s"], "130")
    check_r("strong FOMAML 30k train s", fr["30000"]["train_s"], "399")
    check_r("strong FOMAML 60k train s", fr["60000"]["train_s"], "512")
    check_r("strong params", fr["60000"]["n_params"], "8513")
    check_r("gap vs 10k", fr["10000"]["test_mse"] / sob, "17")
    check_r("gap vs 30k", fr["30000"]["test_mse"] / sob, "7.6")
    check_r("gap vs 60k", fr["60000"]["test_mse"] / sob, "5.5")
    check_r("compute ratio 60k/Sobolev",
            fr["60000"]["train_s"] / ms["sobolev"]["train_s"]["mean"], "36")
    # App. details: "The 60k run took 8.5 ms per step and the 10k and 30k runs 13 ms"
    check_r("strong 60k ms per step", 1e3 * fr["60000"]["train_s"] / 60000, "8.5")
    check_r("strong 10k ms per step", 1e3 * fr["10000"]["train_s"] / 10000, "13")
    check_r("strong 30k ms per step", 1e3 * fr["30000"]["train_s"] / 30000, "13")
st_ = load("maml_strong.json")
if st_:
    check_r("ms/t MAML strong", st_["inf_ms"], "1.8")

sb = load("maml_sobolev_baseline.json")
if sb and ms:
    by_lam = {r["lam"]: r for r in sb["per_lam"]}
    assert sb["config"]["seeds"] == [0, 1, 2] and sb["config"]["iters"] == 8000
    check_r("MLP+Sobolev lam=0 MSE", by_lam[0.0]["test_mse"]["mean"], "0.716")
    check_r("MLP+Sobolev lam=0 std", by_lam[0.0]["test_mse"]["std"], "0.073")
    check_r("MLP+Sobolev lam=1 MSE", by_lam[1.0]["test_mse"]["mean"], "0.216")
    check_r("MLP+Sobolev lam=1 std", by_lam[1.0]["test_mse"]["std"], "0.031")
    check_r("MLP+Sobolev lam=1 train s", by_lam[1.0]["train_s"]["mean"], "78")
    check_r("ms/t MAML+Sob (median)", st.median(by_lam[1.0]["inf_ms"]["seeds"]), "1.0")
    check_r("MLP+Sobolev double-backprop cost ratio",
            by_lam[1.0]["train_s"]["mean"] / by_lam[0.0]["train_s"]["mean"], "1.9")
    check_r("gap vs information-matched MLP",
            by_lam[1.0]["test_mse"]["mean"] / ms["sobolev"]["test_mse"]["mean"], "11")
    # "learn." column: 1 -> h -> h -> 1 with h = 40 (config depth counts the linear layers)
    h_, dep_ = sb["config"]["hidden"], sb["config"]["depth"]
    check_r("MAML std / MAML+Sob params (from config)",
            (h_ + h_) + (dep_ - 2) * (h_ * h_ + h_) + (h_ + 1), "1761")

al = load("alpaca_baseline.json")
if al and ms:
    raw_in = al["alpaca"]["raw_in"]
    check_r("ALPaCA in-class MSE", sum(raw_in) / len(raw_in), "0.026")
    check_r("ALPaCA in-class std", st.stdev(raw_in), "0.017")
    check_r("ALPaCA params", al["alpaca"]["n_params"], "1721")
    check_r("ALPaCA train s", al["alpaca"]["train_s"][0], "12")
    lsq_in = al["meta_lsq_sobolev"]["raw_in"]
    check_r("Meta-LSQ-Sob matched-3-seed in-class", sum(lsq_in) / len(lsq_in), "0.023")
    # App. details: ALPaCA-style "transfers better out of class ... (0.30 against 0.64)";
    # the 0.64 is checked with matern_ooc.json below
    check_r("ALPaCA mean tanh MSE", st.mean(al["alpaca"]["raw_ooc"]), "0.30")
    assert len(raw_in) == 3 and al["config"]["n_iters_alpaca"] == 8000
if al and sb:                                     # Sec. 3: "a quarter as many steps"
    claim("Meta-LSQ needs a quarter as many steps as MAML",
          4 * al["config"]["n_iters_lsq"] == sb["config"]["iters"],
          f"data={al['config']['n_iters_lsq']} vs {sb['config']['iters']}")

am64, am32 = load("alpaca_matched_lsq_f64.json"), load("alpaca_matched_lsq_f32.json")
if am64 and am32 and al:
    assert am64["dtype"] == "torch.float64" and am32["dtype"] == "torch.float32"
    assert am64["n_iters"] == 8000 == al["config"]["n_iters_alpaca"]
    assert am64["seeds"] == al["config"]["seeds"]
    check_r("value-only Meta-LSQ 8k f64, matched seeds", am64["mean_mse"], "0.019")
    check_r("value-only Meta-LSQ 8k f32, matched seeds", am32["mean_mse"], "0.041")
    lower = sum(r["mean_mse"] < a for r, a in zip(am64["per_seed"], al["alpaca"]["raw_in"]))
    assert lower == 2, lower                                  # "lower on two of the three"

print("== Table gek: gradient-enhanced kriging (fair-fit GP comparators) ==")
pg = load("periodic_gek.json")
if pg:
    kk = pg["kernels"]
    assert pg["n_test"] == 200 and len(pg["seeds"]) == 5
    check_r("GEK RBF, unit signal variance", kk["rbf_paper_setting"]["test_mse"]["mean"], "0.079")
    check_r("GEK RBF, unit signal variance, std", kk["rbf_paper_setting"]["test_mse"]["std"], "0.022")
    check_r("GEK RBF, variance profiled", kk["rbf"]["test_mse"]["mean"], "0.013")
    check_r("GEK RBF, variance profiled, std", kk["rbf"]["test_mse"]["std"], "0.008")
    check_r("GEK periodic, period per task", kk["periodic"]["test_mse"]["mean"], "0.025")
    check_r("GEK periodic, period per task, std", kk["periodic"]["test_mse"]["std"], "0.015")
    check_r("GEK spectral-mixture Q=1", kk["sm1"]["test_mse"]["mean"], "3.2e-10")
    claim("GEK spectral-mixture is the lowest (bold)",
          kk["sm1"]["test_mse"]["mean"] == min(v["test_mse"]["mean"] for v in kk.values()))
    for k, want in (("rbf_paper_setting", "0.9"), ("rbf", "2.9"), ("periodic", "69.5"), ("sm1", "102")):
        check_r(f"GEK {k} ms/task", kk[k]["ms_per_task"], want)

print("== App. tail: ridge policies, incl. GCV ==")
gc = load("gcv_ridge.json")
if gc:
    r = gc["results"]
    assert gc["config"]["n_heldout"] == 3 and gc["config"]["gcv_grid"][0] == 1e-8
    assert len(gc["config"]["seeds"]) == 5 and gc["config"]["n_test"] == 200
    # Table ridgepolicy: every quoted cell (mean, pooled median, p99)
    for fam, pol, mean, med, p99 in [
            ("in-class (sin)", "tuned", "0.019", "2.7e-5", "0.39"),
            ("in-class (sin)", "floor", "0.054", "1.5e-3", "1.0"),
            ("in-class (sin)", "auto", "0.024", "5.2e-5", "0.47"),
            ("in-class (sin)", "gcv", "0.0062", "7.8e-7", "0.091"),
            ("OOC tanh", "tuned", "44", "0.100", "1.07e3"),          # printed 1070
            ("OOC tanh", "floor", "0.57", "0.057", "8.8"),
            ("OOC tanh", "auto", "12", "0.065", "211"),
            ("OOC tanh", "gcv", "1948", "0.227", "4.2e4"),
            ("OOC triangle", "tuned", "1.8e4", "27.8", "2.6e5"),
            ("OOC triangle", "floor", "63", "8.8", "914"),
            ("OOC triangle", "auto", "5419", "7.1", "7.4e4"),
            ("OOC triangle", "gcv", "1.5e6", "54.5", "3.2e7")]:
        x = r[fam][pol]
        assert x["n"] == 1000
        check_r(f"{fam[:12]:<12s} {pol:<5s} mean", x["mean"], mean)
        check_r(f"{fam[:12]:<12s} {pol:<5s} median", x["median"], med)
        check_r(f"{fam[:12]:<12s} {pol:<5s} p99", x["p99"], p99)
    # GCV best in class, worst out of class on every mean and p99
    for fam in ("OOC tanh", "OOC triangle"):
        for k in ("mean", "p99"):
            assert r[fam]["gcv"][k] == max(r[fam][p][k] for p in r[fam]), (fam, k)
    for k in ("mean", "median", "p99"):
        assert r["in-class (sin)"]["gcv"][k] == min(r["in-class (sin)"][p][k] for p in r["in-class (sin)"]), k
    # the other bold cells of the table
    for fam, k, pol in (("OOC tanh", "mean", "floor"), ("OOC tanh", "median", "floor"),
                        ("OOC tanh", "p99", "floor"), ("OOC triangle", "mean", "floor"),
                        ("OOC triangle", "median", "auto"), ("OOC triangle", "p99", "floor")):
        claim(f"ridge policies bold: {fam} {k} lowest for {pol}",
              r[fam][pol][k] == min(r[fam][p][k] for p in r[fam]))
    for fam, want in (("in-class (sin)", 1000), ("OOC tanh", 856), ("OOC triangle", 474)):
        n_floor = gc["mu_histogram"][fam]["gcv"].get("1e-08", 0)
        ok = n_floor == want
        print(f"  [{'ok  ' if ok else 'FAIL'}] {'GCV picks 1e-8 on ' + fam:<52s} "
              f"paper={want:<12d} data={n_floor:<12d}")
        if not ok:
            FAILS.append(f"GCV grid-floor selection count {fam}")
    g = gc.get("gcv_guard", {})
    rows = g.get("rows", [])
    ok = bool(g.get("ok")) and all(0.855 <= row["gain_recovered"] <= 0.955
                                   for row in rows)
    pct = "-".join("%.0f" % (row["gain_recovered"] * 100) for row in rows)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'GCV guard: recovers 86-95% of oracle gain':<52s} "
          f"paper=86-95%      data={pct}")
    if not ok:
        FAILS.append("GCV implementation guard")
    if rows:
        check_r("GCV guard, lowest gain recovered (%)", 100 * min(x["gain_recovered"] for x in rows), "86")
        check_r("GCV guard, highest gain recovered (%)", 100 * max(x["gain_recovered"] for x in rows), "95")

print("== derivative rows on held-out tasks (Fig. mechanism a, b; App. details) ==")
rv = load("review.json")
if rv:
    dm = rv["derivative_metrics"]
    check_r("query value  vanilla", dm["vanilla"]["val"]["mean"], "0.059")
    check_r("query grad   vanilla", dm["vanilla"]["grad"]["mean"], "0.067")
    check_r("query Hess   vanilla", dm["vanilla"]["hess"]["mean"], "0.087")
    check_r("query value  Sobolev", dm["sobolev"]["val"]["mean"], "0.019")
    check_r("query grad   Sobolev", dm["sobolev"]["grad"]["mean"], "0.017")
    check_r("query Hess   Sobolev", dm["sobolev"]["hess"]["mean"], "0.019")
    check_r("Hessian reduction factor",
            dm["vanilla"]["hess"]["mean"] / dm["sobolev"]["hess"]["mean"], "4.7")
    assert len(dm["vanilla"]["val"]["per_seed"]) == 5
    ws = {r["w"]: r for r in rv["w_sweep"]}
    check_r("w-sweep value at w=0", ws[0.0]["val_mse"]["mean"], "0.073")
    check_r("w-sweep value at w=10", ws[10.0]["val_mse"]["mean"], "0.023")
    check_r("w-sweep grad  at w=0", ws[0.0]["grad_mse"]["mean"], "0.080")
    check_r("w-sweep grad  at w=10", ws[10.0]["grad_mse"]["mean"], "0.016")
    # means fall monotonically in w ("though not on every seed")
    order = sorted(ws)
    assert order == [0.0, 0.1, 0.3, 1.0, 3.0, 10.0], order
    for key in ("val_mse", "grad_mse"):
        v = [ws[w][key]["mean"] for w in order]
        assert all(a >= b for a, b in zip(v, v[1:])), (key, v)
    assert len(ws[0.0]["val_mse"]["per_seed"]) == 3
    ns_ = {r["N"]: r["test_mse"]["mean"] for r in rv["n_sweep"]}
    for N_, want in ((50, "0.024"), (100, "0.021"), (200, "0.023")):
        check_r(f"N-sweep N={N_}", ns_[N_], want)
    assert "Reconstructed" not in rv.get("_note", "")

print("\n== Table wallclock: equal wall-clock ==")
wc = load("wallclock_budget.json")
if wc:
    su = wc["summary"]
    assert wc["seeds"] == [0, 1, 2] and wc["n_test"] == 200
    cells = {   # budget: per method (mean, population std) as printed
        "0.5": (("3.50", "0.06"), ("0.304", "0.076"), ("0.071", "0.017"), ("0.144", "0.034")),
        "5.0": (("2.07", "0.05"), ("0.173", "0.106"), ("0.034", "0.018"), ("0.085", "0.033")),
        "50.0": (("0.61", "0.06"), ("0.045", "0.009"), ("0.0076", "0.0011"), ("0.020", "0.009")),
    }
    meths = ("MLP-MAML", "Meta-LSQ-vanilla", "Meta-LSQ-Sobolev", "Meta-LSQ-Sobolev-floor")
    for b, row in cells.items():
        for m, (mean, std) in zip(meths, row):
            per = [x["test_mse"] for x in wc["per_seed"][m][b]]
            check_r(f"{m} @{b}s mean", su[m][b]["mse_mean"], mean)
            check_r(f"{m} @{b}s population std", st.pstdev(per), std)
        claim(f"wall-clock bold: Meta-LSQ-Sobolev lowest @{b}s",
              su["Meta-LSQ-Sobolev"][b]["mse_mean"] == min(su[m][b]["mse_mean"] for m in meths))
    check_r("0.5s Sobolev beats 50s MLP-MAML",
            su["MLP-MAML"]["50.0"]["mse_mean"] / su["Meta-LSQ-Sobolev"]["0.5"]["mse_mean"], "8.6")
    # Sec. 5.1 text: "about 115 steps", "about 9100 steps", "a step costs 4 to 5 ms"
    def _steps(m, b):
        return st.mean(x["iters"] for x in wc["per_seed"][m][b])
    check("Sobolev steps in 0.5 s ('about 115')", _steps("Meta-LSQ-Sobolev", "0.5"), 115, 0.05)
    check("MLP-MAML steps in 50 s ('about 9100')", _steps("MLP-MAML", "50.0"), 9100, 0.05)
    ms_step = [1e3 * st.mean(x["train_time"] for x in wc["per_seed"][m][b]) / _steps(m, b)
               for m in ("MLP-MAML", "Meta-LSQ-Sobolev") for b in ("0.5", "5.0", "50.0")]
    # Sec. 5.1: "about 4 ms for Meta-LSQ-Sobolev and 5 ms for MLP-MAML ... about a third more steps"
    claim(f"MLP-MAML ms per step about 5 ({ms_step[:3]})", all(4.5 <= v < 5.5 for v in ms_step[:3]))
    claim(f"Sobolev ms per step about 4 ({ms_step[3:]})", all(3.5 <= v < 4.5 for v in ms_step[3:]))
    check("Sobolev takes about a third more steps in 50 s",
          _steps("Meta-LSQ-Sobolev", "50.0") / _steps("MLP-MAML", "50.0"), 4 / 3, 0.05)
    # the 50 s cell of the Sobolev row must not be a diverged run
    worst = max(r["test_mse"] for r in wc["per_seed"]["Meta-LSQ-Sobolev"]["50.0"])
    ok = worst < 1.0
    print(f"  [{'ok  ' if ok else 'FAIL'}] {'50s cell did NOT diverge in this run':<52s} "
          f"paper=no diverge   data=worst seed {worst:.4g}")
    if not ok:
        FAILS.append("50s divergence reproduction status")
    # App. tail: training with the floor "makes the error 2--2.7x larger"
    r = [su["Meta-LSQ-Sobolev-floor"][b]["mse_mean"] / su["Meta-LSQ-Sobolev"][b]["mse_mean"]
         for b in ("0.5", "5.0", "50.0")]
    check_r("floor cost in class, min", min(r), "2")
    check_r("floor cost in class, max", max(r), "2.7")

print("\n== ridge floor during training, and longer runs (App. tail, App. wmodel) ==")
rd = load("ridge_divergence.json")
if rd:
    nf = rd["runs"]["no floor"]; fl = rd["runs"]["floor 1e-2"]
    assert len(nf) == 3 and len(fl) == 3
    assert all(rows[0]["iters"] == 2000 and rows[-1]["iters"] == 80000 for rows in nf.values())
    w80 = [rows[-1]["grad_w"] for rows in nf.values()]
    check_r("unfloored w at 8e4 steps, min", min(w80), "3.6e2")      # printed 360
    check_r("unfloored w at 8e4 steps, max", max(w80), "2.3e3")      # printed 2300
    # "the mean test MSE stays below its 2000-step value"
    mean_at = [st.mean(rows[i]["mean"] for rows in nf.values()) for i in range(len(rd["checkpoints"]))]
    claim("unfloored: mean test MSE stays below its 2000-step value",
          all(m < mean_at[0] for m in mean_at[1:]),
          "data=" + " ".join(f"{m:.3g}" for m in mean_at))
    w10k = [[r for r in rows if r["iters"] == 10000][0]["grad_w"] for rows in fl.values()]
    claim("floored: w exceeds 1e4 within 1e4 steps", min(w10k) > 1e4, f"data=min {min(w10k):.3g}")
    div20 = [[r for r in rows if r["iters"] == 20000][0]["mean"] > 1 for rows in fl.values()]
    div80 = [[r for r in rows if r["iters"] == 80000][0]["mean"] > 1 for rows in fl.values()]
    claim("floored: one run diverged by 2e4 steps, another by 8e4",
          sum(div20) == 1 and sum(div80) == 2, f"data={sum(div20)}/3 then {sum(div80)}/3")

print("\n== Table sobolev: GMM log-density ==")
so = load("sobolev.json")
if so:
    d8 = so["d8"]
    for k, fit_s, val, grad, hess, sig in [
            ("fastlsq_vanilla", "3.4", "7.1e-3", "5.3e-2", "9.3e-2", "0.05"),
            ("fastlsq_sobolev", "12.9", "3.4e-3", "3.5e-2", "7.9e-2", "0.1"),
            ("mlp_vanilla", "46.1", "8.9e-3", "6.7e-2", "1.1e-1", None),
            ("mlp_sobolev", "70.9", "7.2e-3", "5.7e-2", "9.5e-2", None)]:
        check_r(f"d8 {k} fit s", d8[k]["fit_s"], fit_s)
        check_r(f"d8 {k} val", d8[k]["val"], val)
        check_r(f"d8 {k} grad", d8[k]["grad"], grad)
        check_r(f"d8 {k} hess", d8[k]["hess"], hess)
        if sig:
            check_r(f"d8 {k} selected sigma", d8[k]["sigma"], sig)

sm = load("sobolev_multiseed.json")
if sm:
    med = lambda rows, k, m: _st.median(r[k][m] for r in rows)
    wins = lambda rows, m: sum(r["fastlsq_sobolev"][m] < r["fastlsq_vanilla"][m] for r in rows)
    for dim in ("d4", "d8"):
        assert len(sm[dim]) == 5
    assert all(wins(sm["d8"], m) == 5 for m in ("val", "grad", "hess"))   # "every seed at d=8"
    assert wins(sm["d4"], "val") == 4 and wins(sm["d4"], "grad") == 4 and wins(sm["d4"], "hess") == 3
    for dim, m, k, want in [("d8", "val", "fastlsq_sobolev", "3.9e-3"), ("d8", "val", "fastlsq_vanilla", "7.1e-3"),
                            ("d8", "grad", "fastlsq_sobolev", "0.037"), ("d8", "grad", "fastlsq_vanilla", "0.056"),
                            ("d8", "hess", "fastlsq_sobolev", "0.081"), ("d8", "hess", "fastlsq_vanilla", "0.11")]:
        check_r(f"5-seed median {dim} {m} {k[8:]}", med(sm[dim], k, m), want)
    r0 = sm["d4"][0]                                   # the table's seed: Sobolev worse on all three
    assert r0["seed"] == 0 and all(r0["fastlsq_sobolev"][m] > r0["fastlsq_vanilla"][m]
                                   for m in ("val", "grad", "hess"))
    # "The seed on which values and gradients get worse": there is exactly one
    assert sum(r["fastlsq_sobolev"]["val"] > r["fastlsq_vanilla"]["val"]
               and r["fastlsq_sobolev"]["grad"] > r["fastlsq_vanilla"]["grad"] for r in sm["d4"]) == 1

ds = load("dim_scan.json")
if ds:
    by_d = {r["d"]: r for r in ds}
    assert sorted(by_d) == [4, 8, 16, 32, 64]
    claim("dimension scan: value error at or below 6e-2",
          max(r["fl"]["val"] for r in ds) <= 6e-2, f"data=max {max(r['fl']['val'] for r in ds):.3g}")
    check_r("d=64 grad rRMSE", by_d[64]["fl"]["grad"], "0.89")
    check_r("d=64 Hess rRMSE", by_d[64]["fl"]["hess"], "0.96")
    lowest = min(r["fl"]["sigma"] for r in ds)
    claim("d >= 16: selected bandwidth at the lowest value of the scan",
          all(r["fl"]["sigma"] == lowest for r in ds if r["d"] >= 16), f"data=sigma {lowest}")

print("\n== Table ooc: out-of-class ==")
oo = load("ooc_multiseed.json")
mt = load("matern_ooc.json")
if oo:
    fam = {"sin": "in-class (sin)", "tanh": "OOC-A (tanh)", "triangle": "OOC-B (triangle)"}
    want = {
        ("sin", "mlp"): ("0.70", "0.47"), ("sin", "meta_lsq_sob"): ("0.019", "3e-5"),
        ("sin", "meta_lsq_sob_robust"): ("0.056", "1.7e-3"),
        ("tanh", "mlp"): ("6.0", "4.6"), ("tanh", "meta_lsq_sob"): ("45", "0.11"),
        ("tanh", "meta_lsq_sob_robust"): ("0.56", "0.055"),
        ("triangle", "mlp"): ("4.0", "3.2"), ("triangle", "meta_lsq_sob"): ("1.8e4", "32"),   # printed 18000
        ("triangle", "meta_lsq_sob_robust"): ("63", "9.2"),
    }
    for (f, m), (wm, wmed) in want.items():
        assert len(oo[fam[f]][m]["mean"]["seeds"]) == 5
        check_r(f"{f:<9s} {m:<20s} mean", oo[fam[f]][m]["mean"]["mean"], wm)
        check_r(f"{f:<9s} {m:<20s} med ", oo[fam[f]][m]["median"]["mean"], wmed)
    tanh_ = oo["OOC-A (tanh)"]; tri_ = oo["OOC-B (triangle)"]
    check("tanh tuned ridge vs MLP, median ('about 40x')",
          tanh_["mlp"]["median"]["mean"] / tanh_["meta_lsq_sob"]["median"]["mean"], 40, 0.1)
    check_r("tanh floor vs MLP (mean)",
            tanh_["mlp"]["mean"]["mean"] / tanh_["meta_lsq_sob_robust"]["mean"]["mean"], "11")
    check_r("tanh floor vs MLP (median)",
            tanh_["mlp"]["median"]["mean"] / tanh_["meta_lsq_sob_robust"]["median"]["mean"], "83")
    check_r("triangle floor vs MLP (mean)",
            tri_["meta_lsq_sob_robust"]["mean"]["mean"] / tri_["mlp"]["mean"]["mean"], "16")
    check_r("MLP triangle median, seeds 0-2", st.mean(tri_["mlp"]["median"]["seeds"][:3]), "3.15")

if mt and oo:
    t2 = mt["studentt_nu2"]
    assert all(len(t2[f]["mean"]["seeds"]) == 3 for f in t2)
    check_r("t2 sin mean", t2["in-class (sin)"]["mean"]["mean"], "0.28")
    check_r("t2 sin median", t2["in-class (sin)"]["median"]["mean"], "0.044")
    check_r("t2 tanh mean", t2["OOC-A (tanh)"]["mean"]["mean"], "0.18")
    check_r("t2 tanh median", t2["OOC-A (tanh)"]["median"]["mean"], "0.037")
    check_r("t2 triangle mean", t2["OOC-B (triangle)"]["mean"]["mean"], "14")
    check_r("t2 triangle median (table)", t2["OOC-B (triangle)"]["median"]["mean"], "3.0")
    check_r("t2 triangle median (App. details)", t2["OOC-B (triangle)"]["median"]["mean"], "2.97")
    g = mt["normal"]
    check_r("Gaussian triangle median (3 seeds)", g["OOC-B (triangle)"]["median"]["mean"], "7.99")
    check_r("Gaussian tanh median (3 seeds)", g["OOC-A (tanh)"]["median"]["mean"], "0.061")
    check_r("Gaussian sin median (3 seeds)", g["in-class (sin)"]["median"]["mean"], "0.0015")
    # App. details: "(0.30 against 0.64 mean tanh MSE on the same seeds)", floored Meta-LSQ
    check_r("floored Meta-LSQ mean tanh MSE (3 seeds)", g["OOC-A (tanh)"]["mean"]["mean"], "0.64")
    # bold cells of Table ooc: lowest of MLP-MAML, tuned, floor, t2 in each row
    def _row(f, stat):
        return {"mlp": oo[f]["mlp"][stat]["mean"], "tuned": oo[f]["meta_lsq_sob"][stat]["mean"],
                "floor": oo[f]["meta_lsq_sob_robust"][stat]["mean"], "t2": t2[f][stat]["mean"]}
    for f, stat, best in (("in-class (sin)", "mean", "tuned"), ("in-class (sin)", "median", "tuned"),
                          ("OOC-A (tanh)", "mean", "t2"), ("OOC-A (tanh)", "median", "t2"),
                          ("OOC-B (triangle)", "mean", "mlp"), ("OOC-B (triangle)", "median", "t2")):
        row = _row(f, stat)
        claim(f"Table ooc bold: {f} {stat} lowest for {best}", row[best] == min(row.values()))

print("\n== mechanism / effective rank ==")
er = load("eff_rank.json")
if er:
    det = {r["sigma"]: r for r in er["detail"]}
    E = lambda sg, k: det[sg][k]["mean"]
    check_r("K=10 eff rank H @0.65", E(0.65, "eff_rank_valonly_1e-06"), "9.9")
    check_r("K=10 eff rank stacked @0.65", E(0.65, "eff_rank_stacked_w10.9_1e-06"), "13.1")
    check("derivative rows add 'about three' directions",
          E(0.65, "eff_rank_stacked_w10.9_1e-06") - E(0.65, "eff_rank_valonly_1e-06"), 3, 0.15)
    assert E(3.0, "eff_rank_stacked_w10.9_1e-06") > 19 and E(2.0, "eff_rank_stacked_w10.9_1e-06") < 19
    cH = det[0.65]["log10_cond_restricted_valonly"]["median"]
    cS = det[0.65]["log10_cond_topK_stacked_w10.9"]["median"]
    check_r("cond H median log10 @0.65", cH, "4.7")
    check_r("cond stacked top-K median log10 @0.65", cS, "2.3")
    claim("conditioning improves by more than two orders", cH - cS > 2, f"data={cH - cS:.2f} orders")
    for f in ("valonly_mu1e-05", "stacked_w10.9_mu1e-05"):
        assert abs(E(0.65, "beta_err_" + f) / E(0.65, "norm_beta_true") - 1) < 0.05
    check_r("relRMSE value, values only @0.65", E(0.65, "relrmse_val_valonly_mu1e-05"), "0.091")
    check_r("relRMSE value, stacked @0.65", E(0.65, "relrmse_val_stacked_w10.9_mu1e-05"), "0.017")
    check_r("relRMSE deriv, values only @0.65", E(0.65, "relrmse_der_valonly_mu1e-05"), "0.17")
    check_r("relRMSE deriv, stacked @0.65", E(0.65, "relrmse_der_stacked_w10.9_mu1e-05"), "0.037")
    check("value error falls 'about five times'",
          E(0.65, "relrmse_val_valonly_mu1e-05") / E(0.65, "relrmse_val_stacked_w10.9_mu1e-05"), 5, 0.1)
    check("derivative error falls 'about five times'",
          E(0.65, "relrmse_der_valonly_mu1e-05") / E(0.65, "relrmse_der_stacked_w10.9_mu1e-05"), 5, 0.1)
    check_r("value gain @sigma=2",
            E(2.0, "relrmse_val_valonly_mu1e-05") / E(2.0, "relrmse_val_stacked_w10.9_mu1e-05"), "1.8")
    assert er["config"]["dtype"] == "float64" and er["config"]["K"] == 10 and er["config"]["draws"] == 200

print("\n== noise sweeps (App. wmodel) ==")
ns = load("noise_sweep.json")
if ns:
    by_n = {r["noise"]: r for r in ns}
    assert all(r.get("noise_mode") == "values_only" and r["n_iters"] == 8000 for r in ns)
    for nz, want in ((0.0, "1.9e-6"), (0.1, "8.6e-5"), (0.25, "4.2e-4"), (0.5, "1.2e-3")):
        check_r(f"mu* eps={nz}", by_n[nz]["mu_star"]["mean"], want)
    mus = [by_n[nz]["mu_star"]["mean"] for nz in (0.0, 0.1, 0.25, 0.5)]
    assert mus == sorted(mus)                                   # "rises monotonically"
    claim("learned ridge rises by nearly three orders of magnitude",
          300 < mus[-1] / mus[0] < 1000, f"data={mus[-1] / mus[0]:.0f}x")
    check_r("w seeds noiseless min", min(by_n[0.0]["grad_w"]["seeds"]), "82")
    check_r("w seeds noiseless max", max(by_n[0.0]["grad_w"]["seeds"]), "113")
    noisy_seeds = [w for nz in (0.1, 0.25, 0.5) for w in by_n[nz]["grad_w"]["seeds"]]
    check_r("w seeds noisy min", min(noisy_seeds), "228")
    check_r("w seeds noisy max", max(noisy_seeds), "461")
    assert min(noisy_seeds) > max(by_n[0.0]["grad_w"]["seeds"])   # "on every seed"
    assert all(len(by_n[nz]["grad_w"]["seeds"]) == 3 for nz in by_n)
    noisy_w = [by_n[nz]["grad_w"]["mean"] for nz in (0.1, 0.25, 0.5)]
    claim("w does not rise further between the noise levels",
          all(a >= b for a, b in zip(noisy_w, noisy_w[1:])), "data=" + " ".join(f"{w:.0f}" for w in noisy_w))
    for nz in (0.0, 0.1, 0.25, 0.5):          # "well below MLP-MAML at every noise level"
        assert by_n[nz]["meta_lsq_sob_mse"]["mean"] < by_n[nz]["mlp_mse"]["mean"], nz
    sob_ = [by_n[nz]["meta_lsq_sob_mse"]["mean"] for nz in (0.0, 0.1, 0.25, 0.5)]
    mlp_ = [by_n[nz]["mlp_mse"]["mean"] for nz in (0.0, 0.1, 0.25, 0.5)]
    check_r("App. C Meta-LSQ-Sobolev noise sweep, lowest (0.0049)", min(sob_), "0.0049")
    check_r("App. C Meta-LSQ-Sobolev noise sweep, highest (0.055)", max(sob_), "0.055")
    claim("App. C MLP-MAML 'about 0.65' at every noise level",
          all(0.6 <= v <= 0.7 for v in mlp_), " ".join(f"{v:.3f}" for v in mlp_))
ns_old = load("noise_sweep_2000steps.json")
if ns_old and ns:
    by_o = {r["noise"]: r for r in ns_old}
    # "Neither has converged at the benchmark's 2000 steps, where w climbs more slowly under noise"
    assert by_o[0.5]["grad_w"]["mean"] < by_o[0.0]["grad_w"]["mean"] < by_n[0.0]["grad_w"]["mean"]

gn = load("grad_noise_robustness.json")
if gn:
    by_e = {r["eps_g"]: r for r in gn["per_eps"]}
    assert gn["seeds"] == [0, 1, 2] and gn["config"]["n_iters"] == 2000
    for eps_, w_, mse_ in ((0.0, "10.9", "0.023"), (0.1, "0.29", "0.095"),
                           (0.25, "0.20", "0.098"), (0.5, "0.18", "0.21")):
        check_r(f"learned w @eps_g={eps_}", by_e[eps_]["w_mean"], w_)
        check_r(f"clean-query MSE @eps_g={eps_}", by_e[eps_]["mse_mean"], mse_)
    claim("noisy derivatives lower w by more than an order of magnitude",
          by_e[0.0]["w_mean"] / by_e[0.1]["w_mean"] > 10,
          f"data={by_e[0.0]['w_mean'] / by_e[0.1]['w_mean']:.0f}x")
    # the paper quotes sample std (ddof=1); the JSON stores population std
    e0 = by_e[0.0]
    check_r("learned w sample std", _st.stdev(e0["w_seeds"]), "3.4")
    check_r("learned sigma mean", _st.mean(e0["sigma_seeds"]), "0.65")
    check_r("learned sigma sample std", _st.stdev(e0["sigma_seeds"]), "0.09")
    check_r("learned mu* ('about 1e-5')", _st.mean(e0["mu_seeds"]), "1e-5")

sg = load("struct_grad_noise.json")
if sg:
    assert sg["seeds"] == [0, 1, 2] and sg["config"]["n_iters"] == 1200 and sg["config"]["n_test"] == 150
    mp = {r["eps"]: r for r in sg["magprop"]["per_level"]}
    check_r("magnitude-prop w @eps=0", mp[0.0]["w_mean"], "4.6")
    check_r("magnitude-prop w @eps=0.1", mp[0.1]["w_mean"], "0.31")
    by_p = {r["p"]: r for r in sg["partial"]["per_level"]}
    for p_, wv in [(0.25, "3.8"), (0.5, "4.9"), (1.0, "4.6")]:
        check_r(f"partial gradients w @p={p_}", by_p[p_]["w_mean"], wv)
    for p_, mv in [(0.0, "0.173"), (0.25, "0.120"), (0.5, "0.069"), (1.0, "0.036")]:
        check_r(f"partial gradients MSE @p={p_}", by_p[p_]["mse_mean"], mv)

print("\n== anisotropy (App. details, Fig. regime b) ==")
an = load("aniso_multiseed.json")
if an:
    # The paper quotes PAIRED per-seed ratios, not ratios of means: the
    # absolute MSEs have overlapping error bars while the per-seed ratios are
    # one-sided, so the paired form is both stronger and harder to dispute.
    sc = an["scalar"]["test_mse"]["seeds"]
    dg = an["diag"]["test_mse"]["seeds"]
    fl = an["full"]["test_mse"]["seeds"]
    r_sd = [a / b for a, b in zip(sc, dg)]
    r_df = [a / b for a, b in zip(dg, fl)]
    check_r("paired scalar/diag", sum(r_sd) / len(r_sd), "3.9")
    check_r("paired diag/full", sum(r_df) / len(r_df), "1.8")
    for lbl, r in (("scalar/diag", r_sd), ("diag/full", r_df)):
        w = sum(1 for x in r if x > 1)
        ok = w == len(r) == 5
        print(f"  [{'ok  ' if ok else 'FAIL'}] {'paired ' + lbl + ' wins on every seed':<52s} "
              f"paper=5/5         data={w}/{len(r)}")
        if not ok:
            FAILS.append(f"paired {lbl} wins")
    # Sec. limits: "lowers the error by up to 7x" (scalar against full, ratio of means)
    check_r("scalar/full, ratio of means ('up to 7x')",
            an["scalar"]["test_mse"]["mean"] / an["full"]["test_mse"]["mean"], "7")

print("\n== appendices ==")
tr = load("ooc_tail_ridge.json")
if tr:
    q = tr["quantile_table_pooled"]
    assert all(q[k]["n"] == 600 for k in q) and tr["config"]["seeds"] == [0, 1, 2]
    for key, med, p90, p99, mx, share in [
            ("mu_star", "34", "3.9e3", "2.6e5", "4.8e6", "76"),      # p90 printed 3900
            ("0.001", "12", "282", "4.4e3", "1.7e4", "43"),          # p99 printed 4400
            ("0.01", "7.7", "124", "762", "3.8e3", "31"),            # max printed 3800
            ("0.1", "7.1", "74", "182", "582", "15")]:
        check_r(f"tail mu={key:<8s} median", q[key]["median"], med)
        check_r(f"tail mu={key:<8s} p90", q[key]["p90"], p90)
        check_r(f"tail mu={key:<8s} p99", q[key]["p99"], p99)
        check_r(f"tail mu={key:<8s} max", q[key]["max"], mx)
        check_r(f"tail mu={key:<8s} top-1% share (%)", 100 * q[key]["top1pct_mean_frac"], share)
    claim("tail: p99 falls by three orders from mu* to 1e-1",
          q["mu_star"]["p99"] / q["0.1"]["p99"] >= 1e3, f"data={q['mu_star']['p99'] / q['0.1']['p99']:.0f}x")
    claim("tail table bold: mu=0.1 lowest in every column",
          all(q["0.1"][k] == min(q[m][k] for m in q)
              for k in ("median", "p90", "p99", "max", "top1pct_mean_frac")))

cv = load("cond_vs_extrap.json")
if cv:
    assert cv["config"]["n_seeds"] == 3
    check_r("cond-blowup correlation", cv["aggregate"]["global_corr"]["mean"], "0.21")
    fa = cv["aggregate"]["families"]
    ic, tri = fa["in_class_sin"], fa["ooc_triangle"]
    check_r("stabilised solve in-class MSE (normal eq)", ic["mse_star_normal_mean"]["mean"], "0.0206")
    check_r("stabilised solve in-class MSE (lstsq)", ic["mse_star_stack_mean"]["mean"], "0.0207")
    for f in (ic, tri):
        a, b = f["mse_star_normal_mean"]["mean"], f["mse_star_stack_mean"]["mean"]
        assert abs(a - b) / a < 0.01, (a, b)                        # "by less than 1%"
    check_r("blowup-ratio change triangle, max", max(tri["blow_normal_vs_stack_maxabsdiff"]["vals"]), "3.2")
    check_r("triangle blowup ratio, max", max(tri["blow_star_normal_max"]["vals"]), "9.8e2")   # printed 980
    check("cond at mu*, order", ic["cond_star_median"]["mean"], 1e6, 0.5)
    check("cond at 1e-2, order", ic["cond_big_median"]["mean"], 1e3, 0.5)

cal = load("calibration.json")
if cal:
    cal = sorted(cal, key=lambda e: e["noise"])
    cov = lambda conv: [100 * e[conv]["cov90"]["mean"] for e in cal]
    assert all(len(e["unit"]["cov90"]["seeds"]) == 3 for e in cal)
    for c_ in cov("unit"):
        check_r("unit-noise cov90 ('about 100%')", c_, "100")
    check_r("sigma_y^2 = mu cov90, highest", max(cov("bo_code")), "68")
    check_r("sigma_y^2 = mu cov90, lowest", min(cov("bo_code")), "27")
    check_r("plug-in cov90, lowest", min(cov("emp_bayes")), "31")
    check_r("plug-in cov90, highest", max(cov("emp_bayes")), "64")
    b_, e_ = cov("bo_code"), cov("emp_bayes")
    claim("sigma_y^2 = mu: worse with more noise", all(x > y for x, y in zip(b_, b_[1:])))
    claim("plug-in: better with more noise", all(x < y for x, y in zip(e_, e_[1:])))

pde = load("pde.json")
if pde:
    assert "exact" in pde["error_metric"], pde["error_metric"]
    assert "200 uniform points" in pde["error_metric"]
    assert pde["n_test"] == 50 and pde["n_iters"] == 1000 and len(pde["seeds"]) == 5
    rows = pde["per_seed"]
    check_r("PDE amortised median error", pde["median_meta_L2"], "1.8e-7")
    check_r("PDE per-task median error", pde["median_auto_L2"], "8.0e-7")
    check_r("PDE per-task median error, same ridge", pde["median_auto_matched_mu_L2"], "5.7e-8")
    # Sec. mechanism: "errors near 1e-7, four times lower than ..."
    check_r("PDE amortised vs per-task ('four times lower')",
            pde["median_auto_L2"] / pde["median_meta_L2"], "4")
    win = [r["mean_auto_L2"] / r["mean_meta_L2"] for r in rows]
    assert min(win) > 1, win                                   # "more accurate on every seed"
    check_r("PDE amortised win, min", min(win), "2.0")
    check_r("PDE amortised win, max", max(win), "14")
    loss = sorted(r["mean_meta_L2"] / r["mean_auto_matched_mu_L2"] for r in rows)
    assert loss[0] < 1 and loss[1] > 1, loss                   # level on one, worse on four
    check("PDE same-ridge loss, level seed", loss[0], 1.0, 0.03)
    check_r("PDE same-ridge loss, min of four", loss[1], "2.1")
    check_r("PDE same-ridge loss, max", loss[-1], "5.1")
    check_r("PDE sigma* min", min(r["sigma_star"] for r in rows), "2.9")
    check_r("PDE sigma* max", max(r["sigma_star"] for r in rows), "3.5")
    mus = [r["mu_star"] for r in rows]
    check_r("PDE learned ridge, lowest", min(mus), "1e-10")
    check_r("PDE learned ridge, highest", max(mus), "4e-9")
    assert pde["fastlsq_version"] == "0.7.1"
    assert all(r["mean_meta_svd_L2"] <= 2e-9 and r["mean_auto_svd_L2"] <= 2e-9
               for r in rows)                                  # "2e-9 or less"

ig = load("integro.json")
if ig:
    f = ig["forward"]
    check_r("integro amortised L2", f["amort_mean_L2"], "8.1e-7")
    check_r("integro per-task L2", f["auto_mean_L2"], "5.8e-7")
    check_r("integro sigma*", f["sigma_star"], "4.9")
    check_r("integro amortised ms", f["amort_ms_per_task"], "48")
    check_r("integro per-task ms", f["auto_ms_per_task"], "280")
    assert sorted({r["lam_true"] for r in ig["inverse"]}) == [0.5, 1.0, 2.0, 4.0]
    worst = max(r["rel_err"] for r in ig["inverse"] if r["noise"] == 0.05)
    best = min(r["rel_err"] for r in ig["inverse"] if r["noise"] == 0.05)
    check_r("inverse worst rel err @5% noise (%)", 100 * worst, "3.9")
    check_r("inverse best rel err @5% noise (%)", 100 * best, "0.3")
    assert 100 * worst < 4                                      # Sec. mechanism: "to within 4%"
    clean = [r["rel_err"] for r in ig["inverse"] if r["noise"] == 0.0]
    claim("inverse rel err on clean data 'about 1e-5'", all(3e-6 < x < 3e-5 for x in clean),
          f"data={min(clean):.1e}..{max(clean):.1e}")

tz = load("teaser.json")
if tz and ms:
    # the teaser must show a genuine held-out task picked by the stated rule,
    # not a hand-chosen one, from the seed-0 model of Table 1
    c = tz["config"]
    assert c["selection"].startswith("median relative"), c["selection"]
    check("teaser per-task mean matches Table 1",
          tz["per_task_mse_meta_lsq"]["mean"], ms["sobolev"]["test_mse"]["seeds"][0], 0.02)

cc = load("closedform_classify.json")
if cc:
    s5, s1 = cc["5way_5shot"], cc["5way_1shot"]
    assert s5["seeds"] == [0, 1, 2] and s5["n_test"] == 200 and s5["N_way"] == 5
    check_r("5-way 5-shot Meta-LSQ acc", 100 * s5["lsq_acc_mean"], "89.0")
    check_r("5-way 5-shot Meta-LSQ acc std", 100 * s5["lsq_acc_std"], "0.7")
    check_r("5-way 5-shot MLP acc", 100 * s5["mlp_acc_mean"], "85.1")
    check_r("5-way 5-shot MLP acc std", 100 * s5["mlp_acc_std"], "1.2")
    check_r("5-way 1-shot Meta-LSQ acc", 100 * s1["lsq_acc_mean"], "84.0")
    check_r("5-way 1-shot Meta-LSQ acc std", 100 * s1["lsq_acc_std"], "1.0")
    check_r("5-way 1-shot MLP acc", 100 * s1["mlp_acc_mean"], "80.5")
    check_r("5-way 1-shot MLP acc std", 100 * s1["mlp_acc_std"], "0.03")
    check_r("classifier MLP params", s5["mlp_n_learnable"], "1965")

ca = load("compute_accounting_breakdown.json")
if ca:
    t = ca["timings_us"]
    assert ca["config"]["meta_bs"] == 4 and ca["config"]["K"] == 10 and ca["config"]["N"] == 200
    check_r("inner solve ms", t["inner_solve_nograd"]["mean_us"] / 1e3, "0.81")
    check_r("Meta-LSQ outer step ms", t["full_step"]["mean_us"] / 1e3, "9.6")
    check_r("MLP-MAML step ms", t["mlp_full_step"]["mean_us"] / 1e3, "8.1")

print("== Camera ready: shared-hyperparameter GP (Table 1, App. GP) ==")
gm = load("rebuttal_gek_shared_matched.json")
gd = load("rebuttal_gek_shared_diag.json")
if gm:
    E_ = lambda key: [v for k, v in gm.items() if k.startswith(key)][0]
    gek = gm["GEK shared, matched protocol (2000 steps, l0=1, lam0=1e-4)"]
    sob2, sob6 = E_("Meta-LSQ-Sobolev (2000"), E_("Meta-LSQ-Sobolev (6000")
    gek6 = E_("GEK shared, matched protocol (6000")
    val2, val6 = E_("Meta-LSQ w=0 (2000"), E_("Meta-LSQ w=0 (6000")
    gp2, gp6 = E_("GP shared values-only, matched protocol"), E_("GP shared values-only, matched protocol (6000")
    for lbl, x, mean, std in [                       # Table gpshared (and the shared row of Table 1)
            ("Meta-LSQ-Sob 2k (matched run)", sob2, "0.019", "0.016"),
            ("Meta-LSQ-Sob 6k", sob6, "0.0043", "0.0039"),
            ("shared GEK 2k", gek, "0.026", "0.014"),
            ("shared GEK 6k", gek6, "0.013", "0.008"),
            ("Meta-LSQ w=0 2k (matched run)", val2, "0.059", "0.042"),
            ("Meta-LSQ w=0 6k", val6, "0.034", "0.024"),
            ("shared GP val 2k", gp2, "0.075", "0.028"),
            ("shared GP val 6k", gp6, "0.034", "0.017")]:
        assert len(x["per_seed_mean"]) == 5
        check_r(f"{lbl} MSE", x["mean"], mean)
        check_r(f"{lbl} sample std", st.stdev(x["per_seed_mean"]), std)
    wins = lambda a, b: sum(x < y for x, y in zip(a["per_seed_mean"], b["per_seed_mean"]))
    assert wins(sob2, gek) == 4 and wins(sob6, gek6) == 5, (wins(sob2, gek), wins(sob6, gek6))
    assert wins(val2, gp2) == 4 and wins(val6, gp6) == 3, (wins(val2, gp2), wins(val6, gp6))
if gd:
    check_r("grid-selected shared hp, Sob", gd["grid"]["sob"]["mean"], "0.012")
    check_r("grid-selected shared hp, val", gd["grid"]["val"]["mean"], "0.031")
if gm and ms and fr and sb and al:
    # Table 1: "Bold marks the lowest error among the methods without a per-task search"
    others = [ms["standard"]["test_mse"]["mean"], ms["vanilla"]["test_mse"]["mean"],
              min(v["test_mse"] for v in fr.values()),
              [r for r in sb["per_lam"] if r["lam"] == 1.0][0]["test_mse"]["mean"],
              st.mean(al["alpaca"]["raw_in"]), gek["mean"]]
    claim("Table 1 bold: Meta-LSQ-Sob lowest without a per-task search",
          ms["sobolev"]["test_mse"]["mean"] < min(others))

print("== Camera ready: real data (Table real, App. real) ==")
TG = ["CO", "NO2", "C6H6", "NOx"]
RR = {t: load(f"rebuttal_real_{t}.json") for t in TG}
RF = {t: load(f"rebuttal_relfloor_{t}.json") for t in TG}
if all(RR.values()) and all(RF.values()):
    M = lambda t, k: RR[t]["methods"][k]["mean"]
    C = lambda t, c: RF[t]["methods"][c]["mean"]
    GPK = "GP-RBF, shared meta-learned hp"
    table = {  # Table real: Meta-LSQ (c=10), GP, MAML, lin, quad; bold column
        "CO":   (("0.32", "0.30", "0.28", "0.31", "0.40"), 2),
        "NO2":  (("0.86", "0.84", "3.86", "0.90", "1.74"), 1),
        "C6H6": (("0.0037", "0.0038", "0.00015", "0.048", "0.0003"), 2),
        "NOx":  (("0.85", "0.81", "0.72", "1.34", "2.28"), 2),
    }
    cols = ("Meta-LSQ 10mu*", "shared GP", "MLP-MAML", "per-task linear", "per-task quadratic")
    for t, (cells, bold) in table.items():
        got = (C(t, "c=10"), M(t, GPK), M(t, "MLP-MAML"), M(t, "per-task linear"), M(t, "per-task quadratic"))
        for name, g_, w_ in zip(cols, got, cells):
            check_r(f"real {t} {name}", g_, w_)
        claim(f"Table real bold: {t} lowest for {cols[bold]}", got[bold] == min(got))
        assert RR[t]["seeds"] == [0, 1, 2, 3, 4] and RR[t]["K"] == 10 and RR[t]["K_q"] == 50
    # "MLP-MAML is better on three of the four targets"
    claim("MLP-MAML better than Meta-LSQ on three of four targets",
          sum(M(t, "MLP-MAML") < C(t, "c=10") for t in TG) == 3)
    # "the shared-hyperparameter GP stays within 8% of Meta-LSQ throughout"
    gap = max(abs(M(t, GPK) / C(t, "c=10") - 1) for t in TG)
    claim("shared GP within 8% of Meta-LSQ on every target", gap < 0.08, f"data=max {100 * gap:.1f}%")
    # App. real, full table: mean and sample std over 5 seeds as printed (CO, NO2, C6H6, NOx)
    full = {
        "c=10": (("0.32", "0.06"), ("0.86", "0.06"), ("3.7e-3", "2.4e-3"), ("0.85", "0.09")),
        "Meta-LSQ (w=0)": (("0.38", "0.14"), ("1.03", "0.16"), ("1.3e-3", "0.5e-3"), ("0.95", "0.12")),
        "Meta-LSQ (w=0, mu>=0.01)": (("0.32", "0.04"), ("0.85", "0.06"), ("5.1e-2", "2.0e-2"), ("0.84", "0.07")),
        "Meta-LSQ FD-Sobolev": (("0.87", "0.36"), ("4.53", "2.2"), ("3.8e-3", "1.7e-3"), ("6.58", "3.4")),
        GPK: (("0.30", "0.03"), ("0.84", "0.06"), ("3.8e-3", "2.8e-3"), ("0.81", "0.05")),
        "MLP-MAML": (("0.28", "0.02"), ("3.86", "1.4"), ("1.5e-4", "1.1e-4"), ("0.72", "0.03")),
        "per-task linear": (("0.31", "0.02"), ("0.90", "0.08"), ("4.8e-2", "0.9e-2"), ("1.34", "0.45")),
        "per-task quadratic": (("0.40", "0.18"), ("1.74", "0.89"), ("3.0e-4", "2.2e-4"), ("2.28", "1.4")),
        "global linear": (("0.38", "0.02"), ("6.24", "0.13"), ("5.2e-2", "0.6e-2"), ("3.12", "0.12")),
    }
    for k, vals in full.items():
        for t, (mean, std) in zip(TG, vals):
            x = RF[t]["methods"]["c=10"] if k == "c=10" else RR[t]["methods"][k]
            assert len(x["per_seed_mean"]) == 5
            check_r(f"App real {t} {k} mean", x["mean"], mean)
            check_r(f"App real {t} {k} sample std", st.stdev(x["per_seed_mean"]), std)
    # App. real, floor sweep: mean and pooled worst week as printed
    sweep = {
        "c=1":   (("0.38", "5.8"), ("1.03", "5.4"), ("0.0013", "0.020"), ("0.95", "8.0")),
        "c=3":   (("0.35", "4.1"), ("0.92", "3.5"), ("0.0021", "0.040"), ("0.90", "7.3")),
        "c=10":  (("0.32", "2.4"), ("0.86", "2.9"), ("0.0037", "0.071"), ("0.85", "4.3")),
        "c=30":  (("0.32", "1.6"), ("0.84", "2.9"), ("0.0066", "0.096"), ("0.85", "4.6")),
        "c=100": (("0.35", "1.5"), ("0.89", "2.6"), ("0.012", "0.22"), ("0.94", "4.7")),
        "abs 1e-2": (("0.32", "1.7"), ("0.85", "2.9"), ("0.051", "0.48"), ("0.84", "4.3")),
    }
    for c, cells in sweep.items():
        for t, (mean, worst) in zip(TG, cells):
            check_r(f"App floor {t} {c} mean", C(t, c), mean)
            check_r(f"App floor {t} {c} worst", RF[t]["methods"][c]["pooled_max"], worst)
    check_r("benzene gain over linear (13x)", M("C6H6", "per-task linear") / C("C6H6", "c=10"), "13")
    # learned ridge follows the noise level: noiseless benzene two orders below the noisy targets
    mu = {t: st.mean(RR[t]["learned"]["metalsq_mu"]) for t in TG}
    for t, w_ in zip(TG, ("4.4e-4", "7.6e-4", "5.3e-6", "5.2e-4")):
        check_r(f"learned ridge {t}", mu[t], w_)
    for t in TG:                                   # FD-Sobolev "worse than values alone on every target"
        assert M(t, "Meta-LSQ FD-Sobolev") > M(t, "Meta-LSQ (w=0)"), t
    fdw = [st.mean(RR[t]["learned"]["fd_w"]) for t in TG]
    check_r("FD weight min", min(fdw), "0.35")
    check_r("FD weight max", max(fdw), "0.76")
    check_r("benzene fixed-floor penalty (40x)",
            M("C6H6", "Meta-LSQ (w=0, mu>=0.01)") / M("C6H6", "Meta-LSQ (w=0)"), "40")
    check_r("benzene 10mu* penalty", C("C6H6", "c=10") / C("C6H6", "c=1"), "2.9")
    check_r("benzene 30mu* penalty", C("C6H6", "c=30") / C("C6H6", "c=1"), "5.2")
    # protocol: weeks per target and the chronological 60/40 split
    for key, lo, hi in (("n_tasks", "53", "56"), ("n_train", "32", "34"), ("n_test", "21", "22")):
        sizes = [RR[t][key] for t in TG]
        check_r(f"real data {key}, lowest", min(sizes), lo)
        check_r(f"real data {key}, highest", max(sizes), hi)
    # degree-5 polynomial against linear R^2 on the meta-train weeks
    for t, w_ in zip(TG, ("0.005", "0.021", "0.031", "0.271")):
        check_r(f"degree-5 R^2 gain {t}", RR[t]["nonlinearity"]["gain"], w_)
    check_r("benzene degree-5 R^2", RR["C6H6"]["nonlinearity"]["r2_deg5"], "0.99998")

# ---- ARES beam tuning (Sec. 4.3, Table ares, App. ares) ----
ar_ = load("ares_tuning.json")
if ar_:
    print("\nARES beam tuning")
    F_, L_, MN_, RB_ = ar_["final"], ar_["levels"], ar_["minutes"], ar_["robust"]
    check("ARES trials", ar_["n_trials"], 300, 0)
    for m_, med_, mean_ in (("RL agent", "4.4", "17.1"), ("BO (DESY)", "30.7", "38.0"),
                             ("ALPaCA-style, machine", "7.5", "13.2"), ("Meta-LSQ, machine", "7.8", "13.2"),
                             ("gradient descent, twin", "14.2", "27.7"),
                             ("ALPaCA-style, twin", "1.9", "9.8"), ("Meta-LSQ, twin", "1.6", "9.6")):
        check_r(f"ARES {m_} median um", F_[m_]["median"], med_)
        check_r(f"ARES {m_} mean um", F_[m_]["mean"], mean_)
    for m_, steps_, pct_ in (("RL agent", 39, 62), ("Meta-LSQ, twin", 16, 79), ("ALPaCA-style, twin", 20, 78)):
        check(f"ARES {m_} steps to 99%", L_[f"{m_}|0.99"]["steps"], steps_, 0)
        check(f"ARES {m_} reach 99% (%)", round(100 * L_[f"{m_}|0.99"]["reached"]), pct_, 0)
    for m_ in ("Meta-LSQ, machine", "ALPaCA-style, machine"):   # "worse than both surrogates from values alone"
        if not F_["gradient descent, twin"]["median"] > F_[m_]["median"]:
            FAILS.append(f"ARES gradient descent is not worse than {m_} on the median")
    for m_, pct_ in (("BO (DESY)", 1), ("ALPaCA-style, machine", 46), ("Meta-LSQ, machine", 44),
                     ("gradient descent, twin", 37)):
        check(f"ARES {m_} reach 99% (%)", round(100 * L_[f"{m_}|0.99"]["reached"]), pct_, 0)
        if L_[f"{m_}|0.99"]["steps"] is not None:
            FAILS.append(f"ARES {m_}: paper prints no 99% step count but data has one")
    # bold cells of Table ares, per block
    vals_ = ("RL agent", "BO (DESY)", "ALPaCA-style, machine", "Meta-LSQ, machine")
    twin_ = ("gradient descent, twin", "ALPaCA-style, twin", "Meta-LSQ, twin")
    claim("Table ares bold: agent has the lowest median from values",
          F_["RL agent"]["median"] == min(F_[m]["median"] for m in vals_))
    claim("Table ares bold: the two surrogates have the lowest means from values",
          max(F_[m]["mean"] for m in vals_[2:]) < min(F_[m]["mean"] for m in vals_[:2]))
    claim("Table ares bold: Meta-LSQ lowest median and mean with derivatives",
          all(F_["Meta-LSQ, twin"][k] == min(F_[m][k] for m in twin_) for k in ("median", "mean")))
    check("ARES RL steps to 95%", L_["RL agent|0.95"]["steps"], 8, 0)
    check("ARES Meta-LSQ twin steps to 95%", L_["Meta-LSQ, twin|0.95"]["steps"], 9, 0)
    for m_ in ("Meta-LSQ, machine", "ALPaCA-style, machine"):
        check(f"ARES BO/{m_} error ratio ('about four')", F_["BO (DESY)"]["median"] / F_[m_]["median"], 4.0, 0.06)
    check("ARES RL quad reversals", ar_["reversals"]["RL agent"], 8, 0.05)
    for m_ in ("Meta-LSQ, twin", "ALPaCA-style, twin", "Meta-LSQ, machine", "ALPaCA-style, machine"):
        check(f"ARES {m_} reversals ('about 240')", ar_["reversals"][m_], 240, 0.05)
    for key_, cyc_, want_ in (("RL agent|0.95", "cycle30", "2.2"), ("Meta-LSQ, twin|0.95", "cycle30", "3.3"),
                              ("RL agent|0.99", "cycle30", "4.1"), ("Meta-LSQ, twin|0.99", "cycle30", "6.6"),
                              ("Meta-LSQ, twin|0.99", "cycle0", "0.6")):
        check_r(f"ARES minutes {key_} {cyc_}", MN_[key_][cyc_], want_)
    for m_ in ("Meta-LSQ, twin", "ALPaCA-style, twin"):
        check(f"ARES imperfect model {m_} ('near 11')", RB_["twin"][m_]["median"], 11.0, 0.03)
    check_r("ARES Meta-LSQ values, first 150 trials", RB_["nominal"]["Meta-LSQ, machine"]["median"], "8.8")
    # imperfect model (first 150 trials), quoted in the text
    check_r("ARES imperfect model, gradient descent median", RB_["twin"]["gradient descent, twin"]["median"], "12")
    claim("ARES imperfect model: Meta-LSQ/ALPaCA worse than Meta-LSQ from values, ahead of gradient descent",
          all(RB_["nominal"]["Meta-LSQ, machine"]["median"] < RB_["twin"][m]["median"]
              < RB_["twin"]["gradient descent, twin"]["median"] for m in ("Meta-LSQ, twin", "ALPaCA-style, twin")))
    # Table ares-robust (noise, hysteresis) was removed on 16 Nov 2026: the surrogates
    # were never meta-trained with readout noise or hysteresis.

# ---- ARES, our copy of DESY's agent against their published runs (Sec. 4.3, App. ares) ----
ac_ = load("ares_agent_check.json")
if ac_:
    print("\nARES, our copy of DESY's agent")
    check("agent check trials", ac_["n_trials"], 300, 0)
    check_r("largest difference from a recorded move, in steps", ac_["largest_move_difference"], "3e-6")
    check("published runs that leave the limits", ac_["published_runs_leaving_limits"], 28, 0)
    check_r("published runs, median", ac_["published_best_error"]["median"], "4.4")
    check_r("published runs, mean", ac_["published_best_error"]["mean"], "11.3")
    check_r("our copy in their setting, median", ac_["ours_desy_setting_best_error"]["median"], "4.4")
    check_r("our copy in their setting, mean", ac_["ours_desy_setting_best_error"]["mean"], "11.3")
    bf_ = ac_["beam_final_setting"]
    check_r("agent with free magnets, median", bf_["not clipped"]["median"], "4.3")
    check_r("agent with free magnets, mean", bf_["not clipped"]["mean"], "11.8")
    check_r("agent with clipped magnets, median", bf_["clipped"]["median"], "4.4")
    check_r("agent with clipped magnets, mean", bf_["clipped"]["mean"], "17.1")
    if ar_:
        for k_ in ("median", "mean"):
            check(f"clipped agent agrees with ares_tuning.json, {k_}", bf_["clipped"][k_],
                  ar_["final"]["RL agent"][k_], 1e-6)
        claim("agent's mean above the surrogates' only with clipped magnets",
              bf_["not clipped"]["mean"] < min(ar_["final"][m_]["mean"] for m_ in
                                               ("Meta-LSQ, machine", "ALPaCA-style, machine"))
              < bf_["clipped"]["mean"])

# ---- ARES target grid (Sec. 4.3, Table ares-grid, App. ares) ----
gr_ = load("ares_grid.json")
if gr_:
    print("\nARES target grid")
    check("grid targets per band x 4", gr_["n"], 1200, 0)
    cols_ = ("agent, published", "BO, published", "ALPaCA-style", "Meta-LSQ")
    table_ = {"0.0-0.5": (49, ("6.7", "10"), ("29", "46"), ("6.5", "7.5"), ("6.6", "8.9")),
              "0.5-1.0": (23, ("18", "152"), ("49", "80"), ("7.7", "11"), ("7.7", "12")),
              "1.0-1.5": (9, ("69", "258"), ("109", "147"), ("7.9", "8.6"), ("7.9", "8.4")),
              "1.5-2.0": (19, ("278", "1222"), ("189", "236"), ("8.3", "21"), ("8.3", "21"))}
    for band_, (share_, *cells_) in table_.items():
        b_ = gr_["bands"][band_]
        check(f"grid {band_} share (%)", round(100 * b_["share"]), share_, 0)
        check(f"grid {band_} targets", b_["n"], 300, 0)
        for c_, (med_, mean_) in zip(cols_, cells_):
            check_r(f"grid {band_} {c_} median", b_[c_][0], med_)
            check_r(f"grid {band_} {c_} mean", b_[c_][1], mean_)
    # whole grid: the table prints two figures, the text and the abstract 14.4, 41.9 and 7.3
    for c_, (med_, mean_, text_) in zip(cols_, (("14", "295", "14.4"), ("42", "99", "41.9"),
                                                ("7.3", "11", "7.3"), ("7.3", "12", "7.3"))):
        check_r(f"grid whole {c_} median (table)", gr_["grid"][c_][0], med_)
        check_r(f"grid whole {c_} median (text)", gr_["grid"][c_][0], text_)
        check_r(f"grid whole {c_} mean", gr_["grid"][c_][1], mean_)
    for band_, pub_, own_ in zip(table_, (53, 89, 98, 99), (43, 82, 95, 97)):
        check(f"grid {band_} Meta-LSQ closer than published agent (%)",
              round(100 * gr_["bands"][band_]["lsq_beats_agent"]), pub_, 0)
        check(f"grid {band_} Meta-LSQ closer than our agent's best (%)",
              round(100 * gr_["bands"][band_]["lsq_beats_own_agent_best"]), own_, 0)
    for band_, best_ in zip(table_, ("5.5", "15", "34", "185")):
        check_r(f"grid {band_} our agent, best, median", gr_["bands"][band_]["agent, ours, best"][0], best_)
    for c_ in ("ALPaCA-style", "Meta-LSQ"):      # largest targets: "the surrogates stay near 8 um"
        check(f"grid 1.5-2.0 {c_} median ('near 8')", gr_["bands"]["1.5-2.0"][c_][0], 8, 0.05)
    if not gr_["criterion_met"]:
        FAILS.append("grid: the pre-set win criterion is not met")

print("\n== Sec. 5.2 / Table 3 / App. E: fixed design selected by cross-validation ==")
v2 = load("real_calibration_v2.json")
if v2:
    import numpy as np
    T4 = ["CO", "NO2", "C6H6", "NOx"]
    tr_ = lambda t: v2[t]["test_per_seed_mean"]
    sel_ = lambda t, fam: v2[t]["selected"][fam]
    mean_ = lambda t, k: float(np.mean(tr_(t)[k]))
    std_ = lambda t, k: float(np.std(tr_(t)[k], ddof=1))
    FIRST, GPFIRST = "Meta-LSQ (first design)|x|floor=10", "GP (first design)|x"
    # the selection the paper describes
    claim("CV: Meta-LSQ and GP use T/RH on CO, NO2, NOx; sensor only on benzene",
          all(sel_(t, f).split("|")[1] == ("x" if t == "C6H6" else "xTR")
              for t in T4 for f in ("Meta-LSQ", "GP")))
    claim("CV floors 3, 1, 3, 3 mu* (CO, NO2, benzene, NOx)",
          [sel_(t, "Meta-LSQ").split("floor=")[1] for t in T4] == ["3", "1", "3", "3"])
    claim("CV: MAML sensor only except on NO2",
          [sel_(t, "MAML").split("|")[1] for t in T4] == ["x", "xTR", "x", "x"])
    claim("CV: ALPaCA-style uses T/RH on every target",
          all(sel_(t, "ALPaCA-style").split("|")[1] == "xTR" for t in T4))
    claim("CV: per-task linear uses T/RH except on benzene",
          [sel_(t, "per-task linear").split("|")[1] for t in T4] == ["xTR", "xTR", "x", "xTR"])
    # Table 3, as printed
    tab3 = {"CO":   ("0.25", "0.20", "0.20", "0.28", "0.34", "0.40"),
            "NO2":  ("0.78", "0.69", "0.42", "1.61", "0.84", "1.74"),
            "C6H6": ("1.6", "0.57", "2.2", "1.4", "483", "3.0"),        # printed in units of 1e-4
            "NOx":  ("2.57", "0.62", "0.63", "0.72", "1.56", "2.28")}
    bold = {"CO": ("GP", "ALPaCA-style"), "NO2": ("ALPaCA-style",), "C6H6": ("GP",), "NOx": ("GP",)}
    for t, row in tab3.items():
        fams = ("Meta-LSQ", "GP", "ALPaCA-style", "MAML", "per-task linear", "per-task quadratic")
        keys = tuple(sel_(t, f) for f in fams)
        sc = 1e4 if t == "C6H6" else 1.0
        for lbl, k, p in zip(fams, keys, row):
            check_r(f"Table 3 {t} {lbl}", sc * mean_(t, k), p)
        printed = {f: float(p) for f, p in zip(fams, row)}
        claim(f"Table 3 {t}: bold = lowest at printed precision {bold[t]}",
              {f for f, v in printed.items() if v == min(printed.values())} == set(bold[t]))
    # App. E table: mean +- std
    tabE = {  # row: per target (mean, std) as printed
        "first":  (FIRST, {"CO": ("0.32", "0.05"), "NO2": ("0.87", "0.08"), "C6H6": ("4.0e-3", "2.9e-3"), "NOx": ("0.84", "0.07")}),
        "lsq":    (None,  {"CO": ("0.25", "0.03"), "NO2": ("0.78", "0.08"), "C6H6": ("1.6e-4", "1.9e-4"), "NOx": ("2.57", "1.9")}),
        "gpfirst": (GPFIRST, {"CO": ("0.30", "0.03"), "NO2": ("0.84", "0.06"), "C6H6": ("3.8e-3", "2.8e-3"), "NOx": ("0.81", "0.05")}),
        "gp":     ("GP", {"CO": ("0.20", "0.01"), "NO2": ("0.69", "0.10"), "C6H6": ("5.7e-5", "3.2e-5"), "NOx": ("0.62", "0.02")}),
        "alpaca": ("ALPaCA-style", {"CO": ("0.20", "0.01"), "NO2": ("0.42", "0.04"), "C6H6": ("2.2e-4", "1.3e-4"), "NOx": ("0.63", "0.03")}),
        "maml":   ("MAML", {"CO": ("0.28", "0.02"), "NO2": ("1.61", "0.97"), "C6H6": ("1.4e-4", "1.2e-4"), "NOx": ("0.72", "0.03")}),
        "lin":    ("per-task linear", {"CO": ("0.34", "0.05"), "NO2": ("0.84", "0.20"), "C6H6": ("4.8e-2", "0.9e-2"), "NOx": ("1.56", "0.50")}),
        "quad":   ("per-task quadratic", {"CO": ("0.40", "0.18"), "NO2": ("1.74", "0.89"), "C6H6": ("3.0e-4", "2.2e-4"), "NOx": ("2.28", "1.4")}),
    }
    for row, (src, vals) in tabE.items():
        for t, (m_p, s_p) in vals.items():
            k = src if src in (FIRST, GPFIRST) else sel_(t, "Meta-LSQ" if src is None else src)
            check_r(f"App. E {row} {t} mean", mean_(t, k), m_p)
            check_r(f"App. E {row} {t} std", std_(t, k), s_p)
    # Sec. 5.2 text
    L = {t: mean_(t, sel_(t, "Meta-LSQ")) for t in T4}
    F = {t: mean_(t, FIRST) for t in T4}
    G = {t: mean_(t, sel_(t, "GP")) for t in T4}
    Mm = {t: mean_(t, sel_(t, "MAML")) for t in T4}
    check_r("first design 28x behind MAML on benzene", F["C6H6"] / Mm["C6H6"], "28")
    check_r("first design's error higher, CO (30%)", 100 * (F["CO"] / L["CO"] - 1), "30")
    check_r("first design's error higher, NO2 (12%)", 100 * (F["NO2"] / L["NO2"] - 1), "12")
    check_r("improvement over first design, benzene (25x)", F["C6H6"] / L["C6H6"], "25")
    for t in ("CO", "NO2"):
        a, b = np.array(tr_(t)[sel_(t, "Meta-LSQ")]), np.array(tr_(t)[sel_(t, "MAML")])
        claim(f"{t}: below MAML on every seed", bool((a < b).all()))
    claim("benzene: level with MAML (within 15%)", abs(L["C6H6"] / Mm["C6H6"] - 1) < 0.15,
          f"ratio {L['C6H6'] / Mm['C6H6']:.3f}")
    claim("abstract: ahead of MAML on exactly two targets (CO, NO2)",
          [t for t in T4 if L[t] < Mm[t]] == ["CO", "NO2"])
    claim("GP better than Meta-LSQ on every target (abstract)", all(G[t] < L[t] for t in T4))
    A_ = {t: mean_(t, sel_(t, "ALPaCA-style")) for t in T4}
    claim("abstract: ALPaCA-style more accurate than Meta-LSQ on most (3 of 4) targets",
          sum(A_[t] < L[t] for t in T4) == 3)
    claim("ALPaCA-style best on NO2", A_["NO2"] < min(G["NO2"], Mm["NO2"], L["NO2"]))
    claim("GP best on benzene", G["C6H6"] < min(A_["C6H6"], Mm["C6H6"], L["C6H6"]))
    claim("GP and ALPaCA-style level on CO and NOx (within 3%)",
          all(abs(A_[t] / G[t] - 1) < 0.03 for t in ("CO", "NOx")),
          " ".join(f"{t} {A_[t] / G[t]:.3f}" for t in ("CO", "NOx")))
    # cost: Results sentence and App. E
    tm_ = lambda t, fam, f: float(np.mean([x[f] for x in v2[t]["timing"][sel_(t, fam)]]))
    rng_ = lambda fam, f: (min(tm_(t, fam, f) for t in T4), max(tm_(t, fam, f) for t in T4))
    for fam, (tr_p, ad_p) in {"Meta-LSQ": (("5.4", "8.7"), ("0.29", "0.49")),
                              "GP": (("1.3", "1.4"), ("0.06", "0.07")),
                              "ALPaCA-style": (("6.9", "7.0"), ("0.09", "0.10")),
                              "MAML": (("8.8", "9.2"), ("0.12", "0.13"))}.items():
        lo, hi = rng_(fam, "train_s"); check_r(f"App. E {fam} train s, low", lo, tr_p[0]); check_r(f"App. E {fam} train s, high", hi, tr_p[1])
        lo, hi = rng_(fam, "adapt_ms"); check_r(f"App. E {fam} week ms, low", lo, ad_p[0]); check_r(f"App. E {fam} week ms, high", hi, ad_p[1])
    nm_ = lambda fam: sorted({v2[t]["timing"][sel_(t, fam)][0]["n_meta"] for t in T4})
    claim("meta-parameters: Meta-LSQ 6-10, GP 5-9, ALPaCA 1842, MAML 1761-1841",
          (nm_("Meta-LSQ")[0], nm_("Meta-LSQ")[-1], nm_("GP")[0], nm_("GP")[-1], nm_("ALPaCA-style"),
           nm_("MAML")[0], nm_("MAML")[-1]) == (6, 10, 5, 9, [1842], 1761, 1841),
          f"LSQ {nm_('Meta-LSQ')} GP {nm_('GP')} ALPaCA {nm_('ALPaCA-style')} MAML {nm_('MAML')}")
    check_r("Results: GP trains in 1.4 s", rng_("GP", "train_s")[1], "1.4")
    claim("Results: Meta-LSQ trains in 5-9 s", 4.5 <= rng_("Meta-LSQ", "train_s")[0] and rng_("Meta-LSQ", "train_s")[1] <= 9.5)
    ratios = [tm_(t, "Meta-LSQ", "adapt_ms") / tm_(t, "GP", "adapt_ms") for t in T4]
    check_r("Results: GP adapts 5x faster (low end)", min(ratios), "5")
    check_r("Results: GP adapts 7x faster (high end)", max(ratios), "7")
    nox = np.array(tr_("NOx")[sel_("NOx", "Meta-LSQ")])
    claim("NOx: fails on three of five seeds (above 2)", int((nox > 2).sum()) == 3)
    for p_, v_ in zip(("0.73", "0.78", "2.79", "5.35", "3.19"), nox):
        check_r("App. E NOx per-seed test error", v_, p_)
    cvn = v2["NOx"]["cv"][sel_("NOx", "Meta-LSQ")]
    check_r("NOx last CV block (1.41)", cvn[-1], "1.41")
    # "The last cross-validation block was hard for every method (1.07 to 2.14, Meta-LSQ 1.41)"
    last_ = [v2["NOx"]["cv"][sel_("NOx", f_)][-1] for f_ in v2["NOx"]["selected"]]
    check_r("NOx last CV block, lowest over selected methods", min(last_), "1.07")
    check_r("NOx last CV block, highest over selected methods", max(last_), "2.14")
    # Sec. 5.5: "for the final design cross-validation chose 1 to 3 times"
    floors_ = sorted({float(sel_(t, "Meta-LSQ").split("floor=")[1]) for t in T4})
    claim(f"selected floors are 1 and 3 times mu* ({floors_})", floors_ == [1.0, 3.0])
    check("weeks per target 53-56", min(v2[t]["n_weeks"] for t in T4), 53, 0)
    check("weeks per target 53-56 (max)", max(v2[t]["n_weeks"] for t in T4), 56, 0)
    # temperatures: "half of the test hours are colder than 95% of the training hours";
    # App. E: 56% on NOx. Recomputed from the data with the experiment's own loader.
    try:
        import importlib.util
        spec_ = importlib.util.spec_from_file_location(
            "rc2", os.path.join(ROOT, "experiments", "real_calibration_v2.py"))
        rc2 = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(rc2)
        cold = {}
        for t in T4:
            tk = rc2.load(*rc2.TARGETS[t]); n_ = int(round(0.6 * len(tk)))
            Ttr = np.concatenate([x[0][:, 1] for x in tk[:n_]])
            Tte = np.concatenate([x[0][:, 1] for x in tk[n_:]])
            cold[t] = float(np.mean(Tte < np.percentile(Ttr, 5)))
        claim("test hours colder than 95% of training hours: about half",
              all(0.45 <= v <= 0.6 for v in cold.values()), str({k: round(v, 2) for k, v in cold.items()}))
        check_r("NOx share of colder test hours (56%)", 100 * cold["NOx"], "56")
    except Exception as e:                           # data or pandas missing
        SKIPS.append(f"temperature check ({e})")

print()
if SKIPS:
    print(f"skipped (missing result files): {', '.join(sorted(set(SKIPS)))}")
if FAILS:
    print(f"\n{len(FAILS)} MISMATCH(ES):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("all checked numbers agree with results/")
