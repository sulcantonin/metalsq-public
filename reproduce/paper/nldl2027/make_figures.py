"""Render the NLDL figures from the captured results/ data.

Same contract as paper/full/make_figures.py: the experiment scripts stay
plotting-free and write results/*.json|*.npz; this is the only file that
imports matplotlib.  It differs from the full-paper version in three ways:

  * figure sizes target the NLDL two-column layout (a4paper, 2.5 cm margins,
    so \\linewidth = 3.08 in and \\textwidth = 6.30 in) rather than a
    one-column article;
  * several one-panel figures are merged into wide multi-panel ones to fit
    the 8-page limit (``mechanism``, ``regime``);
  * fonts are Computer Modern to match the 10 pt NLDL body text.

Run from the repo root:  python3 paper/northern-lights/make_figures.py
Each figure is guarded on its input file, so a partial results/ still renders
whatever is available.
"""

import os
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Computer Modern to match the NLDL 10 pt body text.  cmr10 has no U+2212, so
# unicode_minus must be off or every negative tick label warns and renders as a
# missing glyph.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.linewidth": 0.6,
    # Type 42 (TrueType) rather than matplotlib's default Type 3: Type 3 fonts
    # in figures are rejected by many venues and defeat pdfTeX font expansion.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL = 3.08    # \linewidth  (in) in the NLDL two-column layout
WIDE = 6.30   # \textwidth  (in), for figure*

# Consistent method colours across figures.
C_MLP = "#888888"
C_STRONG = "#444444"
C_LSQ = "#1f77b4"
C_SOB = "#d62728"
C_GRN = "#2ca02c"


def _exists(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        warnings.warn(f"missing {name}; skipping its figure")
        return None
    return p


def _load(name):
    p = _exists(name)
    return None if p is None else json.load(open(p))


def _save(fig, name):
    out = os.path.join(FIGDIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] wrote {os.path.relpath(out, ROOT)}")


def _pt(x):
    """Point estimate from a scalar or a {mean,std,seeds} dict."""
    return x["mean"] if isinstance(x, dict) and "mean" in x else x


def _sd(x):
    return float(x.get("std", 0.0)) if isinstance(x, dict) else 0.0


def _smooth_log(y, w):
    """Geometric (log-space) moving average; robust to the per-batch spikes in
    the training meta-loss without distorting the convergence trend."""
    y = np.clip(np.asarray(y, float), 1e-12, None)
    if len(y) < 5:
        return y
    w = max(3, (w | 1))                       # force odd, >= 3
    pad = w // 2
    lyp = np.pad(np.log10(y), pad, mode="edge")
    k = np.ones(w) / w
    return 10 ** np.convolve(lyp, k, mode="valid")


# ---------------------------------------------------------------- teaser
def _teaser_fit(d):
    """Rebuild the shown task's stacked design and fit from the stored scalars.

    teaser.json keeps the task, the learned (sigma, mu, w) and the basis seed; the
    frozen draw is regenerated with the experiment's own make_basis, so this panel
    shows the very fit whose curve teaser_demo.py stored (agreement to ~1e-4)."""
    import sys
    import torch
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from experiments.maml_sinusoid import make_basis
    c = d["config"]
    Wb, bb = make_basis(N=200, seed=c["seed"])
    Wb, bb = Wb.double().numpy()[0], bb.double().numpy()[0]
    W = c["sigma"] * Wb
    xs = np.asarray(d["support_x"]); ys = np.asarray(d["support_y"])
    dys = np.asarray(d["support_dy"]); sw = np.sqrt(c["grad_w"])
    N = W.size
    H = np.sin(np.outer(xs, W) + bb) / np.sqrt(N)
    G = W * np.cos(np.outer(xs, W) + bb) / np.sqrt(N)
    A = np.vstack([H, sw * G]); t = np.concatenate([ys, sw * dys])
    beta = np.linalg.solve(A.T @ A + c["mu"] * np.eye(N), A.T @ t)
    g = np.asarray(d["grid"])
    u = (np.sin(np.outer(g, W) + bb) / np.sqrt(N)) @ beta
    du = (W * np.cos(np.outer(g, W) + bb) / np.sqrt(N)) @ beta
    return A, u, du, beta, t


def fig_teaser():
    """Fig. 1: the method on one real task, then what it buys in training time.

    One colour per kind of information throughout: black for values (and the value
    rows H), green for slopes (and the slope rows Gamma), red for the fit."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    d = _load("teaser.json")
    if d is None:
        return
    A, u, du, beta, tvec = _teaser_fit(d)
    g = np.asarray(d["grid"]); true = np.asarray(d["true"])
    xs = np.asarray(d["support_x"]); ys = np.asarray(d["support_y"])
    dys = np.asarray(d["support_dy"])
    amp = d["shown_task_amplitude"]
    true_d = np.gradient(true, g)
    C_VAL, C_TRUTH, C_TRUTH_D = "#1a1a1a", "#d4d4d4", "#cfe8cf"

    fig = plt.figure(figsize=(WIDE, 1.45))
    # an empty column between (a) and (b) leaves room for the row labels of (b)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.1, 0.9, 1.0, 1.05], wspace=0.5,
                          left=0.015, right=0.9, top=0.83, bottom=0.30)
    a1, a2, a3, a4 = (fig.add_subplot(gs[0, i]) for i in (0, 2, 3, 4))

    def title(ax, letter, text):
        ax.set_title(r"$\mathbf{(" + letter + r")}$ " + text, fontsize=7.6, loc="left", pad=4)

    def clean(ax):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(-5.4, 5.4); ax.set_xticks([-5, 0, 5])
        ax.set_ylim(-1.25 * amp, 2.15 * amp)
        ax.set_xlabel("$x$", labelpad=0)
        ax.tick_params(length=2, pad=1.5)

    # (a) the task: a value fixes a point of the curve, a slope fixes its direction
    # there. The values-only Meta-LSQ fit of the same task (its own meta-learned
    # scalars, stored by teaser_demo.py) goes through every point but bends away
    # from the truth where the points are sparse.
    C_VONLY = "#9467bd"                      # same colour as "values only" in (d)
    vonly = np.asarray(d["meta_lsq_valueonly"])
    a1.plot(g, true, color=C_TRUTH, lw=1.8, zorder=1, solid_capstyle="round")
    a1.fill_between(g, true, vonly, color=C_VONLY, alpha=0.28, lw=0, zorder=1.5)
    a1.plot(g, vonly, color=C_VONLY, lw=1.0, ls=(0, (3.2, 1.6)), zorder=2)
    h = 0.62
    for x0, y0, s0 in zip(xs, ys, dys):
        n = h / np.sqrt(1 + (s0 / amp * 2.5) ** 2)   # roughly equal length on screen
        a1.plot([x0 - n, x0 + n], [y0 - s0 * n, y0 + s0 * n], color=C_GRN, lw=2.0,
                zorder=3, solid_capstyle="round")
    a1.plot(xs, ys, "o", color=C_VAL, ms=3.6, mec="white", mew=0.6, zorder=4)
    clean(a1)
    i_gap = np.argmax(np.abs(vonly - true) * (g < xs.min() + 4))
    a1.annotate("values\nonly", (g[i_gap], vonly[i_gap]),
                xytext=(-5.3, -0.7 * amp), textcoords="data", fontsize=6.0,
                color=C_VONLY, ha="left", va="center", linespacing=0.9,
                arrowprops=dict(arrowstyle="-", color=C_VONLY, lw=0.6,
                                shrinkA=1, shrinkB=1.5))
    a1.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=C_VAL, ms=3.6,
                                  mec="white", mew=0.6),
                       plt.Line2D([], [], color=C_GRN, lw=2.0)],
              labels=["value", "slope"], ncol=2, frameon=False, fontsize=6.4,
              loc="upper center", handlelength=1.2, columnspacing=1.2,
              borderaxespad=0.0, handletextpad=0.4)
    title(a1, "a", "values and slopes of one task")

    # (b) the whole system A beta = t: value rows over slope rows, the coefficients,
    # and the targets, whose 20 cells line up with the rows of A. One support point's
    # value row, slope row and their two targets are outlined.
    show = 20
    M = A[:, :show]
    lim = np.percentile(np.abs(M), 98)
    xb0, xb1 = show + 3.6, show + 5.6          # beta strip
    xt0, xt1 = show + 9.2, show + 11.2         # t strip
    a2.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto",
              interpolation="nearest", extent=(0, show, 20, 0))
    bl = np.percentile(np.abs(beta[:show]), 98)
    a2.imshow(beta[:show, None], cmap="RdBu_r", vmin=-bl, vmax=bl, aspect="auto",
              interpolation="nearest", extent=(xb0, xb1, 20, 0))
    tl = np.abs(tvec).max()
    a2.imshow(tvec[:, None], cmap="RdBu_r", vmin=-tl, vmax=tl, aspect="auto",
              interpolation="nearest", extent=(xt0, xt1, 20, 0))
    for x0_, x1_ in ((0, show), (xt0, xt1)):
        a2.plot([x0_, x1_], [10, 10], color="white", lw=2.0)
    for xm, sym in (((show + xb0) / 2, r"$\times$"), ((xb1 + xt0) / 2, r"$=$")):
        a2.text(xm, 10, sym, ha="center", va="center", fontsize=7.6)
    r = 2                                       # the outlined support point
    for y0, col in ((r, C_VAL), (10 + r, C_GRN)):
        for x0_, x1_ in ((0, show), (xt0, xt1)):
            a2.add_patch(Rectangle((x0_, y0), x1_ - x0_, 1, fill=False, ec=col, lw=0.9,
                                   zorder=5))
    for spine in a2.spines.values():
        spine.set_visible(False)
    a2.set_xticks([]); a2.set_yticks([]); a2.grid(False)
    for y0, col, lab in ((0, C_VAL, r"$H$"), (10, C_GRN, r"$\sqrt{w}\,\Gamma$")):
        a2.add_patch(Rectangle((-2.2, y0 + 0.4), 1.2, 9.2, color=col, clip_on=False,
                               transform=a2.transData))
        a2.text(-3.0, y0 + 5, lab, color=col, fontsize=7.2, ha="right", va="center")
    for y0, col, lab in ((0, C_VAL, r"$y$"), (10, C_GRN, r"$\sqrt{w}\,g$")):
        a2.text(xt1 + 0.6, y0 + 5, lab, color=col, fontsize=6.6, ha="left", va="center")
    a2.set_xlim(-0.01, xt1 + 6.0)               # room for the y and sqrt(w) g labels
    a2.set_ylim(20, -0.01)
    a2.set_xlabel(r"$\beta^\ast = (A^{\!\top}\!A + \mu I)^{-1} A^{\!\top} t$",
                  labelpad=4, fontsize=7.2)
    title(a2, "b", r"one system $A\beta = t$")

    # (c) the fitted function and its closed-form derivative, over the truth. The
    # values-only fit also has a derivative; it is drawn dashed to show that the slope
    # rows are what make u_N' right. teaser.json stores that fit only as a curve on a
    # 400-point grid, so its derivative is taken by central differences there (on the
    # Sobolev fit this agrees with the closed form to 0.04, against errors up to 1.3).
    dvonly = np.gradient(vonly, g)
    a3.plot(g, true, color=C_TRUTH, lw=3.2, zorder=1, solid_capstyle="round")
    a3.plot(g, true_d, color=C_TRUTH_D, lw=3.2, zorder=1, solid_capstyle="round")
    a3.fill_between(g, true_d, dvonly, color=C_VONLY, alpha=0.28, lw=0, zorder=1.5)
    a3.plot(g, dvonly, color=C_VONLY, lw=1.1, ls=(0, (3.2, 1.6)), zorder=2)
    a3.plot(g, u, color=C_SOB, lw=1.25, zorder=3)
    a3.plot(g, du, color=C_GRN, lw=1.25, zorder=3)
    a3.plot(xs, ys, "o", color=C_VAL, ms=2.6, mec="white", mew=0.5, zorder=4)
    clean(a3)
    a3.set_ylim(-1.25 * amp, 2.6 * amp)          # room for the two-row legend
    a3.legend(handles=[plt.Line2D([], [], color=C_SOB, lw=1.25),
                       plt.Line2D([], [], color=C_GRN, lw=1.25),
                       plt.Line2D([], [], color=C_VONLY, lw=1.0, ls=(0, (3.2, 1.6)))],
              labels=[r"fit $u_N$", r"$u_N'$", r"$u_N'$ from values only"], ncol=2,
              frameon=False, fontsize=6.0, loc="upper center", handlelength=1.1,
              columnspacing=0.8, borderaxespad=0.0, handletextpad=0.3)
    title(a3, "c", "fit and its derivative")

    # (d) the same training budget, spent two ways
    wc = _load("wallclock_budget.json")
    if wc is not None:
        su = wc["summary"]
        B = [0.5, 5.0, 50.0]
        series = [("MLP-MAML", "MLP-MAML", C_MLP, "--", "o"),
                  ("Meta-LSQ-vanilla", "Meta-LSQ,\nvalues only", "#9467bd", "-", "s"),
                  ("Meta-LSQ-Sobolev", "Meta-LSQ,\nwith slopes", C_SOB, "-", "o")]
        for key, lab, col, ls, mk in series:
            v = [su[key][str(b)]["mse_mean"] for b in B]
            a4.plot(B, v, ls, color=col, lw=1.4, marker=mk, ms=3.4, mec="white",
                    mew=0.5, zorder=4)
            dy = {"Meta-LSQ-vanilla": 3, "Meta-LSQ-Sobolev": -3}.get(key, 0)
            a4.annotate(lab, (B[-1], v[-1]), textcoords="offset points",
                        xytext=(5, dy), fontsize=6.2, color=col, ha="left",
                        va="center", linespacing=0.95)
        sob05 = su["Meta-LSQ-Sobolev"]["0.5"]["mse_mean"]
        a4.axhline(sob05, color=C_SOB, lw=0.8, ls=":", alpha=0.7, zorder=2)
        a4.set_xscale("log"); a4.set_yscale("log")
        a4.set_xlim(0.36, 75); a4.set_ylim(4e-3, 11.0)
        a4.set_xticks([0.5, 5, 50]); a4.set_xticklabels(["0.5", "5", "50"])
        a4.minorticks_off()
        a4.set_yticks([1e-2, 1e-1, 1, 10])
        a4.set_xlabel("training time (s)", labelpad=1)
        a4.set_ylabel("test MSE", labelpad=1)
        a4.spines["top"].set_visible(False); a4.spines["right"].set_visible(False)
        a4.grid(False)
        a4.grid(alpha=0.18, which="major", lw=0.4, axis="y")
        a4.tick_params(length=2, pad=1.5)
        title(a4, "d", "error vs. training time")

    # inner loop: (a) -> (b) -> (c)
    for src, dst, lab in ((a1, a2, "stack"), (a2, a3, "solve")):
        pa, pb = src.get_position(), dst.get_position()
        x0, x1 = pa.x1 + 0.008, pb.x0 - (0.075 if dst is a2 else 0.008)
        y = pa.y0 + 0.5 * pa.height
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=9,
                                       color="#666666", lw=1.0))
        fig.text((x0 + x1) / 2, y + 0.03, lab, ha="center", va="bottom",
                 fontsize=6.2, color="#666666", style="italic")

    # outer loop: query error at new points back to (sigma, mu, w)
    p1, p3 = a1.get_position(), a3.get_position()
    fig.text((p1.x0 + p3.x1) / 2 + 0.01, 0.055,
             r"outer loop over tasks:  query error of $u_N$  $\rightarrow$  gradient "
             r"through the solve  $\rightarrow$  Adam step on "
             r"$\log\sigma, \log\mu, \log w$",
             ha="center", va="center", fontsize=6.4, color="#9c1c1f",
             bbox=dict(boxstyle="round,pad=0.45", fc="#fbecec", ec="none"))
    _save(fig, "teaser.pdf")


# ---------------------------------------------------------------- converge
def fig_converge():
    p = _exists("maml_sinusoid.npz")
    if p is None:
        return
    d = np.load(p)
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    series = [
        ("curve_mlp", "MLP-MAML (standard)", C_MLP, "o"),
        ("curve_fl",  "Meta-LSQ (vanilla)",  C_LSQ, "s"),
        ("curve_fls", "Meta-LSQ (Sobolev)",  C_SOB, "^"),
    ]
    ps = _exists("maml_strong.npz")
    strong = np.load(ps)["curve"] if ps is not None else None
    # x = meta-iteration (column 1), which is deterministic; the wall-clock
    # column is machine-load-dependent and not used here.
    for key, label, c, m in series:
        if key not in d.files:
            continue
        cv = d[key]
        w = max(7, len(cv) // 6)
        sm = _smooth_log(cv[:, 2], w)
        ax.plot(cv[:, 1], cv[:, 2], color=c, lw=0.5, alpha=0.16)          # raw
        ax.plot(cv[:, 1], sm, color=c, lw=1.4, label=label,               # smoothed
                marker=m, markersize=3, markevery=max(1, len(cv) // 8))
    if strong is not None:
        ws = max(7, len(strong) // 6)
        ax.plot(strong[:, 1], strong[:, 2], color=C_STRONG, lw=0.5, alpha=0.16)
        ax.plot(strong[:, 1], _smooth_log(strong[:, 2], ws), color=C_STRONG,
                lw=1.2, ls="--", label="MLP-MAML (strong FOMAML)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(3e-7, 30)          # headroom below the curves for the legend
    ax.set_xlabel("meta-iteration")
    ax.set_ylabel("meta-loss (query MSE)")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2,
              handlelength=1.6, borderaxespad=0.2)
    _save(fig, "converge.pdf")


# ---------------------------------------------------------------- mechanism
def fig_mechanism():
    """Effect of the derivative rows: query value/grad/Hess on held-out tasks,
    a fixed-w sweep, effective rank and function recovery at K=10."""
    rev = _load("review.json")
    er = _load("eff_rank.json")
    if rev is None or er is None:
        return
    fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(WIDE, 1.75))

    # (a) derivative-aware query metrics: value-only vs Sobolev
    dm = rev["derivative_metrics"]
    mets = ["val", "grad", "hess"]
    van = [_pt(dm["vanilla"][m]) for m in mets]
    sob = [_pt(dm["sobolev"][m]) for m in mets]
    x = np.arange(3)
    w = 0.36
    a1.bar(x - w / 2, van, w, color=C_MLP, label="value-only ($w{=}0$)")
    a1.bar(x + w / 2, sob, w, color=C_SOB, label="Sobolev ($w^\\ast$)")
    for xi, v, s in zip(x, van, sob):
        a1.text(xi, v * 1.25, f"{v / s:.1f}$\\times$", ha="center", fontsize=6)
    a1.set_yscale("log")
    a1.set_xticks(x)
    a1.set_xticklabels(["value", "grad", "Hess"])
    a1.set_ylim(min(sob) / 2.2, max(van) * 12)
    a1.set_ylabel("query MSE (5 seeds)")
    a1.set_title("(a) held-out query error")
    a1.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2,
              borderaxespad=0.2, labelspacing=0.2)

    # (b) fixed-w sweep
    ws = [r["w"] for r in rev["w_sweep"]]
    a2.plot(ws, [_pt(r["val_mse"]) for r in rev["w_sweep"]], "s-",
            color=C_SOB, lw=1.3, ms=3.2, label="value")
    a2.plot(ws, [_pt(r["grad_mse"]) for r in rev["w_sweep"]], "o-",
            color=C_LSQ, lw=1.3, ms=3.2, label="gradient")
    a2.set_yscale("log")
    a2.set_xscale("symlog", linthresh=0.1, linscale=0.6)
    a2.set_xticks([0, 0.1, 1, 10]); a2.set_xticklabels(["0", "0.1", "1", "10"])
    a2.set_xlim(-0.02, 13)
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    a2.yaxis.set_major_locator(FixedLocator([0.02, 0.04, 0.08]))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    a2.yaxis.set_minor_locator(NullLocator()); a2.yaxis.set_minor_formatter(NullFormatter())
    a2.set_xlabel("gradient weight $w$")
    a2.set_ylabel("query MSE")
    a2.set_title("(b) $w$ held fixed")
    a2.legend(frameon=False, fontsize=6.5)

    # (c) effective rank at the 10-shot support set: value block vs stacked
    det = er["detail"]
    K = er["config"]["K"]
    sg = [r["sigma"] for r in det]
    a3.plot(sg, [r["eff_rank_valonly_1e-06"]["mean"] for r in det], "o-",
            color=C_MLP, lw=1.3, ms=3.2, label="values $H$")
    a3.plot(sg, [r["eff_rank_stacked_w10.9_1e-06"]["mean"] for r in det], "s-",
            color=C_SOB, lw=1.3, ms=3.2, label="stacked $[H;\\sqrt{w}\\Gamma]$")
    for lev, lab in ((K, f"$K{{=}}{K}$"), (2 * K, f"$2K{{=}}{2 * K}$")):
        a3.axhline(lev, color="black", ls=":", lw=0.8)
        a3.text(5.4, lev, lab, ha="right", va="bottom", fontsize=5.8)
    a3.axvspan(0.5, 0.8, color=C_GRN, alpha=0.10, lw=0)
    a3.set_xscale("log")
    a3.set_ylim(0, 2 * K * 1.25)
    a3.set_xlabel("bandwidth $\\sigma$")
    a3.set_ylabel("effective rank (tol $10^{-6}$)")
    a3.set_title("(c) rank at $K{=}10$")
    a3.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.2,
              borderaxespad=0.2, labelspacing=0.2)

    # (d) held-out recovery of an in-span target, relative RMSE
    for key, mk, lab in (("val", "o", "value"), ("der", "s", "derivative")):
        a4.plot(sg, [r[f"relrmse_{key}_valonly_mu1e-05"]["mean"] for r in det],
                mk + "--", color=C_MLP, lw=1.1, ms=3.0)
        a4.plot(sg, [r[f"relrmse_{key}_stacked_w10.9_mu1e-05"]["mean"] for r in det],
                mk + "-", color=C_SOB, lw=1.3, ms=3.0)
    a4.plot([], [], "o", color="gray", ms=3.0, label="value")
    a4.plot([], [], "s", color="gray", ms=3.0, label="derivative")
    a4.plot([], [], "--", color=C_MLP, lw=1.1, label="values only")
    a4.plot([], [], "-", color=C_SOB, lw=1.3, label="stacked")
    a4.axvspan(0.5, 0.8, color=C_GRN, alpha=0.10, lw=0)
    a4.set_xscale("log")
    a4.set_yscale("log")
    a4.set_xlabel("bandwidth $\\sigma$")
    a4.set_ylabel("held-out relative RMSE")
    a4.set_title("(d) function recovery")
    a4.set_ylim(top=40.0)
    a4.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=2,
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.2,
              columnspacing=0.6)

    fig.tight_layout(pad=0.25, w_pad=1.1)
    _save(fig, "mechanism.pdf")


# ---------------------------------------------------------------- ooc
def fig_ooc():
    data = _load("ooc_multiseed.json") or _load("ooc.json")
    if data is None:
        return
    fams = list(data.keys())
    # Bars = mean; overlaid tick = median.  The mean/median gap visualises the
    # heavy-tailed extrapolation blowups, and how the ridge floor removes them.
    methods = [("mlp", "MLP-MAML", C_MLP),
               ("meta_lsq_sob", "Meta-LSQ-Sob ($\\mu^\\ast$)", C_SOB),
               ("meta_lsq_sob_robust", "Meta-LSQ-Sob ($\\mu\\!\\geq\\!10^{-2}$)", C_LSQ)]
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    x = np.arange(len(fams))
    w = 0.26
    for i, (key, label, c) in enumerate(methods):
        means = [_pt(data[f][key]["mean"]) for f in fams]
        meds = [_pt(data[f][key]["median"]) for f in fams]
        xpos = x + (i - 1) * w
        ax.bar(xpos, means, w, color=c, label=label, alpha=0.85)
        for xc, md in zip(xpos, meds):
            ax.plot([xc - w / 2, xc + w / 2], [md, md], color="black", lw=1.1,
                    solid_capstyle="butt", zorder=5)
    ax.plot([], [], color="black", lw=1.1, label="median (per task)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["in-class\n(sin)", "OOC-A\n(tanh)", "OOC-B\n(triangle)"],
                       fontsize=6.5)
    ax.set_ylabel("test MSE (bar $=$ mean)")
    ax.set_ylim(1e-5, 4e6)         # headroom above the bars for the legend
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=5.8,
              handlelength=1.3, borderaxespad=0.15, labelspacing=0.25)
    _save(fig, "ooc.pdf")


# ---------------------------------------------------------------- regime
def fig_regime():
    """Where the method lives: dimension scaling, anisotropy, spectrum tails."""
    dim = _load("dim_scan.json")
    ani = _load("aniso_multiseed.json") or _load("aniso.json")
    mat = _load("matern_ooc.json")
    if dim is None or ani is None or mat is None:
        return
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(WIDE, 2.0))

    # (a) isotropic-RFF scaling with ambient dimension
    ds = [r["d"] for r in dim]
    a1.plot(ds, [r["fl"]["val"] for r in dim], "o-", color=C_SOB, lw=1.3, ms=3.2,
            label="value")
    a1.plot(ds, [r["fl"]["grad"] for r in dim], "s-", color=C_LSQ, lw=1.3, ms=3.2,
            label="gradient")
    a1.plot(ds, [r["fl"]["hess"] for r in dim], "^-", color=C_GRN, lw=1.3, ms=3.2,
            label="Hessian")
    mlp_d = [r["d"] for r in dim if "mlp" in r]
    if mlp_d:
        a1.plot(mlp_d, [r["mlp"]["val"] for r in dim if "mlp" in r], "o:",
                color=C_MLP, lw=1.1, ms=2.8, label="MLP value")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xticks(ds)
    a1.set_xticklabels([str(x) for x in ds])
    a1.set_xlabel("ambient dimension $d$")
    a1.set_ylabel("relative RMSE")
    a1.set_title("(a) error vs dimension (held-out $\\sigma$)")
    a1.legend(frameon=False, fontsize=6, loc="upper left", ncol=2,
              columnspacing=0.8, handlelength=1.4)

    # (b) bandwidth parameterisation on an anisotropic family
    order = [k for k in ["scalar", "diag", "full"] if k in ani]
    nl = [ani[k]["n_learnable"] for k in order]
    mse = [_pt(ani[k]["test_mse"]) for k in order]
    sd = [_sd(ani[k]["test_mse"]) for k in order]
    bars = a2.bar(range(len(order)), mse, yerr=sd, capsize=2.5, width=0.55,
                  color=[C_MLP, C_LSQ, C_SOB][:len(order)],
                  error_kw={"lw": 0.8})
    a2.set_yscale("log")
    a2.set_xticks(range(len(order)))
    a2.set_xticklabels([f"{k}\n({n} params)" for k, n in zip(order, nl)], fontsize=6.5)
    a2.set_ylabel("test MSE (anisotropic family)")
    a2.set_title("(b) per-coordinate bandwidth")
    for b, m, s in zip(bars, mse, sd):
        a2.text(b.get_x() + b.get_width() / 2, m + s, f"{m:.2f}",
                ha="center", va="bottom", fontsize=6)
    a2.set_ylim(top=max(m + s for m, s in zip(mse, sd)) * 3)

    # (c) frequency-distribution tails, median OOC error (+ the in-class cost)
    order = [k for k in ["normal", "studentt_nu5", "studentt_nu2", "laplace"] if k in mat]
    labels = {"normal": "Gauss.", "studentt_nu5": "$t_5$",
              "studentt_nu2": "$t_2$", "laplace": "Lapl."}
    fams = [("OOC-B (triangle)", C_SOB, "triangle"),
            ("OOC-A (tanh)", C_LSQ, "tanh"),
            ("in-class (sin)", C_GRN, "in-class")]
    x = np.arange(len(order))
    w = 0.27
    lo = np.inf
    for i, (fam, c, lab) in enumerate(fams):
        med = [_pt(mat[k][fam]["median"]) for k in order]
        lo = min(lo, min(med))
        a3.bar(x + (i - 1) * w, med, w, color=c, label=lab)
    mlp = _load("ooc_multiseed.json")
    if mlp is not None:
        y = _pt(mlp["OOC-B (triangle)"]["mlp"]["median"])
        a3.axhline(y, color="black", ls="--", lw=0.9)
        a3.text(len(order) - 0.55, y * 1.25, "MLP-MAML (triangle)", ha="right",
                va="bottom", fontsize=5.8)
    a3.set_yscale("log")
    a3.set_xticks(x)
    a3.set_xticklabels([labels[k] for k in order])
    a3.set_ylim(lo / 3, 3e2)
    a3.set_xlabel("frequency distribution $p(\\omega)$")
    a3.set_ylabel("test MSE (median)")
    a3.set_title("(c) heavier tails: rough targets")
    a3.legend(frameon=False, fontsize=6, ncol=3, columnspacing=0.7,
              handlelength=1.0, loc="upper left", borderaxespad=0.2)

    fig.tight_layout(pad=0.25, w_pad=1.3)
    _save(fig, "regime.pdf")


# ---------------------------------------------------------------- noise
def fig_noise():
    rows = _load("noise_sweep.json")
    if rows is None:
        return
    nz = [r["noise"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.75))
    a1.plot(nz, [_pt(r["mu_star"]) for r in rows], "o-", color=C_SOB, lw=1.3, ms=3.2,
            label="$\\mu^\\ast$")
    a1.axhline(1e-2, color="black", ls="--", lw=0.9, label="floor $10^{-2}$")
    a1.set_yscale("log")
    a1.set_xlabel("support-noise std")
    a1.set_ylabel("meta-learned ridge $\\mu^\\ast$")
    a1.legend(frameon=False, fontsize=6, loc="lower right")
    a2.errorbar(nz, [_pt(r["meta_lsq_sob_mse"]) for r in rows],
                yerr=[_sd(r["meta_lsq_sob_mse"]) for r in rows],
                fmt="s-", color=C_SOB, lw=1.3, ms=3.2, capsize=2, label="Meta-LSQ-Sob")
    a2.errorbar(nz, [_pt(r["mlp_mse"]) for r in rows],
                yerr=[_sd(r["mlp_mse"]) for r in rows],
                fmt="o-", color=C_MLP, lw=1.3, ms=3.2, capsize=2, label="MLP-MAML")
    a2.set_yscale("log")
    a2.set_xlabel("support-noise std")
    a2.set_ylabel("test MSE (clean query)")
    a2.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.2, w_pad=1.0)
    _save(fig, "noise.pdf")


# ---------------------------------------------------------------- sobolev cost
def fig_sobolev():
    data = _load("sobolev.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=False)
    for ax, dk, metric, ylab in zip(axes, ["d4", "d8"], ["val", "val"],
                                    ["value rRMSE", None]):
        blk = data[dk]
        groups = [("FastLSQ", "fastlsq_vanilla", "fastlsq_sobolev", C_SOB),
                  ("MLP", "mlp_vanilla", "mlp_sobolev", C_MLP)]
        x = np.arange(len(groups))
        w = 0.34
        for j, (label, vk, sk, c) in enumerate(groups):
            ax.bar(x[j] - w / 2, blk[vk][metric], w, color=c, alpha=0.45,
                   label="value-only" if j == 0 else None)
            ax.bar(x[j] + w / 2, blk[sk][metric], w, color=c,
                   label="$+$Sobolev" if j == 0 else None)
            r = blk[vk][metric] / blk[sk][metric]
            ax.text(x[j], max(blk[vk][metric], blk[sk][metric]) * 1.2,
                    f"{r:.1f}$\\times$", ha="center", fontsize=6)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_title(f"$d={dk[1:]}$")
        ax.set_ylim(top=max(blk[g[1]][metric] for g in groups) * 4)
        if ylab:
            ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(pad=0.2, w_pad=1.0)
    _save(fig, "sobolev.pdf")


# ---------------------------------------------------------------- calibration
def fig_calibration():
    data = _load("calibration.json")
    if data is None:
        return
    entry = max(data, key=lambda e: e["noise"])
    nominal = {"cov50": 50, "cov68": 68, "cov90": 90, "cov95": 95}
    order = ["cov50", "cov68", "cov90", "cov95"]
    convs = [("unit", "$\\sigma_y^2{=}1$", C_MLP),
             ("bo_code", "$\\sigma_y^2{=}\\mu$", C_LSQ),
             ("emp_bayes", "residual plug-in $\\hat\\sigma_y^2$", C_SOB)]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    xs = [nominal[k] for k in order]
    ax.plot([40, 100], [40, 100], color="black", ls=":", lw=0.9)
    ax.text(86, 80, "perfect", rotation=33, fontsize=6, ha="center", va="center")
    for key, lab, c in convs:
        ax.plot(xs, [100 * _pt(entry[key][k]) for k in order], "o-",
                color=c, lw=1.3, ms=3.2, label=lab)
    ax.set_xlabel("nominal coverage (%)")
    ax.set_ylabel("empirical coverage (%)")
    ax.set_title(f"posterior calibration (noise $=$ {entry['noise']})")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.91),
              handlelength=1.6, labelspacing=0.3)
    _save(fig, "calibration.pdf")


# ---------------------------------------------------------------- bo
def fig_bo():
    pnpz = _exists("bo.npz")
    summ = _load("bo.json")
    if pnpz is None or summ is None:
        return
    arrs = np.load(pnpz)
    probs = [("branin", "Branin ($d{=}2$)"), ("hartmann6", "Hartmann-6 ($d{=}6$)")]
    methods = [("random", "Random", C_MLP),
               ("meta_lsq", "Meta-LSQ $+$ EI", C_SOB),
               ("gp", "GP $+$ EI", C_LSQ)]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85))
    for ax, (pkey, ptitle) in zip(axes, probs):
        tmin = summ[pkey]["true_min"]
        for mkey, label, c in methods:
            arrname = f"{pkey}_{mkey}"
            if arrname not in arrs.files:
                continue
            reg = np.maximum(arrs[arrname] - tmin, 1e-12)   # (n_seeds, n_iters)
            xs = np.arange(1, reg.shape[1] + 1)
            ax.plot(xs, reg.mean(0), color=c, lw=1.3, label=label)
            ax.fill_between(xs, np.percentile(reg, 25, 0), np.percentile(reg, 75, 0),
                            color=c, alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("evaluations")
        ax.set_title(ptitle, fontsize=7)
    axes[0].set_ylabel("simple regret")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout(pad=0.2, w_pad=1.0)
    _save(fig, "bo.pdf")


def main():
    fig_teaser()
    fig_converge()
    fig_mechanism()
    fig_ooc()
    fig_regime()
    fig_noise()
    fig_sobolev()
    fig_calibration()
    fig_bo()
    print("done.")


if __name__ == "__main__":
    main()
