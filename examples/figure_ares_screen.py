"""What the operator sees: the ARES screen during tuning, for each method.

One of DESY's 300 trials is replayed with every method, recording every magnet setting.
The trial is chosen by a fixed rule, not by eye: the trial on which Meta-LSQ with
simulator gradients ends closest to its own median over all 300 trials
(results/beam_final_ares.jsonl). Each screen shows the best setting the method has found
so far, rendered from the simulated beam moments on AREABSCR1's pixel grid (2448 x 2040
pixels of 3.32 x 2.45 um), zoomed around the target; the dashed ellipse is the target
beam (2 sigma).

Writes figures/ares_screens.png (schematic, model loop and filmstrip) and
figures/ares_screens.gif (the screens side by side over the 150 measurements).

    python examples/figure_ares_screen.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

import ares
from ares_vs_rl import TRIALS_URL, Agent, fetch, load_trials, run_agent, trial_to_task
from beam_full import RESULTS, VARIANTS, run_loop
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
LOCKED = json.load(open(os.path.join(RESULTS, "beam_full_locked.json")))["ares"]
ROWS = [("DESY's RL agent", "values only", "rl"),
        ("DESY's Bayesian opt.", "values only", "bo"),
        ("Meta-LSQ", "values only", ("values", "Meta-LSQ")),
        ("ALPaCA-style", "with simulator gradients", ("twin", "ALPaCA-style")),
        ("Meta-LSQ", "with simulator gradients", ("twin", "Meta-LSQ"))]
COLUMNS = (1, 10, 20, 50, 150)
MAGNIFY = 50
PIX = (3.3198e-6, 2.4469e-6)
KNOBS = ("Q1", "Q2", "CV", "Q3", "CH")


def pick_trial():
    done = {}
    for line in open(os.path.join(RESULTS, "beam_final_ares.jsonl")):
        r = json.loads(line)
        done[r["index"]] = {**done.get(r["index"], {}), **r["result"]}
    idx = sorted(i for i in done if "Meta-LSQ, twin" in done[i])
    final = np.array([done[i]["Meta-LSQ, twin"][0][-1] for i in idx])
    j = int(np.argmin(np.abs(final - np.median(final))))
    return idx[j], float(np.median(final)), len(idx)


def replay(index, task):
    """Every method's 150 settings on this trial, with the seeds of the final run."""
    tu = Tuner(ares)
    trained = torch.load(os.path.join(RESULTS, "beam_full_ares_dev_models.pt"), weights_only=False)
    paths = {}
    for label, info, spec in ROWS:
        gen = torch.Generator().manual_seed(1000 + index)
        if spec == "rl":
            K, y, _ = run_agent(Agent(), task, 150)
        elif spec == "bo":
            from bo_desy import run_bo_desy
            K, y = run_bo_desy(tu, task, gen, 150)
        else:
            variant = LOCKED[spec[0]][spec[1]]
            _, K, y = run_loop(tu, task, trained[variant], VARIANTS[variant][1], gen, 150, return_path=True)
        best = torch.cummin(torch.as_tensor(y), 0).indices
        with torch.no_grad():
            B = ares.beam_on_screen(task, K[best]).numpy()      # best-so-far beam after each measurement
        paths[(label, info)] = (K[best].numpy(), B)
    return paths


def mae_um(b, target):
    return 1e6 * float(np.mean(np.abs(b - target)))


def screen_image(b, x, y):
    gx = np.exp(-0.5 * ((x - b[0]) / max(b[1], PIX[0])) ** 2)
    gy = np.exp(-0.5 * ((y - b[2]) / max(b[3], PIX[1])) ** 2)
    return np.outer(gy, gx)


def view(target):
    half = max(4 * max(target[1], target[3]), 0.25e-3)
    cx, cy = target[0], target[2]
    x = cx + np.arange(-half, half, PIX[0] * max(1, int(2 * half / PIX[0] / 300)))
    y = cy + np.arange(-half * 0.8, half * 0.8, PIX[1] * max(1, int(1.6 * half / PIX[1] / 240)))
    return x, y, half


def draw_screen(ax, b, target, x, y, half, label=None):
    ax.imshow(screen_image(b, x, y), extent=(x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3),
              origin="lower", cmap="inferno", vmin=0, vmax=1, aspect="equal")
    ax.add_patch(patches.Ellipse((target[0] * 1e3, target[2] * 1e3), 4e3 * target[1], 4e3 * target[3],
                                 fill=False, ls="--", lw=1.3, ec="#7fdcff"))
    off = abs(b[0] - target[0]) > half or abs(b[2] - target[2]) > 0.8 * half
    if off:                                      # the spot is outside the zoomed view
        d = np.array([b[0] - target[0], b[2] - target[2]]); d = d / np.linalg.norm(d)
        ax.annotate("", xy=((target[0] + 0.85 * half * d[0]) * 1e3, (target[2] + 0.65 * half * d[1]) * 1e3),
                    xytext=((target[0] + 0.45 * half * d[0]) * 1e3, (target[2] + 0.35 * half * d[1]) * 1e3),
                    arrowprops=dict(arrowstyle="->", color="#ffb000", lw=1.5))
        ax.text(0.5, 0.08, "beam off view", transform=ax.transAxes, ha="center", color="#ffb000", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    if label is not None:
        ax.text(0.03, 0.93, label, transform=ax.transAxes, color="white", fontsize=8, va="top")


def pick_magnification(finals, target):
    """Largest of 50, 25, 20, 10 (else 5) at which every final beam's magnified size stays
    between 0.25 and 2 times the target size."""
    for m in (50, 25, 20, 10, 5):
        ok = all(0.25 <= 1 + m * (b[k] - target[k]) / target[k] <= 2 for b in finals for k in (1, 3))
        if ok:
            return m
    return 5


def draw_magnified(ax, b, target, half):
    """Final beam outline against the target with position and size errors magnified."""
    ax.set_facecolor("black")
    t = target
    ax.add_patch(patches.Ellipse((0, 0), 4e3 * t[1], 4e3 * t[3], fill=False, ls="--", lw=1.3, ec="#7fdcff"))
    cx, cy = MAGNIFY * (b[0] - t[0]) * 1e3, MAGNIFY * (b[2] - t[2]) * 1e3
    w = 4e3 * max(t[1] + MAGNIFY * (b[1] - t[1]), 1e-6)
    h = 4e3 * max(t[3] + MAGNIFY * (b[3] - t[3]), 1e-6)
    ax.add_patch(patches.Ellipse((cx, cy), w, h, fill=False, lw=1.8, ec="#ffb000"))
    ax.plot([cx], [cy], "+", color="#ffb000", ms=8)
    ax.set_xlim(-half * 1e3, half * 1e3); ax.set_ylim(-0.8 * half * 1e3, 0.8 * half * 1e3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{mae_um(b, target):.1f} µm off", fontsize=8, labelpad=2)


def draw_schematic(ax, knobs=None):
    ax.set_xlim(0, 10); ax.set_ylim(-1.6, 1.6); ax.axis("off")
    ax.plot([0.2, 9.0], [0, 0], color="#cc3333", lw=2, zorder=1)
    ax.annotate("", xy=(0.9, 0), xytext=(0.2, 0), arrowprops=dict(arrowstyle="->", color="#cc3333", lw=2))
    ax.text(0.2, 0.35, "electron beam", fontsize=8, color="#cc3333")
    xs = {"Q1": 2.0, "Q2": 3.4, "CV": 4.6, "Q3": 5.8, "CH": 7.0}
    for k, xk in xs.items():
        if k.startswith("Q"):
            ax.add_patch(patches.Ellipse((xk, 0), 0.35, 1.4, fc="#4c72b0", ec="k", zorder=2))
            ax.text(xk, 0.85, k, ha="center", fontsize=9)
        else:
            ax.add_patch(patches.Rectangle((xk - 0.15, -0.45), 0.3, 0.9, fc="#55a868", ec="k", zorder=2))
            ax.text(xk, 0.6, k, ha="center", fontsize=9)
        v = 0.0 if knobs is None else float(knobs[KNOBS.index(k)])
        ax.add_patch(patches.Circle((xk, -1.15), 0.28, fc="white", ec="k"))
        a = np.pi / 2 - v * 2.4
        ax.plot([xk, xk + 0.24 * np.cos(a)], [-1.15, -1.15 + 0.24 * np.sin(a)], color="k", lw=1.5)
    ax.add_patch(patches.Rectangle((9.0, -0.9), 0.25, 1.8, fc="#333333", ec="k"))
    ax.text(9.12, 1.0, "screen", ha="center", fontsize=8)
    ax.text(4.5, 1.45, "blue: focusing magnets   green: steerers", ha="center", fontsize=8, color="#444")
    ax.text(4.5, -1.62, "five dials (magnet settings) to choose", ha="center", fontsize=8, style="italic")


def draw_loop(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    boxes = [(0.3, 6.4, "1  measure", "the beam\non the screen"),
             (3.55, 6.4, "2  fit", "one least-squares\nsolve, no training"),
             (6.8, 6.4, "3  propose", "next dial setting\nfrom the fit")]
    for x0, y0, title, body in boxes:
        ax.add_patch(patches.FancyBboxPatch((x0, y0), 2.9, 3.0, boxstyle="round,pad=0.1", fc="#f4f4f4", ec="#555"))
        ax.text(x0 + 1.45, y0 + 2.3, title, ha="center", fontsize=9, weight="bold")
        ax.text(x0 + 1.45, y0 + 1.05, body, ha="center", va="center", fontsize=8)
    for x0 in (3.25, 6.5):
        ax.annotate("", xy=(x0 + 0.3, 7.9), xytext=(x0, 7.9), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(1.75, 9.75), xytext=(8.25, 9.75),
                arrowprops=dict(arrowstyle="->", lw=1.3, connectionstyle="arc3,rad=0.12"))
    ax.text(5.0, 11.1, "repeat for every measurement", ha="center", fontsize=7.5, color="#444")
    # the idea of the fit: measured points with their slopes, and the fitted curve
    t = np.linspace(0, 1, 200)
    f = lambda u: 0.6 * np.sin(5 * u) + 0.4 * u
    X0, Y0, W, H = 2.0, 0.6, 6.0, 3.6
    ax.plot(X0 + W * t, Y0 + H * (0.5 + 0.45 * f(t)), color="#c44e52", lw=1.5)
    for u in (0.12, 0.38, 0.7, 0.9):
        px, py = X0 + W * u, Y0 + H * (0.5 + 0.45 * f(u))
        s = H * 0.45 * (3 * np.cos(5 * u) + 0.4) / W
        dx = 0.45 / np.sqrt(1 + s ** 2)
        ax.plot(px, py, "o", color="k", ms=4)
        ax.annotate("", xy=(px + dx, py + s * dx), xytext=(px - dx, py - s * dx),
                    arrowprops=dict(arrowstyle="-", lw=2.2, color="#4c72b0"))
    ax.text(5.0, 0.45, "the fit uses each measurement's value (dot) and,\nwith the simulator, its slope (blue)",
            ha="center", fontsize=7.5, style="italic")


def main():
    os.makedirs(FIG, exist_ok=True)
    index, median, n = pick_trial()
    task = trial_to_task(load_trials(fetch(TRIALS_URL, "trials.yaml"))[index])
    target = np.asarray(task["target"])
    paths = replay(index, task)
    x, y, half = view(target)
    global MAGNIFY
    MAGNIFY = pick_magnification([B[-1] for _, B in paths.values()], target)

    fig = plt.figure(figsize=(14.5, 12))
    gs = fig.add_gridspec(1 + len(ROWS), len(COLUMNS) + 2, height_ratios=[1.5] + [1] * len(ROWS),
                          width_ratios=[1.2] + [1] * len(COLUMNS) + [1.05], hspace=0.3, wspace=0.05)
    draw_schematic(fig.add_subplot(gs[0, :4]))
    draw_loop(fig.add_subplot(gs[0, 4:]))
    for r, (label, info, _) in enumerate(ROWS):
        K, B = paths[(label, info)]
        lab = fig.add_subplot(gs[r + 1, 0]); lab.axis("off")
        lab.text(0.95, 0.55, label, ha="right", va="center", fontsize=11, weight="bold")
        lab.text(0.95, 0.35, info, ha="right", va="center", fontsize=8.5, color="#555")
        for c, m in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[r + 1, c + 1])
            draw_screen(ax, B[m - 1], target, x, y, half)
            ax.set_xlabel(f"{mae_um(B[m - 1], target):.1f} µm off", fontsize=8, labelpad=2)
            if r == 0:
                ax.set_title(f"after {m} measurement{'s' if m > 1 else ''}", fontsize=9)
        ax = fig.add_subplot(gs[r + 1, len(COLUMNS) + 1])
        draw_magnified(ax, B[-1], target, half)
        if r == 0:
            ax.set_title(f"remaining error\nmagnified {MAGNIFY}x", fontsize=9)
    fig.suptitle("Tuning the ARES beam: make the spot on the screen fill the dashed target", fontsize=13, y=0.93)
    fig.text(0.5, 0.005, f"DESY trial {index}, chosen by rule: the trial on which Meta-LSQ with gradients ends "
             f"closest to its median over {n} trials ({median:.1f} um). Each screen shows the best setting found "
             f"so far, zoomed to +-{half * 1e3:.2f} mm around the target; numbers are the mean absolute error of "
             f"the four beam parameters. Right: the final beam (solid) against the target (dashed) with the "
             f"differences in position and size magnified {MAGNIFY} times.", ha="center", fontsize=7.5, wrap=True)
    out = os.path.join(FIG, "ares_screens.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    frames = list(range(1, 31)) + list(range(32, 151, 3))
    fig, axes = plt.subplots(1, len(ROWS), figsize=(16, 3.6))
    def frame(i):
        m = frames[i]
        for ax, (label, info, _) in zip(axes, ROWS):
            ax.clear()
            B = paths[(label, info)][1]
            draw_screen(ax, B[m - 1], target, x, y, half)
            ax.set_title(f"{label}\n{info}", fontsize=9)
            ax.set_xlabel(f"{mae_um(B[m - 1], target):.1f} µm off", fontsize=9)
        fig.suptitle(f"after {m} measurement{'s' if m > 1 else ''}  (dashed: target beam)", fontsize=11)
        return []
    FuncAnimation(fig, frame, frames=len(frames)).save(os.path.join(FIG, "ares_screens.gif"),
                                                      writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig)
    print(f"trial {index} (median {median:.2f} um over {n}); wrote {out} and ares_screens.gif")
    for (label, info), (K, B) in paths.items():
        print(f"  {label}, {info}: " + "  ".join(f"{m}: {mae_um(B[m - 1], target):.1f}" for m in COLUMNS))


if __name__ == "__main__":
    main()
