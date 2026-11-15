"""What the operator sees: the ARES screen during tuning, for each method.

One of DESY's 300 trials is replayed with every method, recording every magnet setting.
The trial is chosen by a fixed rule, not by eye: the trial on which Meta-LSQ with
simulator gradients ends closest to its own median over all 300 trials
(results/beam_final_ares.jsonl). Each screen shows the best setting the method has found
so far, rendered from the simulated beam moments on AREABSCR1's pixel grid (2448 x 2040
pixels of 3.32 x 2.45 um), zoomed around the target; the dashed ellipse is the target
beam (2 sigma).

Writes figures/ares_screens/: ares_screens.tex, a TikZ picture that the paper \input's
(beamline, loop, labels and overlays are drawn by LaTeX in the paper's fonts; lengths are
in hundredths of \textwidth), and one PNG per screen, the only raster parts. Also writes
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


# ------------------------------------------------------------------ the TikZ picture
# Lengths along the beam in m, as in ares.beam_on_screen (drifts and magnets up to the screen).
LATTICE = (("drift", 0.17504), ("Q1", 0.122), ("drift", 0.428), ("Q2", 0.122), ("drift", 0.204),
           ("CV", 0.02), ("drift", 0.204), ("Q3", 0.122), ("drift", 0.179), ("CH", 0.02), ("drift", 0.45))
TEX = {"Q1": "$Q_1$", "Q2": "$Q_2$", "Q3": "$Q_3$", "CV": "$C_V$", "CH": "$C_H$"}
FOCUSING = {"Q1": True, "Q2": False, "Q3": True}      # in x, by the sign of the allowed strengths (ares.LOW, ares.HIGH)
ROW_TEX = [("RL agent", "values"), ("BO", "values"), ("Meta-LSQ", "values"),
           ("ALPaCA-style", "+ derivatives"), ("Meta-LSQ", "+ derivatives")]
# layout, in hundredths of \textwidth (y grows upwards, the picture starts at y = 0)
W, GAP, X0, LAST_GAP = 12.4, 1.0, 17.0, 1.2          # screen width, gaps, left edge of the first screen
H = 0.8 * W                                          # the view is +-half by +-0.8 half
TOP, PITCH = -18.8, H + 2.6                          # top of the first row of screens, row pitch (error line below)
AXIS = -7.2                                          # height of the beam axis in the top band


def um(v):
    return (f"{v:.1f}" if v < 100 else f"{v:.0f}") + r"\,\textmu m"


def tikz_beamline():
    x_start, x_end = 7.0, 53.0
    k = (x_end - x_start) / sum(L for _, L in LATTICE)                 # units per metre
    out = [rf"\draw[aresbeam, line width=0.8pt] (0.5,{AXIS}) -- ({x_end:.2f},{AXIS});",
           rf"\draw[aresbeam, line width=0.8pt, -latex] (0.5,{AXIS}) -- (4.6,{AXIS});",
           rf"\node[anchor=south west, text=aresbeam, font=\footnotesize] at (0.5,{AXIS + 1.0:.2f}) {{$e^-$ beam}};"]
    s = 0.0
    for name, L in LATTICE:
        x = x_start + k * (s + L / 2); s += L
        if name == "drift":
            continue
        if name.startswith("Q"):
            r = 3.8
            if FOCUSING[name]:                                         # biconvex lens
                out.append(rf"\filldraw[fill=aresquad, draw=black, line width=0.4pt] ({x:.2f},{AXIS - r}) "
                           rf"to[bend right=34] ({x:.2f},{AXIS + r}) to[bend right=34] ({x:.2f},{AXIS - r}) -- cycle;")
            else:                                                      # biconcave lens
                a = 1.4
                out.append(rf"\filldraw[fill=aresquad, draw=black, line width=0.4pt] ({x - a:.2f},{AXIS - r}) -- "
                           rf"({x + a:.2f},{AXIS - r}) to[bend left=24] ({x + a:.2f},{AXIS + r}) -- "
                           rf"({x - a:.2f},{AXIS + r}) to[bend left=24] ({x - a:.2f},{AXIS - r}) -- cycle;")
        else:                                                          # steerer: a box marked with its kick plane
            out.append(rf"\filldraw[fill=aressteer, draw=black, line width=0.4pt] ({x - 1.15:.2f},{AXIS - 2.2:.2f}) "
                       rf"rectangle ({x + 1.15:.2f},{AXIS + 2.2:.2f});")
            if name == "CV":                                           # kick in the plane of the drawing
                out.append(rf"\draw[line width=0.5pt, latex-latex] ({x:.2f},{AXIS - 3.7:.2f}) -- ({x:.2f},{AXIS + 3.7:.2f});")
            else:                                                      # kick out of the plane of the drawing
                out.append(rf"\filldraw[fill=white, draw=black, line width=0.4pt] ({x:.2f},{AXIS}) circle[radius=0.8];")
                out.append(rf"\fill ({x:.2f},{AXIS}) circle[radius=0.25];")
        out.append(rf"\node[anchor=base, font=\footnotesize] at ({x:.2f},{AXIS + 4.9:.2f}) {{{TEX[name]}}};")
    out.append(rf"\fill[black!85] ({x_end:.2f},{AXIS - 4.6:.2f}) rectangle ({x_end + 0.9:.2f},{AXIS + 4.6:.2f});")
    out.append(rf"\node[anchor=base, font=\footnotesize] at ({x_end + 0.45:.2f},{AXIS + 5.5:.2f}) {{screen}};")
    out.append(rf"\draw[line width=0.5pt, |-|] ({x_start:.2f},{AXIS - 5.6:.2f}) -- ({x_start + 0.5 * k:.2f},{AXIS - 5.6:.2f});")
    out.append(rf"\node[anchor=west] at ({x_start + 0.5 * k + 0.9:.2f},{AXIS - 5.6:.2f}) {{0.5\,m}};")
    return out


def tikz_loop(budget):
    box = (r"draw=black!55, fill=black!4, rounded corners=1.5pt, line width=0.4pt, align=center, "
           r"text width=0.118\textwidth, minimum width=0.126\textwidth, minimum height=0.085\textwidth")
    texts = [("m", 65.3, r"\textbf{measure}\\the beam on\\the screen"),
             ("f", 79.5, r"\textbf{fit}\\one ridge solve\\per parameter"),
             ("p", 93.7, r"\textbf{propose}\\next setting\\from the fits")]
    out = [rf"\node[{box}] ({n}) at ({x},{AXIS - 0.6:.2f}) {{{t}}};" for n, x, t in texts]
    out += [r"\draw[-latex, line width=0.5pt] (m.east) -- (f.west);",
            r"\draw[-latex, line width=0.5pt] (f.east) -- (p.west);",
            rf"\draw[-latex, line width=0.5pt] (p.north) to[bend right=22] node[pos=0.5, above=1.2pt] "
            rf"{{repeat for each of the {budget} measurements}} (m.north);"]
    return out


def write_tikz(folder, paths, target, x, y, half):
    """The picture and its screens. One unit is \textwidth / 100."""
    os.makedirs(folder, exist_ok=True)
    s = W / (2 * half)                                                 # units per metre on a screen
    rx, ry = 2 * target[1] * s, 2 * target[3] * s                      # the target ellipse, 2 sigma
    out = [r"% Written by examples/figure_ares_screen.py. Lengths are in hundredths of \textwidth.",
           r"\providecommand{\aresdir}{figures/ares_screens}",
           r"\begin{tikzpicture}[x=0.01\textwidth, y=0.01\textwidth, line cap=round,",
           r"    every node/.style={inner sep=0pt, outer sep=0pt, font=\scriptsize}]",
           r"\definecolor{aresbeam}{HTML}{B03030}\definecolor{aresquad}{HTML}{B9CCE8}",
           r"\definecolor{aressteer}{HTML}{BFE0C4}\definecolor{arestarget}{HTML}{7FDCFF}",
           r"\definecolor{aresfinal}{HTML}{FFB000}"]
    out += tikz_beamline() + tikz_loop(len(next(iter(paths.values()))[1]))
    col_x = [X0 + c * (W + GAP) + (LAST_GAP if c == len(COLUMNS) else 0) for c in range(len(COLUMNS) + 1)]
    mid = 0.5 * (col_x[0] + col_x[len(COLUMNS) - 1] + W)
    out.append(rf"\node[anchor=base, font=\footnotesize] at ({mid:.2f},{TOP + 3.9:.2f}) "
               r"{best setting found after $n$ measurements};")
    out.append(rf"\node[anchor=base, font=\footnotesize] at ({col_x[-1] + W / 2:.2f},{TOP + 3.9:.2f}) {{final beam,}};")
    for c, m in enumerate(COLUMNS):
        out.append(rf"\node[anchor=base, font=\footnotesize] at ({col_x[c] + W / 2:.2f},{TOP + 1.2:.2f}) {{$n={m}$}};")
    out.append(rf"\node[anchor=base, font=\footnotesize] at ({col_x[-1] + W / 2:.2f},{TOP + 1.2:.2f}) "
               rf"{{differences $\times{MAGNIFY}$}};")
    size = rf"width={W / 100:.4f}\textwidth, height={H / 100:.4f}\textwidth"
    dash = "dash pattern=on 1.7pt off 1.3pt, line width=0.6pt"
    for r, ((label, info, _), (name, uses)) in enumerate(zip(ROWS, ROW_TEX)):
        K, B = paths[(label, info)]
        top = TOP - r * PITCH; cy = top - H / 2
        out.append(rf"\node[anchor=east, align=right, font=\footnotesize] at ({X0 - 1.3},{cy:.2f}) "
                   rf"{{\textbf{{{name}}}\\[-1pt]{{\scriptsize {uses}}}}};")
        for c, m in enumerate(COLUMNS):
            b = B[m - 1]; x0 = col_x[c]; cx = x0 + W / 2
            png = f"screen_{r}_{c}.png"
            plt.imsave(os.path.join(folder, png), screen_image(b, x, y), cmap="inferno", vmin=0, vmax=1, origin="lower")
            out.append(rf"\node[anchor=north west] at ({x0:.2f},{top:.2f}) {{\includegraphics[{size}]{{\aresdir/{png}}}}};")
            out.append(rf"\draw[arestarget, {dash}] ({cx:.2f},{cy:.2f}) ellipse[x radius={rx:.3f}, y radius={ry:.3f}];")
            if abs(b[0] - target[0]) > half or abs(b[2] - target[2]) > 0.8 * half:      # the spot is outside the view
                d = np.array([b[0] - target[0], b[2] - target[2]]); d = d / np.linalg.norm(d)
                out.append(rf"\draw[aresfinal, -latex, line width=0.7pt] ({cx + 0.45 * half * s * d[0]:.2f},"
                           rf"{cy + 0.35 * half * s * d[1]:.2f}) -- ({cx + 0.85 * half * s * d[0]:.2f},"
                           rf"{cy + 0.65 * half * s * d[1]:.2f});")
                out.append(rf"\node[anchor=south west, text=aresfinal] at ({x0 + 0.7:.2f},{top - H + 0.7:.2f}) {{off view}};")
            out.append(rf"\node[anchor=base] at ({cx:.2f},{top - H - 1.85:.2f}) {{{um(mae_um(b, target))}}};")
        # the final beam against the target, position and size errors magnified
        b = B[-1]; x0 = col_x[-1]; cx = x0 + W / 2
        ex, ey = cx + MAGNIFY * (b[0] - target[0]) * s, cy + MAGNIFY * (b[2] - target[2]) * s
        ax_ = 2 * max(target[1] + MAGNIFY * (b[1] - target[1]), 1e-9) * s
        ay_ = 2 * max(target[3] + MAGNIFY * (b[3] - target[3]), 1e-9) * s
        out += [r"\begin{scope}", rf"\clip ({x0:.2f},{top - H:.2f}) rectangle ({x0 + W:.2f},{top:.2f});",
                rf"\fill[black] ({x0:.2f},{top - H:.2f}) rectangle ({x0 + W:.2f},{top:.2f});",
                rf"\draw[arestarget, {dash}] ({cx:.2f},{cy:.2f}) ellipse[x radius={rx:.3f}, y radius={ry:.3f}];",
                rf"\draw[aresfinal, line width=0.8pt] ({ex:.2f},{ey:.2f}) ellipse[x radius={ax_:.3f}, y radius={ay_:.3f}];",
                rf"\draw[aresfinal, line width=0.6pt] ({ex - 0.6:.2f},{ey:.2f}) -- ({ex + 0.6:.2f},{ey:.2f}) "
                rf"({ex:.2f},{ey - 0.6:.2f}) -- ({ex:.2f},{ey + 0.6:.2f});", r"\end{scope}"]
    out.append(r"\end{tikzpicture}")
    path = os.path.join(folder, "ares_screens.tex")
    open(path, "w").write("\n".join(out) + "\n")
    return path


def main():
    os.makedirs(FIG, exist_ok=True)
    index, median, n = pick_trial()
    task = trial_to_task(load_trials(fetch(TRIALS_URL, "trials.yaml"))[index])
    target = np.asarray(task["target"])
    paths = replay(index, task)
    x, y, half = view(target)
    global MAGNIFY
    MAGNIFY = pick_magnification([B[-1] for _, B in paths.values()], target)

    out = write_tikz(os.path.join(FIG, "ares_screens"), paths, target, x, y, half)
    print(f"view +-{half * 1e3:.2f} mm, differences magnified {MAGNIFY} times")

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
