"""Write figures/ares_screens/ares_scheme.tex: the ARES beamline to scale, two strips with
the beam (centre +- 2 sigma) along it for the Fig. 3 trial (start: grey dashed; tuned by
Meta-LSQ with derivatives: red; target at the screen: cyan), and the surrogate loop.

usage (from the repo root): python3 paper/northern-lights-crc/make_ares_scheme.py results/ares_envelope_trial85.json paper/northern-lights-crc/figures/ares_screens/ares_scheme.tex
"""
import sys, json
import numpy as np

ENV, OUT = sys.argv[1:3]
d = json.load(open(ENV))
UPM = 11.24 / 0.5                         # figure units per metre (the 0.5 m scale bar)
OFF = (12.31 - 0.5) / UPM - (0.17504 + 0.061)   # lattice start relative to the drawn beam start
X = lambda s: 0.5 + UPM * (np.asarray(s) + OFF)
MMU = 2.2                                 # figure units per mm of beam
XC, YC = -17.6, -28.6                      # strip baselines (beam position 0)

def band(v, plane, centre_y):
    s = np.asarray(v["s"]); mu = np.asarray(v["mu_" + plane]) * 1e3; sg = np.asarray(v["sigma_" + plane]) * 1e3
    keep = np.unique(np.r_[np.linspace(0, len(s) - 1, 140).astype(int), len(s) - 1])
    s, mu, sg = s[keep], mu[keep], sg[keep]
    up = [(X(a), centre_y + MMU * (m + 2 * g)) for a, m, g in zip(s, mu, sg)]
    lo = [(X(a), centre_y + MMU * (m - 2 * g)) for a, m, g in zip(s, mu, sg)]
    mid = [(X(a), centre_y + MMU * m) for a, m in zip(s, mu)]
    return up, lo, mid

coords = lambda pts: " ".join(f"({x:.2f},{y:.2f})" for x, y in pts)

lines = [r"""% ARES beamline, beam envelope and surrogate loop (one column). Beamline coordinates as
% in ares_screens.tex (to scale); beam envelopes for the Fig. 3 trial (trial TRIALNO) from the
% Cheetah lattice of examples/ares.py, written by write_ares_scheme.py (16 Nov 2026).
\begin{tikzpicture}[x=0.0178\linewidth, y=0.0178\linewidth, line cap=round,
    every node/.style={inner sep=0pt, outer sep=0pt, font=\scriptsize}]
\definecolor{aresbeam}{HTML}{B03030}\definecolor{aresquad}{HTML}{B9CCE8}
\definecolor{aressteer}{HTML}{BFE0C4}\definecolor{arestarget}{HTML}{1F9BD1}
\draw[aresbeam, line width=0.8pt] (0.5,-7.2) -- (53.00,-7.2);
\draw[aresbeam, line width=0.8pt, -latex] (0.5,-7.2) -- (4.6,-7.2);
\node[anchor=south west, text=aresbeam, font=\footnotesize] at (0.5,-6.20) {$e^-$ beam};
\filldraw[fill=aresquad, draw=black, line width=0.4pt] (12.31,-11.0) to[bend right=34] (12.31,-3.4) to[bend right=34] (12.31,-11.0) -- cycle;
\node[anchor=base, font=\footnotesize] at (12.31,-2.30) {$Q_1$};
\filldraw[fill=aresquad, draw=black, line width=0.4pt] (23.27,-11.0) -- (26.07,-11.0) to[bend left=24] (26.07,-3.4) -- (23.27,-3.4) to[bend left=24] (23.27,-11.0) -- cycle;
\node[anchor=base, font=\footnotesize] at (24.67,-2.30) {$Q_2$};
\filldraw[fill=aressteer, draw=black, line width=0.4pt] (29.70,-9.40) rectangle (32.00,-5.00);
\draw[line width=0.5pt, latex-latex] (30.85,-10.90) -- (30.85,-3.50);
\node[anchor=base, font=\footnotesize] at (30.85,-2.30) {$C_V$};
\filldraw[fill=aresquad, draw=black, line width=0.4pt] (37.04,-11.0) to[bend right=34] (37.04,-3.4) to[bend right=34] (37.04,-11.0) -- cycle;
\node[anchor=base, font=\footnotesize] at (37.04,-2.30) {$Q_3$};
\filldraw[fill=aressteer, draw=black, line width=0.4pt] (41.51,-9.40) rectangle (43.81,-5.00);
\filldraw[fill=white, draw=black, line width=0.4pt] (42.66,-7.2) circle[radius=0.8];
\fill (42.66,-7.2) circle[radius=0.25];
\node[anchor=base, font=\footnotesize] at (42.66,-2.30) {$C_H$};
\fill[black!85] (53.00,-11.80) rectangle (53.90,-2.60);
\node[anchor=base, font=\footnotesize] at (53.45,-1.70) {screen};
\draw[line width=0.5pt, |-|] (7.00,-12.80) -- (18.24,-12.80);
\node[anchor=west] at (19.14,-12.80) {0.5\,m};""".replace("TRIALNO", str(d["trial"]))]

target = np.asarray(d["target_m"]) * 1e3          # mm: mu_x, sig_x, mu_y, sig_y
for plane, cy, tmu, tsg, lab in (("x", XC, target[0], target[1], "horizontal"),
                                 ("y", YC, target[2], target[3], "vertical")):
    # magnet positions as faint guides through the strip
    lines.append(rf"\draw[black!12, line width=0.4pt] (0.5,{cy:.2f}) -- (53.0,{cy:.2f});")
    tu = band(d["settings"]["tuned"], plane, cy)
    st = band(d["settings"]["start"], plane, cy)
    lines.append(rf"\fill[aresbeam, opacity=0.30] plot coordinates {{{coords(tu[0])}}} -- plot coordinates {{{coords(tu[1][::-1])}}} -- cycle;")
    lines.append(rf"\draw[aresbeam, line width=0.5pt] plot coordinates {{{coords(tu[2])}}};")
    lines.append(rf"\draw[black!55, line width=0.45pt, dash pattern=on 1.4pt off 1.1pt] plot coordinates {{{coords(st[0])}}};")
    lines.append(rf"\draw[black!55, line width=0.45pt, dash pattern=on 1.4pt off 1.1pt] plot coordinates {{{coords(st[1])}}};")
    lines.append(rf"\draw[arestarget, line width=1.4pt] (53.45,{cy + MMU * (tmu - 2 * tsg):.2f}) -- (53.45,{cy + MMU * (tmu + 2 * tsg):.2f});")
    lines.append(rf"\node[anchor=west] at (54.4,{cy + MMU * tmu:.2f}) {{${plane}$}};")

# faint guides from each magnet down through both strips
for xm in (12.31, 24.67, 30.85, 37.04, 42.66):
    lines.insert(1 + lines.index(r"\node[anchor=west] at (19.14,-12.80) {0.5\,m};") if False else len(lines),
                 rf"\draw[black!18, line width=0.35pt, dash pattern=on 0.8pt off 1.2pt] ({xm},-11.4) -- ({xm},{YC - 0.4:.2f});")

lines.append(rf"\draw[line width=0.5pt, |-|] (5.6,{XC - MMU / 2:.2f}) -- (5.6,{XC + MMU / 2:.2f});")
lines.append(rf"\node[anchor=east] at (5.0,{XC:.2f}) {{1\,mm}};")
LOOPY = YC - 6.0
lines.append(r"""\begin{scope}[every node/.append style={draw=black!55, fill=black!4, rounded corners=1.5pt,
    line width=0.4pt, align=center, text width=0.27\linewidth, minimum height=0.13\linewidth}]""")
lines.append(rf"\node (m) at (8.5,{LOOPY:.2f}) {{\textbf{{measure}}\\the beam on\\the screen}};")
lines.append(rf"\node (f) at (27.2,{LOOPY:.2f}) {{\textbf{{fit}}\\one ridge solve\\per parameter}};")
lines.append(rf"\node (p) at (45.9,{LOOPY:.2f}) {{\textbf{{propose}}\\next setting\\from the fits}};")
lines.append(r"""\end{scope}
\draw[-latex, line width=0.5pt] (m.east) -- (f.west);
\draw[-latex, line width=0.5pt] (f.east) -- (p.west);
\draw[-latex, line width=0.5pt] (p.south) to[bend left=18] (m.south);
\end{tikzpicture}""")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
