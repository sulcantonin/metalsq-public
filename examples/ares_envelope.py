"""Beam envelope along the ARES section for the Fig. 3 trial: the starting setting and
Meta-LSQ's (with derivatives) best setting after 150 measurements. Tracks the same
Cheetah lattice as ares.beam_on_screen in short steps and checks that the end point
equals ares.beam_on_screen. Writes JSON to OUT.

usage (from metalsq-repo/examples): PYTHONPATH=.. ~/.venvs/metalsq-ares/bin/python ares_envelope.py ../../results/ares_envelope_trial85.json
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import cheetah
import ares
from figure_ares_screen import pick_trial, LOCKED
from ares_vs_rl import TRIALS_URL, fetch, load_trials, trial_to_task
from beam_full import RESULTS, VARIANTS, run_loop
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
OUT = sys.argv[1]
STEP = 0.005                                   # metres per tracking slice


def pieces(task, K):
    """(element constructor per slice, length, label) mirroring ares.beam_on_screen."""
    P = ares.to_physical(K.unsqueeze(0))[0]
    t = torch.tensor
    quad = lambda i, k1: (lambda L: cheetah.Quadrupole(length=t(L), k1=k1,
                                                       misalignment=t(task["misalignment"][i])))
    drift = lambda L: cheetah.Drift(length=t(L))
    vcor = lambda ang, Ltot: (lambda L: cheetah.VerticalCorrector(length=t(L), angle=ang * L / Ltot))
    hcor = lambda ang, Ltot: (lambda L: cheetah.HorizontalCorrector(length=t(L), angle=ang * L / Ltot))
    seq = [(drift, 0.17504), (quad(0, P[0]), 0.122), (drift, 0.428), (quad(1, P[1]), 0.122),
           (drift, 0.204), (vcor(P[2], 0.02), 0.02), (drift, 0.204), (quad(2, P[3]), 0.122),
           (drift, 0.179), (hcor(P[4], 0.02), 0.02), (drift, 0.45)]
    return seq


def envelope(task, K):
    t = torch.tensor
    beam = cheetah.ParameterBeam.from_parameters(
        **{k: t(task[k]) for k in ("energy", "mu_x", "mu_px", "mu_y", "mu_py", "sigma_x",
                                   "sigma_px", "sigma_y", "sigma_py", "sigma_tau", "sigma_p")})
    s, rows = 0.0, [(0.0, float(beam.mu_x), float(beam.sigma_x), float(beam.mu_y), float(beam.sigma_y))]
    for make, L in pieces(task, K):
        n = 1 if L == 0.02 else max(1, int(round(L / STEP)))    # steerers: one element, as in Cheetah
        for _ in range(n):
            beam = cheetah.Segment([make(L / n)]).track(beam)
            s += L / n
            rows.append((s, float(beam.mu_x), float(beam.sigma_x), float(beam.mu_y), float(beam.sigma_y)))
    return np.array(rows)


index, median, n = pick_trial()
task = trial_to_task(load_trials(fetch(TRIALS_URL, "trials.yaml"))[index])
tu = Tuner(ares)
trained = torch.load(os.path.join(RESULTS, "beam_full_ares_dev_models.pt"), weights_only=False)
variant = LOCKED["twin"]["Meta-LSQ"]
gen = torch.Generator().manual_seed(1000 + index)
_, K, y = run_loop(tu, task, trained[variant], VARIANTS[variant][1], gen, 150, return_path=True)
best = int(torch.argmin(torch.as_tensor(y)))
settings = {"start": ares.START.clone(), "tuned": K[best].detach().clone()}
out = {"trial": index, "target_m": [float(v) for v in tu.target(task)], "settings": {}}
for name, Kset in settings.items():
    env = envelope(task, Kset)
    with torch.no_grad():
        ref = ares.beam_on_screen(task, Kset.unsqueeze(0))[0].numpy()
    end = env[-1, [1, 2, 3, 4]]
    print(f"{name}: screen (mu_x, sig_x, mu_y, sig_y) tracked {np.round(end*1e6,2)} um, "
          f"beam_on_screen {np.round(ref*1e6,2)} um, max rel diff {np.max(np.abs(end-ref)/np.maximum(np.abs(ref),1e-9)):.2e}")
    out["settings"][name] = {"knobs": [float(v) for v in Kset], "s": env[:, 0].tolist(),
                             "mu_x": env[:, 1].tolist(), "sigma_x": env[:, 2].tolist(),
                             "mu_y": env[:, 3].tolist(), "sigma_y": env[:, 4].tolist()}
print("trial", index, "target (mu_x, sig_x, mu_y, sig_y) um:", np.round(np.array(out["target_m"]) * 1e6, 1))
json.dump(out, open(OUT, "w"))
print("saved", OUT)
