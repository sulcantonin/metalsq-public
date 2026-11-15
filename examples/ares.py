"""Transverse beam tuning in the ARES experimental area (DESY), simulated with Cheetah.

The task is the one DESY used to compare a reinforcement-learning-trained optimiser
with Bayesian optimisation (Kaiser et al., Sci. Rep. 2024; code at
github.com/desy-ml/rl-vs-bo): set three quadrupoles and two steerers so that the beam
on the screen AREABSCR1 has a requested position and size.

    AREAMQZM1 - AREAMQZM2 - AREAMCVM1 - AREAMQZM3 - AREAMCHM1 - AREABSCR1

Geometry is taken from ARESlatticeStage3v1_9.json in the Cheetah repository (element
lengths and drifts between AREASOLA1 and AREABSCR1); the magnet limits, the incoming
beam distribution, the misalignments (+-0.5 mm on the quadrupoles) and the target beam
ranges follow DESY's rl-vs-bo environment. A task is one draw of incoming beam,
misalignments and target beam. The objective is the squared distance between the
measured beam (mu_x, sigma_x, mu_y, sigma_y) and the target, in mm^2; the paper's
reporting metric, the mean absolute error of those four numbers, is printed too.
The finite screen is not simulated.

Part 1  Predict a new task's objective from 30 random measurements.
Part 2  Closed-loop tuning with DESY's Bayesian-optimisation protocol (rl-vs-bo,
        ea_bo.py and bayesopt.py): objective log(||beam - target||), fixed start
        (Q1, Q2, CV, Q3, CH) = (10, -10, 0, 10, 0), five random samples in a window
        of +-5% of each magnet range around the start, then each step proposes a
        setting inside the same window around the last setting, 100 measurements
        in total. Every method follows this protocol, so with values only the
        surrogate is the only difference between GP expected improvement and
        Meta-LSQ (lower confidence bound). With simulator gradients, Meta-LSQ is
        compared with normalised gradient descent. Meta-LSQ is meta-learned on
        local neighbourhoods of random settings, the regime the window works in.

An earlier version used random starting settings and a search over the whole
magnet range; there every method stayed far from the target (Meta-LSQ level with
GP from values, behind gradient descent with gradients), because a fit over the
whole range is a poor guide to the narrow region near the target. The protocol and
all its settings were fixed on separate development tasks before the run reported
by this script.

Result on the 20 held-out tasks (median MAE after 20 / 50 / 100 measurements, um):
  values only     Meta-LSQ 382 / 267 / 220   GP-EI 576 / 328 / 245   random 842 / 746 / 746
  with gradients  Meta-LSQ 164 /  55 /  55   gradient descent 282 / 146 / 59
Meta-LSQ steps cost 5-7 ms against 26 ms for the GP. Every method is still far from
the best reachable MAE (median 0.1 um), so 100 measurements are a short budget here.

The harness (measurements, meta-training, both parts, all baselines) is in
examples/beamtune.py and is shared with examples/lcls_diag0.py.

    pip install cheetah-accelerator scikit-learn
    python examples/ares.py
"""
import sys
import warnings

import torch
import cheetah

from beamtune import Tuner

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
DIM = 5
# knob order Q1, Q2, CV, Q3, CH; limits as in rl-vs-bo/environment.py
LOW = torch.tensor([0.0, -72.0, -6.1782e-3, 0.0, -6.1782e-3])
HIGH = torch.tensor([72.0, 0.0, 6.1782e-3, 72.0, 6.1782e-3])


def to_physical(K):
    """Knobs in [-1, 1] to magnet settings (k1 in 1/m^2, kicks in rad)."""
    return LOW + (K + 1) / 2 * (HIGH - LOW)


# ---------------------------------------------------------------- tasks
def sample_task(gen):
    """Incoming beam and quadrupole misalignments as in rl-vs-bo/backend.py, target beam
    as in rl-vs-bo/environment.py (all uniform)."""
    u = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=gen))
    return dict(
        energy=u(80e6, 160e6), mu_x=u(-1e-3, 1e-3), mu_px=u(-1e-4, 1e-4),
        mu_y=u(-1e-3, 1e-3), mu_py=u(-1e-4, 1e-4), sigma_x=u(1e-5, 5e-4),
        sigma_px=u(1e-6, 5e-5), sigma_y=u(1e-5, 5e-4), sigma_py=u(1e-6, 5e-5),
        sigma_tau=u(1e-6, 5e-5), sigma_p=u(1e-4, 1e-3),
        misalignment=[(u(-5e-4, 5e-4), u(-5e-4, 5e-4)) for _ in range(3)],
        target=[u(-2e-3, 2e-3), u(0, 2e-3), u(-2e-3, 2e-3), u(0, 2e-3)])


def beam_on_screen(task, K):
    """(mu_x, sigma_x, mu_y, sigma_y) on AREABSCR1 for knob settings K (n, 5)."""
    t = torch.tensor
    P = to_physical(K)
    beam = cheetah.ParameterBeam.from_parameters(
        **{k: t(task[k]) for k in ("energy", "mu_x", "mu_px", "mu_y", "mu_py", "sigma_x",
                                   "sigma_px", "sigma_y", "sigma_py", "sigma_tau",
                                   "sigma_p")})
    quad = lambda i, k1: cheetah.Quadrupole(length=t(0.122), k1=k1,
                                            misalignment=t(task["misalignment"][i]))
    drift = lambda L: cheetah.Drift(length=t(L))
    line = cheetah.Segment([
        drift(0.17504), quad(0, P[:, 0]), drift(0.428), quad(1, P[:, 1]), drift(0.204),
        cheetah.VerticalCorrector(length=t(0.02), angle=P[:, 2]), drift(0.204),
        quad(2, P[:, 3]), drift(0.179),
        cheetah.HorizontalCorrector(length=t(0.02), angle=P[:, 4]), drift(0.45)])
    out = line.track(beam)
    return torch.stack([out.mu_x, out.sigma_x, out.mu_y, out.sigma_y], -1)


START = torch.tensor([-1 + 2 * 10 / 72, -1 + 2 * 62 / 72, 0.0, -1 + 2 * 10 / 72, 0.0])
HALF = 0.1          # window half-width in knob units: +-5% of each magnet range


if __name__ == "__main__":
    Tuner(sys.modules[__name__]).main()
