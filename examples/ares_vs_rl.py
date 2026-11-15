"""ARES tuning: Meta-LSQ against DESY's reinforcement-learning-trained optimiser.

All methods are run on the 300 evaluation trials of Kaiser et al. (Sci. Rep. 2024;
trials.yaml in github.com/desy-ml/rl-vs-bo), each a fixed incoming beam, set of
misalignments and target beam, with that paper's budget of 150 measurements and its
fixed start (Q1, Q2, CV, Q3, CH) = (10, -10, 0, 10, 0).

The RL agent is polished-donkey-996, the TD3 agent of the paper, as published in
github.com/RL4AA/rl4aa25-challenge (models/ea/legacy). It is run exactly through the
paper's wrapper chain (FlattenObservation, PolishedDonkeyCompatibility,
NotVecNormalize, RescaleAction): it sees the measured beam, the magnet settings and the
target, and changes the magnets by at most 3 1/m^2 per quadrupole and 0.3/0.6 mrad per
steerer per step. The other methods follow DESY's Bayesian-optimisation protocol from
examples/ares.py (examples/beamtune.py); Meta-LSQ uses the models meta-learned there on
the rl-vs-bo task distribution, not on these trials.

The simulation, the same for every method: beam parameters are the exact moments of the
tracked beam (no screen image, no finite screen, no noise). On DESY's published simulation
runs (Zenodo 7853721) this reproduces their beam sizes exactly and their beam positions up
to the screen offset of each trial, which is not applied here; the agent's magnets are
clipped to the power-supply limits (+-72 1/m^2, +-6.18 mrad), which the paper's
simulation did not do.

Downloads (to ~/.cache/metalsq, pinned commits): trials.yaml from desy-ml/rl-vs-bo, and
model.zip and vec_normalize.pkl from RL4AA/rl4aa25-challenge (GPL-3.0). The agent files
are read without executing pickled code: network weights with torch's weights-only
loader, normalisation statistics with an unpickler that only builds numpy arrays.

Multi-output variants: every measurement gives the four beam parameters, each gets
its own meta-learned Meta-LSQ fit, and the loop minimises the distance computed from
the fits. The structured GP baseline uses four GPs in exactly the same loop, so only
the surrogate differs. The multi-output design was fixed on our own ARES tasks before
this run; nothing was tuned on the 300 trials.

Result on all 300 trials (best MAE so far in um, median / mean, after 20 / 50 / 100 /
150 measurements):
  RL agent (TD3, 6 million training steps)   7.9/28   5.7/21   4.7/18   4.4/17
  measurements only, one combined distance:
    Meta-LSQ                               195/211  102/114   86/92    77/84
    GP Bayesian optimisation (EI)          258/299  172/198  132/157  102/131
    random in the window                   447/482  350/417  311/382  297/362
  measurements only, four beam parameters:
    Meta-LSQ multi-output                   18/27    11/16    8.4/14   7.5/13
    GP multi-output, same loop              28/48    13/20    9.2/15   8.0/13
  with simulator gradients:
    gradient descent                        90/101   44/53    23/36    14/28
    Meta-LSQ, one combined distance         90/101   22/35    11/26    6.7/24
    Meta-LSQ multi-output                  4.4/14   2.3/10   1.9/9.3  1.5/8.8
From measurements alone the agent has the best median, but multi-output Meta-LSQ has
the better mean (fewer trials left far from the target), beats the structured GP at
every budget at an eighth of its cost (8 s against 68 s per trial), and is meta-learned
in seconds on simulated data rather than trained for millions of steps. With simulator
gradients (a differentiable digital twin, which the agent does not use) multi-output
Meta-LSQ beats the agent at every budget, on median and mean.

    pip install cheetah-accelerator scikit-learn
    python examples/ares_vs_rl.py            # all 300 trials, ~40 min on 13 processes
                                             # (most of it the structured GP baseline)
"""
import io
import os
import pickle
import sys
import time
import urllib.request
import warnings
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

import ares
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
BUDGET, CHECKPOINTS = 150, (20, 50, 100, 150)
CACHE = Path.home() / ".cache" / "metalsq" / "ares"
TRIALS_URL = ("https://raw.githubusercontent.com/desy-ml/rl-vs-bo/"
              "fcd68a54aee7f67ba9c761798c65508126f662ab/trials.yaml")
AGENT_URL = ("https://raw.githubusercontent.com/RL4AA/rl4aa25-challenge/"
             "ba23fa2f6447c8ef74a6a83c3eb7cee8173f7218/models/ea/legacy/polished-donkey-996/")


def fetch(url, name):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        # worker processes may all fetch on a first run: download to a file of our own and
        # rename it, so that no process ever reads a partly written file
        part = path.with_name(f"{name}.{os.getpid()}.part")
        urllib.request.urlretrieve(url, part)
        os.replace(part, path)
    return path


# ---------------------------------------------------------------- trials
def load_trials(path):
    """trials.yaml has three fixed levels (trial, section, value); parsed without yaml."""
    trials, cur, sec = {}, None, None
    for line in open(path):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        if indent == 0:
            cur = trials.setdefault(int(key), {})
        elif indent == 2:
            sec = cur.setdefault(key, {})
        else:
            sec[key] = float(value)
    return [trials[i] for i in sorted(trials)]


def trial_to_task(t):
    i, mis, tg = t["incoming"], t["misalignments"], t["target"]
    return dict(energy=i["energy"], mu_x=i["mu_x"], mu_px=i["mu_xp"], mu_y=i["mu_y"],
                mu_py=i["mu_yp"], sigma_x=i["sigma_x"], sigma_px=i["sigma_xp"],
                sigma_y=i["sigma_y"], sigma_py=i["sigma_yp"], sigma_tau=i["sigma_s"],
                sigma_p=i["sigma_p"],
                misalignment=[(mis[f"q{q}_x"], mis[f"q{q}_y"]) for q in (1, 2, 3)],
                target=[tg["mu_x"], tg["sigma_x"], tg["mu_y"], tg["sigma_y"]])


# ---------------------------------------------------------------- the RL agent
class _Inert:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Builds numpy arrays; every other class becomes an inert attribute holder."""
    NUMPY = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
             ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
             ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
             ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.NUMPY:
            return getattr(__import__(module, fromlist=[name]), name)
        if (module, name) == ("copyreg", "_reconstructor"):
            return lambda cls, base, state: cls()
        if module == "builtins" and name in ("object", "dict", "list", "tuple", "set", "slice"):
            return getattr(__import__("builtins"), name)
        return type(name, (_Inert,), {})


class Agent:
    def __init__(self):
        with zipfile.ZipFile(fetch(AGENT_URL + "model.zip", "polished-donkey-996.zip")) as z:
            state = torch.load(io.BytesIO(z.read("policy.pth")), weights_only=True,
                               map_location="cpu")
        self.net = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 5), nn.Tanh())
        self.net.load_state_dict({k.replace("actor.mu.", ""): v for k, v in state.items()
                                  if k.startswith("actor.mu.")})
        self.net = self.net.float().eval()
        with open(fetch(AGENT_URL + "vec_normalize.pkl", "polished-donkey-996.vecnorm"), "rb") as f:
            vn = _NumpyOnlyUnpickler(f).load().__dict__
        rms = vn["obs_rms"].__dict__
        self.mean = torch.tensor(rms["mean"], dtype=torch.float64)
        self.var = torch.tensor(rms["var"], dtype=torch.float64)
        self.clip, self.eps = float(vn["clip_obs"]), float(vn["epsilon"])
        self.high = torch.tensor([3.0, 3.0, 3.0, 3e-4, 6e-4])

    def act(self, magnets, beam, target):
        """magnets (Q1, Q2, CV, Q3, CH); beam and target (mu_x, sigma_x, mu_y, sigma_y).
        Returns the change of the magnets, (Q1, Q2, CV, Q3, CH)."""
        flat = torch.cat([beam, magnets, target])               # FlattenObservation
        o = flat[[4, 5, 7, 6, 8, 9, 11, 10, 12, 0, 2, 1, 3]]     # PolishedDonkeyCompatibility
        o = ((o - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
        with torch.no_grad():
            a = self.net(o.float().unsqueeze(0))[0].double()
        delta = a * self.high                                   # RescaleAction(-1, 1)
        return delta[[0, 1, 3, 2, 4]]                           # back to (Q1, Q2, CV, Q3, CH)


LIMIT = torch.tensor([72.0, 72.0, 6.1782e-3, 72.0, 6.1782e-3])


def to_knobs(m):
    """Inverse of ares.to_physical, extended linearly beyond the unidirectional box."""
    return 2 * (m - ares.LOW) / (ares.HIGH - ares.LOW) - 1


def run_agent(agent, task, budget, clip=True):
    """clip=False lets the magnets leave their limits, as DESY's simulation did."""
    target = torch.tensor(task["target"])
    m = torch.tensor([10.0, -10.0, 0.0, 10.0, 0.0])
    K, y = [], []
    for _ in range(budget):
        k = to_knobs(m)
        with torch.no_grad():
            beam = ares.beam_on_screen(task, k.unsqueeze(0))[0]
        K.append(k); y.append(float(torch.log((beam - target).norm() * 1e3)))
        m = m + agent.act(m, beam, target)
        if clip:
            m = m.clamp(-LIMIT, LIMIT)
    return torch.stack(K), torch.tensor(y), []


# ---------------------------------------------------------------- the comparison
_W = {}


def _init_worker(value_model, grad_model, multi_value, multi_grad):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")
    _W.update(tuner=Tuner(ares), agent=Agent(), value=value_model, grad=grad_model,
              multi_value=multi_value, multi_grad=multi_grad)


def _run_trial(args):
    index, trial = args
    tu, agent = _W["tuner"], _W["agent"]
    task = trial_to_task(trial)
    run, run_multi = tu.run_protocol, tu.run_protocol_multi
    methods = {
        "RL agent (TD3, DESY)": lambda g: run_agent(agent, task, BUDGET),
        "Meta-LSQ, values only": lambda g: run(task, BUDGET, g, tu.propose_metalsq(_W["value"], False)),
        "GP Bayesian optimisation (EI), values": lambda g: run(task, BUDGET, g, tu.propose_gp),
        "random in the window": lambda g: run(task, BUDGET, g, tu.propose_random),
        "gradient descent, with gradients": lambda g: tu.run_gradient_descent(task, BUDGET, g),
        "Meta-LSQ, with gradients":
            lambda g: run(task, BUDGET, g, tu.propose_metalsq(_W["grad"], True), use_grad=True),
        "Meta-LSQ multi-output, values":
            lambda g: run_multi(task, BUDGET, g, tu.propose_multi(_W["multi_value"], False)),
        "GP multi-output (same loop), values":
            lambda g: run_multi(task, BUDGET, g, tu.propose_gp_multi()),
        "Meta-LSQ multi-output, with gradients":
            lambda g: run_multi(task, BUDGET, g, tu.propose_multi(_W["multi_grad"], True),
                                use_grad=True)}
    out = {}
    for name, method in methods.items():
        t0 = time.perf_counter()
        K, y, _ = method(torch.Generator().manual_seed(1000 + index))
        best = torch.cummin(y, 0).indices
        out[name] = ([tu.mae_um(task, K[best[n - 1]]) for n in CHECKPOINTS],
                     time.perf_counter() - t0)
    return out


def main(n_trials=None, workers=None):
    trials = load_trials(fetch(TRIALS_URL, "trials.yaml"))[:n_trials]
    Agent()                                                     # fetch the agent once
    tu = Tuner(ares)
    local = tu.build_local_pool(n_tasks=800, n_points=70, seed=0)
    grad_model = tu.train(local, True, 30, 0.5, "for the loop, with gradients, local")
    value_model = tu.train(local, False, 50, 0.5, "for the loop, values only, local")
    beam_pool = tu.build_local_beam_pool(n_tasks=800, n_points=70, seed=0)
    multi_grad = tu.train_multi(beam_pool, True, 30, 0.5, "multi-output, with gradients")
    multi_value = tu.train_multi(beam_pool, False, 50, 0.5, "multi-output, values only")
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(value_model, grad_model, multi_value, multi_grad)) as pool:
        results = pool.map(_run_trial, list(enumerate(trials)), chunksize=1)
    print(f"\n{len(trials)} trials, {BUDGET} measurements each, in {time.time() - t0:.0f} s "
          f"on {workers} processes; best MAE so far, um (median / mean)")
    for name in results[0]:
        r = np.asarray([res[name][0] for res in results])
        cols = "  ".join(f"{n:3d}: {np.median(r[:, i]):6.1f} / {np.mean(r[:, i]):6.1f}"
                         for i, n in enumerate(CHECKPOINTS))
        secs = np.mean([res[name][1] for res in results])
        print(f"  {name:<40s} {cols}   ({secs:.1f} s per trial)")


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else None
    main(n_trials=n)
