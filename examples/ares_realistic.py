"""ARES tuning under realistic measurement conditions (simulation only).

Same 300 DESY trials, budget, start and protocols as examples/ares_vs_rl.py, but every
measurement now behaves more like a screen on a real machine:
  - noise: each beam parameter (mu_x, sigma_x, mu_y, sigma_y) is read with Gaussian
    noise of standard deviation 5% of its value plus 5 um;
  - finite screen: AREABSCR1 shows +-4.06 mm by +-2.50 mm (2448 x 2040 pixels of
    3.32 x 2.45 um, the values of rl-vs-bo's ARES backend). If the beam centre is off
    the screen the reading is DESY's placeholder (0, 3.5, 0, 2.2), in metres like every
    other reading, as in rl-vs-bo's backend.py (get_beam_parameters).
Every method, the RL agent included, sees the same kind of readings. Each method picks
its best setting from its own noisy readings, as it would on a machine; that setting is
then scored by its true, noise-free mean absolute error to the target. The multi-output
fits leave off-screen readings out (they still count against the budget). Meta-LSQ is
meta-trained on noisy simulated readings of the rl-vs-bo task distribution. Gradient
variants are left out: a real machine gives no gradients.

The noise level and the off-screen rule were fixed before any run; the second final-pick
rule below was not.

Final pick. By default each method's final setting is the one with the best noisy
reading. Methods with a surrogate are also scored with a second rule on the same runs:
refit the surrogate on the readings so far and take the measured setting it predicts to
be best, which does not reward a lucky reading. This rule was added after the first run
showed multi-output Meta-LSQ's mean rising from 100 to 150 measurements; it is standard
Bayesian-optimisation practice and has no free parameter, but it was chosen after
looking at these 300 trials, not before.

Result on all 300 trials (true MAE of the chosen setting in um, median / mean, after
20 / 50 / 100 / 150 measurements; one run):
  RL agent (TD3, DESY), best reading        29/45    25/37    24/35    24/35
  measurements only, one combined distance:
    Meta-LSQ, best reading                 260/313  203/232  148/187  132/162
    Meta-LSQ, surrogate pick               541/596  393/464  248/339  194/278
    GP Bayesian optimisation, best reading 327/425  266/378  224/306  194/272
    GP Bayesian optimisation, surr. pick   328/410  267/346  232/303  212/284
    random in the window                   447/487  351/424  310/387  295/381
  measurements only, four beam parameters:
    Meta-LSQ multi-output, best reading     37/72    24/47    21/44    19/56
    Meta-LSQ multi-output, surrogate pick   32/65    17/37    13/33    10/32
    GP multi-output, best reading          103/199   56/141   47/126   43/132
    GP multi-output, surrogate pick        104/195   51/133   44/121   40/112
Noise costs every method accuracy (noise-free: agent 4.4/17, multi-output Meta-LSQ
7.5/13 after 150). With the surrogate pick, multi-output Meta-LSQ beats the agent on the
median from 50 measurements on (10 against 24 um after 150) and on the mean from 100
on (32 against 35); the agent is ahead after 20 measurements (29/45 against 32/65) and
level on the mean after 50. The pick helps both four-parameter surrogates but not the
one-distance fits. A likely reason, not tested here: the log distance has a cusp at the
target that a smooth fit rounds off, so it misjudges which measured setting is closest,
while the four beam parameters are smooth in the magnet settings and the distance is
only formed afterwards.

Validation of the surrogate pick on 200 fresh tasks (--fresh 200: seed 2027, drawn
like DESY's trials, unseen by meta-training; same metric, one run):
  RL agent (TD3, DESY), best reading        30/41    26/33    24/31    25/31
  Meta-LSQ, one distance, best reading     281/324  209/257  155/207  135/180
  Meta-LSQ, one distance, surrogate pick   554/614  417/483  241/364  184/299
  GP Bayesian optimisation, best reading   367/547  292/656  243/425  204/640
  GP Bayesian optimisation, surr. pick     367/413  292/352  258/323  224/298
  Meta-LSQ multi-output, best reading       37/78    22/53    19/41    18/39
  Meta-LSQ multi-output, surrogate pick     32/75    16/48    12/33    10/31
  GP multi-output, best reading            120/243   58/172   49/155   47/148
  GP multi-output, surrogate pick          118/242   57/170   46/153   44/145
The pick again lowers multi-output Meta-LSQ's error at every budget, on median and
mean, and again does not help the one-distance Meta-LSQ fit. Against the agent the
median result holds (10 against 25 um after 150) but the mean result does not: the
mean is level after 150 measurements (31 against 31) and behind before that. So the
safe claim is a lower median from 50 measurements on, with a comparable mean at the
full budget.

DESY's trials are narrower than the environment's sampling space. On 200 fresh tasks
from the full space (--fresh 200 --wide: target sizes up to 2 mm, misalignments up to
+-0.5 mm, which ares.sample_task and meta-training use) every method is worse and the
agent most of all (after 150: agent 188/573, multi-output Meta-LSQ with surrogate pick
20/126); the agent may not have been trained on that wider space, so this is not a
fair comparison with it.

    pip install cheetah-accelerator scikit-learn
    python examples/ares_realistic.py [--trials N]   # DESY's 300 trials
    python examples/ares_realistic.py --fresh 200 [--wide]   # new tasks, seed 2027
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import torch

import ares
from ares_vs_rl import Agent, LIMIT, TRIALS_URL, fetch, load_trials, to_knobs, trial_to_task
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
BUDGET, CHECKPOINTS = 150, (20, 50, 100, 150)
REL_NOISE, ABS_NOISE = 0.05, 5e-6                # of each beam parameter, in m
SCREEN = (2448 / 2 * 3.3198e-6, 2040 / 2 * 2.4469e-6)   # half-widths, m (rl-vs-bo)
PLACEHOLDER = torch.tensor([0.0, 3.5, 0.0, 2.2])


def observe(task, K, gen):
    """Noisy screen readings (n, 4) in metres and an on-screen mask (n,)."""
    with torch.no_grad():
        b = ares.beam_on_screen(task, K)
    noisy = b + (REL_NOISE * b.abs() + ABS_NOISE) * torch.randn(b.shape, generator=gen)
    noisy[:, [1, 3]] = noisy[:, [1, 3]].abs()
    on = (b[:, 0].abs() < SCREEN[0]) & (b[:, 2].abs() < SCREEN[1])
    noisy[~on] = PLACEHOLDER
    return noisy, on


class RealisticTuner(Tuner):
    """Tuner whose measurements are noisy screen readings; values only."""

    def __init__(self, beamline, noise_seed=0):
        super().__init__(beamline)
        self.noise = torch.Generator().manual_seed(noise_seed)

    def log_distance(self, task, K, grad=False):
        b, _ = observe(task, K, self.noise)
        return torch.log((b - self.target(task)).norm(dim=-1) * 1e3), None

    def run_protocol_multi(self, task, budget, gen, propose, use_grad=False):
        K = torch.cat([self.bl.START.unsqueeze(0), self.window(self.bl.START, 4, gen)])
        B, on = observe(task, K, self.noise)
        B = B * 1e3
        t_fit = []
        while len(B) < budget:
            y = self.log_distance_from_beam(B, task)
            t0 = time.perf_counter()
            if on.sum() >= 3:            # fit on-screen readings only
                k = propose(K[on], B[on], None, y[on], K[-1], gen, self.target(task) * 1e3)
            else:
                k = self.window(K[-1], 1, gen)[0]
            t_fit.append(time.perf_counter() - t0)
            bk, onk = observe(task, k.unsqueeze(0), self.noise)
            K, B, on = torch.cat([K, k.unsqueeze(0)]), torch.cat([B, bk * 1e3]), torch.cat([on, onk])
        self.readings = (B, on)              # kept for the surrogate's final pick
        return K, self.log_distance_from_beam(B, task), t_fit

    # ---------------------------------------------------------------- final pick
    # The default pick is the setting with the best noisy reading. With a surrogate the
    # alternative is to refit it on the first n readings and take the measured setting
    # it predicts to be best, which does not reward a lucky reading.
    @staticmethod
    def _gp(K, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        return GaussianProcessRegressor(
            ConstantKernel() * RBF([0.5] * K.shape[1], (1e-3, 1e2))
            + WhiteKernel(1e-4, (1e-8, 1e-1)),
            normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(K.numpy(), y.numpy())

    def pick_scalar(self, K, y, n, model=None):
        K, y = K[:n], y[:n]
        if model is not None:
            with torch.no_grad():
                pred = model.adapt(K, y, None)(K)
        else:
            pred = torch.tensor(self._gp(K, y).predict(K.numpy()))
        return K[pred.argmin()]

    def pick_multi(self, K, task, n, models=None):
        B, on = self.readings
        K, B, on = K[:n], B[:n], on[:n]
        if on.sum() < 3:
            return K[self.log_distance_from_beam(B, task).argmin()]
        Ko, Bo = K[on], B[on]
        if models is not None:
            with torch.no_grad():
                pred = torch.stack([m.adapt(Ko, Bo[:, i], None)(Ko) for i, m in enumerate(models)], -1)
        else:
            pred = torch.stack([torch.tensor(self._gp(Ko, Bo[:, i]).predict(Ko.numpy()))
                                for i in range(4)], -1)
        return Ko[(pred - self.target(task) * 1e3).norm(dim=-1).argmin()]

    # noisy meta-training data around the start; multi-output pool drops off-screen readings
    def build_noisy_pools(self, n_tasks, n_points, seed, half=0.3):
        gen = torch.Generator().manual_seed(seed)
        scalar, beam = [], []
        for _ in range(n_tasks):
            task = self.bl.sample_task(gen)
            K = (self.bl.START.unsqueeze(0)      # around the start, where every run begins
                 + half * (torch.rand(n_points, self.bl.DIM, generator=gen) * 2 - 1)).clamp(-1, 1)
            b, on = observe(task, K, gen)
            y = torch.log((b - self.target(task)).norm(dim=-1) * 1e3)   # placeholders kept,
            scalar.append((K, y, torch.zeros(len(K), self.bl.DIM)))       # as the loop sees them
            if on.sum() >= 60:
                beam.append((K[on], b[on] * 1e3, torch.zeros(int(on.sum()), 4, self.bl.DIM)))
        return scalar, beam


def run_agent_noisy(agent, task, budget, gen):
    target = torch.tensor(task["target"])
    m = torch.tensor([10.0, -10.0, 0.0, 10.0, 0.0])
    K, y = [], []
    for _ in range(budget):
        k = to_knobs(m)
        beam = observe(task, k.unsqueeze(0), gen)[0][0]
        K.append(k); y.append(float(torch.log((beam - target).norm() * 1e3)))
        m = (m + agent.act(m, beam, target)).clamp(-LIMIT, LIMIT)
    return torch.stack(K), torch.tensor(y), []


_W = {}


def _init_worker(scalar_model, multi_models):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")
    _W.update(agent=Agent(), scalar=scalar_model, multi=multi_models)


def _run_trial(args):
    index, trial = args
    task = trial_to_task(trial) if "incoming" in trial else trial      # DESY trial or fresh task
    out = {}
    methods = {
        "RL agent (TD3, DESY)": lambda tu, g: run_agent_noisy(_W["agent"], task, BUDGET, tu.noise),
        "Meta-LSQ, one distance": lambda tu, g: tu.run_protocol(task, BUDGET, g, tu.propose_metalsq(_W["scalar"], False)),
        "GP-EI, one distance": lambda tu, g: tu.run_protocol(task, BUDGET, g, tu.propose_gp),
        "random in the window": lambda tu, g: tu.run_protocol(task, BUDGET, g, tu.propose_random),
        "Meta-LSQ multi-output": lambda tu, g: tu.run_protocol_multi(task, BUDGET, g, tu.propose_multi(_W["multi"], False)),
        "GP multi-output, same loop": lambda tu, g: tu.run_protocol_multi(task, BUDGET, g, tu.propose_gp_multi()),
    }
    surrogate_pick = {                           # same run, final setting from the surrogate
        "Meta-LSQ, one distance": lambda tu, K, y, n: tu.pick_scalar(K, y, n, _W["scalar"]),
        "GP-EI, one distance": lambda tu, K, y, n: tu.pick_scalar(K, y, n),
        "Meta-LSQ multi-output": lambda tu, K, y, n: tu.pick_multi(K, task, n, _W["multi"]),
        "GP multi-output, same loop": lambda tu, K, y, n: tu.pick_multi(K, task, n),
    }
    for j, (name, method) in enumerate(methods.items()):
        tu = RealisticTuner(ares, noise_seed=10_000 * index + j)
        t0 = time.perf_counter()
        K, y, _ = method(tu, torch.Generator().manual_seed(1000 + index))
        best = torch.cummin(y, 0).indices          # chosen from the noisy readings
        out[name] = ([tu.mae_um(task, K[best[n - 1]]) for n in CHECKPOINTS],
                     time.perf_counter() - t0)
        if name in surrogate_pick:
            out[name + ", surrogate pick"] = (
                [tu.mae_um(task, surrogate_pick[name](tu, K, y, n)) for n in CHECKPOINTS], 0.0)
    return out


def fresh_task(gen, wide=False):
    """A new task drawn like DESY's 300 trials: their targets have beam sizes in
    [0, 0.5] mm and their misalignments lie within +-0.4 mm, narrower than the
    environment's sampling space (sizes up to 2 mm, +-0.5 mm), which ares.sample_task
    follows. wide=True keeps the environment's space."""
    task = ares.sample_task(gen)
    if not wide:
        u = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=gen))
        task["target"][1], task["target"][3] = u(0, 5e-4), u(0, 5e-4)
        task["misalignment"] = [(u(-4e-4, 4e-4), u(-4e-4, 4e-4)) for _ in range(3)]
    return task


def main(n_trials=None, workers=None, fresh=None, fresh_seed=2027, wide=False):
    if fresh:     # new tasks, unseen by meta-training (seed 0) and not DESY's trials
        gen = torch.Generator().manual_seed(fresh_seed)
        trials = [fresh_task(gen, wide) for _ in range(fresh)]
    else:
        trials = load_trials(fetch(TRIALS_URL, "trials.yaml"))[:n_trials]
    Agent()
    tu = RealisticTuner(ares)
    scalar_pool, beam_pool = tu.build_noisy_pools(n_tasks=800, n_points=70, seed=0)
    print(f"{len(scalar_pool)} noisy meta-training tasks; {len(beam_pool)} with >= 60 on-screen readings")
    scalar_model = tu.train(scalar_pool, False, 50, 0.5, "one distance, noisy")
    multi_models = tu.train_multi(beam_pool, False, 50, 0.5, "multi-output, noisy")
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(scalar_model, multi_models)) as pool:
        results = pool.map(_run_trial, list(enumerate(trials)), chunksize=1)
    print(f"\n{len(trials)} {'fresh tasks' if fresh else 'DESY trials'}, {BUDGET} noisy measurements each, finite screen, in "
          f"{time.time() - t0:.0f} s; true MAE of the best-looking setting, um (median / mean)")
    for name in results[0]:
        r = np.asarray([res[name][0] for res in results])
        cols = "  ".join(f"{n:3d}: {np.median(r[:, i]):6.1f} / {np.mean(r[:, i]):6.1f}"
                         for i, n in enumerate(CHECKPOINTS))
        secs = np.mean([res[name][1] for res in results])
        print(f"  {name:<46s} {cols}" + (f"   ({secs:.1f} s per trial)" if secs else ""))


if __name__ == "__main__":
    arg = lambda flag: int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else None
    main(n_trials=arg("--trials"), fresh=arg("--fresh"), wide="--wide" in sys.argv)
