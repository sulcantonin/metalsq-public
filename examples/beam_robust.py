"""Robustness of every tuning method on ARES and DIAG0, with the locked settings.

Every method talks to the beamline through a Machine wrapper, so the scenario changes
the machine and nothing in the methods:

  twin        the "digital twin" that supplies gradients is imperfect: it knows no
              misalignments, and its incoming beam is off (sizes and divergences x U(0.8,
              1.2), offsets +-0.2 mm (ARES) / +-0.1 mm (DIAG0), angles +-20 / +-10 urad,
              energy x U(0.98, 1.02)). Readings are the machine's; only the gradients and
              curvatures come from the twin. Value-only methods are unaffected and not run.
  noise       readings carry DESY's noise (5 % of each beam parameter plus 5 um) and the
              screen is finite (+-4.06 x +-2.50 mm, ARES's AREABSCR1; the same size is
              assumed for DIAG0's OTRDG02). Off the screen the reading is DESY's placeholder
              (0, 3.5, 0, 2.2) "m", as in rl-vs-bo's backend; surrogates leave such readings
              out of their fits. Gradients (twin methods) stay exact.
  hysteresis  each quadrupole's field lags its setting: the effective setting is the set
              value minus 1 % of its range (0.02 knob units) in the direction the magnet was
              last moved. The machine starts freshly cycled (no offset). Gradients come from
              a twin that does not know about hysteresis.
  wide        tasks from the wider ranges the methods were meta-trained or trained on but
              never tuned for (ARES: ares.sample_task, targets up to 2 mm; DIAG0: misalignments
              +-300 um, offsets +-1 mm, beta 2-100 m, energy 80-110 MeV).

The score is the true, noise-free MAE (um) of the best-looking setting so far, at the
field actually reached. Each run records every setting and the true MAE per step, so
beam_machine_time.py can be applied to it (files results/beam_robust_<bl>_<scenario>.jsonl).
The first 150 evaluation tasks of beam_final.py are used (DESY's trials for ARES; seed
4041 for DIAG0); the wide tasks use seed 5051.

    python examples/beam_robust.py --beamline ares --scenario twin
"""
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import torch

import beam_final as bf
from beam_full import BEAMLINES, CHECKPOINTS, RESULTS, VARIANTS, run_loop
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
BUDGET, N_TASKS = 150, 150
QUADS = {"ares": (0, 1, 3), "lcls_diag0": (2, 3, 4, 5)}
SCREEN = (2448 / 2 * 3.3198e-6, 2040 / 2 * 2.4469e-6)
PLACEHOLDER = torch.tensor([0.0, 3.5, 0.0, 2.2])
REL_NOISE, ABS_NOISE, HYST = 0.05, 5e-6, 0.02
SCENARIOS = ("twin", "noise", "hysteresis", "wide")
TWIN_METHODS = ("gradient descent, twin", "Meta-LSQ, twin", "ALPaCA-style, twin")


# ---------------------------------------------------------------- tasks
def twin_of(task, name, seed):
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=g))
    t = dict(task)
    t["misalignment"] = [(0.0, 0.0) for _ in task["misalignment"]]
    off, ang = (2e-4, 2e-5) if name == "ares" else (1e-4, 1e-5)
    for k in ("sigma_x", "sigma_y", "sigma_px", "sigma_py"):
        t[k] = task[k] * u(0.8, 1.2)
    for k in ("cor_x", "cor_y"):
        if k in task:
            t[k] = task[k] * u(0.8, 1.2)
    for k in ("mu_x", "mu_y"):
        t[k] = task[k] + u(-off, off)
    for k in ("mu_px", "mu_py"):
        t[k] = task[k] + u(-ang, ang)
    t["energy"] = task["energy"] * u(0.98, 1.02)
    return t


def wide_task(name, gen):
    if name == "ares":
        return BEAMLINES[name].sample_task(gen)          # the environment's full ranges
    bl = BEAMLINES[name]
    u = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=gen))
    energy = u(80e6, 110e6)
    gamma = energy / 0.51099895e6
    task = dict(energy=energy, mu_x=u(-1e-3, 1e-3), mu_px=u(-1e-4, 1e-4), mu_y=u(-1e-3, 1e-3),
                mu_py=u(-1e-4, 1e-4), sigma_p=u(1e-4, 2e-3),
                misalignment=[(u(-3e-4, 3e-4), u(-3e-4, 3e-4)) for _ in range(4)])
    for plane in ("x", "y"):
        eps = u(0.2e-6, 2.0e-6) / gamma
        beta, alpha = u(2, 100), u(-5, 5)
        task[f"sigma_{plane}"] = math.sqrt(beta * eps)
        task[f"sigma_p{plane}"] = math.sqrt((1 + alpha ** 2) / beta * eps)
        task[f"cor_{plane}"] = -alpha * eps
    k_target = torch.rand(1, bl.DIM, generator=gen) * 2 - 1
    with torch.no_grad():
        task["target"] = bl.beam_on_screen(task, k_target)[0].tolist()
    return task


def tasks(name, scenario):
    if scenario == "wide":
        gen = torch.Generator().manual_seed(5051)
        return [wide_task(name, gen) for _ in range(N_TASKS)]
    out = bf.tasks(name)[:N_TASKS]
    if scenario == "twin":
        out = [dict(t, _twin=twin_of(t, name, 7000 + i)) for i, t in enumerate(out)]
    return out


# ---------------------------------------------------------------- the machine
class Machine:
    """The beamline as the methods see it, for one run."""

    def __init__(self, name, scenario, seed):
        base = BEAMLINES[name]
        self.base, self.name, self.scenario = base, name, scenario
        self.DIM, self.START, self.HALF = base.DIM, base.START, base.HALF
        self.sample_task = base.sample_task
        self.noise = torch.Generator().manual_seed(seed)
        self.quads = torch.zeros(base.DIM, dtype=torch.bool)
        self.quads[list(QUADS[name])] = True
        self.last, self.dir = None, torch.zeros(base.DIM)

    def effective(self, K):
        """Hysteresis: settings applied one after the other; returns the fields reached."""
        out = []
        for k in K.detach():
            if self.last is not None:
                d = k - self.last
                moved = (d.abs() > 1e-12) & self.quads
                self.dir[moved] = torch.sign(d[moved])
            self.last = k.clone()
            out.append(k - HYST * self.dir * self.quads)
        return torch.stack(out)

    def beam_on_screen(self, task, K):
        if self.scenario == "hysteresis":
            K = K + (self.effective(K) - K.detach())          # keeps d/dK of the twin
        true = self.base.beam_on_screen(task, K)
        if self.scenario == "twin":
            twin = self.base.beam_on_screen(task["_twin"], K)
            return true.detach() + twin - twin.detach()       # machine values, twin derivatives
        if self.scenario == "noise":
            with torch.no_grad():
                noise = (REL_NOISE * true.abs() + ABS_NOISE) * torch.randn(true.shape, generator=self.noise)
                on = (true[:, 0].abs() < SCREEN[0]) & (true[:, 2].abs() < SCREEN[1])
            out = true + noise
            out = torch.cat([out[:, :1], out[:, 1:2].abs(), out[:, 2:3], out[:, 3:].abs()], 1)
            return torch.where(on.unsqueeze(1), out, PLACEHOLDER.expand_as(out))
        return true


def replay_effective(name, K):
    m = Machine(name, "hysteresis", 0)
    return m.effective(torch.as_tensor(K))


# ---------------------------------------------------------------- runs
_W = {}


def _init(name, scenario, models, methods):
    bf._init(name, models, methods)
    _W.update(name=name, scenario=scenario)


def _run(method, task, index):
    name, scenario = _W["name"], _W["scenario"]
    machine = Machine(name, scenario, 20_000 + index)
    tu = Tuner(machine)
    gen = torch.Generator().manual_seed(1000 + index)
    if method == "RL agent":
        if name == "ares":
            from ares_vs_rl import LIMIT, to_knobs
            target = torch.tensor(task["target"])
            m = torch.tensor([10.0, -10.0, 0.0, 10.0, 0.0])
            K, y = [], []
            for _ in range(BUDGET):
                k = to_knobs(m)
                with torch.no_grad():
                    beam = machine.beam_on_screen(task, k.unsqueeze(0))[0]
                K.append(k); y.append(float(torch.log((beam - target).norm() * 1e3)))
                m = (m + bf._W["agent"].act(m, beam, target)).clamp(-LIMIT, LIMIT)
            K, y = torch.stack(K), torch.tensor(y)
        else:
            import rl_diag0
            saved, rl_diag0.bl = rl_diag0.bl, machine
            try:
                K, y = bf._W["agent"].run(task, BUDGET)
            finally:
                rl_diag0.bl = saved
    elif method == "BO (DESY)":
        from bo_desy import run_bo_desy
        K, y = run_bo_desy(tu, task, gen, BUDGET)
    elif method == "gradient descent, twin":
        K, y, _ = tu.run_gradient_descent(task, BUDGET, gen)
    else:
        family, level = method.split(", ")
        variant = bf.LOCKED[name]["values" if level == "machine" else "twin"][family]
        _, K, y = run_loop(tu, task, bf._W["models"][variant], VARIANTS[variant][1], gen, BUDGET,
                           return_path=True)
    K = torch.as_tensor(K)
    best = torch.cummin(torch.as_tensor(y), 0).indices
    field = replay_effective(name, K) if scenario == "hysteresis" else K
    with torch.no_grad():
        B = BEAMLINES[name].beam_on_screen(task, field)          # true beam at the field reached
    mae = ((B - torch.as_tensor(task["target"])).abs().mean(-1) * 1e6)[best]
    return {"curve": mae.tolist(), "K": [[round(float(v), 6) for v in k] for k in K], "best": best.tolist()}


def _job(args):
    index, task, missing = args
    return index, {m: _run(m, task, index) for m in missing}


def main(name, scenario, methods=None, workers=None):
    methods = methods or (list(TWIN_METHODS) if scenario == "twin" else list(bf.METHODS))
    all_tasks = tasks(name, scenario)
    trained = torch.load(os.path.join(RESULTS, f"beam_full_{name}_dev_models.pt"), weights_only=False)
    models = {v: trained[v] for level in bf.LOCKED[name].values() for v in level.values()}
    out = os.path.join(RESULTS, f"beam_robust_{name}_{scenario}.jsonl")
    if os.environ.get("SMOKE"):
        all_tasks, out = all_tasks[:int(os.environ["SMOKE"])], out.replace(".jsonl", "_smoke.jsonl")
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.setdefault(r["index"], {}).update(r["result"])
    jobs = [(i, t, [m for m in methods if m not in done.get(i, {})]) for i, t in enumerate(all_tasks)]
    jobs = [j for j in jobs if j[2]]
    print(f"{out}: {len(jobs)} tasks to run", flush=True)
    if jobs:
        t0 = time.time()
        with Pool(workers or max(1, (os.cpu_count() or 2) - 2), initializer=_init,
                  initargs=(name, scenario, models, methods)) as p, open(out, "a") as f:
            for n, (i, res) in enumerate(p.imap_unordered(_job, jobs), 1):
                done.setdefault(i, {}).update(res)
                f.write(json.dumps({"index": i, "result": res}) + "\n"); f.flush()
                if n % 25 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} tasks, {time.time() - t0:.0f} s", flush=True)
    print(f"\n{name}, scenario {scenario}: {len(done)} tasks; true MAE of the best-looking setting, um (median / mean)")
    for m in bf.METHODS:
        rows = [done[i][m]["curve"] for i in sorted(done) if m in done[i]]
        if rows:
            r = np.asarray(rows)
            print(f"  {m:26s}" + "  ".join(f"{c:4d}: {np.median(r[:, c - 1]):7.2f} / {np.mean(r[:, c - 1]):7.2f}"
                                          for c in CHECKPOINTS))


if __name__ == "__main__":
    arg = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
    main(arg("--beamline") or "ares", arg("--scenario") or "twin",
         methods=[m.strip() for m in arg("--methods").split(";")] if arg("--methods") else None,
         workers=int(arg("--workers")) if arg("--workers") else None)
