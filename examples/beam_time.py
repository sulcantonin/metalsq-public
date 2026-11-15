"""How long each method takes to tune the beam (companion to beam_final.py).

Replays every method of beam_final.py on the same tasks and seeds, recording after every
measurement the true MAE of the best setting found so far, and the compute time of every
decision (proposing the next setting; the simulated measurement is not counted).

Reported per method:
  measurements to reach 20 um (DESY's convergence threshold, Kaiser et al.) and 5 um, as
      the median over tasks and the fraction of tasks that reach it within 150;
  compute per decision (median over all decisions) and per run of 150 measurements.
On a machine, the time of a measurement (setting magnets, reading the screen) comes on
top of the compute; the number of measurements is therefore the machine-time figure.

    python examples/beam_time.py --beamline ares [--methods "A;B"]
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import torch

import beam_final as bf
from beam_full import BEAMLINES, RESULTS, VARIANTS, run_loop
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
BUDGET, THRESHOLDS = 150, (20.0, 5.0)


def run_agent_timed(agent, task, budget, times):
    from ares_vs_rl import LIMIT, to_knobs
    import ares
    target = torch.tensor(task["target"])
    m = torch.tensor([10.0, -10.0, 0.0, 10.0, 0.0])
    K, y = [], []
    for _ in range(budget):
        k = to_knobs(m)
        with torch.no_grad():
            beam = ares.beam_on_screen(task, k.unsqueeze(0))[0]
        K.append(k); y.append(float(torch.log((beam - target).norm() * 1e3)))
        t0 = time.perf_counter()
        m = (m + agent.act(m, beam, target)).clamp(-LIMIT, LIMIT)
        times.append(time.perf_counter() - t0)
    return torch.stack(K), torch.tensor(y)


def _run(method, task, gen):
    W, name, tu = bf._W, bf._W["name"], bf._W["tu"]
    times = []
    if method == "RL agent":
        if name == "ares":
            K, y = run_agent_timed(W["agent"], task, BUDGET, times)
        else:
            K, y = W["agent"].run(task, BUDGET, times)
    elif method == "BO (DESY)":
        from bo_desy import run_bo_desy
        K, y = run_bo_desy(tu, task, gen, BUDGET, times)
    elif method == "gradient descent, twin":
        K, y, _ = tu.run_gradient_descent(task, BUDGET, gen)          # one gradient step: no fit
    else:
        family, level = method.split(", ")
        variant = bf.LOCKED[name]["values" if level == "machine" else "twin"][family]
        _, K, y = run_loop(tu, task, W["models"][variant], VARIANTS[variant][1], gen, BUDGET,
                           return_path=True, times=times)
    best = torch.cummin(torch.as_tensor(y), 0).indices
    with torch.no_grad():
        B = tu.bl.beam_on_screen(task, K)
    mae = ((B - tu.target(task)).abs().mean(-1) * 1e6)[best]           # best-so-far MAE per step
    return {"curve": mae.tolist(), "times": times,
            "K": [[round(float(v), 6) for v in k] for k in K],     # every setting, for machine time
            "best": best.tolist()}


def _job(args):
    index, task, missing = args
    return index, {m: _run(m, task, torch.Generator().manual_seed(1000 + index)) for m in missing}


def steps_to(curve, eps):
    hit = np.nonzero(np.asarray(curve) <= eps)[0]
    return int(hit[0]) + 1 if len(hit) else None


def summarise(name, done, methods):
    print(f"\n{name}: {len(done)} tasks; measurements to reach a threshold (median over tasks, "
          f"tasks not reaching it count as > {BUDGET}) and compute per decision")
    print(f"  {'':26s}" + "".join(f"   to {e:g} um: median (reached)" for e in THRESHOLDS)
          + "   decision: median   run of 150")
    for m in methods:
        rows = [done[i][m] for i in sorted(done) if m in done[i]]
        if not rows:
            continue
        line = f"  {m:26s}"
        for eps in THRESHOLDS:
            s = [steps_to(r["curve"], eps) for r in rows]
            reached = [v for v in s if v is not None]
            med = np.median([v if v is not None else BUDGET + 1 for v in s])
            line += f"   {('>150' if med > BUDGET else f'{med:.0f}'):>6s} ({100 * len(reached) / len(s):3.0f}%)        "
        if any(r["times"] for r in rows):
            t = np.concatenate([r["times"] for r in rows])
            per_run = np.median([sum(r["times"]) for r in rows])
            line += f"   {1e3 * np.median(t):9.3f} ms  {per_run:8.3f} s"
        else:
            line += "         (no model to fit)"
        print(line)


def main(name, methods=None, workers=None):
    methods = methods or list(bf.METHODS)
    tasks = bf.tasks(name)
    trained = torch.load(os.path.join(RESULTS, f"beam_full_{name}_dev_models.pt"), weights_only=False)
    models = {v: trained[v] for level in bf.LOCKED[name].values() for v in level.values()}
    out = os.path.join(RESULTS, f"beam_time_{name}.jsonl")
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done[r["index"]] = {**done.get(r["index"], {}), **r["result"]}
    if os.environ.get("SMOKE"):
        tasks = tasks[:int(os.environ["SMOKE"])]
    jobs = [(i, t, [m for m in methods if m not in done.get(i, {})]) for i, t in enumerate(tasks)]
    jobs = [j for j in jobs if j[2]]
    print(f"{out}: {len(jobs)} tasks to run")
    if jobs:
        t0 = time.time()
        with Pool(workers or max(1, (os.cpu_count() or 2) - 2), initializer=bf._init,
                  initargs=(name, models, methods)) as p, open(out, "a") as f:
            for n, (i, res) in enumerate(p.imap_unordered(_job, jobs), 1):
                done[i] = {**done.get(i, {}), **res}
                f.write(json.dumps({"index": i, "result": res}) + "\n"); f.flush()
                if n % 25 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} tasks, {time.time() - t0:.0f} s", flush=True)
    summarise(name, done, list(bf.METHODS))


if __name__ == "__main__":
    arg = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
    main(arg("--beamline") or "ares",
         methods=[m.strip() for m in arg("--methods").split(";")] if arg("--methods") else None,
         workers=int(arg("--workers")) if arg("--workers") else None)
