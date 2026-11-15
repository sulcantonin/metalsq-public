"""Final evaluation on ARES (DESY's 300 trials) and DIAG0 (300 held-out tasks).

Every method gets 150 measurements from the same start; the score is the true MAE of the
best setting found so far (noise-free simulation, as in DESY's simulation study).

    machine (values only)
        RL agent         ARES: DESY's TD3 agent (ares_vs_rl.py); DIAG0: our TD3 agent (rl_diag0.py)
        BO (DESY)        DESY's BoTorch Bayesian optimisation (bo_desy.py)
        Meta-LSQ         locked values variant (results/beam_full_locked.json)
        ALPaCA-style     locked values variant
    twin (values and simulator derivatives)
        gradient descent normalised gradient steps on the measured objective (beamtune.py)
        Meta-LSQ         locked twin variant
        ALPaCA-style     locked twin variant

The surrogate variants were locked on development tasks (beam_full.py, seed 3031) before
this script was run; the DIAG0 evaluation tasks use seed 4041 and DESY's trials were
never used for any choice. Each finished task is appended to
results/beam_final_<beamline>.jsonl; rerunning resumes and adds methods not yet run.

    python examples/beam_final.py --beamline ares [--methods "A;B"]
    python examples/beam_final.py --beamline lcls_diag0
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import torch

from beam_full import BEAMLINES, CHECKPOINTS, RESULTS, VARIANTS, run_loop
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
BUDGET, N_DIAG0, DIAG0_SEED = 150, 300, 4041
LOCKED = json.load(open(os.path.join(RESULTS, "beam_full_locked.json")))
METHODS = ("RL agent", "BO (DESY)", "Meta-LSQ, machine", "ALPaCA-style, machine",
           "gradient descent, twin", "Meta-LSQ, twin", "ALPaCA-style, twin")


def tasks(name):
    if name == "ares":
        from ares_vs_rl import TRIALS_URL, fetch, load_trials, trial_to_task
        return [trial_to_task(t) for t in load_trials(fetch(TRIALS_URL, "trials.yaml"))]
    gen = torch.Generator().manual_seed(DIAG0_SEED)
    return [BEAMLINES[name].sample_task(gen) for _ in range(N_DIAG0)]


_W = {}


def _init(name, models, methods):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")
    _W.update(name=name, models=models, methods=methods, tu=Tuner(BEAMLINES[name]))
    if "RL agent" in methods:
        if name == "ares":
            from ares_vs_rl import Agent
            _W["agent"] = Agent()
        else:
            from rl_diag0 import Diag0Agent
            _W["agent"] = Diag0Agent()


def _run(method, task, gen):
    name, tu = _W["name"], _W["tu"]
    if method == "RL agent":
        if name == "ares":
            from ares_vs_rl import run_agent
            K, y, _ = run_agent(_W["agent"], task, BUDGET)
        else:
            K, y = _W["agent"].run(task, BUDGET)
    elif method == "BO (DESY)":
        from bo_desy import run_bo_desy
        K, y = run_bo_desy(tu, task, gen, BUDGET)
    elif method == "gradient descent, twin":
        K, y, _ = tu.run_gradient_descent(task, BUDGET, gen)
    else:
        family, level = method.split(", ")
        variant = LOCKED[name]["values" if level == "machine" else "twin"][family]
        return run_loop(tu, task, _W["models"][variant], VARIANTS[variant][1], gen, BUDGET)
    best = torch.cummin(torch.as_tensor(y), 0).indices
    return [tu.mae_um(task, K[best[n - 1]]) for n in CHECKPOINTS]


def _job(args):
    index, task, missing = args
    out = {}
    for m in missing:
        t0 = time.perf_counter()
        out[m] = (_run(m, task, torch.Generator().manual_seed(1000 + index)), time.perf_counter() - t0)
    return index, out


def summarise(name, done, methods):
    n = len(done)
    print(f"\n{name}: {n} tasks, {BUDGET} measurements; true MAE of the best setting so far, um (median / mean)")
    for m in methods:
        rows = [done[i][m][0] for i in sorted(done) if m in done[i]]
        if not rows:
            continue
        r = np.asarray(rows)
        s = np.mean([done[i][m][1] for i in done if m in done[i]])
        print(f"  {m:<26s} ({len(rows):3d})" + "  ".join(
            f"{c:4d}: {np.median(r[:, j]):7.2f} / {np.mean(r[:, j]):7.2f}" for j, c in enumerate(CHECKPOINTS))
            + f"   ({s:.1f} s)")
    for a, b in [("Meta-LSQ, machine", "ALPaCA-style, machine"), ("Meta-LSQ, twin", "ALPaCA-style, twin")]:
        idx = [i for i in done if a in done[i] and b in done[i]]
        if idx:
            ra = np.asarray([done[i][a][0] for i in idx]); rb = np.asarray([done[i][b][0] for i in idx])
            print(f"  {a} closer than {b} in: " + "  ".join(
                f"{c}: {np.mean(ra[:, j] < rb[:, j]):.0%}" for j, c in enumerate(CHECKPOINTS)))


def main(name, methods=None, workers=None):
    methods = methods or list(METHODS)
    all_tasks = tasks(name)
    trained = torch.load(os.path.join(RESULTS, f"beam_full_{name}_dev_models.pt"), weights_only=False)
    models = {v: trained[v] for level in LOCKED[name].values() for v in level.values()}
    out = os.path.join(RESULTS, f"beam_final_{name}.jsonl")
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done[r["index"]] = {**done.get(r["index"], {}), **r["result"]}
    todo = [(i, t) for i, t in enumerate(all_tasks) if any(m not in done.get(i, {}) for m in methods)]
    print(f"{out}: {len(all_tasks) - len(todo)} of {len(all_tasks)} tasks have all requested methods")
    if todo:
        t0 = time.time()
        if os.environ.get("SMOKE"):
            todo = todo[:int(os.environ["SMOKE"])]
            out = out.replace(".jsonl", "_smoke.jsonl")
        with Pool(workers or max(1, (os.cpu_count() or 2) - 2), initializer=_init,
                  initargs=(name, models, [m for m in methods])) as p, open(out, "a") as f:
            jobs = [(i, t, [m for m in methods if m not in done.get(i, {})]) for i, t in todo]
            for n, (i, res) in enumerate(p.imap_unordered(_job, jobs), 1):
                done[i] = {**done.get(i, {}), **res}
                f.write(json.dumps({"index": i, "result": res}) + "\n"); f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} tasks, {time.time() - t0:.0f} s", flush=True)
    summarise(name, done, list(METHODS))


if __name__ == "__main__":
    arg = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
    main(arg("--beamline") or "ares",
         methods=[m.strip() for m in arg("--methods").split(";")] if arg("--methods") else None,
         workers=int(arg("--workers")) if arg("--workers") else None)
