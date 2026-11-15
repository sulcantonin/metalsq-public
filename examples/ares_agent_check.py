"""Is our copy of DESY's agent their agent? A check against their published runs.

DESY published every step of their agent's 300 simulated runs (Zenodo 7853721,
simulation.zip, group rl; fetch them first with desy_runs.py). This script

  1. feeds our copy of the agent (ares_vs_rl.Agent) the recorded observations and compares
     its move with the recorded one, at each of the 300 x 150 steps;
  2. runs our copy in closed loop in DESY's own setting (their screen offset in the
     readings, magnets free to leave their limits) and compares the path and the error
     with the published runs;
  3. runs it in the setting of beam_final.py (no screen offset), with the magnets clipped
     to their limits as there and without, and scores it as there (true error of the
     setting that looked best to the method).

Nothing here is tuned. It writes results/ares_agent_check.json.

    python examples/desy_runs.py fetch simulation rl
    python examples/ares_agent_check.py          # about 4 min on one process
"""
import json
import os
import warnings

import numpy as np
import torch

import ares
from ares_vs_rl import LIMIT, TRIALS_URL, Agent, fetch, load_trials, run_agent, to_knobs, trial_to_task
from beamtune import Tuner
from desy_runs import load_runs

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BUDGET = 150
STEP = torch.tensor([3.0, 3.0, 3e-4, 3.0, 6e-4])      # the agent's largest move (Q1, Q2, CV, Q3, CH)


def closed_loop(agent, task, offset):
    """DESY's setting: readings relative to the misaligned screen, no clipping."""
    target = torch.tensor(task["target"])
    off = torch.tensor([offset[0], 0.0, offset[1], 0.0])
    m, path, err = torch.tensor([10.0, -10.0, 0.0, 10.0, 0.0]), [], []
    for _ in range(BUDGET):
        with torch.no_grad():
            beam = ares.beam_on_screen(task, to_knobs(m).unsqueeze(0))[0] - off
        path.append(m); err.append(float((beam - target).abs().mean()) * 1e6)
        m = m + agent.act(m, beam, target)
    return torch.stack(path), min(err)


def main():
    agent, tu = Agent(), Tuner(ares)
    tasks = [trial_to_task(t) for t in load_trials(fetch(TRIALS_URL, "trials.yaml"))]
    runs = {int(r["name"].split("_")[1]): r for r in load_runs("simulation", "rl")}
    assert sorted(runs) == list(range(len(tasks))), "run desy_runs.py fetch simulation rl first"
    move, path, pub, ours, left = [], [], [], [], 0
    table = {"clipped": [], "not clipped": []}
    for i, task in enumerate(tasks):
        r = runs[i]
        M, B, T = torch.tensor(r["magnets"]), torch.tensor(r["beam"]), torch.tensor(r["target"])
        move.append(max(float(((M[t] + agent.act(M[t], B[t], T) - M[t + 1]) / STEP).abs().max())
                        for t in range(BUDGET)))
        left += bool((M.abs() > LIMIT * (1 + 1e-6)).any())
        P, e = closed_loop(agent, task, r["misalignments"][6:8].astype(float))
        path.append(float(((P - M[:BUDGET]) / STEP).abs().max()))
        ours.append(e)
        pub.append(float(np.abs(r["beam"][:BUDGET] - r["target"]).mean(1).min()) * 1e6)
        for name, clip in (("clipped", True), ("not clipped", False)):
            K, y, _ = run_agent(agent, task, BUDGET, clip=clip)
            table[name].append(tu.mae_um(task, K[int(torch.argmin(y))]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(tasks)}", flush=True)
    stat = lambda v: dict(median=float(np.median(v)), mean=float(np.mean(v)))
    out = dict(
        n_trials=len(tasks),
        largest_move_difference=float(max(move)),             # in units of the agent's largest move
        published_runs_leaving_limits=int(left),
        closed_loop_path_within_1pct=float(np.mean(np.array(path) < 0.01)),
        closed_loop_path_median_difference=float(np.median(path)),
        published_best_error=stat(pub), ours_desy_setting_best_error=stat(ours),
        beam_final_setting={k: stat(v) for k, v in table.items()})
    json.dump(out, open(os.path.join(RESULTS, "ares_agent_check.json"), "w"), indent=1)
    f = lambda s: f"{s['median']:.1f} / {s['mean']:.1f}"
    print(f"moves: our copy differs from the recorded move by at most {max(move):.1e} of a full step")
    print(f"DESY's published runs leave the magnet limits on {left} of {len(tasks)} trials")
    print(f"closed loop in DESY's setting: {np.mean(np.array(path) < 0.01):.0%} of the runs follow the published "
          f"settings within 1% of a step; best error, um (median / mean): published {f(stat(pub))}, ours {f(stat(ours))}")
    print("setting of beam_final.py, error of the best-looking setting, um (median / mean): "
          + ", ".join(f"{k} {f(stat(v))}" for k, v in table.items()))


if __name__ == "__main__":
    main()
