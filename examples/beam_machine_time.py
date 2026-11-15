"""How long tuning would take on the machine, from the replays of beam_time.py.

Two measures, both from the recorded run of every method on every task:

1. Measurements needed to remove 90 / 95 / 99 % of the starting error (the true MAE of the
   start setting), judged on the best setting found so far.

2. Machine time to get there, with a simple, stated cost model (all parameters below are
   assumptions, not ARES or SLAC specifications; --cycle and the others change them):
     - every measurement: T_MEAS for the screen readout;
     - every change of the settings: the magnets ramp in parallel, T_RAMP for a sweep over
       a magnet's full range (time proportional to the largest move), then T_SETTLE;
     - hysteresis: the field of an iron-dominated quadrupole depends on the direction it was
       approached from, so a quadrupole whose change reverses direction is cycled first
       (T_CYCLE, all reversing quadrupoles in parallel); steerers are taken as free of
       hysteresis;
     - at the end, the magnets are moved back to the best setting found (with the same
       ramp, settle and hysteresis rules).
   The algorithms' own compute is left out: it is milliseconds per step for every method.

    python examples/beam_machine_time.py --beamline ares [--cycle 30] [--meas 1] [--ramp 10] [--settle 1]
"""
import argparse
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QUADS = {"ares": (0, 1, 3), "lcls_diag0": (2, 3, 4, 5)}          # knob indices of the quadrupoles
LEVELS = (0.90, 0.95, 0.99)
EPS = 1e-4                                                       # smaller knob changes are no move
METHODS = ("RL agent", "BO (DESY)", "Meta-LSQ, machine", "ALPaCA-style, machine",
           "gradient descent, twin", "Meta-LSQ, twin", "ALPaCA-style, twin")


def step_costs(K, quads, p):
    """Time of every measurement (move from the previous setting, then read the screen), and
    the state of each quadrupole's last direction of change after it."""
    K = np.asarray(K)
    last_dir = np.zeros(K.shape[1])
    costs, dirs = [p.meas], [last_dir.copy()]
    for n in range(1, len(K)):
        c, last_dir = move_cost(K[n - 1], K[n], last_dir, quads, p)
        costs.append(c + p.meas); dirs.append(last_dir.copy())
    return np.array(costs), dirs


def move_cost(a, b, last_dir, quads, p):
    d = np.asarray(b) - np.asarray(a)
    moved = np.abs(d) > EPS
    if not moved.any():
        return 0.0, last_dir
    new_dir = last_dir.copy()
    new_dir[moved] = np.sign(d[moved])
    reversal = any(moved[q] and last_dir[q] != 0 and new_dir[q] != last_dir[q] for q in quads)
    t = p.ramp * np.abs(d).max() / 2 + p.settle + (p.cycle if reversal else 0.0)
    return t, new_dir


def time_to_level(run, quads, p, level):
    """(measurements, machine seconds) until the best setting so far has removed `level` of
    the starting error, including the final move back to that best setting; None if never."""
    curve, K, best = np.asarray(run["curve"]), np.asarray(run["K"]), run["best"]
    goal = (1 - level) * curve[0]
    hit = np.nonzero(curve <= goal)[0]
    if not len(hit):
        return None, None
    n = int(hit[0])
    costs, dirs = step_costs(K[:n + 1], quads, p)
    back, _ = move_cost(K[n], K[best[n]], dirs[n], quads, p)
    return n + 1, float(costs.sum() + back)


def reversals(run, quads):
    K = np.asarray(run["K"])
    last, count = np.zeros(K.shape[1]), 0
    for n in range(1, len(K)):
        d = K[n] - K[n - 1]
        for q in quads:
            if abs(d[q]) > EPS:
                s = np.sign(d[q])
                count += last[q] != 0 and s != last[q]
                last[q] = s
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beamline", default="ares")
    ap.add_argument("--meas", type=float, default=1.0, help="screen readout per measurement, s")
    ap.add_argument("--ramp", type=float, default=10.0, help="full-range ramp of a magnet, s")
    ap.add_argument("--settle", type=float, default=1.0, help="settling after a change, s")
    ap.add_argument("--cycle", type=float, default=30.0, help="hysteresis cycle of a quadrupole, s")
    p = ap.parse_args()
    done, path = {}, os.path.join(HERE, "results", f"beam_time_{p.beamline}.jsonl")
    with (open(path) if os.path.exists(path) else gzip.open(path + ".gz", "rt")) as f:   # stored gzipped
        for line in f:
            r = json.loads(line)
            done.setdefault(r["index"], {}).update(r["result"])
    quads = QUADS[p.beamline]
    print(f"{p.beamline}: {len(done)} tasks. Machine-time model: readout {p.meas:g} s, full-range ramp "
          f"{p.ramp:g} s, settle {p.settle:g} s, quadrupole hysteresis cycle {p.cycle:g} s on a reversal.")
    print("  measurements (and machine minutes) to remove 90 / 95 / 99 % of the starting error; median "
          "over tasks, tasks that never get there count as not reached")
    for m in METHODS:
        runs = [done[i][m] for i in sorted(done) if m in done[i]]
        if not runs:
            continue
        row = f"  {m:24s}"
        for lv in LEVELS:
            res = [time_to_level(r, quads, p, lv) for r in runs]
            steps = [s if s is not None else np.inf for s, _ in res]
            mins = [t / 60 if t is not None else np.inf for _, t in res]
            reached = np.mean([s is not None for s, _ in res])
            ms, mt = np.median(steps), np.median(mins)
            row += (f"  {lv:.0%}: " + (">150" if np.isinf(ms) else f"{ms:4.0f}") + " / "
                    + ("  -  " if np.isinf(mt) else f"{mt:5.1f}") + f" min ({reached:4.0%})")
        row += f"   quad reversals per run: {np.median([reversals(r, quads) for r in runs]):.0f}"
        print(row)


if __name__ == "__main__":
    main()
