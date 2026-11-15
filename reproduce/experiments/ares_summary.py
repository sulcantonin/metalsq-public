"""Summarise the ARES beam-tuning runs of examples/ into results/ares_tuning.json.

Reads the per-trial files written by examples/beam_final.py (final accuracy, 300 DESY
trials), examples/beam_time.py (error after every measurement and every setting) and
examples/beam_robust.py (robustness scenarios, first 150 trials), and stores the
numbers the paper quotes. The machine-time model is the one of
examples/beam_machine_time.py.

    python reproduce/experiments/ares_summary.py
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EX = os.path.join(REPO, "examples")
sys.path.insert(0, EX)
import beam_machine_time as bmt  # noqa: E402

METHODS = ("RL agent", "BO (DESY)", "Meta-LSQ, machine", "ALPaCA-style, machine",
           "gradient descent, twin", "Meta-LSQ, twin", "ALPaCA-style, twin")


def load(name):
    """Per-trial records of one run file; the larger files are stored gzipped."""
    d, path = {}, os.path.join(EX, "results", name)
    with (open(path) if os.path.exists(path) else gzip.open(path + ".gz", "rt")) as f:
        for line in f:
            r = json.loads(line)
            d.setdefault(r["index"], {}).update(r["result"])
    return d


def main():
    final, timed = load("beam_final_ares.jsonl"), load("beam_time_ares.jsonl")
    out = {"n_trials": len(final), "final": {}, "closer": {}, "levels": {}, "reversals": {}, "minutes": {}, "robust": {}}
    for m in METHODS:
        r = np.array([final[i][m][0][-1] for i in sorted(final)])          # after 150 measurements
        out["final"][m] = {"median": float(np.median(r)), "mean": float(np.mean(r))}
    for a, b in (("Meta-LSQ, twin", "ALPaCA-style, twin"), ("Meta-LSQ, machine", "ALPaCA-style, machine")):
        out["closer"][a] = float(np.mean([final[i][a][0][-1] < final[i][b][0][-1] for i in final]))
    quads = bmt.QUADS["ares"]
    for cyc in (0.0, 30.0):
        p = argparse.Namespace(meas=1.0, ramp=10.0, settle=1.0, cycle=cyc)
        for m in METHODS:
            runs = [timed[i][m] for i in sorted(timed)]
            for lv in bmt.LEVELS:
                res = [bmt.time_to_level(r, quads, p, lv) for r in runs]
                steps = [s if s is not None else np.inf for s, _ in res]
                mins = [t / 60 if t is not None else np.inf for _, t in res]
                key = f"{m}|{lv:.2f}"
                if cyc == 0.0:
                    s = float(np.median(steps))
                    out["levels"][key] = {"steps": None if np.isinf(s) else s,
                                          "reached": float(np.mean([s_ is not None for s_, _ in res]))}
                t = float(np.median(mins))
                out["minutes"].setdefault(key, {})[f"cycle{int(cyc)}"] = None if np.isinf(t) else t
            if cyc == 0.0:
                out["reversals"][m] = float(np.median([bmt.reversals(r, quads) for r in runs]))
    first = sorted(timed)[:150]
    out["robust"]["nominal"] = {m: {"median": float(np.median([timed[i][m]["curve"][-1] for i in first])),
                                    "mean": float(np.mean([timed[i][m]["curve"][-1] for i in first]))} for m in METHODS}
    for sc in ("twin", "noise", "hysteresis", "wide"):
        R = load(f"beam_robust_ares_{sc}.jsonl")
        out["robust"][sc] = {m: {"median": float(np.median([R[i][m]["curve"][-1] for i in R])),
                                 "mean": float(np.mean([R[i][m]["curve"][-1] for i in R]))}
                             for m in METHODS if m in R[0]}
    dst = os.path.join(os.path.dirname(HERE), "results", "ares_tuning.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)
    for m in METHODS:
        f, l99 = out["final"][m], out["levels"][f"{m}|0.99"]
        print(f"  {m:24s} median {f['median']:6.2f} mean {f['mean']:6.2f}  99%: {l99['steps']} ({l99['reached']:.0%})"
              f"  reversals {out['reversals'][m]:.0f}")
    print("  closer:", out["closer"])


if __name__ == "__main__":
    main()
