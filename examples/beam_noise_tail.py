"""Why the surrogate loops have a high mean error in the noise scenario of beam_robust.py.

Reads the stored runs of the noise scenario (results/beam_robust_ares_noise.jsonl, every
setting of every run), recomputes the true beam at each setting and counts the runs in
which the beam is off the screen at more than half of the settings ("lost"). For each
method it reports the lost trials with their final errors and the mean error over the
remaining trials. Nothing is rerun.

    python examples/beam_noise_tail.py      # writes results/beam_noise_tail.json
"""
import gzip
import json
import os
import warnings

import numpy as np
import torch

import ares

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCREEN = (2448 / 2 * 3.3198e-6, 2040 / 2 * 2.4469e-6)      # half-widths of AREABSCR1, m (as in beam_robust.py)
METHODS = ("RL agent", "BO (DESY)", "ALPaCA-style, machine", "Meta-LSQ, machine",
           "ALPaCA-style, twin", "Meta-LSQ, twin")
N_TASKS = 150


def load(name):
    path = os.path.join(RESULTS, name)
    opener = open if os.path.exists(path) else gzip.open
    done = {}
    with opener(path if os.path.exists(path) else path + ".gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            done.setdefault(r["index"], {}).update(r["result"])
    return done


def off_screen(task, K):
    with torch.no_grad():
        B = ares.beam_on_screen(task, torch.tensor(K))
    return ((B[:, 0].abs() >= SCREEN[0]) | (B[:, 2].abs() >= SCREEN[1])).numpy()


def main():
    from ares_vs_rl import TRIALS_URL, fetch, load_trials, trial_to_task
    tasks = [trial_to_task(t) for t in load_trials(fetch(TRIALS_URL, "trials.yaml"))][:N_TASKS]
    done = load("beam_robust_ares_noise.jsonl")
    idx = sorted(done)
    out = {"n_trials": len(idx)}
    print(f"noise scenario, {len(idx)} trials; error after 150 measurements, um")
    for m in METHODS:
        final = np.array([done[i][m]["curve"][-1] for i in idx])
        off = np.array([off_screen(tasks[i], done[i][m]["K"]) for i in idx])
        lost = off.mean(1) > 0.5
        out[m] = dict(median=float(np.median(final)), mean=float(final.mean()),
                      off_screen_share=float(off.mean()),
                      lost=[[int(idx[j]), float(final[j])] for j in np.where(lost)[0]],
                      mean_without_lost=float(final[~lost].mean()))
        print(f"  {m:24s} {np.median(final):6.1f} / {final.mean():6.1f}   lost beam on {int(lost.sum())} trials "
              f"({', '.join(f'{final[j]:.0f}' for j in np.where(lost)[0])})   mean over the others {final[~lost].mean():.1f}")
    json.dump(out, open(os.path.join(RESULTS, "beam_noise_tail.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
