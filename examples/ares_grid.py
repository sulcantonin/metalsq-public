"""The surrogate tuning loops on DESY's target grid, against DESY's published results.

The 300 trials of the RL-vs-BO study only ask for beam sizes up to 0.5 mm. DESY also ran
their agent (and their BO) on a grid of 160,000 targets that covers the environment's full
range (evaluation/eval_*_target_grid.py of github.com/desy-ml/rl-vs-bo): incoming beam and
misalignments of trial 0, start (10, -10, 0, 10, 0), 150 steps, and

    mu_x, mu_y        20 values from -2 to 2 mm (linear)
    sigma_x, sigma_y  20 values from 20 um to 2 mm (geometric)

Their results are in simulation.zip of Zenodo record 7853721 (rl_grid/polished_donkey.pkl,
bo_grid/*.pkl; read by desy_runs.py without storing the archive): the target and the final
beam of each run. This script draws a stratified sample of the grid (by the larger of the
two target sizes: below 0.5 mm, 0.5-1, 1-1.5 and 1.5-2 mm), runs the locked values-only
Meta-LSQ and ALPaCA-style loops of beam_full.py on it, unchanged, and compares them per band
with DESY's own numbers for the same targets. Our copy of the agent is run too, in DESY's
screen coordinates, so that its final beam can be checked against the published one.

DESY's simulated readings are relative to a misaligned screen (trial 0: 374 um, -46 um). The
surrogate loops have no screen offset, so they are given the target shifted by that offset,
which is the same task; the agent is fed the readings and the target in screen coordinates,
as in DESY's environment.

Scores (um): mean absolute error of the four beam parameters,
    published   DESY's final beam against the target (agent: last step; BO: best setting)
    best        the best setting found in 150 measurements (the metric of beam_final.py)

Win criterion, fixed before the run: Meta-LSQ's median and mean are below the published
agent's in every band above 0.5 mm, and its mean over the whole grid (bands weighted by
their share of the grid) is below the agent's.

Result (one run, 2026-11-15, 300 targets per band, seed 6061; median / mean in um):
    larger target size   agent, published   BO, published   ALPaCA-style   Meta-LSQ
    below 0.5 mm              6.7 /   10      28.7 /  46      6.5 /  7.5    6.6 /  8.9
    0.5 - 1 mm               18.4 /  152      49.1 /  80      7.7 / 10.6    7.7 / 11.8
    1 - 1.5 mm               69.0 /  258     109.3 / 147      7.9 /  8.6    7.9 /  8.4
    1.5 - 2 mm              277.7 / 1223     188.8 / 236      8.3 / 21.2    8.3 / 21.2
    whole grid (weighted)    14.4 /  295      41.9 /  99      7.3 / 10.9    7.3 / 11.9
The criterion is met. Meta-LSQ is closer than the published agent on 53 / 89 / 98 / 99 % of
the targets in the four bands. Against our own run of the agent scored by its best setting
(5.5, 15.0, 33.7, 185 um medians) it is behind below 0.5 mm (43 % of targets) and ahead on
82 / 95 / 97 % above. Notes: (1) Meta-LSQ and ALPaCA-style are level; (2) our run of the
agent agrees with DESY's per band (medians 6.6, 16.7, 71.4, 286 um) but not beam by beam
(median difference of the final beams 16 um), and DESY's means carry a few runs that end
hundreds of mm away; (3) the grid uses one incoming beam (trial 0), noise-free; (4) the
range of targets the agent was trained on is not stated in DESY's repository; (5) the error
that is reachable at all is below 15 um on the targets checked, so no method is at a floor.

    python examples/ares_grid.py run [--per-band 300] [--workers 10]
    python examples/ares_grid.py summary
"""
import io
import json
import os
import sys
import time
import warnings
import zipfile
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ares                                                              # noqa: E402
import desy_runs                                                         # noqa: E402
from ares_vs_rl import (LIMIT, TRIALS_URL, Agent, fetch, load_trials,    # noqa: E402
                        to_knobs, trial_to_task)
from beam_full import RESULTS, VARIANTS, run_loop                        # noqa: E402
from beamtune import Tuner                                               # noqa: E402

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
BUDGET, SEED = 150, 6061
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0001))               # mm, larger target size
LOCKED = json.load(open(os.path.join(RESULTS, "beam_full_locked.json")))["ares"]["values"]
GRID = os.path.join(desy_runs.CACHE, "grid.npz")
OUT = os.path.join(RESULTS, "ares_grid.jsonl")


# ---------------------------------------------------------------- DESY's published grid
def published():
    """targets (n, 4), the agent's and BO's final beams (n, 4; nan where BO has none), in m."""
    if not os.path.exists(GRID):
        z = zipfile.ZipFile(desy_runs.RangeFile(desy_runs.URL.format("simulation")))

        def read(member):
            with z.open(member) as f:
                obj = desy_runs.SafeUnpickler(io.BufferedReader(f)).load()
            return (np.stack([o["target_beam"] for o in obj]).astype(float),
                    np.stack([np.asarray(o["final_beam"], float) for o in obj]))
        T, F = read("simulation/rl_grid/polished_donkey.pkl")
        key = lambda t: tuple(np.round(np.asarray(t, float) * 1e8).astype(np.int64))
        index = {key(t): i for i, t in enumerate(T)}
        bo = np.full_like(F, np.nan)
        members = [m for m, _, _ in desy_runs.directory("simulation") if m.split("/")[1] == "bo_grid"]
        for m in members:
            try:
                Tb, Fb = read(m)
            except Exception:
                continue
            for t, f in zip(Tb, Fb):
                i = index.get(key(t))
                if i is not None:
                    bo[i] = f
        np.savez_compressed(GRID, target=T, agent=F, bo=bo)
    d = np.load(GRID)
    return d["target"], d["agent"], d["bo"]


def band_of(T):
    s = np.maximum(T[:, 1], T[:, 3]) * 1e3
    return np.digitize(s, [b[0] for b in BANDS[1:]])


def sample(per_band):
    T, _, _ = published()
    band, rng, idx = band_of(T), np.random.default_rng(SEED), []
    for b in range(len(BANDS)):
        idx += sorted(rng.choice(np.flatnonzero(band == b), per_band, replace=False).tolist())
    return idx


# ---------------------------------------------------------------- one grid target
_W = {}


def _init():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")
    trial = load_trials(fetch(TRIALS_URL, "trials.yaml"))[0]
    trained = torch.load(os.path.join(RESULTS, "beam_full_ares_dev_models.pt"), weights_only=False)
    off = torch.tensor([trial["misalignments"]["screen_x"], 0.0, trial["misalignments"]["screen_y"], 0.0])
    _W.update(base=trial_to_task(trial), off=off, tu=Tuner(ares), agent=Agent(),
              models={family: trained[variant] for family, variant in LOCKED.items()})


def run_agent(agent, task, target_screen, off, steps):
    """DESY's episode: `steps` actions; readings and target relative to the screen."""
    m = torch.tensor([10.0, -10.0, 0.0, 10.0, 0.0])
    K, beams = [], []
    for _ in range(steps + 1):
        k = to_knobs(m)
        with torch.no_grad():
            beam = ares.beam_on_screen(task, k.unsqueeze(0))[0] - off
        K.append(k)
        beams.append(beam)
        m = (m + agent.act(m, beam, target_screen)).clamp(-LIMIT, LIMIT)
    return torch.stack(K), torch.stack(beams)


def _job(args):
    index, target = args
    tu, off = _W["tu"], _W["off"]
    T = torch.tensor(target)
    task = dict(_W["base"], target=(T + off).tolist())            # same task without a screen offset
    rec = dict(index=index, target=target)
    t0 = time.perf_counter()
    K, beams = run_agent(_W["agent"], task, T, off, BUDGET)
    err = (beams - T).abs().mean(1) * 1e6
    y = torch.log((beams[:BUDGET] - T).norm(dim=1))
    rec["agent_final"] = float(err[-1])
    rec["agent_best"] = float(err[:BUDGET][y.argmin()])
    rec["agent_final_beam"] = beams[-1].tolist()
    for family, models in _W["models"].items():
        variant = LOCKED[family]
        maes = run_loop(tu, task, models, VARIANTS[variant][1], torch.Generator().manual_seed(1000 + index), BUDGET)
        rec[family] = maes[-1]
        rec[family + " @50"] = maes[1]
    rec["seconds"] = time.perf_counter() - t0
    return rec


def run(per_band, workers):
    T, _, _ = published()
    idx = sample(per_band)
    done = {json.loads(line)["index"] for line in open(OUT)} if os.path.exists(OUT) else set()
    todo = [(i, T[i].tolist()) for i in idx if i not in done]
    print(f"{len(idx)} grid targets ({per_band} per band), {len(todo)} to do", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool, open(OUT, "a") as f:
        for n, rec in enumerate(pool.imap_unordered(_job, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({rec['seconds']:.0f} s the last, {time.time() - t0:.0f} s elapsed)", flush=True)


# ---------------------------------------------------------------- summary
def wmedian(values, weights):
    o = np.argsort(values)
    c = np.cumsum(np.asarray(weights)[o])
    return float(np.asarray(values)[o][np.searchsorted(c, c[-1] / 2)])


def summary():
    T, A, B = published()
    pub_agent, pub_bo = np.abs(A - T).mean(1) * 1e6, np.abs(B - T).mean(1) * 1e6
    band_all = band_of(T)
    share = np.bincount(band_all, minlength=len(BANDS)) / len(T)
    recs = {}
    for line in open(OUT):
        r = json.loads(line)
        recs[r["index"]] = r
    idx = np.array(sorted(recs))
    band = band_of(T[idx])
    cols = {"agent, published": pub_agent[idx], "agent, ours, final": np.array([recs[i]["agent_final"] for i in idx]),
            "agent, ours, best": np.array([recs[i]["agent_best"] for i in idx]), "BO, published": pub_bo[idx],
            "ALPaCA-style": np.array([recs[i]["ALPaCA-style"] for i in idx]),
            "Meta-LSQ": np.array([recs[i]["Meta-LSQ"] for i in idx])}
    ours = np.array([recs[i]["agent_final_beam"] for i in idx])
    gap = np.abs(ours - A[idx]).max(1) * 1e6
    print(f"{len(idx)} sampled grid targets. Our agent's final beam against DESY's published one, same target: "
          f"median |difference| {np.median(gap):.2f} um, 90th percentile {np.percentile(gap, 90):.2f} um, "
          f"within 1 um on {100 * np.mean(gap < 1):.0f}%")
    out = {"n": len(idx), "agent_check_median_um": float(np.median(gap)), "bands": {}}
    print("\nmedian / mean error (um) by the larger target beam size; share = part of DESY's grid")
    print(f"  {'band (mm)':12s}{'share':>7s}{'n':>6s}" + "".join(f"{c:>22s}" for c in cols))
    for b, (lo, hi) in enumerate(BANDS):
        m = band == b
        cells = {c: (float(np.nanmedian(v[m])), float(np.nanmean(v[m]))) for c, v in cols.items()}
        out["bands"][f"{lo}-{min(hi, 2.0)}"] = dict(share=float(share[b]), n=int(m.sum()), **{c: list(v) for c, v in cells.items()},
                                                    bo_available=int(np.isfinite(cols["BO, published"][m]).sum()),
                                                    lsq_beats_agent=float(np.mean(cols["Meta-LSQ"][m] < cols["agent, published"][m])),
                                                    lsq_beats_own_agent_best=float(np.mean(cols["Meta-LSQ"][m] < cols["agent, ours, best"][m])))
        print(f"  {lo:.1f}-{min(hi, 2.0):.1f}     {share[b]:7.2f}{int(m.sum()):6d}"
              + "".join(f"{cells[c][0]:12.1f} /{cells[c][1]:8.1f}" for c in cols))
    w = share[band] / np.bincount(band, minlength=len(BANDS))[band]
    print("\nwhole grid (bands weighted by their share): median / mean")
    out["grid"] = {}
    for c, v in cols.items():
        ok = np.isfinite(v)
        med, mean = wmedian(v[ok], w[ok]), float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
        out["grid"][c] = [med, mean]
        print(f"  {c:22s}{med:10.1f} /{mean:9.1f}")
    print(f"  (DESY's full grid, all 160,000 targets, agent: {np.median(pub_agent):.1f} / {pub_agent.mean():.1f})")
    upper = [k for k in out["bands"] if not k.startswith("0.0")]
    win = all(out["bands"][k]["Meta-LSQ"][0] < out["bands"][k]["agent, published"][0]
              and out["bands"][k]["Meta-LSQ"][1] < out["bands"][k]["agent, published"][1] for k in upper) \
        and out["grid"]["Meta-LSQ"][1] < out["grid"]["agent, published"][1]
    out["criterion_met"] = bool(win)
    print("\nwin criterion (median and mean below the published agent in every band above 0.5 mm, and "
          f"grid mean below): {'met' if win else 'NOT met'}")
    for k in out["bands"]:
        print(f"  {k} mm: Meta-LSQ closer than the published agent on {100 * out['bands'][k]['lsq_beats_agent']:.0f}% of targets")
    json.dump(out, open(os.path.join(RESULTS, "ares_grid_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    arg = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
    if sys.argv[1] == "run":
        run(int(arg("--per-band") or 300), int(arg("--workers") or 10))
    else:
        summary()
