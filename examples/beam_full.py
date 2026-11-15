"""Full-information beam tuning on ARES (DESY) and DIAG0 (SLAC LCLS-II).

The loop is DESY's: fixed start, the start plus four random settings in the window
(+-5% of each knob range, i.e. +-0.1 in knob units) around it, then one proposal per
step inside the window around the last setting, 150 measurements. Each proposal fits one
surrogate per beam parameter, minimises the composed log distance to the target over
2000 candidates in the window and, when slopes are available, refines it by gradient
descent on the fitted model.

Surrogates (beam_structure.py), each given the same information:
    order 0   values                       (a real machine)
    order 1   values + slopes              (a differentiable twin)
    order 2   values + slopes + curvatures (the twin's second derivatives)
    +c        constraint rows from linear optics (usable at any order)
for Meta-LSQ (closed-form feature derivatives) and ALPaCA-style (autograd derivatives
of a learned encoder). Settings are chosen on development tasks (--dev), never on the
evaluation trials.

    python examples/beam_full.py --beamline ares --dev 60            # development tasks
    python examples/beam_full.py --beamline lcls_diag0 --dev 60
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import torch

import ares
try:
    import lcls_diag0          # SLAC LCLS-II DIAG0, a second beamline; optional
except ImportError:
    lcls_diag0 = None
from beam_structure import FullALPaCA, FullLSQ, constraints, measure, pool_sampler
from beamtune import Tuner

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")
BEAMLINES = {"ares": ares}
if lcls_diag0 is not None:
    BEAMLINES["lcls_diag0"] = lcls_diag0
BUDGET, CHECKPOINTS, N_CAND, EPS, KAPPA = 150, (20, 50, 100, 150), 2000, 1e-3, 0.1
OFF_SCREEN_MM = torch.tensor([0.0, 3.5, 0.0, 2.2]) * 1e3      # DESY's off-screen reading (m), in mm
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCALE = float(os.environ.get("STEPS_SCALE", 1))       # below 1 only for smoke tests (nothing cached)

# name -> (family, derivative order, constraint rows, support size for meta-training)
VARIANTS = {
    "Meta-LSQ, values": ("lsq", 0, False, 50),
    "Meta-LSQ, values + constraints": ("lsq", 0, True, 50),
    "Meta-LSQ, slopes": ("lsq", 1, False, 30),
    "Meta-LSQ, slopes + constraints": ("lsq", 1, True, 30),
    "Meta-LSQ, slopes + curvatures": ("lsq", 2, False, 30),
    "Meta-LSQ, slopes + curvatures + constraints": ("lsq", 2, True, 30),
    "ALPaCA-style, values": ("alp", 0, False, 50),
    "ALPaCA-style, values + constraints": ("alp", 0, True, 50),
    "ALPaCA-style, slopes": ("alp", 1, False, 30),
    "ALPaCA-style, slopes + constraints": ("alp", 1, True, 30),
    "ALPaCA-style, slopes + curvatures": ("alp", 2, False, 30),
    "ALPaCA-style, slopes + curvatures + constraints": ("alp", 2, True, 30),
}


def task_sampler(name):
    if name == "ares":                 # drawn like DESY's 300 trials (see ares_realistic.py)
        from ares_realistic import fresh_task
        return lambda gen: fresh_task(gen)
    return BEAMLINES[name].sample_task


def build_pool(name, n_tasks=400, n_points=60, seed=0, half=0.3):
    """Meta-training tasks: random local neighbourhoods, with slopes and curvatures."""
    path = os.path.join(RESULTS, f"beam_full_{name}_pool.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    bl, gen = BEAMLINES[name], torch.Generator().manual_seed(seed)
    sample, pool, t0 = BEAMLINES[name].sample_task, [], time.time()
    for _ in range(n_tasks):
        task = sample(gen)
        K = ((torch.rand(1, bl.DIM, generator=gen) * 2 - 1)
             + half * (torch.rand(n_points, bl.DIM, generator=gen) * 2 - 1)).clamp(-1, 1)
        pool.append((K, *measure(bl, task, K, 2)))
    print(f"{name}: {n_tasks} meta-training tasks with curvatures in {time.time() - t0:.0f} s")
    os.makedirs(RESULTS, exist_ok=True)
    torch.save(pool, path)
    return pool


def train_variant(name, variant, pool):
    family, order, constr, Ks = VARIANTS[variant]
    rows = constraints(name)
    models = []
    for i in range(4):
        torch.manual_seed(i)
        kw = dict(use_slopes=order >= 1, use_curvature=order >= 2,
                  constraint_rows=rows[i] if constr else ())
        if family == "lsq":
            m = FullLSQ(BEAMLINES[name].DIM, sigma=0.5, **kw)
            m.meta_fit(pool_sampler(pool, i, Ks, order), steps=int(1500 * SCALE), batch_size=4, lr=1e-2, seed=1)
        else:
            m = FullALPaCA(BEAMLINES[name].DIM, **kw)
            m.meta_fit(pool_sampler(pool, i, Ks, order), steps=int(8000 * SCALE), batch_size=4, lr=1e-3, seed=1)
        models.append(m)
    return models


def run_loop(tu, task, models, order, gen, budget=BUDGET, return_path=False, times=None):
    bl = tu.bl
    target = tu.target(task) * 1e3
    K = torch.cat([bl.START.unsqueeze(0), tu.window(bl.START, 4, gen)])
    B, J, H = measure(bl, task, K, order)
    while len(K) < budget:
        y = torch.log((B - target).norm(dim=-1))
        t_start = time.perf_counter()
        # Off-screen readings (DESY's placeholder, only in the noisy-screen scenario of
        # beam_robust.py) are left out of the fits, as in ares_realistic.py.
        on = ~(B == OFF_SCREEN_MM).all(-1)
        last = K[-1]
        if on.sum() < 3:
            k = tu.window(last, 1, gen)[0]
            if times is not None:
                times.append(time.perf_counter() - t_start)
            b, j, h = measure(bl, task, k.unsqueeze(0), order)
            K, B = torch.cat([K, k.unsqueeze(0)]), torch.cat([B, b])
            J = None if j is None else torch.cat([J, j])
            H = None if h is None else torch.cat([H, h])
            continue
        Ko, Bo = K[on], B[on]
        Jo = None if J is None else J[on]
        Ho = None if H is None else H[on]
        with torch.no_grad():
            fits = [m.adapt(Ko, Bo[:, i], None if Jo is None else Jo[:, i], None if Ho is None else Ho[:, i])
                    for i, m in enumerate(models)]
            C = tu.window(last, N_CAND, gen)
            acq = torch.log((torch.stack([f(C) for f in fits], -1) - target).norm(dim=-1) + EPS)
            if order == 0:
                sd = torch.stack([m.posterior_sd(Ko, C) if hasattr(m, "posterior_sd")
                                  else Tuner.posterior_sd(m, Ko, C) for m in models]).mean(0)
                acq = acq - KAPPA * (y[on].max() - y[on].min()) * sd
            k = C[acq.argmin()].clone()
            if order >= 1:
                def value(z):
                    b = torch.stack([f(z.unsqueeze(0))[0] for f in fits])
                    return float(torch.log((b - target).norm() + EPS))

                def gradient(z):
                    b = torch.stack([f(z.unsqueeze(0))[0] for f in fits])
                    g = torch.stack([f.gradient(z.unsqueeze(0))[0] for f in fits])
                    r = b - target
                    return (r.unsqueeze(1) * g).sum(0) / (r.norm() ** 2 + EPS)
                k = tu.descend(value, gradient, k, step=bl.HALF / 2)
                k = torch.max(torch.min(k, last + bl.HALF), last - bl.HALF).clamp(-1, 1)
        if times is not None:
            times.append(time.perf_counter() - t_start)
        b, j, h = measure(bl, task, k.unsqueeze(0), order)
        K, B = torch.cat([K, k.unsqueeze(0)]), torch.cat([B, b])
        J = None if j is None else torch.cat([J, j])
        H = None if h is None else torch.cat([H, h])
    y = torch.log((B - target).norm(dim=-1))
    best = torch.cummin(y, 0).indices
    maes = [tu.mae_um(task, K[best[n - 1]]) for n in CHECKPOINTS]
    return (maes, K, y) if return_path else maes


def _init_train():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")


def _train_job(args):
    name, v, pool = args
    return v, train_variant(name, v, pool)


_W = {}


def _init(name, trained):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")
    _W.update(name=name, trained=trained, tu=Tuner(BEAMLINES[name]))


def _job(args):
    index, task = args
    out = {}
    for variant, models in _W["trained"].items():
        t0 = time.perf_counter()
        mae = run_loop(_W["tu"], task, models, VARIANTS[variant][1], torch.Generator().manual_seed(1000 + index))
        out[variant] = (mae, time.perf_counter() - t0)
    return index, out


def main(name, n_dev, variants=None, workers=None, seed=3031):
    pool = build_pool(name)
    variants = variants or list(VARIANTS)
    path = os.path.join(RESULTS, f"beam_full_{name}_dev_models.pt")
    trained = torch.load(path, weights_only=False) if os.path.exists(path) and SCALE == 1 else {}
    missing = [v for v in variants if v not in trained]
    if missing:                                   # one variant per process, one thread each
        t0 = time.time()
        with Pool(min(len(missing), workers or max(1, (os.cpu_count() or 2) - 2)),
                  initializer=_init_train) as p:
            for v, models in p.imap_unordered(_train_job, [(name, v, pool) for v in missing]):
                trained[v] = models
                print(f"  trained {v} ({time.time() - t0:.0f} s)", flush=True)
                if SCALE == 1:
                    torch.save(trained, path)
    trained = {v: trained[v] for v in variants}
    gen = torch.Generator().manual_seed(seed)
    tasks = [task_sampler(name)(gen) for _ in range(n_dev)]
    out_path = os.path.join(RESULTS, f"beam_full_{name}_dev.jsonl" if SCALE == 1 else f"beam_full_{name}_smoke.jsonl")
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done[r["index"]] = r["result"]
    todo = [(i, t) for i, t in enumerate(tasks) if i not in done or any(v not in done[i] for v in variants)]
    if todo:
        with Pool(workers or max(1, (os.cpu_count() or 2) - 2), initializer=_init, initargs=(name, trained)) as p, \
                open(out_path, "a") as f:
            for n, (i, res) in enumerate(p.imap_unordered(_job, todo), 1):
                done[i] = {**done.get(i, {}), **res}
                f.write(json.dumps({"index": i, "result": done[i]}) + "\n"); f.flush()
                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} dev tasks", flush=True)
    print(f"\n{name}, {n_dev} development tasks (seed {seed}): best true MAE, um (median / mean)")
    for v in variants:
        r = np.asarray([done[i][v][0] for i in range(n_dev)])
        s = np.mean([done[i][v][1] for i in range(n_dev)])
        print(f"  {v:<50s}" + "  ".join(f"{c}: {np.median(r[:, j]):6.2f} / {np.mean(r[:, j]):6.2f}"
                                          for j, c in enumerate(CHECKPOINTS)) + f"   ({s:.1f} s)")


if __name__ == "__main__":
    arg = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
    main(arg("--beamline") or "ares", int(arg("--dev") or 60),
         variants=[v.strip() for v in arg("--variants").split(";")] if arg("--variants") else None,
         workers=int(arg("--workers")) if arg("--workers") else None)
