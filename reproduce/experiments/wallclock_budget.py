"""Experiment #1.3: equal-wall-clock comparison.

Instead of comparing iteration counts, compare the best test MSE each method can
reach within a fixed WALL-CLOCK training budget B (single-thread). For
B in {0.5, 5, 50} s we train each of:

    - MLP-MAML standard  (hidden=40, depth=3, inner_steps=1)
    - Meta-LSQ-vanilla   (FastLSQ-Meta, sobolev=False)
    - Meta-LSQ-Sobolev   (FastLSQ-Meta, sobolev=True)

We run each training loop until ~B seconds of *training* time elapse. To do this
robustly we (1) calibrate per-iteration time with a short warmup, (2) pick an
n_iters that should land near B, (3) call the stock train_* function with that
n_iters and read back its REPORTED train_time (the train_* functions time only
their own loop, excluding eval). If the measured time misses B by a lot we make
one corrective re-run with a rescaled n_iters. Then we eval test MSE on 200
tasks. 3 seeds; we report mean +/- std of test MSE in a 3x3 (method x budget)
table.

Read-only import of experiments.maml_sinusoid; no existing file is modified.
"""

import os, sys, time, json, math
import torch

import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from experiments.maml_sinusoid import (
    train_mlp_maml, eval_mlp_maml,
    train_fastlsq_meta, eval_fastlsq_meta,
)

torch.set_default_dtype(torch.float32)

BUDGETS = [0.5, 5.0, 50.0]
SEEDS = [0, 1, 2]
N_TEST = 200

# Per-method config + a calibration call to learn per-iter cost.
# We never let n_iters explode: 50 s of FastLSQ is a lot of cheap iters, so we
# cap it. MLP iters are ~50x more expensive, so its cap is much lower.
METHODS = {
    "MLP-MAML": dict(
        cap=200000,
        min_iters=1,
    ),
    "Meta-LSQ-vanilla": dict(
        cap=400000,
        min_iters=1,
    ),
    "Meta-LSQ-Sobolev": dict(
        cap=400000,
        min_iters=1,
    ),
    # Same method with the deployment ridge floor of Sec. "Where Does It Break?"
    # active during meta-training. The unfloored run diverges at the 50 s budget
    # because Adam keeps shrinking an already-negligible mu; this column tests
    # whether the floor we advocate actually prevents that.
    "Meta-LSQ-Sobolev-floor": dict(
        cap=400000,
        min_iters=1,
    ),
}
MU_FLOOR = 1e-2


def train_method(name, n_iters, seed):
    """Run the stock training loop for n_iters; return (meta_or_net, train_time)."""
    # log_every huge so the per-iter .item() sync in the curve logging does not
    # distort wall-clock (curve only logs at it==0 and it==n_iters-1).
    big = max(n_iters, 1) + 10
    if name == "MLP-MAML":
        net, t, _ = train_mlp_maml(
            n_iters=n_iters, hidden=40, depth=3, inner_steps=1,
            seed=seed, log_every=big)
        return net, t
    elif name == "Meta-LSQ-vanilla":
        meta, t, _ = train_fastlsq_meta(
            sobolev=False, n_iters=n_iters, seed=seed, log_every=big)
        return meta, t
    elif name == "Meta-LSQ-Sobolev":
        meta, t, _ = train_fastlsq_meta(
            sobolev=True, n_iters=n_iters, seed=seed, log_every=big)
        return meta, t
    elif name == "Meta-LSQ-Sobolev-floor":
        meta, t, _ = train_fastlsq_meta(
            sobolev=True, n_iters=n_iters, seed=seed, log_every=big,
            mu_floor=MU_FLOOR)
        return meta, t
    raise ValueError(name)


def eval_method(name, model):
    if name == "MLP-MAML":
        mse, _ = eval_mlp_maml(model, n_test=N_TEST)
        return mse
    else:
        mse, _ = eval_fastlsq_meta(model, n_test=N_TEST)
        return mse


def calibrate_per_iter(name, seed, warm_seconds=1.0, max_warm=4000):
    """Measure STEADY-STATE per-iter seconds.

    A short cold probe badly underestimates the sustained per-iter cost (a 20-iter
    MLP probe measured ~3 ms/iter but the sustained rate was ~13 ms/iter -- autograd
    graph reuse, Adam state, single-thread thermal/cache effects). So we run a warmup
    sized to take ~warm_seconds of wall-clock and read its measured train_time, which
    reflects the steady-state rate the real run will see.
    """
    # bootstrap: tiny probe to get an order-of-magnitude, then a ~warm_seconds run.
    _, t0 = train_method(name, 20, seed)
    pi0 = t0 / 20.0
    warm_iters = int(min(max_warm, max(40, warm_seconds / max(pi0, 1e-9))))
    _, t1 = train_method(name, warm_iters, seed)
    return t1 / max(warm_iters, 1)


def run_budget(name, budget, seed, cap, min_iters):
    """Train so that the (measured) training time is <= ~budget seconds.

    We treat ``budget`` as a CEILING: it is the wall-clock each method is allowed to
    spend. We size n_iters from a steady-state per-iter estimate, then if the first
    real run overshoots we correct DOWN; if it undershoots a lot we correct UP, but we
    cap every individual train call at ~1.3*budget of expected cost so no single run
    can blow far past the budget. This keeps the comparison fair and bounded.
    """
    per_iter = calibrate_per_iter(name, seed, warm_seconds=min(1.0, 0.5 * budget))

    def size(target_s, pi):
        n = int(round(target_s / max(pi, 1e-9)))
        return max(min_iters, min(cap, n))

    # aim slightly under budget to respect the ceiling given calibration noise.
    n_iters = size(0.95 * budget, per_iter)
    model, train_time = train_method(name, n_iters, seed)

    # corrective passes: keep within [0.7B, 1.15B] if we can. For small budgets a
    # re-run is cheap, so we allow up to 2. For large budgets (>=10s) a re-run costs
    # ~B seconds; the steady-state calibration above already lands within ~10-15%, so
    # we allow at most ONE downward correction (to honor the ceiling) and never a
    # second run. Each corrective run is sized to cost <= ~budget (never above).
    max_passes = 2 if budget < 10.0 else 1
    for _ in range(max_passes):
        if n_iters >= cap:
            break  # pinned at cap: cannot get closer to budget, accept it
        too_low = train_time < 0.7 * budget
        too_high = train_time > 1.15 * budget
        # for large budgets only correct DOWNWARD (overshoot) -- an undershoot just
        # means the method got slightly less than B and we accept it (still <= ceiling).
        if budget >= 10.0:
            too_low = False
        if not (too_low or too_high):
            break
        # re-estimate per-iter from the actual run (most representative sample).
        pi = train_time / max(n_iters, 1)
        n2 = size(0.95 * budget, pi)
        n2 = max(min_iters, min(cap, n2))
        if n2 == n_iters:
            break
        model, train_time = train_method(name, n2, seed)
        n_iters = n2

    mse = eval_method(name, model)
    return {"iters": int(n_iters), "train_time": float(train_time),
            "test_mse": float(mse)}


def main():
    t_start = time.time()
    results = {name: {f"{b}": [] for b in BUDGETS} for name in METHODS}

    for seed in SEEDS:
        for name, cfg in METHODS.items():
            for b in BUDGETS:
                r = run_budget(name, b, seed, cfg["cap"], cfg["min_iters"])
                results[name][f"{b}"].append({"seed": seed, **r})
                print(f"[seed {seed}] {name:<18} B={b:>4}s  "
                      f"iters={r['iters']:>7d}  t={r['train_time']:>6.2f}s  "
                      f"mse={r['test_mse']:.4f}",
                      flush=True)

    # aggregate mean/std across seeds
    summary = {}
    for name in METHODS:
        summary[name] = {}
        for b in BUDGETS:
            mses = torch.tensor([e["test_mse"] for e in results[name][f"{b}"]])
            times = torch.tensor([e["train_time"] for e in results[name][f"{b}"]])
            iters = [e["iters"] for e in results[name][f"{b}"]]
            summary[name][f"{b}"] = {
                "mse_mean": float(mses.mean()),
                "mse_std": float(mses.std(unbiased=False)),
                "time_mean": float(times.mean()),
                "iters_mean": float(sum(iters) / len(iters)),
            }

    out = {
        "experiment": "1.3 equal-wall-clock comparison",
        "budgets_s": BUDGETS,
        "seeds": SEEDS,
        "n_test": N_TEST,
        "methods": list(METHODS.keys()),
        "per_seed": results,
        "summary": summary,
        "total_wall_s": time.time() - t_start,
    }
    os.makedirs(_os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    path = _os.path.join(_REPO_ROOT, "results", "wallclock_budget.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # pretty 3x3 table
    print("\n=== test MSE (mean +/- std over 3 seeds) vs wall-clock budget ===")
    hdr = f"{'method':<18}" + "".join(f"{('B='+str(b)+'s'):>22}" for b in BUDGETS)
    print(hdr)
    for name in METHODS:
        row = f"{name:<18}"
        for b in BUDGETS:
            s = summary[name][f"{b}"]
            row += f"{s['mse_mean']:>12.4f}+/-{s['mse_std']:<7.4f}"
        print(row)
    print(f"\n[saved] {path}")
    print(f"[total wall] {out['total_wall_s']:.1f}s")


if __name__ == "__main__":
    main()
