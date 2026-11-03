"""A non-synthetic task family: gas-sensor calibration under drift.

Every family elsewhere in the paper is synthetic, which is the limitation the
discussion names. This is a real one.

Data: UCI Air Quality (De Vito et al.), one year of hourly co-located readings
from a metal-oxide sensor array and a reference analyser in an Italian city.
Task family: recover the sensor's CALIBRATION CURVE -- true CO concentration
(reference analyser, mg/m^3) as a function of the tin-oxide sensor response
PT08.S1(CO). One task per ISO week. The curve is smooth, monotone and NOT
periodic, which matters: a diurnal signal would have flattered a sinusoidal
basis, and this deliberately does not.

The tasks genuinely differ. Per-week least-squares slope of the calibration
curve varies ~1.9x across the year -- this is the sensor drift the dataset was
collected to study, and it is what makes the family a meta-learning problem
rather than one regression repeated 54 times.

Split is CHRONOLOGICAL: meta-train on the earlier weeks, meta-test on the later
ones, so test tasks are drifted relative to anything seen in training. A random
split would be easier and less honest.

Meta-LSQ runs VALUE-ONLY (w = 0, two learnable scalars). Real data carries no
measured derivatives, and finite-differencing the observations would make the
"derivative" rows a transform of the value rows rather than new information, so
the Sobolev variant is deliberately not claimed here.

Baselines on the identical splits:
  MLP-MAML        the paper's standard baseline, 2x40 ReLU, one inner step
  per-task linear least squares on the K support points -- the honest comparator
                  for a near-linear calibration curve, and the one a
                  practitioner would actually use
  global linear   one calibration fitted on all meta-train points, applied
                  unchanged to every test task (i.e. ignore drift)

Run:  OMP_NUM_THREADS=1 python3 experiments/real_calibration.py --save results/real_calibration.json
"""

import os, sys, math, json, time, argparse, zipfile, io, urllib.request

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import (
    make_basis, fastlsq_inner_solve, fastlsq_predict, MAMLNet, maml_inner_step,
)

torch.set_default_dtype(torch.float32)

SEEDS = (0, 1, 2, 3, 4)
DATA_DIR = os.path.join(_REPO_ROOT, "data")
CSV = os.path.join(DATA_DIR, "AirQualityUCI.csv")
URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
RESULT_PATH = os.path.join(_REPO_ROOT, "results", "real_calibration.json")

# Two calibration targets, chosen by a stated rule rather than by result: we
# screened all five (sensor, reference) pairs in the dataset for how much a
# degree-5 fit improves on a straight line, and took the LEAST and the MOST
# nonlinear. The paper claims a regime (smooth, nonlinear, low-dimensional), so
# a family with no nonlinearity to exploit is a genuine test of that claim and
# is reported alongside, not instead of, the one that suits the method.
TARGETS = {
    "CO (near-linear)":  ("PT08.S1(CO)", "CO(GT)"),
    "NOx (nonlinear)":   ("PT08.S3(NOx)", "NOx(GT)"),
}

K = 10          # support points per task, matching the sinusoid protocol
K_Q = 50        # query points per task
MIN_PTS = 60    # a week must have this many valid readings to be a task


def ensure_data():
    if os.path.exists(CSV):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  downloading {URL}")
    b = urllib.request.urlopen(URL, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(b)).extractall(DATA_DIR)


def load_tasks(sensor="PT08.S1(CO)", ref="CO(GT)"):
    """Return [(x, y)] per week, raw units, chronologically ordered."""
    import pandas as pd
    ensure_data()
    df = pd.read_csv(CSV, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(how="all")
    sub = df[["Date", ref, sensor]].replace(-200, np.nan).dropna()
    dt = pd.to_datetime(sub["Date"], format="%d/%m/%Y")
    iso = dt.dt.isocalendar()
    key = (iso.year.astype(int) * 100 + iso.week.astype(int)).values
    order, tasks = [], []
    for w in sorted(set(key)):
        m = key == w
        if m.sum() >= MIN_PTS:
            order.append(w)
            tasks.append((sub[sensor].values[m].astype(np.float64),
                          sub[ref].values[m].astype(np.float64)))
    return tasks, order


def split_tasks(tasks, frac=0.6):
    """Chronological split: earlier weeks meta-train, later weeks meta-test."""
    n = int(round(frac * len(tasks)))
    return tasks[:n], tasks[n:]


def normaliser(train_tasks):
    """Standardise with META-TRAIN statistics only (no test leakage)."""
    x = np.concatenate([t[0] for t in train_tasks])
    y = np.concatenate([t[1] for t in train_tasks])
    return (x.mean(), x.std(), y.mean(), y.std())


def draw(task, norm, gen, k=K, kq=K_Q):
    """Random support/query split within one week."""
    xm, xs_, ym, ys_ = norm
    x = torch.tensor((task[0] - xm) / xs_, dtype=torch.float32)
    y = torch.tensor((task[1] - ym) / ys_, dtype=torch.float32)
    p = torch.randperm(x.numel(), generator=gen)
    si, qi = p[:k], p[k:k + kq]
    return x[si].unsqueeze(-1), y[si], x[qi].unsqueeze(-1), y[qi]


def batch(tasks, norm, gen, bs):
    idx = torch.randint(len(tasks), (bs,), generator=gen)
    return [draw(tasks[int(i)], norm, gen) for i in idx]


# ---------------- Meta-LSQ, value only ----------------

def train_metalsq(tasks, norm, n_iters=2000, N=200, lr=5e-3, bs=4, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu = torch.tensor(math.log(1e-4), requires_grad=True)
    opt = torch.optim.Adam([log_sigma, log_mu], lr=lr)
    t0 = time.time()
    for _ in range(n_iters):
        opt.zero_grad()
        loss = 0.0
        for xs_, ys_, xq_, yq_ in batch(tasks, norm, gen, bs):
            W, beta = fastlsq_inner_solve(xs_, ys_, ys_, Wb, bb,
                                          log_sigma.exp(), log_mu.exp(),
                                          sobolev=False)
            loss = loss + (fastlsq_predict(xq_, W, bb, beta) - yq_).pow(2).mean()
        (loss / bs).backward()
        opt.step()
    return {"Wb": Wb, "bb": bb, "sigma": float(log_sigma.exp()),
            "mu": float(log_mu.exp()), "train_s": time.time() - t0}


def eval_metalsq(m, tasks, norm, seed=0, mu_floor=0.0):
    """``mu_floor`` applies the deployment ridge floor of Sec. "Where Does It
    Break?" at test time, exactly as the out-of-class table does."""
    gen = torch.Generator().manual_seed(seed + 9999)
    mu_use = max(m["mu"], mu_floor)
    out = []
    with torch.no_grad():
        for t in tasks:
            xs_, ys_, xq_, yq_ = draw(t, norm, gen)
            W, beta = fastlsq_inner_solve(xs_, ys_, ys_, m["Wb"], m["bb"],
                                          torch.tensor(m["sigma"]),
                                          torch.tensor(mu_use), sobolev=False)
            out.append(float((fastlsq_predict(xq_, W, m["bb"], beta) - yq_).pow(2).mean()))
    return out


# ---------------- MLP-MAML ----------------

def train_maml(tasks, norm, n_iters=8000, lr=1e-3, inner_lr=1e-2, bs=4, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    net = MAMLNet(hidden=40, depth=3)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(n_iters):
        opt.zero_grad()
        loss = 0.0
        for xs_, ys_, xq_, yq_ in batch(tasks, norm, gen, bs):
            p = net.init_params()
            p = maml_inner_step(net, p, xs_, ys_, inner_lr)
            loss = loss + (net(xq_, p) - yq_).pow(2).mean()
        (loss / bs).backward()
        opt.step()
    return net, time.time() - t0


def eval_maml(net, tasks, norm, inner_lr=1e-2, seed=0):
    gen = torch.Generator().manual_seed(seed + 9999)
    out = []
    for t in tasks:
        xs_, ys_, xq_, yq_ = draw(t, norm, gen)
        p = net.init_params()
        p = maml_inner_step(net, p, xs_, ys_, inner_lr)
        with torch.no_grad():
            out.append(float((net(xq_, p) - yq_).pow(2).mean()))
    return out


# ---------------- classical baselines ----------------

def eval_pertask_linear(tasks, norm, seed=0, deg=1):
    """Least squares on the K support points of each test task."""
    gen = torch.Generator().manual_seed(seed + 9999)
    out = []
    for t in tasks:
        xs_, ys_, xq_, yq_ = draw(t, norm, gen)
        c = np.polyfit(xs_.squeeze(-1).numpy(), ys_.numpy(), deg)
        out.append(float(((np.polyval(c, xq_.squeeze(-1).numpy()) - yq_.numpy()) ** 2).mean()))
    return out


def eval_global_linear(train_tasks, test_tasks, norm, seed=0):
    """One calibration fitted on all meta-train data, drift ignored."""
    xm, xs_, ym, ys_ = norm
    X = np.concatenate([(t[0] - xm) / xs_ for t in train_tasks])
    Y = np.concatenate([(t[1] - ym) / ys_ for t in train_tasks])
    c = np.polyfit(X, Y, 1)
    gen = torch.Generator().manual_seed(seed + 9999)
    out = []
    for t in test_tasks:
        _, _, xq_, yq_ = draw(t, norm, gen)
        out.append(float(((np.polyval(c, xq_.squeeze(-1).numpy()) - yq_.numpy()) ** 2).mean()))
    return out


def nonlinearity(tasks):
    """How much a degree-5 fit beats a line, pooled. The screen that picked the targets."""
    x = np.concatenate([t[0] for t in tasks]); y = np.concatenate([t[1] for t in tasks])
    def r2(deg):
        r = y - np.polyval(np.polyfit(x, y, deg), x)
        return 1 - r.var() / y.var()
    return {"r2_linear": r2(1), "r2_deg5": r2(5), "gain": r2(5) - r2(1)}


def stats(v):
    t = torch.tensor(v, dtype=torch.float64)
    return {"mean": float(t.mean()), "median": float(t.median()),
            "n": int(t.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=RESULT_PATH)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters-lsq", type=int, default=2000)
    ap.add_argument("--iters-maml", type=int, default=8000)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]

    all_out = {}
    for tgt_name, (sensor, ref) in TARGETS.items():
        print(f"\n########## target: {tgt_name}  [{sensor} -> {ref}] ##########")
        all_out[tgt_name] = run_one(sensor, ref, seeds, args)
    p = args.save if os.path.isabs(args.save) else os.path.join(_REPO_ROOT, args.save)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\n[saved] {p}")


def run_one(sensor, ref, seeds, args):
    tasks, weeks = load_tasks(sensor, ref)
    tr, te = split_tasks(tasks)
    norm = normaliser(tr)
    print(f"tasks (weeks): {len(tasks)}   meta-train {len(tr)}   meta-test {len(te)}")
    nl = nonlinearity(tr)
    print(f"chronological split at week {weeks[len(tr)]}; y standardised on meta-train only")
    print(f"nonlinearity of this calibration curve: linear R2 {nl['r2_linear']:.3f}, "
          f"deg5 R2 {nl['r2_deg5']:.3f}, gain {nl['gain']:+.3f}")

    res = {m: {"per_seed_mean": [], "per_seed_median": []}
           for m in ("Meta-LSQ (w=0)", "Meta-LSQ (w=0, floor)", "MLP-MAML",
                     "per-task linear", "global linear")}
    sig, mus, tl, tm = [], [], [], []

    for s in seeds:
        m = train_metalsq(tr, norm, n_iters=args.iters_lsq, seed=s)
        v = eval_metalsq(m, te, norm, seed=s)
        sig.append(m["sigma"]); mus.append(m["mu"]); tl.append(m["train_s"])
        res["Meta-LSQ (w=0)"]["per_seed_mean"].append(stats(v)["mean"])
        res["Meta-LSQ (w=0)"]["per_seed_median"].append(stats(v)["median"])

        vf = eval_metalsq(m, te, norm, seed=s, mu_floor=1e-2)
        res["Meta-LSQ (w=0, floor)"]["per_seed_mean"].append(stats(vf)["mean"])
        res["Meta-LSQ (w=0, floor)"]["per_seed_median"].append(stats(vf)["median"])

        net, t_ = train_maml(tr, norm, n_iters=args.iters_maml, seed=s)
        v = eval_maml(net, te, norm, seed=s); tm.append(t_)
        res["MLP-MAML"]["per_seed_mean"].append(stats(v)["mean"])
        res["MLP-MAML"]["per_seed_median"].append(stats(v)["median"])

        v = eval_pertask_linear(te, norm, seed=s)
        res["per-task linear"]["per_seed_mean"].append(stats(v)["mean"])
        res["per-task linear"]["per_seed_median"].append(stats(v)["median"])

        v = eval_global_linear(tr, te, norm, seed=s)
        res["global linear"]["per_seed_mean"].append(stats(v)["mean"])
        res["global linear"]["per_seed_median"].append(stats(v)["median"])
        print(f"  seed {s} done  (sigma {sig[-1]:.3f}, mu {mus[-1]:.2e})")

    print(f"\n=== held-out weeks, normalised query MSE ({len(seeds)} seeds) ===")
    print(f"{'method':<18}{'mean':>18}{'median':>18}")
    out = {}
    for k, v in res.items():
        a = torch.tensor(v["per_seed_mean"]); b = torch.tensor(v["per_seed_median"])
        out[k] = {"mean": float(a.mean()), "mean_std": float(a.std(unbiased=True)),
                  "median": float(b.mean()), "median_std": float(b.std(unbiased=True)),
                  "per_seed_mean": v["per_seed_mean"]}
        print(f"{k:<18}{out[k]['mean']:>10.4f} +/-{out[k]['mean_std']:<6.4f}"
              f"{out[k]['median']:>10.4f} +/-{out[k]['median_std']:<6.4f}")

    lsq = out["Meta-LSQ (w=0, floor)"]["mean"]
    mlp = out["MLP-MAML"]["mean"]; lin = out["per-task linear"]["mean"]
    print(f"\nMeta-LSQ(floor) vs MLP-MAML        : {mlp / lsq:.2f}x")
    print(f"Meta-LSQ(floor) vs per-task linear : {lin / lsq:.2f}x")
    print(f"unfloored mean/median ratio        : "
          f"{out['Meta-LSQ (w=0)']['mean'] / out['Meta-LSQ (w=0)']['median']:.1f}"
          f"  (heavy tail; floored: "
          f"{out['Meta-LSQ (w=0, floor)']['mean'] / out['Meta-LSQ (w=0, floor)']['median']:.1f})")
    print(f"learned sigma {np.mean(sig):.3f} +/- {np.std(sig):.3f}   "
          f"mu {np.mean(mus):.2e} +/- {np.std(mus):.1e}")
    print(f"  (on the noiseless sinusoid benchmark the learned mu is ~1e-5; "
          f"real data should push it up -- assumption A2)")
    print(f"train_s: Meta-LSQ {np.mean(tl):.1f}  MLP-MAML {np.mean(tm):.1f}")

    blob = {"n_tasks": len(tasks), "n_train": len(tr), "n_test": len(te),
            "K": K, "K_q": K_Q, "seeds": list(seeds), "results": out,
            "sigma": float(np.mean(sig)), "sigma_std": float(np.std(sig)),
            "mu": float(np.mean(mus)), "mu_std": float(np.std(mus)),
            "train_s_metalsq": float(np.mean(tl)),
            "train_s_maml": float(np.mean(tm)),
            "ratio_vs_maml": mlp / lsq, "ratio_vs_pertask_linear": lin / lsq,
            "sensor": sensor, "ref": ref, "nonlinearity": nl}
    return blob


if __name__ == "__main__":
    main()
