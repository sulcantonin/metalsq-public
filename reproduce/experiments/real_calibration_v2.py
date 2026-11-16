"""UCI Air Quality calibration, fixed design selected by cross-validation (Sec. 5.2).

The first real-data study (rebuttal_real_hypotheses.py) gave Meta-LSQ the sensor reading
alone, a zero prior mean and 200 features. Exploratory runs (two rounds, which also scored
the held-out weeks) pointed to three changes; this script fixes them up front and decides
everything else by cross-validation on the training weeks only.

Fixed design, every method on the same tasks:
  Meta-LSQ  N = 1000 random features with one bandwidth per input; a polynomial trend
            block t(x) = (1, inputs, sensor^2) with its own meta-learned Tikhonov strength;
            a meta-learned prior mean on the trend coefficients. Per task, in dual form,
              beta* = beta_bar + D^-1 F^T (F D^-1 F^T + I)^-1 (y - F beta_bar),
            which is the ridge solve pulled toward beta_bar instead of zero.
  GP        exact RBF kernel, one length-scale per input, the same learned trend as its
            mean function, shared noise; all meta-learned by query MSE.
  MAML      MLP-MAML as in the paper (two hidden layers of 40, one inner step, 8000 steps).
  ALPaCA    ALPaCA-style: Bayesian linear regression on a learned tanh encoder (two layers
            of 40, 40 features plus a bias), with a meta-learned prior mean on the
            last-layer weights, as in the original ALPaCA, and a learned isotropic
            precision; 8000 Adam steps at 1e-3, like MAML.
  timing    every test run logs training seconds, per-week adaptation and prediction
            milliseconds, and the number of meta-learned parameters (one thread).
  per-task  least squares on each week's support points: line, quadratic in the sensor.
Selected by blocked 4-fold cross-validation over the training weeks (contiguous blocks):
  inputs    sensor reading alone, or sensor + temperature + relative humidity (all methods
            except the per-task quadratic);
  floor     Meta-LSQ deployment ridge floor c * mu*, c in {1, 3, 10, 30}.
The held-out (later) weeks are then scored once, 5 seeds. For reference the run also
recomputes the paper's first configuration (Meta-LSQ: sensor only, 200 features, no trend,
no prior mean, floor 10 mu*; GP: sensor only, zero mean) on the same tasks.

Hours with a missing temperature or humidity are dropped for every method, so the task
set differs slightly from rebuttal_real_hypotheses.py.

Run:  OMP_NUM_THREADS=1 python3 experiments/real_calibration_v2.py --target CO
      (one process per target; --merge combines results/real_calibration_v2_*.json)
"""

import os, sys, math, json, time, argparse

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.maml_sinusoid import MAMLNet, maml_inner_step

torch.set_default_dtype(torch.float64)
CSV = os.path.join(_REPO_ROOT, "data", "AirQualityUCI.csv")
RESULTS = os.path.join(_REPO_ROOT, "results")
TARGETS = {"CO": ("PT08.S1(CO)", "CO(GT)"), "NO2": ("PT08.S4(NO2)", "NO2(GT)"),
           "C6H6": ("PT08.S2(NMHC)", "C6H6(GT)"), "NOx": ("PT08.S3(NOx)", "NOx(GT)")}
K, KQ, MIN_PTS = 10, 50, 60
ITERS, ITERS_MAML = 2000, 8000
TEST_SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 4
FLOORS = (1, 3, 10, 30)
INPUTS = {"x": [0], "xTR": [0, 1, 2]}


# ---------------- data ----------------
def load(sensor, ref):
    import pandas as pd
    df = pd.read_csv(CSV, sep=";", decimal=",").dropna(axis=1, how="all").dropna(how="all")
    sub = df[["Date", ref, sensor, "T", "RH"]].replace(-200, np.nan).dropna()
    dt = pd.to_datetime(sub["Date"], format="%d/%m/%Y")
    iso = dt.dt.isocalendar()
    key = (iso.year.astype(int) * 100 + iso.week.astype(int)).values
    tasks = []
    for w in sorted(set(key)):
        m = key == w
        if m.sum() >= MIN_PTS:
            X = np.stack([sub[sensor].values[m], sub["T"].values[m], sub["RH"].values[m]], 1)
            tasks.append((X.astype(np.float64), sub[ref].values[m].astype(np.float64)))
    return tasks


def normaliser(tr):
    X = np.concatenate([t[0] for t in tr]); y = np.concatenate([t[1] for t in tr])
    return X.mean(0), X.std(0), y.mean(), y.std()


def draw(task, norm, gen, cols):
    xm, xs, ym, ys = norm
    X = torch.tensor((task[0] - xm) / xs)[:, cols]
    y = torch.tensor((task[1] - ym) / ys)
    p = torch.randperm(len(y), generator=gen)
    s, q = p[:K], p[K:K + KQ]
    return X[s], y[s], X[q], y[q]


def trend(X):
    """1, every input, and the square of the sensor reading."""
    return torch.cat([torch.ones(len(X), 1), X, X[:, :1] ** 2], 1)


def rff(X, Wb, bb, sig):
    return torch.sin(X @ (sig[:, None] * Wb) + bb) / math.sqrt(Wb.shape[1])


# ---------------- Meta-LSQ ----------------
def lsq_pred(Xs, ys, Xq, P, c=1.0):
    Fs, Fq = rff(Xs, P["Wb"], P["bb"], P["sig"]), rff(Xq, P["Wb"], P["bb"], P["sig"])
    N = Fs.shape[1]
    reg = (c * P["mu"]).expand(N)
    bbar = torch.zeros(N)
    if P["trend"]:
        Ts, Tq = trend(Xs), trend(Xq)
        Fs, Fq = torch.cat([Fs, Ts], 1), torch.cat([Fq, Tq], 1)
        reg = torch.cat([reg, P["mu_t"].expand(Ts.shape[1])])
        bbar = torch.cat([bbar, P["bbar"]])
    Dinv = 1.0 / reg
    G = (Fs * Dinv) @ Fs.T + torch.eye(len(ys))
    beta = bbar + Dinv * (Fs.T @ torch.linalg.solve(G, ys - Fs @ bbar))
    return Fq @ beta


def train_lsq(tr, norm, cols, seed, N=1000, use_trend=True):
    d = len(cols)
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1000)
    g = torch.Generator().manual_seed(seed)
    Wb = torch.randn(d, N, generator=g); bb = torch.rand(1, N, generator=g) * 2 * math.pi
    ls = torch.zeros(d, requires_grad=True)
    lm = torch.tensor(math.log(1e-4), requires_grad=True)
    lt = torch.tensor(math.log(1e-4), requires_grad=True)
    bbar = torch.zeros(d + 2, requires_grad=True)
    groups = [{"params": [ls, lm] + ([lt] if use_trend else []), "lr": 5e-3}]
    if use_trend:
        groups.append({"params": [bbar], "lr": 2e-2})
    opt = torch.optim.Adam(groups)
    P = lambda: {"Wb": Wb, "bb": bb, "sig": ls.exp(), "mu": lm.exp(), "mu_t": lt.exp(),
                 "bbar": bbar, "trend": use_trend}
    for _ in range(ITERS):
        opt.zero_grad(); loss = 0.0
        for i in torch.randint(len(tr), (4,), generator=gen):
            Xs, ys, Xq, yq = draw(tr[int(i)], norm, gen, cols)
            loss = loss + (lsq_pred(Xs, ys, Xq, P()) - yq).pow(2).mean()
        (loss / 4).backward(); opt.step()
    with torch.no_grad():
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in P().items()}


def eval_lsq(P, te, norm, cols, seed, c):
    gen = torch.Generator().manual_seed(seed + 9999); out = []
    with torch.no_grad():
        for t in te:
            Xs, ys, Xq, yq = draw(t, norm, gen, cols)
            out.append(float((lsq_pred(Xs, ys, Xq, P, c) - yq).pow(2).mean()))
    return out


# ---------------- shared GP ----------------
def gp_pred(Xs, ys, Xq, P):
    def k(A, B):
        D = (A[:, None, :] - B[None, :, :]) / P["l"]
        return torch.exp(-0.5 * (D ** 2).sum(-1))
    m = (lambda X: trend(X) @ P["mb"]) if P["trend"] else (lambda X: torch.zeros(len(X)))
    Kss = k(Xs, Xs) + P["lam"] * torch.eye(len(Xs))
    for jit in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        L, info = torch.linalg.cholesky_ex(Kss + jit * torch.eye(len(Xs)))
        if int(info) == 0:
            break
    alpha = torch.cholesky_solve((ys - m(Xs))[:, None], L)[:, 0]
    return m(Xq) + k(Xq, Xs) @ alpha


def train_gp(tr, norm, cols, seed, use_trend=True):
    d = len(cols)
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1000)
    ll = torch.zeros(d, requires_grad=True)
    ln = torch.tensor(math.log(1e-2), requires_grad=True)
    mb = torch.zeros(d + 2, requires_grad=True)
    groups = [{"params": [ll, ln], "lr": 5e-3}]
    if use_trend:
        groups.append({"params": [mb], "lr": 2e-2})
    opt = torch.optim.Adam(groups)
    P = lambda: {"l": ll.exp(), "lam": ln.exp(), "mb": mb, "trend": use_trend}
    for _ in range(ITERS):
        opt.zero_grad(); loss = 0.0
        for i in torch.randint(len(tr), (4,), generator=gen):
            Xs, ys, Xq, yq = draw(tr[int(i)], norm, gen, cols)
            loss = loss + (gp_pred(Xs, ys, Xq, P()) - yq).pow(2).mean()
        (loss / 4).backward(); opt.step()
    with torch.no_grad():
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in P().items()}


def eval_gp(P, te, norm, cols, seed):
    gen = torch.Generator().manual_seed(seed + 9999); out = []
    with torch.no_grad():
        for t in te:
            Xs, ys, Xq, yq = draw(t, norm, gen, cols)
            out.append(float((gp_pred(Xs, ys, Xq, P) - yq).pow(2).mean()))
    return out


# ---------------- MLP-MAML ----------------
def train_maml(tr, norm, cols, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1000)
    net = MAMLNet(hidden=40, depth=3, in_dim=len(cols)).double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(ITERS_MAML):
        opt.zero_grad(); loss = 0.0
        for i in torch.randint(len(tr), (4,), generator=gen):
            Xs, ys, Xq, yq = draw(tr[int(i)], norm, gen, cols)
            p = maml_inner_step(net, net.init_params(), Xs, ys, 1e-2)
            loss = loss + (net(Xq, p) - yq).pow(2).mean()
        (loss / 4).backward(); opt.step()
    return net


def eval_maml(net, te, norm, cols, seed):
    gen = torch.Generator().manual_seed(seed + 9999); out = []
    for t in te:
        Xs, ys, Xq, yq = draw(t, norm, gen, cols)
        p = maml_inner_step(net, net.init_params(), Xs, ys, 1e-2)
        with torch.no_grad():
            out.append(float((net(Xq, p) - yq).pow(2).mean()))
    return out


# ---------------- ALPaCA-style: learned encoder, learned prior mean ----------------
class Encoder(torch.nn.Module):
    """Two tanh layers of 40 and 40 features, as in the paper's ALPaCA-style model."""
    def __init__(self, d, hidden=40, p=40):
        super().__init__()
        self.l1 = torch.nn.Linear(d, hidden); self.l2 = torch.nn.Linear(hidden, p)

    def forward(self, X):
        h = torch.tanh(self.l2(torch.tanh(self.l1(X))))
        return torch.cat([h, torch.ones(len(X), 1)], 1)        # bias column


def alpaca_pred(Xs, ys, Xq, enc, lam, bbar):
    """Bayesian linear regression on learned features, shrunk toward a learned
    prior mean bbar (as in the original ALPaCA), isotropic prior precision lam."""
    Ps, Pq = enc(Xs), enc(Xq)
    G = Ps.T @ Ps + lam * torch.eye(Ps.shape[1])
    beta = bbar + torch.linalg.solve(G, Ps.T @ (ys - Ps @ bbar))
    return Pq @ beta


def train_alpaca(tr, norm, cols, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1000)
    enc = Encoder(len(cols)).double()
    ll = torch.tensor(math.log(1e-2), requires_grad=True)
    bbar = torch.zeros(41, requires_grad=True)
    opt = torch.optim.Adam(list(enc.parameters()) + [ll, bbar], lr=1e-3)
    for _ in range(ITERS_MAML):
        opt.zero_grad(); loss = 0.0
        for i in torch.randint(len(tr), (4,), generator=gen):
            Xs, ys, Xq, yq = draw(tr[int(i)], norm, gen, cols)
            loss = loss + (alpaca_pred(Xs, ys, Xq, enc, ll.exp(), bbar) - yq).pow(2).mean()
        (loss / 4).backward(); opt.step()
    return {"enc": enc, "lam": ll.exp().detach(), "bbar": bbar.detach()}


def eval_alpaca(m, te, norm, cols, seed):
    gen = torch.Generator().manual_seed(seed + 9999); out = []
    with torch.no_grad():
        for t in te:
            Xs, ys, Xq, yq = draw(t, norm, gen, cols)
            out.append(float((alpaca_pred(Xs, ys, Xq, m["enc"], m["lam"], m["bbar"]) - yq).pow(2).mean()))
    return out


def eval_pertask(te, norm, cols, seed, quad):
    gen = torch.Generator().manual_seed(seed + 9999); out = []
    F = lambda X: torch.cat([torch.ones(len(X), 1), X] + ([X[:, :1] ** 2] if quad else []), 1)
    for t in te:
        Xs, ys, Xq, yq = draw(t, norm, gen, cols)
        beta = torch.linalg.lstsq(F(Xs), ys[:, None]).solution[:, 0]
        out.append(float((F(Xq) @ beta - yq).pow(2).mean()))
    return out


# ---------------- one configuration, trained and scored ----------------
def score(family, inp, tr, te, seed, timing=None):
    """Per-task query MSEs of one configuration; Meta-LSQ returns one list per floor.
    If ``timing`` is a dict, it receives the training time (s), the per-week
    adaptation-and-prediction time (ms) and the number of meta-learned parameters."""
    norm = normaliser(tr); cols = INPUTS[inp]; d = len(cols)
    t0 = time.perf_counter(); P = None; n_meta = 0
    if family == "Meta-LSQ":
        P = train_lsq(tr, norm, cols, seed); n_meta = d + 2 + (d + 2)
        t1 = time.perf_counter()
        r = {c: eval_lsq(P, te, norm, cols, seed, c) for c in FLOORS}
        t2 = time.perf_counter(); t2 = t1 + (t2 - t1) / len(FLOORS)   # one floor's evaluation
    elif family == "Meta-LSQ (first design)":
        P = train_lsq(tr, norm, cols, seed, N=200, use_trend=False); n_meta = d + 1
        t1 = time.perf_counter(); r = {10: eval_lsq(P, te, norm, cols, seed, 10)}; t2 = time.perf_counter()
    elif family in ("GP", "GP (first design)"):
        trend_ = family == "GP"
        g = train_gp(tr, norm, cols, seed, use_trend=trend_); n_meta = d + 1 + ((d + 2) if trend_ else 0)
        t1 = time.perf_counter(); r = eval_gp(g, te, norm, cols, seed); t2 = time.perf_counter()
    elif family == "MAML":
        net = train_maml(tr, norm, cols, seed); n_meta = sum(p.numel() for p in net.parameters())
        t1 = time.perf_counter(); r = eval_maml(net, te, norm, cols, seed); t2 = time.perf_counter()
    elif family == "ALPaCA-style":
        m = train_alpaca(tr, norm, cols, seed)
        n_meta = sum(p.numel() for p in m["enc"].parameters()) + 1 + m["bbar"].numel()
        t1 = time.perf_counter(); r = eval_alpaca(m, te, norm, cols, seed); t2 = time.perf_counter()
    elif family in ("per-task linear", "per-task quadratic"):
        t1 = time.perf_counter()
        r = eval_pertask(te, norm, cols, seed, quad=family.endswith("quadratic"))
        t2 = time.perf_counter()
    else:
        raise ValueError(family)
    if timing is not None:
        timing.update({"train_s": t1 - t0, "adapt_ms": 1e3 * (t2 - t1) / len(te), "n_meta": int(n_meta)})
    return r, P


CANDIDATES = {"Meta-LSQ": ("x", "xTR"), "GP": ("x", "xTR"), "MAML": ("x", "xTR"),
              "ALPaCA-style": ("x", "xTR"),
              "per-task linear": ("x", "xTR"), "per-task quadratic": ("x",)}
REFERENCE = {"Meta-LSQ (first design)": "x", "GP (first design)": "x"}


def run_target(target):
    sensor, ref = TARGETS[target]
    tasks = load(sensor, ref)
    n = int(round(0.6 * len(tasks)))
    tr, te = tasks[:n], tasks[n:]
    t0 = time.time()
    print(f"[{target}] weeks {len(tasks)}; train {len(tr)}; test {len(te)}", flush=True)

    # 1. blocked cross-validation over the training weeks
    blocks = np.array_split(np.arange(len(tr)), N_FOLDS)
    cv = {}
    for f, b in enumerate(blocks):
        val = [tr[i] for i in b]; fit = [tr[i] for i in range(len(tr)) if i not in set(b)]
        for fam, inps in CANDIDATES.items():
            for inp in inps:
                r, _ = score(fam, inp, fit, val, seed=f)
                if isinstance(r, dict):
                    for c, v in r.items():
                        cv.setdefault(f"{fam}|{inp}|floor={c}", []).append(float(np.mean(v)))
                else:
                    cv.setdefault(f"{fam}|{inp}", []).append(float(np.mean(r)))
        print(f"  CV fold {f} done ({time.time() - t0:.0f} s)", flush=True)
    selected = {}
    for k, v in cv.items():
        fam = k.split("|")[0]
        if fam not in selected or np.mean(v) < np.mean(cv[selected[fam]]):
            selected[fam] = k

    # 2. the held-out weeks, once, with the selected configurations and the references
    test, learned, timing = {}, {}, {}
    run = [(k.split("|")[0], k.split("|")[1], k) for k in selected.values()]
    run += [(fam, inp, f"{fam}|{inp}|floor=10" if fam.startswith("Meta") else f"{fam}|{inp}")
            for fam, inp in REFERENCE.items()]
    for seed in TEST_SEEDS:
        for fam, inp, key in run:
            tm = {}
            r, P = score(fam, inp, tr, te, seed, timing=tm)
            timing.setdefault(key, []).append(tm)
            if isinstance(r, dict):
                c = int(key.split("floor=")[1]); r = r[c]
                if P is not None and fam == "Meta-LSQ":
                    learned.setdefault("sigma", []).append([float(s) for s in P["sig"]])
                    learned.setdefault("mu", []).append(float(P["mu"]))
                    learned.setdefault("mu_trend", []).append(float(P["mu_t"]))
            test.setdefault(key, []).append(float(np.mean(r)))
        print(f"  test seed {seed} done ({time.time() - t0:.0f} s)", flush=True)

    out = {"target": target, "sensor": sensor, "ref": ref, "n_weeks": len(tasks),
           "n_train": len(tr), "n_test": len(te), "K": K, "K_q": KQ, "n_folds": N_FOLDS,
           "test_seeds": list(TEST_SEEDS), "cv": cv, "selected": selected,
           "test_per_seed_mean": test, "learned_metalsq": learned, "timing": timing,
           "wall_s": time.time() - t0}
    path = os.path.join(RESULTS, f"real_calibration_v2_{target}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")


def merge():
    out = {}
    for t in TARGETS:
        out[t] = json.load(open(os.path.join(RESULTS, f"real_calibration_v2_{t}.json")))
    path = os.path.join(RESULTS, "real_calibration_v2.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")
    for t, r in out.items():
        print(f"\n{t}: selected {r['selected']}")
        for k, v in r["test_per_seed_mean"].items():
            print(f"   {k:40s} {np.mean(v):.4g} ± {np.std(v, ddof=1):.2g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run_target(a.target)
