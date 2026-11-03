"""Compute accounting: measured timing breakdown of one Meta-LSQ-Sobolev outer step.

Verifies the paper's ~1.5x backward/forward claim and breaks down where the time
goes inside the inner closed-form solve: (a) feature assembly (sin/cos of W^T x + b
for value+gradient rows), (b) forming A^T A, (c) torch.linalg.solve, and (d) the
backward pass (loss.backward) vs the forward pass. For context, also times one
MLP-MAML standard outer step (forward + inner-grad + outer-backward).

Single-thread, float32, torch + time.perf_counter with warmup. 3 seeds.
"""

import sys, os, math, time, json
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import torch

from experiments.maml_sinusoid import (
    sample_tasks, make_basis,
    fastlsq_inner_solve, fastlsq_predict,
    MAMLNet, maml_inner_step,
)

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

# ---- config (matches prompt: K=10, N=200, batch 4) ----
K       = 10
K_q     = 10
N       = 200
META_BS = 4
SEEDS   = [0, 1, 2]
WARMUP  = 50
REPEATS = 400          # repeats per measured quantity, averaged
GRAD_W  = 10.0         # learned w ~ 10 per repo notes


def median_us(times):
    """times: list of per-repeat seconds. Return (mean_us, median_us)."""
    t = torch.tensor(times, dtype=torch.float64)
    return float(t.mean() * 1e6), float(t.median() * 1e6)


def time_block(fn, repeats, warmup):
    """Time a zero-arg callable. Returns list of per-call seconds (len=repeats)."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


# ============================================================================
#  Meta-LSQ-Sobolev: instrumented inner solve, broken into sub-stages.
#  Mirrors fastlsq_inner_solve exactly so stage sums reconcile with the whole.
# ============================================================================

def assemble_features(xs, ys, dys, W, bb, sqrt_w):
    """(a) feature assembly: sin/cos of W^T x + b for value+gradient rows."""
    Z = xs @ W + bb                                  # (K, N)
    sN = math.sqrt(W.shape[1])
    H = torch.sin(Z) / sN
    G = (W * torch.cos(Z)) / sN
    A = torch.cat([H, sqrt_w * G], dim=0)            # (2K, N)
    t = torch.cat([ys, sqrt_w * dys], dim=0)         # (2K,)
    return A, t


def form_AtA(A, t, mu, N):
    """(b) forming A^T A (and A^T b)."""
    AtA = A.T @ A + mu * torch.eye(N)
    Atb = A.T @ t
    return AtA, Atb


def solve_system(AtA, Atb):
    """(c) the factorization/solve."""
    return torch.linalg.solve(AtA, Atb)


def run_seed(seed):
    gen = torch.Generator().manual_seed(seed + 1000)
    Wb, bb = make_basis(N=N, seed=seed)

    # meta-params (leaf tensors, requires_grad for the full fwd/bwd timing)
    log_sigma = torch.tensor(0.0, requires_grad=True)
    log_mu    = torch.tensor(math.log(1e-4), requires_grad=True)
    log_gw    = torch.tensor(math.log(GRAD_W), requires_grad=True)

    # one fixed meta-batch of tasks for this seed
    xs_b, ys_b, dys_b, xq_b, yq_b = sample_tasks(META_BS, K, K_q, gen, noise=0.0)

    # ------- pre-derived constants for the sub-stage timers (no grad) -------
    with torch.no_grad():
        sigma0 = log_sigma.exp(); mu0 = log_mu.exp(); gw0 = log_gw.exp()
        W0 = sigma0 * Wb
        sqrt_w0 = torch.sqrt(gw0)
        # use task 0 for sub-stage timing
        xs0, ys0, dys0 = xs_b[0], ys_b[0], dys_b[0]
        A0, t0 = assemble_features(xs0, ys0, dys0, W0, bb, sqrt_w0)
        AtA0, Atb0 = form_AtA(A0, t0, mu0, N)

    res = {}

    # (a) feature assembly
    res['a_assembly'] = median_us(time_block(
        lambda: assemble_features(xs0, ys0, dys0, W0, bb, sqrt_w0),
        REPEATS, WARMUP))

    # (b) forming A^T A
    res['b_form_AtA'] = median_us(time_block(
        lambda: form_AtA(A0, t0, mu0, N), REPEATS, WARMUP))

    # (c) solve
    res['c_solve'] = median_us(time_block(
        lambda: solve_system(AtA0, Atb0), REPEATS, WARMUP))

    # ---- whole inner solve (one task), no grad: matches lib fastlsq_inner_solve ----
    res['inner_solve_nograd'] = median_us(time_block(
        lambda: fastlsq_inner_solve(xs0, ys0, dys0, Wb, bb, sigma0, mu0,
                                    sobolev=True, grad_w=gw0),
        REPEATS, WARMUP))

    # ====================================================================
    #  FORWARD vs BACKWARD for one full outer step (meta-batch of 4 tasks)
    # ====================================================================
    def forward_only():
        """Build the meta-loss graph for the whole batch (the forward pass)."""
        sigma = log_sigma.exp(); mu = log_mu.exp(); gw = log_gw.exp()
        meta_loss = 0.0
        for tk in range(META_BS):
            W, beta = fastlsq_inner_solve(
                xs_b[tk], ys_b[tk], dys_b[tk], Wb, bb, sigma, mu,
                sobolev=True, grad_w=gw)
            qpred = fastlsq_predict(xq_b[tk], W, bb, beta)
            meta_loss = meta_loss + (qpred - yq_b[tk]).pow(2).mean()
        return meta_loss / META_BS

    # time forward (graph construction). Detach result so no graph retained between reps.
    fwd_times = []
    for _ in range(WARMUP):
        forward_only()
    for _ in range(REPEATS):
        if log_sigma.grad is not None:
            log_sigma.grad = None; log_mu.grad = None; log_gw.grad = None
        t0 = time.perf_counter()
        loss = forward_only()
        fwd_times.append(time.perf_counter() - t0)
        del loss
    res['forward'] = median_us(fwd_times)

    # time backward only (rebuild graph each rep outside the timer, time .backward())
    bwd_times = []
    for _ in range(WARMUP):
        l = forward_only(); l.backward()
    for _ in range(REPEATS):
        log_sigma.grad = None; log_mu.grad = None; log_gw.grad = None
        loss = forward_only()             # not timed
        t0 = time.perf_counter()
        loss.backward()                   # timed
        bwd_times.append(time.perf_counter() - t0)
    res['backward'] = median_us(bwd_times)

    # full step = forward (graph) + backward, timed together for a sanity total
    full_times = []
    for _ in range(WARMUP):
        l = forward_only(); l.backward()
    for _ in range(REPEATS):
        log_sigma.grad = None; log_mu.grad = None; log_gw.grad = None
        t0 = time.perf_counter()
        loss = forward_only()
        loss.backward()
        full_times.append(time.perf_counter() - t0)
    res['full_step'] = median_us(full_times)

    # ====================================================================
    #  MLP-MAML standard outer step for context
    #  forward (inner SGD step with create_graph) + outer backward
    # ====================================================================
    torch.manual_seed(seed)
    net = MAMLNet(hidden=40, depth=3).to(torch.device("cpu"))
    inner_lr = 0.01

    def mlp_forward():
        meta_loss = 0.0
        for tk in range(META_BS):
            params = net.init_params()
            params = maml_inner_step(net, params, xs_b[tk], ys_b[tk], inner_lr,
                                     first_order=False)
            qpred = net(xq_b[tk], params)
            meta_loss = meta_loss + (qpred - yq_b[tk]).pow(2).mean()
        return meta_loss / META_BS

    def zero_net_grads():
        for p in net.parameters():
            p.grad = None

    # forward
    mlp_fwd = []
    for _ in range(WARMUP):
        mlp_forward()
    for _ in range(REPEATS):
        zero_net_grads()
        t0 = time.perf_counter()
        loss = mlp_forward()
        mlp_fwd.append(time.perf_counter() - t0)
        del loss
    res['mlp_forward'] = median_us(mlp_fwd)

    # backward
    mlp_bwd = []
    for _ in range(WARMUP):
        l = mlp_forward(); l.backward()
    for _ in range(REPEATS):
        zero_net_grads()
        loss = mlp_forward()
        t0 = time.perf_counter()
        loss.backward()
        mlp_bwd.append(time.perf_counter() - t0)
    res['mlp_backward'] = median_us(mlp_bwd)

    # full
    mlp_full = []
    for _ in range(WARMUP):
        l = mlp_forward(); l.backward()
    for _ in range(REPEATS):
        zero_net_grads()
        t0 = time.perf_counter()
        loss = mlp_forward(); loss.backward()
        mlp_full.append(time.perf_counter() - t0)
    res['mlp_full_step'] = median_us(mlp_full)

    return res


def aggregate(per_seed, key, idx):
    """idx: 0=mean_us field, 1=median_us field. Aggregate the 'mean_us' across seeds."""
    vals = torch.tensor([per_seed[s][key][idx] for s in range(len(per_seed))],
                        dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


def main():
    all_seeds = [run_seed(s) for s in SEEDS]

    # use mean_us (index 0) as the headline number, report std across seeds
    keys = ['a_assembly', 'b_form_AtA', 'c_solve', 'inner_solve_nograd',
            'forward', 'backward', 'full_step',
            'mlp_forward', 'mlp_backward', 'mlp_full_step']
    agg = {}
    for k in keys:
        m, s = aggregate(all_seeds, k, 0)        # mean across repeats, then across seeds
        med, _ = aggregate(all_seeds, k, 1)
        agg[k] = {'mean_us': m, 'std_across_seeds_us': s, 'median_us': med}

    # ---- derived quantities ----
    fwd = agg['forward']['mean_us']
    bwd = agg['backward']['mean_us']
    bf_ratio = bwd / fwd

    # sub-stage fractions of the (no-grad) inner solve (per single task)
    a = agg['a_assembly']['mean_us']
    b = agg['b_form_AtA']['mean_us']
    c = agg['c_solve']['mean_us']
    inner = agg['inner_solve_nograd']['mean_us']
    stage_sum = a + b + c
    frac = {
        'a_assembly':  a / inner,
        'b_form_AtA':  b / inner,
        'c_solve':     c / inner,
        'overhead':    (inner - stage_sum) / inner,   # cat/exp/sqrt glue not in a/b/c
    }

    mlp_fwd = agg['mlp_forward']['mean_us']
    mlp_bwd = agg['mlp_backward']['mean_us']
    mlp_bf = mlp_bwd / mlp_fwd

    summary = {
        'config': {'K': K, 'N': N, 'meta_bs': META_BS, 'K_q': K_q,
                   'grad_w': GRAD_W, 'seeds': SEEDS,
                   'repeats': REPEATS, 'warmup': WARMUP, 'threads': 1,
                   'dtype': 'float32'},
        'timings_us': agg,
        'backward_over_forward': bf_ratio,
        'inner_solve_fractions': frac,
        'inner_stage_sum_us': stage_sum,
        'mlp_backward_over_forward': mlp_bf,
    }

    os.makedirs(_os.path.join(_REPO_ROOT, 'results'), exist_ok=True)
    outpath = _os.path.join(_REPO_ROOT, 'results', 'compute_accounting_breakdown.json')
    with open(outpath, 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- console table ----
    print(f"\n=== Meta-LSQ-Sobolev compute accounting (K={K}, N={N}, batch={META_BS}, "
          f"grad_w={GRAD_W}, single-thread, {REPEATS} reps x 3 seeds) ===\n")
    print("Inner closed-form solve sub-stages (per task, no-grad):")
    print(f"  {'stage':<28}{'mean us':>10}{'fraction':>12}")
    print(f"  {'(a) feature assembly':<28}{a:>10.2f}{frac['a_assembly']*100:>11.1f}%")
    print(f"  {'(b) form A^T A':<28}{b:>10.2f}{frac['b_form_AtA']*100:>11.1f}%")
    print(f"  {'(c) linalg.solve':<28}{c:>10.2f}{frac['c_solve']*100:>11.1f}%")
    print(f"  {'    glue/overhead':<28}{inner-stage_sum:>10.2f}{frac['overhead']*100:>11.1f}%")
    print(f"  {'TOTAL inner solve':<28}{inner:>10.2f}{100.0:>11.1f}%")
    print()
    print("One full outer step (batch of 4 tasks):")
    print(f"  {'forward (graph build)':<28}{fwd:>10.2f}")
    print(f"  {'backward (loss.backward)':<28}{bwd:>10.2f}")
    print(f"  {'full step (fwd+bwd)':<28}{agg['full_step']['mean_us']:>10.2f}")
    print(f"  backward / forward ratio = {bf_ratio:.2f}x   (paper claims ~1.5x)")
    print()
    print("MLP-MAML standard outer step (context):")
    print(f"  {'forward (inner-grad+graph)':<28}{mlp_fwd:>10.2f}")
    print(f"  {'backward':<28}{mlp_bwd:>10.2f}")
    print(f"  {'full step':<28}{agg['mlp_full_step']['mean_us']:>10.2f}")
    print(f"  backward / forward ratio = {mlp_bf:.2f}x")
    print(f"\n[saved] {outpath}")

    return summary


if __name__ == "__main__":
    main()
