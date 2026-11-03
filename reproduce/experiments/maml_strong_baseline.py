"""Strong MLP-MAML baseline on sinusoid (4x64 / 5 inner steps / FOMAML, long budget).

Addresses the 'strawman risk' (paper Section 7): the canonical MAML sinusoid
result of test MSE ~0.3 requires a larger network, multi-step inner adaptation,
and 60k+ outer iterations. We use FOMAML to make the wall-clock manageable.

Output goes into the revised Table 2 in the paper.
"""

import os
import sys
import time
import torch

# Allow `python experiments/maml_strong_baseline.py` to find the `experiments`
# package without setting PYTHONPATH (repo root = parent of this file's dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.maml_sinusoid import train_mlp_maml, eval_mlp_maml

torch.set_default_dtype(torch.float32)
SEED = 0


def run(n_iters=60000, hidden=64, depth=4, inner_steps=5,
        inner_lr=0.005, outer_lr=1e-3, first_order=True,
        n_test=200, K=10, K_q=50, log_every=5000):
    print(f"Strong MLP-MAML: depth={depth} hidden={hidden} "
          f"inner_steps={inner_steps} first_order={first_order} n_iters={n_iters}")
    t0 = time.time()
    net, train_time, curve = train_mlp_maml(
        n_iters=n_iters,
        meta_bs=4,
        K=K,
        K_q=K_q,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        outer_lr=outer_lr,
        hidden=hidden,
        depth=depth,
        first_order=first_order,
        seed=SEED,
        log_every=log_every,
        clip_grad=10.0,   # documented for the strong baseline (paper §6 setup)
    )
    print(f"Training wall-clock: {train_time:.1f}s")
    mse, inf = eval_mlp_maml(net, n_test=n_test, K=K, K_q=K_q,
                              inner_lr=inner_lr, inner_steps=inner_steps, seed=SEED)
    print(f"Test MSE: {mse:.4f}    Inference: {inf*1000:.3f} ms/task")

    # Param count
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Param count: {n_params}")

    print("\nConvergence curve (wallclock_s, iter, meta_loss):")
    idx = [int(i * (len(curve)-1) / 7) for i in range(8)]
    for i in idx:
        t, it, l = curve[i]
        print(f"  t={t:7.1f}s  iter={it:6d}  loss={l:.4f}")

    return {
        "train_s": train_time,
        "test_mse": mse,
        "inf_ms": inf * 1000,
        "n_params": n_params,
        "curve": curve,
    }


if __name__ == "__main__":
    import sys
    # One stronger MLP-MAML configuration. 60k iterations is the literature-standard
    # number, but is expensive on CPU. The 20k FOMAML configuration is a useful
    # intermediate point and finishes in a few minutes.
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    n_iters = int(pos[0]) if pos else 20000
    le = 25 if save else max(1, n_iters // 8)   # finer curve when capturing for figures
    print(f"\n========== Strong baseline ({n_iters} iters) ==========")
    r = run(n_iters=n_iters, hidden=64, depth=4, inner_steps=5,
            first_order=True, log_every=le)

    print("\n=== Summary ===")
    print(f"{'config':<25} {'train s':>10} {'test MSE':>10} {'inf ms':>8} {'params':>8}")
    print(f"{'strong FOMAML':<25} {r['train_s']:>10.1f} {r['test_mse']:>10.4f} {r['inf_ms']:>8.3f} {r['n_params']:>8d}")

    if save:
        import json
        import numpy as np
        np.savez(save + ".npz", curve=np.array(r["curve"], dtype=float))
        with open(save + ".json", "w") as f:
            json.dump({"train_s": r["train_s"], "test_mse": r["test_mse"],
                       "inf_ms": r["inf_ms"], "n_params": r["n_params"],
                       "n_iters": n_iters}, f, indent=2)
        print(f"[saved] {save}.npz and {save}.json")
