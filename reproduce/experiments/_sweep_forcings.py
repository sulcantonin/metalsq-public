"""Sweep: how many forcings, and how strong a prior, does kernel recovery need?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.kernel_identification as K

K.set_rank(12)
print(f"{'forcings':>9} {'noise':>7} {'tik':>8} {'||K_hat-K||/||K||':>18} {'time s':>8}",
      flush=True)
for nf in (1, 4, 8):
    for nz in (0.01, 0.05):
        for tik in (1e-3, 1e-2, 1e-1, 1.0):
            r = [K.multi_forcing_problem(sd, nz, nf, tik=tik, iters=250) for sd in (0, 1)]
            e = sum(x["kernel_l2"] for x in r) / len(r)
            print(f"{nf:>9} {nz:>7.1%} {tik:>8.0e} {e:>18.4f} {r[-1]['time_s']:>8.1f}",
                  flush=True)
