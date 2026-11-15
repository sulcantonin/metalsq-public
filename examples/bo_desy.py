"""DESY's Bayesian optimisation for beam tuning, as in rl-vs-bo's simulation runs.

Follows evaluation/eval_bo_sim.py and bayesopt.py of github.com/desy-ml/rl-vs-bo
(commit fcd68a54): actions in the RescaleAction(-3, 3) space (knob units times 3);
the start setting plus random settings drawn uniformly in a window of +-0.3 around the
start until there are five; then at every step a SingleTaskGP with standardised
outputs is fitted by marginal likelihood, Expected Improvement (best_f = best reward
so far) is maximised with optimize_acqf (q=1, 10 restarts, 256 raw samples, maxiter
200) inside the window +-0.3 around the last setting, clipped to [-3, 3]. The reward is
-log(sum |beam - target|) (the on-screen bonus is a constant for an infinite screen).

The kernel is fixed to the Matern-5/2 kernel with the gamma priors that BoTorch's
SingleTaskGP used by default when rl-vs-bo was written (the paper states Matern-5/2;
newer BoTorch versions changed the default), so the result does not depend on the
installed BoTorch version.
"""
import torch

STEP = 0.3          # 0.1 x the half-width of the rescaled action space [-3, 3]


def _gp(X, Y):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior
    covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1], lengthscale_prior=GammaPrior(3.0, 6.0)),
                        outputscale_prior=GammaPrior(2.0, 0.15))
    return SingleTaskGP(X, Y.unsqueeze(-1), covar_module=covar, outcome_transform=Standardize(m=1))


def run_bo_desy(tu, task, gen, budget=150, times=None):
    """Returns knob settings K (n, d) and the loss -reward per measurement (lower is better)."""
    from botorch.acquisition import ExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    bl = tu.bl
    target = torch.as_tensor(task["target"], dtype=torch.float64)

    def reward(x):
        with torch.no_grad():
            b = bl.beam_on_screen(task, (x / 3).unsqueeze(0))[0]
        return -torch.log((b - target).abs().sum())

    X = [3 * bl.START.clone()]
    Y = [reward(X[0])]
    while len(X) < 5:
        x = (X[0] + STEP * (2 * torch.rand(bl.DIM, generator=gen) - 1)).clamp(-3, 3)
        X.append(x); Y.append(reward(x))
    torch.manual_seed(int(torch.randint(2 ** 31, (1,), generator=gen)))   # optimize_acqf's own draws
    import time
    while len(X) < budget:
        t_start = time.perf_counter()
        Xt, Yt = torch.stack(X), torch.stack(Y)
        gp = _gp(Xt, Yt)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        last = Xt[-1]
        bounds = torch.stack([(last - STEP).clamp(-3, 3), (last + STEP).clamp(-3, 3)])
        cand, _ = optimize_acqf(ExpectedImprovement(gp, best_f=Yt.max()), bounds, q=1,
                                num_restarts=10, raw_samples=256, options={"maxiter": 200})
        x = cand[0].detach()
        if times is not None:
            times.append(time.perf_counter() - t_start)
        X.append(x); Y.append(reward(x))
    return torch.stack(X) / 3, -torch.stack(Y).detach()
