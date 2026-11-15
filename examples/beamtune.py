"""Shared harness for the beam-tuning examples (ares.py, lcls_diag0.py).

A beamline is any object (in practice, a module) with
    DIM                 number of knobs, all scaled to [-1, 1]
    START, HALF         fixed starting setting and half-width of the moving window
    sample_task(gen)    one task: a dict with at least "target" = [mu_x, sigma_x, mu_y,
                        sigma_y] in metres
    beam_on_screen(task, K) -> (n, 4) beam parameters on the screen, differentiable
                        in the knob settings K (n, DIM)

Part 1 predicts a task's objective from random measurements. Part 2 is DESY's
Bayesian-optimisation protocol (rl-vs-bo): objective log(||beam - target||), fixed
start, five samples in a window around it, then one proposal per step inside the
window around the last setting. Every method follows it, so with values only the
surrogate is the only difference between GP expected improvement and Meta-LSQ.

Tried and not adopted: a per-knob bandwidth (MetaLSQ(bandwidth="diagonal")) and
polynomial columns (poly_degree=1 or 2) for the multi-output fits. On two independent
sets of our own ARES development tasks neither gave a consistent gain: with gradients
the shared bandwidth stayed best, and from values the best combination (per-knob
bandwidth plus linear columns) improved the median slightly on one set and worsened
the mean on the other. The learned per-knob bandwidths are physically sensible
(~2-3 for Q1 and Q3, ~0.3 for Q2, ~0.03 for both steerers, whose effect is nearly
linear), but the shared-bandwidth fit already copes. They were not run on DESY's trials.
"""
import time

import numpy as np
import torch

from metalsq import MetaLSQ, Task

N_FEATURES = 500


class Tuner:
    def __init__(self, beamline):
        self.bl = beamline

    # ------------------------------------------------------------ measurements
    def uniform(self, n, gen):
        return torch.rand(n, self.bl.DIM, generator=gen) * 2 - 1

    def target(self, task):
        return torch.tensor(task["target"])

    def measure(self, task, K, grad=True):
        """Squared distance to the target beam [mm^2] and, if asked, its gradient."""
        K = K.detach().clone().requires_grad_(grad)
        y = ((self.bl.beam_on_screen(task, K) - self.target(task)) ** 2).sum(-1) * 1e6
        if not grad:
            return y.detach(), None
        g, = torch.autograd.grad(y.sum(), K)
        return y.detach(), g

    def log_distance(self, task, K, grad=True):
        """DESY's BO objective: log of the distance to the target beam (in mm)."""
        K = K.detach().clone().requires_grad_(grad)
        d = (self.bl.beam_on_screen(task, K) - self.target(task)).norm(dim=-1) * 1e3
        y = torch.log(d)
        if not grad:
            return y.detach(), None
        g, = torch.autograd.grad(y.sum(), K)
        return y.detach(), g

    def mae_um(self, task, k):
        """Mean absolute error of the four beam parameters, in um (DESY's metric)."""
        with torch.no_grad():
            b = self.bl.beam_on_screen(task, k.unsqueeze(0))[0]
        return float((b - self.target(task)).abs().mean()) * 1e6

    # ------------------------------------------------------------ meta-training data
    def build_pool(self, n_tasks, n_points, seed):
        """Measurements at random settings over the whole range (squared distance)."""
        gen = torch.Generator().manual_seed(seed)
        pool = []
        for _ in range(n_tasks):
            task = self.bl.sample_task(gen)
            K = self.uniform(n_points, gen)
            pool.append((K, *self.measure(task, K)))
        return pool

    def build_local_pool(self, n_tasks, n_points, seed, half=0.3):
        """Measurements in a neighbourhood of a random setting (log distance), the
        regime the moving window works in."""
        gen = torch.Generator().manual_seed(seed)
        pool = []
        for _ in range(n_tasks):
            task = self.bl.sample_task(gen)
            K = (self.uniform(1, gen)
                 + half * (torch.rand(n_points, self.bl.DIM, generator=gen) * 2 - 1)).clamp(-1, 1)
            pool.append((K, *self.log_distance(task, K)))
        return pool

    @staticmethod
    def pool_sampler(pool, K_support):
        def sample(n_tasks, gen):
            out = []
            for i in torch.randint(len(pool), (n_tasks,), generator=gen).tolist():
                K, y, g = pool[i]
                p = torch.randperm(len(y), generator=gen)
                s, q = p[:K_support], p[K_support:]
                out.append(Task(x=K[s], y=y[s], dy=g[s], xq=K[q], yq=y[q]))
            return out
        return sample

    def train(self, pool, sobolev, K_support, sigma, label, **model_kwargs):
        m = MetaLSQ(in_dim=self.bl.DIM, n_features=N_FEATURES, sobolev=sobolev, sigma=sigma,
                    seed=0, **model_kwargs)
        t0 = time.time()
        m.meta_fit(self.pool_sampler(pool, K_support), steps=1500, batch_size=4, lr=1e-2, seed=1)
        print(f"meta-trained {label} on {K_support}-point tasks in {time.time() - t0:.1f} s -> {m}")
        return m

    # ------------------------------------------------------------ part 1
    def part1(self, models, tasks, gen, n=30):
        print(f"\nPart 1: predicting a new task's objective from {n} measurements")
        X = self.uniform(4000, gen)
        errs = {name: [] for name in models}
        for task in tasks:
            K = self.uniform(n, gen)
            y, g = self.measure(task, K)
            yt, _ = self.measure(task, X, grad=False)
            for name, (model, use_grad) in models.items():
                with torch.no_grad():
                    f = model.adapt(K, y, g if use_grad else None)
                    errs[name].append(float(((f(X) - yt).pow(2).mean() / yt.var()).sqrt()))
        for name, e in errs.items():
            print(f"  {name:<34s} relative RMS error: median {np.median(e):.3f}, mean {np.mean(e):.3f}")

    # ------------------------------------------------------------ part 2
    def window(self, center, n, gen):
        return (center + self.bl.HALF
                * (torch.rand(n, self.bl.DIM, generator=gen) * 2 - 1)).clamp(-1, 1)

    @staticmethod
    def posterior_sd(model, K, X):
        """Posterior std of a value-only fit at X, up to the noise scale, max-normalised."""
        H, h = model.features(K), model.features(X)
        P = H @ H.T + model.mu * torch.eye(len(K))
        q = (h * h).sum(1) - ((h @ H.T) * torch.linalg.solve(P, H @ h.T).T).sum(1)
        sd = q.clamp_min(0).sqrt()
        return sd / sd.max()

    @staticmethod
    def descend(value, gradient, k, n_steps=40, step=0.1):
        """Projected descent with normalised gradient steps, halving the step whenever it
        does not lower the value (beamline gradients can reach ~10^3)."""
        v = value(k)
        for _ in range(n_steps):
            g = gradient(k)
            k_new = (k - step * g / (g.norm() + 1e-12)).clamp(-1, 1)
            v_new = value(k_new)
            if v_new < v:
                k, v = k_new, v_new
            else:
                step /= 2
        return k

    def run_protocol(self, task, budget, gen, propose, use_grad=False):
        """Start setting plus four random samples in the window around it, then one
        proposal per step in the window around the last setting."""
        K = torch.cat([self.bl.START.unsqueeze(0), self.window(self.bl.START, 4, gen)])
        y, G = self.log_distance(task, K, grad=use_grad)
        t_fit = []
        while len(y) < budget:
            t0 = time.perf_counter()
            k = propose(K, y, G, K[-1], gen)
            t_fit.append(time.perf_counter() - t0)
            yk, gk = self.log_distance(task, k.unsqueeze(0), grad=use_grad)
            K, y = torch.cat([K, k.unsqueeze(0)]), torch.cat([y, yk])
            if use_grad:
                G = torch.cat([G, gk])
        return K, y, t_fit

    def propose_metalsq(self, model, use_grad, kappa=0.1):
        half = self.bl.HALF

        def propose(K, y, G, last, gen):
            with torch.no_grad():
                f = model.adapt(K, y, G if use_grad else None)
                C = self.window(last, 2000, gen)
                acq = f(C)
                if not use_grad:                   # lower confidence bound
                    acq = acq - kappa * (y.max() - y.min()) * self.posterior_sd(model, K, C)
                k = C[acq.argmin()].clone()
                if use_grad:                       # refine with the fit's gradient
                    k = self.descend(lambda z: float(f(z.unsqueeze(0))[0]),
                                     lambda z: f.gradient(z.unsqueeze(0))[0], k, step=half / 2)
                    k = torch.max(torch.min(k, last + half), last - half).clamp(-1, 1)
            return k
        return propose

    def propose_gp(self, K, y, G, last, gen):
        """GP with expected improvement in the window, as in DESY's BO."""
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        gp = GaussianProcessRegressor(
            ConstantKernel() * RBF([0.5] * self.bl.DIM, (1e-3, 1e2))
            + WhiteKernel(1e-4, (1e-8, 1e-1)),
            normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(K.numpy(), y.numpy())
        C = self.window(last, 2000, gen)
        mu, sd = gp.predict(C.numpy(), return_std=True)
        best = float(y.min())
        z = (best - mu) / np.maximum(sd, 1e-12)
        ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        return C[int(np.argmax(ei))]

    def propose_random(self, K, y, G, last, gen):
        return self.window(last, 1, gen)[0]

    def run_gradient_descent(self, task, budget, gen):
        """Uses gradients too: after the same five starting samples, normalised-gradient
        descent on the machine, one measurement per step, halving the step after a
        measurement that got worse."""
        step = self.bl.HALF
        K, y, _ = self.run_protocol(task, 5, gen, None)
        k_best, v_best = K[y.argmin()].clone(), float(y.min())
        _, g = self.log_distance(task, k_best.unsqueeze(0))
        Ks, ys = [K], [y]
        while sum(len(v) for v in ys) < budget:
            k = (k_best - step * g[0] / (g[0].norm() + 1e-12)).clamp(-1, 1)
            yk, gk = self.log_distance(task, k.unsqueeze(0))
            Ks.append(k.unsqueeze(0)); ys.append(yk)
            if float(yk) < v_best:
                v_best, k_best, g = float(yk), k.clone(), gk
            else:
                step /= 2
        return torch.cat(Ks), torch.cat(ys), []

    def best_reachable_mae(self, task, gen, n_starts=64, n_steps=300, step=0.1):
        """Reference: lowest MAE found by multi-start descent on the simulator itself,
        with unlimited measurements; all starts move together in one batched call."""
        k = self.uniform(n_starts, gen)
        v, g = self.measure(task, k)
        steps = torch.full((n_starts, 1), step)
        for _ in range(n_steps):
            k_new = (k - steps * g / (g.norm(dim=1, keepdim=True) + 1e-12)).clamp(-1, 1)
            v_new, g_new = self.measure(task, k_new)
            better = (v_new < v).unsqueeze(1)
            k = torch.where(better, k_new, k)
            g = torch.where(better, g_new, g)
            v = torch.where(better[:, 0], v_new, v)
            steps = torch.where(better, steps, steps / 2)
        return self.mae_um(task, k[v.argmin()])

    def part2(self, value_model, grad_model, tasks, gen, budget=100):
        print(f"\nPart 2: closed-loop tuning with DESY's protocol ({budget} measurements); "
              f"mean absolute error of (mu_x, sigma_x, mu_y, sigma_y) vs the target, um")
        run = self.run_protocol
        methods = {
            "Meta-LSQ, values only":
                lambda t, g: run(t, budget, g, self.propose_metalsq(value_model, False)),
            "GP Bayesian optimisation (EI), values":
                lambda t, g: run(t, budget, g, self.propose_gp),
            "random in the window":
                lambda t, g: run(t, budget, g, self.propose_random),
            "gradient descent, with gradients":
                lambda t, g: self.run_gradient_descent(t, budget, g),
            "Meta-LSQ, with gradients":
                lambda t, g: run(t, budget, g, self.propose_metalsq(grad_model, True), use_grad=True)}
        mae = {m: [] for m in methods}; times = {m: [] for m in methods}; reach = []
        for task in tasks:
            reach.append(self.best_reachable_mae(task, gen))
            seed = int(torch.randint(2 ** 31, (1,), generator=gen))
            for name, run_method in methods.items():
                K, y, t = run_method(task, torch.Generator().manual_seed(seed))
                best_so_far = torch.cummin(y, 0).indices
                mae[name].append([self.mae_um(task, K[best_so_far[n - 1]]) for n in (20, 50, 100)])
                times[name] += t
        for name in methods:
            r = np.asarray(mae[name])
            cols = "  ".join(f"after {n:3d}: {np.median(r[:, i]):6.1f}"
                             for i, n in enumerate((20, 50, 100)))
            ft = f"{1e3 * np.mean(times[name]):6.1f} ms/step" if times[name] else ""
            print(f"  {name:<40s} median MAE {cols}   {ft}")
        print(f"  {'best reachable (unlimited, on the simulator)':<40s} "
              f"median MAE {np.median(reach):6.1f}")
        return mae, reach

    # ------------------------------------------------------------ multi-output variant
    # Instead of the scalar distance, every measurement gives the four beam parameters
    # (and, with gradients, their Jacobian). Each parameter gets its own meta-learned
    # Meta-LSQ fit; the loop minimises the distance computed from the four fits.
    def measure_beam(self, task, K, grad=True):
        """Beam parameters on the screen [mm], (n, 4), and their Jacobians (n, 4, DIM)."""
        K = K.detach().clone().requires_grad_(grad)
        B = self.bl.beam_on_screen(task, K) * 1e3
        if not grad:
            return B.detach(), None
        J = torch.stack([torch.autograd.grad(B[:, i].sum(), K, retain_graph=i < 3)[0]
                         for i in range(4)], 1)
        return B.detach(), J

    def build_local_beam_pool(self, n_tasks, n_points, seed, half=0.3):
        gen = torch.Generator().manual_seed(seed)
        pool = []
        for _ in range(n_tasks):
            task = self.bl.sample_task(gen)
            K = (self.uniform(1, gen)
                 + half * (torch.rand(n_points, self.bl.DIM, generator=gen) * 2 - 1)).clamp(-1, 1)
            pool.append((K, *self.measure_beam(task, K)))
        return pool

    def train_multi(self, pool, sobolev, K_support, sigma, label, **model_kwargs):
        """One Meta-LSQ model per beam parameter, meta-learned on that parameter."""
        models = []
        for i in range(4):
            output_pool = [(K, B[:, i], J[:, i]) for K, B, J in pool]
            models.append(self.train(output_pool, sobolev, K_support, sigma,
                                     f"{label}, beam parameter {i}", **model_kwargs))
        return models

    def log_distance_from_beam(self, B, task):
        return torch.log((B - self.target(task) * 1e3).norm(dim=-1))

    def run_protocol_multi(self, task, budget, gen, propose, use_grad=False):
        """DESY's protocol, but recording the four beam parameters at every measurement."""
        K = torch.cat([self.bl.START.unsqueeze(0), self.window(self.bl.START, 4, gen)])
        B, J = self.measure_beam(task, K, grad=use_grad)
        t_fit = []
        while len(B) < budget:
            y = self.log_distance_from_beam(B, task)
            t0 = time.perf_counter()
            k = propose(K, B, J, y, K[-1], gen, self.target(task) * 1e3)
            t_fit.append(time.perf_counter() - t0)
            bk, jk = self.measure_beam(task, k.unsqueeze(0), grad=use_grad)
            K, B = torch.cat([K, k.unsqueeze(0)]), torch.cat([B, bk])
            if use_grad:
                J = torch.cat([J, jk])
        return K, self.log_distance_from_beam(B, task), t_fit

    def propose_multi(self, models, use_grad, kappa=0.1, eps=1e-3):
        half = self.bl.HALF

        def propose(K, B, J, y, last, gen, target):
            with torch.no_grad():
                fits = [m.adapt(K, B[:, i], J[:, i] if use_grad else None)
                        for i, m in enumerate(models)]
                C = self.window(last, 2000, gen)
                pred = torch.stack([f(C) for f in fits], -1)
                acq = torch.log((pred - target).norm(dim=-1) + eps)
                if not use_grad:          # lower confidence bound, averaged uncertainty
                    sd = torch.stack([self.posterior_sd(m, K, C) for m in models]).mean(0)
                    acq = acq - kappa * (y.max() - y.min()) * sd
                k = C[acq.argmin()].clone()
                if use_grad:              # refine on the composed objective
                    def value(z):
                        b = torch.stack([f(z.unsqueeze(0))[0] for f in fits])
                        return float(torch.log((b - target).norm() + eps))

                    def gradient(z):
                        b = torch.stack([f(z.unsqueeze(0))[0] for f in fits])
                        g = torch.stack([f.gradient(z.unsqueeze(0))[0] for f in fits])
                        r = b - target
                        return (r.unsqueeze(1) * g).sum(0) / (r.norm() ** 2 + eps)
                    k = self.descend(value, gradient, k, step=half / 2)
                    k = torch.max(torch.min(k, last + half), last - half).clamp(-1, 1)
            return k
        return propose

    def propose_gp_multi(self, kappa=0.1, eps=1e-3):
        """The same structured loop with GP surrogates: one GP (RBF + noise, fitted by
        marginal likelihood) per beam parameter, and the same lower confidence bound on
        the composed log distance. Only the surrogate differs from propose_multi."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        def propose(K, B, J, y, last, gen, target):
            C = self.window(last, 2000, gen)
            means, sds = [], []
            for i in range(4):
                gp = GaussianProcessRegressor(
                    ConstantKernel() * RBF([0.5] * self.bl.DIM, (1e-3, 1e2))
                    + WhiteKernel(1e-4, (1e-8, 1e-1)),
                    normalize_y=True, n_restarts_optimizer=1, random_state=0
                ).fit(K.numpy(), B[:, i].numpy())
                mu, sd = gp.predict(C.numpy(), return_std=True)
                means.append(torch.tensor(mu)); sds.append(torch.tensor(sd / max(sd.max(), 1e-12)))
            pred = torch.stack(means, -1)
            acq = (torch.log((pred - target).norm(dim=-1) + eps)
                   - kappa * (y.max() - y.min()) * torch.stack(sds).mean(0))
            return C[acq.argmin()].clone()
        return propose

    # ------------------------------------------------------------ everything
    def main(self, n_test=20, test_seed=123, sigma_global=0.3, sigma_local=0.5):
        t0 = time.time()
        pool = self.build_pool(n_tasks=800, n_points=70, seed=0)
        print(f"simulated {len(pool)} training tasks in {time.time() - t0:.1f} s")
        grad_model = self.train(pool, True, 30, sigma_global, "with gradients")
        value_model = self.train(pool, False, 30, sigma_global, "values only")
        models = {"Meta-LSQ, with gradients": (grad_model, True),
                  "Meta-LSQ, values only": (value_model, False),
                  "untrained defaults, values only":
                      (MetaLSQ(in_dim=self.bl.DIM, n_features=N_FEATURES, sobolev=False), False)}
        local = self.build_local_pool(n_tasks=800, n_points=70, seed=0)
        loop_grad = self.train(local, True, 30, sigma_local, "for the loop, with gradients, local")
        loop_value = self.train(local, False, 50, sigma_local, "for the loop, values only, local")

        gen = torch.Generator().manual_seed(test_seed)
        tasks = [self.bl.sample_task(gen) for _ in range(n_test)]
        self.part1(models, tasks, gen)
        return self.part2(loop_value, loop_grad, tasks, gen)
