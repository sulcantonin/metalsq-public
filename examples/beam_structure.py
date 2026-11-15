"""Everything a differentiable twin and linear beam optics give a closed-form surrogate.

Observation rows (from the twin, per measurement and beam parameter):
    values          b(K)
    slopes          db/dK                      (d rows)
    curvatures      d2b/dK_a dK_b, a <= b      (d(d+1)/2 rows)

Constraint rows (exact for uncoupled linear optics, checked against Cheetah by
check_constraints): at fixed unlabeled collocation points, D^alpha b = 0 for
    - beam sizes and steerers: sigma_x, sigma_y do not depend on either steerer;
    - the other plane: mu_x does not depend on the vertical steerer, mu_y not on the
      horizontal one;
    - linearity in the own steerer: d2 mu / d(steerer)^2 = 0, and on ARES the slope
      of mu_x in CH (after all quadrupoles) and of mu_y in CV (before Q3 only) do not
      depend on the quadrupoles upstream of the steerer.
Constraint rows need no gradients from the machine, so they also apply to values-only
tuning.

Meta-LSQ gets these rows through its closed-form feature derivatives; ALPaCA-style gets
the same rows through autograd derivatives of its encoder. The weights of the slope,
curvature and constraint rows are meta-learned with each model's other parameters.

Integral rows are not used: nothing in this task observes an integral of a beam
parameter over the magnet settings.
"""
import math

import torch
from torch import nn
from torch.func import hessian, jacrev, vmap

from metalsq import MetaLSQ, Task, baselines

D = torch.float64
OUTPUTS = ("mu_x", "sigma_x", "mu_y", "sigma_y")


def e(d, *idx):
    a = [0] * d
    for i in idx:
        a[i] += 1
    return tuple(a)


# knobs: ARES (Q1, Q2, CV, Q3, CH); DIAG0 (XC, YC, Q6, Q7, Q8, Q9)
def constraints(name):
    if name == "ares":
        d, Q1, Q2, CV, Q3, CH = 5, 0, 1, 2, 3, 4
        return {0: [e(d, CV), e(d, CH, CH), e(d, Q1, CH), e(d, Q2, CH), e(d, Q3, CH)],
                1: [e(d, CV), e(d, CH)],
                2: [e(d, CH), e(d, CV, CV), e(d, Q1, CV), e(d, Q2, CV)],
                3: [e(d, CV), e(d, CH)]}
    if name == "lcls_diag0":
        d, XC, YC = 6, 0, 1
        return {0: [e(d, YC), e(d, XC, XC)], 1: [e(d, XC), e(d, YC)],
                2: [e(d, XC), e(d, YC, YC)], 3: [e(d, XC), e(d, YC)]}
    raise ValueError(name)


def measure(bl, task, K, order=2):
    """Beam parameters in mm (n, 4), slopes (n, 4, d) and curvatures (n, 4, d, d) of the
    twin at K. The settings are independent, so per-setting derivatives come from
    gradients of batch sums."""
    K = K.detach().clone().requires_grad_(order > 0)
    B = bl.beam_on_screen(task, K) * 1e3
    if order == 0:
        return B.detach(), None, None
    n, d = K.shape
    J = torch.stack([torch.autograd.grad(B[:, i].sum(), K, create_graph=order > 1, retain_graph=True)[0]
                     for i in range(4)], 1)
    if order == 1:
        return B.detach(), J.detach(), None
    H = torch.zeros(n, 4, d, d)
    for i in range(4):
        for a in range(d):
            H[:, i, a] = torch.autograd.grad(J[:, i, a].sum(), K, retain_graph=True, allow_unused=True)[0]
    return B.detach(), J.detach(), H.detach()


def check_constraints(bl, name, n_tasks=10, n_points=8, seed=11):
    """Largest |D^alpha b| over random tasks and settings for every constraint row."""
    gen = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(n_tasks):
        task = bl.sample_task(gen)
        K = torch.rand(n_points, bl.DIM, generator=gen) * 2 - 1
        _, J, H = measure(bl, task, K)
        for i, rows in constraints(name).items():
            for a in rows:
                idx = [k for k, o in enumerate(a) for _ in range(o)]
                v = J[:, i, idx[0]] if len(idx) == 1 else H[:, i, idx[0], idx[1]]
                worst = max(worst, float(v.abs().max()))
    return worst


def _pairs(d):
    return [(a, b) for a in range(d) for b in range(a, d)]


def _collocation(d, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, d, generator=g, dtype=D) * 2 - 1


class FullLSQ(MetaLSQ):
    """Meta-LSQ for one beam parameter with slope, curvature and constraint rows."""

    def __init__(self, in_dim, n_features=500, sigma=0.5, use_slopes=True, use_curvature=True,
                 constraint_rows=(), w2=0.1, cw=1.0, **kwargs):
        super().__init__(in_dim=in_dim, n_features=n_features, sobolev=use_slopes, sigma=sigma, **kwargs)
        self.use_curvature, self.constraint_rows = use_curvature, tuple(constraint_rows)
        self.log_w2 = nn.Parameter(torch.tensor(math.log(w2), dtype=D), requires_grad=use_curvature)
        self.log_cw = nn.Parameter(torch.tensor(math.log(cw), dtype=D),
                                   requires_grad=bool(self.constraint_rows))
        self.xc = _collocation(in_dim)

    def adapt(self, x, y, dy=None, H=None, **kwargs):
        x = torch.as_tensor(x, dtype=D)

        def rows(basis, W):
            out = []
            if self.use_curvature and H is not None:
                s = self.log_w2.exp().sqrt()
                for a, b in _pairs(basis.in_dim):
                    out.append((s * basis.derivative(x, W, e(basis.in_dim, a, b)), s * torch.as_tensor(H, dtype=D)[:, a, b]))
            if self.constraint_rows:
                c = self.log_cw.exp()
                zeros = torch.zeros(len(self.xc), dtype=D)
                out += [(c * basis.derivative(self.xc, W, al), zeros) for al in self.constraint_rows]
            return out
        return super().adapt(x, y, dy, rows=rows, **kwargs)

    def query_loss(self, tasks):
        loss = 0.0
        for tk in tasks:
            f = self.adapt(tk.x, tk.y, tk.dy, tk.extra.get("H"))
            loss = loss + (f(tk.xq) - torch.as_tensor(tk.yq, dtype=D)).pow(2).mean()
        return loss / len(tasks)


class _ALPFit:
    def __init__(self, m, beta):
        self.m, self.beta = m, beta

    def __call__(self, X):
        return self.m._phi(torch.as_tensor(X, dtype=D)) @ self.beta

    def gradient(self, X):
        return torch.einsum("mdn,n->md", self.m.phi_gradient(torch.as_tensor(X, dtype=D)), self.beta)


class FullALPaCA(baselines.ALPaCA):
    """ALPaCA-style for one beam parameter with the same slope, curvature and
    constraint rows, built from autograd derivatives of its encoder."""

    def __init__(self, in_dim, use_slopes=True, use_curvature=True, constraint_rows=(),
                 w2=0.1, cw=1.0, hidden=40, p=40):
        super().__init__(in_dim, hidden=hidden, p=p, sobolev=use_slopes, dtype=D)
        self.use_curvature, self.constraint_rows = use_curvature, tuple(constraint_rows)
        self.log_w2 = nn.Parameter(torch.tensor(math.log(w2), dtype=D), requires_grad=use_curvature)
        self.log_cw = nn.Parameter(torch.tensor(math.log(cw), dtype=D),
                                   requires_grad=bool(self.constraint_rows))
        self.xc = _collocation(in_dim)

    def _phi_hessian(self, x):
        """(n, p+1, d, d) second derivatives of the features, in closed form for two tanh
        layers: with u = W2 h + b2, h = tanh(W1 x + b1), phi = tanh(u),
        d2 phi = (1 - phi^2) d2 u - 2 phi (1 - phi^2) du du,
        d2 u_j / dx_a dx_b = sum_k W2_jk (-2 h_k (1 - h_k^2)) W1_ka W1_kb."""
        l1, l2 = self.enc[0], self.enc[2]
        h = torch.tanh(l1(x))                                         # (n, H)
        phi = torch.tanh(l2(h))                                       # (n, p)
        G = torch.einsum("jk,nk,ka->nja", l2.weight, 1 - h ** 2, l1.weight)          # du/dx
        U2 = torch.einsum("jk,nk,ka,kb->njab", l2.weight, -2 * h * (1 - h ** 2), l1.weight, l1.weight)
        s = (1 - phi ** 2)[..., None, None]
        Hphi = s * U2 - 2 * (phi[..., None, None] * s) * G[..., :, None] * G[..., None, :]
        return torch.cat([Hphi, torch.zeros(len(x), 1, x.shape[1], x.shape[1], dtype=D)], 1)

    def _system(self, x, y, dy, H):
        Phi = self._phi(x)
        A, t = [Phi], [y]
        d = x.shape[1]
        if self.sobolev and dy is not None:
            sw = self.log_w.exp().sqrt()
            G = self.phi_gradient(x)
            A.append(sw * G.transpose(0, 1).reshape(len(x) * d, -1))
            t.append(sw * dy.reshape(len(x), d).T.reshape(-1))
        if self.use_curvature and H is not None:
            s = self.log_w2.exp().sqrt()
            Hp = self._phi_hessian(x)
            for a, b in _pairs(d):
                A.append(s * Hp[:, :, a, b]); t.append(s * H[:, a, b])
        if self.constraint_rows:
            c = self.log_cw.exp()
            Jc, Hc = self.phi_gradient(self.xc).transpose(1, 2), self._phi_hessian(self.xc)   # (n, p+1, d)
            for al in self.constraint_rows:
                idx = [k for k, o in enumerate(al) for _ in range(o)]
                R = Jc[:, :, idx[0]] if len(idx) == 1 else Hc[:, :, idx[0], idx[1]]
                A.append(c * R); t.append(torch.zeros(len(self.xc), dtype=D))
        return torch.cat(A), torch.cat(t)

    def _solve(self, x, y, dy, H):
        A, t = self._system(x, y, dy, H)
        lhs = A.T @ A + self.log_lam.exp() * torch.eye(A.shape[1], dtype=D)
        return torch.linalg.solve(lhs, A.T @ t)

    def adapt(self, x, y, dy=None, H=None):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=D)
            beta = self._solve(x, torch.as_tensor(y, dtype=D),
                               None if dy is None else torch.as_tensor(dy, dtype=D),
                               None if H is None else torch.as_tensor(H, dtype=D))
        return _ALPFit(self, beta)

    def posterior_sd(self, K, C):
        with torch.no_grad():
            Phi, h = self._phi(torch.as_tensor(K, dtype=D)), self._phi(torch.as_tensor(C, dtype=D))
            P = Phi.T @ Phi + self.log_lam.exp() * torch.eye(Phi.shape[1], dtype=D)
            sd = (h * torch.linalg.solve(P, h.T).T).sum(1).clamp_min(0).sqrt()
            return sd / sd.max()

    def query_loss(self, tasks):
        loss = 0.0
        for tk in tasks:
            x = torch.as_tensor(tk.x, dtype=D)
            beta = self._solve(x, torch.as_tensor(tk.y, dtype=D),
                               None if tk.dy is None else torch.as_tensor(tk.dy, dtype=D),
                               tk.extra.get("H"))
            loss = loss + (self._phi(torch.as_tensor(tk.xq, dtype=D)) @ beta - torch.as_tensor(tk.yq, dtype=D)).pow(2).mean()
        return loss / len(tasks)


def pool_sampler(pool, output, K_support, order):
    """Tasks for one beam parameter from a pool of (K, B, J, H) measurements."""
    def sample(n_tasks, gen):
        out = []
        for i in torch.randint(len(pool), (n_tasks,), generator=gen).tolist():
            K, B, J, H = pool[i]
            p = torch.randperm(len(K), generator=gen)
            s, q = p[:K_support], p[K_support:]
            out.append(Task(x=K[s], y=B[s, output], xq=K[q], yq=B[q, output],
                            dy=J[s, output] if order >= 1 else None,
                            extra={"H": H[s, output]} if order >= 2 else {}))
        return out
    return sample
