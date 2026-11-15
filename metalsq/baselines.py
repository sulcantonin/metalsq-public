"""Baselines with the same interface as :class:`metalsq.MetaLSQ`, for comparisons on
any task family.

Every model has

    meta_fit(sampler, steps, batch_size=4, lr=None, seed=0) -> list of losses
    adapt(x, y, dy=None) -> f, with f(xq) of shape (Q,) or (Q, P)
    n_meta_parameters -> number of meta-learned parameters (a cost proxy)

so one evaluation loop serves all of them. y may be (M,) or (M, P), as for Meta-LSQ.
Only ALPaCA (with ``sobolev=True``) uses derivative observations; the others accept
``dy`` and ignore it.

The models are ports of the paper's baselines (``reproduce/experiments``), generalised
from 1-D sinusoids to d inputs and P outputs:

* :class:`PerTaskRidge`  polynomial ridge fit on each support set, nothing meta-learned.
* :class:`GlobalMLP`     one MLP for all tasks, no adaptation.
* :class:`MAML`          MLP-MAML (Finn et al. 2017), optionally first order.
* :class:`SharedGP`      exact RBF GP posterior mean whose length scales and noise are
                         meta-learned by query MSE, the same protocol as Meta-LSQ.
* :class:`ALPaCA`        a learned tanh encoder with a closed-form ridge read-out whose
                         ridge is meta-learned (posterior mean only, as in the paper's
                         ALPaCA-style baseline); optionally fits derivative rows too.
"""
import math

import torch
from torch import nn
from torch.func import functional_call

from .basis import PolynomialColumns
from .model import ridge_solve


def _as_x(x, dtype):
    x = torch.as_tensor(x, dtype=dtype)
    return x.unsqueeze(-1) if x.dim() == 1 else x


def _mlp(in_dim, out_dim, hidden, depth, act):
    """``depth`` Linear layers, so depth=3 has two hidden layers."""
    layers, d = [], in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class _Baseline(nn.Module):
    """Shared meta-training loop. Subclasses implement ``_fit(x, y, training)``, which
    returns a predictor and keeps the graph to the meta-parameters when training."""

    dtype = torch.float32
    default_lr = 1e-3

    def _fit(self, x, y, training, dy=None):
        raise NotImplementedError

    def adapt(self, x, y, dy=None):
        dy = None if dy is None else torch.as_tensor(dy, dtype=self.dtype)
        with torch.no_grad():
            f = self._fit(_as_x(x, self.dtype), torch.as_tensor(y, dtype=self.dtype), False, dy)

        def predict(xq):
            with torch.no_grad():
                return f(_as_x(xq, self.dtype))
        return predict

    def query_loss(self, tasks):
        loss = 0.0
        for t in tasks:
            dy = None if t.dy is None else torch.as_tensor(t.dy, dtype=self.dtype)
            f = self._fit(_as_x(t.x, self.dtype), torch.as_tensor(t.y, dtype=self.dtype), True, dy)
            pred = f(_as_x(t.xq, self.dtype))
            yq = torch.as_tensor(t.yq, dtype=self.dtype).reshape(pred.shape)
            loss = loss + (pred - yq).pow(2).mean()
        return loss / len(tasks)

    def meta_fit(self, sampler, steps=2000, batch_size=4, lr=None, seed=0, log_every=0):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return []
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(params, lr=lr or self.default_lr)
        history = []
        for step in range(steps):
            opt.zero_grad()
            loss = self.query_loss(sampler(batch_size, gen))
            loss.backward()
            opt.step()
            history.append(loss.item())
            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"step {step:5d}  loss {history[-1]:.3e}  {type(self).__name__}")
        return history

    @property
    def n_meta_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PerTaskRidge(_Baseline):
    """Ridge regression on all monomials up to ``degree`` (constant included), fitted
    to each support set alone. Nothing is meta-learned."""

    def __init__(self, in_dim, degree=1, mu=1e-6, dtype=torch.float64):
        super().__init__()
        self.dtype, self.mu = dtype, mu
        self.poly = PolynomialColumns(in_dim, degree, dtype)

    def _fit(self, x, y, training, dy=None):
        beta = ridge_solve(self.poly.features(x), y, self.mu)
        return lambda xq: self.poly.features(xq) @ beta


class GlobalMLP(_Baseline):
    """One MLP trained on the query points of all meta-training tasks; it ignores the
    support set, so it shows what the family mean alone achieves."""

    def __init__(self, in_dim, out_dim=1, hidden=40, depth=3, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.net = _mlp(in_dim, out_dim, hidden, depth, nn.ReLU).to(dtype)

    def _fit(self, x, y, training, dy=None):
        squeeze = y.dim() == 1
        return lambda xq: self.net(xq).squeeze(-1) if squeeze else self.net(xq)


class MAML(_Baseline):
    """MLP-MAML. The defaults (two hidden layers of 40 ReLU units, one inner step of
    0.01, Adam 1e-3 outside) are the paper's MLP-MAML on sinusoids."""

    def __init__(self, in_dim, out_dim=1, hidden=40, depth=3, inner_lr=0.01, inner_steps=1,
                 first_order=False, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.inner_lr, self.inner_steps, self.first_order = inner_lr, inner_steps, first_order
        self.net = _mlp(in_dim, out_dim, hidden, depth, nn.ReLU).to(dtype)

    def _forward(self, params, x, squeeze):
        out = functional_call(self.net, params, (x,))
        return out.squeeze(-1) if squeeze else out

    def _fit(self, x, y, training, dy=None):
        squeeze = y.dim() == 1
        params = dict(self.net.named_parameters())
        with torch.enable_grad():                       # adapt() runs under no_grad
            for _ in range(self.inner_steps):
                loss = (self._forward(params, x, squeeze) - y).pow(2).mean()
                grads = torch.autograd.grad(loss, list(params.values()),
                                            create_graph=training and not self.first_order)
                params = {k: v - self.inner_lr * g for (k, v), g in zip(params.items(), grads)}
        if not training:
            params = {k: v.detach() for k, v in params.items()}
        return lambda xq: self._forward(params, xq, squeeze)


class SharedGP(_Baseline):
    """Exact GP posterior mean with a unit-variance RBF kernel. The length scales (one
    per input with ``ard=True``) and the noise are shared by all tasks and meta-learned
    by query MSE through the posterior mean, the protocol of the paper's shared GP.
    With P outputs the kernel is shared and each output is its own right-hand side."""

    default_lr = 5e-3

    def __init__(self, in_dim, lengthscale=1.0, noise=1e-4, ard=True, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        shape = (in_dim,) if ard else ()
        self.log_l = nn.Parameter(torch.full(shape, math.log(lengthscale), dtype=dtype))
        self.log_noise = nn.Parameter(torch.tensor(math.log(noise), dtype=dtype))

    def kernel(self, a, b):
        l = self.log_l.exp()
        d2 = ((a.unsqueeze(1) - b.unsqueeze(0)) / l).pow(2).sum(-1)
        return torch.exp(-0.5 * d2)

    def _fit(self, x, y, training, dy=None):
        K = self.kernel(x, x) + self.log_noise.exp() * torch.eye(x.shape[0], dtype=self.dtype)
        alpha = torch.linalg.solve(K, y)
        return lambda xq: self.kernel(xq, x) @ alpha


class ALPaCA(_Baseline):
    """Learned encoder (two tanh layers, ``p`` features plus a bias column) with a
    closed-form ridge read-out per task; encoder weights and the ridge are
    meta-learned. Only the posterior mean is used.

    With ``sobolev=True`` the read-out also fits derivative observations: a slope row
    is the encoder's Jacobian (closed form for two tanh layers), weighted by a
    meta-learned sqrt(w), the same construction as Meta-LSQ's."""

    def __init__(self, in_dim, hidden=40, p=40, lam=1e-2, sobolev=False, w=1.0, dtype=torch.float32):
        super().__init__()
        self.dtype, self.sobolev = dtype, sobolev
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, p), nn.Tanh()).to(dtype)
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam), dtype=dtype))
        self.log_w = nn.Parameter(torch.tensor(math.log(w), dtype=dtype), requires_grad=sobolev)

    def _phi(self, x):
        Phi = self.enc(x)
        return torch.cat([Phi, torch.ones(Phi.shape[0], 1, dtype=self.dtype)], 1)

    def phi_gradient(self, x):
        """(M, d, p+1) derivative of the features (bias column included) w.r.t. x."""
        l1, l2 = self.enc[0], self.enc[2]
        h = torch.tanh(l1(x))                                         # (M, H)
        phi = torch.tanh(l2(h))                                       # (M, p)
        inner = (1 - h ** 2).unsqueeze(-1) * l1.weight.unsqueeze(0)   # (M, H, d)
        J = (1 - phi ** 2).unsqueeze(-1) * (l2.weight.unsqueeze(0) @ inner)   # (M, p, d)
        J = torch.cat([J, torch.zeros(J.shape[0], 1, J.shape[2], dtype=self.dtype)], 1)
        return J.transpose(1, 2)

    def _fit(self, x, y, training, dy=None):
        Phi = self._phi(x)
        A, t = Phi, y
        if self.sobolev and dy is not None:
            sw = self.log_w.exp().sqrt()
            G = self.phi_gradient(x)                                  # (M, d, p+1)
            M, d = G.shape[0], G.shape[1]
            A = torch.cat([Phi, sw * G.transpose(0, 1).reshape(M * d, -1)])
            if y.dim() == 2:
                t = torch.cat([y, sw * dy.reshape(M, d, -1).transpose(0, 1).reshape(M * d, -1)])
            else:
                t = torch.cat([y, sw * dy.reshape(M, d).T.reshape(-1)])
        lhs = A.T @ A + self.log_lam.exp() * torch.eye(A.shape[1], dtype=self.dtype)
        beta = torch.linalg.solve(lhs, A.T @ t)
        return lambda xq: self._phi(xq) @ beta
