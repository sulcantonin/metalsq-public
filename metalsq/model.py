"""Meta-LSQ: a closed-form Sobolev inner solver with three meta-learned scalars.

Inner loop (per task): one ridge solve over the frozen basis, fitting values and,
optionally, derivatives,

    A = [H; sqrt(w) G_1; ...; sqrt(w) G_d],   t = [y; sqrt(w) dy_1; ...],
    beta* = argmin ||A beta - t||^2 + mu ||beta||^2.

Outer loop: Adam on log(sigma), log(mu), log(w) through the solve. torch.linalg.solve
is differentiated implicitly, so there is no unrolling and no second-order graph.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import nn

from .basis import SineBasis


@dataclass
class Task:
    """One few-shot task: support set (x, y, optional dy) and query set (xq, yq).

    Shapes: x (M, d), y (M,), dy (M, d), xq (Q, d), yq (Q,). 1-D inputs may be
    given as (M,)."""
    x: torch.Tensor
    y: torch.Tensor
    xq: Optional[torch.Tensor] = None
    yq: Optional[torch.Tensor] = None
    dy: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)


def ridge_solve(A, t, mu):
    """argmin ||A b - t||^2 + mu ||b||^2. Uses the n x n dual system when A has fewer
    rows than columns (the usual few-shot case), the N x N primal system otherwise."""
    n, N = A.shape
    if n < N:
        K = A @ A.T + mu * torch.eye(n, dtype=A.dtype)
        return A.T @ torch.linalg.solve(K, t)
    return torch.linalg.solve(A.T @ A + mu * torch.eye(N, dtype=A.dtype), A.T @ t)


class Adapted:
    """The fitted model of one task, with closed-form derivatives of every order."""

    def __init__(self, basis, W, beta):
        self.basis, self.W, self.beta = basis, W, beta

    def __call__(self, x):
        return self.basis.features(x, self.W) @ self.beta

    predict = __call__

    def derivative(self, x, alpha):
        """D^alpha u at x; ``alpha`` is a length-d tuple of derivative orders."""
        return self.basis.derivative(x, self.W, alpha) @ self.beta

    def gradient(self, x):
        """(M, d) gradient."""
        return torch.einsum("mdn,n->md", self.basis.gradient(x, self.W), self.beta)

    def laplacian(self, x):
        return self.basis.laplacian(x, self.W) @ self.beta


class MetaLSQ(nn.Module):
    """Meta-learner whose only parameters are the bandwidth sigma, the ridge mu and,
    with ``sobolev=True``, the derivative weight w.

    Parameters
    ----------
    in_dim, n_features, spectrum, seed : passed to :class:`SineBasis`.
    sobolev : fit support derivatives as well as values.
    sigma, mu, w : initial values of the three scalars.
    """

    def __init__(self, in_dim=1, n_features=200, sobolev=True, sigma=1.0, mu=1e-4,
                 w=1.0, spectrum="gaussian", seed=0, dtype=torch.float64):
        super().__init__()
        self.basis = SineBasis(in_dim, n_features, spectrum=spectrum, seed=seed, dtype=dtype)
        self.sobolev = sobolev
        self.dtype = dtype
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma), dtype=dtype))
        self.log_mu = nn.Parameter(torch.tensor(math.log(mu), dtype=dtype))
        self.log_w = nn.Parameter(torch.tensor(math.log(w), dtype=dtype),
                                  requires_grad=sobolev)

    @property
    def sigma(self):
        return self.log_sigma.exp()

    @property
    def mu(self):
        return self.log_mu.exp()

    @property
    def w(self):
        return self.log_w.exp()

    @property
    def W(self):
        """Current frequencies sigma * W_hat (carries the gradient to sigma)."""
        return self.sigma * self.basis.W_hat

    def _t(self, v):
        return None if v is None else torch.as_tensor(v, dtype=self.dtype)

    def adapt(self, x, y, dy=None, rows=None, ridge_scale=1.0, ridge_floor=0.0):
        """Fit one task by a single ridge solve and return an :class:`Adapted` model.

        rows : optional callable ``rows(basis, W) -> [(A_block, t_block), ...]`` with
            further linear constraints on beta, e.g. differential-operator or
            integral rows built from ``basis.derivative``. They enter the stacked
            system as given (weight them yourself).
        ridge_scale, ridge_floor : deployment ridge ``max(ridge_scale * mu, floor)``.
            On noisy data a floor of 10-30 times the learned ridge was a good default
            in the paper; leave both at their defaults during meta-training.
        """
        W = self.W
        x, y, dy = self._t(x), self._t(y), self._t(dy)
        blocks = [self.basis.features(x, W)]
        targets = [y.reshape(-1)]
        if self.sobolev and dy is not None:
            sw = self.w.sqrt()
            G = self.basis.gradient(x, W)                         # (M, d, N)
            dy = dy.reshape(G.shape[0], G.shape[1])
            blocks.append(sw * G.transpose(0, 1).reshape(-1, G.shape[-1]))
            targets.append(sw * dy.T.reshape(-1))
        if rows is not None:
            for A_k, t_k in rows(self.basis, W):
                blocks.append(A_k)
                targets.append(t_k.reshape(-1))
        A, t = torch.cat(blocks), torch.cat(targets)
        mu = torch.clamp(self.mu * ridge_scale, min=ridge_floor)
        return Adapted(self.basis, W, ridge_solve(A, t, mu))

    def query_loss(self, tasks: List[Task]):
        """Mean query MSE after adaptation, the default meta-objective."""
        loss = 0.0
        for tk in tasks:
            f = self.adapt(tk.x, tk.y, tk.dy)
            loss = loss + (f(tk.xq) - self._t(tk.yq).reshape(-1)).pow(2).mean()
        return loss / len(tasks)

    def meta_fit(self, sampler: Callable[[int, torch.Generator], List[Task]],
                 steps=2000, batch_size=4, lr=5e-3, loss=None, seed=0, log_every=0):
        """Meta-train the scalars with Adam.

        sampler : ``sampler(batch_size, generator) -> list of Task``.
        loss : optional ``loss(model, tasks) -> scalar tensor`` replacing the query
            MSE, e.g. a label-free PDE residual at held-out collocation points.
        Returns the list of per-step losses.
        """
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)
        loss_fn = loss or (lambda model, tasks: model.query_loss(tasks))
        history = []
        for step in range(steps):
            opt.zero_grad()
            value = loss_fn(self, sampler(batch_size, gen))
            value.backward()
            opt.step()
            history.append(value.item())
            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"step {step:5d}  loss {history[-1]:.3e}  {self!r}")
        return history

    def __repr__(self):
        s = f"MetaLSQ(sigma={self.sigma.item():.4g}, mu={self.mu.item():.3g}"
        if self.sobolev:
            s += f", w={self.w.item():.4g}"
        return s + f", N={self.basis.n_features}, d={self.basis.in_dim})"
