"""Meta-LSQ: a closed-form Sobolev inner solver with three meta-learned scalars.

Inner loop (per task): one ridge solve over the frozen basis, fitting values and,
optionally, derivatives,

    A = [H; sqrt(w) G_1; ...; sqrt(w) G_d],   t = [y; sqrt(w) dy_1; ...],
    beta* = argmin ||A beta - t||^2 + mu ||beta||^2.

Outer loop: Adam on log(sigma), log(mu), log(w) through the solve. torch.linalg.solve
is differentiated implicitly, so there is no unrolling and no second-order graph.

Options: ``bandwidth="diagonal"`` learns one bandwidth per input instead of one shared
(knobs on different physical scales), and ``poly_degree`` adds polynomial columns
(constant, linear, ...) to the design next to the sines.

Prior mean: with ``prior_mean=True`` the ridge pulls the coefficients towards a
meta-learned beta_0 instead of towards zero,

    beta* = argmin ||A beta - t||^2 + mu ||beta - beta_0||^2 = beta_0 + ridge(A, t - A beta_0),

so the model learns what all tasks share and each solve fits only the difference. It
adds one coefficient per column and output. Useful when the family has a non-zero mean
(dynamics of physics variants, for example); for zero-mean families leave it off.

Several outputs: pass y as (M, P) (and dy as (M, d, P)). All outputs share the design,
the scalars and one factorisation, with one right-hand side per output, so P outputs
cost about one solve. For separate scalars per output, train one model per output.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import nn

from .basis import PolynomialColumns, SineBasis


@dataclass
class Task:
    """One few-shot task: support set (x, y, optional dy) and query set (xq, yq).

    Shapes: x (M, d), y (M,), dy (M, d), xq (Q, d), yq (Q,). 1-D inputs may be
    given as (M,). With P outputs: y (M, P), dy (M, d, P), yq (Q, P)."""
    x: torch.Tensor
    y: torch.Tensor
    xq: Optional[torch.Tensor] = None
    yq: Optional[torch.Tensor] = None
    dy: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)


def ridge_solve(A, t, mu):
    """argmin ||A b - t||^2 + mu ||b||^2. Uses the n x n dual system when A has fewer
    rows than columns (the usual few-shot case), the N x N primal system otherwise.
    t may be (n,) or (n, P); P right-hand sides share one factorisation."""
    n, N = A.shape
    if n < N:
        K = A @ A.T + mu * torch.eye(n, dtype=A.dtype)
        return A.T @ torch.linalg.solve(K, t)
    return torch.linalg.solve(A.T @ A + mu * torch.eye(N, dtype=A.dtype), A.T @ t)


class Adapted:
    """The fitted model of one task, with closed-form derivatives of every order."""

    def __init__(self, basis, W, beta, poly=None):
        self.basis, self.W, self.beta, self.poly = basis, W, beta, poly

    def _cat(self, sine, poly_fn):
        return sine if self.poly is None else torch.cat([sine, poly_fn()], -1)

    def __call__(self, x):
        return self._cat(self.basis.features(x, self.W), lambda: self.poly.features(x)) @ self.beta

    predict = __call__

    def derivative(self, x, alpha):
        """D^alpha u at x; ``alpha`` is a length-d tuple of derivative orders."""
        return self._cat(self.basis.derivative(x, self.W, alpha),
                         lambda: self.poly.derivative(x, alpha)) @ self.beta

    def gradient(self, x):
        """(M, d) gradient, or (M, d, P) with P outputs."""
        G = self._cat(self.basis.gradient(x, self.W), lambda: self.poly.gradient(x))
        return torch.einsum("mdn,n...->md...", G, self.beta)

    def laplacian(self, x):
        d = self.basis.in_dim
        return sum(self.derivative(x, [2 if j == k else 0 for j in range(d)]) for k in range(d))


class MetaLSQ(nn.Module):
    """Meta-learner whose only parameters are the bandwidth sigma, the ridge mu and,
    with ``sobolev=True``, the derivative weight w.

    Parameters
    ----------
    in_dim, n_features, spectrum, seed : passed to :class:`SineBasis`.
    sobolev : fit support derivatives as well as values.
    sigma, mu, w : initial values of the three scalars.
    bandwidth : "scalar" (one sigma) or "diagonal" (one sigma per input dimension).
    poly_degree : if > 0, add all monomials up to this degree as extra columns.
    prior_mean : meta-learn the mean beta_0 the ridge shrinks towards (default: zero).
    n_outputs : number of outputs P, needed for the shape of beta_0.
    """

    def __init__(self, in_dim=1, n_features=200, sobolev=True, sigma=1.0, mu=1e-4,
                 w=1.0, spectrum="gaussian", seed=0, dtype=torch.float64,
                 bandwidth="scalar", poly_degree=0, prior_mean=False, n_outputs=1):
        super().__init__()
        if bandwidth not in ("scalar", "diagonal"):
            raise ValueError('bandwidth must be "scalar" or "diagonal"')
        self.basis = SineBasis(in_dim, n_features, spectrum=spectrum, seed=seed, dtype=dtype)
        self.poly = PolynomialColumns(in_dim, poly_degree, dtype) if poly_degree > 0 else None
        self.sobolev = sobolev
        self.dtype = dtype
        log_s = torch.full((in_dim,) if bandwidth == "diagonal" else (), math.log(sigma), dtype=dtype)
        self.log_sigma = nn.Parameter(log_s)
        self.log_mu = nn.Parameter(torch.tensor(math.log(mu), dtype=dtype))
        self.log_w = nn.Parameter(torch.tensor(math.log(w), dtype=dtype),
                                  requires_grad=sobolev)
        n_cols = n_features + (self.poly.n_columns if self.poly is not None else 0)
        self.beta0 = nn.Parameter(torch.zeros(n_cols, n_outputs, dtype=dtype)) if prior_mean else None

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
        """Current frequencies sigma * W_hat, row k scaled by sigma_k in diagonal mode
        (carries the gradient to sigma)."""
        return self.sigma.unsqueeze(-1) * self.basis.W_hat

    def features(self, x, W=None):
        """Value columns of the design at x: sines, then polynomial columns."""
        H = self.basis.features(x, self.W if W is None else W)
        return H if self.poly is None else torch.cat([H, self.poly.features(x)], -1)

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

        y may be (M,) or (M, P); with P outputs, dy is (M, d, P), extra-row targets
        are (r,) for all outputs or (r, P), and the fit returns (Q, P) predictions.
        """
        W = self.W
        x, y, dy = self._t(x), self._t(y), self._t(dy)
        multi = y.dim() == 2
        P = y.shape[1] if multi else None
        blocks = [self.features(x, W)]
        targets = [y if multi else y.reshape(-1)]
        n_cols = blocks[0].shape[1]
        if self.sobolev and dy is not None:
            sw = self.w.sqrt()
            G = self.basis.gradient(x, W)                         # (M, d, N)
            if self.poly is not None:
                G = torch.cat([G, self.poly.gradient(x)], -1)
            blocks.append(sw * G.transpose(0, 1).reshape(-1, G.shape[-1]))
            if multi:                                             # rows ordered (k, m)
                targets.append(sw * dy.reshape(G.shape[0], G.shape[1], P).transpose(0, 1).reshape(-1, P))
            else:
                targets.append(sw * dy.reshape(G.shape[0], G.shape[1]).T.reshape(-1))
        if rows is not None:
            for A_k, t_k in rows(self.basis, W):
                if A_k.shape[1] < n_cols:                         # rows on the sines only
                    A_k = torch.cat([A_k, A_k.new_zeros(A_k.shape[0], n_cols - A_k.shape[1])], 1)
                blocks.append(A_k)
                if multi:
                    t_k = t_k.reshape(A_k.shape[0], -1)
                    targets.append(t_k.expand(-1, P) if t_k.shape[1] == 1 else t_k)
                else:
                    targets.append(t_k.reshape(-1))
        A, t = torch.cat(blocks), torch.cat(targets)
        mu = torch.clamp(self.mu * ridge_scale, min=ridge_floor)
        if self.beta0 is None:
            return Adapted(self.basis, W, ridge_solve(A, t, mu), self.poly)
        if multi and P != self.beta0.shape[1]:
            raise ValueError(f"y has {P} outputs but beta_0 was built for {self.beta0.shape[1]}")
        b0 = self.beta0 if multi else self.beta0[:, 0]
        return Adapted(self.basis, W, b0 + ridge_solve(A, t - A @ b0, mu), self.poly)

    def prior(self):
        """The prior-mean model alone (what the fit returns with no data)."""
        if self.beta0 is None:
            raise ValueError("model has no prior mean")
        b0 = self.beta0[:, 0] if self.beta0.shape[1] == 1 else self.beta0
        return Adapted(self.basis, self.W, b0, self.poly)

    def query_loss(self, tasks: List[Task]):
        """Mean query MSE after adaptation, the default meta-objective. With several
        outputs it averages over them, so put them on comparable scales first (see
        :class:`metalsq.Standardizer`)."""
        loss = 0.0
        for tk in tasks:
            f = self.adapt(tk.x, tk.y, tk.dy)
            pred = f(tk.xq)
            loss = loss + (pred - self._t(tk.yq).reshape(pred.shape)).pow(2).mean()
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
        sig = self.sigma.detach()
        sig = f"{sig.item():.4g}" if sig.dim() == 0 else "[" + ", ".join(f"{v:.3g}" for v in sig.tolist()) + "]"
        s = f"MetaLSQ(sigma={sig}, mu={self.mu.item():.3g}"
        if self.sobolev:
            s += f", w={self.w.item():.4g}"
        poly = f", poly={self.poly.n_columns}" if self.poly is not None else ""
        prior = f", prior mean |beta_0|={self.beta0.norm().item():.3g}" if self.beta0 is not None else ""
        return s + f", N={self.basis.n_features}{poly}{prior}, d={self.basis.in_dim})"
