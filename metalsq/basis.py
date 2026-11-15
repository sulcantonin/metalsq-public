"""Frozen random Fourier basis with closed-form derivatives of every order.

    u(x) = (1/sqrt N) * sum_j beta_j sin(W_j^T x + b_j),   W = sigma * W_hat

Differentiating a sinusoid shifts its phase by a quarter period and multiplies it
by the frequency, so for a multi-index alpha

    D^alpha sin(W_j^T x + b_j) = (prod_k W_jk^alpha_k) sin(W_j^T x + b_j + |alpha| pi/2).

Every derivative is therefore one sinusoid evaluation, linear in beta, with no
autograd graph. The frequencies are drawn once and frozen; only the scale sigma is
learned, so W is passed in explicitly and may carry a gradient.
"""
import math

import torch

SPECTRA = ("gaussian", "student_t", "laplace")


class SineBasis:
    """N frozen sine features on R^d.

    Parameters
    ----------
    in_dim : input dimension d.
    n_features : number of features N.
    spectrum : distribution of the unit-scale frequencies W_hat. "gaussian" gives an
        RBF kernel; "student_t" (with ``nu`` degrees of freedom) and "laplace" have
        heavier tails, i.e. rougher sample paths, which helps on non-smooth targets.
    seed : seed of the frozen draw.
    """

    def __init__(self, in_dim, n_features=200, spectrum="gaussian", nu=3.0, seed=0,
                 dtype=torch.float64):
        if spectrum not in SPECTRA:
            raise ValueError(f"spectrum must be one of {SPECTRA}, got {spectrum!r}")
        g = torch.Generator().manual_seed(seed)
        d, n = in_dim, n_features
        if spectrum == "gaussian":
            W = torch.randn(d, n, generator=g, dtype=dtype)
        elif spectrum == "student_t":                           # multivariate t, integer nu
            z = torch.randn(d, n, generator=g, dtype=dtype)
            chi2 = torch.randn(int(nu), n, generator=g, dtype=dtype).pow(2).sum(0)
            W = z / torch.sqrt(chi2 / int(nu))
        else:
            u = torch.rand(d, n, generator=g, dtype=dtype) - 0.5
            W = -torch.sign(u) * torch.log1p(-2.0 * u.abs())
        self.W_hat = W
        self.b = torch.rand(n, generator=g, dtype=dtype) * 2 * math.pi
        self.in_dim, self.n_features, self.dtype = d, n, dtype

    def _phase(self, x, W):
        x = torch.as_tensor(x, dtype=self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x @ W + self.b                                   # (M, N)

    def features(self, x, W):
        """Value features, (M, N)."""
        return torch.sin(self._phase(x, W)) / math.sqrt(self.n_features)

    def derivative(self, x, W, alpha):
        """Features of D^alpha u, (M, N). ``alpha`` is a length-d tuple of orders."""
        alpha = tuple(int(a) for a in alpha)
        if len(alpha) != self.in_dim:
            raise ValueError(f"alpha must have length {self.in_dim}")
        scale = torch.ones(self.n_features, dtype=self.dtype)
        for k, a in enumerate(alpha):
            if a:
                scale = scale * W[k] ** a
        shift = (sum(alpha) % 4) * math.pi / 2
        return scale * torch.sin(self._phase(x, W) + shift) / math.sqrt(self.n_features)

    def gradient(self, x, W):
        """Features of each partial derivative, (M, d, N)."""
        c = torch.cos(self._phase(x, W)) / math.sqrt(self.n_features)
        return c.unsqueeze(1) * W.unsqueeze(0)

    def laplacian(self, x, W):
        """Features of the Laplacian, (M, N)."""
        return -(W ** 2).sum(0) * self.features(x, W)


class PolynomialColumns:
    """All monomials of total degree <= ``degree`` in d inputs, as extra columns of the
    design next to the sine features. Their derivatives are monomials too, so they
    enter value, derivative and operator rows in closed form like the sines.

    Beam responses and many physical maps are close to low-order polynomials (a
    position nearly linear in a kick, a size squared nearly quadratic in a focusing
    strength); the sines then only have to represent what is left."""

    def __init__(self, in_dim, degree=1, dtype=torch.float64):
        from itertools import combinations_with_replacement
        exps = [[0] * in_dim]
        for deg in range(1, degree + 1):
            for combo in combinations_with_replacement(range(in_dim), deg):
                e = [0] * in_dim
                for k in combo:
                    e[k] += 1
                exps.append(e)
        self.exponents = torch.tensor(exps)                  # (P, d)
        self.n_columns = len(exps)
        self.in_dim, self.dtype = in_dim, dtype

    def _x(self, x):
        x = torch.as_tensor(x, dtype=self.dtype)
        return x.unsqueeze(-1) if x.dim() == 1 else x

    def derivative(self, x, alpha):
        """Columns of D^alpha (monomials), (M, P)."""
        x = self._x(x)
        alpha = torch.tensor([int(a) for a in alpha])
        e = self.exponents
        keep = (e >= alpha).all(1)
        coef = torch.ones(self.n_columns, dtype=self.dtype)
        for k in range(self.in_dim):
            for j in range(int(alpha[k])):
                coef = coef * (e[:, k] - j).clamp_min(0).to(self.dtype)
        powers = (e - alpha).clamp_min(0).to(self.dtype)      # (P, d)
        cols = torch.prod(x.unsqueeze(1) ** powers.unsqueeze(0), dim=-1)
        return cols * (coef * keep.to(self.dtype))

    def features(self, x):
        return self.derivative(x, (0,) * self.in_dim)

    def gradient(self, x):
        """(M, d, P)."""
        return torch.stack([self.derivative(x, [int(j == k) for j in range(self.in_dim)])
                            for k in range(self.in_dim)], 1)
