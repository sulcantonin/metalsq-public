import math

import pytest
import torch

from metalsq import MetaLSQ, SineBasis, Task, ridge_solve, tasks


@pytest.mark.parametrize("alpha", [(1, 0), (0, 2), (1, 1), (2, 1), (0, 3)])
def test_derivative_matches_autograd(alpha):
    basis = SineBasis(2, 16, seed=1)
    W = 1.3 * basis.W_hat
    beta = torch.randn(16, dtype=torch.float64)
    x = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    u = basis.features(x, W) @ beta
    d = u
    for k, order in enumerate(alpha):
        for _ in range(order):
            d = torch.autograd.grad(d.sum(), x, create_graph=True)[0][:, k]
    closed = basis.derivative(x.detach(), W, alpha) @ beta
    assert torch.allclose(d, closed, atol=1e-10)


def test_gradient_and_laplacian():
    basis = SineBasis(3, 32, seed=2)
    W = 0.7 * basis.W_hat
    x = torch.randn(4, 3, dtype=torch.float64)
    G = basis.gradient(x, W)
    for k in range(3):
        e = [0, 0, 0]; e[k] = 1
        assert torch.allclose(G[:, k], basis.derivative(x, W, e), atol=1e-12)
    lap = sum(basis.derivative(x, W, [2 if j == k else 0 for j in range(3)]) for k in range(3))
    assert torch.allclose(basis.laplacian(x, W), lap, atol=1e-12)


@pytest.mark.parametrize("n_rows", [8, 40])
def test_dual_and_primal_solves_agree(n_rows):
    A = torch.randn(n_rows, 20, dtype=torch.float64)
    t = torch.randn(n_rows, dtype=torch.float64)
    mu = 1e-3
    primal = torch.linalg.solve(A.T @ A + mu * torch.eye(20, dtype=torch.float64), A.T @ t)
    assert torch.allclose(ridge_solve(A, t, mu), primal, atol=1e-9)


def test_meta_gradient_matches_finite_differences():
    batch = tasks.sinusoid(K=10, K_q=20)(3, torch.Generator().manual_seed(0))
    model = MetaLSQ(sobolev=True, sigma=0.8, mu=1e-3, w=2.0)
    model.query_loss(batch).backward()
    for name in ("log_sigma", "log_mu", "log_w"):
        p = getattr(model, name)
        eps = 1e-5
        with torch.no_grad():
            p += eps; up = float(model.query_loss(batch))
            p -= 2 * eps; down = float(model.query_loss(batch))
            p += eps
        fd = (up - down) / (2 * eps)
        assert math.isclose(float(p.grad), fd, rel_tol=1e-4, abs_tol=1e-8), name


def test_adapt_interpolates_with_tiny_ridge():
    model = MetaLSQ(sobolev=False, sigma=0.7, mu=1e-10)
    x = torch.linspace(-3, 3, 8, dtype=torch.float64).unsqueeze(-1)
    y = torch.sin(x[:, 0])
    f = model.adapt(x, y)
    assert torch.allclose(f(x), y, atol=1e-6)


def test_extra_rows_are_used():
    model = MetaLSQ(sobolev=False, sigma=1.0, mu=1e-10)
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xc = torch.tensor([[0.5]], dtype=torch.float64)
    rows = lambda basis, W: [(basis.derivative(xc, W, (2,)), torch.tensor([3.0], dtype=torch.float64))]
    f = model.adapt(x, y, rows=rows)
    assert abs(f.derivative(xc, (2,)).item() - 3.0) < 1e-6


def test_meta_fit_learns_the_sinusoid_family():
    sampler = tasks.sinusoid(K=10, K_q=10)
    test = tasks.sinusoid(K=10, K_q=50)(50, torch.Generator().manual_seed(123))

    def errors(model):
        with torch.no_grad():
            return torch.stack([(model.adapt(t.x, t.y, t.dy)(t.xq) - t.yq).pow(2).mean()
                                for t in test])

    model = MetaLSQ(sobolev=True)
    before = errors(model)
    model.meta_fit(sampler, steps=2000, lr=5e-3, seed=1000)   # the paper's protocol
    after = errors(model)
    # The mean carries rare extrapolation blowups under the near-zero ridge (paper,
    # Sec. 4.4), so the median is the robust check.
    assert after.median() < before.median() / 10 and after.median() < 1e-3
    assert after.mean() < 0.1
    assert 0.3 < model.sigma.item() < 1.0 and model.w.item() > 1.0
