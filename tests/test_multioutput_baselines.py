import math

import pytest
import torch

from metalsq import MetaLSQ, Standardizer, Task, baselines, tasks

D = torch.float64


def _two_output_tasks(n, seed, K=10, Kq=20, d=2):
    """y = (sin(a.x + phi), a_0 cos(x_1)) with gradients, a 2-output family."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        a = torch.rand(d, generator=g, dtype=D) + 0.5
        phi = float(torch.rand(1, generator=g)) * math.pi
        f = lambda z: torch.stack([torch.sin(z @ a + phi), a[0] * torch.cos(z[:, 1])], 1)
        def df(z):
            g0 = torch.cos(z @ a + phi).unsqueeze(1) * a                    # (M, d)
            g1 = torch.stack([torch.zeros(len(z), dtype=D), -a[0] * torch.sin(z[:, 1])], 1)
            return torch.stack([g0, g1], -1)                                 # (M, d, 2)
        x = torch.rand(K, d, generator=g, dtype=D) * 4 - 2
        xq = torch.rand(Kq, d, generator=g, dtype=D) * 4 - 2
        out.append(Task(x=x, y=f(x), dy=df(x), xq=xq, yq=f(xq)))
    return out


@pytest.mark.parametrize("sobolev", [False, True])
def test_multi_output_equals_separate_solves(sobolev):
    t = _two_output_tasks(1, 0)[0]
    model = MetaLSQ(in_dim=2, sobolev=sobolev, sigma=0.7, mu=1e-4, w=2.0, poly_degree=1)
    xc = torch.zeros(3, 2, dtype=D)
    rows = lambda basis, W: [(basis.laplacian(xc, W), torch.ones(3, dtype=D))]
    joint = model.adapt(t.x, t.y, t.dy if sobolev else None, rows=rows)
    for p in range(2):
        single = model.adapt(t.x, t.y[:, p], t.dy[..., p] if sobolev else None, rows=rows)
        assert torch.allclose(joint(t.xq)[:, p], single(t.xq), atol=1e-10)
        assert torch.allclose(joint.gradient(t.xq)[..., p], single.gradient(t.xq), atol=1e-10)
    assert joint(t.xq).shape == (20, 2) and joint.gradient(t.xq).shape == (20, 2, 2)


def test_single_output_unchanged():
    t = tasks.sinusoid(K=10, K_q=20)(1, torch.Generator().manual_seed(0))[0]
    model = MetaLSQ(sobolev=True, sigma=0.6, mu=1e-5, w=3.0)
    f1 = model.adapt(t.x, t.y, t.dy)
    f2 = model.adapt(t.x, t.y.unsqueeze(-1), t.dy.unsqueeze(-1))
    assert f1(t.xq).shape == (20,)
    assert torch.allclose(f1(t.xq), f2(t.xq)[:, 0], atol=1e-12)


def test_multi_output_meta_gradient_matches_finite_differences():
    batch = _two_output_tasks(3, 1)
    model = MetaLSQ(in_dim=2, sobolev=True, sigma=0.8, mu=1e-3, w=2.0)
    model.query_loss(batch).backward()
    for name in ("log_sigma", "log_mu", "log_w"):
        p = getattr(model, name)
        eps = 1e-5
        with torch.no_grad():
            p += eps; up = float(model.query_loss(batch))
            p -= 2 * eps; down = float(model.query_loss(batch))
            p += eps
        assert math.isclose(float(p.grad), (up - down) / (2 * eps), rel_tol=1e-4, abs_tol=1e-8), name


def test_standardizer_round_trip_and_chain_rule():
    g = torch.Generator().manual_seed(0)
    a = torch.tensor([[2.0, -1.0], [0.5, 3.0], [1.0, 1.0]], dtype=D)       # (d=3, P=2)
    make = lambda x: Task(x=x, y=x @ a + torch.tensor([5.0, -2.0], dtype=D),
                          dy=a.unsqueeze(0).expand(len(x), -1, -1), xq=x, yq=x @ a)
    train = [make(torch.randn(30, 3, generator=g, dtype=D) * torch.tensor([1.0, 10.0, 0.1], dtype=D) + 4)
             for _ in range(5)]
    std = Standardizer.fit(train)
    x, y, dy = std.support(train[0])
    assert torch.allclose(std.unscale_y(y), train[0].y, atol=1e-12)
    # The standardized map is linear with slope a_kp * s_x_k / s_y_p, which scale_dy must give.
    slope = torch.linalg.lstsq(torch.cat([x, torch.ones(len(x), 1, dtype=D)], 1), y).solution[:3]
    assert torch.allclose(dy[0], slope, atol=1e-8)
    assert torch.allclose(x.mean(0), torch.zeros(3, dtype=D), atol=0.5)


def test_standardizer_single_output_shapes():
    batch = tasks.sinusoid(K=10, K_q=10)(4, torch.Generator().manual_seed(0))
    std = Standardizer.fit(batch)
    t = std.task(batch[0])
    assert t.y.shape == (10,) and t.dy.shape == (10, 1) and t.xq.shape == (10, 1)


ALL = [
    lambda d: baselines.PerTaskRidge(d, degree=1),
    lambda d: baselines.GlobalMLP(d, out_dim=2),
    lambda d: baselines.MAML(d, out_dim=2),
    lambda d: baselines.MAML(d, out_dim=2, first_order=True, inner_steps=2),
    lambda d: baselines.SharedGP(d),
    lambda d: baselines.ALPaCA(d),
]


@pytest.mark.parametrize("make", ALL)
def test_baseline_interface_multi_output(make):
    model = make(2)
    sampler = lambda n, g: _two_output_tasks(n, int(torch.randint(10 ** 6, (1,), generator=g)))
    hist = model.meta_fit(sampler, steps=3)
    assert all(math.isfinite(h) for h in hist)
    t = _two_output_tasks(1, 7)[0]
    pred = model.adapt(t.x, t.y, t.dy)(t.xq)
    assert pred.shape == (20, 2) and torch.isfinite(pred).all()


def test_baselines_single_output_shapes():
    t = tasks.sinusoid(K=10, K_q=20)(1, torch.Generator().manual_seed(0))[0]
    for model in (baselines.PerTaskRidge(1), baselines.GlobalMLP(1), baselines.MAML(1),
                  baselines.SharedGP(1), baselines.ALPaCA(1)):
        assert model.adapt(t.x, t.y)(t.xq).shape == (20,)


def test_per_task_ridge_fits_a_plane_exactly():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(10, 3, generator=g, dtype=D)
    y = x @ torch.tensor([1.0, -2.0, 0.5], dtype=D) + 3
    xq = torch.randn(5, 3, generator=g, dtype=D)
    pred = baselines.PerTaskRidge(3, mu=1e-12).adapt(x, y)(xq)
    assert torch.allclose(pred, xq @ torch.tensor([1.0, -2.0, 0.5], dtype=D) + 3, atol=1e-6)


def test_maml_inner_step_lowers_support_loss():
    t = tasks.sinusoid(K=10, K_q=20)(1, torch.Generator().manual_seed(3))[0]
    model = baselines.MAML(1, inner_lr=0.01, inner_steps=5)
    before = (model.net(t.x.float()).squeeze(-1) - t.y.float()).pow(2).mean()
    after = (model.adapt(t.x, t.y)(t.x) - t.y.float()).pow(2).mean()
    assert after < before


def test_shared_gp_meta_fit_learns_the_sinusoid_family():
    sampler = tasks.sinusoid(K=10, K_q=10)
    test = tasks.sinusoid(K=10, K_q=50)(30, torch.Generator().manual_seed(123))
    err = lambda m: torch.stack([(m.adapt(t.x, t.y)(t.xq) - t.yq).pow(2).mean() for t in test]).median()
    model = baselines.SharedGP(1, lengthscale=3.0, noise=1e-2)
    before = err(model)
    model.meta_fit(sampler, steps=300, seed=1)
    assert err(model) < before / 3


def test_prior_mean_matches_the_closed_form():
    t = _two_output_tasks(1, 2)[0]
    model = MetaLSQ(in_dim=2, sobolev=True, sigma=0.7, mu=1e-2, w=2.0, poly_degree=1,
                    prior_mean=True, n_outputs=2)
    with torch.no_grad():
        model.beta0.normal_(generator=torch.Generator().manual_seed(0))
    f = model.adapt(t.x, t.y, t.dy)
    # Direct solve of (A^T A + mu I) beta = A^T t + mu beta_0 with the same stacked rows.
    W = model.W
    G = torch.cat([model.basis.gradient(t.x, W), model.poly.gradient(t.x)], -1)
    sw = model.w.sqrt()
    A = torch.cat([model.features(t.x), sw * G.transpose(0, 1).reshape(-1, G.shape[-1])])
    T = torch.cat([t.y, sw * t.dy.transpose(0, 1).reshape(-1, 2)])
    mu = model.mu
    beta = torch.linalg.solve(A.T @ A + mu * torch.eye(A.shape[1], dtype=D), A.T @ T + mu * model.beta0)
    assert torch.allclose(f.beta, beta, atol=1e-7)


def test_prior_mean_zero_is_the_plain_fit_and_huge_ridge_returns_the_prior():
    t = _two_output_tasks(1, 3)[0]
    plain = MetaLSQ(in_dim=2, sobolev=False, sigma=0.7, mu=1e-3)
    prior = MetaLSQ(in_dim=2, sobolev=False, sigma=0.7, mu=1e-3, prior_mean=True, n_outputs=2)
    assert torch.allclose(plain.adapt(t.x, t.y)(t.xq), prior.adapt(t.x, t.y)(t.xq), atol=1e-12)
    with torch.no_grad():
        prior.beta0.normal_(generator=torch.Generator().manual_seed(1))
    far = prior.adapt(t.x, t.y, ridge_scale=1e12)(t.xq)
    assert torch.allclose(far, prior.prior()(t.xq), atol=1e-6)


def test_prior_mean_meta_gradient_and_single_output():
    batch = tasks.sinusoid(K=5, K_q=10)(3, torch.Generator().manual_seed(0))
    for b in batch:
        b.y, b.yq = b.y + 2.0, b.yq + 2.0                  # a family with a non-zero mean
    model = MetaLSQ(sobolev=True, sigma=0.8, mu=1e-1, w=2.0, prior_mean=True)
    model.query_loss(batch).backward()
    for j in (0, 7, 150):
        eps = 1e-5
        with torch.no_grad():
            model.beta0[j, 0] += eps; up = float(model.query_loss(batch))
            model.beta0[j, 0] -= 2 * eps; down = float(model.query_loss(batch))
            model.beta0[j, 0] += eps
        assert math.isclose(float(model.beta0.grad[j, 0]), (up - down) / (2 * eps), rel_tol=1e-4, abs_tol=1e-9)
    assert model.adapt(batch[0].x, batch[0].y)(batch[0].xq).shape == (10,)


def test_alpaca_feature_gradient_matches_autograd():
    model = baselines.ALPaCA(3, p=8, dtype=torch.float64)
    x = torch.randn(4, 3, dtype=D, requires_grad=True)
    phi = model._phi(x)
    auto = torch.stack([torch.autograd.grad(phi[:, j].sum(), x, retain_graph=True)[0]
                        for j in range(phi.shape[1])], -1)          # (M, d, p+1)
    assert torch.allclose(model.phi_gradient(x.detach()), auto, atol=1e-10)


def test_alpaca_sobolev_uses_slopes_and_trains():
    model = baselines.ALPaCA(2, sobolev=True, dtype=torch.float64)
    t = _two_output_tasks(1, 4)[0]
    with_slopes = model.adapt(t.x, t.y, t.dy)(t.xq)
    without = model.adapt(t.x, t.y)(t.xq)
    assert with_slopes.shape == (20, 2) and not torch.allclose(with_slopes, without)
    sampler = lambda n, g: _two_output_tasks(n, int(torch.randint(10 ** 6, (1,), generator=g)))
    hist = model.meta_fit(sampler, steps=3)
    assert all(math.isfinite(h) for h in hist) and model.log_w.grad is not None
