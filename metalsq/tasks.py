"""Task samplers. A sampler is any callable ``sampler(n_tasks, generator) -> [Task]``."""
import math

import torch

from .model import Task


def sinusoid(K=10, K_q=10, noise=0.0, dtype=torch.float64):
    """The few-shot sinusoid family of Finn et al. (2017): y = a sin(x + phi),
    a ~ U[0.1, 5], phi ~ U[0, pi], x ~ U[-5, 5]. Support derivatives are exact;
    ``noise`` is added to the support values only."""

    def sample(n_tasks, generator):
        tasks = []
        for _ in range(n_tasks):
            a = float(torch.rand(1, generator=generator)) * 4.9 + 0.1
            phi = float(torch.rand(1, generator=generator)) * math.pi
            x = torch.rand(K, 1, generator=generator, dtype=dtype) * 10 - 5
            xq = torch.rand(K_q, 1, generator=generator, dtype=dtype) * 10 - 5
            y = a * torch.sin(x[:, 0] + phi)
            if noise:
                y = y + noise * torch.randn(K, generator=generator, dtype=dtype)
            tasks.append(Task(x=x, y=y, dy=a * torch.cos(x + phi),
                              xq=xq, yq=a * torch.sin(xq[:, 0] + phi)))
        return tasks

    return sample


class Standardizer:
    """Affine rescaling of inputs and targets, with statistics from meta-training
    tasks only, so that nothing about the test tasks leaks into the model.

    Inputs are scaled per coordinate and targets per output. Derivative targets
    follow by the chain rule: d(y/s_y)/d(x_k/s_x_k) = dy/dx_k * s_x_k / s_y.

    >>> std = Standardizer.fit(train_tasks)
    >>> sampler = std.wrap(train_sampler)          # meta-train on standardized tasks
    >>> f = model.adapt(*std.support(task))        # adapt on a standardized support set
    >>> y_pred = std.unscale_y(f(std.scale_x(task.xq)))
    """

    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean, self.x_std, self.y_mean, self.y_std = x_mean, x_std, y_mean, y_std

    @classmethod
    def fit(cls, tasks, eps=1e-12):
        """Statistics over the support and query points of ``tasks``."""
        def cat(parts):
            parts = [p if p.dim() > 1 else p.unsqueeze(-1) for p in parts if p is not None]
            return torch.cat(parts)
        x = cat([t.x for t in tasks] + [t.xq for t in tasks])
        y = cat([t.y for t in tasks] + [t.yq for t in tasks])
        x_std, y_std = x.std(0), y.std(0)
        x_std = torch.where(x_std > eps, x_std, torch.ones_like(x_std))
        y_std = torch.where(y_std > eps, y_std, torch.ones_like(y_std))
        return cls(x.mean(0), x_std, y.mean(0), y_std)

    def scale_x(self, x):
        x = torch.as_tensor(x, dtype=self.x_mean.dtype)
        return ((x.unsqueeze(-1) if x.dim() == 1 else x) - self.x_mean) / self.x_std

    def scale_y(self, y):
        if y is None:
            return None
        y = torch.as_tensor(y, dtype=self.y_mean.dtype)
        if y.dim() == 1:                                     # single output
            return (y - self.y_mean.squeeze()) / self.y_std.squeeze()
        return (y - self.y_mean) / self.y_std

    def unscale_y(self, y):
        if y.dim() == 1:
            return y * self.y_std.squeeze() + self.y_mean.squeeze()
        return y * self.y_std + self.y_mean

    def scale_dy(self, dy):
        """dy is (M, d) for one output or (M, d, P)."""
        if dy is None:
            return None
        dy = torch.as_tensor(dy, dtype=self.x_std.dtype)
        if dy.dim() == 2:
            return dy * self.x_std / self.y_std.squeeze()
        return dy * self.x_std.unsqueeze(-1) / self.y_std

    def support(self, task):
        """(x, y, dy) of the support set, standardized, ready for ``adapt``."""
        return self.scale_x(task.x), self.scale_y(task.y), self.scale_dy(task.dy)

    def task(self, task):
        return Task(x=self.scale_x(task.x), y=self.scale_y(task.y),
                    xq=None if task.xq is None else self.scale_x(task.xq),
                    yq=self.scale_y(task.yq), dy=self.scale_dy(task.dy), extra=task.extra)

    def wrap(self, sampler):
        """A sampler that returns standardized tasks."""
        return lambda n_tasks, generator: [self.task(t) for t in sampler(n_tasks, generator)]
