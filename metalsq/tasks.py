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
