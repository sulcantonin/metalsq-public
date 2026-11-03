# Meta-LSQ

Few-shot meta-learning in which the inner loop is a single regularised least-squares
solve. Each task is fitted over a frozen random Fourier (sine) basis, and only three
scalars are meta-learned: the bandwidth σ, the ridge μ and the weight w of the
derivative rows. Because the derivative of a sinusoid is again a sinusoid, the fitted
model has closed-form derivatives of any order. The same solve can therefore take
derivative observations, differential-operator rows or integral rows next to the
values, and none of them needs an autograd graph.

This repository contains the `metalsq` package and, under [`reproduce/`](reproduce/),
the scripts, results and LaTeX source of the paper

> A. Sulc. *Meta-LSQ: A Closed-Form Sobolev Inner Solver for Few-Shot
> Meta-Learning.* Northern Lights Deep Learning Conference (NLDL), PMLR 352, 2027.

## Install

```bash
pip install -e .            # torch and numpy are the only dependencies
pytest -q tests             # about 5 s on a laptop CPU
```

## Usage

```python
import torch
from metalsq import MetaLSQ, tasks

model = MetaLSQ(in_dim=1, n_features=200, sobolev=True)
model.meta_fit(tasks.sinusoid(K=10, K_q=10), steps=2000, lr=5e-3)   # about 3 s
print(model)            # MetaLSQ(sigma=0.589, mu=7.58e-06, w=9.696, N=200, d=1)

# Adapting to a new task is one ridge solve.
task = tasks.sinusoid(K=10, K_q=50)(1, torch.Generator().manual_seed(0))[0]
f = model.adapt(task.x, task.y, task.dy)        # dy is optional (values only)
f(task.xq)                      # predictions
f.gradient(task.xq)             # first derivatives
f.derivative(task.xq, (3,))     # any order, closed form
```

A task sampler is any function `sampler(n_tasks, generator)` that returns a list of
`Task(x, y, xq, yq, dy=None)`. Two hooks cover problems that are not plain regression:

* `adapt(..., rows=fn)` adds linear constraints on the coefficients. `fn(basis, W)`
  returns `(A_block, t_block)` pairs, built for instance from `basis.derivative`,
  which gives the features of any derivative.
* `meta_fit(..., loss=fn)` replaces the query MSE as the meta-objective.

[`examples/helmholtz.py`](examples/helmholtz.py) uses both. It meta-learns the
bandwidth and ridge for a family of Helmholtz equations from the PDE residual alone,
with no solutions seen in training, and reaches an RMS error of about 1e-7 against the
exact solutions.

For deployment on noisy data, `adapt(..., ridge_scale=c)` multiplies the learned
ridge by `c`. In the paper, c between 10 and 30 controlled the error on the worst
tasks without costing much on noiseless ones (Sec. 4.4).

## Examples

| script | what it shows | time |
|---|---|---|
| `examples/sinusoid.py` | the 10-shot sinusoid benchmark; test MSE 0.013 with derivatives, 0.047 without | ~3 s |
| `examples/helmholtz.py` | operator rows and a label-free meta-objective | ~15 s |

## Scope

The method is meant for smooth, low-dimensional regression families that a sine
basis represents well. With a near-zero ridge, the fit can extrapolate badly
away from the support points on unfamiliar tasks, and non-smooth targets are hard
for a Gaussian spectrum (`spectrum="student_t"` helps). The paper discusses both in
Sec. 4.4.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT
