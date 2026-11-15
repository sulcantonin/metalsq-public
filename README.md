# Meta-LSQ

Few-shot meta-learning in which the inner loop is a single regularised least-squares
solve. Each task is fitted over a frozen random Fourier (sine) basis, and only three
scalars are meta-learned: the bandwidth σ, the ridge μ and the weight w of the
derivative rows. Because the derivative of a sinusoid is again a sinusoid, the fitted
model has closed-form derivatives of any order. The same solve can therefore take
derivative observations, differential-operator rows or integral rows next to the
values, and none of them needs an autograd graph.

This repository contains the `metalsq` package and everything behind the paper

> A. Sulc. *Meta-LSQ: A Closed-Form Sobolev Inner Solver for Few-Shot
> Meta-Learning.* Northern Lights Deep Learning Conference (NLDL), PMLR 352, 2027.

[Paper (PDF)](reproduce/paper/nldl2027/main.pdf) ·
[Project page](https://sulcantonin.github.io/metalsq-public/) ·
[Reproducing the paper](reproduce/README.md)

```
metalsq/      the package (torch and numpy only)
tests/        unit tests of the package
examples/     two short examples of the package, and the beam-tuning experiment of Sec. 4.3
reproduce/    research scripts, stored results, data and LaTeX source of the paper
docs/         the project page
```

## Install

```bash
pip install -e .            # torch and numpy are the only dependencies
pytest -q tests             # about 10 s on a laptop CPU
```

## Usage

```python
import torch
from metalsq import MetaLSQ, tasks

model = MetaLSQ(in_dim=1, n_features=200, sobolev=True)
model.meta_fit(tasks.sinusoid(K=10, K_q=10), steps=2000, lr=5e-3)   # about 4 s
print(model)            # MetaLSQ(sigma=0.6151, mu=7.66e-06, w=10.13, N=200, d=1)

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

Further options:

* `MetaLSQ(bandwidth="diagonal")` learns one bandwidth per input, and
  `MetaLSQ(poly_degree=1)` adds a constant and linear terms to the solve (with
  closed-form derivatives, like the sines).
* `MetaLSQ(spectrum="student_t")` draws heavier-tailed frequencies for less smooth
  targets.
* `adapt(..., ridge_scale=c)` multiplies the learned ridge by `c` and
  `adapt(..., ridge_floor=m)` keeps it at or above `m`. On the paper's real data a
  floor of 10 to 30 times the learned ridge kept most of the gain on noisy targets
  (Sec. 4.5); that range was chosen on held-out data and still needs validating.
* For several outputs, pass `y` as (M, P) and `dy` as (M, d, P). The outputs share the
  scalars and one factorisation of the solve.
* `Standardizer.fit(tasks)` rescales inputs and targets with statistics from
  meta-training tasks only, and rescales derivative targets to match.
* `metalsq.baselines` has baselines with the same `meta_fit` / `adapt` interface:
  per-task polynomial ridge, a global MLP, MLP-MAML, a shared-hyperparameter GP and an
  ALPaCA-style encoder with a ridge read-out. They ignore derivative observations.

`MetaLSQ` works in double precision (the baselines default to single) and solves the
dual system when a task has fewer rows than features. The paper's sinusoid and real-data tables were produced by
the research scripts in `reproduce/`, which run in single precision, so the package's
numbers differ somewhat from those tables.

## Examples

| script | what it shows | time |
|---|---|---|
| `examples/sinusoid.py` | 10-shot sinusoid regression with the package: test MSE 0.013 with derivatives, 0.047 without (`--values`) | ~4 s |
| `examples/helmholtz.py` | operator rows and a label-free meta-objective | ~15 s |

## Reproducing the paper

Every number in the paper is recomputed from stored result files by a script, and the
figures are drawn from the same files:

```bash
cd reproduce
pip install -r requirements.txt
python3 paper/nldl2027/check_numbers.py   # exits non-zero on any disagreement
python3 paper/nldl2027/make_figures.py    # renders paper/nldl2027/figures/
cd paper/nldl2027 && ./build.sh           # clean LaTeX build with page and reference checks
```

[`reproduce/README.md`](reproduce/README.md) lists, for every table and figure, the
result file and the command that produced it.

### Beam tuning on ARES (Sec. 4.3, App. K)

The closed-loop experiment on DESY's ARES experimental area is in `examples/`. It
needs more than the package:

```bash
pip install cheetah-accelerator scikit-learn scipy botorch matplotlib
```

| script | role |
|---|---|
| `ares.py`, `beamtune.py` | the simulated beamline (Cheetah) and the tuning harness |
| `ares_vs_rl.py` | DESY's 300 trials and their trained agent, downloaded at run time and loaded without executing pickles |
| `bo_desy.py` | DESY's Bayesian-optimisation protocol |
| `beam_structure.py`, `beam_full.py` | the surrogate models, their meta-training, the tuning loop, and the choice of variants on development tasks |
| `beam_final.py` | all methods on the 300 trials (Table 4) |
| `beam_time.py`, `beam_machine_time.py` | error after every measurement and the machine-time model (Sec. 4.3) |
| `beam_robust.py`, `beam_noise_tail.py` | the robustness scenarios (Table 5) and the trials with a lost beam |
| `ares_agent_check.py` | our copy of DESY's agent against every step of their published runs, and its error with and without clipped magnets |
| `ares_grid.py`, `desy_runs.py` | DESY's grid of targets over the full range (Table 6), read from their Zenodo record 7853721 |
| `figure_ares_screen.py` | Fig. 2 |

`examples/results/` holds the per-trial records of these runs (the larger ones
gzipped) and the trained surrogates, and `reproduce/experiments/ares_summary.py`
turns them into `reproduce/results/ares_tuning.json`, which the paper's numbers are
checked against. All of it is simulation; no method here was run on the machine.
DESY's agent runs from its published weights inside a re-implementation of their
wrappers, and their Bayesian optimisation is re-implemented from their code. Our copy
of the agent makes the moves recorded in their published runs at every step. The
tables hold its magnets to their limits, which their simulation did not do; with free
magnets its mean error is 11.8 µm instead of 17.1 (the median stays near 4.4). The
scripts also know a second beamline (`lcls_diag0`), which is not part of this
repository.

## Scope

The method is meant for smooth, low-dimensional regression families that a sine
basis represents well. With a near-zero ridge the fit can extrapolate badly away from
the support points on unfamiliar tasks, and non-smooth targets are hard for a Gaussian
spectrum. On the paper's real sensor-calibration data it is less accurate than MAML on
three of four targets. The paper discusses these limits in Sec. 4.5 and 4.6.

## Citation

```bibtex
@inproceedings{sulc2027metalsq,
  title     = {Meta-{LSQ}: A Closed-Form {S}obolev Inner Solver for Few-Shot Meta-Learning},
  author    = {Sulc, Antonin},
  booktitle = {Proceedings of the 8th Northern Lights Deep Learning Conference (NLDL)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {352},
  publisher = {PMLR},
  year      = {2027}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License and third-party material

The code is under the MIT license. `reproduce/data/AirQualityUCI.csv` is the UCI Air
Quality data set (De Vito et al. 2008, CC BY 4.0). The ARES scripts download DESY's
evaluation trials (`desy-ml/rl-vs-bo`), their trained agent (`RL4AA/rl4aa25-challenge`,
GPL-3.0) and their published optimisation runs (Zenodo 7853721, CC BY 4.0) at run
time; none of these files is redistributed here.
