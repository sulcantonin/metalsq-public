# Reproducing the paper

Everything that produced the numbers in the NLDL 2027 paper. The scripts in
`experiments/` are the original research scripts, kept as they were run; they are
separate from the `metalsq` package and do not import it. The beam-tuning experiment
of Sec. 4.3 is in `../examples/` and does use the package.

```
experiments/        one script per experiment; most take --save <path>
results/            the JSON/NPZ files the paper's numbers and figures come from
results/superseded/ earlier versions of re-run results, kept for the record
data/               UCI Air Quality (De Vito et al. 2008, CC BY 4.0)
paper/nldl2027/     LaTeX source, figures, the built PDF and the two checking scripts
../examples/        the ARES beam-tuning scripts and their per-trial records (results/)
```

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD OMP_NUM_THREADS=1      # run from this folder, single thread
```

The paper's wall-clock numbers were measured single-threaded on CPU. Times on another
machine will differ; the errors should not, up to seed-level noise.

## Checking the paper against the results

```bash
python3 paper/nldl2027/check_numbers.py   # recomputes every quoted number from results/
python3 paper/nldl2027/make_figures.py    # renders paper/nldl2027/figures/ from results/
cd paper/nldl2027 && ./build.sh           # clean LaTeX build and page/TODO checks
```

`check_numbers.py` exits non-zero if any number printed in the paper disagrees with
the result files.

## Where each result comes from

| paper | result file | command |
|---|---|---|
| Table 1, sinusoid benchmark | `maml_multiseed.json`, `strong_frontier.json` | `python3 experiments/run_campaign.py maml strong` |
| Wall-clock comparison | `wallclock_budget.json` | `python3 experiments/wallclock_budget.py` |
| ALPaCA-style baseline | `alpaca_baseline.json` | `python3 experiments/alpaca_baseline.py` |
| ARES beam tuning (Sec. 4.3, Tables 4 and 5, the machine-time paragraph; App. K) | `ares_tuning.json` | in `../examples` (needs Cheetah and BoTorch): `python beam_full.py --beamline ares --dev 60`, `python beam_final.py --beamline ares`, `python beam_time.py --beamline ares`, `python beam_robust.py --beamline ares --scenario twin\|noise\|hysteresis\|wide`; then `python3 experiments/ares_summary.py`, which reads the per-trial records in `../examples/results/` (stored there, the larger ones gzipped) |
| ARES noise scenario, trials with a lost beam (Sec. 4.3, robustness paragraph) | `ares_noise_tail.json` | `python beam_noise_tail.py` in `../examples`, then copy `../examples/results/beam_noise_tail.json` here |
| ARES, our copy of DESY's agent against their published runs, and its error with free magnets (Sec. 4.3; App. K, "Protocol") | `ares_agent_check.json` | `python desy_runs.py fetch simulation rl`, then `python ares_agent_check.py` in `../examples`, then copy `../examples/results/ares_agent_check.json` here |
| ARES target grid (Sec. 4.3, Table 6; App. K) | `ares_grid.json` | `python ares_grid.py run`, then `python ares_grid.py summary` in `../examples` (reads DESY's published grid results from Zenodo record 7853721), then copy `../examples/results/ares_grid_summary.json` here |
| ARES screen figure (Fig. 2) | `paper/nldl2027/figures/ares_screens.png` | `python figure_ares_screen.py` in `../examples` |
| … at matched budget | `alpaca_matched_lsq_f64.json`, `_f32.json` | `python3 experiments/alpaca_matched_lsq.py [--float64] --save ...` (draws are float32 in both modes, so the two runs and ALPaCA share tasks and basis) |
| MLP-MAML with derivatives | `maml_sobolev_baseline.json` | `python3 experiments/maml_sobolev_baseline.py --iters 8000 --seeds 0 1 2 --lams 0.0 0.1 1.0 --act silu --save ...` |
| GP comparators | `periodic_gek.json`, `rebuttal_gek_shared_matched.json` | `python3 experiments/periodic_gek.py --paper-rbf --save results/periodic_gek.json`, `python3 experiments/rebuttal_gek_shared_matched.py --part mse`; the grid search over shared hyperparameters (`rebuttal_gek_shared_diag.json`): `python3 experiments/rebuttal_gek_shared_diag.py --save results/rebuttal_gek_shared_diag.json` |
| Real data (Table 3) | `rebuttal_real_*.json`, `rebuttal_relfloor_*.json` | `python3 experiments/rebuttal_real_hypotheses.py --part real\|relfloor --target CO\|NO2\|C6H6\|NOx --save results/rebuttal_real_<target>.json` (`rebuttal_relfloor_<target>.json` for `--part relfloor`) |
| Derivative rows, rank and conditioning (Fig. 3) | `review.json`, `eff_rank.json` | `python3 experiments/review_experiments.py --save results/review.json`, `python3 experiments/run_campaign.py effrank` |
| Gradient-noise and partial-derivative predictions | `grad_noise_robustness.json`, `struct_grad_noise.json` | the scripts of the same name |
| Helmholtz family | `pde.json` | `python3 experiments/pde_helmholtz_meta.py --svd --save results/pde.json` |
| Volterra family | `integro.json` | `python3 experiments/integro_meta.py --save results/integro.json` |
| Out-of-class transfer (Table 7) | `ooc_multiseed.json`, `ooc_tail_ridge.json` | `python3 experiments/run_campaign.py ooc`, `python3 experiments/ooc_tail_ridge.py` |
| Value-noise sweep | `noise_sweep.json`, `noise_sweep_2000steps.json` (the same sweep at the benchmark's 2000 steps, which App. C refers to), `noise_wgrid_seed0.json` | `python3 experiments/run_campaign.py noise` (8000 steps; the 2000-step file comes from calling `noise_sweep(n_iters=2000)` in `run_campaign.py`); `python3 experiments/noise_wgrid.py 0 2000 results/noise_wgrid_seed0.json` |
| Anisotropy | `aniso_multiseed.json`, `anisotropy_study.json` | `python3 experiments/run_campaign.py aniso`, `python3 experiments/anisotropy_study.py` |
| Frequency distributions | `matern_ooc.json` | `python3 experiments/run_campaign.py matern` |
| Ridge selection (GCV, leave-out CV; Table F.2) | `gcv_ridge.json` (all fits solved in float64), `autoridge_cv.json` (earlier standalone CV run) | `python3 experiments/gcv_ridge.py --save results/gcv_ridge.json`, `python3 experiments/autoridge_cv.py` |
| Log-density fits | `sobolev.json`, `sobolev_multiseed.json`, `dim_scan.json`, `operator_mix_gmm.json` | `python3 experiments/sobolev_test.py [--multiseed] --save ...`, `dimensionality_scan.py --save results/dim_scan.json`, `operator_mix_gmm.py` |
| Long runs and training with the ridge floor (App. C, App. F) | `ridge_divergence.json` | `python3 experiments/ridge_divergence.py` |
| Extrapolation against round-off (App. F) | `cond_vs_extrap.json` | `python3 experiments/cond_vs_extrap.py` |
| Cost of a solve and of an outer step (App. L) | `compute_accounting_breakdown.json` | `python3 experiments/compute_accounting_breakdown.py` |
| The task shown in Fig. 1 | `teaser.json` | `python3 experiments/teaser_demo.py --save results/teaser.json` |
| Posterior calibration | `calibration.json` | `python3 experiments/run_campaign.py calib` |
| Classification | `closedform_classify.json` | `python3 experiments/closedform_classify.py` |
| Mixing observation types on the Airy family (not in the camera-ready; kept for the record) | `observation_types.json` | `python3 experiments/observation_types.py --save results/observation_types.json` |

The noise sweep reuses the MLP-MAML column of an earlier run. To train it from
scratch, call `noise_sweep()` from `run_campaign.py` without `mlp_from`.
