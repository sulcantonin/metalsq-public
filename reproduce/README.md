# Reproducing the paper

Everything that produced the numbers in the NLDL 2027 paper. These are the original
research scripts, kept as they were run. They are separate from the `metalsq`
package and do not import it.

```
experiments/        one script per experiment; most take --save <path>
results/            the JSON/NPZ files the paper's numbers and figures come from
results/superseded/ earlier versions of re-run results, kept for the record
data/               UCI Air Quality (De Vito et al. 2008, CC BY 4.0)
paper/nldl2027/     LaTeX source, figures and the two checking scripts
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
| … at matched budget | `alpaca_matched_lsq_f64.json`, `_f32.json` | `python3 experiments/alpaca_matched_lsq.py [--float64] --save ...` |
| MLP-MAML with derivatives | `maml_sobolev_baseline.json` | `python3 experiments/maml_sobolev_baseline.py --iters 8000 --seeds 0 1 2 --lams 0.0 0.1 1.0 --act silu --save ...` |
| GP comparators | `periodic_gek.json`, `rebuttal_gek_shared_matched.json` | `python3 experiments/periodic_gek.py`, `python3 experiments/rebuttal_gek_shared_matched.py --part mse` |
| Real data (Table 3) | `rebuttal_real_*.json`, `rebuttal_relfloor_*.json` | `python3 experiments/rebuttal_real_hypotheses.py --part real\|relfloor --target CO\|NO2\|C6H6\|NOx` |
| Derivative rows, rank and conditioning (Fig. 2) | `review.json`, `eff_rank.json` | `python3 experiments/review_experiments.py`, `python3 experiments/run_campaign.py effrank` |
| Gradient-noise and partial-derivative predictions | `grad_noise_robustness.json`, `struct_grad_noise.json` | the scripts of the same name |
| Helmholtz family | `pde.json` | `python3 experiments/pde_helmholtz_meta.py --svd --save results/pde.json` |
| Volterra family | `integro.json` | `python3 experiments/integro_meta.py` |
| Out-of-class transfer (Table 4) | `ooc_multiseed.json`, `ooc_tail_ridge.json` | `python3 experiments/run_campaign.py ooc`, `python3 experiments/ooc_tail_ridge.py` |
| Value-noise sweep | `noise_sweep.json`, `noise_wgrid_seed0.json` | `python3 experiments/run_campaign.py noise`; `python3 experiments/noise_wgrid.py 0 2000 results/noise_wgrid_seed0.json` |
| Anisotropy | `aniso_multiseed.json`, `anisotropy_study.json` | `python3 experiments/run_campaign.py aniso`, `python3 experiments/anisotropy_study.py` |
| Frequency distributions | `matern_ooc.json` | `python3 experiments/run_campaign.py matern` |
| Ridge selection (GCV, leave-out CV) | `gcv_ridge.json`, `autoridge_cv.json` | `python3 experiments/gcv_ridge.py`, `python3 experiments/autoridge_cv.py` |
| Log-density fits | `sobolev.json`, `sobolev_multiseed.json`, `dim_scan.json`, `operator_mix_gmm.json` | `python3 experiments/sobolev_test.py [--multiseed] --save ...`, `dimensionality_scan.py`, `operator_mix_gmm.py` |
| Posterior calibration | `calibration.json` | `python3 experiments/run_campaign.py calib` |
| Classification | `closedform_classify.json` | `python3 experiments/closedform_classify.py` |

The noise sweep reuses the MLP-MAML column of an earlier run. To train it from
scratch, call `noise_sweep()` from `run_campaign.py` without `mlp_from`.
