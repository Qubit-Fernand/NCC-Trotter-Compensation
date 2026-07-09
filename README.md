# NCC-Trotter-Compensation

This repository contains numerical experiments, data checkpoints, and plotting
scripts for NCC/Trotter compensation. The active scaling code is organized under
`scaling/`.

This README describes the main files and directories that are intended to be
part of the repository. Local run caches, Bohrium staging directories, Overleaf
manuscript folders, logs, and personal memory files are not documented here.

## Repository Layout

- `scaling/`
  Main NCC scaling directory, including active implementations, batch JSON
  files, plotting notebooks, the primary data directory, and some legacy code.
- `scaling/data/`
  Primary data directory for scaling experiments. The `scaling/plot_*_scaling.ipynb`
  notebooks read JSON/NPZ checkpoints from this directory by default.
- `scaling/additonal_optimized_run/`
  Cache-optimized scripts for NCC channel log runs. These are currently used to
  split local `expected_r_min` searches from Bohrium sampled-only searches.
- `scaling/old_ver._rejective_samp./`
  Historical rejection-sampling implementation. It still contains a few old
  data subdirectories, such as `data_1_state/` and `data_3_17/`, but it is not
  the current primary data entry point.
- `figures/`
  Figure output directory.
- `scaling/plot_original_scaling.ipynb`
  Reads original-compensation data from `scaling/data/` and writes figures to
  `figures/`.
- `scaling/plot_log_scaling.ipynb`
  Reads log-compensation data from `scaling/data/` and writes figures to
  `figures/`.
- `scaling/plot_combined_scaling.ipynb`
  Compares original and log scaling results, writing figures to `figures/`.
- `scaling/plot_four_series_scaling.ipynb`
  Combines the four operator/channel and original/log series, writing figures to
  `figures/`.
- `tests/`
  Smoke and unit tests for the current scaling code.
- `1-norm/`
  Notes and CSV data for Pauli sampling overhead.
- `resource/` and `hncc_resource_plot.ipynb`
  HNCC resource-estimation scripts, precomputed data, and plotting notebooks.
  `resource/hncc_resource_estimate.py` is the main algorithmic implementation
  and CLI. `resource/hncc_resource_collect.py` imports it and only handles
  parameter sweeps and JSON table output under `resource/data/*.json`.
  `hncc_resource_plot.ipynb` reads those JSON files for plotting.
- `requirements.txt`
  Python dependency list.

## Important Data Caveat

The channel-pairing angle bug in the active channel samplers was fixed on
2026-07-09. The Pauli decompositions and `eta_pair_sum` values were correct,
but the old code converted `eta_pair_sum` to the Pauli-pairing angle by solving
`sin(theta) / cos(theta)^2 = eta`. For the signed channel sampler with branches
`U_theta = cos(theta) I + i sin(theta) P` and `-P(.)P`, the branch expectation
is proportional to `I + i tan(theta) ad_P`, so unbiased sampling requires
`theta = arctan(eta_pair_sum)`.

The active files `scaling/NCC_channel_original.py` and
`scaling/NCC_channel_log.py` now use this `arctan` parameterization and validate
against the actual paired sampled-branch expectation.

Existing sampled LCQC data under `scaling/data/` that was generated before this
fix still contains the paired-branch bias. Deterministic expected-map curves
from those runs remain valid for the intended truncated map, because the old
expected path used the hand-written target expectation rather than the biased
finite-sample branch expectation. When using `scaling/data/` for paper figures
or tables, distinguish sampled data generated before this fix from expected or
deterministic data.

## Active Code

Active scaling scripts:

- `scaling/NCC_operator_original.py`
- `scaling/NCC_operator_log.py`
- `scaling/NCC_operator_find_r_min.py`
- `scaling/NCC_channel_original.py`
- `scaling/NCC_channel_log.py`
- `scaling/NCC_channel_find_r_min.py`
- `scaling/Pauli_Hamiltonian_BCH.py`

Naming conventions:

- `operator` refers to operator-level compensation and sampling.
- `channel` refers to quantum-channel-level compensation and sampling.
- `original` is the fixed low-order compensation scheme.
- `log` is the precision-dependent higher-order compensation scheme.

## Batch Configs

Common batch configurations are stored under `scaling/`:

- `scaling/run_original.json`
- `scaling/run_log.json`
- `scaling/run_channel_original_N8_eps_sweep.json`
- `scaling/run_channel_log_N8_eps_sweep.json`

These JSON files usually set `out_dir` to `scaling/data`.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run operator batches:

```bash
python scaling/NCC_operator_find_r_min.py --batch-file scaling/run_original.json
python scaling/NCC_operator_find_r_min.py --batch-file scaling/run_log.json
```

Run channel batches:

```bash
python scaling/NCC_channel_find_r_min.py --batch-file scaling/run_channel_original_N8_eps_sweep.json
python scaling/NCC_channel_find_r_min.py --batch-file scaling/run_channel_log_N8_eps_sweep.json
```

For long jobs, unbuffered output is recommended:

```bash
PYTHONUNBUFFERED=1 python scaling/NCC_channel_find_r_min.py --batch-file scaling/run_channel_log_N8_eps_sweep.json 2>&1 | tee logs/run_channel_log.tmux.log
```

The plotting notebooks are located at `scaling/plot_*_scaling.ipynb`. They
automatically locate `scaling/data/`, so the kernel can be started either from
the repository root or from `scaling/`. PDF outputs are still written under the
repository-level `figures/` directory.

## Cache-Optimized Split Runs

Scripts under `scaling/additonal_optimized_run/` support the current NCC
channel log cache workflow:

- Run the `expected_r_min` search locally.
- Run sampled-only searches on Bohrium.
- Load the same cache set on both sides: `static_cache`,
  `slim_evolution_cache`, and `full_evolution_cache`.

## Resource Estimate Note

The expected-precision constraint in the resource-estimation code is not an
exact re-evaluation for every integer Trotter step `r`. It first computes
per-step error values on an `x_grid`, then uses `log_interpolate_error` to
evaluate `step_error(T/r)` by log-log interpolation. If `T/r` lies outside the
grid, it uses log-log extrapolation from the boundary points.

This is appropriate for resource trends and initial bracket prediction. For
paper claims about exact finite-size `expected_r_min`, use exact integer-`r`
evaluation from `scaling/additonal_optimized_run/NCC_channel_expected_r_min_cached.py`
or `scaling/NCC_channel_find_r_min.py`.

## Outputs

Default outputs are written to `scaling/data/`:

- `.json` files store parameters, repeat results, and search histories.
- `.npz` files store numerical arrays for notebook post-processing and plotting.

The `find_r_min` scripts search for the minimum Trotter step count `r` that
meets the target precision `epsilon`.
