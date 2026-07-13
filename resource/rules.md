# HNCC Resource Data Processing

This folder keeps the HNCC resource-estimate scripts and data products.

## Current Entrypoints

- `hncc_resource_estimate.py`
  Core numerical implementation and legacy all-in-one plotting CLI.

- `hncc_resource_collect.py`
  Legacy all-in-one data collector. It runs the full resource pipeline and
  writes final rows directly to `resource/data/hncc_resource_data*.json`.

- `r_for_expected_precision.py`
  Split-pipeline stage for the expected-precision constraint only. It computes
  `r_precision` from
  `r * expected_error(T / r) <= epsilon`. The default expected-error norm is now
  the basis-state trace norm.

- `r_for_1_norm.py`
  Split-pipeline stage for the accumulated 1-norm constraint only. It computes
  `r_1_norm` from
  `lambda(T / r)^r <= cap`.

- `hncc_resource.py`
  Split-pipeline final stage. It reads the two constraint files, sets
  `r_final = max(r_precision, r_1_norm)`, recomputes final bias / norm overhead
  at `r_final`, multiplies by per-step gate costs, and writes final resource
  rows. For reproducing existing legacy final files without rerunning dense
  step-error curves, use `--use-reference-final`.

## Current Split-Pipeline Workflow

The current resource-estimate workflow is the split pipeline in the `resource`
root, not the legacy all-in-one collector.

1. Run `r_for_expected_precision.py` to generate
   `data/split_constraints/r_for_expected_precision_N*_trace.json` by default.
   This stage imports `cluster/hncc_resource_estimate.py`, builds the step-error
   curves, and solves the expected-precision constraint
   `r * expected_error(T / r) <= epsilon`. Use `--error-norm fro` only to
   reproduce the legacy Frobenius files without the `_trace` suffix.

2. Run `r_for_1_norm.py` to generate
   `data/split_constraints/r_for_1_norm_N*_cap_*.json`. This stage also imports
   `cluster/hncc_resource_estimate.py`, but it only needs the Pauli 1-norm
   polynomials for NCC/HNCC and solves the accumulated normalization constraint
   `lambda(T / r)^r <= cap`.

3. Run `hncc_resource.py` to merge the two constraint files. By default it reads
   the `_trace` expected-precision file and writes final rows with a `_trace`
   suffix. For each method and scan point, it uses
   `r = max(r_precision, r_1_norm)`, records `active_constraint`, recomputes the
   final bias and normalization overhead at that selected `r`, and applies the
   shared per-step gate-cost model. For plotting-only updates where
   `r_1_norm <= r_precision` for every row, use `--skip-final-bias-recompute` to
   skip the expensive final bias recomputation and reuse `bias_precision`.

The current resource-estimate precision constraint uses the basis-state
worst-case trace norm. The old plotted legacy resource data used the
basis-state Frobenius proxy implemented in `cluster/hncc_resource_estimate.py`
(`max_basis_fro_error` and the historical HNCC/NCC/extrapolation `ord="fro"`
calls). Neither version is a diamond-norm constraint. The repetition panel in
the plots is separate: it comes from the accumulated sampling normalization
factor `lambda^r`, not from the error norm.

Current main data notes:

- For `N=8`, the split files can be generated directly and then merged with
  `hncc_resource.py`.

- For `N=10`, the trace-norm expected-precision file
  `data/split_constraints/r_for_expected_precision_N10_trace.json` was generated
  locally with `python resource/r_for_expected_precision.py --N 10 --scan both`.
  It contains both the target-precision sweep and the simulation-time sweep. The
  HNCC trace-norm step-error calculation uses a low-rank outer-product
  representation of each basis-state density difference, with batched
  applications of the BCH matrices for each step size. This computes the same
  trace norm as the dense density-matrix eigensolver but avoids repeated
  full-size `1024 x 1024` decompositions.

- The current `N=10`, cap `1.75`, trace-norm final resource rows were generated
  with `python resource/hncc_resource.py --N 10 --sampling-overhead-cap 1.75
  --scan both --skip-final-bias-recompute`, using the existing 1-norm file
  `data/split_constraints/r_for_1_norm_N10_cap_1p75.json`.

- For `N=10`, the legacy Frobenius split files seed `r_for_expected_precision_N10.json`
  from `data/hncc_resource_data_N10_cap_e.json`, because recomputing the dense
  HNCC step-error curve remains slow. The final split files are then produced
  with `hncc_resource.py --use-reference-final`, which adds
  `r_precision`, `r_1_norm`, and `active_constraint` provenance to the validated
  legacy final rows.

## Legacy Final Data

These files are final rows produced by the legacy all-in-one pipeline:

- `data/hncc_resource_data_N8_cap_1.75.json`
  `N=8`, accumulated 1-norm cap `1.75`.

- `data/hncc_resource_data_N8_cap_e.json`
  `N=8`, accumulated 1-norm cap `e`.

- `data/hncc_resource_data_N10.json`
  `N=10`, accumulated 1-norm cap `1.75`.

- `data/hncc_resource_data_N10_cap_e.json`
  `N=10`, accumulated 1-norm cap `e`.

- `data/hncc_resource_data_N10_eps_sanity.json`
  Sanity run only: `N=10`, cap `1.75`, eps scan only, `num_x=12`.
  Do not treat it as a peer of the four main data files.

The legacy final files contain `steps`, `x`, `bias`, `norm_overhead`, gate
counts, and repetitions, but they do not separately record which constraint
determined the step count.

## Split Constraint Data

Split-pipeline products live under `data/split_constraints/`.

- `r_for_expected_precision_N8.json`
  Expected-precision constraint for `N=8`. This is independent of the 1-norm
  cap, so it is shared by cap `1.75` and cap `e`.

- `r_for_1_norm_N8_cap_1p75.json`
  1-norm constraint for `N=8`, cap `1.75`.

- `r_for_1_norm_N8_cap_e.json`
  1-norm constraint for `N=8`, cap `e`.

- `hncc_resource_split_N8_cap_1p75.json`
  Final rows reconstructed from the two split constraints for `N=8`, cap
  `1.75`.

- `hncc_resource_split_N8_cap_e.json`
  Final rows reconstructed from the two split constraints for `N=8`, cap `e`.

- `r_for_expected_precision_N10.json`
  Expected-precision rows for `N=10`, seeded from the existing final
  `data/hncc_resource_data_N10_cap_e.json`. This avoids rerunning the expensive
  dense HNCC step-error curve.

- `r_for_1_norm_N10_cap_1p75.json`
  1-norm constraint for `N=10`, cap `1.75`.

- `r_for_1_norm_N10_cap_e.json`
  1-norm constraint for `N=10`, cap `e`.

- `hncc_resource_split_N10_cap_1p75.json`
  Final rows for `N=10`, cap `1.75`, reconstructed by adding split provenance
  fields to `data/hncc_resource_data_N10.json`.

- `hncc_resource_split_N10_cap_e.json`
  Final rows for `N=10`, cap `e`, reconstructed by adding split provenance
  fields to `data/hncc_resource_data_N10_cap_e.json`.

The final split rows include the extra provenance fields:

- `r_precision`
- `r_1_norm`
- `active_constraint`

where `active_constraint` is one of `expected_precision`, `one_norm`, or
`both`.

## Reproduction Status

Bottom line as of 2026-07-06: the split pipeline successfully reproduces the
four main legacy final datasets:

- `N=8`, cap `1.75`
- `N=8`, cap `e`
- `N=10`, cap `1.75`
- `N=10`, cap `e`

The reproduced split outputs are suitable replacements for the legacy
all-in-one outputs when plotting or inspecting the resource estimates. The only
known numerical difference is one `N=8`, cap `e` `bias` value at about `1e-8`,
which does not change the selected step count, norm overhead, gate counts,
repetitions, or weights.

Verified locally:

- `data/split_constraints/hncc_resource_split_N8_cap_1p75.json` reproduces
  `data/hncc_resource_data_N8_cap_1.75.json` exactly, field by field.

- `data/split_constraints/hncc_resource_split_N8_cap_e.json` reproduces
  `data/hncc_resource_data_N8_cap_e.json` in all resource-decision fields
  (`steps`, `x`, `norm_overhead`, gate counts, repetitions, weights). A few
  `bias` entries differ only at about `1e-8`, consistent with recomputing /
  reinterpolating floating point values.

- `data/split_constraints/hncc_resource_split_N10_cap_1p75.json` reproduces
  `data/hncc_resource_data_N10.json` exactly, field by field.

- `data/split_constraints/hncc_resource_split_N10_cap_e.json` reproduces
  `data/hncc_resource_data_N10_cap_e.json` exactly, field by field.

Implementation note:

- Direct split-pipeline `N=10` expected-precision recomputation was killed
  before the matrix-construction fix. The core now builds dense Pauli operators
  from sparse Pauli-string actions instead of caching dense per-string matrices,
  which fixes the memory spike. However, recomputing the full dense HNCC
  step-error curve for `N=10` remains slow. The current `N=10` split files
  therefore seed expected-precision rows from the already validated legacy final
  data and use norm-only recomputation plus `hncc_resource.py
  --use-reference-final` to add split provenance fields.

## Smoke Data

`data/split_constraints_smoke/` contains a small `N=4` smoke test for validating
the split-pipeline wiring. It is not part of the plotted or manuscript data.
