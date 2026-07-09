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
  `r * expected_error(T / r) <= epsilon`.

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
