# Channel Data Manifest

Generated: 2026-07-10

Scope: channel checkpoints used by `scaling/plot_four_series_scaling.ipynb`.

## Current Summary

After the latest local reconciliation:

- Complete channel checkpoints in `scaling/data/`: 25 canonical stems.
- Incomplete canonical checkpoints in `scaling/data/`: 0 stems.
- Missing canonical checkpoint stems: 9 stems, of which 2 are the retired `N=9` N-sweep endpoints.
- Excluding the deferred `N=9` endpoints, every missing stem is now part of the `N=8, T=2` original-channel epsilon sweep.
- The `N=8, T=2` log-channel epsilon sweep is complete for `eps in [0.1, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]`, with expected `r_min` fields filled in the promoted canonical files.

Recent local fixes:

- `NCC_channel_log_sampling_r_N8_T0.8_eps0.001_trials1000_repeats10_q4_s4` was replaced from `~/Downloads/out/data/` after verifying the local 1/10 checkpoint was an exact prefix of the downloaded 10/10 Bohrium output.
- `NCC_channel_log_sampling_r_N8_T1_eps0.001_trials1000_repeats10_q4_s4` was completed by combining the earlier prefix with an A100 sampled-only continuation.
- `NCC_channel_original_sampling_r_N8_T1_eps0.001_trials1000_repeats10` was rebuilt from the matching 4-repeat prefix and 6-repeat tail; the canonical JSON/NPZ now hold all 10 repeats.
- `NCC_channel_log_sampling_r_N8_T2_eps0.04_trials1000_repeats10_q4_s4`, `eps0.02`, `eps0.01`, `eps0.005`, `eps0.0025`, and `eps0.001` were promoted from split sampled-only Bohrium/A100 outputs plus separate expected-search results.
- The duplicate non-canonical `...T0.8...c16_m64_benchmark` files were removed from formal `scaling/data/`. A raw extracted copy remains under `.bohr/extracted/22657524/`.

## Expected-Only Override Files

These JSON/NPZ pairs expose the fields used by expected-only overrides.  The
three entries marked **extracted from sampled** were not independently rerun:
their `expected_r_min` and error value were copied from the named canonical
sampled checkpoint, and their `searches` arrays are intentionally empty.  Do
not treat them as independent evidence for the expected-r search.

| File stem | Mode | `eps` | `expected_r_min` | Provenance |
| --- | --- | ---: | ---: | --- |
| `NCC_channel_original_expected_r_min_N8_T2_eps0.1` | original | 0.1 | 38 | **Extracted from sampled** `...eps0.1_trials1000_repeats10_standard_sampled.json` (sole recorded pair) |
| `NCC_channel_original_expected_r_min_N8_T2_eps0.04` | original | 0.04 | 51 | standalone expected-only run |
| `NCC_channel_original_expected_r_min_N8_T2_eps0.02` | original | 0.02 | 63 | standalone expected-only run |
| `NCC_channel_original_expected_r_min_N8_T2_eps0.01` | original | 0.01 | 79 | standalone expected-only Bohrium run |
| `NCC_channel_original_expected_r_min_N8_T2_eps0.005` | original | 0.005 | 99 | standalone expected-only Bohrium run |
| `NCC_channel_original_expected_r_min_N8_T2_eps0.0025` | original | 0.0025 | 125 | monotone bracket: r124 full-256-basis fail; r125 Bohrium expected pass; r123 cross-check pending |
| `NCC_channel_original_expected_r_min_N8_T2_eps0.001` | original | 0.001 | 169 | **Extracted from sampled** `...eps0.001_trials1000_repeats10_standard_sampled.json` (first of two recorded pairs at the same r) |
| `NCC_channel_log_expected_r_min_N8_T2_eps0.1_q4_s4` | log | 0.1 | 21 | standalone expected-only run |
| `NCC_channel_log_expected_r_min_N8_T2_eps0.04_q4_s4` | log | 0.04 | 26 | standalone expected-only run |
| `NCC_channel_log_expected_r_min_N8_T2_eps0.02_q4_s4` | log | 0.02 | 31 | standalone expected-only run |
| `NCC_channel_log_expected_r_min_N8_T2_eps0.01_q4_s4` | log | 0.01 | 32 | standalone expected-only run |
| `NCC_channel_log_expected_r_min_N8_T2_eps0.005_q4_s4` | log | 0.005 | 43 | standalone expected-only run |
| `NCC_channel_log_expected_r_min_N8_T2_eps0.0025_q4_s4` | log | 0.0025 | 51 | standalone expected-only run |
| `NCC_channel_log_expected_r_min_N8_T2_eps0.001_q4_s4` | log | 0.001 | 63 | **Extracted from sampled** `...eps0.001_trials1000_repeats10_q4_s4_standard_sampled.json` (sole recorded pair) |

## Expected Plot Grid

- N sweep: `N=4..9`, `T=2`, `eps=0.001`
- T sweep: `N=8`, `T in [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]`, `eps=0.001`
- Epsilon sweep: `N=8`, `T=2`, `eps in [0.1, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]`
- Log channel uses `q4_s4`.
- `N=9` is listed separately as retired and is not counted in the active remaining-action summary. If the N sweep is extended later, the likely next target is `N=10`, not `N=9`.

## Complete Local Channel Files

Files below have both `.json` and `.npz` in `scaling/data/`, and `sampled_r_mins` has 10 entries.

| File stem | Mode | Parameters |
| --- | --- | --- |
| `NCC_channel_original_sampling_r_N4_T2_eps0.001_trials1000_repeats10` | original | `N=4, T=2.0, eps=0.001` |
| `NCC_channel_original_sampling_r_N5_T2_eps0.001_trials1000_repeats10` | original | `N=5, T=2.0, eps=0.001` |
| `NCC_channel_original_sampling_r_N6_T2_eps0.001_trials1000_repeats10` | original | `N=6, T=2.0, eps=0.001` |
| `NCC_channel_original_sampling_r_N7_T2_eps0.001_trials1000_repeats10` | original | `N=7, T=2.0, eps=0.001` |
| `NCC_channel_original_sampling_r_N8_T0.2_eps0.001_trials1000_repeats10` | original | `N=8, T=0.2, eps=0.001` |
| `NCC_channel_original_sampling_r_N8_T0.4_eps0.001_trials1000_repeats10` | original | `N=8, T=0.4, eps=0.001` |
| `NCC_channel_original_sampling_r_N8_T0.6_eps0.001_trials1000_repeats10` | original | `N=8, T=0.6, eps=0.001` |
| `NCC_channel_original_sampling_r_N8_T0.8_eps0.001_trials1000_repeats10` | original | `N=8, T=0.8, eps=0.001` |
| `NCC_channel_original_sampling_r_N8_T1_eps0.001_trials1000_repeats10` | original | `N=8, T=1.0, eps=0.001` |
| `NCC_channel_log_sampling_r_N4_T2_eps0.001_trials1000_repeats10_q4_s4` | log | `N=4, T=2.0, eps=0.001` |
| `NCC_channel_log_sampling_r_N5_T2_eps0.001_trials1000_repeats10_q4_s4` | log | `N=5, T=2.0, eps=0.001` |
| `NCC_channel_log_sampling_r_N6_T2_eps0.001_trials1000_repeats10_q4_s4` | log | `N=6, T=2.0, eps=0.001` |
| `NCC_channel_log_sampling_r_N7_T2_eps0.001_trials1000_repeats10_q4_s4` | log | `N=7, T=2.0, eps=0.001` |
| `NCC_channel_log_sampling_r_N8_T0.2_eps0.001_trials1000_repeats10_q4_s4` | log | `N=8, T=0.2, eps=0.001` |
| `NCC_channel_log_sampling_r_N8_T0.4_eps0.001_trials1000_repeats10_q4_s4` | log | `N=8, T=0.4, eps=0.001` |
| `NCC_channel_log_sampling_r_N8_T0.6_eps0.001_trials1000_repeats10_q4_s4` | log | `N=8, T=0.6, eps=0.001` |
| `NCC_channel_log_sampling_r_N8_T0.8_eps0.001_trials1000_repeats10_q4_s4` | log | `N=8, T=0.8, eps=0.001` |
| `NCC_channel_log_sampling_r_N8_T1_eps0.001_trials1000_repeats10_q4_s4` | log | `N=8, T=1.0, eps=0.001` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.001_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.001` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.0025_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.0025` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.005_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.005` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.01_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.01` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.02_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.02` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.04_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.04` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.1_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.1` |

## Incomplete Local Checkpoints

No current canonical channel checkpoint in `scaling/data/` is incomplete.

| File stem | Mode | Parameters | Completed repeats | `sampled_r_mins` |
| --- | --- | --- | ---: | --- |
| _None_ | | | | |

## Missing Canonical Files

Each entry means both `.json` and `.npz` are missing unless noted otherwise. Deferred `N=9` endpoints are listed in the next section.

| File stem | Roles |
| --- | --- |
| `NCC_channel_original_sampling_r_N8_T2_eps0.001_trials1000_repeats10` | N sweep original `N=8`; T sweep original `T=2`; epsilon sweep original `eps=0.001` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.1_trials1000_repeats10` | epsilon sweep original `eps=0.1` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.04_trials1000_repeats10` | epsilon sweep original `eps=0.04` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.02_trials1000_repeats10` | epsilon sweep original `eps=0.02` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.01_trials1000_repeats10` | epsilon sweep original `eps=0.01` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.005_trials1000_repeats10` | epsilon sweep original `eps=0.005` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.0025_trials1000_repeats10` | epsilon sweep original `eps=0.0025` |

## Retired N=9 Endpoints

These are still missing, but are excluded from the active remaining-action summary because they are not planned to be computed. If the N sweep is extended later, the likely next target is `N=10`.

| File stem | Roles |
| --- | --- |
| `NCC_channel_original_sampling_r_N9_T2_eps0.001_trials1000_repeats10` | N sweep original `N=9` |
| `NCC_channel_log_sampling_r_N9_T2_eps0.001_trials1000_repeats10_q4_s4` | N sweep log `N=9` |

## Notes

- `scaling/plot_four_series_scaling.ipynb` discovers channel files with `NCC_channel_*_sampling_r_*.json` and reads `N/T/epsilon/repeats` from JSON `params`.
- Prefer actual `len(sampled_r_mins)` over declared `params["repeats"]` when choosing between candidates.
- Log-channel `N=8, T=2, eps=0.0025` repeat 6 was filled from the A100 local evaluation `repeat6_556_559_result.json`, giving `sampled_r_min=559`.
