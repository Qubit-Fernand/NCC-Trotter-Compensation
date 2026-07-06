# Channel Data Manifest

Generated: 2026-07-01

Scope: channel checkpoints used by `plot_four_series_scaling.ipynb`.

## Current Summary

After the latest local reconciliation:

- Complete channel checkpoints in `scaling/data/`: 18 canonical stems.
- Incomplete canonical checkpoints in `scaling/data/`: 3 stems.
- Missing canonical checkpoint stems: 13 stems, of which 2 are the deferred `N=9` N-sweep endpoints.
- Excluding the deferred `N=9` endpoints, every missing stem is part of the `N=8, T=2` epsilon sweep.
- One non-`T=2` exception remains incomplete: log channel `N=8, T=1.0, eps=0.001` is still 1/10. The older 10/10 copy under `scaling/old_ver._rejective_samp./data_1_state/` uses `r_max=1024`, while the current canonical log run uses `r_max=4096`, so it has not been used as a replacement.

Recent local fixes:

- `NCC_channel_log_sampling_r_N8_T0.8_eps0.001_trials1000_repeats10_q4_s4` was replaced from `~/Downloads/out/data/` after verifying the local 1/10 checkpoint was an exact prefix of the downloaded 10/10 Bohrium output.
- `NCC_channel_original_sampling_r_N8_T1_eps0.001_trials1000_repeats10` was rebuilt from the matching 4-repeat prefix and 6-repeat tail; the canonical JSON/NPZ now hold all 10 repeats.
- The duplicate non-canonical `...T0.8...c16_m64_benchmark` files were removed from formal `scaling/data/`. A raw extracted copy remains under `.bohrium/extracted/22657524/`.

## Expected Plot Grid

- N sweep: `N=4..9`, `T=2`, `eps=0.001`
- T sweep: `N=8`, `T in [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]`, `eps=0.001`
- Epsilon sweep: `N=8`, `T=2`, `eps in [0.1, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]`
- Log channel uses `q4_s4`.
- `N=9` is listed separately as deferred and is not counted in the active remaining-action summary.

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
| `NCC_channel_log_sampling_r_N8_T2_eps0.1_trials1000_repeats10_q4_s4` | log | `N=8, T=2.0, eps=0.1` |

## Incomplete Local Checkpoints

These files declare `repeats=10` in `params`, but fewer than 10 repeats are present in `sampled_r_mins`.

| File stem | Mode | Parameters | Completed repeats | `sampled_r_mins` |
| --- | --- | --- | ---: | --- |
| `NCC_channel_log_sampling_r_N8_T1_eps0.001_trials1000_repeats10_q4_s4` | log | `N=8, T=1.0, eps=0.001` | 1/10 | `[342]` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.001_trials1000_repeats10` | original | `N=8, T=2.0, eps=0.001` | 2/10 | `[792, 817]` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.1_trials1000_repeats10` | original | `N=8, T=2.0, eps=0.1` | 1/10 | `[104]` |

## Missing Canonical Files

Each entry means both `.json` and `.npz` are missing unless noted otherwise. Deferred `N=9` endpoints are listed in the next section.

| File stem | Roles |
| --- | --- |
| `NCC_channel_log_sampling_r_N8_T2_eps0.001_trials1000_repeats10_q4_s4` | N sweep log `N=8`; T sweep log `T=2`; epsilon sweep log `eps=0.001` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.04_trials1000_repeats10` | epsilon sweep original `eps=0.04` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.04_trials1000_repeats10_q4_s4` | epsilon sweep log `eps=0.04` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.02_trials1000_repeats10` | epsilon sweep original `eps=0.02` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.02_trials1000_repeats10_q4_s4` | epsilon sweep log `eps=0.02` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.01_trials1000_repeats10` | epsilon sweep original `eps=0.01` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.01_trials1000_repeats10_q4_s4` | epsilon sweep log `eps=0.01` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.005_trials1000_repeats10` | epsilon sweep original `eps=0.005` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.005_trials1000_repeats10_q4_s4` | epsilon sweep log `eps=0.005` |
| `NCC_channel_original_sampling_r_N8_T2_eps0.0025_trials1000_repeats10` | epsilon sweep original `eps=0.0025` |
| `NCC_channel_log_sampling_r_N8_T2_eps0.0025_trials1000_repeats10_q4_s4` | epsilon sweep log `eps=0.0025` |

## Deferred N=9 Endpoints

These are still missing, but are excluded from the active remaining-action summary.

| File stem | Roles |
| --- | --- |
| `NCC_channel_original_sampling_r_N9_T2_eps0.001_trials1000_repeats10` | N sweep original `N=9` |
| `NCC_channel_log_sampling_r_N9_T2_eps0.001_trials1000_repeats10_q4_s4` | N sweep log `N=9` |

## Notes

- `plot_four_series_scaling.ipynb` discovers channel files with `NCC_channel_*_sampling_r_*.json` and reads `N/T/epsilon/repeats` from JSON `params`.
- Prefer actual `len(sampled_r_mins)` over declared `params["repeats"]` when choosing between candidates.
- The downloaded Bohrium output in `~/Downloads/out/data/` matched and replaced the previous 1/10 `N8, T=0.8, eps=0.001` log checkpoint.
