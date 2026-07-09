# Channel Epsilon Sweep Cache Workflow Notes

Generated: 2026-07-08

This note records the provenance convention for the `N=8` channel epsilon-sweep
data in this directory, plus a few closely related channel checkpoints completed
with the same split sampled-only workflow.

## Channel Log Sweep

For `NCC_channel_log`, the `eps=0.1` point is the ordinary completed sampled
run. The other epsilon-sweep points are being completed through a cache-rescue
workflow:

- recover or merge evolution-cache entries produced during previous Bohrium
  sampled searches;
- run expected `r_min` search separately when needed;
- run sampled-only jobs separately, loading existing caches before probing new
  `r` values;
- dump newly built lightweight evolution-cache entries during the sampled-only
  run for later reuse.

The intended cache loading order for sampled-only jobs is:

```text
static_cache.pkl.gz
slim_evolution_cache.pkl.gz
full_evolution_cache.pkl.gz
per-epsilon additional evolution cache, if available
```

The per-epsilon additional cache is not a new physical model assumption. It is
only a computational reuse artifact: it stores previously built evolution-cache
entries for selected `r` values so later sampled searches do not rebuild them.

## Splitting Repeats By Shifting The Seed

The sampled runs use deterministic pseudo-random seeds. A nominal
`repeats=10`, `seed=7` run can be split into several independent jobs by
starting later jobs from the seed that the original script would have used for
that repeat.

The seed convention used in the sampled-only continuation jobs is:

```text
seed_for_repeat_k = 7 + (k - 1) * 100003
```

Here `k` is the 1-based repeat index in the original 10-repeat run. For example:

```text
repeat 1 starts from seed 7
repeat 4 starts from seed 300016
repeat 7 starts from seed 600025
```

Thus a partial original run can be completed without changing the statistical
experiment:

```text
original repeats 1-3: already available
continuation job for repeats 4-6: repeats=3, seed=300016
continuation job for repeats 7-10: repeats=4, seed=600025
```

When merging, concatenate the sampled results in original repeat order and keep
the per-repeat `sampled_r_mins`, searched `r` values, and metadata aligned. The
split jobs may load additional evolution caches and may dump new cache entries,
but the random samples are controlled by the shifted seed schedule above.

## Combining Expected Search With Sampled Runs

For the expensive channel epsilon-sweep points, the final dataset may be
assembled from two logically different computations:

```text
expected-only search: determines the deterministic expected r_min and related
                      expected-error metadata;
sampled-only runs:    determine sampled_r_mins over multiple random repeats.
```

The expected-only part and sampled-only part can be run in separate jobs because
they share the same model parameters but produce different fields. The final
merged JSON/NPZ should combine them into one record for the same case:

```text
N, T, epsilon, trials, repeats, q0, s0
expected r_min / expected-error search trace
sampled_r_mins and sampled search traces for all repeats
```

When doing this merge, treat the expected result as a case-level deterministic
quantity and the sampled results as repeat-level stochastic quantities. Do not
duplicate the expected search once per repeat. The sampled repeats should be
ordered by their original repeat index, including any pieces recovered by the
seed-shift continuation rule above.

This split is also a practical optimization over the early `find_r_min` scripts.
Those scripts called a combined `search_r_min` once per sampled repeat; inside
each call, the deterministic expected search was run before the stochastic
sampled search. With a shared evolution cache this did not usually rebuild every
heavy object, but it still repeated the same expected-search control flow and
could recompute deterministic error quantities. The cost depends on what the
cache stores:

```text
operator cache with deterministic bias or operators:
    expected search is mostly scalar lookup or a small spectral-norm check;

channel cache with exact_output_batch and deterministic_output_batch:
    expected search still needs a trace-norm sweep over basis-state outputs,
    but avoids rebuilding the deterministic propagated batch;

slim channel cache without deterministic_output_batch:
    expected search may rebuild deterministic output batches and can become
    much slower than a pure cached lookup.
```

Therefore the later workflow of running expected-only once, running sampled-only
for all repeats, and merging the deterministic expected result into every
repeat-level sampled record is the cleaner and usually faster layout.

The resulting promoted file can use the canonical unsuffixed name once it has
the intended expected fields and all sampled repeats. Partial sampled-only
checkpoints should keep the `_standard_sampled` suffix until they are promoted.

## Channel Original Sweep

The `NCC_channel_original` epsilon sweep may use the same split workflow:
expected search and sampled-only search can be run separately, with rescued or
merged evolution-cache entries used as acceleration. Detailed provenance for
those points should be filled in once the final original-channel cache and
sample outputs are merged.

Placeholder:

```text
TODO: record final original-channel eps-sweep cache sources, brackets, and
merged sampled output names.
```

## Current Standard Sampled Checkpoints

Some current channel-log and channel-original epsilon-sweep files may be partial
checkpoints with only 1, 2, or 3 completed repeats, even if the filename says
`repeats10`. These current standard sampled checkpoints are marked explicitly
with the suffix when they are intentionally kept as non-canonical partial
checkpoints:

```text
..._standard_sampled.json
..._standard_sampled.npz
```

At the time of this note, the special standard sampled checkpoints are:

```text
NCC_channel_log_sampling_r_N8_T2_eps0.001_trials1000_repeats10_q4_s4_standard_sampled
NCC_channel_log_sampling_r_N8_T1_eps0.001_trials1000_repeats10_q4_s4
NCC_channel_original_sampling_r_N8_T2_eps0.001_trials1000_repeats10_standard_sampled
NCC_channel_original_sampling_r_N8_T2_eps0.1_trials1000_repeats10_standard_sampled
```

This suffix makes it clear that these files are the current standard sampled
checkpoints, while leaving the unsuffixed canonical names available for future
fully merged or promoted outputs.

Additional provenance for `NCC_channel_log_sampling_r_N8_T1_eps0.001_trials1000_repeats10_q4_s4`:

```text
repeats 1-6: early-version run, computed together before the split workflow
             was introduced;
repeats 7-10: A100 sampled-only continuation, using static cache only because
              the available evolution caches are for T=2.0 and must not be
              loaded for this T=1.0 case.
```

The A100 continuation uses the seed-shift rule with `seed=600025`,
`repeats=4`, and the bracket `(256, 768]` with first probe `r=384`, chosen from
the first six repeats' search trajectories.

## Sample Bracket Guide

The following bracket guide is copied from
`.bohr/cache/SAMPLE_BRACKET_GUIDE.md`.

### NCC Channel Log Sample Bracket Guide

This note records rough sampled `r_min` brackets for `N=8`, `T=2.0`,
`q0=s0=4`, based on the two currently confirmed channel-log sampled points:

```text
eps=0.1    sampled r_min ~= 129
eps=0.001  sampled r_min ~= 1126
```

Use these as submission guides, not final claims.

### Poly / Log-Log Fit

Approximate fit:

```text
r ~= C * eps^(-0.47)
```

```text
eps       r estimate   recommended bracket   first sample
0.04      192          (160, 256]            192 or 208
0.02      280          (256, 384]            320
0.01      386          (320, 512]            384
0.005     545          (448, 768]            576
0.0025    772          (640, 1024]           768 or 832
```

### Linear In log(1/eps) Fit

Approximate fit:

```text
r ~= -369.5 + 216.5 log(1/eps)
```

```text
eps       r estimate   recommended bracket   first sample
0.04      327          (256, 384]            320
0.02      477          (384, 512]            448
0.01      627          (512, 768]            640
0.005     777          (640, 896]            768
0.0025    928          (768, 1024]           896
```

### Practical Rule

For Bohrium sampled-only jobs, prefer the linear-in-log brackets when we want a
more conservative starting interval. The sample-only dump-cache wrapper supports
optional case fields:

```json
"sample_low": 256,
"sample_high": 384
```

The interval semantics are `(sample_low, sample_high]`.

If the bracket is wrong, the wrapper automatically repairs it:

- If `sample_high` is still too small, it expands right by doubling until the
  target is reached or `r_max` is hit.
- If `sample_low` already satisfies the target, it expands left by halving until
  the lower side is outside the target.

The wrapper still dumps newly built sample-baseline evolution-cache entries via
`--cache-update-dir`, so bracket probes can be reused later.
