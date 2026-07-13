# Expected-Channel Workflows

This directory is reserved for precision-only expected-channel searches and
related helpers. Use it for scripts or notes that evaluate deterministic
expected-map error without sampled trajectory runs.

Keep existing production scripts in place until their cache paths and result
provenance have been checked.

## One-Step Proxy Search

`NCC_channel_step_proxy_r_min_cached.py` searches for the smallest `r` satisfying

```text
r * max_z ||E_step(T/r)(|z><z|) - U_step(T/r)(|z><z|)|| <= epsilon.
```

It reuses the `NCC_channel_log` static/evolution caches and supports
`--norm trace` or `--norm frobenius`.

Canonical trace-proxy checkpoints live directly in this directory, using sweep
role suffixes such as `_time`, `_eps`, `_eps_T1`, `_N`, and `_N_T1`. Raw
machine- or batch-specific outputs are kept under `raw/`. See `manifest.md` for
the source host and raw checkpoint behind each canonical file.
