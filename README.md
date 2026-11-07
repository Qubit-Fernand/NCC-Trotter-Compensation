# Project Memory

**项目目录：** 对应的文稿位于同级目录 [../overleaf/](../overleaf/)，两者是独立 Git 仓库。

- Tomorrow TODO (2026-07-13): investigate and optimize the channel exact-error
  bottleneck. Start by locating Xinzhao's WeChat discussion with Shuo about why
  the resource-estimate error calculation is very fast while the current
  256-basis channel error calculation is slow, even though both are ultimately
  maximizing the trace norm expression. Hypothesis to verify: the resource code
  amortizes the expensive expression over a precomputed step-size grid and then
  uses interpolation/bracketing, whereas the channel expected-r search recomputes
  exact integer-r outputs over all 256 computational-basis states. Optimize this
  path without changing the claimed exact finite-size result, or explicitly
  separate a fast resource-style bracket from a final exact verifier.
- 2026-07-10 Bohrium original-channel expected-only checkpoint: for project
  `240827`, use the temporary new `ACCESS_KEY` from this memory file only in
  the command environment; the default shell key gives `Permission error` for
  these jobs. Commands used for inspection were
  `ACCESS_KEY="$NEWKEY" bohr job list -r -n 20 --json`,
  `ACCESS_KEY="$NEWKEY" bohr job describe -j 23000681 -j 23000682 -j 23000680`,
  and `ACCESS_KEY="$NEWKEY" bohr job log -j ... -o
  .bohr/results/Bohrium_inspect_running/original_N8_expected_dump_20260708_logs`.
  The three `original_N8_expected_dump_20260708` jobs were still `Running` after
  about 27.7 hours and had log tails stuck at expected-batch construction:
  eps `0.01` job `23000681` has observed `r=80` pass, `r=64` fail, `r=72`
  barely fail, then stuck at `r=76`; adopt `expected_r_min=73` per discussion.
  eps `0.005` job `23000682` has `r=99` pass, `r=96` fail, then stuck at
  `r=97`; adopt `expected_r_min=99` per discussion. eps `0.0025` job
  `23000680` has `r=125` pass and `r=112` fail, then stuck at `r=118`; record
  only an estimate, likely `r_min≈122` with working interval `120--125`. The
  partial/provisional data was written to
  `scaling/data/NCC_channel_original_expected_r_min_N8_T2_bohrium_partial_20260710.json`.
- 2026-07-08 channel-pairing theta bug [resolved 2026-07-09]: in the scaling code, the Pauli
  decomposition and `eta_pair_sum` values are correct; the bug is the
  conversion from `eta` to the Pauli-pairing angle `theta`. The old channel
  sampling code in `scaling/NCC_channel_original.py` and
  `scaling/NCC_channel_log.py` solved `sin(theta)/cos(theta)^2 = eta`, but
  the actual signed-channel sampler with branches
  `U_theta = cos(theta) I + i sin(theta) P` and `-P(.)P` has expectation
  `I + i tan(theta) ad_P`. Therefore strict unbiased sampling of
  `I + i eta ad_P` requires `tan(theta) = eta`. The fix changed both active
  channel implementations to `theta = arctan(eta_pair_sum)` and changed
  validation to enumerate the actual paired sampled-branch expectation. Data
  provenance warning: existing sampled LCQC data under `scaling/data/` that
  was generated before this fix still contains the paired-branch bias
  `(cos(theta)-1)` times the low-order linear term. Deterministic/expected-map
  curves from those runs are still correct for the intended truncated map,
  because the old validation and expected path used the hand-written target
  expectation rather than the biased finite-sample branch expectation.
- 2026-07-07 resource-estimate vs exact expected-r note: the resource
  expected-precision constraint is not an exact re-evaluation for every
  integer Trotter step `r`. It precomputes per-step error values on `x_grid`
  and uses `log_interpolate_error` to evaluate `step_error(T/r)` by log-log
  interpolation, or log-log extrapolation outside the grid. This is reasonable
  for resource trends and for predicting an initial bracket, but exact
  finite-size `expected_r_min` claims should be verified with
  `scaling/additonal_optimized_run/NCC_channel_expected_r_min_cached.py` or
  `scaling/NCC_channel_find_r_min.py`, which perform exact integer-r
  evaluations. Current observed mismatch can also come from product-formula
  order and truncation-order differences, e.g. resource HNCC2/S2 q=s=6 vs
  channel-log q=s=4 or first-order variants.
- 2026-07-06 afternoon plan: use
  `scaling/additonal_optimized_run/` for the NCC channel log split workflow.
  Run `expected_r_min` locally with the expected-only script, and submit the
  Bohrium side as sampled-only. Both local expected-only and remote sampled-only
  runs must load all three caches first: `static_cache`, `slim_evolution_cache`,
  and `full_evolution_cache`.
- Bohrium sampled-only/cache-dump submit rule: for later cache-based Bohrium
  tasks, stage and load the three cache inputs in order:
  `static_cache`, then `slim_evolution_cache`, then `full_evolution_cache`.
  The current canonical NCC_channel_log cache files live in the repository-root
  `cache/` directory, not under `.bohr/cache/`.
  When using a dump-cache script, keep `--cache-update-dir` enabled; those
  dumped evolution-cache updates are intended to preserve cache entries created
  during the run and accelerate future tasks.
- Sample-only cache minimization note: sampled `r_min` decisions only require
  the sampled average channel output minus the exact total-evolution output on
  the 256 computational-basis states. They do not require
  `deterministic_output_batch`; that field is only used for logging
  expectation bias/fluctuation. For future sampled-only Bohrium runs, prefer
  `rebuild_full_evolution_cache_flexible.py --cache-profile sample-baseline`
  to store `exact_output_batch` while omitting deterministic/basis/step/V
  fields, or `--cache-profile sample-minimal` to store only sampling data plus
  `U_exact` and build `exact_output_batch` at runtime. The local
  `NCC_channel_find_r_min_sample_only_cache*.py` wrappers under
  `.bohr/active/log_N8_eps_split_cache_20260705/cache_assets/` use this
  exact-output baseline only and report `expectation_bias/sample_fluctuation`
  as `NaN` during sampled-only searches. The dump-cache wrapper also prunes
  `deterministic_output_batch`, basis labels/states, step-exact, and V-exact
  fields from cache-update files; dumped updates keep `exact_output_batch`.
- Original-channel sample-baseline cache rule: the same minimization applies to
  `NCC_channel_original`. The static cache in
  `.bohr/cache/NCC_channel_original_cache_N8/NCC_channel_original_cache_N8.pkl.gz`
  uses original static keys `(N, J, h)`, e.g. `(8, 1.0, 1.0)`, because static
  data is independent of total time. Evolution caches derived from it must use
  the cleaner channel evolution key `(N, T, r)`, e.g. `(8, 2.0, r)`. For
  sampled-only original jobs, build `sample-baseline` entries that keep
  `exact_output_batch` plus sampling fields and prune
  `deterministic_output_batch`, `basis_states`, `basis_labels`, callables,
  `step_exact`, and `V_exact`. The local builder and note for this are in
  `.bohr/cache/NCC_channel_original_cache_N8/build_original_evolution_cache.py`
  and `.bohr/cache/NCC_channel_original_cache_N8/README.md`.
- Static cache reuse rule: `static_cache` is independent of total evolution
  time `T` and Trotter step count `r`; it only depends on the static
  Hamiltonian and mode-specific static parameters. For `original`, the key is
  effectively `(N, J, h)`. For `log`, the key is effectively
  `(N, q0, s0, J, h, K)`. In contrast, `evolution_cache` depends on
  `(N, T, r)` and should not be reused across different `T` unless explicitly
  rebuilt or verified.
- Scaling code/data layout after 2026-07-06 cleanup: active NCC scaling
  implementations and batch configs live under `scaling/`; stored scaling
  checkpoints live under `scaling/data/`; plotting notebooks read from
  `scaling/data/`; local optimized/cache helper scripts live under
  `scaling/additonal_optimized_run/` (directory name intentionally currently
  uses this spelling).
- 我有两个 Bohrium 账号，其中中国手机号对应的 accesskey 是 [REDACTED]，这个是写进系统 zshrc 的你已经配置好提交任务用了很多回的。但是余额已经是负数了。
- 你提交 加载 cache 版本的那 5 个 split 任务的时候，请考虑使用我美国手机号注册的新的 Access Key [REDACTED]，这个创立了一个 id 为 240827 的 project 供你使用。是临时替换设置，还是暂时覆盖一下系统环境变量名看你。切记提交的时候加载 static cache 和 evolution cache。
- Related Overleaf manuscript project:
  `/Users/AntiEntropy/Documents/Research/NCC Trotter Compensation/overleaf`
- Current code and notebook repository:
  `/Users/AntiEntropy/Documents/Research/NCC Trotter Compensation/Code`
- `.bohr` layout after cleanup: top-level entries are `active/`,
  `archive/`, `cache/`, `results/`, and `submitted/` plus `README.md`.
  Use `active/` for current staging and rebuild work, `.bohr/cache/` for
  legacy cache rescues, original-channel caches, and cache helper scripts,
  `submitted/` for historical Bohrium input directories, and `archive/` for
  old debug/smoke/metadata artifacts. Current NCC_channel_log handoff caches
  live in repository-root `cache/` (`static_cache.pkl.gz`,
  `slim_evolution_cache.pkl.gz`, `full_evolution_cache.pkl.gz`, and
  `evolution_cache_eps_*.pkl.gz`). Bohrium
  result files live under `.bohr/results/`: merged usable data is in
  `.bohr/results/data/`, partial data is in
  `.bohr/results/data_unfinished/`, downloaded Bohrium bundles and their
  in-place extracted contents are in `.bohr/results/Bohrium_downloaded/<jobid>/`,
  and old running-job inspection snapshots are in
  `.bohr/results/Bohrium_inspect_running/`. When unpacking an `out.zip`,
  extract it in place under the matching `Bohrium_downloaded/<jobid>/` directory rather
  than using separate raw/unzipped trees. Historical submitted task directories
  under `.bohr/submitted/` use a date prefix, e.g.
  `20260514_log_N8_T_sweep`. The `inspect` directories are provenance/log
  snapshots rather than primary numerical results.
- A100 cache rescue on 2026-07-05: extracted `NCC_channel_log`
  `EVOLUTION_CACHE` from the running log task
  `.venv/bin/python NCC_channel_find_r_min.py --batch-file run_log_eps_reverse_20260701.json`
  on remote PID `3279512`.
  Local files:
  `.bohr/cache/20260705_A100_cache_rescue/evolution_cache_dump_log_pid3279512_20260705_180633.pkl.gz`
  and
  `.bohr/cache/20260705_A100_cache_rescue/evolution_cache_live_summary_log_pid3279512_20260705_181254.json`.
  The summary reports 19 cache entries for `(N, T, r)` keys with `N=8`,
  `T=2.0`, and `r` values:
  `1, 2, 4, 8, 16, 20, 21, 22, 24, 32, 64, 96, 112, 120, 124, 125, 126, 127, 128`.
  The remote uncompressed 4.3G dump was deleted after the local gzip passed
  `gzip -t`; the running process remained alive.
- A100 static cache rescue on 2026-07-05: extracted `NCC_channel_log`
  static data from the same live log process `3279512`. Local file:
  `.bohr/cache/20260705_A100_cache_rescue/static_cache_NCC_channel_log_N8_q4_s4_J1_h1_K1_20260705_185310.pkl.gz`.
  It was validated locally with `pickle.load`; key is
  `(8, 4, 4, 1.0, 1.0, 1)` for `(N, q0, s0, J, h, K)`. A copy for Bohrium
  staging lives at
  `.bohr/active/log_N8_eps_split_cache_20260705/cache_assets/static_cache_NCC_channel_log_N8_q4_s4_J1_h1_K1.pkl.gz`.
- Bohrium cache-submit lesson from 2026-07-06: for the `NCC_channel_log`
  `N=8, T=2.0, q0=s0=4` eps-split jobs, do not upload the whole active
  `cache_assets/` directory blindly. It contains several duplicate 486M
  partial/full rebuild files. Uploading the 486M partial full cache is not a
  mistake when the task intentionally wants low-`r` full-batch reuse; the
  mistake is accidentally uploading multiple redundant partial/full snapshots.
  For the slim-cache Bohrium test, the correct minimal input `cache_assets/` is
  the three-file asset set under
  `.bohr/active/log_N8_eps_split_cache_20260705/submit_eps004_slim_project240827_assets/cache_assets/`:
  `NCC_channel_find_r_min_cached.py`,
  `static_cache_NCC_channel_log_N8_q4_s4_J1_h1_K1.pkl.gz` (~20M), and
  `evolution_cache_NCC_channel_log_N8_T2_pid3279512.pkl.gz` (~122M). The whole
  task input directory is about 142M and uploads successfully.
- Bohrium eps `0.01` slim-cache submission on 2026-07-06 used the newer
  project `240827` and temporary `ACCESS_KEY=...` only for the command
  environment, together with the skill wrapper
  `/Users/AntiEntropy/.codex/skills/bohrium-batch-jobs/scripts/bohr_direct.py`
  to avoid VPN/TUN route timeouts. It created local task list
  `.bohr/active/log_N8_eps_split_cache_20260705/tasks_eps001.jsonl` and
  submitted from
  `.bohr/active/log_N8_eps_split_cache_20260705/submit_eps001_slim_project240827/task_000001`.
  The submitted job is `JobId=22985852`, CLI/web `JobGroupId=16334603`,
  and the local `job_group_bid` returned by group creation is `7205339`.
  The job name is `log_N8_eps001_slim_cache_1`, machine type `c8_m64_cpu`.
  Immediate `bohr job list -j 16334603 --json` showed status `Running` and
  message `Preparing input files and computation environment`.
- Practical command pattern for these slim-cache submits: first run a dry run
  with `bohrium_batch.py`, inspect `du -sh task_000001`, `job.json`,
  `task.sh`, and `gzip -t` both cache files. Then submit with the same helper
  using `--extra-dir .bohr/active/log_N8_eps_split_cache_20260705/submit_eps004_slim_project240827_assets/cache_assets`,
  `--project-id 240827`, `--machine-type c8_m64_cpu`, and the run command
  `PYTHONUNBUFFERED=1 python cache_assets/NCC_channel_find_r_min_cached.py --batch-file params.json --static-cache cache_assets/static_cache_NCC_channel_log_N8_q4_s4_J1_h1_K1.pkl.gz --evolution-cache cache_assets/evolution_cache_NCC_channel_log_N8_T2_pid3279512.pkl.gz --require-static-cache --require-evolution-cache 2>&1 | tee job.log; tar -czf data.tar.gz data_log_eps_split_cache_20260705`.
- Bohrium eps `0.005` fullpartial input was prepared and submitted on
  2026-07-06. Task list:
  `.bohr/active/log_N8_eps_split_cache_20260705/tasks_eps0005.jsonl`.
  Submitted input:
  `.bohr/active/log_N8_eps_split_cache_20260705/submit_eps0005_fullpartial_project240827_dryrun/task_000001`.
  It is about 628M and intentionally includes both the 122M slim evolution
  cache and the 486M partial full evolution cache, plus the 20M static cache.
  It uses `cache_assets/NCC_channel_find_r_min_multi_cache.py`, loading slim
  first and the partial full cache second:
  `--evolution-cache cache_assets/evolution_cache_NCC_channel_log_N8_T2_pid3279512.pkl.gz --evolution-cache cache_assets/evolution_cache_FULL_NCC_channel_log_N8_T2_partial_r1_2_4_8_16_20.pkl.gz`.
  `gzip -t` passed for all three cache files and the wrapper `--help` works
  from inside the generated task directory. The submitted job is
  `JobId=22985868`, CLI/web `JobGroupId=16334608`, and local `job_group_bid`
  `7205342`; immediate `bohr job list -j 16334608 --json` showed status
  `Running` with message `Preparing input files and computation environment`.
- Full evolution cache rebuild status for `NCC_channel_log`, `N=8`, `T=2.0`,
  `q0=s0=4`: Mac mini completed the powers/full cache at
  `.bohr/log_N8_eps_split_cache_20260705/cache_assets/evolution_cache_FULL_NCC_channel_log_N8_T2_powers_of_two.pkl.gz`.
  Its summary reports full entries for
  `r = 1, 2, 4, 8, 16, 20, 32, 64, 128`, size `802711485` bytes, sha256
  `e84d579dc9d68523a478b81b727286499e4fcc4f6b372e1632e65849abb6a09b`.
  The adjacent `.partial.pkl.gz` is a checkpoint/duplicate of the same rebuild
  family and should not be mistaken for the active canonical final file.
- Supplemental local full-cache rebuild on 2026-07-06 produced
  `.bohr/active/log_N8_eps_split_cache_20260705/cache_assets/evolution_cache_FULL_NCC_channel_log_N8_T2_r24_48_20260706.pkl.gz`
  with keys `r = 24, 48`, size `195601913` bytes, sha256
  `9b4e351b0a5e51c051d6aebd7776985d5cef468d6c7f11cf12ab68c9c89bd3a1`.
  `r=24` came from the slim evolution cache; `r=48` was built from static data
  with `build_tilde_V`. These files are intended for later manual merging with
  the Mac mini powers cache.
