# NCC-Trotter-Compensation

这个仓库主要放 NCC/Trotter compensation 的数值实验、数据 checkpoint 和画图脚本。当前主线的 scaling 代码已经集中到 `scaling/` 下面。

本 README 只介绍当前目录中没有被 `.gitignore` 排除的主要文件/文件夹；本地运行缓存、Bohrium 工作目录、Overleaf 文章目录、日志和个人 memory 不在这里展开。

## 当前结构

- `scaling/`
  当前 NCC scaling 主目录，包含 active implementation、batch JSON、画图 notebook、主数据目录和少量 legacy 代码。
- `scaling/data/`
  当前 scaling 实验的主数据目录。`scaling/plot_*_scaling.ipynb` 默认从这里读取 JSON/NPZ checkpoint。
- `scaling/additonal_optimized_run/`
  NCC channel log 的 cache 优化运行脚本。目前用于把本地 `expected_r_min` 搜索和 Bohrium 上的 sampled-only 搜索拆开。
- `scaling/old_ver._rejective_samp./`
  历史 rejection-sampling 版本。这里仍有少量历史数据子目录，例如 `data_1_state/`、`data_3_17/`，但它们不是当前主数据入口。
- `figures/`
  图像输出目录。
- `scaling/plot_original_scaling.ipynb`
  从 `scaling/data/` 读取 original compensation 数据并画 scaling 图，输出到 `figures/`。
- `scaling/plot_log_scaling.ipynb`
  从 `scaling/data/` 读取 log compensation 数据并画 scaling 图，输出到 `figures/`。
- `scaling/plot_combined_scaling.ipynb`
  对比 original/log scaling，输出到 `figures/`。
- `scaling/plot_four_series_scaling.ipynb`
  汇总 operator/channel 与 original/log 四组曲线，输出到 `figures/`。
- `tests/`
  当前 scaling 代码的 smoke/unit tests。
- `1-norm/`
  Pauli sampling overhead 相关说明和 CSV 数据。
- `resource/` 与 `hncc_resource_plot.ipynb`
  HNCC resource estimate/collect 脚本、预计算数据和对应画图 notebook。其中
  `resource/hncc_resource_estimate.py` 是核心算法实现和可直接画图的 CLI；
  `resource/hncc_resource_collect.py` import 前者，只负责扫参数并把表格数据落到
  `resource/data/*.json`；`hncc_resource_plot.ipynb` 再读取这些 JSON 画图。
  注意 resource estimate 中的 expected-precision 约束不是对每个整数
  Trotter step `r` 都重新精确计算一次。它先在一组 `x_grid` 步长上精确
  计算 per-step error curve，然后用 `log_interpolate_error` 对
  `step_error(T/r)` 做 log-log 插值；若 `T/r` 落在网格外，则用边缘点做
  log-log 外推。该方法适合 resource trend 和初始 bracket 预测，但文章中
  需要精确 finite-size `expected_r_min` 时，应以
  `scaling/additonal_optimized_run/NCC_channel_expected_r_min_cached.py` 或
  `scaling/NCC_channel_find_r_min.py` 的 exact integer-r evaluation 为准。
- `requirements.txt`
  Python 依赖列表。

## 主线代码

Active scaling scripts:

- `scaling/NCC_operator_original.py`
- `scaling/NCC_operator_log.py`
- `scaling/NCC_operator_find_r_min.py`
- `scaling/NCC_channel_original.py`
- `scaling/NCC_channel_log.py`
- `scaling/NCC_channel_find_r_min.py`
- `scaling/Pauli_Hamiltonian_BCH.py`

其中：

- `operator` 版本对应算符层面的补偿与采样实现。
- `channel` 版本对应量子信道层面的补偿与采样实现。
- `original` 是固定低阶补偿方案。
- `log` 是精度相关的高阶补偿方案。

## Batch Configs

常用 batch 配置也放在 `scaling/` 下：

- `scaling/run_original.json`
- `scaling/run_log.json`
- `scaling/run_channel_original_N8_eps_sweep.json`
- `scaling/run_channel_log_N8_eps_sweep.json`

这些 JSON 文件通常设置 `out_dir` 为 `scaling/data`。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行 operator batch：

```bash
python scaling/NCC_operator_find_r_min.py --batch-file scaling/run_original.json
python scaling/NCC_operator_find_r_min.py --batch-file scaling/run_log.json
```

运行 channel batch：

```bash
python scaling/NCC_channel_find_r_min.py --batch-file scaling/run_channel_original_N8_eps_sweep.json
python scaling/NCC_channel_find_r_min.py --batch-file scaling/run_channel_log_N8_eps_sweep.json
```

长任务建议使用 unbuffered output：

```bash
PYTHONUNBUFFERED=1 python scaling/NCC_channel_find_r_min.py --batch-file scaling/run_channel_log_N8_eps_sweep.json 2>&1 | tee logs/run_channel_log.tmux.log
```

画图 notebook 位于 `scaling/plot_*_scaling.ipynb`。这些 notebook 会自动定位
`scaling/data/`，因此从仓库根目录或从 `scaling/` 目录启动 kernel 都可以；
PDF 输出仍写入仓库根目录下的 `figures/`。

## Cache 优化拆分

`scaling/additonal_optimized_run/` 中的脚本用于当前 NCC channel log 的 cache workflow：

- 本地跑 `expected_r_min` 搜索。
- Bohrium 远端跳过 expected search，只跑 sampled search。
- 两边都应先加载同一组 cache：`static_cache`、`slim_evolution_cache`、`full_evolution_cache`。

## 输出

默认输出在 `scaling/data/`：

- `.json` 保存参数、repeat 结果和搜索历史。
- `.npz` 保存数值数组，供 notebook 后处理和画图。

`find_r_min` 脚本的目标是搜索满足目标精度 `epsilon` 的最小 Trotter 步数 `r`。
