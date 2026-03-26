# NCC-Trotter-Compensation

这个仓库主要研究两类 NCC/Trotter compensation 版本：

- `operator` 版本
- `channel` 版本

两类版本下又各自分成两种补偿方案：

- `original`
- `log`

## 版本划分

### 1. operator 与 channel

- `operator` 版本对应算符层面的补偿与采样实现。
- `channel` 版本对应量子信道层面的补偿与采样实现。

相关脚本包括：

- `NCC_operator_original.py`
- `NCC_operator_log.py`
- `NCC_operator_find_r_min.py`
- `NCC_channel_original.py`
- `NCC_channel_log.py`
- `NCC_channel_find_r_min.py`

### 2. original 与 log

- `original` 版本对应之前 PRX Quantum 中的补偿方案，主要补偿到 2 阶和 3 阶。
- `log` 版本对应当前 note 中的补偿方案，补偿阶数随精度参数变化，直到 `log(1 / epsilon)` 量级。

可以粗略理解为：

- `original`：固定低阶补偿
- `log`：精度相关的可变高阶补偿

## `find_r_min` 的作用

`find_r_min` 系列脚本用于搜索在给定目标精度 `epsilon` 下，达到误差要求所需的最小 Trotter 步数 `r`。

这里的核心做法是：

- 对候选 `r` 做误差评估
- 用二分搜索定位最小可行 `r`
- 同时记录采样误差、期望偏差以及重复实验统计结果

对应脚本包括：

- `NCC_find_r_min.py`
- `NCC_operator_find_r_min.py`
- `NCC_channel_find_r_min.py`

## 批量运行

仓库中提供了一些 JSON 配置文件用于批量运行不同参数组合，例如：

- `run.json`
- `run_original.json`
- `run_log.json`
- `run_log_only.json`

这些文件通常按 case 列出不同的：

- `mode`
- `N`
- `T`
- `epsilon`
- 以及部分 `log` 版本使用的 `q0`、`s0`

脚本会按配置顺序逐个运行 case，并把结果写入 `data/` 目录。

## 用法介绍

### 1. 安装依赖

先安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 单个 case 运行

如果只想运行单个参数点，可以直接传命令行参数。

例如运行 operator/original 版本：

```bash
python NCC_find_r_min.py --mode original --N 8 --T 2.0 --epsilon 0.001 --trials 1000 --repeats 10 --r-max 1024
```

例如运行 operator/log 版本：

```bash
python NCC_find_r_min.py --mode log --N 8 --T 2.0 --epsilon 0.001 --q0 6 --s0 6 --trials 1000 --repeats 10 --r-max 1024
```

例如运行 channel 版本：

```bash
python NCC_channel_find_r_min.py --batch-file run_original.json
```

### 3. 读取 JSON 批量实验

现在更常用的方式是通过 JSON 文件一次性描述一批实验，再让脚本顺序执行。

例如：

```bash
python NCC_find_r_min.py --batch-file run.json
```

或只跑 log 部分：

```bash
python NCC_find_r_min.py --batch-file run_log_only.json
```

channel 版本类似：

```bash
python NCC_channel_find_r_min.py --batch-file run_original.json
```

JSON 文件的基本结构是：

```json
{
  "defaults": {
    "out_dir": "data",
    "trials": 1000,
    "repeats": 10,
    "r_max": 1024
  },
  "cases": [
    { "mode": "original", "N": 8, "T": 0.2, "epsilon": 0.001 },
    { "mode": "log", "N": 8, "T": 2.0, "epsilon": 0.001, "q0": 6, "s0": 6 }
  ]
}
```

其中：

- `defaults` 给出一批实验共享的默认参数
- `cases` 给出每一个具体实验点
- `mode` 指定是 `original` 还是 `log`
- `N`、`T`、`epsilon` 是主要扫描参数
- `q0`、`s0` 主要用于 `log` 版本

### 4. 推荐的带日志运行方式

如果实验很长，建议配合 `tmux` 和日志文件运行，例如：

```bash
tmux new -s run_log
PYTHONUNBUFFERED=1 python NCC_find_r_min.py --batch-file run_log_only.json 2>&1 | tee logs/run_log.tmux.log
```

channel 版本也类似：

```bash
tmux new -s run_channel
PYTHONUNBUFFERED=1 python NCC_channel_find_r_min.py --batch-file run_original.json 2>&1 | tee logs/run_channel.tmux.log
```

这样可以：

- 在后台长期运行实验
- 实时查看输出日志
- 避免因为终端断开而中断任务

## 输出说明

输出结果通常保存在 `data/` 目录中，主要包括：

- `json`：保存运行参数、每次 repeat 的结果、搜索历史等
- `npz`：保存数值数组，便于后处理和画图

## 说明

如果只想抓住这个仓库的主线，可以记成下面三点：

1. 分成 `operator` 和 `channel` 两个大版本。
2. 每个版本里又分成 `original` 和 `log` 两种补偿方案。
3. `find_r_min` 脚本负责用二分搜索找出达到目标精度所需的最小 `r`。
