# 恢复、失败重跑和性能回归

`run --resume` 校验旧 PASS 记录的配置、工具与入口源码、输入、环境身份和必需快照摘要。满足条件才复用，报告区分 `reused` 与本次新执行；身份未知、变更或产物损坏会重新执行。`--rerun-failed` 必须配合 `--resume`，通过可验证旧 PASS 复用实现失败项重跑，而不是盲信旧状态。

`report --baseline` 比较两个已有性能报告。设备、工作负载、软件和计时政策须一致，两边至少各 5 个样本；IQR/中位数超过 20% 标为 `NOISY`。中位数涨幅超过 `--threshold-percent` 才判 `REGRESSION`。不可比较、噪声、缺样本不能判作性能通过。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst run docs/manual/examples/workflow-cpu.json --output runs/regression-first
python -m musaopst run docs/manual/examples/workflow-cpu.json --resume runs/regression-first --output runs/regression-resumed
python -m musaopst run docs/manual/examples/workflow-cpu.json --resume runs/regression-first --rerun-failed --output runs/regression-rerun
```

结果检查 `summary.reused`、`summary.fresh_gpu_passed` 和用例的 `provenance`，不能把复用数当作新 GPU 通过数。

在同一设备环境中执行两次相同 GPU 性能配置，例如：

```bash
python -m musaopst run examples/gpu_smoke.json --case torch_softmax --benchmark --output runs/perf-baseline
python -m musaopst run examples/gpu_smoke.json --case torch_softmax --benchmark --output runs/perf-current
python -m musaopst report runs/perf-current --baseline runs/perf-baseline --threshold-percent 10 --output runs/perf-comparison
```

产物为 `baseline.json`、`baseline.md`。基线比较退出码：全部 `OK` 为 0，出现退化为 1，其余不确定/不可比较为 2。该门禁只说明所测条件，不自动消除其他任务占用及设备频率变化。
