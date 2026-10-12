# 常见问题与错误定位

[返回手册目录](README.md)

## 功能说明

先查看 `summary.md` 中的用例状态、最后执行阶段与日志，再打开该用例的日志和 `result.json`。构建失败、参考失败与精度失败属于不同问题，不宜统一当作 kernel 错误处理。

## 常见情况

| 现象 | 检查方式 |
|---|---|
| `No module named musaopst` | 确认当前 Python 环境已安装项目，或在源码根目录运行 |
| 找不到 torch_musa / Triton / TileLang | 使用安装了对应 MUSA 版本依赖的环境，运行 doctor；不要用 CPU 结果代替 |
| `CONFIG_ERROR` 或 CLI 参数错误 | 先运行 validate，检查 schema_version、字段名和参数组合 |
| 输出目录非空 | 换一个新的输出目录；旧结果保留用于复现 |
| `BUILD_FAILED` | 读构建日志，检查 mcc、头文件、编译参数与目标架构 |
| `REFERENCE_FAILED` | 单独检查 NumPy 参考函数、输入绑定与返回结构 |
| `ACCURACY_FAILED` | 先读 contract_errors，再看逐输出政策和失败坐标；shape/dtype 错误不能靠放宽容差解决 |
| `UNSUPPORTED` | 查看具体入口限制；例如旧向量加入口未支持跨步，或 native Profiler 无独立采集器 |
| `TIMEOUT` | 查看停在哪一阶段；首次编译可比热启动更慢，确认原因后调整 timeout_s |
| `EXPECTED_FAILURE` | 指定的负向期望满足，精度通过数仍为 0 |
| 恢复没有命中旧结果 | 配置、源码、输入、设备/软件身份或快照完整性发生变化时，会重新执行 |
| 性能 `NOISY` / `INCOMPARABLE` | 核对样本波动、设备、输入和计时政策，不能直接判定快慢 |

GPU 严重故障后工具停止后续设备验证；它不会 reset 设备，结束自身 worker 也不保证硬件恢复。

## 使用示例

以下命令会运行一个安全的 CPU 负向套件，`run` 返回非零是预期现象：

```bash
python -m musaopst validate examples/cpu_negative.json
python -m musaopst run examples/cpu_negative.json --output runs/manual-errors
python -m musaopst report runs/manual-errors --output runs/manual-errors-report
```

打开 `runs/manual-errors-report/summary.md`，对照状态和失败阶段定位问题。报告重绘不会把失败改成成功，因此也可能返回非零。

在有设备的环境中，`examples/gpu_negative.json` 用安全扰动输出验证测试器能抓错；它不会故意制造越界或死锁。日常排查优先使用 CPU 用例检查配置和比较政策，再运行一个小的 GPU 用例。
