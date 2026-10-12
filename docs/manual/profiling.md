# 采集 Profiler 时间线

`profile` 使用与 `run` 相同的配置、筛选和设备参数，在正确性通过后独立采集，不将 Profiler 调用混入正式 benchmark。torch_musa、MUSA C++ 扩展、Triton、TileLang 四种 Python GPU 路径使用 `torch.profiler` 的 CPU 与 `PrivateUse1` 活动；报告必须找到真实 kernel 事件，否则返回不支持，不能将 CPU trace 当作 GPU trace。

原生独立子进程目前没有独立 GPU 采集器，不支持本功能；父进程 trace 不能替代子进程设备事件。这里不采硬件性能计数器。

用例的 `profile` 可设置 `enabled`、`warmup`、`iterations`、`record_shapes`、`profile_memory`。CLI `profile` 强制开启采集；默认预热 1 次、采集 3 次。`profile_memory` 是可用框架信息，不替代完整显存泄漏诊断。

## 使用示例

在项目根目录、具备 torch_musa Profiler 支持的环境执行：

```bash
python -m musaopst profile examples/gpu_smoke.json --case torch_softmax --device 0 --output runs/torch-profile
```

预期用例目录含 `profile_trace.json`、`profile_table.txt`，报告的 `profile` 字段包含 GPU kernel 数、名称和时长。trace 可用兼容 Chrome Trace 格式的查看器打开。

要控制采样，给配置增加以下用例字段，再运行 `profile`（以下仅是配置片段）：

```json
{"profile":{"warmup":1,"iterations":5,"record_shapes":true,"profile_memory":false}}
```

只有 CPU 活动、设备无 Profiler 支持或无实际 kernel（例如空输入）时，结果应按具体原因读取，不能因文件已生成就判定采集成功。
