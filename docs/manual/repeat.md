# 有限重复测试

重复测试在首次精度通过后执行。每轮恢复需要恢复的输入，执行、同步、保存输出副本并与独立参考比较，首次异常即停止并保留失败快照。次数和时长是有限条件，不表示长期稳定性认证。

`repeat.count` 与 `repeat.duration_s` 默认均为 0，表示不追加重复。只设次数按次数结束，只设时长按持续时间结束；两者都大于 0 时先到任一上限即停止。CLI `--repeat` 和 `--duration` 覆盖对应字段；按时长测试时显式 `--repeat 0`，避免原配置次数先结束。整个用例仍受 `timeout_s` 限制。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst run docs/manual/examples/workflow-cpu.json --repeat 5 --output runs/repeat-count
python -m musaopst run docs/manual/examples/workflow-cpu.json --repeat 0 --duration 0.2 --output runs/repeat-duration
```

在已有 MUSA 环境可执行一个短设备例：

```bash
python -m musaopst run examples/gpu_smoke.json --case torch_softmax --repeat 5 --output runs/gpu-repeat
```

预期报告 `repeat` 字段包括请求次数/时长、实际完成次数、持续时间、异常和前后显存信息；失败时有 `repeat_failure.npz`。torch_musa/C++/Triton/TileLang 在接口可用时记录活动分配与缓存保留；native 记录 runtime free/total，它们不是同一统计量。当前 native 每次调用启动新 host 进程/context，因此不能据此证明同一个长驻 context 无泄漏。
