# 接入 Triton MUSA kernel

设置 `backend: "triton"`。入口 `factory(case, inputs, workdir)` 接收设备张量，定义/选择 Triton kernel、预分配输出、确定 grid、BLOCK 等 launch 参数，返回零参数 `launch()`。launch 必须返回真实 MUSA Tensor 或列表。

需要可用于 MUSA 的 Triton 构建；普通 NVIDIA 环境中的包存在并不保证 MUSA kernel 能运行。平台将 JIT 准备与正式计时分开，记录工厂声明的 launch 参数、输出分配和编译身份。当前 `build` 不把 runtime JIT 序列化为可搬移产物，编译缓存由 Triton 管理。

多个 shape 是否复用同一产物取决于 kernel 的特化参数。需要 `launch.rebind`、运行期参数设计和实际编译身份记录；不能只因 Python 函数相同就宣称一次编译跨 shape。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst init workspace/triton-op --backend triton
python -m musaopst validate workspace/triton-op/cases.json
python -m musaopst run workspace/triton-op/cases.json --device 0 --output runs/triton-op
python -m musaopst run examples/gpu_smoke.json --case triton_softmax --benchmark --output runs/triton-softmax
```

预期报告包含独立精度比较、工厂记录的 launch 信息及 MUSA Event 样本。编辑工程时需显式维护尾部 mask、stride 与输出契约，不要隐式调用 `.contiguous()` 后声称覆盖原布局。

关键示例：[triton_ops.py](../../musaopst/examples/triton_ops.py)、[advanced_triton.py](../../musaopst/examples/advanced_triton.py)。示例数值算子主要支持 FP32/FP16/BF16，复数指针不在当前实现范围；详细支持由各入口检查。
