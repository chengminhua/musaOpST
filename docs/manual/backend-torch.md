# 接入 torch_musa 算子

设置 `backend: "torch_musa"`。普通入口为 `module:function`，按 `binding` 将设备输入和 `attrs` 传给函数；返回一个 MUSA Tensor 或 Tensor 列表/元组。输出必须位于所选设备，返回 CPU Tensor 会失败。

平台负责按真实布局上传输入、选择 stream、执行同步、输出元数据与快照、计时和清理。普通 eager callable 通常在调用内分配输出，计时会记录包含分配。若要预分配，可为入口挂载 `prepare(case, inputs, workdir)`，返回零参数 `launch()`，通过 `launch.allocation_included=False` 明确实际分配政策；该声明必须与代码一致。

独立参考使用 NumPy/CPU 代码，在独立输入副本上计算，不调用待测函数。默认先量化输入再升精度；详见[参考计算](reference.md)。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst init workspace/torch-op --backend torch_musa
python -m musaopst validate workspace/torch-op/cases.json
python -m musaopst run workspace/torch-op/cases.json --device 0 --output runs/torch-op
```

编辑生成的 `operator_impl.py` 和 `independent_reference.py` 接入自己的算子，再同步修改 `cases.json`。内置 Softmax 可直接运行：

```bash
python -m musaopst run examples/gpu_smoke.json --case torch_softmax --output runs/torch-softmax
```

接入代码参考 [torch_ops.py](../../musaopst/examples/torch_ops.py)、[advanced_ops.py](../../musaopst/examples/advanced_ops.py) 与共享 [GPUBackend](../../musaopst/backends/gpu.py)。用例报告记录实际设备、dtype、stride 和输出契约。框架 eager 能处理多个 shape，不代表已证明某个编译图跨 shape 复用。
