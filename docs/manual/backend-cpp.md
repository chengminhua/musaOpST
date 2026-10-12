# 接入 MUSA C++ 扩展

设置 `backend: "musa_cpp"`。入口是 `factory(case, inputs, workdir)`，其中 `inputs` 已是保持所声明布局的 MUSA Tensor 结构；工厂编译/加载扩展、准备输出，再返回零参数 `launch()`。每次 launch 调用绑定的 C++ 接口并返回实际 MUSA 输出。

平台不猜测任意 `.so` 的函数名或 ABI。扩展需要明确的 Python 绑定、源码/头文件、编译参数和 launch 信息。内建实现通过 `build_cpp_descriptor` 管理缓存，`load_cpp_artifact` 校验加载。支持独立构建的工厂需挂载 `build_descriptor(case, workdir)`；准备结果可由 `run --prepared` 执行。

预分配输出可将 `launch.allocation_included=False`；支持运行期 shape 需实现 `launch.rebind(new_inputs, new_case)` 并记录产物身份，不能将重新编译称为复用。

## 使用示例

在项目根目录、具有对应 SDK/torch_musa/主机编译器的环境执行：

```bash
python -m musaopst init workspace/cpp-op --backend musa_cpp
python -m musaopst validate workspace/cpp-op/cases.json
python -m musaopst run workspace/cpp-op/cases.json --device 0 --output runs/cpp-op
```

预期工程包括配置、Python 工厂、独立参考、`.mu`、绑定 `.cpp` 和头文件；报告记录构建摘要、缓存命中、实际输出及计时政策。仅构建和加载已有产物见[独立构建](build.md)。

关键示例：[advanced_cpp.py](../../musaopst/examples/advanced_cpp.py)、[advanced_cpp.mu](../../musaopst/examples/advanced_cpp.mu)、[advanced_cpp_bindings.cpp](../../musaopst/examples/advanced_cpp_bindings.cpp)。示例 kernel 用于说明接入，不是性能调优实现；dtype/rank/布局限制由工厂显式检查。
