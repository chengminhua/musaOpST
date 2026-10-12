# 接入 TileLang MUSA kernel

设置 `backend: "tilelang"`。入口 `factory(case, inputs, workdir)` 使用 MUSA 目标编译 TileLang kernel，明确输出缓冲区与 launch 参数，返回零参数 `launch()`。返回值须为实际 MUSA Tensor；独立参考仍由统一参考接口完成。

需要安装支持 MUSA 的 TileLang 及其配套编译依赖。当前独立 `build` 不支持仅靠 Python callable 生成可搬移 JIT 产物；准备和 JIT 在执行 worker 中完成。工厂应在准备阶段完成必要编译，再进入正式计时。

内置普通示例与专门的 `axpby_dynamic` 有不同能力：后者使用符号 N、要求两个同 dtype 的连续一维张量，并保持编译期属性不变。源码摘要加保留 JIT 对象只能证明所记录的复用范围，不能等同于获取了最终设备二进制摘要。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst init workspace/tilelang-op --backend tilelang
python -m musaopst validate workspace/tilelang-op/cases.json
python -m musaopst run workspace/tilelang-op/cases.json --device 0 --output runs/tilelang-op
python -m musaopst run examples/gpu_smoke.json --case tilelang_vector_add --benchmark --output runs/tilelang-add
```

预期生成可编辑 factory 与独立参考；运行报告包含实际 shape/stride、比较结果、编译元数据和计时政策。

关键示例：[tilelang_ops.py](../../musaopst/examples/tilelang_ops.py)、[advanced_tilelang.py](../../musaopst/examples/advanced_tilelang.py)。新 kernel 应明确 bounds check、输出索引/分配和布局限制；当前内置例不是任意 rank、复数或任意动态 shape 的通用适配器。
