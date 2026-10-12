# 新建算子测试工程

`init` 生成可编辑的配置、算子入口和独立参考。指定后端时生成一个 `axpby` 示例，即 `alpha * a + beta * b`；工程包含实际实现，不是空占位文件。

后端可选 `cpu`、`torch_musa`、`musa_cpp`、`triton`、`tilelang`、`native`。CPU 工程只用于平台验证。新建目标应为空目录；工具拒绝覆盖已有工程。

需要指定算子名称、shape、dtype 或模板种类时使用 `generate` 和描述文件。内建模板包括各后端 `axpby`，以及 CPU/C++/TileLang/native 的 `vector_add`、torch_musa/Triton 的 `softmax`。其他算子需要提供完整 `case`、实现入口和独立参考。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst init workspace/my_add --backend cpu
python -m musaopst validate workspace/my_add/cases.json
python -m musaopst run workspace/my_add/cases.json --output runs/my-add
```

预期生成 `cases.json`、`operator_impl.py`、`independent_reference.py`、`operator.descriptor.json`、`generation.json`、`run.sh`。C++ 和原生工程还会复制示例源文件及头文件。

也可以创建自己的描述文件：

```bash
mkdir -p workspace
cat > workspace/add.descriptor.json <<'JSON'
{"descriptor_version":1,"backend":"cpu","operation":"axpby","name":"scaled_add","shape":[257],"dtype":"float32","attrs":{"alpha":2.0,"beta":-0.5}}
JSON
python -m musaopst generate workspace/add.descriptor.json --output workspace/scaled-add
python -m musaopst run workspace/scaled-add/cases.json --output runs/scaled-add
```

对于自定义 `descriptor.case`，工具不会猜测或自动复制全部外部依赖；入口模块必须可导入，源码用 `sources` 显式声明。
