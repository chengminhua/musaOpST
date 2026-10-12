# 动态 shape

[返回功能手册](README.md)

动态 shape 需要分清两件事：从配置展开多个具体 shape，以及让同一个准备好的执行对象接受多个 shape。仅有 `${N}` 参数化不能证明动态编译能力，逐个重新编译也不等于复用同一产物。

## 字段与执行规则

| 字段 | 含义 |
|---|---|
| `symbols` | 符号名称到有限非负整数列表的映射 |
| `shape_mode: "specialize"` | 默认方式，每个展开用例独立准备执行 |
| `shape_mode: "runtime"` | 同一动态组使用同一个新 worker，先 prepare，再对后续输入 rebind |
| `compile_group` | runtime 必填；相同组名表示共享执行会话 |

组内用例必须连续排列，使用相同 backend、entry、attrs、配置目录及 stream 政策，不能跳过组内单例。dtype、rank、布局能否改变取决于适配器；不能仅靠填写相同组名保证任何变化都被支持。动态组不会按单条旧结果恢复，以免丢失本次产物复用证据。

## CPU 平台流程示例

[core-dynamic-shapes.json](examples/core-dynamic-shapes.json) 对 N=1、17、513 运行同一个 CPU identity callable：

```bash
python -m musaopst validate docs/manual/examples/core-dynamic-shapes.json
python -m musaopst run docs/manual/examples/core-dynamic-shapes.json --output runs/manual-dynamic-shapes
```

预期 3 个 CPU `PASS`，`dynamic_execution.same_worker` 为 true，`prepare_calls` 为 1。这只验证平台分组与重绑定逻辑，不证明任何 GPU 编译器能力。

其中关键配置片段为：

```json
{
  "symbols": {"N": [1, 17, 513]},
  "inputs": [{"shape": ["${N}"], "dtype": "float32"}],
  "outputs": [{"shape": ["${N}"], "dtype": "float32"}],
  "shape_mode": "runtime",
  "compile_group": "identity_runtime"
}
```

`symbols` 是有限绑定，不接受无限范围或任意 Python 表达式。约束写法见 [配置](configuration.md#组合约束)。

## GPU 使用示例与后端差异

在已配置 C++ 扩展工具链的 MUSA 环境中，运行同一扩展处理三个长度的示例：

```bash
python -m musaopst run examples/suites/dynamic_matrix.json --case 'musa_cpp_runtime_shape_*' --device 0 --output runs/manual-dynamic-shapes-gpu
```

预期有三个非空用例，各自通过独立参考比较。检查 `dynamic_execution`、后端 launch 信息和构建产物标识，确认是否实际复用，而不是只看 PASS 数量。

| 后端 | 当前示例的复用范围与证据 |
|---|---|
| torch_musa | 同一 eager callable 接收不同 shape；没有单一编译二进制的承诺 |
| MUSA C++ 扩展 | 同一已构建扩展，通过运行时长度参数执行 |
| Triton | 使用支持运行期长度的 kernel；查看编译标识，其他 constexpr 变化可能产生新特化 |
| TileLang | 专门的 `axpby_dynamic` 支持连续一维符号 N；证据为保留的 JIT 对象及生成设备源码标识 |
| 原生 MUSA | 同一 host/kernel 可执行产物接收不同长度；各次 host 调用仍是独立子进程 |

TileLang 此示例不证明任意多维、非连续布局都能动态复用，也不把源码标识等同于设备二进制身份。原生子进程之间复用可执行文件，不等于保持同一个设备 context。自定义 factory 应明确实现 rebind，重新 JIT 的路径需在元数据中如实记录。

## 数据相关输出是另一种能力

输入 shape 不变但筛选结果条数变化，属于数据相关输出。使用 `shape: "reference"`、`rank`、`max_elements` 由独立参考限定结果，而不是由实际输出自行决定契约。见 [输出契约](output-contracts.md#数据相关输出)。该能力不能从普通符号 N 示例自动推断。
