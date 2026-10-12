# 功能使用手册

每篇文档围绕一个功能，按“功能说明 → 参数与范围 → 使用示例 → 结果与限制”组织。首次使用建议先阅读 [安装](installation.md)，再按项目 [README](../../README.md#快速开始) 跑通一个 CPU 示例。

文中的命令默认在源码项目根目录执行，并已激活安装了 musaopst 的 Python 环境。`python -m musaopst` 与安装后的 `musaopst` 命令等价。示例输出位于 `runs/`；同一示例再次执行时请换一个空目录。标为“配置片段”的 JSON 需要放入完整用例，不能单独当作配置文件运行。

## 安装和执行

| 功能 | 你可以学到什么 |
|---|---|
| [安装](installation.md) | 源码、wheel、GPU 环境和可选依赖 |
| [环境检查](doctor.md) | 检查依赖、逻辑设备及指定后端 smoke |
| [生成测试工程](project-generation.md) | 用 init 或算子描述创建可编辑工程 |
| [执行用例](execution.md) | 正确性测试、设备选择、筛选、多配置和退出码 |
| [独立造数与准备](data-preparation.md) | 先冻结输入，再运行已有准备结果 |
| [独立构建](build.md) | 构建 C++/原生程序并执行已有产物 |
| [编译缓存](cache.md) | 缓存命中条件、查看和清理 |

## 配置、数据和正确性

| 功能 | 你可以学到什么 |
|---|---|
| [用例配置](configuration.md) | 完整 JSON、参数化、符号 shape、属性和约束 |
| [生成输入](input-generation.md) | 随机种子、分布、固定值与 dtype 量化 |
| [文件输入](file-input.md) | NPY、NPZ、BIN、端序和输入摘要 |
| [输入绑定](input-binding.md) | 位置/名称、可选输入、TensorList 和零输入 |
| [布局](layouts.md) | stride、offset、转置、切片和设备实际布局 |
| [输出契约](output-contracts.md) | 多输出、shape/dtype、布局、别名和数据相关 shape |
| [独立参考](reference.md) | 编写参考函数及选择参考计算精度 |
| [精度比较](comparison.md) | 容差、exact、NaN/Inf、逐输出政策和额外指标 |
| [预期失败](expected-failures.md) | 匹配阶段和异常类型，识别意外成功 |
| [编译期常量](compiled-constants.md) | 固定值与真正编译特化的区别 |
| [动态 shape](dynamic-shapes.md) | 静态展开与同 worker 产物复用的区别 |

## 性能与重复测试

| 功能 | 你可以学到什么 |
|---|---|
| [性能测量](benchmarking.md) | 批量/单次 Event、同步 wall time、stream、缓存与原地操作 |
| [Profiler](profiling.md) | 正确性通过后独立采集真实设备事件 |
| [有限重复](repeat.md) | 按次数/时长运行、异常及显存统计 |
| [回归管理与性能基线](regression.md) | 恢复、失败重跑、合并和可比性能门禁 |

## 报告与复现

| 功能 | 你可以学到什么 |
|---|---|
| [查看报告](reports.md) | JSON、Markdown、JUnit、状态与覆盖计数 |
| [离线重比较](recompare.md) | 用新政策重比快照，导出失败元素 CSV |
| [快照回放](replay.md) | 使用保存的实际输入重新执行 |
| [复现包](bundle.md) | 归档源码、输入与依赖信息，并在兼容环境回放 |

## 自动用例准备

| 功能 | 你可以学到什么 |
|---|---|
| [有限 Fuzz](fuzz.md) | 受限随机采样、约束过滤、自定义生成器 |
| [原型导入](prototype-import.md) | 从受支持的描述提取已知信息 |
| [模型规格抽取](model-import.md) | ONNX 与 TensorFlow GraphDef 的输入输出规格 |

## 后端接入

| 后端 | 接入方式 |
|---|---|
| [torch_musa](backend-torch.md) | 普通张量 callable 或预分配 factory |
| [MUSA C++ 扩展](backend-cpp.md) | 编译扩展、显式绑定与可选构建描述 |
| [Triton](backend-triton.md) | MUSA JIT kernel 与 launch factory |
| [TileLang](backend-tilelang.md) | MUSA kernel、输出约定与运行时 JIT |
| [原生 MUSA](backend-native.md) | 独立 host 程序、请求/结果及张量文件协议 |

问题定位请看 [常见问题](troubleshooting.md)。整体限制见 [功能与后端支持范围](../features.md)。小型手册配置放在 [examples/](examples/)，功能矩阵放在项目 [examples/suites/](../../examples/suites/)。
