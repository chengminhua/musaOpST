# 功能与后端支持范围

本页描述当前实现范围，具体使用方法在 [功能手册](manual/README.md)。后端通用接口不代表任意算子、dtype 或布局都能直接运行；表中的限制以现有内建示例为准。

| 能力 | torch_musa | MUSA C++ 扩展 | Triton | TileLang | 原生 MUSA |
|---|---|---|---|---|---|
| 接入方式 | callable / factory | 扩展绑定 + factory | JIT + factory | JIT + factory | 显式 host 文件协议 |
| 独立构建与已有产物 | 无需编译时为 NOT_REQUIRED | 需 build_descriptor | 运行时 JIT | 运行时 JIT | 支持 |
| 实际输入回放、比较和报告 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 正 stride / offset 布局 | 由算子入口决定 | advanced 入口 | advanced 入口 | advanced 入口 | 结构化协议入口 |
| TensorList 内建示例 | 非空列表 | 两项 | 两项 | 两项 | 两项 |
| 动态 shape 示例 | eager，非编译复用 | 运行期参数 | 同组编译 kernel | 一维符号 N | 同一 binary，每调用新 host |
| 编译期常量示例 | 未实现 | 未实现 | 有限标量缩放 | 有限标量缩放 | 未实现 |
| 数据相关输出示例 | nonzero | 未实现通用分配 | 未实现通用分配 | 未实现通用分配 | 未实现通用分配 |
| Profiler | 设备事件 | 设备事件 | 设备事件 | 设备事件 | 未实现独立采集器 |
| 三种计时模式 | 支持 | 支持 | 支持 | 支持 | 新协议支持；兼容向量加协议仅批量 Event |
| 显存统计 | allocated / reserved | 同左 | 同左 | 同左 | free / total，不等同活动/缓存 |

数学核示例主要覆盖 float32、float16、bfloat16。更多整数及复数类型的 identity 示例只证明数据搬运和契约比较，不能外推为所有数学算子支持。当前 Torch clone 对 uint16/uint32 有限制；Triton/TileLang 复数入口和原生复数文件协议未实现。

负 stride、零 stride 广播和重叠 storage 明确不支持。厂商私有 blocked format、任意 tensor 常量、任意 ABI 推断、跨主机架构工具链自动配置未实现。官方 MUSA kernel 仿真没有可运行接入；CPU 平台模式不代表仿真。

## 功能目录与实现状态

“已实现”表示所列范围内有可运行功能；“部分实现”表示仍有上述限制。“不适用”表示当前范围明确排除项。五后端细分实现范围可读取 [features.json](features.json)。

| 编号 | 功能 | 实现状态 | 使用说明 |
|---|---|---|---|
| F001 | 统一配置到报告闭环 | 已实现 | [查看说明](manual/execution.md) |
| F002 | 参数化组合与名称筛选 | 已实现 | [查看说明](manual/configuration.md) |
| F003 | 多输入、多输出与输出契约 | 已实现 | [查看说明](manual/output-contracts.md) |
| F004 | 独立参考与量化后升精度 | 已实现 | [查看说明](manual/reference.md) |
| F005 | 精度误差与非有限值政策 | 已实现 | [查看说明](manual/comparison.md) |
| F006 | 基础Event性能采样 | 已实现 | [查看说明](manual/benchmarking.md) |
| F007 | JSON/Markdown/JUnit报告与CI退出 | 已实现 | [查看说明](manual/reports.md) |
| F008 | 五条MUSA后端统一接入 | 已实现 | [查看说明](manual/execution.md) |
| F009 | 算子原型自动生成配置 | 部分实现 | [查看说明](manual/prototype-import.md) |
| F010 | 从模型抽取算子规格 | 部分实现 | [查看说明](manual/model-import.md) |
| F011 | 有限Fuzz与自定义生成器 | 已实现 | [查看说明](manual/fuzz.md) |
| F012 | 符号shape与动态编译描述 | 部分实现 | [查看说明](manual/dynamic-shapes.md) |
| F013 | 可选输入 | 已实现 | [查看说明](manual/input-binding.md) |
| F014 | 命名输入与TensorList绑定 | 已实现 | [查看说明](manual/input-binding.md) |
| F015 | 标量属性类型 | 已实现 | [查看说明](manual/configuration.md) |
| F016 | 更多dtype | 部分实现 | [查看说明](manual/input-generation.md) |
| F017 | 更多输入分布 | 已实现 | [查看说明](manual/input-generation.md) |
| F018 | 从文件读取输入 | 已实现 | [查看说明](manual/file-input.md) |
| F019 | 原始格式与执行格式 | 部分实现 | [查看说明](manual/layouts.md) |
| F020 | 编译期常量输入 | 部分实现 | [查看说明](manual/compiled-constants.md) |
| F021 | 自动生成完整测试工程 | 已实现 | [查看说明](manual/project-generation.md) |
| F022 | 原生kernel测试脚手架 | 已实现 | [查看说明](manual/backend-native.md) |
| F023 | 通用原生执行协议 | 已实现 | [查看说明](manual/backend-native.md) |
| F024 | 仅准备/仅构建/运行已有工程 | 部分实现 | [查看说明](manual/build.md) |
| F025 | 开发机与运行机分离/主机交叉编译 | 部分实现 | [查看说明](manual/build.md) |
| F026 | 预期失败用例声明 | 已实现 | [查看说明](manual/expected-failures.md) |
| F027 | 失败元素CSV | 已实现 | [查看说明](manual/recompare.md) |
| F028 | Profiler与时间线 | 部分实现 | [查看说明](manual/profiling.md) |
| F030 | 逐输出比较政策 | 已实现 | [查看说明](manual/comparison.md) |
| F031 | 丰富的非连续布局 | 部分实现 | [查看说明](manual/layouts.md) |
| F032 | 输出布局和别名契约 | 已实现 | [查看说明](manual/output-contracts.md) |
| F033 | 参数语义约束 | 部分实现 | [查看说明](manual/configuration.md) |
| F034 | 参考精度可配置 | 已实现 | [查看说明](manual/reference.md) |
| F035 | ULP/余弦/RMSE与自定义比较 | 已实现 | [查看说明](manual/comparison.md) |
| F036 | 独立造数与离线重比较 | 已实现 | [查看说明](manual/data-preparation.md) |
| F037 | 实际快照回放 | 已实现 | [查看说明](manual/replay.md) |
| F038 | 便携复现包 | 部分实现 | [查看说明](manual/bundle.md) |
| F039 | 跨运行编译缓存 | 部分实现 | [查看说明](manual/cache.md) |
| F040 | 后端doctor smoke | 已实现 | [查看说明](manual/doctor.md) |
| F041 | 批量回归/标签/失败重跑/续跑/合并 | 已实现 | [查看说明](manual/regression.md) |
| F042 | 性能回归门禁 | 已实现 | [查看说明](manual/regression.md) |
| F043 | 更多计时政策 | 部分实现 | [查看说明](manual/benchmarking.md) |
| F044 | 原地算子性能测试 | 部分实现 | [查看说明](manual/benchmarking.md) |
| F045 | 运行期动态输出shape | 部分实现 | [查看说明](manual/output-contracts.md) |
| F046 | 零输入算子 | 已实现 | [查看说明](manual/input-binding.md) |
| F047 | 更多算子与环境矩阵 | 部分实现 | [查看说明](manual/README.md) |
| F050 | 避免.all布尔替代数值比较 | 已实现 | [查看说明](manual/comparison.md) |
| F051 | 显式逐元素 atol/rtol 精度阈值 | 已实现 | [查看说明](manual/comparison.md) |
| F052 | worker隔离/超时/严重故障停止 | 已实现 | [查看说明](manual/troubleshooting.md) |
| F053 | 全跳过/必需未执行不能CI通过 | 已实现 | [查看说明](manual/reports.md) |
| F054 | 有限重复与活动/缓存显存统计 | 部分实现 | [查看说明](manual/repeat.md) |
| F055 | CPU kernel仿真机制与MUSA仿真入口 | 未实现 | [查看说明](manual/troubleshooting.md) |
| F056 | 远程调度/多卡/分布式/并发压力 | 不适用 | [查看说明](manual/troubleshooting.md) |
| F057 | 通用backward/gradcheck/opcheck | 不适用 | [查看说明](manual/troubleshooting.md) |
| F058 | 自动调优搜索 | 不适用 | [查看说明](manual/troubleshooting.md) |
| F059 | sanitizer/调试器/指令模拟器 | 不适用 | [查看说明](manual/troubleshooting.md) |
| F060 | mcu/msys深度集成 | 不适用 | [查看说明](manual/troubleshooting.md) |
| F061 | GUI/数据库/远程调度服务 | 不适用 | [查看说明](manual/troubleshooting.md) |
| F062 | 长期稳定性认证 | 不适用 | [查看说明](manual/troubleshooting.md) |
| F063 | 平台抓错与GPU安全错误输出 | 已实现 | [查看说明](manual/expected-failures.md) |
| F064 | 新旧验收记录隔离 | 已实现 | [查看说明](manual/reports.md) |
| F065 | 每功能×每GPU后端状态矩阵 | 已实现 | [查看说明](manual/reports.md) |

当前范围为单卡、串行、前向测试。多卡分布式、通用 backward、自动调优、GUI/数据库、长期稳定性认证等不提供占位实现。
