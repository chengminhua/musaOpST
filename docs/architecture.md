# 架构、兼容语义与扩展方向

musaopst 使用一条共享执行链：`CLI → 配置展开与校验 → 新解释器 worker → 后端执行 → 独立参考 → 契约与数值比较 → 有限重复/性能/Profiler → 报告`。CPU 平台模式和五种 MUSA 接入方式共享逻辑用例、数据构造、比较器及报告模型；CPU 结果不能计入 GPU 验收。

| 模块 | 当前职责 |
|---|---|
| `config.py`、`schema.json` | 兼容 schema v1/v2；参数与符号展开、约束、输入树、输出契约、逐输出政策、运行边界 |
| `binding.py`、`data.py` | 位置/名称与 optional/TensorList 绑定；生成或加载输入、按目标 dtype 量化、保留实际 stride/offset、创建独立参考副本 |
| `runner.py`、`device_identity.py` | 单卡串行、新解释器、逐例超时、动态组调度、身份及快照校验后续跑、严重设备故障后停止本轮设备验证 |
| `worker.py` | 分阶段执行与同步，保存实际输入/输出/参考快照，执行比较、有限重复、性能后复查和独立 Profiler 采集 |
| `backends/gpu.py` | torch_musa/C++扩展/Triton/TileLang 的公共 MUSA 张量适配；真实布局、当前或独立流、Event/wall计时、显存与PrivateUse1采集 |
| `backends/native.py` | 保留 `vector_add_f32_v1` 兼容协议；仅连续FP32向量加及旧计时ABI |
| `backends/native_protocol.py`、`examples/native_host.hpp` | 显式JSON+物理storage文件协议；多输入/输出、输入树、布局、独立host同步与计时；不猜任意 `.mu/.so` ABI |
| `backends/build_cache.py`、`cache.py` | native与带build descriptor的C++内容寻址构建缓存；校验产物SHA、预编译复用、私有缓存list/prune |
| `compare.py`、`expectations.py` | 先数量/shape/dtype/布局/别名契约，后逐元素与可选额外指标；负向期望独立于精度PASS |
| `workflows.py`、`artifacts.py` | data/prepare/build、prepared执行输入、离线重比与CSV、快照回放、源码/数据复现包和完整性验证 |
| `generation.py`、`models.py` | 六后端限定算子工程、受限原型导入、有限Fuzz；ONNX/GraphDef只抽取规格，不自动实现kernel或参考 |
| `reporting.py`、`regression.py` | JSON/Markdown/JUnit、覆盖与退出规则；合并保留来源、匹配环境和计时政策后比较性能基线 |

## 数据与比较边界

参考输入在待测执行前独立复制。低精度输入先按目标dtype量化，再按 `reference_policy=float64/float32/preserve` 处理参考输入；整数保留整数精度。待测输出在参考计算前完成同步、独立快照并落盘，避免原地操作或缓冲区复用污染参考。不同输入快照与错误shape不会通过广播被掩盖。

v2逐输出政策优先级为公共政策、当前输出dtype政策、当前输出专属政策。复数实部和虚部分别满足容差/非有限政策；最大绝对和相对误差取最大分量误差。RMSE使用复数差值模平方，cosine使用Hermitian内积的实部；这些指标是额外门槛，不能覆盖默认比较失败。自定义比较器使用副本，其源码也纳入报告、复现包和续跑身份。

平台允许显式正stride、非负offset、transpose和正步长slice，并核对设备实际布局。零/负stride、重叠storage和厂商私有blocked-format转换不在当前支持范围。扩展dtype的造数/比较支持不等于每个GPU算子支持；native协议及Triton/TileLang示例不支持复数，数学kernel主要实现FP32/FP16/BF16。

## 执行隔离与动态组

调度器用 `subprocess` 启动全新Python解释器，不继承已经初始化的设备上下文继续运行。超时只处理本任务worker的新进程组；原生程序受worker进程组监督。单独build命令有自身编译超时。工具不reset GPU，不升级驱动，也不操作其他任务；结束进程不保证故障硬件恢复。

普通用例独立启动worker。显式runtime compile_group在一个worker内prepare一次，通过声明的rebind接口换实际输入和shape；分组要求相邻、backend/entry/attrs/stream一致。Torch eager只记录调度复用，不称JIT编译一次。Triton记录编译产物并拒绝组内产生多份特化产物；TileLang只为受限的连续一维axpby提供符号N入口，其证据是同一JIT对象路径与生成源码摘要，不能表述为已证明二进制身份。结构化原生协议重用同一可执行文件，但每次调用仍启动新的host进程和设备上下文。

编译期常量目前只有Triton/TileLang的受限constant_scale：非空连续向量与同dtype有限标量，将实际量化后的标量嵌入constexpr/TIR，记录常量及编译产物证据。普通fixed输入或运行期attrs不是这项能力。

## 构建、缓存与复现

native描述提供源文件/头文件/编译参数；MUSA C++ entry显式提供 `build_descriptor(case, workdir)`。两者都有持久内容摘要缓存，命中前校验产物SHA；`build` 可单独构建，`run --prepared`验证快照及预编译产物再加载。C++离线构建在新解释器中进行，torch_musa导入可能探测设备，但不执行被测kernel。Triton/TileLang依赖运行期张量和JIT，上游缓存并未统一为可搬移的本工具构建产物。

`--offload-arch` 指定GPU ISA，不是主机交叉编译。本次aarch64→x86_64和反方向尝试均缺目标C++头文件/标准库环境，尚无完整跨主机工具链和ABI兼容保证。

回放复用实际量化输入，并重建原stride/offset；转置或切片快照不会再次应用变换。bundle收集已知入口、显式sources、相邻原生资源、工具源码及快照，不递归整个外部目录，不打包SDK或未知传递依赖。常见凭据文件、路径冲突和缺失文件被拒绝。源码与原报告不匹配会明确记录，不能称精确历史回放。

## 性能、重复与报告

正式性能在正确性和有限重复通过后执行；编译/JIT、参考、输入复制和预热分开。四条torch张量后端及结构化原生协议提供batch_event、single_event、wall_time，记录实际stream、分配范围和cache政策。批均值的P90不是单次调用尾延迟；Event区间仍可能含host提交间隙。cache interference只代表缓冲区写入干扰，不证明硬件冷缓存。

原地算子须显式allow_inplace并使用single_event或wall_time；每个样本前恢复输入，恢复不计入正式时间，并记录缓存影响。批Event原地循环拒绝。旧native v1仅支持batch_event/current/warm，不支持原地计时；额外模式明确Unsupported。

Profiler是独立诊断阶段，四条torch张量后端要求PrivateUse1 trace含GPU kernel事件，未采集硬件计数器。独立native子进程尚无专用采集器，CPU-only外层trace不算GPU分析。

有限重复只证明报告所列次数/时长与输入条件。torch分配器区分allocated/reserved；native只有runtime free/total，且每次重启上下文，不能据此宣称长期无泄漏。

PASS必须有真实比较证据。EXPECTED_FAILURE仅表示负向测试满足声明，不计精度PASS；必需未执行、全跳过、意外成功和未预期失败均保留独立结果与CI语义。续跑校验配置、实际输入、源码/自定义比较器、工具链、快照和设备UUID/驱动；reused与fresh分列。合并仅导入记录，不能制造新GPU通过。性能基线还要求口径与环境可比、至少5样本并检查噪声，不可比不算性能通过。

## 配置兼容与后续路线图

继续读取v1配置，保留其三种分布、末轴stride2及按首输出dtype作用于整例的比较政策；v2功能通过显式schema版本启用。兼容原生v1协议仍保留自身dtype、布局和计时限制，不用新的报告标签假装旧host实现新能力。

剩余工作是扩大明确能力的适用范围：更多算子的动态shape/变长输出与常量输入；更广dtype和布局；独立native采集器；Triton/TileLang构建缓存的可搬移协议；目标主机sysroot/ABI与更多GPU代际、SDK和框架版本的实测；原型映射和模型规格到可运行实现的受限接入。官方 MUSA kernel 模拟器仍需实际可用环境和协议，当前没有占位适配器。

当前范围不包括通用backward/gradcheck/opcheck、多卡/分布式/并发压力、自动调优、完整sanitizer或指令模拟器、mcu/msys深度集成、GUI/数据库/远程调度，以及长期稳定性认证。实现状态和验证证据分别见 [逐项清单](features.md)，具体运行入口见 [功能手册](manual/README.md)。
