---
name: musaopst-develop
description: 开发或审查 musaOpST 平台代码、后端适配器、算子接入模板及相关回归测试，保持配置兼容、独立精度比较和真实设备证据。仅操作已有工具进行测试时使用 musaopst-use 技能。
---

# musaOpST 开发

在现有实现上完成用户指定的修改，按影响范围验证。展示名 `musaOpST`、分发与 CLI 名 `musaopst`、Python 包名 `musaopst` 各有用途；文件按职责命名，不批量删除配置和机器协议中的版本标识。

## 仓库定位与导航

从用户指定路径或当前目录向上寻找 `pyproject.toml` 和 `musaopst/cli.py`，确认分发名为 `musaopst`。若技能仍在仓库 `skills/` 下，可检查技能目录上两级；复制到全局技能目录后仍需重新定位项目。以下都是项目根目录下的路径，先读所在目录适用的 `AGENTS.md`（如有）。

| 改动范围 | 实现入口 | 对应文档或测试 |
|---|---|---|
| CLI、配置、输入绑定 | `musaopst/cli.py`、`config.py`、`schema.json`、`binding.py` | `docs/manual/configuration.md`、`tests/test_config_data_compare.py` |
| 实际数据、布局、参考、比较 | `musaopst/data.py`、`compare.py`、`expectations.py` | `docs/manual/reference.md`、`comparison.md`、`tests/test_integer_reference_integrity.py`、`tests/test_core.py` 的 `CompareTests` |
| 调度、阶段错误和证据 | `musaopst/runner.py`、`worker.py`、`reporting.py` | `docs/architecture.md`、`tests/test_runner.py`、`test_report_integrity.py` |
| 四条框架张量后端 | `musaopst/backends/gpu.py`、`musaopst/examples/` | `docs/manual/backend-torch.md`、`backend-cpp.md`、`backend-triton.md`、`backend-tilelang.md` |
| 原生程序与缓存 | `musaopst/backends/native_protocol.py`、`build_cache.py`、`musaopst/examples/native_host.hpp` | `docs/manual/backend-native.md`、`cache.md`、`tests/test_backend_protocols.py` |
| 复现与回归工作流 | `musaopst/workflows.py`、`artifacts.py`、`regression.py` | `docs/manual/regression.md`、`tests/test_workflows.py`、`test_resume_evidence.py` |
| 工程生成、模型规格 | `musaopst/generation.py`、`models.py` | `docs/manual/project-generation.md`、`model-import.md` |

表中省略目录的同列文件与该行前一个完整路径同目录。以当前源码为准，先追踪实际调用链，再决定修改位置；不要创建独立脚本绕开共享配置、比较器与报告。

## 修改过程中维持的关键行为

- CPU/离线入口按需导入 GPU 依赖。真实设备执行在新解释器 worker 中完成，超时与进程组只管理本任务；保留阶段状态，执行完成需要相应同步。
- 先量化目标输入，再独立复制给待测和参考。保留实际输入、输出、参考快照与设备元数据，避免原地写入和输出复用污染参考。
- 比较先验证数量、shape、dtype、布局/别名契约，再逐元素检查。整数严格比较不能经过浮点转换；非有限值遵守明确政策；自定义或额外指标不能覆盖基础失败。
- 不支持的 dtype、stride、offset、动态形状或计时政策显式拒绝，不能暗中转连续、CPU 回退或把重新特化称为同一产物复用。
- 未运行比较不能 PASS。预期失败保留实际失败及匹配证据，环境和严重设备故障不能被普通负向用例吞掉。复用、合并、离线重比较与本次新 GPU 通过分别统计。
- 构建缓存校验源码/头文件、参数、工具链、ABI 与产物摘要；不完整构建不可命中。旧配置或公开入口的兼容性需要实际回归，不能只改示例来绕过旧用法。
- 正确性先于正式性能；分离 JIT/构建、预热、参考和采样。原地计时逐次恢复输入且恢复在计时外；GPU Profiler 必须有设备事件，不能拿 CPU trace 充当 GPU trace。

## 接入新算子或后端

优先从现有 `init --backend` 生成工程或复用同类示例，阅读该后端手册。普通 torch_musa callable 使用统一绑定；C++ 扩展、Triton、TileLang 采用 `factory(case, inputs, workdir)` 返回零参数 `launch()`，并返回实际 MUSA 输出。需要预分配、编译身份或动态重绑定时按现有接口提供元数据及 `launch.rebind`。

独立原生程序使用 JSON 请求/结果与物理张量文件协议，不猜任意 `.mu/.so` 的 ABI。协议版本保持原义；host 负责实际设备执行、错误检查、同步与真实输出元数据。新增支持范围必须有相应实现和可执行示例，接口存在不等于功能完成。

同步更新受影响的 `docs/manual/` 章节及 `docs/features.json` / `docs/features.md` 能力范围；只在功能范围改变时更新能力清单，不把未实测后端写成通过。遵守 `LICENSE` 及第三方文件自己的许可和归属信息。

## 验证与交付

针对改动先运行相关用例。例如修改比较和数据时：

```bash
python -m unittest discover -s tests -p 'test_config_data_compare.py' -v
```

共享执行链、兼容接口或发布改动应运行完整离线回归，使用新的空输出目录保存证据：

```bash
python tools/run_offline_tests.py --output runs/offline-regression
python -m musaopst --help
```

新增行为用能区分正确与错误实现的正反例验证，尤其是错误 shape、错误数值、参考异常、超时、缓存失效、回放和复用计数；不要仅断言内部函数被调用。纯文案修改检查相关命令、链接即可，不为文字建立脆弱测试。

适用的 GPU 改动在当前授权且可用的设备上运行最小真实例，再扩展到必要 dtype、尾部和布局；测试前重新确认环境。无 GPU 时完成离线验证并明确未上板范围，不引用历史 GPU 日志作为当前源码通过证据。不升级共享驱动/SDK、不 reset GPU、不终止他人任务，不因技能被调用而自动发布仓库或安装包。

最终列出改动、实际测试命令与结果、源码与报告位置、兼容影响，以及未验证或阻塞的具体范围。若任务是准备发布，阅读 `docs/releasing.md`，运行 `tools/check_release.py`，检查打包清单和安装后的 CLI/资源，保持凭据、虚拟环境、缓存及本地运行产物不进入发布内容。
