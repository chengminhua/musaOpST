# musaOpST

面向摩尔线程 MUSA 的单算子测试工具，用统一的 JSON 用例完成精度验证、性能测量和结果报告。当前版本为 **v0.1.0**，见 [更新记录](CHANGELOG.md)。

[快速开始](#快速开始) · [功能使用手册](docs/manual/README.md) · [后端支持范围](docs/features.md) · [常见问题](docs/manual/troubleshooting.md)

## 简介

musaOpST 适合在算子开发过程中验证自定义实现，也可用于建立算子回归用例。工具支持 torch_musa、MUSA C++ 扩展、Triton、TileLang 和独立原生 MUSA 程序，采用单卡、串行、前向执行方式。

一次测试的流程为：

```text
配置校验 → 构造输入 → 构建或加载 → 设备执行 → 独立参考计算
        → 输出契约与精度比较 → 有限重复 / 性能测量 → 生成报告
```

帮助、配置校验、数据准备、结果重比较和报告读取可在没有 GPU SDK 的环境使用。CPU 模式用于检查测试流程，结果会明确标记为平台测试；设备后端不可用时不会自动回退 CPU。

## 功能介绍

| 功能 | 用途 | 使用说明 |
|---|---|---|
| 用例配置 | 参数化、符号 shape、属性与组合约束 | [配置用例](docs/manual/configuration.md) |
| 输入构造 | 随机分布、固定数据、文件输入、可选输入和 TensorList | [生成输入](docs/manual/input-generation.md) · [文件输入](docs/manual/file-input.md) · [输入绑定](docs/manual/input-binding.md) |
| 工程与后端接入 | 生成可编辑工程，接入五种 MUSA 执行路径 | [生成工程](docs/manual/project-generation.md) · [后端接入](docs/manual/README.md#后端接入) |
| 精度测试 | 独立参考、输出契约、逐输出政策、非有限值和错误明细 | [参考计算](docs/manual/reference.md) · [精度比较](docs/manual/comparison.md) |
| 性能分析 | Event / 同步 wall time、Profiler、性能基线 | [性能测量](docs/manual/benchmarking.md) · [Profiler](docs/manual/profiling.md) |
| 结果复现 | 保存实际输入、离线重比较、快照回放和复现包 | [重比较](docs/manual/recompare.md) · [回放](docs/manual/replay.md) · [复现包](docs/manual/bundle.md) |
| 回归管理 | 批量配置、标签筛选、失败重跑、恢复和报告合并 | [执行用例](docs/manual/execution.md) · [回归管理](docs/manual/regression.md) |
| 自动用例准备 | 有限 Fuzz、受限原型导入、ONNX / GraphDef 规格抽取 | [Fuzz](docs/manual/fuzz.md) · [原型导入](docs/manual/prototype-import.md) · [模型规格](docs/manual/model-import.md) |

详细的输入形式、dtype、布局和编译限制见 [支持范围](docs/features.md)。后端可接入，不代表任意算子组合都已支持。

## 命名约定

展示名称和 GitHub 仓库名为 **musaOpST**；本地源码文件夹、Python 发行包、导入包和 CLI 统一为 `musaopst`。使用 `import musaopst`、`python -m musaopst` 或 `musaopst` 命令。具体对照见 [命名说明](docs/naming.md)。

## 环境准备

- Python 3.10 或更高版本、NumPy 1.24 或更高版本。
- 设备测试需已有兼容的 MUSA 驱动、运行时和所选后端依赖。
- C++ 扩展与原生程序需要对应的 MUSA 编译工具链；Triton、TileLang 需使用支持 MUSA 的版本。

在源码项目根目录创建独立 Python 环境并安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m musaopst --help
```

若 GPU 依赖已安装在专用 Python 环境中，应使用该环境安装本项目。安装步骤、可选模型解析依赖以及 wheel 安装方式见 [安装说明](docs/manual/installation.md)。

## 快速开始

下面先用 CPU 跑通流程，再给出 GPU 运行方式。每次选择新的输出目录，工具不会覆盖已有测试结果。

### 1. 生成可编辑用例

```bash
python -m musaopst init demo --backend cpu
```

生成目录包含 `cases.json`、算子入口、独立参考和运行脚本。示例计算 `alpha × x + beta × y`，可以先修改 shape、dtype、属性和精度阈值，再替换算子实现。

### 2. 校验并执行

```bash
python -m musaopst validate demo/cases.json
python -m musaopst run demo/cases.json --output runs/demo
```

`validate` 检查配置并列出展开后的用例；`run` 执行输入构造、算子和参考计算，输出比较结果。这里的 CPU 结果仅表示平台流程验证。

### 3. 查看报告

打开 `runs/demo/summary.md` 阅读汇总，或在新目录重新生成报告：

```bash
python -m musaopst report runs/demo --output runs/demo-report
```

结果目录中的 `results.json` 用于程序读取，`junit.xml` 可接入 CI。输入、实际输出、参考输出和日志保存在各用例子目录中。

### 4. 在 MUSA 设备上测试

在已配置 torch_musa 的环境中，确认可用的逻辑设备编号后执行；以下以逻辑设备 `0` 为例：

```bash
python -m musaopst doctor --device 0 --output runs/doctor
python -m musaopst init demo-musa --backend torch_musa
python -m musaopst run demo-musa/cases.json --device 0 --output runs/demo-musa
```

`--device` 是**当前进程可见设备列表内的序号**（与 `torch.musa.set_device` 语义一致），不是物理 GPU 编号；启动时会打印序号到物理卡的映射。容器按 `MUSA_VISIBLE_DEVICES` 重新编号后应传 `0`：

| 可见性 | 用法与含义 |
|---|---|
| `MUSA_VISIBLE_DEVICES=6` | `--device 0` → 物理 GPU 6（列表只有一项） |
| `MUSA_VISIBLE_DEVICES=2,5` | `--device 0` → 物理 GPU 2；`--device 1` → 物理 GPU 5 |
| 未设置，机器有 8 张卡 | `--device 1` → 物理 GPU 1 |

越界会在启动 worker 前以退出码 2 拒绝，并打印请求值、可见列表和正确用法；例如 `MUSA_VISIBLE_DEVICES=6` 时传 `--device 6` 会提示改用 `--device 0`。这属于参数错误，不会记为 `ENVIRONMENT_ERROR`。`doctor`、`profile`、`replay` 使用同一套解析规则。CPU 平台用例不做设备校验。

```bash
MUSA_VISIBLE_DEVICES=6 python -m musaopst run cases/benchmark.json \
  --case '*sparse_attn_bwd_csa*' --benchmark --device 0 --output runs/attn-bwd
```

`--case` 同时接受通配符和子串：`--case '*sparse_attn_bwd_csa*'` 与 `--case sparse_attn_bwd_csa` 等价；零匹配时会列出可用用例名并以退出码 2 结束。不确定名称时先列出来：

```bash
python -m musaopst run cases/benchmark.json --list-cases
```

切换到 `musa_cpp`、`triton`、`tilelang` 或 `native` 可生成对应工程。只需验证某一后端时，可使用 `doctor --smoke --backend torch_musa`；详见 [环境检查](docs/manual/doctor.md)。

## 常用命令速查

| 操作 | 命令 |
|---|---|
| 查看帮助 | `python -m musaopst --help` |
| 校验配置 | `python -m musaopst validate cases.json` |
| 正确性测试 | `python -m musaopst run cases.json --output runs/test` |
| 正确性通过后测性能 | `python -m musaopst run cases.json --benchmark --output runs/perf` |
| 按名称选择用例 | `python -m musaopst run cases.json --case '*softmax*' --output runs/selected` |
| 按子串选择用例 | `python -m musaopst run cases.json --case softmax --output runs/selected` |
| 只列出用例名 | `python -m musaopst run cases.json --list-cases` |
| 有限重复测试 | `python -m musaopst run cases.json --repeat 20 --output runs/repeat` |
| 单独生成数据 | `python -m musaopst data cases.json --output runs/data` |
| 离线重比较 | `python -m musaopst compare runs/test --output runs/recompare` |
| 用实际输入回放 | `python -m musaopst replay runs/test --output runs/replay` |
| 生成复现包 | `python -m musaopst bundle runs/test --output runs/reproduce.tar.gz` |

每个命令的参数、适用条件、产物及示例见 [按功能分类的使用手册](docs/manual/README.md)。

## 使用边界

- 参考函数必须独立于被测实现。工具先检查输出数量、shape、dtype 和布局契约，再比较数值。
- 正确性通过后才进行正式性能采样。批量 Event、单次 Event 和同步 wall time 分别记录，不能混用统计口径。
- dtype、非连续布局、动态输出等能力取决于具体算子入口；不支持的组合明确报告。
- 仅支持单卡、串行、前向测试。有限重复不构成长期稳定性认证。
- 多卡分布式、通用 backward、自动调优、GUI 和数据库不在当前范围内。

## 项目结构

```text
musaopst/
├── musaopst/       # CLI、配置、比较器和后端实现
├── examples/        # 可直接运行的示例与功能矩阵
├── docs/manual/     # 按功能拆分的使用说明和示例
├── tests/           # 平台自动化测试
├── tools/           # 开发与验收脚本
├── skills/          # 供 agent 使用与开发的技能
├── AGENTS.md        # agent 工作入口
├── CONTRIBUTING.md  # 开发与验证约定
└── LICENSE          # 木兰宽松许可证第 2 版
```

Agent 可从 [AGENTS.md](AGENTS.md) 进入，或使用 [skills/](skills/README.md) 中的使用与开发技能。贡献代码和本地验证见 [CONTRIBUTING.md](CONTRIBUTING.md)，发布准备及 GitHub 填写说明见 [发布指南](docs/releasing.md)。

开发者可继续阅读 [架构说明](docs/architecture.md) 和 [后续方向](docs/roadmap.md)。使用问题见 [常见问题](docs/manual/troubleshooting.md)。

## 许可与来源

本项目原创内容采用 [MulanPSL-2.0](LICENSE)，许可范围见 [NOTICE](NOTICE) 和 [许可证说明](docs/license.md)。测试流程设计参考了 msOpST，参考版本与范围见 [设计来源](docs/design-reference.md)。

musaOpST 是独立项目，面向 MUSA 的运行实现由本项目维护，不承诺与参考项目的配置或 API 兼容。项目名称及对其他项目、产品的引用不表示官方隶属、认证或背书关系。
