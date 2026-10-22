---
name: musaopst-use
description: 使用 musaOpST 编写或校验用例、运行单算子精度与性能测试、读取报告、重比较和回放。适用于已有工具的使用任务；修改平台实现或后端适配器时使用开发技能。
---

# musaOpST 使用

帮助用户完成所请求的测试，并交付实际配置、运行命令和结果。展示名为 `musaOpST`，安装分发与 CLI 为 `musaopst`，Python 包为 `musaopst`。

## 找到项目与环境

- 从用户指定路径或当前目录向上查找同时包含 `pyproject.toml` 和 `musaopst/cli.py` 的目录，并确认项目分发名为 `musaopst`。技能位于仓库 `skills/` 时可以检查技能目录的上两级；安装到全局技能目录后，不要把技能目录当成仓库。
- 下文路径均相对于确认后的项目根目录。只读当前任务需要的 `docs/manual/` 文档，不要复制整份手册进对话。若只有已安装包，先用 `python -m musaopst --help` 与子命令帮助确认接口；确实需要源码或示例时再定位用户的源码目录。
- 使用用户指定或项目独立的 Python 环境。帮助、配置校验、离线比较和 CPU 平台测试不要求 GPU SDK；GPU 运行要使用已具备匹配 MUSA 依赖的环境，不擅自替换框架或升级驱动。

## 选择最小可验证流程

| 用户意图 | 先读的项目文档 | 主要入口 |
|---|---|---|
| 首次使用、检查环境 | `docs/manual/installation.md`、`doctor.md` | `init`、`doctor` |
| 写用例或接入已有算子 | `docs/manual/configuration.md` 及对应 `backend-*.md` | `init --backend`、`validate`、`run` |
| 文件、布局、复杂绑定 | `docs/manual/file-input.md`、`layouts.md`、`input-binding.md` | 统一 JSON 配置 |
| 精度失败排查 | `docs/manual/comparison.md`、`output-contracts.md`、`recompare.md` | `compare` |
| 性能、Profiler、有限重复 | `docs/manual/benchmarking.md`、`profiling.md`、`repeat.md` | `run --benchmark`、`profile` |
| 回放、复现、恢复与汇总 | `docs/manual/replay.md`、`bundle.md`、`regression.md`、`reports.md` | `replay`、`bundle`、`run --resume`、`report` |

表内省略目录的手册文件同样位于 `docs/manual/`。先校验配置与展开结果，再运行用户需要的范围。`--case` 接受 shell 通配模式或子串，要检查最终选中了多少用例；字面 `[` 可写成 `[[]`，名称不确定时先用 `--list-cases` 列出。所有结果写入新的空目录，保留原配置和原报告。

以下是仅验证平台流程的 CPU 示例，在项目根目录、使用尚不存在的目录名执行：

```bash
python -m musaopst init workspace/cpu-demo --backend cpu
python -m musaopst validate workspace/cpu-demo/cases.json
python -m musaopst run workspace/cpu-demo/cases.json --output runs/cpu-demo
python -m musaopst report runs/cpu-demo --output runs/cpu-demo-summary
```

真实设备测试先检查当前环境和用户授权设备，不从历史记录推断卡号或空闲状态。按任务所需选择后端，例如：

```bash
python -m musaopst doctor --output runs/doctor
python -m musaopst doctor --smoke --backend torch_musa --device 0 --output runs/torch-smoke
```

上例 `0` 是当前环境的逻辑设备号，只有对应用户选择时才使用。运行阶段坚持单卡串行、有限次数；设备严重故障时停止本轮设备验证，保留日志，不 reset GPU 或终止其他人的任务。缺少后端或设备时记录环境阻塞，并继续已授权的离线工作。

## 配置与结论必须保持的语义

- `backend: "cpu"` 只验证平台。不要把 GPU 配置改成 CPU 后报告设备通过，也不要把 `doctor` 的探测成功等同于完整算子验收。
- 参考函数应独立实现，使用目标 dtype 量化后的同一份实际输入。保持明确的输出数量、shape、dtype、布局与别名契约；不能用广播、隐式转连续或放宽容差来掩盖错误。
- 使用已有 schema 版本语义。新增逐输出比较等功能需要对应版本；版本号是兼容协议，不是待删除的文件命名后缀。
- 正式性能只在正确性通过后运行。报告区分单次 Event、批量 Event 均值、同步 wall time；批均值的 P90 不是单次尾延迟。保留采样次数、分配、stream 和缓存干扰政策。
- 预期失败只表示匹配声明的负向测试，不算精度通过；全跳过、必需项未执行或比较缺失不能宣称通过。不要为抓错构造越界或死锁 GPU kernel，可安全扰动返回值或使用 CPU 负向用例。

## 复用报告时避免混淆

`compare` 只从实际/参考快照重比较，不重新执行设备或参考；其结果是新的离线结论。`replay` 使用保存的实际输入启动新 worker，只有实际 GPU 执行才能产生新的 GPU 记录。

`run --resume` 校验身份与摘要后复用旧结果；`--rerun-failed` 需配合 `--resume`。检查 `summary.reused`、`summary.fresh_gpu_passed` 和逐例 `provenance`。合并报告只导入既有记录，也不能计为本次新执行。源码或快照摘要变化后不得手工修改记录以强制复用。

交付时指出配置、`results.json`、`summary.md`、`junit.xml` 和必要失败日志的实际位置，分别说明本次新执行、历史复用、离线通过、真实 GPU 通过和未执行原因。有限重复只能证明所测次数与条件，不等于长期稳定性认证。
