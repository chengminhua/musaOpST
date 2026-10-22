# 执行、筛选和批量运行

`run` 按配置顺序串行执行。每个普通用例使用新启动的 worker；声明运行期 shape 组时，同组在一个新 worker 内复用后端实例。流程包括造数、构建/加载、执行同步、独立参考、比较，以及启用的重复测试和性能采样。

| 参数 | 用途 |
|---|---|
| 一个或多个配置路径 | 批量加载；多文件运行自动为名称加 `索引:文件名::` 前缀 |
| `--output` | 运行必填，结果写入新的空目录 |
| `--device` | 可见设备列表内的序号（`torch.musa.set_device` 语义），不是物理 GPU 编号，默认 0；启动时打印序号到物理卡的映射 |
| `--case '模式'` | 按最终用例名筛选：shell 通配模式（须加引号）或直接子串；多文件时包含前缀 |
| `--tag` | 按标签筛选，可重复；同时满足全部指定标签 |
| `--list-cases` | 只打印筛选后的 name/backend/tags，不执行用例 |
| `--benchmark` | 启用配置中的性能政策 |
| `--repeat` / `--duration` | 覆盖有限重复次数/秒数 |
| `--prepared` | 执行已准备或构建的目录，不能同时提供配置路径 |
| `--resume` / `--rerun-failed` | 详见[回归管理](regression.md) |

配置错误、必需用例失败、未执行或全部跳过均不能判作通过。CPU 与 GPU 结果分别标识，不会静默回退。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst validate docs/manual/examples/workflow-cpu.json
python -m musaopst run docs/manual/examples/workflow-cpu.json --output runs/cpu-workflow
python -m musaopst run docs/manual/examples/workflow-cpu.json --tag smoke --case 'workflow_add[[]0]' --output runs/cpu-filtered
```

第一个配置展开为 4 个 CPU 用例；筛选命令只运行 `workflow_add[0]`。名称中的 `[` 是通配符特殊字符，因此用 `[[]` 匹配字面方括号。预期产物为 `results.json`、`summary.md`、`junit.xml` 和 `case-0000/` 等用例目录；目录内有实际输入/输出/参考快照及 `worker.log`。

在具备对应 MUSA 后端的环境执行小型设备测试：

```bash
python -m musaopst run examples/gpu_smoke.json --case 'torch_softmax' --device 0 --output runs/torch-smoke
```

`--device` 是**可见设备列表内的序号**，不是物理 GPU 编号；启动时打印 `--device 0 → 物理 GPU 6 (MUSA_VISIBLE_DEVICES=6, count=1)` 这样的映射并写入 `summary.md` 头部。容器只暴露一张卡时（`MUSA_VISIBLE_DEVICES=6`）必须传 `--device 0`；`MUSA_VISIBLE_DEVICES=2,5` 时 `--device 0/1` 分别对应物理 GPU 2/5。越界会在启动 worker 前以退出码 2 拒绝，消息包含请求值、可见列表、`device_count()` 和正确用法，不再归为 `ENVIRONMENT_ERROR`。`results.json` 保留 `environment.device_selection`（序号），并新增 `environment.device_resolution` 记录 requested/ordinal/physical_id/visible_devices/method。

`--case` 同时接受通配符和子串：`--case 'torch_softmax'`、`--case 'torch_*'`、`--case softmax` 都可用。零匹配时消息会回显 pattern 并列出可用用例名（同时写入 `summary.md`），以退出码 2 结束；`--tag` 零匹配同样处理。先查看名称：

```bash
python -m musaopst run examples/gpu_smoke.json --list-cases
```

多配置示例：`python -m musaopst run first.json second.json --output runs/batch`，其中两个文件须是已完成校验的实际配置。
