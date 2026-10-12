# 执行、筛选和批量运行

`run` 按配置顺序串行执行。每个普通用例使用新启动的 worker；声明运行期 shape 组时，同组在一个新 worker 内复用后端实例。流程包括造数、构建/加载、执行同步、独立参考、比较，以及启用的重复测试和性能采样。

| 参数 | 用途 |
|---|---|
| 一个或多个配置路径 | 批量加载；多文件运行自动为名称加 `索引:文件名::` 前缀 |
| `--output` | 必填，结果写入新的空目录 |
| `--device` | GPU 逻辑设备号，默认 0 |
| `--case '模式'` | 按最终用例名匹配 shell 通配模式，多文件时包含前缀；须加引号 |
| `--tag` | 按标签筛选，可重复；同时满足全部指定标签 |
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

多配置示例：`python -m musaopst run first.json second.json --output runs/batch`，其中两个文件须是已完成校验的实际配置。
