# 使用实际输入快照回放

`replay` 将原结果中的实际输入转成文件输入配置，然后启动新的 worker 重新执行。使用的是当时实际量化后的输入值，而不是仅凭同一个 seed 重新造数。

回放前检查输入快照摘要、数量和 shape；原可选输入、TensorList、dtype、stride 与 storage offset 语义保留。原算子和参考模块仍须可导入。回放不承诺复制整个 SDK 环境；若要搬到其他机器，结合[复现包](bundle.md)检查依赖和路径。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst run docs/manual/examples/workflow-cpu.json --output runs/replay-source
python -m musaopst replay runs/replay-source --output runs/replayed
```

可用 `--case 'workflow_add[[]0]'` 选择手册示例中的第一个用例（`[[]` 匹配名称里的 `[`），用 `--device 0` 选择当前 GPU 逻辑设备。

预期产物：`runs/replayed-inputs/` 含回放配置、输入快照、来源摘要和模块搜索信息；`runs/replayed/` 含本次新执行报告。原报告不变。CPU 回放仍是 CPU 测试；真实 GPU 回放才会形成新的 GPU 执行记录。
