# 阅读、重绘和合并报告

每次 `run` 自动生成机器可读 `results.json`、便于阅读的 `summary.md`、CI 使用的 `junit.xml`。用例目录保存阶段状态、输出快照、精度指标、性能样本和日志。首先查看状态与失败阶段，再查看 `worker.log` 和比较详情。

`report` 可以从 JSON 重绘摘要，不需要 GPU；`--merge` 汇总多个结果。合并记录标记 `imported`，保留来源及未执行覆盖，不能算成本次新执行。为保留原件，重绘与合并都建议指定新的 `--output`。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst run docs/manual/examples/workflow-cpu.json --output runs/report-a
python -m musaopst run docs/manual/examples/workflow-cpu.json --case 'workflow_add[[]0]' --output runs/report-b
python -m musaopst report runs/report-a --output runs/report-redrawn
python -m musaopst report runs/report-a runs/report-b --merge --output runs/report-merged
```

重绘目录包含重新计算汇总的 JSON、Markdown、JUnit；合并目录还复制各源用例资料，并给用例名增加来源前缀。

常见状态包括 `PASS`、`CONFIG_ERROR`、`BUILD_FAILED`、`REFERENCE_FAILED`、`EXECUTION_FAILED`、`ACCURACY_FAILED`、`TIMEOUT`、`ENVIRONMENT_ERROR`、`UNSUPPORTED` 和 `SKIPPED`。必需项未执行或全跳过不通过；匹配预期失败的用例仍保留实际失败状态，不计精度 PASS。具体状态以实际 JSON 为准。
