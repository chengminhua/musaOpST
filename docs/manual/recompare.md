# 离线重比较与失败元素 CSV

`compare` 读取已运行结果中的 `actual.npz`、`reference.npz` 和实际输出元数据，使用同一比较器重新计算。它不会重新执行算子，也不会重新算参考；原报告与原判定保留不变。

`--policy` 可提供一个比较政策 JSON 对象，其字段覆盖原政策及逐输出对应字段。`--csv-limit` 限制每个用例导出的失败元素行数，默认 10000；报告记录总不匹配数与截断标记。形状等契约失败时不进行数值比较，CSV 只有表头，原因写入 `comparison.json` 的 `csv.reason`。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst run docs/manual/examples/workflow-cpu.json --output runs/recompare-source
python -m musaopst compare runs/recompare-source --csv-limit 1000 --output runs/recompare-default
mkdir -p workspace
cat > workspace/strict-policy.json <<'JSON'
{"atol":0.0,"rtol":0.0}
JSON
python -m musaopst compare runs/recompare-source --policy workspace/strict-policy.json --csv-limit 100 --output runs/recompare-strict
```

预期产物：`recompare.json`、`summary.md`，每用例 `comparison.json` 和 `failures.csv`。严格政策可能导致原来允许舍入误差的用例失败，这是新的离线政策结论，不能修改原始设备验收结论。CSV 记录输出编号、索引、实际值、参考值、绝对/相对误差及原因；全局指标仍以 JSON 为准。
