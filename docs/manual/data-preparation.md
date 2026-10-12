# 独立造数与执行准备

`data` 根据统一配置创建实际输入快照；`prepare` 在此基础上记录入口、参考和后续构建执行计划。两者都不启动待测 GPU kernel，不生成精度通过结论。

低精度输入先量化到目标 dtype，再写入快照。文件输入会校验 shape、dtype、长度和配置的摘要；可选输入和 TensorList 的结构保存在配置中，快照按张量叶子存储。NPZ 保存逻辑值，后续依据布局定义重建 stride/offset。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst data docs/manual/examples/workflow-cpu.json --output runs/input-data
python -m musaopst prepare docs/manual/examples/workflow-cpu.json --output runs/prepared
python -m musaopst run --prepared runs/prepared --output runs/from-prepared
```

`data`/`prepare` 目录都包含 `config.original.json`、`config.resolved.json`、`preparation.json`，以及每用例的 `inputs.npz`、`inputs.meta.json`、`case.json`。`prepare` 的 `preparation.json` 另有执行计划。

`run --prepared` 校验这些文件后在新目录执行，不覆盖准备记录。修改配置或快照后应重新准备；不要手工替换文件再沿用旧摘要。
