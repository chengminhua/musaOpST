# 算子原型导入

`import-prototype` 只提取可靠声明的信息，不执行导入的源文件。支持 JSON 描述、受限 INI、Python 字面量 `MUSA_OPST_DESCRIPTOR`，以及 C++/MUSA 的 `/* MUSA_OPST_DESCRIPTOR {JSON} */` 注释。

仅有普通 Python 函数签名时，会列出函数及位置参数，但不能推断 shape、dtype、后端和参考函数；结果标记 `runnable_descriptor=false`。工具不会猜测任意 C++ ABI。输出是待审阅的原型描述，须先补齐缺失字段，再交给 `generate`。

## 使用示例

在项目根目录执行：

```bash
mkdir -p workspace
cat > workspace/operator_description.py <<'PYCODE'
MUSA_OPST_DESCRIPTOR = {
    "descriptor_version": 1,
    "backend": "cpu",
    "operation": "axpby",
    "name": "imported_add",
    "shape": [17],
    "dtype": "float32"
}
PYCODE
python -m musaopst import-prototype workspace/operator_description.py --output workspace/prototype.json
python -m musaopst generate workspace/prototype.json --output workspace/imported-project
python -m musaopst run workspace/imported-project/cases.json --output runs/imported-project
```

`prototype.json` 包含原始来源、`descriptor`、`unresolved` 和是否可生成标记；生成工程含配置与实际模板实现。

INI 使用 `[operator]` 和按序编号的 `[input.0]`、`[output.0]` 等节。`operator` 写 `name/backend/entry/reference`，每个张量节写 JSON 数组 `shape` 与 `dtype`，可加 `name`；属性用 `attrs={...}`。复杂布局或 TensorList 建议直接使用完整 JSON 描述。
