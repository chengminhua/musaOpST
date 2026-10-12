# 文件输入

[返回功能手册](README.md)

文件输入用于复用已知数据、复现故障或建立固定回归样本。工具在设备执行前检查文件内容、shape、dtype、字节长度及可选的摘要，不会悄悄把不匹配的数据转换成配置要求的精度。

## 格式与字段

在张量描述中使用 `source` 替代 `distribution`：

| 字段 | 说明 |
|---|---|
| `source.path` | 文件路径；相对路径以配置文件所在目录为基准 |
| `source.format` | `npy`、`npz` 或 `bin`；省略时根据扩展名判断 |
| `source.key` | NPZ 必填，明确要读取的数组名称 |
| `source.byte_order` | BIN 必填，`little` 或 `big` |
| `source.sha256` | 可选的文件内容摘要；提供后必须匹配 |
| 输入 `shape`、`dtype` | 必填，必须与文件实际数据相符 |

NPY/NPZ 禁止 pickle 和 object 数组。BIN 的 dtype 由输入 `dtype` 指定，文件必须正好容纳该 shape 的元素，不能有多余字节。使用 `base_shape` 时，文件需符合变换前的 shape；转置或切片完成后才得到最终 `shape`。

## 完整使用示例

以下命令在项目根目录执行，生成三个内容等价的文件和完整配置；需要 NumPy。示例使用新的数据目录，避免覆盖现有数据。

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('runs/manual-file-input-data')
root.mkdir(parents=True, exist_ok=False)
x = np.array([0, 1, -2, 0.5, 4], dtype=np.float32)
np.save(root / 'x.npy', x)
np.savez(root / 'x.npz', data=x)
x.astype('<f4').tofile(root / 'x.bin')
cases = []
for fmt in ('npy', 'npz', 'bin'):
    source = {'path': f'x.{fmt}', 'format': fmt}
    source['sha256'] = hashlib.sha256((root / source['path']).read_bytes()).hexdigest()
    if fmt == 'npz':
        source['key'] = 'data'
    if fmt == 'bin':
        source['byte_order'] = 'little'
    cases.append({
        'name': f'file_{fmt}', 'backend': 'cpu',
        'entry': 'musaopst.examples.cpu_ops:identity',
        'reference': 'musaopst.examples.references:identity',
        'inputs': [{'shape': [5], 'dtype': 'float32', 'source': source}],
        'outputs': [{'shape': [5], 'dtype': 'float32', 'compare': {'exact': True}}]
    })
(root / 'case.json').write_text(json.dumps({'schema_version': 2, 'cases': cases}, indent=2))
PY
python -m musaopst validate runs/manual-file-input-data/case.json
python -m musaopst run runs/manual-file-input-data/case.json --output runs/manual-file-input
```

预期得到 3 个 CPU `PASS`，输入来源记录包含实际文件摘要。`validate` 只检查配置结构；真正的文件内容检查在 `data`、`prepare` 或 `run` 构造输入时进行。

## BF16 文件

BF16 BIN 使用明确的 16 位 BF16 编码，不是 IEEE FP16 编码。下面仅是输入配置片段：

```json
{
  "shape": [5],
  "dtype": "bfloat16",
  "source": {"path": "x-bf16.bin", "format": "bin", "byte_order": "little"}
}
```

BF16 的 NPY/NPZ 可保存 uint16 编码，或已严格量化成 BF16 的 float32 值。未量化的任意 float32 数据会被拒绝。可直接运行现成 GPU 示例：

```bash
python -m musaopst run examples/suites/contracts_matrix.json --case torch_file_bf16_bin --device 0 --output runs/manual-file-input-bf16
```

常见错误包括 NPZ 忘记 key、BIN 端序缺失、文件长度不符、float64 文件声明成 float32、摘要与修改后的文件不匹配。不要通过去掉检查来掩盖这些问题，应修正数据或声明。对于已有测试报告的输入，优先使用 [快照回放](replay.md)，它还保留输入结构及布局。
