# 有限 Fuzz 用例生成

`generate` 根据描述文件中的 `fuzz` 采样参数，先生成并冻结配置，之后才由 `run` 执行。它记录 seed、尝试次数、重复或约束不满足的拒绝原因；达到尝试上限仍不足目标数量时返回非零，不偷偷减少覆盖。

`fuzz.parameters` 支持 `choice`（`values`）和闭区间 `int`（`min`、`max`、可选 `step`）。`count` 为目标不同用例数，`max_attempts` 为最大尝试次数；CLI `--count`、`--seed` 可覆盖数量和种子。去重后数量可能不足，因此应给采样空间留余量。

## 使用示例

在项目根目录执行，基于手册 CPU 用例生成完整描述文件：

```bash
mkdir -p workspace
python - <<'PYCODE'
import json
from pathlib import Path
base=json.loads(Path('docs/manual/examples/workflow-cpu.json').read_text())['cases'][0]
base['parameters']={'n':[1],'dtype':['float32']}
descriptor={'descriptor_version':1,'schema_version':2,'case':base,
 'fuzz':{'count':4,'max_attempts':40,'parameters':{
  'n':{'kind':'int','min':1,'max':31},
  'dtype':{'kind':'choice','values':['float32','float16']}}}}
Path('workspace/fuzz.json').write_text(json.dumps(descriptor,indent=2))
PYCODE
python -m musaopst generate workspace/fuzz.json --seed 42 --count 4 --output workspace/fuzz-cases
python -m musaopst validate workspace/fuzz-cases/cases.json
python -m musaopst run workspace/fuzz-cases/cases.json --output runs/fuzz-cases
```

预期产物为冻结的 `cases.json`、含拒绝原因及 `complete` 的 `generation.json`，后续运行生成普通测试报告。

复杂生成器可将 `fuzz.parameters` 换为 `"generator":"my_generator:sample"`，同时设置 `timeout_s`（大于 0 且不超过 60）。模块中的 `sample(*, seed, index)` 应返回参数名到值的字典；它在有超时的新进程中执行。生成器属于受信任代码，不是安全沙箱；必须自行保证相同参数输出一致。框架仍校验、去重并冻结返回用例。
