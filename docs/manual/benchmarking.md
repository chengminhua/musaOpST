# 性能测量与计时口径

只有正确性通过后才进入正式 benchmark。编译/JIT、输入上传、独立参考、预热与正式采样分开，结束后再次比较输出。输出分配是否计入由后端入口声明；普通 eager 算子往往含分配，预分配 factory 可排除分配，应以报告政策为准。

| `benchmark` 字段 | 含义 |
|---|---|
| `enabled` | 是否启用，也可用 CLI `--benchmark` 开启 |
| `mode` | `batch_event`、`single_event` 或 `wall_time` |
| `warmup` | 正式计时前的预热次数 |
| `samples` | 原始样本数 |
| `iterations` | 批量 Event 每样本调用次数；单次 Event 必须为 1；wall time 实际每样本一次 |
| `stream` | `current` 或 `independent` |
| `cache` | `warm` 或 `interference` |
| `cache_bytes` | 干扰缓冲区字节数，使用干扰政策时必须大于 0 |
| `allow_inplace` | 原地算子显式开启，并选单次 Event/wall time |

批量 Event 的样本是“这一批耗时/调用次数”，P90 是批均值的分位数。单次 Event 每样本测一次；同步 wall time 还包含主机提交与同步开销，不能与纯 Event 数值混同。Event 区间也可能受主机提交间隙影响，不等于纯 kernel 指令时间。缓存干扰只扫描缓冲区，不保证绝对冷缓存。

原地算子在每个采样前恢复输入并同步，恢复在计时区间外；批量连续修改输入的模式被拒绝。旧原生兼容协议只支持原有批量 Event/current/warm 政策，其他模式需使用通用原生协议。

## 使用示例

在项目根目录执行，将真实 torch_musa 小例改为单次 Event：

```bash
mkdir -p workspace
python - <<'PYCODE'
import json
from pathlib import Path
config=json.loads(Path('examples/gpu_smoke.json').read_text())
config['schema_version']=2
config['cases']=[config['cases'][0]]
config['cases'][0]['benchmark']={'enabled':True,'mode':'single_event','warmup':3,'samples':10,'iterations':1,'stream':'independent','cache':'warm'}
Path('workspace/benchmark.json').write_text(json.dumps(config,indent=2))
PYCODE
python -m musaopst validate workspace/benchmark.json
python -m musaopst run workspace/benchmark.json --device 0 --output runs/benchmark
```

结果 `benchmark` 对象保存 `raw_samples_ms`、原始区间耗时、median、P90 及完整政策。改用其他模式时写入新配置、新目录；跨运行比较见[性能回归](regression.md)。CPU 模式可验证平台的计时流程，但不是 MUSA Event 验收。
