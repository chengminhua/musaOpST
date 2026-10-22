# 环境检查与后端 smoke

`doctor` 在独立进程中探测 Python 包、MUSA runtime、设备和工具链。普通探测不执行测试 kernel；`--smoke` 在探测成功后执行所选后端的小算子，包含实际输出比较。

| 参数 | 用途 |
|---|---|
| `--device` | 可见设备列表内的序号（`torch.musa.set_device` 语义），不是物理 GPU 编号，默认 0；越界以退出码 2 拒绝并提示映射 |
| `--output` | 保存探测报告的目录；建议总是指定 |
| `--timeout` | 环境探测进程超时秒数，默认 60；smoke 用例各有自己的超时 |
| `--smoke` | 探测成功后运行后端测试 |
| `--backend` | 限定 smoke 后端，可重复；省略则检查五种后端 |

无 SDK 环境可以读取帮助和调用探测，但缺设备时返回非零，不会回退 CPU。环境信息缺失显示 `unknown`。已有 `doctor.json` 的目录不能复用。

## 使用示例

以下命令均在项目根目录执行；GPU 命令应使用当前已安装 MUSA 依赖的 Python 环境。

```bash
python -m musaopst doctor --output runs/doctor
python -m musaopst doctor --smoke --backend torch_musa --backend triton --device 0 --output runs/doctor-smoke
```

预期产物：`doctor.json`、`doctor.log`；开启 smoke 后另有 `smoke/results.json`、`smoke/summary.md` 和用例日志。探测通过不等于全部后端用例通过，需检查 `smoke_summary`。
