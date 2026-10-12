# 开发与贡献

项目展示名为 **musaOpST**，Python 包为 `musaopst`。运行入口、配置和示例见 [README](README.md) 与 [功能手册](docs/manual/README.md)，代码职责见 [架构](docs/architecture.md)。

## 开发环境

使用独立 Python 环境；GPU 测试沿用已有兼容环境，不由本项目安装驱动或 SDK。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m musaopst --help
```

## 验证修改

核心平台测试无需 MUSA SDK：

```bash
python tools/run_offline_tests.py --output runs/platform-tests
```

输出目录须不存在。测试包含比较器抓错、文件输入、配置展开、worker 超时和报告等正反例。扩展后端的修改还需在当前可用设备上验证受影响用例，并保留实际输出、环境和源码摘要；不能将 CPU 测试写成设备通过。

示例 GPU 运行方式：

```bash
python -m musaopst doctor --device 0 --output runs/doctor
python -m musaopst run examples/gpu_smoke.json --case torch_softmax --device 0 --output runs/gpu-check
```

逻辑设备号应按实际环境调整。保持单卡、串行及有限测试；不通过故意越界或死锁 kernel 验证失败路径。

## 变更要求

- 修改配置、CLI 或后端行为时同步对应手册和可运行示例。
- 保持旧 schema 的读取兼容。配置和协议版本是接口身份，不随文件改名删除。
- 参考实现独立于待测实现；使用量化后的同值独立输入，先检查输出契约，再比较数值。
- 计时口径、复用结果和本次执行分别记录；不扩大 PASS 的含义。
- 提交前清理中间产物，按 [发布指南](docs/releasing.md) 检查源码和安装包。

原创贡献沿用 [MulanPSL-2.0](LICENSE)。不要给第三方代码换上本项目版权头；需要引入时保留原许可及来源说明。
