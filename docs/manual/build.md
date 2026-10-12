# 独立构建与运行已有产物

`build` 先冻结实际输入，再调用声明的构建接口。原生 MUSA 接口给出源码或已构建 binary；C++ 扩展工厂提供 `build_descriptor(case, workdir)`，或在配置中提供显式 `artifact`。

| 后端 | 独立构建行为 |
|---|---|
| MUSA C++、native | 编译并记录产物摘要、工具链和架构 |
| CPU、torch_musa | `NOT_REQUIRED`，无需独立编译 |
| Triton、TileLang | `UNSUPPORTED`，需通过 `run` 在运行时 JIT |

`--arch` 使用 `mp_<数字>` 格式。必须根据实际目标设备选择，不能照抄别的设备架构。构建机器需要匹配目标的 SDK、主机编译工具链和框架 ABI；该命令不会安装或升级它们。

## 使用示例

在项目根目录执行，先创建 C++ 示例工程：

```bash
python -m musaopst init workspace/cpp-build --backend musa_cpp
# 将 MUSA_OPST_ARCH 设置为你实际目标的 mp_<数字>，然后执行：
: "${MUSA_OPST_ARCH:?请先设置实际目标架构}"
python -m musaopst build workspace/cpp-build/cases.json --arch "$MUSA_OPST_ARCH" --output runs/cpp-build
python -m musaopst run --prepared runs/cpp-build --device 0 --output runs/cpp-prebuilt
```

预期构建目录含 `preparation.json`、解析后配置、每用例输入快照和构建日志。运行前检查配置、产物文件及其 SHA；损坏或变更的产物不能复用。构建成功本身不表示精度通过，后续 `run` 才产生精度报告。

换成 `--backend native` 可生成原生工程。对于自定义 C++ 工厂，缺少 `build_descriptor` 时会明确返回不支持，仍可自行实现运行时构建入口。
