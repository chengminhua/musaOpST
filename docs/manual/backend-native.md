# 接入独立原生 MUSA 程序

设置 `backend: "native"`。Python 入口 `factory(case, inputs, workdir)` 返回构建与协议描述；host 程序负责设备分配、传输、kernel launch、错误检查、同步、输出和计时。它不需要采用 PyTorch 扩展 ABI。

通用协议标识为 `musaopst_native_v2`，这是机器协议版本，不能随文件改名而删除。描述至少给出 `protocol`、`operator` 和 `source`，也可提供经过摘要校验的 `binary`。工具以以下参数调用程序：

```text
program --request REQUEST_JSON --result RESULT_JSON
```

请求包含设备号、输入张量文件清单、可选输入/TensorList 结构、属性、输出契约和计时政策。输入文件按小端物理存储编码，元数据给出 shape、dtype、元素 stride、storage offset 和字节数；BF16 使用 16 位编码。输出需返回实际元数据及文件，平台先校验字节数和契约再比较，不能照抄期望 shape 掩盖实际错误。

结果需包含协议、状态、实际执行设备及设备号、输出列表；计时时还返回真实样本。只有成功同步后才能写出成功完成记录。可复用 `native_host.hpp` 的 `run_native(argc, argv, prepare, launch)`：开发者实现准备和 launch hook，host helper 管理同步、Event、文件及释放。

## 使用示例

在项目根目录、有 MUSA SDK 与兼容主机工具链的环境执行：

```bash
python -m musaopst init workspace/native-op --backend native
python -m musaopst validate workspace/native-op/cases.json
python -m musaopst run workspace/native-op/cases.json --device 0 --output runs/native-op
```

没有 torch_musa 自动探测目标架构时，先将 `MUSA_OPST_ARCH` 设置为实际设备的 `mp_<数字>`；工具不会猜测架构。独立构建见[build](build.md)。预期用例目录保存 `native_*_request.json`、`native_*_result.json`、张量 `.bin` 文件和运行日志，主报告保存统一比较与性能结论。

关键实现：[native_protocol_ops.py](../../musaopst/examples/native_protocol_ops.py)、[native_kernels.mu](../../musaopst/examples/native_kernels.mu)、[native_host.hpp](../../musaopst/examples/native_host.hpp)、[native_protocol.py](../../musaopst/backends/native_protocol.py)。

当前通用文件编码不含复数；旧 `vector_add_f32_v1` 仅保留简单兼容路径，不具备通用协议全部功能。原生每次调用启动新进程/context，Event 不包含进程启动及文件 I/O；无独立 GPU Profiler 采集器，不能使用父进程 CPU trace 替代。
