# 更新记录

## 未发布

- `--device` 现在在父进程、启动 worker 之前解析并校验：按 `torch.musa.set_device` 语义解释为可见设备列表内的序号，并打印序号到物理卡的映射（如 `--device 0 → 物理 GPU 6`）。越界以退出码 2 拒绝，不再产生 `ENVIRONMENT_ERROR`；`doctor`、`profile`、`replay` 共用同一解析器。
- `results.json` 新增 `environment.device_resolution`（requested / ordinal / physical_id / visible_devices / method）；既有的 `environment.device_selection` 仍表示序号。`summary.md` 头部回显设备映射。
- `--case` 支持子串匹配（通配符仍可用），零匹配时列出可用用例名并以退出码 2 结束；新增 `--list-cases` 只打印 name/backend/tags。`--tag` 零匹配使用同样的提示。
- worker 中出现的 `invalid device ordinal` 在父进程重分类为 `CONFIG_ERROR`，并附带正确用法提示。

## v0.1.0 — 2026-10-12

首次发布 musaOpST。

- 提供统一的配置校验、输入构造、独立参考、输出契约与精度比较、性能采样和报告流程。
- 接入 torch_musa、MUSA C++ 扩展、Triton、TileLang 和独立原生 MUSA 五种后端；具体适用范围见 [功能说明](docs/features.md)。
- 提供工程生成、快照回放、离线重比较、复现包、回归管理和按功能拆分的使用手册。
- Python 发行包、导入包、CLI 与本地源码文件夹统一命名为 `musaopst`，展示名称和 GitHub 仓库名为 `musaOpST`。

当前范围为单卡、串行、前向测试。CPU 平台验证不能替代真实设备验收；配置与机器协议版本独立于发行版本。
