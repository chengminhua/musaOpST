# 许可证

项目采用 **木兰宽松许可证第 2 版（MulanPSL-2.0）**，完整中英文文本位于根目录 [LICENSE](../LICENSE)。

本仓原创源码、示例、测试、技能和文档采用同一许可证，根目录 [NOTICE](../NOTICE) 明确范围。可写注释的源码带 `SPDX-License-Identifier: MulanPSL-2.0`；JSON 和测试用二进制输入由根许可与 NOTICE 覆盖。wheel 和源码包都附带 LICENSE、NOTICE，包元数据声明相同 SPDX 标识。

NumPy、torch_musa、Triton、TileLang 等依赖仍适用各自许可，本项目不打包或重新许可它们。后续引入第三方实现时，应逐文件保留来源、原版权和许可说明。

设计参考的版本、模块及范围见 [设计来源](design-reference.md)。用于审阅的上游源码和文档副本不随本项目源码包或 wheel 分发；原资料仍适用其自身许可。

参考项目的许可按文件区分：

| 参考资料 | 许可依据 |
|---|---|
| msopgen 根目录及注明相同许可的源码 | [MulanPSL-2.0](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/LICENSE) |
| `tools/msopst/st/interface/st_report.py` | 文件头单独注明 [Apache-2.0](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/st/interface/st_report.py) |
| 上游 `docs/` 文档 | [CC BY 4.0](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/docs/zh/legal/license_notice.md) |

这份参考说明不改变各资料的许可，也不代替实际复用时应履行的义务。若引入或改编第三方代码、文字、图表或模板，应记录具体来源和修改范围，按适用许可保留版权、署名、许可及其他必要声明，并随分发产物提供所需许可文本或链接。不得用本项目统一的许可标记覆盖第三方原有声明。

本项目是独立工具。项目名称及第三方名称仅用于标识项目、说明技术适配或交代来源，不表示官方隶属、认证或背书关系，也不授予第三方商标使用许可。具体条款以各自适用的许可证为准。
