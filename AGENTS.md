# Agent 工作入口

这是 musaOpST 的源码根目录。发行包与命令名为 `musaopst`，Python 包为 `musaopst`；不要从展示名推断 import 路径。

按任务读取所需技能：

- 使用工具、编写用例、运行与解释报告：[musaopst-use](skills/musaopst-use/SKILL.md)。
- 开发功能、接入后端、修改测试与准备发布：[musaopst-develop](skills/musaopst-develop/SKILL.md)。

先确认当前目录的 `pyproject.toml` 和实际实现，用户手册位于 `docs/manual/`。源码验收记录不随仓库发布；历史结论不能代替当前源码的新执行证据。

维护时保留 GPU 依赖按需导入、spawn worker 隔离、实际输入快照、独立参考、输出契约先于数值比较、正确性先于性能等不变量。CPU 模式只能证明平台流程。命令、配置字段或支持范围发生变化时，同步修改相应功能手册和有意义的测试。

项目使用 MulanPSL-2.0，新增原创源码保持 SPDX 标记；引入第三方代码前核对并保留其许可。生成报告、构建缓存及凭据不加入源码，发布检查见 `docs/releasing.md`。设备连接、逻辑卡号及环境从当前任务获取，不使用历史服务器信息作为默认值。
