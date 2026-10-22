# Agent 技能

| 技能 | 适用任务 |
|---|---|
| [musaopst-use](musaopst-use/SKILL.md) | 编写配置、运行用例、分析比较结果、回放和回归 |
| [musaopst-develop](musaopst-develop/SKILL.md) | 修改工具、接入后端、补充测试及准备发布 |

每个目录包含标准 `SKILL.md` 和可选的 agent UI 元数据。它们提供工作入口与关键约束，详细参数仍以源码及 `docs/manual/` 为准。

可以直接让 agent 读取仓库内相应的 `SKILL.md`。支持技能目录的工具也可以安装这两个独立目录。例如复制到 Codex 的用户技能目录（目标目录已存在时先核对，不覆盖已有技能）：

```bash
skills_destination="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$skills_destination"
for skill_name in musaopst-use musaopst-develop; do
    if [ -e "$skills_destination/$skill_name" ]; then
        echo "已存在，请先检查：$skills_destination/$skill_name"
    else
        cp -R "skills/$skill_name" "$skills_destination/$skill_name"
    fi
done
```

安装后可在支持技能调用的 agent 中使用 `$musaopst-use` 或 `$musaopst-develop`；新会话应重新确认技能是否已发现。复制到其他目录时，技能按当前工作目录和 `pyproject.toml` 定位项目，不把本机路径或服务器信息当作默认配置。

使用和开发各有一条示例请求。

- “使用 `$musaopst-use`，在当前源码仓生成一个 CPU 用例，运行并解释报告。”
- “使用 `$musaopst-develop`，给比较器增加一项能力，并验证对应成功和失败路径。”
