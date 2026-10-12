# 命名约定

项目展示名称和 GitHub 仓库名为 **musaOpST**。本地源码文件夹、Python 发行包、导入包和命令统一使用全小写的 **musaopst**，名称中没有连字符或下划线。

| 用途 | 名称或用法 |
|---|---|
| 展示名称 | `musaOpST` |
| GitHub 仓库 | `chengminhua/musaOpST` |
| 本地源码文件夹 | `musaopst/` |
| Python 发行包 | `musaopst` |
| Python 包目录 | `musaopst/` |
| Python 导入 | `import musaopst` |
| 模块执行 | `python -m musaopst` |
| 命令行程序 | `musaopst` |

克隆时显式指定本地文件夹名：

```bash
git clone git@github.com:chengminhua/musaOpST.git musaopst
cd musaopst
```

安装后使用：

```bash
musaopst --version
python -m musaopst --help
```

```python
import musaopst
print(musaopst.__version__)
```

正式版本从 `0.1.0` 开始，变更记录见 [CHANGELOG](../CHANGELOG.md)。wheel 文件名为 `musaopst-0.1.0-py3-none-any.whl`。配置 schema 和机器协议各自的版本号描述数据格式，与项目发行版本独立维护。
