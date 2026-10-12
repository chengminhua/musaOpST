# 打包复现材料

`bundle` 将结果目录、已知算子/参考/自定义比较源码、配置显式声明的 `sources`、工具源码和摘要清单打成 gzip tar。配置中应显式声明不能可靠自动发现的本地头文件与资源。

包内记录依赖要求，不打包系统驱动、SDK、整个虚拟环境或未知传递依赖。遇到常见凭据文件名会拒绝打包，但日志或自定义源码中的敏感内容仍需自行检查。源码已与原执行不同或依赖未解析时，`unresolved_sources` 明确列出，不能宣称完整复现。当前归档上限为 512 MiB。

## 使用示例

在项目根目录执行：

```bash
python -m musaopst run docs/manual/examples/workflow-cpu.json --output runs/bundle-source
python -m musaopst bundle runs/bundle-source --output bundles/cpu-example.tar.gz
```

预期归档结构为 `bundle.json`、`source/` 和 `evidence/`。`bundle.json` 包含文件 SHA、原执行与打包源码的对比、未解决依赖和回放命令。归档复制的历史结果不算本次新 GPU 验收。

解包至一个新目录后，先阅读 `bundle.json` 并安装兼容依赖。在解包目录可运行：

```bash
PYTHONPATH=source python -m musaopst replay evidence/results.json --output replay
```

搬迁的自定义模块可能仍含绝对资源路径；按清单修正后重新执行。不能仅凭归档成功判断可跨机器运行。
