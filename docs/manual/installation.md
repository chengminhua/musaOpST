# 安装

[返回手册目录](README.md)

## 功能说明

musaOpST 是 Python CLI。核心平台只依赖 Python 3.10+ 和 NumPy 1.24+；GPU SDK 由实际后端在执行时加载。安装本工具不会安装或升级设备驱动、MUSA SDK、Triton 或 TileLang。

## 源码安装示例

在源码项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m musaopst --version
python -m musaopst --help
```

`-e` 使当前源码修改立即可用。查看到命令帮助只表示安装成功；GPU 能力还需 [doctor](doctor.md) 检查和实际用例验证。

GPU 框架已安装在专用环境时，先激活该环境，再运行 `python -m pip install -e .`。如需让新的独立环境读取基础环境中的 GPU 包，可以使用 `python -m venv --system-site-packages .venv`；前提是基础环境本身可用。

## wheel 安装示例

在选择的 Python 环境中安装 `dist/` 下当前 wheel：

```bash
python -m pip install dist/musaopst-*.whl
python -m musaopst --version
```

离线且环境已有符合要求的 NumPy 时，可加 `--no-deps`。wheel 不携带 GPU 工具链，也不提供五个后端的框架安装包。

## 可选模型解析依赖

按需安装：

```bash
python -m pip install -e '.[onnx]'
python -m pip install -e '.[tf-spec]'
```

前者用于 ONNX，后者安装 TensorBoard 的 GraphDef protobuf；只用于 [规格抽取](model-import.md)，不执行整个模型。

## 验证安装

```bash
python -m musaopst init runs/manual-install-project --backend cpu
python -m musaopst validate runs/manual-install-project/cases.json
python -m musaopst run runs/manual-install-project/cases.json --output runs/manual-install-result
```

预期生成 `results.json`、`summary.md` 和 `junit.xml`，结果明确为 CPU 平台测试。重复运行时使用新的目录名。
