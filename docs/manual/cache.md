# 编译缓存

工具管理 MUSA C++ 扩展与通用原生协议的内容缓存。缓存身份包含声明源码/头文件、编译参数、架构、工具链和相关框架 ABI；复用时验证产物摘要。失败或未完成的构建不作为有效缓存。

默认根目录为当前目录的 `.musaopst-cache`。可用环境变量 `MUSA_OPST_CACHE` 调整构建时的缓存位置，或用 `cache --root` 指定管理目标。Triton/TileLang 的框架 JIT 缓存不在本命令管理范围；旧原生向量加兼容协议使用自己的结果目录缓存。

## 使用示例

在项目根目录执行：

```bash
export MUSA_OPST_CACHE="$PWD/.musaopst-cache"
python -m musaopst cache list
python -m musaopst cache list --root .musaopst-cache
```

输出 JSON 列出路径、缓存键和 `valid`。首次构建前列表为空是正常情况。执行[独立构建](build.md)后，可再次检查。

确实需要重新编译时：

```bash
python -m musaopst cache prune --root .musaopst-cache
```

该操作删除所选目录中具有工具缓存 manifest 的 native/C++ 条目，包括列出的无效产物；不会清理上游 JIT 缓存或任意目录。清理前停止当前项目的构建和运行。下一次执行会按需重建。
