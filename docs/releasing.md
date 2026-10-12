# 发布与 GitHub 建仓

发布源码保留 `musaopst/`、`examples/`、`tests/`、`tools/`、`docs/`、`skills/` 及根目录说明和许可证。`.npy/.npz/.bin` 示例输入是测试依赖，不能按扩展名一概删除。

环境、缓存、结果与构建产物不提交：`.venv*/`、`.cache/`、`.musaopst-cache/`、`runs/`、`reports/`、`build/`、`dist/`。原始验收报告可能包含机器路径、设备和环境元数据，应另行保存；需要公开时单独审阅。上游源码审阅副本不随本项目分发。

## 本地验证与构建

在安装了项目依赖的独立 Python 环境，从项目根目录执行：

```bash
python tools/check_release.py --output ../release-check.json
python tools/run_offline_tests.py --output ../release-tests
python -m pip install 'build>=1.2' 'setuptools>=77.0.3' 'packaging>=24.2'
python -m build --outdir ../release-dist
```

`check_release.py` 核查发布目录、命名元数据、源码许可头、文档本地链接及常见私有环境/凭据模式，输出文件摘要。它跳过已知本地环境和生成目录；未知根文件、源码内二进制构建物、历史报告等会要求处理。扫描不等于全面秘密检测，还应检查实际 `git diff` 和归档文件清单。

源码包包含手册、技能、测试与示例；wheel 包含运行代码、kernel 源文件、schema、模板和许可文件。确认 wheel 的 `License-Expression` 为 `MulanPSL-2.0`，LICENSE 和 NOTICE 都存在。用隔离环境安装 wheel 后，从源码目录之外检查：

```bash
musaopst --version
python -c 'import musaopst; print(musaopst.__version__, musaopst.__file__)'
musaopst init demo --backend cpu
musaopst validate demo/cases.json
musaopst run demo/cases.json --output result
```

安装验证中的 CPU PASS 只证明平台流程。改动设备代码后，GPU 验收仍须使用同份冻结源码另做记录。

## GitHub 新建页面填写

| 字段 | 建议填写 |
|---|---|
| Owner | 选择你要持有项目的个人或组织账号 |
| Repository name | `musaOpST` |
| Description | `musaOpST: single-operator correctness and performance testing for Moore Threads MUSA.` |
| Visibility | 准备公开源码时选 `Public`；需先内部审阅则选 `Private` |
| Template | 不使用模板 |
| Add README | 不勾选，源码已经包含 |
| Add .gitignore | 选 None，使用本仓规则 |
| Choose a license | 选 None，上传本仓完整 MulanPSL-2.0 LICENSE |

页面选 None 只是避免生成另一份 LICENSE，不表示本项目无许可证。对已有本地项目，GitHub 也建议不要预填 README、ignore 和许可文件，以免首次推送产生冲突，见 [官方建仓说明](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)。

创建后可添加 Topics：`musa`、`operator-testing`、`torch-musa`、`triton`、`tilelang`、`benchmark`。About 中不要填写尚不存在的项目网站或假定可用的 PyPI 链接。

## 首次提交和推送

只把源码目录作为 Git 仓库，不在包含私有归档和报告的上一级目录初始化。尚未初始化时执行：

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git diff --cached
```

确认暂存内容后提交；如果 Git 提示缺少作者身份，设置本仓的 `user.name` 和 `user.email`。再将创建页面给出的真实 URL 用作远端：

```bash
git commit -m "Initial release of musaOpST v0.1.0"
# 先把 YOUR_GITHUB_REPOSITORY_URL 设置为刚创建仓库的实际 URL
: "${YOUR_GITHUB_REPOSITORY_URL:?请先设置实际仓库 URL}"
git remote add origin "$YOUR_GITHUB_REPOSITORY_URL"
git push -u origin main
```

已有 Git 历史或远端时先检查 `git status`、`git remote -v`，不重复初始化或覆盖远端。上传安装包可使用后续 GitHub Release；本指南不包含自动发布操作。
