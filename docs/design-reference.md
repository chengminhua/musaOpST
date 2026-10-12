# 设计来源

本项目在单算子测试流程设计上参考了 [msopgen 项目中的 msOpST](https://github.com/Ascend/msopgen/tree/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst)，感谢参考项目贡献者的开源工作。审阅版本为 `e8f6d68c01f6af37ce30c2adf10e4ae475d94882`，以下链接均固定到该版本。

| 参考内容 | 来源 | 本项目采用的思路 |
|---|---|---|
| 测试流程与阶段划分 | [msopst.py](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/scripts/msopst.py) | 将配置、准备、执行、比较和报告组织为明确阶段 |
| 用例配置与生成 | [case_generator.py](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/st/interface/case_generator.py) | 通过配置描述测试用例，支持参数化准备 |
| 输入准备与参考计算 | [data_generator.py](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/st/interface/data_generator.py) | 分开组织输入构造、参考计算和数据保存 |
| 精度比较与结果汇总 | [compare_data.py](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/st/interface/compare_data.py)、[result_comparer.py](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/st/interface/result_comparer.py) | 记录比较结果、失败阶段和错误明细 |
| 报告组织 | [st_report.py](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/st/interface/st_report.py) | 汇总用例状态、阶段结果和计数 |
| 构建与设备执行 | [acl_op_runner.py](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/tools/msopst/st/interface/acl_op_runner.py) | 将构建、执行和结果读取分开处理 |
| 使用文档组织 | [项目 README](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/README.md)、[msOpST 快速入门](https://github.com/Ascend/msopgen/blob/e8f6d68c01f6af37ce30c2adf10e4ae475d94882/docs/zh/quick_start/msopst_quick_start.md) | 按简介、环境、命令、步骤与结果组织使用说明 |

上述清单记录设计参考范围。musaOpST 的配置接口、MUSA 后端、比较政策和报告格式由本项目维护，具体行为以本项目实现和手册为准，不承诺与参考项目的配置或 API 兼容。用于审阅的上游源码和文档副本不包含在当前源码包或 wheel 中。

各参考文件仍适用其原有许可证；本项目的许可不覆盖这些上游材料。实际引入或改编第三方内容时，需另行记录对应文件及修改情况并履行适用许可要求，见 [许可证说明](license.md)。

musaOpST 是独立项目。对参考项目及产品名称的使用仅为交代来源，不表示官方隶属、认证或背书关系。
