# SPDX-License-Identifier: MulanPSL-2.0
"""接入模板。放在配置所在目录，entry 使用 custom_op:launch。"""

def launch(x, **attrs):
    # torch_musa 后端接收已在 MUSA 的 tensor；这里必须真正执行待测实现。
    return x.clone()


def reference(x, **attrs):
    # 接收独立的 NumPy 升精度输入。自行实现数学定义，不调用 launch。
    return x.copy()


def factory(case, inputs, workdir):
    # musa_cpp/triton/tilelang 工厂负责编译、创建输出、设置 launch 参数。
    # 返回的零参数 callable 只做 launch，输出分配尽量留在工厂阶段。
    raise NotImplementedError('替换成你的编译和 launch 实现；此模板不声称支持任意 .mu/.so')
