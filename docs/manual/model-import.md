# ONNX 与 TensorFlow 模型规格抽取

`extract-model` 提取节点名称、算子类型、属性以及已知输入输出 shape/dtype，生成待补齐的用例候选。它不执行模型，不捕获运行期真实输入，也不自动提供 MUSA 实现和独立参考。符号维度及未推断 shape 保留为未解决信息。

ONNX 使用可选 `onnx` 包，并尝试 shape inference；不跟随外部权重数据。TensorFlow 路径接受二进制 GraphDef `.pb`；可用 TensorFlow protobuf，也可只安装 TensorBoard protobuf，无需 TensorFlow 执行器。SavedModel 目录及模型运行不在支持范围。

## ONNX 示例

在项目根目录的项目独立 Python 环境执行：

```bash
python -m pip install '.[onnx]'
mkdir -p workspace
python - <<'PYCODE'
import onnx
from onnx import helper, TensorProto
x=helper.make_tensor_value_info('x',TensorProto.FLOAT,[2,3])
y=helper.make_tensor_value_info('y',TensorProto.FLOAT,[2,3])
node=helper.make_node('Relu',['x'],['y'],name='relu')
model=helper.make_model(helper.make_graph([node],'demo',[x],[y]),opset_imports=[helper.make_opsetid('',13)])
onnx.checker.check_model(model)
onnx.save(model,'workspace/relu.onnx')
PYCODE
python -m musaopst extract-model workspace/relu.onnx --output runs/onnx-spec
```

## GraphDef 示例

```bash
python -m pip install '.[tf-spec]'
mkdir -p workspace
python - <<'PYCODE'
from pathlib import Path
from tensorboard.compat.proto import graph_pb2, types_pb2
graph=graph_pb2.GraphDef()
x=graph.node.add(name='x',op='Placeholder')
x.attr['dtype'].type=types_pb2.DT_FLOAT
for size in (2,3): x.attr['shape'].shape.dim.add(size=size)
y=graph.node.add(name='relu',op='Relu')
y.input.append('x'); y.attr['T'].type=types_pb2.DT_FLOAT
shape=y.attr['_output_shapes'].list.shape.add()
for size in (2,3): shape.dim.add(size=size)
Path('workspace/relu.pb').write_bytes(graph.SerializeToString())
PYCODE
python -m musaopst extract-model workspace/relu.pb --format tf --output runs/tf-spec
```

两种命令均生成 `case_candidates.json` 和 `summary.md`。检查每个候选的 `unresolved`，为具体节点补齐实际输入政策、后端入口和独立参考，再转成统一用例配置。缺解析依赖时记录 `ENVIRONMENT_UNAVAILABLE` 并返回非零；这不是设备测试结果。
