import onnx
import onnxruntime as ort
import numpy as np
from onnx import shape_inference
import onnxruntime as ort
import numpy as np

def test_onnx_input_shape():
    # 加载 ONNX 模型
    model = onnx.load("./log/partseg/Nico_v2_GCN_ONNX/model.onnx")

    # 查看计算图
    # print(onnx.helper.printable_graph(model.graph))

    # 使用 onnxruntime 进行推理，看看是否能正常执行
    session = ort.InferenceSession("./log/partseg/Nico_v2_GCN_ONNX/model.onnx")

    # 打印输入输出信息
    for input in session.get_inputs():
        print(f"Input name: {input.name}, shape: {input.shape}, type: {input.type}")

    for output in session.get_outputs():
        print(f"Output name: {output.name}, shape: {output.shape}, type: {output.type}")


    from onnx import shape_inference

    # 进行 shape 推理
    inferred_model = shape_inference.infer_shapes(model)

    # 保存新的 ONNX
    onnx.save(inferred_model, "./log/partseg/Nico_v2_GCN_ONNX/model_mixed.onnx")

    # 再次检查 shape
    print(onnx.helper.printable_graph(inferred_model.graph))


def test_mid_input():
    # 加载 ONNX 模型
    onnx_model_path = "./log/partseg/Nico_v2_GCN_ONNX/model.onnx"
    session = ort.InferenceSession(onnx_model_path)

    # 获取模型的输入名称和形状
    for inp in session.get_inputs():
        print(f"Input Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

    # 构造随机输入
    dummy_inputs = {
        inp.name: np.random.rand(*[dim if dim else 1 for dim in inp.shape]).astype(np.float32)
        for inp in session.get_inputs()
    }

    # 运行推理
    outputs = session.run(None, dummy_inputs)

    # 打印所有输出的形状
    for i, out in enumerate(session.get_outputs()):
        print(f"Output Name: {out.name}, Shape: {outputs[i].shape}")


# 用 onnx.helper 查找 GatherElements 层的输入
def onnx_helper():
    # 加载 ONNX 模型
    model = onnx.load("./log/partseg/Nico_v2_GCN_ONNX/model.onnx")

    # # 遍历所有的节点
    # for node in model.graph.node:
    #     if node.op_type == "GatherElements":
    #         print(f"Found GatherElements Node: {node.name}")
    #         print(f"  Inputs: {node.input}")
    #         print(f"  Outputs: {node.output}")
    #
    #         # 查找 indices 的形状
    #         for tensor in model.graph.input:
    #             if tensor.name == node.input[1]:  # indices 通常是第二个输入
    #                 print(f"  Indices Shape: {tensor.type.tensor_type.shape}")

    nferred_model = shape_inference.infer_shapes(model)
    # 遍历所有的节点
    for tensor in nferred_model.graph.value_info:
        if tensor.name == "/ptb_9/Add_output_0":
            print(f"Tensor Name: {tensor.name}")
            print(f"Shape: {tensor.type.tensor_type.shape}")

    for tensor in nferred_model.graph.value_info:
        if tensor.name == "/ptb_9/Expand_1_output_0":
            print(f"Tensor Name: {tensor.name}")
            print(f"Shape: {tensor.type.tensor_type.shape}")

# 直接推理onnx模型
def onnxPredict():
    # 1. 加载 ONNX 模型
    onnx_model_path = "./log/partseg/Nico_v2_GCN_ONNX/model.onnx"
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    # 2. 获取模型输入名称和形状
    input_details = session.get_inputs()
    for i, input_info in enumerate(input_details):
        print(f"Input {i}: name={input_info.name}, shape={input_info.shape}, dtype={input_info.type}")

    # 3. 构造输入数据 (确保与模型输入形状匹配)
    input_name_1 = input_details[0].name  # 假设第一个输入是 data

    # 生成随机输入数据
    input_data_1 = np.random.randn(4, 2048, 4).astype(np.float32)  # 这里替换为你的实际数据
    input_data_2 = np.random.randint(0, 2048, size=(4, 2048, 3), dtype=np.int64)  # 确保索引范围合理

    # 4. 进行推理
    outputs = session.run(None, {input_name_1: input_data_1})

    # 5. 查看输出
    print("Output shape:", outputs[0].shape)

if __name__ == "__main__":
    # test_onnx_input_shape()
    # test_mid_input()
    # onnx_helper()
    onnxPredict();