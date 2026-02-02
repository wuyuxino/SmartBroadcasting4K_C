from ultralytics import YOLO
import onnxruntime as ort
import numpy as np

# 1. 加载训练好的PT模型
model = YOLO("best.pt")

# 2. 导出FP32 ONNX（核心：half=False，simplify=True）
onnx_path = "best.onnx"
model.export(
    format="onnx",
    imgsz=[2176, 3840],  # 显式列表，(高,宽)匹配C++输入维度
    opset=13,
    simplify=True,       # 必须开启，简化ONNX结构
    dynamic=False,       # 固定batch=1，匹配C++端
    batch=1,
    half=False,          # 关键：先导出FP32 ONNX，后续trtexec转FP16
    device="0",
    agnostic_nms=False,
    retina_masks=False
)
print(f"✅ ONNX模型导出成功：{onnx_path}")

# 3. 验证ONNX模型有效性（关键：提前排查ONNX问题）
ort_session = ort.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
# 构造测试输入（1,3,2176,3840），FP32
test_input = np.random.randn(1, 3, 2176, 3840).astype(np.float32)
# 执行推理
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input})
# 打印输出维度（验证是否为(1,5,171360)）
print(f"✅ ONNX推理成功，输出维度：{[o.shape for o in outputs]}")
# 检查输出是否为有效值（非全0/极小值）
print(f"✅ 输出置信度范围：{np.min(outputs[0][0,4,:]):.6f} ~ {np.max(outputs[0][0,4,:]):.6f}")