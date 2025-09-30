import sys
import os
import torch
# 加入父目录到系统路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from xunlian import UNet  # 导入UNet模型

# 加载模型
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()  # 切换到推理模式

# 示例输入（与训练时的输入尺寸一致）
dummy_input = torch.randn(1, 3, 256, 256)  # batch=1, 3通道, 256x256

# 关键修改：禁用dynamo，使用传统导出方式，降低opset版本增强兼容性
torch.onnx.export(
    model,
    dummy_input,
    "unet_segmentation.onnx",
    export_params=True,
    opset_version=11,  # 降低opset版本（11比12兼容性更好）
    input_names=["input"],
    output_names=["output"],
    dynamo=False  # 禁用新导出器，使用传统逻辑
)
print("ONNX模型导出成功！")