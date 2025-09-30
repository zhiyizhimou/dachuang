from flask import Flask, request, jsonify
from flask_cors import CORS  # 解决跨域问题
import onnxruntime as rt
from PIL import Image
import io
import numpy as np
import torchvision.transforms as transforms
import base64  # 用于图像Base64编码（前端展示）

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载ONNX模型
session = rt.InferenceSession("unet_segmentation.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 图像预处理（与训练时一致）
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 掩码后处理（生成纯二值掩码，类似“图二”）
def postprocess_mask(mask_prob):
    mask_prob = mask_prob.squeeze()  # 去除 batch 和 channel 维度
    # 二值化（阈值0.5，也可改用Otsu自动阈值，参考之前的enhance_mask_clarity）
    binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255
    return binary_mask

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "请上传图像"}), 400
    
    # 读取上传的图像
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    original_size = image.size  # 记录原图尺寸（后续恢复掩码大小）
    
    # 预处理图像
    input_tensor = preprocess(image).unsqueeze(0).numpy()  # 增加batch维度
    
    # ONNX模型推理
    output = session.run([output_name], {input_name: input_tensor})[0]
    mask_prob = 1 / (1 + np.exp(-output))  # 等价于torch.sigmoid（ONNX未直接输出sigmoid结果时需手动计算）
    
    # 后处理生成掩码
    binary_mask = postprocess_mask(mask_prob)
    mask_image = Image.fromarray(binary_mask).resize(original_size, Image.NEAREST)
    
    # 转为Base64编码（方便前端直接展示）
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return jsonify({"mask": mask_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 允许外部访问（部署时需配置）