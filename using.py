import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# -------------------------- 1. 参数配置 --------------------------
MODEL_PATH = "best_model.pth"
INPUT_IMAGE_PATH = "C:/Users/上杉初绘/Desktop/app/dachuang/MoNuSeg/training/images/TCGA-2Z-A9J9-01A-01-TS1.tif"
N_CHANNELS = 3
THRESHOLD_OFFSET = 0.1  # 阈值偏移量（无需打印）
MORPH_KERNEL = np.ones((2, 2), np.uint8)
MORPH_ITER = 1
# 彩色配置
COLOR_MAP = {
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0)
}
# ----------------------------------------------------------------


# 2. 模型结构（无修改）
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# 3. 核心函数（删除所有调试打印）
def get_dynamic_threshold(prob_mask):
    """动态计算阈值（无打印）"""
    prob_mean = prob_mask.mean()
    prob_std = prob_mask.std()
    dynamic_thresh = prob_mean + prob_std + THRESHOLD_OFFSET
    return max(0.1, min(dynamic_thresh, 0.8))


def generate_black_white_mask(prob_mask):
    """生成黑白掩码（无打印）"""
    prob_smoothed = cv2.GaussianBlur(prob_mask, (3, 3), 0)
    dynamic_thresh = get_dynamic_threshold(prob_smoothed)
    binary_mask = (prob_smoothed > dynamic_thresh).astype(np.uint8) * 255
    # 形态学操作
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=MORPH_ITER)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_ITER)
    return binary_mask


def add_foreground_border(mask, border_color=(255, 255, 255), border_width=1):
    """添加前景边界（无打印）"""
    contours, _ = cv2.findContours((mask[:, :, 0] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_with_border = mask.copy()
    cv2.drawContours(mask_with_border, contours, -1, border_color, border_width)
    return mask_with_border


def single2color_mask(black_white_mask, color):
    """生成彩色掩码（无打印）"""
    h, w = black_white_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[black_white_mask == 255] = color
    return add_foreground_border(color_mask)


def predict_masks(input_path, model, device, transform):
    """推理生成掩码（无打印）"""
    image = Image.open(input_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
    
    black_white_mask = generate_black_white_mask(pred_prob)
    black_white_mask = Image.fromarray(black_white_mask).resize(original_size, Image.NEAREST)
    return image, np.array(black_white_mask)


# 4. 主函数（仅保留图片相关打印）
def main():
    # 设备信息（必要）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载模型（必要状态打印）
    model = UNet(n_channels=N_CHANNELS, n_classes=1).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("模型加载成功")
    except RuntimeError as e:
        raise RuntimeError(f"模型结构不匹配：{e}") from e

    # 生成掩码
    original_image, black_white_mask = predict_masks(INPUT_IMAGE_PATH, model, device, transform)

    # 1. 保存黑白掩码（图片相关打印）
    black_white_img = Image.fromarray(black_white_mask)
    black_white_img.save("black_white_mask_clear.png")
    print("清晰黑白掩码已保存至: black_white_mask_clear.png")

    # 2. 保存彩色掩码（图片相关打印）
    color_masks = {}
    for color_name, color in COLOR_MAP.items():
        color_mask = single2color_mask(black_white_mask, color)
        color_masks[color_name] = color_mask
        color_img_path = f"color_mask_{color_name}_clear.png"
        Image.fromarray(color_mask).save(color_img_path)
        print(f"{color_name}彩色掩码已保存至: {color_img_path}")

    # 3. 可视化并保存对比图（图片相关打印）
    plt.figure(figsize=(24, 8))
    # 原图
    plt.subplot(1, 6, 1)
    plt.imshow(original_image)
    plt.title("输入原图", fontsize=12)
    plt.axis("off")
    # 黑白掩码
    plt.subplot(1, 6, 2)
    plt.imshow(black_white_mask, cmap="gray")
    plt.title("清晰黑白掩码", fontsize=12)
    plt.axis("off")
    # 彩色掩码
    for i, (name, mask) in enumerate(color_masks.items(), 3):
        plt.subplot(1, 6, i)
        plt.imshow(mask)
        plt.title(f"{name}彩色掩码", fontsize=12)
        plt.axis("off")

    comparison_path = "mask_comparison_clear.png"
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.show()
    print(f"完整对比图已保存至: {comparison_path}")
    print("所有图片生成流程完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序异常：{e}")