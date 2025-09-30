import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置随机种子，确保结果可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sliding_window_infer(image_path, model, device, transform, threshold=0.5, patch_size=256, overlap=32):
    """滑动窗口推理，支持自定义阈值"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    img_np = np.array(image)
    
    pred_mask = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, patch_size - overlap):
        for x in range(0, w, patch_size - overlap):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            y_start = max(0, y_end - patch_size)
            x_start = max(0, x_end - patch_size)
            
            patch = Image.fromarray(img_np[y_start:y_end, x_start:x_end])
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).squeeze().cpu().numpy()
            
            pred_mask[y_start:y_end, x_start:x_end] += patch_pred
            count[y_start:y_end, x_start:x_end] += 1
    
    count[count == 0] = 1
    pred_mask = pred_mask / count
    pred_mask = (pred_mask > threshold).astype(np.uint8) * 255  # 使用自定义阈值
    return Image.fromarray(pred_mask)


class CancerCellDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 增强数据多样性
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # 随机旋转
            angle = random.uniform(-15, 15)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
            
            # 随机亮度和对比度调整
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                image = transforms.functional.adjust_brightness(image, brightness)
                image = transforms.functional.adjust_contrast(image, contrast)
            
            # 随机弹性形变（需要imgaug库支持）
            if random.random() > 0.5:
                try:
                    import imgaug.augmenters as iaa
                    from imgaug.augmentables.segmaps import SegmentationMapsOnImage
                    
                    img_np = np.array(image)
                    mask_np = np.array(mask)
                    segmap = SegmentationMapsOnImage(mask_np, shape=img_np.shape)
                    
                    aug = iaa.ElasticTransformation(alpha=50, sigma=5)
                    img_aug, seg_aug = aug(image=img_np, segmentation_maps=segmap)
                    
                    image = Image.fromarray(img_aug)
                    mask = Image.fromarray(seg_aug.get_arr())
                except ImportError:
                    print("警告: imgaug库未安装，跳过弹性形变增强")

        # 应用数据变换
        if self.transform:
            image = self.transform['train'](image)
            mask = self.transform['mask'](mask)

        mask = (mask > 0.5).float()
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):  # 增强dropout
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 增加正则化
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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.3, weight_dice=0.7):  # 调整权重，增强Dice影响
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets, smooth=1):
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sigmoid.sum() + targets.sum() + smooth)
        bce_loss = nn.BCELoss()(inputs_sigmoid, targets)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


def find_optimal_threshold(model, val_loader, device, thresholds=np.arange(0.3, 0.8, 0.05)):
    """在验证集上搜索最优二值化阈值"""
    model.eval()
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        f1_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > threshold).float()
                
                for pred, mask in zip(preds_binary, masks):
                    intersection = (pred * mask).sum()
                    precision = intersection / (pred.sum() + 1e-8) if pred.sum() > 0 else 0
                    recall = intersection / (mask.sum() + 1e-8) if mask.sum() > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
                    f1_scores.append(f1)
        
        mean_f1 = np.mean(f1_scores)
        print(f"阈值: {threshold:.2f}, F1分数: {mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = threshold
    
    print(f"最优阈值: {best_threshold:.2f}, 最佳F1分数: {best_f1:.4f}")
    return best_threshold


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, patience=8, min_improvement=1e-4):
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    
    best_val_loss = float('inf')
    best_val_dice = -float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            # 计算训练Dice
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            intersection = (preds_binary * masks).sum()
            union = preds_binary.sum() + masks.sum()
            dice = (2. * intersection) / (union + 1e-8)
            train_dice += dice.item() * images.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss_avg = train_loss / len(train_loader.dataset)
        train_dice_avg = train_dice / len(train_loader.dataset)
        train_losses.append(train_loss_avg)
        train_dice_scores.append(train_dice_avg)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).float()
                intersection = (preds_binary * masks).sum()
                union = preds_binary.sum() + masks.sum()
                dice = (2. * intersection) / (union + 1e-8)
                val_dice += dice.item() * images.size(0)

        val_loss_avg = val_loss / len(val_loader.dataset)
        val_dice_avg = val_dice / len(val_loader.dataset)
        val_losses.append(val_loss_avg)
        val_dice_scores.append(val_dice_avg)

        scheduler.step(val_loss_avg)
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss_avg:.4f}, Train Dice: {train_dice_avg:.4f}, '
              f'Val Loss: {val_loss_avg:.4f}, Val Dice: {val_dice_avg:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 早停判断
        loss_improved = (best_val_loss - val_loss_avg) > min_improvement
        dice_improved = (val_dice_avg - best_val_dice) > min_improvement
        
        if loss_improved or dice_improved:
            if loss_improved:
                best_val_loss = val_loss_avg
            if dice_improved:
                best_val_dice = val_dice_avg
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_count = 0
            print(f"保存最佳模型 (Val Loss: {best_val_loss:.4f}, Val Dice: {best_val_dice:.4f})")
        else:
            early_stop_count += 1
            print(f"早停计数: {early_stop_count}/{patience}")
            if early_stop_count >= patience:
                print("验证指标连续停滞，触发早停！")
                break

    # 绘制训练曲线
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_dice_scores, label='Training Dice')
    plt.plot(val_dice_scores, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    return model


def evaluate_model(model, test_loader, device, threshold=0.5):
    """使用指定阈值评估模型"""
    model.eval()
    metrics = {
        'dice': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > threshold).float()

            for pred, mask in zip(preds_binary, masks):
                intersection = (pred * mask).sum()
                union = pred.sum() + mask.sum()
                dice = (2. * intersection) / (union + 1e-8) if union > 0 else 0
                
                precision = intersection / (pred.sum() + 1e-8) if pred.sum() > 0 else 0
                recall = intersection / (mask.sum() + 1e-8) if mask.sum() > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
                
                metrics['dice'].append(dice.item())
                metrics['precision'].append(precision.item())
                metrics['recall'].append(recall.item())
                metrics['f1'].append(f1.item())

    return {k: np.mean(v) for k, v in metrics.items()}


def visualize_predictions(model, test_loader, device, threshold=0.5, num_samples=5):
    model.eval()
    num_samples = min(num_samples, len(test_loader.dataset))
    samples = random.sample(list(test_loader.dataset), num_samples)

    for i, (image, mask) in enumerate(samples):
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > threshold).astype(np.float32)  # 使用最优阈值

        # 反归一化显示
        image_np = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0.0, 1.0)
        
        mask_np = mask.squeeze().numpy()
        sharpened_mask = laplacian_sharpen(mask_np, alpha=0.5)
        sharpened_pred = laplacian_sharpen(pred_mask, alpha=0.5)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sharpened_mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(sharpened_pred, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.savefig(f'prediction_{i + 1}.png')
        plt.close()


def laplacian_sharpen(mask, alpha=0.5):
    mask_np = np.array(mask, dtype=np.float32)
    if mask_np.max() <= 1:
        mask_np = mask_np * 255
    laplacian = cv2.Laplacian(mask_np, cv2.CV_32F, ksize=3)
    sharpened = np.clip(mask_np - alpha * laplacian, 0, 255).astype(np.uint8)
    return sharpened


def main():
    # 数据路径
    train_input_dir = "C:/Users/上杉初绘/Desktop/app/dachuang/MoNuSeg/training/images"
    val_input_dir = "C:/Users/上杉初绘/Desktop/app/dachuang/MoNuSeg/test/images"
    test_input_dir = "C:/Users/上杉初绘/Desktop/app/dachuang/MoNuSeg/test/images"

    train_target_dir = "C:/Users/上杉初绘/Desktop/app/dachuang/MoNuSeg/training/manual"
    val_target_dir = "C:/Users/上杉初绘/Desktop/app/dachuang/MoNuSeg/test/manual"
    test_target_dir = "C:/Users/上杉初绘/Desktop/app/dachuang/MoNuSeg/test/manual"

    # 获取图像路径
    train_input_paths = sorted(glob.glob(os.path.join(train_input_dir, "*.*")))
    val_input_paths = sorted(glob.glob(os.path.join(val_input_dir, "*.*")))
    test_input_paths = sorted(glob.glob(os.path.join(test_input_dir, "*.*")))

    train_target_paths = sorted(glob.glob(os.path.join(train_target_dir, "*.*")))
    val_target_paths = sorted(glob.glob(os.path.join(val_target_dir, "*.*")))
    test_target_paths = sorted(glob.glob(os.path.join(test_target_dir, "*.*")))

    # 验证数据完整性
    assert len(train_input_paths) == len(train_target_paths), "训练集数据不匹配"
    assert len(val_input_paths) == len(val_target_paths), "验证集数据不匹配"
    assert len(test_input_paths) == len(test_target_paths), "测试集数据不匹配"

    print(f"训练集: {len(train_input_paths)} 对图像")
    print(f"验证集: {len(val_input_paths)} 对图像")
    print(f"测试集: {len(test_input_paths)} 对图像")

    # 数据变换
    transform = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'mask': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    }

    # 数据集和加载器
    batch_size = 8 if torch.cuda.is_available() else 2
    num_workers = 4 if torch.cuda.is_available() else 0

    train_dataset = CancerCellDataset(train_input_paths, train_target_paths, transform, augment=True)
    val_dataset = CancerCellDataset(val_input_paths, val_target_paths, transform, augment=False)
    test_dataset = CancerCellDataset(test_input_paths, test_target_paths, transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型初始化
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss(weight_bce=0.3, weight_dice=0.7)  # 增加Dice权重
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)  # 增强L2正则化
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, min_lr=1e-6)

    # 训练模型
    num_epochs = 40
    print(f"开始训练，共 {num_epochs} 个epoch")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                       num_epochs, device, patience=8, min_improvement=1e-4)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 搜索最优阈值
    print("\n正在搜索最优二值化阈值...")
    optimal_threshold = find_optimal_threshold(model, val_loader, device)

    # 用最优阈值评估测试集
    print("\n使用最优阈值评估测试集...")
    metrics = evaluate_model(model, test_loader, device, threshold=optimal_threshold)
    print(f"测试集Dice系数: {metrics['dice']:.4f}")
    print(f"测试集精确率: {metrics['precision']:.4f}")
    print(f"测试集召回率: {metrics['recall']:.4f}")
    print(f"测试集F1分数: {metrics['f1']:.4f}")

    # 可视化预测结果
    visualize_predictions(model, test_loader, device, threshold=optimal_threshold)
    print("预测结果已保存")


if __name__ == "__main__":
    main()
    