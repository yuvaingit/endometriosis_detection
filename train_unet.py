import os
import sys
from pathlib import Path

# Auto-reexec with virtual environment if torch is not installed in current Python
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
except ImportError:
    candidate_venvs = [
        Path.home() / "Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python",
        Path.home() / "Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python3",
        Path(__file__).resolve().parent / ".venv/bin/python",
        Path(__file__).resolve().parent / "venv/bin/python",
    ]
    for venv_py in candidate_venvs:
        if venv_py.exists():
            print(f"[train_unet] Auto-reexecuting with virtualenv: {venv_py}")
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    print("Error: PyTorch not found. Run with starter virtualenv:")
    print("  ~/Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python train_unet.py")
    sys.exit(1)

from dataset_loader import GlendaDataset


# Simple, effective U-Net building blocks for multi-class segmentation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):  # 5 classes: background (0) + 4 lesion types
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        if x.shape != x2.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape != x1.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        
        return self.outc(x)


class CombinedDiceAndCrossEntropyLoss(nn.Module):
    """
    Combines weighted Cross-Entropy Loss with Multi-Class Dice Loss.
    Addresses severe medical segmentation background class imbalance (>98% background).
    """
    def __init__(self, num_classes=5, weight_ce=1.0, weight_dice=1.2):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        # Downweight background (0.2) relative to lesions (1.0)
        self.register_buffer("class_weights", torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0]))

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        probs = F.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
        
        # Penalize low dice scores, emphasizing lesion classes 1..4
        dice_loss = 1.0 - (dice[0] * 0.2 + dice[1:].sum() * 0.8 / (self.num_classes - 1))
        
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


def calculate_lesion_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
    """Calculate binary lesion IoU (any lesion class > 0)."""
    preds_binary = (pred_masks > 0)
    gts_binary = (gt_masks > 0)
    intersection = (preds_binary & gts_binary).sum().item()
    union = (preds_binary | gts_binary).sum().item()
    if union == 0:
        return 1.0 if not preds_binary.any() else 0.0
    return intersection / union


def train_model(epochs=20, batch_size=4, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device} for {epochs} epochs")
    
    # 1. Load Datasets & DataLoaders with standardized image size (256x256)
    img_size = (256, 256)
    train_dataset = GlendaDataset("coco.json", "frames", split="train", val_ratio=0.2, img_size=img_size)
    val_dataset = GlendaDataset("coco.json", "frames", split="val", val_ratio=0.2, img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Initialize Model, Loss, Optimizer, and LR Scheduler
    model = SimpleUNet(in_channels=3, num_classes=5).to(device)
    criterion = CombinedDiceAndCrossEntropyLoss(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    best_iou = 0.0
    
    print("\n" + "="*70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Lesion IoU':<12} | {'LR':<10}")
    print("-" * 70)
    
    # 3. Training Loop across specified epochs
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        # Validation evaluation
        model.eval()
        running_val_loss = 0.0
        val_ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                for p, g in zip(preds, masks):
                    val_ious.append(calculate_lesion_iou(p, g))
                    
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_iou = np.mean(val_ious)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "checkpoints/glenda_unet_best.pth")
            marker = "⭐ (Best)"
        else:
            marker = ""
            
        print(f"[{epoch+1:2d}/{epochs:2d}]   | {avg_train_loss:<12.4f} | {avg_val_loss:<10.4f} | {avg_val_iou:<12.4f} | {current_lr:<10.2e} {marker}")
        
    # Save final checkpoint
    torch.save(model.state_dict(), "checkpoints/glenda_unet.pth")
    print("=" * 70)
    print(f"🎉 Success! Training complete across {epochs} epochs.")
    print(f"📁 Best checkpoint: checkpoints/glenda_unet_best.pth (Val Loss: {best_val_loss:.4f}, IoU: {best_iou:.4f})")
    print(f"📁 Final checkpoint: checkpoints/glenda_unet.pth")


if __name__ == "__main__":
    train_model(epochs=20)
