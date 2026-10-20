import os
import sys
from pathlib import Path

# Auto-reexec with virtual environment if torch is not installed in current Python
try:
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    candidate_venvs = [
        Path.home() / "Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python",
        Path.home() / "Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python3",
        Path(__file__).resolve().parent / ".venv/bin/python",
        Path(__file__).resolve().parent / "venv/bin/python",
    ]
    for venv_py in candidate_venvs:
        if venv_py.exists():
            print(f"[evaluate_unet] Auto-reexecuting with virtualenv: {venv_py}")
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    print("Error: PyTorch/Pillow not found. Run with starter virtualenv:")
    print("  ~/Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python evaluate_unet.py")
    sys.exit(1)

from dataset_loader import GlendaDataset
from train_unet import SimpleUNet

# Color mapping matching Glenda dataset classes:
# 0: Background
# 1: Endo-Peritoneum (#be3ecc)
# 2: Endo-Ovar (#5d59fe)
# 3: Endo-TIE (#91d28a)
# 4: Endo-Uterus (#eeec32)
CLASS_COLORS = {
    0: (15, 17, 23),
    1: (190, 62, 204),
    2: (93, 89, 254),
    3: (145, 210, 138),
    4: (238, 236, 50),
}

def colorize_mask(mask_np: np.ndarray) -> Image.Image:
    """Map class ID indices to distinct RGB colors."""
    h, w = mask_np.shape
    color_arr = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id, rgb in CLASS_COLORS.items():
        color_arr[mask_np == c_id] = rgb
    return Image.fromarray(color_arr)

def create_overlay(orig_img: Image.Image, mask_np: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend segmented lesion predictions onto original frame."""
    color_mask = colorize_mask(mask_np).convert("RGBA")
    base = orig_img.convert("RGBA")
    
    # Only blend non-background pixels
    mask_alpha = (mask_np > 0).astype(np.uint8) * int(255 * alpha)
    color_mask.putalpha(Image.fromarray(mask_alpha, mode="L"))
    
    blended = Image.alpha_composite(base, color_mask)
    return blended.convert("RGB")

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Running evaluation on device: {device}")
    
    # 1. Load Validation Dataset
    val_dataset = GlendaDataset("coco.json", "frames", split="val", val_ratio=0.2, img_size=(256, 256))
    if len(val_dataset) == 0:
        print("❌ Error: Validation dataset is empty!")
        return
        
    # 2. Load Trained Model
    model = SimpleUNet(in_channels=3, num_classes=5).to(device)
    checkpoint_path = "checkpoints/glenda_unet.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✅ Loaded trained model weights from {checkpoint_path}")
    
    # 3. Create evaluation output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 4. Run inference on validation samples
    num_samples_to_eval = min(5, len(val_dataset))
    print(f"🎨 Generating visual comparisons for {num_samples_to_eval} validation samples...")
    
    ious = []
    
    with torch.no_grad():
        for i in range(num_samples_to_eval):
            image_tensor, mask_tensor = val_dataset[i]
            
            # Forward pass
            input_tensor = image_tensor.unsqueeze(0).to(device)
            output = model(input_tensor) # Shape: [1, num_classes, H, W]
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # Shape: [H, W]
            gt_mask = mask_tensor.numpy()
            
            # Binary Lesion IoU
            pred_binary = (pred_mask > 0)
            gt_binary = (gt_mask > 0)
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            sample_iou = (intersection / union) if union > 0 else (1.0 if not pred_binary.any() else 0.0)
            ious.append(sample_iou)
            
            # Image reconstructions
            img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            orig_img = Image.fromarray(img_np)
            gt_img = colorize_mask(gt_mask)
            pred_img = colorize_mask(pred_mask)
            overlay_img = create_overlay(orig_img, pred_mask)
            
            # 4-panel comparison: [Original | Ground Truth | Prediction Mask | Overlay]
            w, h = orig_img.size
            panel_w = w * 4 + 50
            panel_h = h + 45
            comparison = Image.new("RGB", (panel_w, panel_h), (25, 27, 34))
            draw = ImageDraw.Draw(comparison)
            
            comparison.paste(orig_img, (10, 35))
            comparison.paste(gt_img, (w + 20, 35))
            comparison.paste(pred_img, (w * 2 + 30, 35))
            comparison.paste(overlay_img, (w * 3 + 40, 35))
            
            font = ImageFont.load_default()
            draw.text((10, 10), "1. Original Frame", fill=(220, 220, 220), font=font)
            draw.text((w + 20, 10), "2. Ground Truth Mask", fill=(220, 220, 220), font=font)
            draw.text((w * 2 + 30, 10), "3. Model Prediction", fill=(220, 220, 220), font=font)
            draw.text((w * 3 + 40, 10), f"4. Overlay (Lesion IoU: {sample_iou:.2f})", fill=(220, 220, 220), font=font)
            
            save_path = output_dir / f"val_pred_{i+1}.jpg"
            comparison.save(save_path, quality=95)
            print(f"  [{i+1}/{num_samples_to_eval}] Saved: {save_path} (Lesion IoU: {sample_iou:.3f})")

    avg_iou = np.mean(ious)
    print("\n" + "="*50)
    print(f"📊 Mean Sample Lesion IoU: {avg_iou:.3f}")
    print(f"🎉 Success! Evaluation comparisons saved to: {output_dir.resolve()}")
    print("="*50)

if __name__ == "__main__":
    evaluate()
