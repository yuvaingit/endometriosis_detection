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
        Path(__file__).resolve().parent.parent / ".venv/bin/python",
    ]
    for venv_py in candidate_venvs:
        if venv_py.exists():
            print(f"[predict_lesion] Auto-reexecuting with virtualenv: {venv_py}")
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    print("Error: PyTorch not found. Run with starter virtualenv:")
    print("  ~/Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python predict_lesion.py")
    sys.exit(1)

# Ensure parent directory (Glenda_v1.5_classes) is on sys.path
file_dir = Path(__file__).resolve().parent
project_root = file_dir.parent if (file_dir.parent / "train_unet.py").exists() else file_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from train_unet import SimpleUNet

# Class names and RGB color palette for Glenda dataset
CLASS_NAMES = {
    0: "Background",
    1: "Endo-Peritoneum",
    2: "Endo-Ovar",
    3: "Endo-TIE",
    4: "Endo-Uterus",
}

CLASS_COLORS = {
    0: (20, 22, 28),
    1: (190, 62, 204),  # Purple
    2: (93, 89, 254),   # Blue
    3: (145, 210, 138), # Green
    4: (238, 236, 50),  # Yellow
}

def colorize_mask(mask_arr: np.ndarray) -> Image.Image:
    """Map class ID indices to distinct RGB colors."""
    h, w = mask_arr.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id, rgb in CLASS_COLORS.items():
        color_img[mask_arr == c_id] = rgb
    return Image.fromarray(color_img)

def create_overlay(orig_img: Image.Image, mask_arr: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend segmented lesion predictions onto the input frame."""
    color_mask = colorize_mask(mask_arr).convert("RGBA")
    base = orig_img.convert("RGBA")
    
    # Only make non-background pixels translucent
    mask_alpha = (mask_arr > 0).astype(np.uint8) * int(255 * alpha)
    color_mask.putalpha(Image.fromarray(mask_alpha, mode="L"))
    
    blended = Image.alpha_composite(base, color_mask)
    return blended.convert("RGB")

def predict_single_image(image_path, checkpoint_path=None, output_path=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Running inference on device: {device}")
    
    # 1. Resolve paths
    if checkpoint_path is None:
        best_ckpt = project_root / "checkpoints/glenda_unet_best.pth"
        std_ckpt = project_root / "checkpoints/glenda_unet.pth"
        checkpoint_path = best_ckpt if best_ckpt.exists() else std_ckpt
    else:
        checkpoint_path = Path(checkpoint_path) if os.path.isabs(checkpoint_path) else project_root / checkpoint_path
        
    img_path = Path(image_path) if os.path.isabs(image_path) else project_root / image_path
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
        
    if not img_path.exists():
        print(f"❌ Error: Image not found at {img_path}")
        return
        
    # 2. Load Trained Model
    model = SimpleUNet(in_channels=3, num_classes=5).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model weights from: {checkpoint_path.name}")
    print(f"🖼️  Input frame: {img_path.name}")
    
    # 3. Load and Preprocess Target Image
    original_image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = original_image.size
    
    # Resize to model input size (256x256)
    input_img = original_image.resize((256, 256), Image.Resampling.BILINEAR)
    image_tensor = torch.tensor(np.array(input_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 4. Model Inference
    with torch.no_grad():
        output = model(image_tensor) # Shape: [1, 5, 256, 256]
        pred_mask_256 = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
    # Resize predicted mask back to original image dimensions
    pred_mask_img = Image.fromarray(pred_mask_256).resize((orig_w, orig_h), Image.Resampling.NEAREST)
    pred_mask_full = np.array(pred_mask_img)
    
    # Detect classes present in prediction
    unique_classes = [int(c) for c in np.unique(pred_mask_full) if c > 0]
    detected_names = [CLASS_NAMES.get(c, f"Class {c}") for c in unique_classes]
    
    print("\n" + "="*50)
    print("DETECTION SUMMARY:")
    if detected_names:
        for c in unique_classes:
            px_count = int((pred_mask_full == c).sum())
            pct = (px_count / (orig_w * orig_h)) * 100
            print(f"  • {CLASS_NAMES.get(c)}: {px_count} pixels ({pct:.2f}% of frame)")
    else:
        print("  • No lesion detected (clean background).")
    print("="*50 + "\n")
    
    # 5. Create 3-Panel Visual Comparison: [Original | Color-Coded Mask | Overlay]
    color_mask = colorize_mask(pred_mask_full)
    overlay = create_overlay(original_image, pred_mask_full)
    
    panel_w = orig_w * 3 + 40
    panel_h = orig_h + 45
    comparison = Image.new("RGB", (panel_w, panel_h), (25, 27, 34))
    draw = ImageDraw.Draw(comparison)
    
    comparison.paste(original_image, (10, 35))
    comparison.paste(color_mask, (orig_w + 20, 35))
    comparison.paste(overlay, (orig_w * 2 + 30, 35))
    
    font = ImageFont.load_default()
    draw.text((10, 10), "1. Input Laparoscopic Frame", fill=(220, 220, 220), font=font)
    draw.text((orig_w + 20, 10), "2. Predicted Lesion Mask", fill=(220, 220, 220), font=font)
    label_classes = f" ({', '.join(detected_names)})" if detected_names else " (No lesion)"
    draw.text((orig_w * 2 + 30, 10), f"3. Lesion Overlay{label_classes}", fill=(220, 220, 220), font=font)
    
    if output_path is None:
        output_path = project_root / f"predicted_{img_path.stem}.jpg"
    else:
        output_path = Path(output_path) if os.path.isabs(output_path) else project_root / output_path
        
    comparison.save(output_path, quality=95)
    print(f"🎉 Success! Prediction comparison saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_frame = sys.argv[1]
    else:
        sample_frame = "frames/c_170_v_(video_5222.mp4)_f_64.jpg"
    predict_single_image(sample_frame)
