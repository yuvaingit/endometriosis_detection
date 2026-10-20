import os
import sys
import json
from pathlib import Path

# Auto-reexec with virtual environment if torch / PIL are not installed in current Python
try:
    import torch
    from PIL import Image, ImageDraw
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    candidate_venvs = [
        Path.home() / "Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python",
        Path.home() / "Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python3",
        Path(__file__).resolve().parent / ".venv/bin/python",
        Path(__file__).resolve().parent / "venv/bin/python",
    ]
    for venv_py in candidate_venvs:
        if venv_py.exists():
            print(f"[dataset_loader] Auto-reexecuting with virtualenv: {venv_py}")
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    print("Error: PyTorch / PIL not found. Run with starter virtualenv:")
    print("  ~/Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python dataset_loader.py")
    sys.exit(1)


class GlendaDataset(Dataset):
    def __init__(self, coco_json_path, frames_dir, split="train", val_ratio=0.2, seed=42, img_size=(256, 256)):
        base_dir = Path(__file__).resolve().parent
        coco_json_path = Path(coco_json_path) if os.path.isabs(coco_json_path) else base_dir / coco_json_path
        self.frames_dir = Path(frames_dir) if os.path.isabs(frames_dir) else base_dir / frames_dir
        self.img_size = img_size  # (H, W) tuple to standardize image sizes across batches
        
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)
            
        self.categories = {cat['id']: cat['name'] for cat in self.coco.get('categories', [])}
        self.images = self.coco.get('images', [])
        
        # Group annotations by image_id
        self.img_to_annots = {}
        for ann in self.coco.get('annotations', []):
            self.img_to_annots.setdefault(ann['image_id'], []).append(ann)
            
        # Group images by video source (extracted from filename like 'c_178_v_(video_5419.mp4)_f_794.jpg')
        # This ensures no video data leakage between train and val!
        video_groups = {}
        for img in self.images:
            fname = img['file_name']
            video_id = fname.split('_f_')[0] if '_f_' in fname else fname
            video_groups.setdefault(video_id, []).append(img)
            
        # Deterministic split by video groups
        video_keys = sorted(list(video_groups.keys()))
        np.random.seed(seed)
        np.random.shuffle(video_keys)
        
        split_idx = int(len(video_keys) * (1 - val_ratio))
        train_videos = set(video_keys[:split_idx])
        
        # Filter images based on split
        if split == "train":
            self.selected_images = [img for v_id in train_videos for img in video_groups[v_id]]
        else:
            val_videos = set(video_keys[split_idx:])
            self.selected_images = [img for v_id in val_videos for img in video_groups[v_id]]
            
        print(f"📦 Loaded {split.upper()} split: {len(self.selected_images)} images across {len(train_videos) if split=='train' else len(video_keys)-split_idx} videos (target size: {self.img_size}).")

    def __len__(self):
        return len(self.selected_images)

    def __getitem__(self, idx):
        img_info = self.selected_images[idx]
        file_name = img_info['file_name']
        img_path = self.frames_dir / file_name
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        # Get annotations for this image
        annots = self.img_to_annots.get(img_info['id'], [])
        
        # Create a blank segmentation mask (HxW) using PIL Image
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # Draw polygon masks with category IDs
        for ann in annots:
            cat_id = ann.get('category_id', 1)
            for seg in ann.get('segmentation', []):
                if len(seg) >= 6:
                    polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg)-1, 2)]
                    draw.polygon(polygon, outline=cat_id, fill=cat_id)
                    
        # Standardize sizes across diverse input frames
        if self.img_size is not None:
            target_h, target_w = self.img_size
            image = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
            mask_img = mask_img.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
            
        mask = np.array(mask_img, dtype=np.uint8)
                    
        # Convert to tensors
        image_tensor = torch.tensor(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        return image_tensor, mask_tensor

if __name__ == "__main__":
    train_dataset = GlendaDataset("coco.json", "frames", split="train", img_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    images, masks = next(iter(train_loader))
    print(f"🎉 Success! Batch image tensor shape: {images.shape}")
    print(f"🎉 Success! Batch mask tensor shape: {masks.shape}")
    print(f"🏷️  Unique class IDs present in this batch mask: {torch.unique(masks).tolist()}")
