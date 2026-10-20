import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import torch

# Ensure parent directory (Glenda_v1.5_classes) is in sys.path
file_dir = Path(__file__).resolve().parent
project_root = file_dir.parent if (file_dir.parent / "train_unet.py").exists() else file_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from train_unet import SimpleUNet

# Page configuration
st.set_page_config(
    page_title="Endometriosis AI Lesion Segmentation",
    page_icon="🔬",
    layout="wide"
)

# Dark theme custom styling
st.markdown("""
    <style>
    .main { background-color: #0f1117; color: #e2e8f0; }
    .stApp { max-width: 1350px; margin: 0 auto; }
    h1 { color: #f8fafc; font-weight: 700; margin-bottom: 0.2rem; }
    .stMetric { background-color: #1a1f2c; padding: 15px; border-radius: 10px; border: 1px solid #2d3748; }
    .legend-badge {
        display: inline-block;
        padding: 4px 10px;
        margin-right: 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Class metadata
CLASS_NAMES = {
    0: "Background",
    1: "Endo-Peritoneum",
    2: "Endo-Ovar",
    3: "Endo-TIE",
    4: "Endo-Uterus"
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (190, 62, 204),   # Purple
    2: (93, 89, 254),    # Blue
    3: (145, 210, 138),  # Green
    4: (238, 236, 50)    # Yellow
}

@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=3, num_classes=5).to(device)
    
    best_path = project_root / "checkpoints/glenda_unet_best.pth"
    std_path = project_root / "checkpoints/glenda_unet.pth"
    checkpoint_path = best_path if best_path.exists() else std_path
        
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, device, checkpoint_path.name
    return None, device, None

# Header
st.title("🔬 Endometriosis Lesion Segmentation AI")
st.markdown("Real-time deep learning multi-class lesion segmentation on laparoscopic video frames.")

# Render Class Color Badges
badge_html = " ".join([
    f'<span class="legend-badge" style="background-color: rgb{CLASS_COLORS[i]};">{CLASS_NAMES[i]}</span>'
    for i in range(1, 5)
])
st.markdown(f"**Target Lesion Classes:** {badge_html}", unsafe_allow_html=True)
st.write("")

model, device, ckpt_name = load_model()

if model is None:
    st.error("❌ Model checkpoint not found in `checkpoints/` folder. Please train the model first!")
else:
    st.sidebar.header("⚙️ Controls & Inputs")
    st.sidebar.caption(f"🚀 Device: `{device}` | Model: `{ckpt_name}`")
    
    alpha_val = st.sidebar.slider("Overlay Transparency", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    
    uploaded_file = st.sidebar.file_uploader("Upload Laparoscopic Frame", type=["jpg", "jpeg", "png"])
    
    # Pre-populated samples
    st.sidebar.markdown("---")
    st.sidebar.subheader("Or Select Sample Frame")
    sample_frames = [
        "c_8_v_(video_148.mp4)_f_1079.jpg",
        "c_170_v_(video_5222.mp4)_f_64.jpg",
        "c_100_v_(video_3062.mp4)_f_1458.jpg"
    ]
    selected_sample = st.sidebar.selectbox("Select Sample", ["None"] + sample_frames)

    image_to_process = None
    file_name = "Image"

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        file_name = uploaded_file.name
    elif selected_sample != "None":
        sample_path = project_root / "frames" / selected_sample
        if sample_path.exists():
            image_to_process = Image.open(sample_path).convert("RGB")
            file_name = selected_sample

    if image_to_process is not None:
        orig_w, orig_h = image_to_process.size
        
        # Preprocessing
        resized_img = image_to_process.resize((256, 256), Image.Resampling.BILINEAR)
        img_tensor = torch.tensor(np.array(resized_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # Resize mask back to original resolution
        mask_pil = Image.fromarray(pred_mask.astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.NEAREST)
        mask_np = np.array(mask_pil)
        
        # Create Color-Coded Overlay
        overlay_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for class_id, color in CLASS_COLORS.items():
            if class_id != 0:
                overlay_rgb[mask_np == class_id] = color
                
        base_np = np.array(image_to_process)
        has_lesion = (mask_np > 0)[:, :, np.newaxis]
        blended_np = np.where(
            has_lesion,
            (base_np * (1 - alpha_val) + overlay_rgb * alpha_val).astype(np.uint8),
            base_np
        )
        blended_img = Image.fromarray(blended_np)
        
        # Calculate Lesion Statistics
        total_pixels = orig_w * orig_h
        class_counts = {cid: int(np.sum(mask_np == cid)) for cid in range(1, 5)}
        
        # Display Results Layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Original Frame")
            st.image(image_to_process, use_container_width=True, caption=f"{file_name} ({orig_w}x{orig_h})")
            
        with col2:
            st.subheader("2. Segmentation Mask")
            st.image(overlay_rgb, use_container_width=True, caption="Multi-Class Color Mask")
            
        with col3:
            st.subheader("3. AI Overlay View")
            st.image(blended_img, use_container_width=True, caption=f"Alpha: {alpha_val:.2f}")
            
        st.markdown("---")
        st.subheader("📊 Lesion Analysis & Pixel Breakdown")
        
        m_cols = st.columns(4)
        for idx, cid in enumerate(range(1, 5)):
            cnt = class_counts[cid]
            pct = (cnt / total_pixels) * 100
            with m_cols[idx]:
                st.metric(
                    label=CLASS_NAMES[cid],
                    value=f"{cnt:,} px",
                    delta=f"{pct:.2f}% of frame" if cnt > 0 else "None detected"
                )
    else:
        st.info("👈 Upload an image or select a sample frame from the sidebar to inspect AI predictions!")
