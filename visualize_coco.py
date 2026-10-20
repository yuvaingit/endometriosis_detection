#!/usr/bin/env python3
"""
COCO Dataset Visualizer for Glenda Dataset (Endometriosis Ultrasound / Laparoscopy)
Loads coco.json, draws segmentation polygons, bounding boxes, and category labels,
and exports high-resolution visual overlays and an interactive HTML gallery.
"""

import os
import sys
from pathlib import Path

# Auto-reexec with virtual environment if PIL is not installed in current Python
try:
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
            print(f"[visualize_coco] Auto-reexecuting with virtual environment: {venv_py}")
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    print("Error: Required library 'Pillow' is not installed in current Python.")
    print("Please run with:")
    print("  ~/Documents/endometriosis_ultrasound_ai_starter/.venv/bin/python visualize_coco.py")
    sys.exit(1)

import json
import argparse
from typing import Dict, List, Any, Tuple


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color (#RRGGBB) to (R, G, B) tuple."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (255, 0, 0)


def draw_annotations(
    image: Image.Image,
    annotations: List[Dict[str, Any]],
    categories: Dict[int, Dict[str, Any]],
    alpha: int = 110,
    outline_width: int = 2
) -> Image.Image:
    """
    Render polygon segmentations and bounding boxes with labels onto the image.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # First pass: filled semi-transparent polygons
    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_info = categories.get(cat_id, {})
        color_hex = ann.get("color") or cat_info.get("color", "#FF3B30")
        r, g, b = hex_to_rgb(color_hex)

        segmentations = ann.get("segmentation", [])
        for seg in segmentations:
            if len(seg) >= 6:
                polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg) - 1, 2)]
                overlay_draw.polygon(polygon, fill=(r, g, b, alpha))

    combined = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(combined)

    font = ImageFont.load_default()

    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_info = categories.get(cat_id, {})
        cat_name = cat_info.get("name", f"Class {cat_id}")
        short_name = cat_name.split("_", 1)[-1] if "_" in cat_name else cat_name
        color_hex = ann.get("color") or cat_info.get("color", "#FF3B30")
        r, g, b = hex_to_rgb(color_hex)

        # Polygon boundary
        for seg in ann.get("segmentation", []):
            if len(seg) >= 6:
                polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg) - 1, 2)]
                draw.line(polygon + [polygon[0]], fill=(r, g, b, 255), width=outline_width)

        # Bounding box
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            bx, by, bw, bh = bbox
            draw.rectangle([bx, by, bx + bw, by + bh], outline=(r, g, b, 255), width=outline_width)

            label_text = f"{short_name}"
            if "area" in ann and ann["area"]:
                label_text += f" ({int(ann['area'])}px)"

            if hasattr(draw, "textbbox"):
                t_bbox = draw.textbbox((bx, by), label_text, font=font)
                tw, th = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
            else:
                tw, th = len(label_text) * 6, 10

            badge_y = max(0, by - th - 6)
            draw.rectangle([bx, badge_y, bx + tw + 6, badge_y + th + 6], fill=(r, g, b, 230))
            draw.text((bx + 3, badge_y + 2), label_text, fill=(255, 255, 255, 255), font=font)

    return combined.convert("RGB")


def create_side_by_side(
    original: Image.Image,
    mask: Image.Image,
    annotated: Image.Image,
    title_text: str = ""
) -> Image.Image:
    """Create a 3-panel comparison: Original | GT Mask | Annotated Overlay."""
    w, h = original.size
    panel = Image.new("RGB", (w * 3 + 20, h + 40), (25, 25, 30))
    draw = ImageDraw.Draw(panel)

    panel.paste(original, (5, 35))
    if mask:
        panel.paste(mask.convert("RGB").resize((w, h)), (w + 10, 35))
    panel.paste(annotated, (w * 2 + 15, 35))

    font = ImageFont.load_default()
    draw.text((10, 10), "1. Original Frame", fill=(220, 220, 220), font=font)
    draw.text((w + 15, 10), "2. Ground-Truth Mask (annots/)", fill=(220, 220, 220), font=font)
    draw.text((w * 2 + 20, 10), f"3. Annotated Overlay ({title_text})", fill=(220, 220, 220), font=font)
    return panel


def generate_html_gallery(
    output_dir: Path,
    sample_records: List[Dict[str, Any]],
    categories: Dict[int, Dict[str, Any]],
    total_images: int,
    total_annots: int
) -> None:
    badge_items = []
    for cat in categories.values():
        c_color = cat.get("color", "#4f46e5")
        c_name = cat.get("name", "")
        badge_items.append(f'<span class="badge" style="background-color: {c_color}; color: #fff;">{c_name}</span>')
    cat_badges = " ".join(badge_items)

    cards_html = []
    for rec in sample_records:
        cats_in_img = ", ".join(rec["classes"])
        f_name = rec["file_name"]
        a_count = rec["annot_count"]
        o_rel = rec["overlay_rel"]
        s_rel = rec.get("sbs_rel")

        sbs_snippet = ""
        if s_rel:
            sbs_snippet = f'<br><a class="side-link" href="{s_rel}" target="_blank">View Side-by-Side (Original | Mask | Overlay)</a>'

        card = f"""<div class="card">
  <div class="card-header">
    <strong>{f_name}</strong>
    <span class="pill">{a_count} annotations</span>
  </div>
  <a href="{o_rel}" target="_blank">
    <img src="{o_rel}" alt="{f_name}" loading="lazy" />
  </a>
  <div class="card-footer">
    <span class="tag">Classes: {cats_in_img}</span>
    {sbs_snippet}
  </div>
</div>"""
        cards_html.append(card)

    cards_str = "\n".join(cards_html)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Glenda Dataset COCO Visualizations</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f1117;
            color: #e2e8f0;
            padding: 24px;
        }
        header {
            max-width: 1400px;
            margin: 0 auto 28px;
            padding: 20px 24px;
            background: #1a1f2c;
            border-radius: 12px;
            border: 1px solid #2d3748;
        }
        h1 { font-size: 24px; margin-bottom: 8px; color: #f8fafc; }
        .stats { color: #94a3b8; font-size: 14px; margin-bottom: 12px; }
        .categories { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
        .badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            text-shadow: 0 1px 2px rgba(0,0,0,0.6);
        }
        .grid {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 20px;
        }
        .card {
            background: #1a1f2c;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #2d3748;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.4);
            border-color: #4a5568;
        }
        .card-header {
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            background: #232936;
        }
        .pill {
            background: #3b82f6;
            color: white;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
        }
        .card img {
            width: 100%;
            display: block;
            background: #000;
            cursor: pointer;
        }
        .card-footer {
            padding: 12px 16px;
            font-size: 12px;
            color: #94a3b8;
            line-height: 1.6;
        }
        .tag { color: #cbd5e1; font-weight: 500; }
        .side-link {
            color: #60a5fa;
            text-decoration: none;
            font-weight: 500;
        }
        .side-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <header>
        <h1>🔬 Glenda Dataset & COCO Annotations Gallery</h1>
        <div class="stats">
            Total Dataset: <strong>__TOTAL_IMAGES__</strong> frames &bull;
            Annotations: <strong>__TOTAL_ANNOTS__</strong> lesions &bull;
            Visualized in this batch: <strong>__SAMPLE_COUNT__</strong> samples
        </div>
        <div class="categories">
            __CAT_BADGES__
        </div>
    </header>

    <div class="grid">
        __CARDS_STR__
    </div>
</body>
</html>"""

    html_content = html_template.replace("__TOTAL_IMAGES__", str(total_images))
    html_content = html_content.replace("__TOTAL_ANNOTS__", str(total_annots))
    html_content = html_content.replace("__SAMPLE_COUNT__", str(len(sample_records)))
    html_content = html_content.replace("__CAT_BADGES__", cat_badges)
    html_content = html_content.replace("__CARDS_STR__", cards_str)

    (output_dir / "index.html").write_text(html_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="COCO Visualizer for Glenda Endometriosis Dataset")
    parser.add_argument("--coco", type=str, default="coco.json", help="Path to coco.json annotation file")
    parser.add_argument("--frames", type=str, default="frames", help="Directory containing frame images")
    parser.add_argument("--annots", type=str, default="annots", help="Directory containing mask images")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of sample images to render and save")
    parser.add_argument("--output-dir", type=str, default="visualized_samples", help="Directory to store visualizations")
    parser.add_argument("--side-by-side", action="store_true", default=True, help="Generate 3-panel comparison images")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    coco_path = Path(args.coco) if os.path.isabs(args.coco) else base_dir / args.coco
    frames_dir = Path(args.frames) if os.path.isabs(args.frames) else base_dir / args.frames
    annots_dir = Path(args.annots) if os.path.isabs(args.annots) else base_dir / args.annots
    output_dir = Path(args.output_dir) if os.path.isabs(args.output_dir) else base_dir / args.output_dir

    if not coco_path.exists():
        print(f"Error: COCO file not found at {coco_path}")
        sys.exit(1)

    print(f"Loading COCO annotations from: {coco_path}")
    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat for cat in coco_data.get("categories", [])}
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    print(f"Found {len(images)} images, {len(annotations)} annotations, and {len(categories)} categories.")

    cat_counts = {}
    for ann in annotations:
        c_id = ann.get("category_id")
        cat_counts[c_id] = cat_counts.get(c_id, 0) + 1

    print("\n" + "="*65)
    print("DATASET CLASS DISTRIBUTION:")
    print("-" * 65)
    for c_id, cat in categories.items():
        count = cat_counts.get(c_id, 0)
        print(f"  ID {c_id:2d} | {cat['name']:26s} | {count:4d} lesions | Color: {cat.get('color', 'N/A')}")
    print("=" * 65 + "\n")

    img_to_annots = {}
    for ann in annotations:
        img_id = ann["image_id"]
        img_to_annots.setdefault(img_id, []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    sbs_dir = output_dir / "side_by_side"
    if args.side_by_side:
        sbs_dir.mkdir(exist_ok=True)

    # Pick diverse sample images representing every class
    selected_img_ids = []
    for c_id in categories.keys():
        for img in images:
            ann_list = img_to_annots.get(img["id"], [])
            if any(a.get("category_id") == c_id for a in ann_list):
                if img["id"] not in selected_img_ids:
                    selected_img_ids.append(img["id"])
                if len([i for i in selected_img_ids if any(a.get("category_id") == c_id for a in img_to_annots.get(i, []))]) >= 3:
                    break

    for img in images:
        if len(selected_img_ids) >= args.num_samples:
            break
        if img["id"] not in selected_img_ids and img["id"] in img_to_annots:
            selected_img_ids.append(img["id"])

    sample_images = [img for img in images if img["id"] in selected_img_ids]
    print(f"Rendering visualizations for {len(sample_images)} selected diverse samples...")

    gallery_records = []

    for idx, img_info in enumerate(sample_images):
        file_name = img_info["file_name"]
        img_path = frames_dir / file_name

        if not img_path.exists():
            continue

        raw_img = Image.open(img_path).convert("RGB")
        ann_list = img_to_annots.get(img_info["id"], [])

        annotated_img = draw_annotations(raw_img, ann_list, categories)

        overlay_filename = f"overlay_{Path(file_name).stem}.jpg"
        overlay_path = overlays_dir / overlay_filename
        annotated_img.save(overlay_path, quality=95)

        sbs_rel = None
        if args.side_by_side:
            mask_filename = Path(file_name).with_suffix(".png").name
            mask_path = annots_dir / mask_filename
            mask_img = Image.open(mask_path) if mask_path.exists() else None

            class_names = [categories[a["category_id"]]["name"].split("_", 1)[-1] for a in ann_list if a.get("category_id") in categories]
            sbs_img = create_side_by_side(raw_img, mask_img, annotated_img, ", ".join(set(class_names)))
            sbs_filename = f"sbs_{Path(file_name).stem}.jpg"
            sbs_path = sbs_dir / sbs_filename
            sbs_img.save(sbs_path, quality=95)
            sbs_rel = f"side_by_side/{sbs_filename}"

        classes_found = list(set([categories[a["category_id"]]["name"] for a in ann_list if a.get("category_id") in categories]))

        gallery_records.append({
            "file_name": file_name,
            "annot_count": len(ann_list),
            "classes": classes_found,
            "overlay_rel": f"overlays/{overlay_filename}",
            "sbs_rel": sbs_rel,
        })
        print(f"  [{idx+1}/{len(sample_images)}] Processed {file_name} ({len(ann_list)} annotations)")

    generate_html_gallery(output_dir, gallery_records, categories, len(images), len(annotations))

    print("\n" + "=" * 65)
    print(f"SUCCESS: Generated {len(gallery_records)} visualizations!")
    print(f"Visualizations directory: {output_dir}")
    print(f"Interactive HTML Gallery: file://{output_dir}/index.html")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
