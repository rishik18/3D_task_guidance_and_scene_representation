#!/usr/bin/env python3
"""
Resize all images in INPUT_DIR from 1280×720 to 1920×1080
and write them to OUTPUT_DIR. Requires Pillow (`pip install pillow`).
"""

from pathlib import Path
from PIL import Image

# -------- configure these paths --------
INPUT_DIR  = Path(r"D:\CSU_data\mockup_video_data_v2\tmp_color_199735")
OUTPUT_DIR = Path(r"D:\CSU_data\mockup_video_data_v2\tmp_color_199735_resize")
# ---------------------------------------

TARGET_SIZE = (1920, 1080)          # (width, height)

def main() -> None:
    # Create output directory if it doesn’t exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loop over common image extensions
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_path in INPUT_DIR.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue  # skip non‑images

        try:
            with Image.open(img_path) as im:
                # Optional: verify original resolution
                if im.size != (1280, 720):
                    print(f"Skipping {img_path.name}: size {im.size} is not 1280×720")
                    continue

                # Resize with high‑quality Lanczos filter
                im_resized = im.resize(TARGET_SIZE, Image.LANCZOS)

                # Preserve sub‑folder structure (if any)
                relative = img_path.relative_to(INPUT_DIR)
                save_path = OUTPUT_DIR / relative
                save_path.parent.mkdir(parents=True, exist_ok=True)
                im_resized.save(save_path)

                print(f"✓ {img_path.name} → {save_path}")
        except Exception as e:
            print(f"✗ {img_path.name}: {e}")

    print("\nDone!")

if __name__ == "__main__":
    main()
