#!/usr/bin/env python3
"""
visualize_attention_plus.py

Wrapper/enhancer for the repo's `visualize_attention.py`.

Features:
- Accepts local image path or image URL. If URL, downloads it.
- Calls the repo's `visualize_attention.py` (using the same Python interpreter) so the model & attention maps are generated exactly the same way as before.
- After `visualize_attention.py` finishes, this script searches the `attention_maps_output` folder for the produced images for the given input image, composes a neat grid (raw image + head-1..head-N + head-mean) and saves a combined PNG.
- Also displays the combined image (if running in a notebook environment with a display) and prints the saved path.

Usage (example):
    python visualize_attention_plus.py \
        --model_path models/Q2L-CvT_w24-384.pkl \
        --pic_path https://picsum.photos/384/384 \
        --config_file configs/coco_cvt_w24.json \
        --threshold 0.5 \
        --img_size 384 \
        --backbone CvT-w24 \
        --output_dir attention_maps_output

Note: run this script inside the same environment you use to run visualize_attention.py (e.g. the conda env `q2l_env`).

If `visualize_attention.py` names output files differently in your fork, adjust the `find_matching_outputs` function accordingly.
"""

import argparse
import os
import sys
import subprocess
import shutil
import time
from urllib.parse import urlparse

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    raise RuntimeError("Pillow is required. Please install with `pip install pillow` in your environment.")

try:
    import requests
except Exception:
    raise RuntimeError("requests is required. Please install with `pip install requests` in your environment.")

import matplotlib.pyplot as plt


def download_image(url, out_path):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    return out_path


def is_url(path):
    try:
        p = urlparse(path)
        return p.scheme in ("http", "https")
    except Exception:
        return False


def find_matching_outputs(output_dir, basename):
    """
    Return a list of file paths that match the given basename in the output_dir.
    We expect files like:
      {basename}_raw.png
      {basename}_head-1.png
      {basename}_head-2.png
      {basename}_head-mean.png
    But we'll be permissive and match any file that contains the basename.
    """
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if basename in f and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, f))
    return sorted(all_files)


def compose_grid(image_paths, save_path, title_text=None, per_row=None):
    """
    Compose images in image_paths horizontally (1 row) or grid if per_row is set.
    We'll put the raw image first (if present), then the heads in filename order.
    """
    imgs = [Image.open(p).convert('RGBA') for p in image_paths]

    # Normalize heights: scale all images to same height (height of the tallest)
    heights = [im.height for im in imgs]
    max_h = max(heights)
    resized = []
    for im in imgs:
        if im.height != max_h:
            new_w = int(im.width * (max_h / im.height))
            im = im.resize((new_w, max_h), Image.LANCZOS)
        resized.append(im)

    # If per_row is None: single row
    if per_row is None:
        per_row = len(resized)

    rows = []
    for i in range(0, len(resized), per_row):
        row_imgs = resized[i:i+per_row]
        total_w = sum(im.width for im in row_imgs)
        row_img = Image.new('RGBA', (total_w, max_h), (255,255,255,255))
        x = 0
        for im in row_imgs:
            row_img.paste(im, (x, 0), mask=im)
            x += im.width
        rows.append(row_img)

    grid_w = max(r.width for r in rows)
    grid_h = sum(r.height for r in rows)
    grid = Image.new('RGBA', (grid_w, grid_h + 60), (255,255,255,255))

    y = 0
    for r in rows:
        grid.paste(r, (0, y), mask=r)
        y += r.height

    # Draw title text if provided
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    if title_text:
        draw.text((10, grid_h + 10), title_text, fill=(0,0,0), font=font)

    # Save as PNG (flatten to RGB)
    rgb = Image.new('RGB', grid.size, (255,255,255))
    rgb.paste(grid, mask=grid.split()[3])
    rgb.save(save_path)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--pic_path', required=True, help='Local path or URL to image')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--backbone', type=str, default='CvT-w24')
    parser.add_argument('--output_dir', default='attention_maps_output')
    parser.add_argument('--keep_temp', action='store_true', help='Keep downloaded temp image')
    parser.add_argument('--per_row', type=int, default=None, help='Images per row in composite grid')
    args, unknown = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare input image: download if URL
    input_is_url = is_url(args.pic_path)
    if input_is_url:
        parsed = urlparse(args.pic_path)
        basename = os.path.basename(parsed.path) or 'input_image'
        tmp_local = os.path.join('/tmp', f'downloaded_{int(time.time())}_{basename}')
        print(f"Downloading image URL to: {tmp_local}")
        download_image(args.pic_path, tmp_local)
        pic_for_visualize = tmp_local
    else:
        pic_for_visualize = args.pic_path

    # Call visualize_attention.py using same python interpreter
    vis_script = os.path.join(os.getcwd(), 'visualize_attention.py')
    if not os.path.exists(vis_script):
        print('ERROR: visualize_attention.py not found in current working directory.')
        sys.exit(1)

    cmd = [sys.executable, vis_script,
           '--model_path', args.model_path,
           '--pic_path', pic_for_visualize,
           '--config_file', args.config_file,
           '--threshold', str(args.threshold),
           '--img_size', str(args.img_size),
           '--backbone', args.backbone]

    print('Running original visualize script to generate attention maps...')
    print(' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print('visualize_attention.py returned non-zero exit status.')
        raise

    # Try to find generated outputs
    basename = os.path.splitext(os.path.basename(pic_for_visualize))[0]
    matches = find_matching_outputs(args.output_dir, basename)
    if not matches:
        # If nothing found by basename, try to collect any images created recently
        print('No matching files found by basename. Scanning output folder for recent images...')
        all_imgs = []
        for root, dirs, files in os.walk(args.output_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_imgs.append(os.path.join(root, f))
        # choose the most recent 6-10 images
        all_imgs = sorted(all_imgs, key=lambda p: os.path.getmtime(p), reverse=True)
        matches = all_imgs[:6]

    if not matches:
        print('Could not find any attention map images in the output folder. Exiting.')
        sys.exit(1)

    # Heuristics: prefer raw image first (largest file or contains 'raw')
    matches_sorted = sorted(matches, key=lambda p: (0 if 'raw' in os.path.basename(p).lower() else 1, -os.path.getsize(p)))

    # Compose grid and save
    out_name = f'combined_{basename}.png'
    out_path = os.path.join(args.output_dir, out_name)
    title_text = f'Composite: {basename}  (generated by visualize_attention_plus)'

    print('Composing final grid image...')
    compose_grid(matches_sorted, out_path, title_text=title_text, per_row=args.per_row)

    print(f'Combined image saved to: {out_path}')

    # Display inline if possible
    try:
        img = Image.open(out_path)
        plt.figure(figsize=(10,10))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    except Exception:
        pass

    # Clean up
    if input_is_url and not args.keep_temp:
        try:
            os.remove(pic_for_visualize)
        except Exception:
            pass


if __name__ == '__main__':
    main()
