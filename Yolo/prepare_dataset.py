"""
Prepare the ISPRS Potsdam dataset for YOLO segmentation training.

This script takes the raw downloaded Kaggle data and:
  1. Converts RGB segmentation masks to YOLO polygon label files
  2. Splits images/labels into train/val sets
  3. Creates the final directory structure YOLO expects

Expected raw download structure (adjust paths as needed):
  raw_data/
    images/         (or any folder with the aerial images)
    masks/          (or any folder with the RGB segmentation masks)

Final output:
  dataset/
    images/
      train/
      val/
    labels/
      train/
      val/
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from convert_masks_to_yolo import convert_dataset, auto_detect_colors


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_matching_pairs(image_dir: str, mask_dir: str):
    """
    Match images to their corresponding masks by filename stem.
    Returns list of (image_path, mask_path) tuples.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    images = {f.stem: f for f in image_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS}
    masks = {f.stem: f for f in mask_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS}

    # Try exact stem match first
    paired_stems = set(images.keys()) & set(masks.keys())

    if not paired_stems:
        print("  No exact filename matches. Trying fuzzy matching...")
        fuzzy_pairs = set()

        # Strategy 1: strip common suffixes from mask names (_label, _mask, etc.)
        mask_suffixes = ["_label", "_mask", "_gt", "_seg", "_labels", "_masks"]
        for mask_stem in masks:
            for suffix in mask_suffixes:
                if mask_stem.endswith(suffix):
                    base = mask_stem[: -len(suffix)]
                    if base in images:
                        fuzzy_pairs.add((base, mask_stem))

        # Strategy 2: prefix swap (e.g. Image_0 <-> Label_0, img_0 <-> mask_0)
        if not fuzzy_pairs:
            import re
            prefix_map = {
                "Label": "Image", "label": "image", "Mask": "Image", "mask": "image",
                "GT": "Image", "gt": "image", "Seg": "Image", "seg": "image",
            }
            for mask_stem in masks:
                for mask_prefix, img_prefix in prefix_map.items():
                    if mask_stem.startswith(mask_prefix):
                        candidate = img_prefix + mask_stem[len(mask_prefix):]
                        if candidate in images:
                            fuzzy_pairs.add((candidate, mask_stem))
                            break

        # Strategy 3: match by extracting the numeric ID from filenames
        if not fuzzy_pairs:
            img_by_num = {}
            for stem in images:
                nums = re.findall(r"\d+", stem)
                if nums:
                    img_by_num[nums[-1]] = stem
            for mask_stem in masks:
                nums = re.findall(r"\d+", mask_stem)
                if nums and nums[-1] in img_by_num:
                    fuzzy_pairs.add((img_by_num[nums[-1]], mask_stem))

        if fuzzy_pairs:
            pairs = [(images[img_stem], masks[msk_stem]) for img_stem, msk_stem in sorted(fuzzy_pairs)]
            print(f"  Matched {len(pairs)} pairs via fuzzy matching.")
            return pairs

        print("\n  [WARNING] Could not match images to masks by filename.")
        print("  Image stems sample:", list(images.keys())[:5])
        print("  Mask stems sample:", list(masks.keys())[:5])
        print("  Please ensure image and mask filenames share a common base name.\n")
        return []

    pairs = [(images[stem], masks[stem]) for stem in sorted(paired_stems)]
    print(f"  Matched {len(pairs)} image/mask pairs by filename.")
    return pairs


def prepare(args):
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    print(f"Image directory : {image_dir}")
    print(f"Mask directory  : {mask_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Val split       : {args.val_split}")
    print()

    # Step 0: Detect colors in masks
    if args.detect_colors:
        print("--- Scanning mask colors ---")
        auto_detect_colors(str(mask_dir))

    # Step 1: Match images to masks
    print("--- Matching images to masks ---")
    pairs = find_matching_pairs(str(image_dir), str(mask_dir))
    if not pairs:
        sys.exit(1)

    # Step 2: Convert masks to YOLO labels in a temp directory
    temp_labels = output_dir / "_temp_labels"
    print("\n--- Converting masks to YOLO labels ---")
    convert_dataset(str(mask_dir), str(temp_labels), args.min_area, args.epsilon)

    # Step 3: Train/val split
    print("--- Splitting into train/val ---")
    indices = np.arange(len(pairs))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_split, random_state=args.seed, shuffle=True
    )
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}\n")

    # Step 4: Copy files into final structure
    for split, idxs in [("train", train_idx), ("val", val_idx)]:
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i in idxs:
            img_src, mask_src = pairs[i]

            shutil.copy2(img_src, img_out / img_src.name)

            label_file = temp_labels / (mask_src.stem + ".txt")
            if label_file.exists():
                # Save label with same stem as the image
                shutil.copy2(label_file, lbl_out / (img_src.stem + ".txt"))

    # Cleanup temp directory
    shutil.rmtree(temp_labels, ignore_errors=True)

    print(f"--- Dataset prepared at '{output_dir}' ---")
    print(f"  {output_dir}/images/train/  ({len(train_idx)} images)")
    print(f"  {output_dir}/images/val/    ({len(val_idx)} images)")
    print(f"  {output_dir}/labels/train/  (YOLO .txt labels)")
    print(f"  {output_dir}/labels/val/    (YOLO .txt labels)")
    print()
    print(f"Update 'path' in dataset.yaml to point to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ISPRS Potsdam dataset for YOLO segmentation training."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing the raw aerial images.",
    )
    parser.add_argument(
        "--mask_dir", type=str, required=True,
        help="Directory containing the RGB segmentation masks.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset",
        help="Output directory for the prepared dataset. Default: dataset/",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Fraction of data for validation. Default: 0.2",
    )
    parser.add_argument(
        "--min_area", type=int, default=100,
        help="Min contour area for polygon extraction. Default: 100",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.002,
        help="Polygon simplification factor. Default: 0.002",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split. Default: 42",
    )
    parser.add_argument(
        "--detect_colors", action="store_true",
        help="Scan mask colors before converting (recommended on first run).",
    )
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
