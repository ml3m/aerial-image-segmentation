"""
Convert RGB segmentation masks (ISPRS Potsdam format) to YOLO segmentation labels.

ISPRS Potsdam class color map (standard):
    0 - Impervious surfaces : (255, 255, 255) white
    1 - Building            : (0,   0,   255) blue
    2 - Low vegetation      : (0,   255, 255) cyan
    3 - Tree                : (0,   255, 0)   green
    4 - Car                 : (255, 255, 0)   yellow
    5 - Clutter/background  : (255, 0,   0)   red

Input:  directory of RGB mask images (.png/.tif)
Output: directory of YOLO-format .txt label files with polygon coordinates
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CLASS_COLOR_MAP = {
    0: (255, 255, 255),  # Impervious surfaces - white
    1: (0,   0,   255),  # Building - blue
    2: (0,   255, 255),  # Low vegetation - cyan
    3: (0,   255,   0),  # Tree - green
    4: (255, 255,   0),  # Car - yellow
    5: (255,   0,   0),  # Clutter/background - red
}

CLASS_NAMES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
]

# Reverse map: RGB tuple -> class index
RGB_TO_CLASS = {v: k for k, v in CLASS_COLOR_MAP.items()}


def mask_to_yolo_polygons(mask_path: str, min_area: int = 100, epsilon_factor: float = 0.002):
    """
    Read an RGB mask and extract YOLO-format polygon annotations.

    Args:
        mask_path: Path to the RGB mask image.
        min_area: Minimum contour area in pixels to keep (filters noise).
        epsilon_factor: Controls polygon simplification (fraction of perimeter).

    Returns:
        List of strings, each in YOLO segmentation format:
        "<class_idx> <x1> <y1> <x2> <y2> ... <xn> <yn>"
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        print(f"  [WARNING] Could not read mask: {mask_path}")
        return []

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    h, w = mask_rgb.shape[:2]
    annotations = []

    for class_idx, color in CLASS_COLOR_MAP.items():
        binary = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            points = approx.reshape(-1, 2)
            norm_points = points.astype(np.float64)
            norm_points[:, 0] /= w
            norm_points[:, 1] /= h

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_points)
            annotations.append(f"{class_idx} {coords}")

    return annotations


def auto_detect_colors(mask_dir: str, sample_count: int = 5):
    """Scan a few masks and report unique RGB values found, to help verify the color map."""
    mask_files = sorted(Path(mask_dir).glob("*"))
    mask_files = [f for f in mask_files if f.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")]
    samples = mask_files[:sample_count]

    all_colors = set()
    for mf in samples:
        img = cv2.imread(str(mf), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape(-1, 3)
        unique = np.unique(pixels, axis=0)
        for c in unique:
            all_colors.add(tuple(c))

    print("\n--- Unique RGB colors found in sample masks ---")
    for color in sorted(all_colors):
        label = RGB_TO_CLASS.get(color, "UNKNOWN")
        if isinstance(label, int):
            label = f"class {label} ({CLASS_NAMES[label]})"
        print(f"  RGB{color} -> {label}")
    print()

    unknown = [c for c in all_colors if c not in RGB_TO_CLASS]
    if unknown:
        print(f"  [WARNING] {len(unknown)} color(s) not in the class map!")
        print("  You may need to update CLASS_COLOR_MAP in this script.\n")

    return all_colors


def convert_dataset(mask_dir: str, output_dir: str, min_area: int = 100, epsilon_factor: float = 0.002):
    """Convert all masks in a directory to YOLO segmentation label files."""
    mask_path = Path(mask_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    mask_files = sorted([f for f in mask_path.iterdir() if f.suffix.lower() in extensions])

    if not mask_files:
        print(f"No mask images found in {mask_dir}")
        sys.exit(1)

    print(f"Found {len(mask_files)} mask files in '{mask_dir}'")
    print(f"Output labels will be written to '{output_dir}'\n")

    stats = {name: 0 for name in CLASS_NAMES}
    empty_count = 0

    for mask_file in tqdm(mask_files, desc="Converting masks"):
        annotations = mask_to_yolo_polygons(str(mask_file), min_area, epsilon_factor)

        label_name = mask_file.stem + ".txt"
        label_path = out_path / label_name

        if annotations:
            with open(label_path, "w") as f:
                f.write("\n".join(annotations) + "\n")
            for ann in annotations:
                cls_idx = int(ann.split()[0])
                stats[CLASS_NAMES[cls_idx]] += 1
        else:
            with open(label_path, "w") as f:
                f.write("")
            empty_count += 1

    print("\n--- Conversion Summary ---")
    total = sum(stats.values())
    print(f"  Total polygon annotations: {total}")
    for name, count in stats.items():
        print(f"  {name:25s}: {count}")
    print(f"  Images with no annotations: {empty_count}")
    print(f"  Label files saved to: {out_path.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RGB segmentation masks to YOLO polygon labels."
    )
    parser.add_argument(
        "--mask_dir", type=str, required=True,
        help="Directory containing RGB mask images.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save YOLO .txt label files.",
    )
    parser.add_argument(
        "--min_area", type=int, default=100,
        help="Minimum contour area (pixels) to keep. Default: 100.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.002,
        help="Polygon simplification factor (fraction of arc length). Default: 0.002.",
    )
    parser.add_argument(
        "--detect_colors", action="store_true",
        help="Scan masks and print detected RGB colors before converting.",
    )

    args = parser.parse_args()

    if args.detect_colors:
        auto_detect_colors(args.mask_dir)

    convert_dataset(args.mask_dir, args.output_dir, args.min_area, args.epsilon)


if __name__ == "__main__":
    main()
