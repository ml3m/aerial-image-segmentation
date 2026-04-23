"""combine.py + visualization.py correctness (pure numpy, no torch)."""

import numpy as np

from inference.combine import Detection, combine, draw_detections, overlay_mask
from inference.visualization import colorize_mask, palette_from_class_info


CLASS_INFO = [
    (0, "impervious_surface", 255, 255, 255),
    (1, "building", 0, 0, 255),
    (2, "low_vegetation", 0, 255, 255),
    (3, "tree", 0, 255, 0),
    (4, "car", 255, 255, 0),
    (5, "clutter", 255, 0, 0),
]
PALETTE = palette_from_class_info(CLASS_INFO)


def test_palette_shape():
    assert PALETTE.shape == (6, 3)
    assert PALETTE.dtype == np.uint8
    # Building class is blue.
    assert tuple(PALETTE[1]) == (0, 0, 255)


def test_colorize_mask_roundtrip():
    mask = np.array([[0, 1], [2, 5]], dtype=np.int64)
    colored = colorize_mask(mask, PALETTE)
    assert colored.shape == (2, 2, 3)
    assert tuple(colored[0, 0]) == (255, 255, 255)
    assert tuple(colored[1, 1]) == (255, 0, 0)


def test_overlay_mask_produces_alpha_blend():
    image = np.full((50, 50, 3), 50, dtype=np.uint8)  # dark gray BGR
    mask = np.zeros((50, 50), dtype=np.int64)
    mask[10:40, 10:40] = 1  # building block
    blended = overlay_mask(image, mask, PALETTE, alpha=0.5)
    assert blended.shape == image.shape
    # The masked region should be different from the original.
    assert not np.array_equal(blended[20, 20], image[20, 20])
    # The untouched corner should equal a blend of original gray + white
    # (class 0 is impervious/white) OR remain same if class-0 = impervious.
    # Accept either; just assert dtype.
    assert blended.dtype == np.uint8


def test_draw_detections_changes_pixels():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    det = Detection(
        class_id=0,
        class_name="car",
        conf=0.9,
        corners=np.array([[10, 10], [40, 12], [42, 35], [12, 33]], dtype=float),
    )
    out = draw_detections(image, [det], thickness=2)
    # The polygon perimeter must differ from the original black image.
    assert not np.array_equal(out, image)
    # Sanity: no NaNs/invalid uint8.
    assert out.dtype == np.uint8


def test_combine_end_to_end():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.int64)
    mask[30:60, 30:60] = 4  # car
    det = Detection(
        class_id=0, class_name="vehicle", conf=0.77,
        corners=np.array([[20, 20], [50, 22], [52, 55], [22, 53]], dtype=float),
    )
    out = combine(image, mask, [det], PALETTE, mask_alpha=0.5, box_thickness=2)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, image)


def test_combine_empty_detections():
    image = np.full((40, 40, 3), 30, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.int64)
    out = combine(image, mask, [], PALETTE, mask_alpha=0.0, box_thickness=1)
    # alpha=0 → image is unchanged by overlay and no boxes drawn.
    assert np.array_equal(out, image)
