from __future__ import annotations

"""Phone-region slicing utilities.

This module is intentionally separate from border detection so slicing logic can
evolve independently.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.detect_border import BorderRect


def _load_image(image_path: Path) -> np.ndarray:
    """Load image from disk and raise a clear error when unavailable."""

    if not image_path.exists() or not image_path.is_file():
        msg = f"Image path does not exist or is not a file: {image_path}"
        raise FileNotFoundError(msg)
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Failed to read image: {image_path}"
        raise FileNotFoundError(msg)
    return image


def _tile_positions(total_size: int, tile_size: int, overlap_ratio: float) -> list[int]:
    """Return starting offsets that cover `total_size` with overlap."""

    if total_size <= 0:
        return []

    tile_size = min(tile_size, total_size)
    if tile_size == total_size:
        return [0]

    step = max(1, int(round(tile_size * (1.0 - overlap_ratio))))
    positions = [0]
    while True:
        next_pos = positions[-1] + step
        if next_pos + tile_size >= total_size:
            break
        positions.append(next_pos)

    last_pos = total_size - tile_size
    if positions[-1] != last_pos:
        positions.append(last_pos)
    return positions


def slice_phone_region(
    image_path: Path,
    rect: BorderRect,
    clip_output_dir: Path,
    slice_width: int = 512,
    slice_height: int = 512,
    overlap_ratio: float = 0.2,
) -> int:
    """Slice detected phone region from original image and save clips.

    Clips are written to `<clip_output_dir>/<image_name>/` as
    `<image_name>_c_<index>_x_<abs_x>_y_<abs_y>_w_<w>_h_<h><original_ext>`.
    An index CSV is also written as `<image_name>_slice_index.csv`.
    """

    if rect.w <= 0 or rect.h <= 0:
        return 0
    if slice_width <= 0:
        msg = f"slice_width must be > 0, got {slice_width}"
        raise ValueError(msg)
    if slice_height <= 0:
        msg = f"slice_height must be > 0, got {slice_height}"
        raise ValueError(msg)
    if overlap_ratio < 0 or overlap_ratio >= 1:
        msg = f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
        raise ValueError(msg)

    image = _load_image(image_path)
    image_height, image_width = image.shape[:2]
    pad_x = int(round(slice_width * overlap_ratio))
    pad_y = int(round(slice_height * overlap_ratio))
    x0 = max(0, rect.x - pad_x)
    y0 = max(0, rect.y - pad_y)
    x1 = min(image_width, rect.x + rect.w + pad_x)
    y1 = min(image_height, rect.y + rect.h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return 0

    phone_roi = image[y0:y1, x0:x1]
    roi_height, roi_width = phone_roi.shape[:2]
    tile_w = min(slice_width, roi_width)
    tile_h = min(slice_height, roi_height)

    x_positions = _tile_positions(roi_width, tile_w, overlap_ratio)
    y_positions = _tile_positions(roi_height, tile_h, overlap_ratio)

    image_name = image_path.stem
    suffix = image_path.suffix if image_path.suffix else ".jpg"
    image_clip_dir = clip_output_dir / image_name
    image_clip_dir.mkdir(parents=True, exist_ok=True)
    index_path = image_clip_dir / f"{image_name}_slice_index.csv"
    if index_path.exists() and index_path.is_file():
        index_path.unlink()
    for old_slice in image_clip_dir.glob(f"{image_name}_*{suffix}"):
        if old_slice.is_file():
            old_slice.unlink()

    index_rows: list[tuple[str, int, int, int, int, int]] = []
    clip_index = 1
    for y in y_positions:
        for x in x_positions:
            clip = phone_roi[y : y + tile_h, x : x + tile_w]
            abs_x = x0 + x
            abs_y = y0 + y
            clip_h, clip_w = clip.shape[:2]
            clip_name = (
                f"{image_name}_c_{clip_index}_x_{abs_x}_y_{abs_y}_w_{clip_w}_h_{clip_h}{suffix}"
            )
            clip_path = image_clip_dir / clip_name
            if not cv2.imwrite(str(clip_path), clip):
                msg = f"Failed to write clip image: {clip_path}"
                raise RuntimeError(msg)
            index_rows.append((clip_name, abs_x, abs_y, clip_w, clip_h, clip_index))
            clip_index += 1

    with index_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "x", "y", "w", "h", "c"])
        writer.writerows(index_rows)

    return clip_index - 1
