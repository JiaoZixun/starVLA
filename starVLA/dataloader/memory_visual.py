"""Small, deterministic RGB patch conversion shared by spatial-memory data paths."""

from __future__ import annotations

import numpy as np
from PIL import Image


def rgb_to_memory_patches(
    rgb: np.ndarray,
    *,
    rows: int = 2,
    cols: int = 4,
    patch_size: int = 8,
    resize_size: int | None = None,
) -> np.ndarray:
    """Convert an HWC uint8 RGB image to spatially ordered thumbnail patches.

    ``resize_size`` optionally resizes the full frame to a square using
    bilinear interpolation before the grid is split. Each tile is then reduced
    independently with BOX filtering. With the default ``None``, the input
    pixels are not resized before tiling.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB pixels, got {image.dtype}")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError(f"rows must be a positive integer, got {rows!r}")
    if not isinstance(cols, int) or isinstance(cols, bool) or cols < 1:
        raise ValueError(f"cols must be a positive integer, got {cols!r}")
    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size < 1:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}")
    if resize_size is not None and (
        not isinstance(resize_size, int) or isinstance(resize_size, bool) or resize_size < 1
    ):
        raise ValueError(f"resize_size must be a positive integer or None, got {resize_size!r}")

    frame = Image.fromarray(image, mode="RGB")
    if resize_size is not None:
        frame = frame.resize(
            (resize_size, resize_size),
            Image.Resampling.BILINEAR,
        )

    width, height = frame.size
    tile_width, tile_height = width // cols, height // rows
    if tile_width < 1 or tile_height < 1:
        raise ValueError(f"RGB image {frame.size} is too small for a {rows}x{cols} patch grid")

    patches = np.empty((rows * cols, patch_size, patch_size, 3), dtype=np.uint8)
    index = 0
    for row in range(rows):
        for col in range(cols):
            left, top = col * tile_width, row * tile_height
            right = width if col == cols - 1 else (col + 1) * tile_width
            bottom = height if row == rows - 1 else (row + 1) * tile_height
            tile = frame.crop((left, top, right, bottom))
            patches[index] = np.asarray(
                tile.resize((patch_size, patch_size), Image.Resampling.BOX),
                dtype=np.uint8,
            )
            index += 1
    return patches


def image_to_memory_tokens(
    rgb: np.ndarray,
    *,
    resize_size: int | None = None,
) -> np.ndarray:
    """Shared public entry point for spatial RGB memory tokens.

    The default preserves source resolution before tiling. Benchmarks that
    resize their source frames must pass that resolution explicitly.
    """
    return rgb_to_memory_patches(
        rgb,
        rows=2,
        cols=4,
        patch_size=8,
        resize_size=resize_size,
    )


__all__ = ["image_to_memory_tokens", "rgb_to_memory_patches"]
