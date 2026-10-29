"""Training-only image augmentation that keeps image-space grounding labels consistent.

The primary image gets a random similarity transform (scale, rotation, shift) and photometric
jitter; normalised (x, y) labels on that image are moved by the same transform and invalidated
when they leave the frame. Secondary images get photometric jitter and are blanked with
``secondary_dropout`` probability so the policy cannot rely on the wrist / second view alone.
Randomness comes from ``torch``'s per-worker generator, so DataLoader workers do not repeat.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand(()))


class GroundingAugment:
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.scale = tuple(cfg.get("scale", (0.88, 1.08)))
        self.rotate_deg = float(cfg.get("rotate_deg", 5.0))
        self.shift = float(cfg.get("shift", 0.06))
        self.brightness = float(cfg.get("brightness", 0.25))
        self.contrast = float(cfg.get("contrast", 0.25))
        self.saturation = float(cfg.get("saturation", 0.25))
        self.blur_p = float(cfg.get("blur_p", 0.2))
        self.noise_std = float(cfg.get("noise_std", 6.0))
        self.secondary_dropout = float(cfg.get("secondary_dropout", 0.15))

    def _photometric(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(_uniform(1 - self.brightness, 1 + self.brightness))
        image = ImageEnhance.Contrast(image).enhance(_uniform(1 - self.contrast, 1 + self.contrast))
        image = ImageEnhance.Color(image).enhance(_uniform(1 - self.saturation, 1 + self.saturation))
        if float(torch.rand(())) < self.blur_p:
            image = image.filter(ImageFilter.GaussianBlur(radius=_uniform(0.3, 1.2)))
        if self.noise_std > 0:
            array = np.asarray(image, dtype=np.float32)
            noise = torch.randn(array.shape).numpy() * _uniform(0.0, self.noise_std)
            image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
        return image

    def __call__(self, images: list[Image.Image], uv: np.ndarray | None = None, uv_valid: np.ndarray | None = None):
        """images[0] is the primary view; uv [N,2] normalised (x=col, y=row) on it."""
        if not self.enabled:
            return images, uv, uv_valid
        width, height = images[0].size
        scale = _uniform(*self.scale)
        angle = math.radians(_uniform(-self.rotate_deg, self.rotate_deg))
        tx, ty = _uniform(-self.shift, self.shift) * width, _uniform(-self.shift, self.shift) * height
        cx, cy = width / 2.0, height / 2.0
        cos, sin = math.cos(angle) * scale, math.sin(angle) * scale
        # Forward map (pixel centre coords): p' = A (p - c) + c + t
        forward = np.array([[cos, -sin, cx + tx - cos * cx + sin * cy], [sin, cos, cy + ty - sin * cx - cos * cy], [0, 0, 1]])
        inverse = np.linalg.inv(forward)
        primary = images[0].transform((width, height), Image.AFFINE, tuple(inverse[:2].reshape(-1)), resample=Image.BILINEAR)
        out = [self._photometric(primary)]
        for image in images[1:]:
            if float(torch.rand(())) < self.secondary_dropout:
                out.append(Image.new("RGB", image.size))
            else:
                out.append(self._photometric(image))
        if uv is not None:
            uv = np.asarray(uv, dtype=np.float32).copy()
            pixels = np.c_[uv[:, 0] * width - 0.5, uv[:, 1] * height - 0.5, np.ones(len(uv))] @ forward.T
            uv[:, 0], uv[:, 1] = (pixels[:, 0] + 0.5) / width, (pixels[:, 1] + 0.5) / height
            inside = (uv >= 0).all(-1) & (uv <= 1).all(-1)
            uv_valid = np.asarray(uv_valid, dtype=bool) & inside
        return out, uv, uv_valid
