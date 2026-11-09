"""Deterministic dominant-colour extraction in CIELab space."""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib import colors
from PIL import Image
from sklearn.cluster import KMeans

from utils.color_space import lab_to_rgb_numpy, rgb_to_lab_numpy


class ColorExtractor:
    def __init__(self, n_colors=3, random_state=42):
        if n_colors < 1:
            raise ValueError("n_colors must be positive")
        self.n_colors = int(n_colors)
        self.random_state = int(random_state)
        self.xkcd_colors = {
            name.replace("xkcd:", ""): tuple(int(channel * 255) for channel in colors.to_rgb(value))
            for name, value in colors.XKCD_COLORS.items()
        }
        names = list(self.xkcd_colors)
        rgb = np.asarray([self.xkcd_colors[name] for name in names], dtype=np.float32) / 255.0
        self._xkcd_names = names
        self._xkcd_lab = rgb_to_lab_numpy(rgb)

    def get_dominant_palette(self, pil_img, pixel_mask=None):
        """Return RGB/Lab centres and names for the selected pixels.

        ``pixel_mask`` is a boolean H×W mask. Stage 2 supplies its selected
        window here; omitting it retains the standalone whole-image interface.
        """
        image = pil_img.convert("RGB")
        if pixel_mask is None:
            image = image.resize((50, 50))
            rgb_pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3) / 255.0
        else:
            mask = np.asarray(pixel_mask, dtype=bool)
            pixels = np.asarray(image, dtype=np.float32)
            if mask.shape != pixels.shape[:2]:
                raise ValueError(f"pixel_mask shape {mask.shape} does not match image {pixels.shape[:2]}")
            rgb_pixels = pixels[mask].reshape(-1, 3) / 255.0
            if len(rgb_pixels) == 0:
                raise ValueError("Cannot extract a palette from an empty selected region")
        lab_pixels = rgb_to_lab_numpy(rgb_pixels)
        cluster_count = min(self.n_colors, len(lab_pixels))
        kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=self.random_state)
        labels = kmeans.fit_predict(lab_pixels)
        counts = np.bincount(labels, minlength=cluster_count)
        order = np.argsort(-counts, kind="stable")
        # Rounding suppresses insignificant BLAS reduction-order jitter while
        # retaining much more precision than an RGB uint8 palette.
        lab_palette = np.round(kmeans.cluster_centers_[order].astype(np.float32), decimals=3)
        rgb_palette = lab_to_rgb_numpy(lab_palette)
        rgb_u8 = np.rint(rgb_palette * 255.0).astype(np.uint8)
        names = [self.lab_to_xkcd_name(lab) for lab in lab_palette]
        return {
            "rgb": [tuple(int(channel) for channel in colour) for colour in rgb_u8],
            "lab": [tuple(float(channel) for channel in colour) for colour in lab_palette],
            "names": names,
        }

    def get_dominant_colors(self, pil_img):
        """Legacy RGB-only interface."""
        return self.get_dominant_palette(pil_img)["rgb"]

    def rgb_to_xkcd_name(self, rgb):
        rgb_array = np.asarray(rgb, dtype=np.float32).reshape(1, 3)
        if rgb_array.max() > 1.0:
            rgb_array = rgb_array / 255.0
        return self.lab_to_xkcd_name(rgb_to_lab_numpy(rgb_array)[0])

    def lab_to_xkcd_name(self, lab):
        distances = np.sum((self._xkcd_lab - np.asarray(lab, dtype=np.float32)) ** 2, axis=1)
        return self._xkcd_names[int(np.argmin(distances))]

    def get_top_color_names(self, pil_img, topk=3):
        return self.get_dominant_palette(pil_img)["names"][:topk]

    def get_top_color_names_batch(self, pil_imgs: List[Image.Image], topk=3):
        """Batch interface; extraction errors deliberately propagate to the caller."""
        return [self.get_top_color_names(image, topk=topk) for image in pil_imgs]
