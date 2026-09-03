from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def classify_artwork(image_path: str | Path, output_path: str | Path | None = None) -> dict:
    """Classify a normalized raster into broad reconstruction routes.

    This classifier intentionally uses generic image statistics only. It does not
    assume any brand colour, poster family, or GFC-specific composition.
    """
    image_path = Path(image_path)
    rgb = _load_rgb(image_path)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edge_threshold = max(18.0, float(np.percentile(grad, 78)))
    edge_density = float((grad >= edge_threshold).mean())

    sigma = max(3.0, min(h, w) * 0.012)
    smooth = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    residual = np.linalg.norm(lab - smooth, axis=2)
    texture = float(np.mean(residual))
    smooth_fraction = float((residual <= max(5.0, float(np.percentile(residual, 45)))).mean())

    thumb_w = max(8, min(192, w))
    thumb_h = max(8, int(round(h * thumb_w / max(w, 1))))
    thumb = cv2.resize(rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    pixels = thumb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.8)
    k = min(12, max(2, len(pixels)))
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    quant_error = float(np.sqrt(compactness / max(len(pixels), 1)))
    palette = np.clip(np.rint(centers), 0, 255).astype(np.uint8).tolist()

    # Broad routing scores, not semantic truth.
    hard_score = float(np.clip(0.58 * (edge_density / 0.18) + 0.42 * (18.0 / max(quant_error, 1.0)), 0, 1))
    smooth_score = float(np.clip(0.62 * smooth_fraction + 0.38 * (1.0 - min(edge_density / 0.28, 1.0)), 0, 1))
    photo_score = float(np.clip(0.55 * min(texture / 18.0, 1.0) + 0.45 * min(quant_error / 28.0, 1.0), 0, 1))

    if smooth_score >= hard_score and smooth_score >= photo_score:
        primary = "smooth_composite"
    elif hard_score >= photo_score:
        primary = "hard_graphic_composite"
    else:
        primary = "mixed_or_photographic"

    routes = {
        "background_gradient_fit": smooth_score >= 0.42,
        "panel_detection": edge_density >= 0.035 and smooth_score >= 0.28,
        "hard_graphic_vectorization": hard_score >= 0.42,
        "photographic_fallback_possible": photo_score >= 0.62,
    }

    result = {
        "schema": "poster-vector-rebuilder.artwork-classification.v1",
        "image": str(image_path),
        "dimensions": [w, h],
        "primary_class": primary,
        "scores": {
            "smooth_composite": round(smooth_score, 6),
            "hard_graphic_composite": round(hard_score, 6),
            "mixed_or_photographic": round(photo_score, 6),
        },
        "features": {
            "edge_density": round(edge_density, 6),
            "mean_local_lab_residual": round(texture, 6),
            "smooth_fraction": round(smooth_fraction, 6),
            "palette_quantization_rmse": round(quant_error, 6),
            "sample_palette_rgb": palette,
        },
        "routes": routes,
        "note": "This is a generic routing classifier. It does not assume a specific brand, colour family, poster template, or artwork type.",
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
