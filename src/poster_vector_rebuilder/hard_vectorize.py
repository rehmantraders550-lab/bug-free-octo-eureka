from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageOps

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)


@dataclass(frozen=True)
class VectorizeConfig:
    colors: int = 8
    min_area: float = 6.0
    simplify: float = 0.0025
    cleanup_radius: int = 0
    backend: str = "auto"


def _read_rgba(path: str | Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    return np.asarray(image, dtype=np.uint8)


def _read_mask(path: str | Path | None, size: tuple[int, int], alpha: np.ndarray) -> np.ndarray:
    h, w = size
    valid = alpha > 0
    if path is None:
        return valid
    mask = Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    return valid & (np.asarray(mask, dtype=np.uint8) >= 128)


def _quantize(rgb: np.ndarray, valid: np.ndarray, colors: int) -> tuple[np.ndarray, np.ndarray]:
    pixels = rgb[valid]
    if pixels.size == 0:
        raise ValueError("Vectorization mask contains no pixels")
    unique = np.unique(pixels.reshape(-1, 3), axis=0)
    k = max(1, min(int(colors), len(unique)))
    if len(unique) <= k:
        palette = unique.astype(np.uint8)
        lut = {tuple(c.tolist()): i for i, c in enumerate(palette)}
        labels = np.full(valid.shape, -1, dtype=np.int32)
        labels[valid] = np.asarray([lut[tuple(c.tolist())] for c in pixels], dtype=np.int32)
        return labels, palette

    cv2.setRNGSeed(0)
    data = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.25)
    _, compact_labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    palette = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    labels = np.full(valid.shape, -1, dtype=np.int32)
    labels[valid] = compact_labels.reshape(-1)
    return labels, palette


def _cleanup(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    r = int(radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def _contour_depth(index: int, hierarchy: np.ndarray) -> int:
    depth = 0
    parent = int(hierarchy[index][3])
    while parent >= 0:
        depth += 1
        parent = int(hierarchy[parent][3])
    return depth


def _approx_contour(contour: np.ndarray, simplify: float) -> np.ndarray:
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    epsilon = max(0.15, float(simplify) * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return contour if len(approx) < 3 else approx


def _path_d(contours: Iterable[np.ndarray]) -> str:
    parts: list[str] = []
    for contour in contours:
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue
        parts.append(f"M {int(pts[0, 0])} {int(pts[0, 1])}")
        for x, y in pts[1:]:
            parts.append(f"L {int(x)} {int(y)}")
        parts.append("Z")
    return " ".join(parts)


def _hex(color: np.ndarray) -> str:
    r, g, b = [int(v) for v in color]
    return f"#{r:02x}{g:02x}{b:02x}"


def _opencv_vectorize(
    rgba: np.ndarray,
    valid: np.ndarray,
    output_svg: Path,
    report_path: Path,
    config: VectorizeConfig,
) -> dict:
    h, w = valid.shape
    rgb = rgba[..., :3]
    labels, palette = _quantize(rgb, valid, config.colors)

    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {"version": "1.1", "width": str(w), "height": str(h), "viewBox": f"0 0 {w} {h}"},
    )
    artwork = ET.SubElement(
        root,
        f"{{{SVG_NS}}}g",
        {
            "id": "HARD_GRAPHIC",
            f"{{{INKSCAPE_NS}}}label": "HARD_GRAPHIC",
            f"{{{INKSCAPE_NS}}}groupmode": "layer",
        },
    )

    reconstructed = np.zeros_like(rgb)
    reconstructed_valid = np.zeros(valid.shape, dtype=bool)
    path_count = 0
    contour_count = 0
    boundary_error_pixels = 0
    palette_report: list[dict] = []

    order = sorted(range(len(palette)), key=lambda i: int(np.count_nonzero(labels == i)), reverse=True)
    for layer_no, color_idx in enumerate(order, start=1):
        raw_mask = np.where(labels == color_idx, 255, 0).astype(np.uint8)
        mask = _cleanup(raw_mask, config.cleanup_radius)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or not contours:
            continue
        hierarchy = hierarchy[0]
        kept_indices = [i for i, c in enumerate(contours) if abs(cv2.contourArea(c)) >= config.min_area]
        if not kept_indices:
            continue

        approx_by_index = {i: _approx_contour(contours[i], config.simplify) for i in kept_indices}
        layer = ET.SubElement(
            artwork,
            f"{{{SVG_NS}}}g",
            {
                "id": f"color_{layer_no:02d}",
                f"{{{INKSCAPE_NS}}}label": f"Color {layer_no:02d} {_hex(palette[color_idx])}",
            },
        )

        top_levels = [i for i in kept_indices if int(hierarchy[i][3]) < 0]
        vector_mask = np.zeros(valid.shape, dtype=np.uint8)
        layer_paths = 0
        for top_i in top_levels:
            subtree: list[int] = []
            stack = [top_i]
            while stack:
                idx = stack.pop()
                if idx in approx_by_index:
                    subtree.append(idx)
                child = int(hierarchy[idx][2])
                while child >= 0:
                    stack.append(child)
                    child = int(hierarchy[child][0])
            if not subtree:
                continue
            d = _path_d([approx_by_index[i] for i in subtree])
            if not d:
                continue
            ET.SubElement(
                layer,
                f"{{{SVG_NS}}}path",
                {"d": d, "fill": _hex(palette[color_idx]), "fill-rule": "evenodd", "stroke": "none"},
            )
            layer_paths += 1
            path_count += 1
            contour_count += len(subtree)

        for i in sorted(approx_by_index, key=lambda j: _contour_depth(j, hierarchy)):
            fill = 255 if _contour_depth(i, hierarchy) % 2 == 0 else 0
            cv2.fillPoly(vector_mask, [approx_by_index[i]], fill)
        vector_mask = (vector_mask > 0) & valid
        reconstructed[vector_mask] = palette[color_idx]
        reconstructed_valid |= vector_mask
        boundary_error_pixels += int(np.count_nonzero((vector_mask != (raw_mask > 0)) & valid))
        palette_report.append(
            {
                "layer": f"color_{layer_no:02d}",
                "rgb": [int(v) for v in palette[color_idx]],
                "hex": _hex(palette[color_idx]),
                "source_pixels": int(np.count_nonzero(raw_mask)),
                "vector_pixels": int(np.count_nonzero(vector_mask)),
                "paths": layer_paths,
            }
        )

    if path_count == 0:
        raise ValueError("No hard-edge vector paths survived the current thresholds")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_svg, encoding="utf-8", xml_declaration=True)

    measured = valid & reconstructed_valid
    mae = (
        float(np.mean(np.abs(rgb[measured].astype(np.int16) - reconstructed[measured].astype(np.int16))))
        if np.any(measured)
        else None
    )
    valid_pixels = int(np.count_nonzero(valid))
    coverage = float(np.count_nonzero(reconstructed_valid & valid) / valid_pixels)
    report = {
        "schema": "poster-vector-hard-graphic-v1",
        "backend": "opencv",
        "source_size": {"width": w, "height": h},
        "mask_pixels": valid_pixels,
        "mask_coverage_ratio": float(valid_pixels / (w * h)),
        "palette_size": len(palette_report),
        "path_count": path_count,
        "contour_count": contour_count,
        "vector_coverage_ratio": coverage,
        "boundary_error_ratio": float(boundary_error_pixels / max(valid_pixels, 1)),
        "rgb_mae_on_vector_coverage": mae,
        "confidence_class": "A" if coverage >= 0.995 else "B",
        "palette": palette_report,
        "outputs": {"svg": str(output_svg), "report": str(report_path)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _masked_temp_png(rgba: np.ndarray, valid: np.ndarray, path: Path) -> None:
    out = rgba.copy()
    out[..., 3] = np.where(valid, out[..., 3], 0).astype(np.uint8)
    Image.fromarray(out, mode="RGBA").save(path)


def _vtracer_vectorize(
    rgba: np.ndarray,
    valid: np.ndarray,
    output_svg: Path,
    report_path: Path,
    config: VectorizeConfig,
) -> dict:
    exe = shutil.which("vtracer")
    if not exe:
        raise RuntimeError("VTracer executable was not found on PATH")
    h, w = valid.shape
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pvr-vtracer-") as td:
        source = Path(td) / "masked.png"
        _masked_temp_png(rgba, valid, source)
        cmd = [
            exe,
            "--input", str(source),
            "--output", str(output_svg),
            "--colormode", "color",
            "--hierarchical", "stacked",
            "--mode", "spline",
            "--filter_speckle", str(max(1, int(round(config.min_area)))),
            "--color_precision", "6",
            "--layer_difference", "16",
            "--corner_threshold", "60",
            "--length_threshold", "4.0",
            "--max_iterations", "10",
            "--splice_threshold", "45",
            "--path_precision", "8",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"VTracer failed: {proc.stderr.strip() or proc.stdout.strip()}")

    tree = ET.parse(output_svg)
    root = tree.getroot()
    path_count = sum(1 for elem in root.iter() if elem.tag.endswith("path"))
    root.set("viewBox", root.get("viewBox") or f"0 0 {w} {h}")
    root.set("width", root.get("width") or str(w))
    root.set("height", root.get("height") or str(h))
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)
    report = {
        "schema": "poster-vector-hard-graphic-v1",
        "backend": "vtracer",
        "source_size": {"width": w, "height": h},
        "mask_pixels": int(np.count_nonzero(valid)),
        "mask_coverage_ratio": float(np.count_nonzero(valid) / (w * h)),
        "path_count": path_count,
        "confidence_class": "B",
        "outputs": {"svg": str(output_svg), "report": str(report_path)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def vectorize_hard_graphic(
    image_path: str | Path,
    output_svg: str | Path,
    *,
    mask_path: str | Path | None = None,
    report_path: str | Path | None = None,
    colors: int = 8,
    min_area: float = 6.0,
    simplify: float = 0.0025,
    cleanup_radius: int = 0,
    backend: str = "auto",
) -> dict:
    """Vectorize a hard-edged raster element into editable solid-fill SVG paths.

    Pixels outside the selected mask are never vectorized. ``auto`` prefers a
    locally installed VTracer executable and falls back to the deterministic
    OpenCV contour engine, so the core pipeline remains operational without it.
    """
    if backend not in {"auto", "opencv", "vtracer"}:
        raise ValueError("backend must be one of: auto, opencv, vtracer")
    if colors < 1:
        raise ValueError("colors must be >= 1")
    if min_area < 0:
        raise ValueError("min_area must be >= 0")
    if simplify < 0:
        raise ValueError("simplify must be >= 0")

    output_svg = Path(output_svg)
    report_path = Path(report_path) if report_path else output_svg.with_suffix(".report.json")
    rgba = _read_rgba(image_path)
    valid = _read_mask(mask_path, rgba.shape[:2], rgba[..., 3])
    config = VectorizeConfig(
        colors=colors,
        min_area=min_area,
        simplify=simplify,
        cleanup_radius=cleanup_radius,
        backend=backend,
    )

    selected = backend
    if selected == "auto":
        selected = "vtracer" if shutil.which("vtracer") else "opencv"
    if selected == "vtracer":
        return _vtracer_vectorize(rgba, valid, output_svg, report_path, config)
    return _opencv_vectorize(rgba, valid, output_svg, report_path, config)
