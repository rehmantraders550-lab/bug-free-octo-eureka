"""Confidence-aware background primitive discovery and SVG construction.

This module deliberately fits only pixels admitted by ``background_known``.
It detects the long, low-frequency panel boundaries that are useful in a
poster background and makes a compact, editable SVG model.  It is not a
foreground tracer and it never uses masked pixels as fitting authority.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class PanelLine:
    """A long boundary observed on authoritative background pixels."""

    angle_deg: float
    offset_px: float
    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float
    edge_support: float
    color_shift_lab: float
    confidence: float


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        mask = np.array(im.convert("L"))
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask >= 128


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(values, [50, 99.3])
    if hi <= lo + 1e-6:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _line_geometry(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, np.ndarray]:
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        raise ValueError("zero-length line")
    tangent = np.array([dx / length, dy / length], dtype=np.float32)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    # An undirected line: use the stable representative in [-90, 90).
    if angle >= 90:
        angle -= 180
        normal *= -1
    if angle < -90:
        angle += 180
        normal *= -1
    offset = float(np.dot(normal, np.array([x1, y1], dtype=np.float32)))
    return angle, offset, normal


def _sample_colour_shift(lab: np.ndarray, known: np.ndarray, line: tuple[float, float, float, float]) -> float:
    """Measure an across-line Lab discontinuity only where both sides are known."""
    x1, y1, x2, y2 = line
    angle, _, normal = _line_geometry(x1, y1, x2, y2)
    del angle
    length = math.hypot(x2 - x1, y2 - y1)
    count = max(16, int(length / 12))
    t = np.linspace(0.08, 0.92, count)
    points = np.column_stack((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    h, w = known.shape
    deltas: list[float] = []
    for distance in (5.0, 9.0, 14.0):
        a = np.rint(points + normal * distance).astype(int)
        b = np.rint(points - normal * distance).astype(int)
        valid = (
            (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
            & (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h)
        )
        a, b = a[valid], b[valid]
        if len(a) == 0:
            continue
        safe = known[a[:, 1], a[:, 0]] & known[b[:, 1], b[:, 0]]
        if safe.any():
            da = lab[a[safe, 1], a[safe, 0]]
            db = lab[b[safe, 1], b[safe, 0]]
            deltas.extend(np.linalg.norm(da - db, axis=1).tolist())
    return float(np.median(deltas)) if deltas else 0.0


def background_safe_edge_map(rgb: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an edge-strength map and a binary candidate map.

    A distance margin suppresses foreground-mask borders, which otherwise form
    deceptively strong Hough candidates around text and product silhouettes.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = lab[:, :, 0]
    gx = cv2.Scharr(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lum, cv2.CV_32F, 0, 1)
    luminance = cv2.magnitude(gx, gy)
    a_grad = cv2.magnitude(cv2.Scharr(lab[:, :, 1], cv2.CV_32F, 1, 0), cv2.Scharr(lab[:, :, 1], cv2.CV_32F, 0, 1))
    b_grad = cv2.magnitude(cv2.Scharr(lab[:, :, 2], cv2.CV_32F, 1, 0), cv2.Scharr(lab[:, :, 2], cv2.CV_32F, 0, 1))
    strength = _normalise(luminance + 0.38 * (a_grad + b_grad))

    margin = max(4, round(min(rgb.shape[:2]) * 0.006))
    safe = known & (distance_transform_edt(known) >= margin)
    safe_u8 = safe.astype(np.uint8) * 255
    blurred = cv2.GaussianBlur((strength * 255).astype(np.uint8), (3, 3), 0)
    low = max(8, int(np.percentile(blurred[safe], 58))) if safe.any() else 32
    high = max(low + 8, int(np.percentile(blurred[safe], 87))) if safe.any() else 96
    edges = cv2.Canny(blurred, low, high)
    edges = cv2.bitwise_and(edges, safe_u8)
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close, iterations=1)
    return (strength * 255).astype(np.uint8), edges


def detect_panel_boundaries(rgb: np.ndarray, known: np.ndarray, max_lines: int = 3) -> tuple[list[PanelLine], np.ndarray, np.ndarray]:
    """Detect robust long panel boundaries from known background only."""
    strength, edges = background_safe_edge_map(rgb, known)
    h, w = known.shape
    min_length = max(80, int(min(h, w) * 0.22))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(30, min_length // 3), minLineLength=min_length, maxLineGap=max(12, min_length // 10))
    if raw is None:
        return [], strength, edges

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    candidates: list[PanelLine] = []
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        angle, offset, _ = _line_geometry(float(x1), float(y1), float(x2), float(y2))
        length = math.hypot(x2 - x1, y2 - y1)
        # Poster planes are generally low-frequency diagonal/horizontal edges.
        # Vertical lines are usually typography, badges or product artefacts.
        if abs(angle) > 42:
            continue
        colour_shift = _sample_colour_shift(lab, known, (x1, y1, x2, y2))
        samples = max(2, int(length / 7))
        xs = np.rint(np.linspace(x1, x2, samples)).astype(int)
        ys = np.rint(np.linspace(y1, y2, samples)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        support = float((edges[ys[valid], xs[valid]] > 0).mean()) if valid.any() else 0.0
        score = 0.40 * min(1.0, length / (w * 0.88)) + 0.31 * support + 0.29 * min(1.0, colour_shift / 18.0)
        if support < 0.12 or colour_shift < 1.2:
            continue
        candidates.append(PanelLine(angle, offset, float(x1), float(y1), float(x2), float(y2), length, support, colour_shift, score))

    # Angle/offset non-maximum suppression gives one simple editable boundary
    # for each visual panel, rather than several Hough fragments of the same line.
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    chosen: list[PanelLine] = []
    for candidate in candidates:
        duplicate = any(abs(candidate.angle_deg - old.angle_deg) < 2.2 and abs(candidate.offset_px - old.offset_px) < 26 for old in chosen)
        if not duplicate:
            chosen.append(candidate)
        if len(chosen) >= max_lines:
            break
    chosen.sort(key=lambda item: _line_y_at_x(item, w / 2))
    return chosen, strength, edges


def _line_y_at_x(line: PanelLine, x: float) -> float:
    dx = line.x2 - line.x1
    if abs(dx) < 1e-5:
        return line.y1
    return line.y1 + (x - line.x1) * (line.y2 - line.y1) / dx


def _intersections_for_line(line: PanelLine, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x in (0.0, float(width)):
        y = _line_y_at_x(line, x)
        if 0 <= y <= height:
            points.append((x, y))
    if abs(line.y2 - line.y1) > 1e-6:
        for y in (0.0, float(height)):
            x = line.x1 + (y - line.y1) * (line.x2 - line.x1) / (line.y2 - line.y1)
            if 0 <= x <= width:
                points.append((x, y))
    if len(points) < 2:
        return (0.0, _line_y_at_x(line, 0.0)), (float(width), _line_y_at_x(line, float(width)))
    # furthest pair is the complete line extent in the rectangular canvas
    best = max(((a, b) for i, a in enumerate(points) for b in points[i + 1:]), key=lambda pair: math.dist(*pair))
    return best


def _hex(rgb: np.ndarray) -> str:
    r, g, b = np.clip(np.rint(rgb), 0, 255).astype(int).tolist()
    return f"#{r:02X}{g:02X}{b:02X}"


def _known_median(rgb: np.ndarray, known: np.ndarray, y0: int, y1: int) -> np.ndarray:
    region = rgb[y0:y1]
    mask = known[y0:y1]
    if mask.any():
        return np.median(region[mask], axis=0)
    return np.median(rgb[known], axis=0) if known.any() else np.median(rgb.reshape(-1, 3), axis=0)


def _build_svg(width: int, height: int, rgb: np.ndarray, known: np.ndarray, lines: list[PanelLine]) -> str:
    stops = []
    for idx, (y0, y1) in enumerate(zip(np.linspace(0, height, 7, dtype=int)[:-1], np.linspace(0, height, 7, dtype=int)[1:])):
        stops.append((idx / 5, _hex(_known_median(rgb, known, y0, y1))))
    defs = ["<linearGradient id=\"base-gradient\" gradientUnits=\"userSpaceOnUse\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"%s\">" % height]
    defs.extend(f'<stop offset="{offset:.4f}" stop-color="{colour}"/>' for offset, colour in stops)
    defs.append("</linearGradient>")
    body = [f'<rect id="Base-Gradient" x="0" y="0" width="{width}" height="{height}" fill="url(#base-gradient)"/>']
    for index, line in enumerate(lines, start=1):
        (x1, y1), (x2, y2) = _intersections_for_line(line, width, height)
        # Estimate the side colour from safe pixels. The polygon must only
        # continue this measured field; unsupported parts never fit themselves.
        normal = np.array([-(y2 - y1), x2 - x1], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-6)
        samples = []
        for t in np.linspace(0.1, 0.9, 18):
            point = np.array([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]) + normal * 20
            px, py = np.rint(point).astype(int)
            if 0 <= px < width and 0 <= py < height and known[py, px]:
                samples.append(rgb[py, px])
        colour = _hex(np.median(samples, axis=0) if samples else _known_median(rgb, known, 0, height))
        # Alternate top/bottom sides to avoid all panels stacking on one side.
        if index % 2:
            points = f"0,0 {width},0 {x2:.2f},{y2:.2f} {x1:.2f},{y1:.2f}"
        else:
            points = f"0,{height} {width},{height} {x2:.2f},{y2:.2f} {x1:.2f},{y1:.2f}"
        opacity = min(0.36, max(0.08, line.color_shift_lab / 55.0))
        body.append(f'<polygon id="Panel-{index:02d}" points="{points}" fill="{colour}" opacity="{opacity:.4f}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<title>Measured background reconstruction</title><desc>Known pixels are measured; masked areas are geometric continuation only.</desc>'
        f'<defs>{"".join(defs)}</defs><g id="00_BACKGROUND">{''.join(body)}</g></svg>'
    )


def _metrics(reference: np.ndarray, rendered: np.ndarray, known: np.ndarray) -> dict[str, float | None]:
    if rendered.shape[:2] != reference.shape[:2]:
        rendered = cv2.resize(rendered, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)
    if not known.any():
        return {"mean_deltaE2000": None, "median_deltaE2000": None, "rgb_mae_8bit": None, "tile_ssim": None}
    ref_lab = rgb2lab(reference / 255.0)
    out_lab = rgb2lab(rendered / 255.0)
    delta = deltaE_ciede2000(ref_lab, out_lab)
    mae = np.abs(reference.astype(np.float32) - rendered.astype(np.float32)).mean(axis=2)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    gray_out = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    # SSIM is reported on the known-pixel bounding crop, with all unknown
    # locations neutralized so foreground does not affect the result.
    safe_ref, safe_out = gray_ref.copy(), gray_out.copy()
    fill = int(np.median(gray_ref[known]))
    safe_ref[~known] = fill
    safe_out[~known] = fill
    return {
        "mean_deltaE2000": float(delta[known].mean()),
        "median_deltaE2000": float(np.median(delta[known])),
        "rgb_mae_8bit": float(mae[known].mean()),
        "tile_ssim": float(structural_similarity(safe_ref, safe_out, data_range=255)),
    }


def fit_background(job_dir: str | Path, image_path: str | Path | None = None, known_mask_path: str | Path | None = None) -> dict[str, Any]:
    """Run Phase 2.4B, render the compact model and write acceptance artefacts."""
    job = Path(job_dir)
    image_path = Path(image_path) if image_path else job / "work" / "normalized_reference.png"
    known_mask_path = Path(known_mask_path) if known_mask_path else job / "masks" / "background_known.png"
    rgb = _load_rgb(image_path)
    known = _load_mask(known_mask_path, rgb.shape[:2])
    lines, strength, edges = detect_panel_boundaries(rgb, known)

    analysis = job / "analysis"
    vector = job / "vector"
    delivery = job / "delivery"
    for folder in (analysis, vector, delivery):
        folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strength).save(analysis / "background_edge_strength.png")
    Image.fromarray(edges).save(analysis / "background_edges.png")
    inferred = (~known).astype(np.uint8) * 255
    Image.fromarray(inferred).save(analysis / "background_inference_mask.png")

    svg = _build_svg(rgb.shape[1], rgb.shape[0], rgb, known, lines)
    svg_path = vector / "background_master.svg"
    svg_path.write_text(svg, encoding="utf-8")
    preview_path = analysis / "background_preview.png"
    inkscape = shutil.which("inkscape")
    if not inkscape:
        raise RuntimeError("Inkscape is required for deterministic SVG render validation")
    subprocess.run([inkscape, str(svg_path), "--export-type=png", f"--export-filename={preview_path}"], check=True, capture_output=True, text=True)
    rendered = _load_rgb(preview_path)
    metrics = _metrics(rgb, rendered, known)
    delta = deltaE_ciede2000(rgb2lab(rgb / 255.0), rgb2lab(rendered / 255.0))
    heat = np.clip(delta / max(8.0, float(np.percentile(delta[known], 95) if known.any() else 8.0)), 0, 1)
    heat_rgb = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    Image.fromarray(cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)).save(analysis / "background_error_heatmap.png")
    report: dict[str, Any] = {
        "phase": "2.4B",
        "authority": "Metrics and boundary scores use background_known pixels only. Inference mask is not optimized.",
        "image": {"width": rgb.shape[1], "height": rgb.shape[0], "known_coverage": float(known.mean())},
        "detected_panel_boundaries": [asdict(line) for line in lines],
        "model": {"base": "editable multi-stop linear gradient", "panels": len(lines), "embedded_raster_images": 0, "svg_filters": 0},
        "known_pixel_metrics": metrics,
        "outputs": {
            "svg": "vector/background_master.svg", "preview": "analysis/background_preview.png", "edges": "analysis/background_edges.png",
            "inference_mask": "analysis/background_inference_mask.png", "heatmap": "analysis/background_error_heatmap.png",
        },
        "acceptance": {"automatic_panel_detection": bool(lines), "corel_safe_primitives": True, "thresholds_met": bool(metrics["median_deltaE2000"] is not None and metrics["median_deltaE2000"] < 3.0 and metrics["mean_deltaE2000"] < 4.0)},
    }
    (analysis / "background_fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Corel master is structurally the same restricted SVG, copied explicitly
    # after Inkscape has successfully parsed and rasterized it.
    shutil.copyfile(svg_path, delivery / "background_master_corel.svg")
    return report
