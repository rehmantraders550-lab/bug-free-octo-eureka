from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math

import cv2
import numpy as np
from scipy.optimize import minimize
from skimage.color import deltaE_ciede2000, rgb2lab


@dataclass
class Boundary:
    angle_deg: float
    rho: float
    t_min: float
    t_max: float
    length: float
    support_ratio: float
    color_shift_delta_e: float
    confidence: float
    inlier_count: int


@dataclass
class PanelHypothesis:
    angle_deg: float
    rho_low: float
    rho_high: float
    width_px: float
    t_min: float
    t_max: float
    mean_color_shift_delta_e: float
    opacity_estimate: float
    confidence: float
    interior_rgb: list[int]
    exterior_rgb: list[int]
    color_shift_rgb: list[float]
    optimization_success: bool
    optimization_message: str


def _load_rgb(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask >= 128


def _write_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_out = image
    if image.ndim == 3 and image.shape[2] == 3:
        image_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_out)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def _robust_normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sample = values[mask]
    if sample.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.percentile(sample, 25))
    hi = float(np.percentile(sample, 97.5))
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_background_safe_edges(
    rgb: np.ndarray,
    background_known: np.ndarray,
    *,
    exclusion_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = background_known.shape
    diag = math.hypot(w, h)
    if exclusion_radius is None:
        exclusion_radius = max(3, int(round(diag * 0.006)))
    kernel_size = max(3, exclusion_radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    safe = cv2.erode(background_known.astype(np.uint8), kernel, iterations=1) > 0
    if int(safe.sum()) < 64:
        raise ValueError("background_known has too little interior support after boundary exclusion")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    lum_grad = cv2.magnitude(sx, sy)

    lab_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_grad = np.zeros((h, w), dtype=np.float32)
    for channel in range(3):
        gx = cv2.Scharr(lab_cv[..., channel], cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(lab_cv[..., channel], cv2.CV_32F, 0, 1)
        lab_grad += cv2.magnitude(gx, gy) ** 2
    lab_grad = np.sqrt(lab_grad)

    lum_n = _robust_normalize(lum_grad, safe)
    lab_n = _robust_normalize(lab_grad, safe)
    combined = 0.45 * lum_n + 0.55 * lab_n

    safe_gray = gray[safe]
    med = float(np.median(safe_gray)) if safe_gray.size else 128.0
    low = int(max(5, 0.50 * med))
    high = int(min(255, max(low + 10, 1.35 * med)))
    canny = cv2.Canny(gray.astype(np.uint8), low, high) > 0

    strength_sample = combined[safe]
    threshold = float(np.percentile(strength_sample, 82.0)) if strength_sample.size else 1.0
    threshold = max(0.18, threshold)
    edges = ((combined >= threshold) | canny) & safe

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0
    edges &= safe

    diagnostics = {
        "known_coverage": float(background_known.mean()),
        "safe_known_coverage": float(safe.mean()),
        "mask_boundary_exclusion_radius_px": int(exclusion_radius),
        "combined_edge_threshold": float(threshold),
        "canny_low": low,
        "canny_high": high,
        "edge_coverage": float(edges.mean()),
    }
    return edges.astype(np.uint8) * 255, safe, diagnostics


def _canonical_angle(angle_deg: float) -> float:
    angle = angle_deg % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def _angle_diff(a: float, b: float) -> float:
    d = abs(_canonical_angle(a) - _canonical_angle(b))
    return min(d, 180.0 - d)


def _line_basis(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(_canonical_angle(angle_deg))
    d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    n = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    return d, n


def _line_from_pair(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float] | None:
    delta = p2 - p1
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return None
    angle = _canonical_angle(math.degrees(math.atan2(delta[1], delta[0])))
    _, n = _line_basis(angle)
    rho = float(np.dot(n, (p1 + p2) * 0.5))
    return angle, rho


def _fit_line_ransac(points: np.ndarray, *, tolerance: float, trials: int = 160) -> tuple[float, float, np.ndarray]:
    if len(points) < 2:
        raise ValueError("At least two points are required for RANSAC line fitting")
    rng = np.random.default_rng(0)
    best_inliers = None
    best_score = (-1, float("inf"))
    for _ in range(trials):
        ids = rng.choice(len(points), size=2, replace=False)
        model = _line_from_pair(points[ids[0]], points[ids[1]])
        if model is None:
            continue
        angle, rho = model
        _, n = _line_basis(angle)
        distances = np.abs(points @ n - rho)
        inliers = distances <= tolerance
        count = int(inliers.sum())
        if count < 2:
            continue
        residual = float(np.mean(distances[inliers]))
        score = (count, -residual)
        if score > (best_score[0], -best_score[1]):
            best_inliers = inliers
            best_score = (count, residual)
    if best_inliers is None:
        best_inliers = np.ones(len(points), dtype=bool)

    p = points[best_inliers]
    center = p.mean(axis=0)
    centered = p - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    angle = _canonical_angle(math.degrees(math.atan2(direction[1], direction[0])))
    d, n = _line_basis(angle)
    rho = float(np.dot(n, center))
    distances = np.abs(points @ n - rho)
    refined_inliers = distances <= tolerance
    if int(refined_inliers.sum()) >= 2:
        p = points[refined_inliers]
        center = p.mean(axis=0)
        centered = p - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        angle = _canonical_angle(math.degrees(math.atan2(direction[1], direction[0])))
        d, n = _line_basis(angle)
        rho = float(np.dot(n, center))
    return angle, rho, refined_inliers


def _line_t_range_in_image(angle_deg: float, rho: float, w: int, h: int) -> tuple[float, float]:
    d, n = _line_basis(angle_deg)
    p0 = n * rho
    candidates: list[float] = []
    eps = 1e-9
    if abs(d[0]) > eps:
        for x in (0.0, float(w - 1)):
            t = (x - p0[0]) / d[0]
            y = p0[1] + t * d[1]
            if -1e-6 <= y <= h - 1 + 1e-6:
                candidates.append(float(t))
    if abs(d[1]) > eps:
        for y in (0.0, float(h - 1)):
            t = (y - p0[1]) / d[1]
            x = p0[0] + t * d[0]
            if -1e-6 <= x <= w - 1 + 1e-6:
                candidates.append(float(t))
    if len(candidates) < 2:
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
        t = (corners - p0) @ d
        return float(t.min()), float(t.max())
    return min(candidates), max(candidates)


def _sample_rgb_or_lab(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    map_x = xy[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = xy[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        image.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(len(xy), -1)


def _boundary_contrast(
    lab: np.ndarray,
    known: np.ndarray,
    angle_deg: float,
    rho: float,
    *,
    probe_px: float,
    t_min: float | None = None,
    t_max: float | None = None,
    samples: int = 192,
) -> tuple[float, float]:
    h, w = known.shape
    d, n = _line_basis(angle_deg)
    p0 = n * rho
    if t_min is None or t_max is None:
        t_min, t_max = _line_t_range_in_image(angle_deg, rho, w, h)
    if t_max <= t_min:
        return 0.0, 0.0
    ts = np.linspace(t_min, t_max, samples)
    center = p0[None, :] + ts[:, None] * d[None, :]
    plus = center + probe_px * n[None, :]
    minus = center - probe_px * n[None, :]

    def in_bounds(xy: np.ndarray) -> np.ndarray:
        return (
            (xy[:, 0] >= 0)
            & (xy[:, 0] <= w - 1)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] <= h - 1)
        )

    valid = in_bounds(plus) & in_bounds(minus)
    if not np.any(valid):
        return 0.0, 0.0
    pi = np.rint(plus[valid]).astype(int)
    mi = np.rint(minus[valid]).astype(int)
    known_pair = known[pi[:, 1], pi[:, 0]] & known[mi[:, 1], mi[:, 0]]
    valid_ids = np.flatnonzero(valid)
    valid_ids = valid_ids[known_pair]
    support = len(valid_ids) / max(1, samples)
    if len(valid_ids) < 4:
        return 0.0, float(support)
    lp = _sample_rgb_or_lab(lab, plus[valid_ids])
    lm = _sample_rgb_or_lab(lab, minus[valid_ids])
    de = deltaE_ciede2000(lp, lm)
    return float(np.median(de)), float(support)


def detect_boundaries(
    rgb: np.ndarray,
    known: np.ndarray,
    edge_map: np.ndarray,
    *,
    min_line_fraction: float = 0.08,
    angle_tolerance_deg: float = 4.0,
    offset_tolerance_fraction: float = 0.018,
) -> list[Boundary]:
    h, w = known.shape
    diag = math.hypot(w, h)
    min_length = max(18, int(round(diag * min_line_fraction)))
    max_gap = max(4, int(round(diag * 0.025)))
    threshold = max(20, int(round(diag * 0.025)))
    raw = cv2.HoughLinesP(edge_map, 1, np.pi / 360.0, threshold, minLineLength=min_length, maxLineGap=max_gap)
    if raw is None:
        return []

    lab = rgb2lab(rgb.astype(np.float32) / 255.0)
    probe = max(2.0, diag * 0.004)
    initial: list[dict[str, Any]] = []
    for item in raw[:, 0, :]:
        x1, y1, x2, y2 = map(float, item)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        model = _line_from_pair(np.array([x1, y1]), np.array([x2, y2]))
        if model is None:
            continue
        angle, rho = model
        d, _ = _line_basis(angle)
        mid = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        t_center = float(np.dot(d, mid))
        t_min = t_center - length * 0.5
        t_max = t_center + length * 0.5
        contrast, support = _boundary_contrast(
            lab,
            known,
            angle,
            rho,
            probe_px=probe,
            t_min=t_min,
            t_max=t_max,
            samples=max(32, int(length / 3)),
        )
        if support < 0.12:
            continue
        strength = min(1.0, length / (0.45 * diag)) * min(1.0, support / 0.65) * min(1.0, max(contrast, 0.5) / 8.0)
        initial.append(
            {
                "coords": (x1, y1, x2, y2),
                "angle": angle,
                "rho": rho,
                "length": length,
                "contrast": contrast,
                "support": support,
                "score": strength,
            }
        )
    if not initial:
        return []

    initial.sort(key=lambda x: x["score"], reverse=True)
    offset_tol = max(3.0, diag * offset_tolerance_fraction)
    clusters: list[list[dict[str, Any]]] = []
    for seg in initial:
        assigned = False
        for cluster in clusters:
            ref = cluster[0]
            if _angle_diff(seg["angle"], ref["angle"]) <= angle_tolerance_deg and abs(seg["rho"] - ref["rho"]) <= offset_tol:
                cluster.append(seg)
                assigned = True
                break
        if not assigned:
            clusters.append([seg])

    edge_bool = edge_map > 0
    boundaries: list[Boundary] = []
    for cluster in clusters:
        cluster_mask = np.zeros_like(edge_map, dtype=np.uint8)
        for seg in cluster:
            x1, y1, x2, y2 = [int(round(v)) for v in seg["coords"]]
            cv2.line(cluster_mask, (x1, y1), (x2, y2), 255, 4)
        ys, xs = np.nonzero((cluster_mask > 0) & edge_bool)
        points = np.column_stack([xs, ys]).astype(np.float64)
        if len(points) < max(12, min_length // 4):
            pts: list[np.ndarray] = []
            for seg in cluster:
                x1, y1, x2, y2 = seg["coords"]
                count = max(4, int(seg["length"] / 8))
                pts.append(np.column_stack([np.linspace(x1, x2, count), np.linspace(y1, y2, count)]))
            points = np.vstack(pts)
        angle, rho, inliers = _fit_line_ransac(points, tolerance=max(1.5, diag * 0.0025))
        d, _ = _line_basis(angle)
        inlier_points = points[inliers]
        t = inlier_points @ d
        t_min = float(t.min())
        t_max = float(t.max())
        length = float(t_max - t_min)
        contrast, support = _boundary_contrast(
            lab,
            known,
            angle,
            rho,
            probe_px=probe,
            t_min=t_min,
            t_max=t_max,
            samples=max(48, int(length / 3)),
        )
        inlier_ratio = float(inliers.mean())
        length_score = min(1.0, length / (0.35 * diag))
        contrast_score = min(1.0, contrast / 7.0)
        support_score = min(1.0, support / 0.60)
        confidence = 0.30 * length_score + 0.25 * inlier_ratio + 0.25 * support_score + 0.20 * contrast_score
        if length < min_length * 0.80 or support < 0.10 or confidence < 0.28:
            continue
        boundaries.append(
            Boundary(
                angle_deg=float(angle),
                rho=float(rho),
                t_min=t_min,
                t_max=t_max,
                length=length,
                support_ratio=support,
                color_shift_delta_e=contrast,
                confidence=float(confidence),
                inlier_count=int(inliers.sum()),
            )
        )

    boundaries.sort(key=lambda b: b.confidence, reverse=True)
    deduped: list[Boundary] = []
    for b in boundaries:
        duplicate = any(_angle_diff(b.angle_deg, x.angle_deg) <= 2.0 and abs(b.rho - x.rho) <= max(3.0, diag * 0.008) for x in deduped)
        if not duplicate:
            deduped.append(b)
    return deduped[:24]


def _mean_parallel_angle(a: float, b: float) -> float:
    ar = math.radians(2 * _canonical_angle(a))
    br = math.radians(2 * _canonical_angle(b))
    mean = 0.5 * math.degrees(math.atan2(math.sin(ar) + math.sin(br), math.cos(ar) + math.cos(br)))
    return _canonical_angle(mean)


def _reproject_rho(boundary: Boundary, target_angle: float) -> float:
    d0, n0 = _line_basis(boundary.angle_deg)
    center_t = 0.5 * (boundary.t_min + boundary.t_max)
    point = n0 * boundary.rho + d0 * center_t
    _, n = _line_basis(target_angle)
    return float(np.dot(n, point))


def _reproject_extent(boundary: Boundary, target_angle: float) -> tuple[float, float]:
    d0, n0 = _line_basis(boundary.angle_deg)
    p1 = n0 * boundary.rho + d0 * boundary.t_min
    p2 = n0 * boundary.rho + d0 * boundary.t_max
    d, _ = _line_basis(target_angle)
    vals = [float(np.dot(d, p1)), float(np.dot(d, p2))]
    return min(vals), max(vals)


def _optimize_panel_geometry(
    lab: np.ndarray,
    known: np.ndarray,
    angle0: float,
    rho_low0: float,
    rho_high0: float,
    *,
    probe: float,
    angle_window: float,
    rho_window: float,
    min_width: float,
) -> tuple[float, float, float, bool, str]:
    x0 = np.array([angle0, rho_low0, rho_high0], dtype=np.float64)

    def objective(x: np.ndarray) -> float:
        angle, lo, hi = map(float, x)
        c1, s1 = _boundary_contrast(lab, known, angle, lo, probe_px=probe, samples=160)
        c2, s2 = _boundary_contrast(lab, known, angle, hi, probe_px=probe, samples=160)
        contrast = 0.5 * (c1 + c2)
        support = 0.5 * (s1 + s2)
        reg_angle = 0.035 * ((angle - angle0) / max(angle_window, 1e-6)) ** 2
        reg_rho = 0.025 * (((lo - rho_low0) / max(rho_window, 1e-6)) ** 2 + ((hi - rho_high0) / max(rho_window, 1e-6)) ** 2)
        support_penalty = 3.0 * max(0.0, 0.22 - support)
        return -contrast + reg_angle + reg_rho + support_penalty

    bounds = [
        (angle0 - angle_window, angle0 + angle_window),
        (rho_low0 - rho_window, rho_low0 + rho_window),
        (rho_high0 - rho_window, rho_high0 + rho_window),
    ]
    constraint = {"type": "ineq", "fun": lambda x: x[2] - x[1] - min_width}
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=[constraint], options={"maxiter": 80, "ftol": 1e-5})
    angle, lo, hi = map(float, result.x)
    if lo > hi:
        lo, hi = hi, lo
    return angle, lo, hi, bool(result.success), str(result.message)


def _panel_color_stats(
    rgb: np.ndarray,
    lab: np.ndarray,
    known: np.ndarray,
    angle: float,
    rho_low: float,
    rho_high: float,
) -> tuple[list[int], list[int], list[float], float, float, float, float]:
    h, w = known.shape
    d, n = _line_basis(angle)
    yy, xx = np.mgrid[0:h, 0:w]
    projection = xx * n[0] + yy * n[1]
    width = max(1.0, rho_high - rho_low)
    band = max(3.0, min(width * 0.22, math.hypot(w, h) * 0.035))
    inside = known & (projection >= rho_low) & (projection <= rho_high)
    outside = known & (((projection >= rho_low - band) & (projection < rho_low)) | ((projection > rho_high) & (projection <= rho_high + band)))
    if int(inside.sum()) < 8:
        interior_rgb = np.median(rgb[known], axis=0) if np.any(known) else np.array([0, 0, 0])
    else:
        interior_rgb = np.median(rgb[inside], axis=0)
    if int(outside.sum()) < 8:
        exterior_rgb = np.median(rgb[known], axis=0) if np.any(known) else interior_rgb.copy()
    else:
        exterior_rgb = np.median(rgb[outside], axis=0)
    interior_lab = rgb2lab((interior_rgb.reshape(1, 1, 3) / 255.0).astype(np.float32))[0, 0]
    exterior_lab = rgb2lab((exterior_rgb.reshape(1, 1, 3) / 255.0).astype(np.float32))[0, 0]
    de = float(deltaE_ciede2000(interior_lab[None, :], exterior_lab[None, :])[0])
    opacity = float(np.clip(de / 22.0, 0.06, 0.86))

    ys, xs = np.nonzero(inside)
    if len(xs):
        t = xs * d[0] + ys * d[1]
        t_min, t_max = float(np.min(t)), float(np.max(t))
    else:
        t_min, t_max = _line_t_range_in_image(angle, 0.5 * (rho_low + rho_high), w, h)
    shift = (interior_rgb.astype(np.float64) - exterior_rgb.astype(np.float64)).tolist()
    return (
        [int(round(v)) for v in interior_rgb],
        [int(round(v)) for v in exterior_rgb],
        [float(v) for v in shift],
        de,
        opacity,
        float(t_min),
        float(t_max),
    )


def generate_panel_hypotheses(
    rgb: np.ndarray,
    known: np.ndarray,
    boundaries: list[Boundary],
    *,
    max_hypotheses: int = 12,
) -> list[PanelHypothesis]:
    h, w = known.shape
    diag = math.hypot(w, h)
    lab = rgb2lab(rgb.astype(np.float32) / 255.0)
    probe = max(2.0, diag * 0.004)
    min_sep = max(8.0, diag * 0.015)
    max_sep = diag * 0.72
    hypotheses: list[PanelHypothesis] = []
    for i in range(len(boundaries)):
        for j in range(i + 1, len(boundaries)):
            b1, b2 = boundaries[i], boundaries[j]
            parallel_error = _angle_diff(b1.angle_deg, b2.angle_deg)
            if parallel_error > 5.0:
                continue
            angle0 = _mean_parallel_angle(b1.angle_deg, b2.angle_deg)
            rho1 = _reproject_rho(b1, angle0)
            rho2 = _reproject_rho(b2, angle0)
            lo0, hi0 = sorted([rho1, rho2])
            sep = hi0 - lo0
            if not (min_sep <= sep <= max_sep):
                continue
            e1 = _reproject_extent(b1, angle0)
            e2 = _reproject_extent(b2, angle0)
            overlap = max(0.0, min(e1[1], e2[1]) - max(e1[0], e2[0]))
            union = max(e1[1], e2[1]) - min(e1[0], e2[0])
            overlap_ratio = overlap / max(1.0, union)
            if overlap_ratio < 0.08:
                continue

            angle, lo, hi, success, message = _optimize_panel_geometry(
                lab,
                known,
                angle0,
                lo0,
                hi0,
                probe=probe,
                angle_window=4.5,
                rho_window=max(5.0, diag * 0.018),
                min_width=max(4.0, min_sep * 0.55),
            )
            interior, exterior, shift_rgb, panel_de, opacity, t_min, t_max = _panel_color_stats(rgb, lab, known, angle, lo, hi)
            boundary_shift = 0.5 * (b1.color_shift_delta_e + b2.color_shift_delta_e)
            parallel_score = max(0.0, 1.0 - parallel_error / 5.0)
            pair_conf = math.sqrt(max(0.0, b1.confidence * b2.confidence))
            shift_score = min(1.0, max(panel_de, boundary_shift) / 8.0)
            confidence = 0.45 * pair_conf + 0.20 * parallel_score + 0.15 * min(1.0, overlap_ratio / 0.6) + 0.20 * shift_score
            if confidence < 0.30:
                continue
            hypotheses.append(
                PanelHypothesis(
                    angle_deg=float(_canonical_angle(angle)),
                    rho_low=float(lo),
                    rho_high=float(hi),
                    width_px=float(hi - lo),
                    t_min=t_min,
                    t_max=t_max,
                    mean_color_shift_delta_e=float(panel_de),
                    opacity_estimate=float(opacity),
                    confidence=float(confidence),
                    interior_rgb=interior,
                    exterior_rgb=exterior,
                    color_shift_rgb=shift_rgb,
                    optimization_success=success,
                    optimization_message=message,
                )
            )

    hypotheses.sort(key=lambda p: p.confidence, reverse=True)
    deduped: list[PanelHypothesis] = []
    for p in hypotheses:
        duplicate = False
        for q in deduped:
            if _angle_diff(p.angle_deg, q.angle_deg) <= 2.5:
                overlap = max(0.0, min(p.rho_high, q.rho_high) - max(p.rho_low, q.rho_low))
                smaller = min(p.width_px, q.width_px)
                if smaller > 0 and overlap / smaller > 0.70:
                    duplicate = True
                    break
        if not duplicate:
            deduped.append(p)
        if len(deduped) >= max_hypotheses:
            break
    return deduped


def _sample_known_pixels(rgb: np.ndarray, known: np.ndarray, max_samples: int = 60000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(known)
    if len(xs) == 0:
        raise ValueError("background_known contains no authoritative pixels")
    if len(xs) > max_samples:
        rng = np.random.default_rng(0)
        ids = np.sort(rng.choice(len(xs), max_samples, replace=False))
        xs, ys = xs[ids], ys[ids]
    colors = rgb[ys, xs].astype(np.float64)
    return xs.astype(np.float64), ys.astype(np.float64), colors


def _base_features(xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    x = xs / max(1.0, w - 1) * 2.0 - 1.0
    y = ys / max(1.0, h - 1) * 2.0 - 1.0
    return np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])


def _panel_indicator(xs: np.ndarray, ys: np.ndarray, panel: PanelHypothesis) -> np.ndarray:
    _, n = _line_basis(panel.angle_deg)
    proj = xs * n[0] + ys * n[1]
    return ((proj >= panel.rho_low) & (proj <= panel.rho_high)).astype(np.float64)


def _fit_linear_color_model(features: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff, *_ = np.linalg.lstsq(features, colors, rcond=None)
    pred = np.clip(features @ coeff, 0.0, 255.0)
    return coeff, pred


def _metrics(colors: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rgb_mae = float(np.mean(np.abs(colors - pred)))
    true_lab = rgb2lab((colors.reshape(-1, 1, 3) / 255.0).astype(np.float32)).reshape(-1, 3)
    pred_lab = rgb2lab((pred.reshape(-1, 1, 3) / 255.0).astype(np.float32)).reshape(-1, 3)
    de = deltaE_ciede2000(true_lab, pred_lab)
    return {
        "mean_deltaE2000": float(np.mean(de)),
        "median_deltaE2000": float(np.median(de)),
        "p90_deltaE2000": float(np.percentile(de, 90)),
        "rgb_mae_8bit": rgb_mae,
    }


def select_background_model(
    rgb: np.ndarray,
    known: np.ndarray,
    panels: list[PanelHypothesis],
    *,
    max_panels: int = 3,
    complexity_penalty: float = 0.04,
) -> dict[str, Any]:
    h, w = known.shape
    xs, ys, colors = _sample_known_pixels(rgb, known)
    base = _base_features(xs, ys, w, h)
    candidates: list[dict[str, Any]] = []

    for k in range(0, min(max_panels, len(panels)) + 1):
        features = [base]
        for panel in panels[:k]:
            features.append(_panel_indicator(xs, ys, panel)[:, None])
        design = np.hstack(features)
        coeff, pred = _fit_linear_color_model(design, colors)
        metrics = _metrics(colors, pred)
        feature_count = int(design.shape[1])
        score = metrics["mean_deltaE2000"] + complexity_penalty * max(0, feature_count - base.shape[1])
        candidates.append(
            {
                "name": "base_only" if k == 0 else f"base_plus_{k}_panel" + ("s" if k != 1 else ""),
                "panel_count": k,
                "radial_shading": False,
                "feature_count": feature_count,
                "selection_score": float(score),
                "metrics": metrics,
                "coefficients": coeff.tolist(),
            }
        )

    _, base_pred = _fit_linear_color_model(base, colors)
    residual = np.mean(np.abs(colors - base_pred), axis=1) + 1e-3
    weight_sum = float(residual.sum())
    cx = float(np.sum(xs * residual) / weight_sum)
    cy = float(np.sum(ys * residual) / weight_sum)
    sigma = max(8.0, 0.38 * math.hypot(w, h))
    radial = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))[:, None]
    design = np.hstack([base, radial])
    coeff, pred = _fit_linear_color_model(design, colors)
    metrics = _metrics(colors, pred)
    score = metrics["mean_deltaE2000"] + complexity_penalty
    candidates.append(
        {
            "name": "base_plus_radial_shading",
            "panel_count": 0,
            "radial_shading": True,
            "radial_center": [cx, cy],
            "radial_sigma": sigma,
            "feature_count": int(design.shape[1]),
            "selection_score": float(score),
            "metrics": metrics,
            "coefficients": coeff.tolist(),
        }
    )

    candidates.sort(key=lambda c: c["selection_score"])
    best = candidates[0]
    base_candidate = next(c for c in candidates if c["name"] == "base_only")
    improvement = 100.0 * (base_candidate["metrics"]["mean_deltaE2000"] - best["metrics"]["mean_deltaE2000"]) / max(
        1e-9, base_candidate["metrics"]["mean_deltaE2000"]
    )
    return {
        "best_model": best["name"],
        "best": best,
        "baseline": base_candidate,
        "mean_deltaE_improvement_percent_vs_base": float(improvement),
        "candidates": candidates,
        "complexity_penalty": float(complexity_penalty),
        "sample_count": int(len(xs)),
    }


def _draw_overlay(rgb: np.ndarray, boundaries: list[Boundary], panels: list[PanelHypothesis]) -> np.ndarray:
    overlay = rgb.copy()
    h, w = overlay.shape[:2]
    for b in boundaries[:12]:
        d, n = _line_basis(b.angle_deg)
        p0 = n * b.rho
        t0, t1 = _line_t_range_in_image(b.angle_deg, b.rho, w, h)
        a = np.rint(p0 + d * t0).astype(int)
        c = np.rint(p0 + d * t1).astype(int)
        cv2.line(overlay, tuple(a), tuple(c), (255, 64, 64), 1, cv2.LINE_AA)
    for p in panels[:6]:
        for rho in (p.rho_low, p.rho_high):
            d, n = _line_basis(p.angle_deg)
            p0 = n * rho
            t0, t1 = _line_t_range_in_image(p.angle_deg, rho, w, h)
            a = np.rint(p0 + d * t0).astype(int)
            c = np.rint(p0 + d * t1).astype(int)
            cv2.line(overlay, tuple(a), tuple(c), (64, 255, 64), 2, cv2.LINE_AA)
    return overlay


def detect_background_panels(
    image_path: str | Path,
    background_known_path: str | Path,
    output_dir: str | Path,
    *,
    max_panels: int = 3,
    max_hypotheses: int = 12,
) -> dict[str, Any]:
    image_path = Path(image_path)
    background_known_path = Path(background_known_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb(image_path)
    known = _load_mask(background_known_path, rgb.shape[:2])
    if int(known.sum()) < 64:
        raise ValueError("background_known has too little authoritative support")

    edges, safe, edge_diag = build_background_safe_edges(rgb, known)
    edge_path = output_dir / "background_edges.png"
    _write_png(edge_path, edges)

    boundaries = detect_boundaries(rgb, safe, edges)
    panels = generate_panel_hypotheses(rgb, safe, boundaries, max_hypotheses=max_hypotheses)
    model_selection = select_background_model(rgb, safe, panels, max_panels=max_panels)

    overlay = _draw_overlay(rgb, boundaries, panels)
    overlay_path = output_dir / "panel_boundary_overlay.png"
    _write_png(overlay_path, overlay)

    report = {
        "schema": "poster-vector-rebuilder.phase24b.v1",
        "stage": "Phase 2.4B automatic panel boundary detection + constrained geometry optimization",
        "source": str(image_path),
        "background_known": str(background_known_path),
        "dimensions": [int(rgb.shape[1]), int(rgb.shape[0])],
        "authoritative_pixel_count": int(known.sum()),
        "safe_authoritative_pixel_count": int(safe.sum()),
        "edge_diagnostics": edge_diag,
        "boundary_count": len(boundaries),
        "panel_hypothesis_count": len(panels),
        "boundaries": [asdict(x) for x in boundaries],
        "panel_hypotheses": [asdict(x) for x in panels],
        "model_selection": model_selection,
        "rules": {
            "visible_pixels_authoritative": True,
            "hidden_pixels_used_for_fitting": False,
            "mask_boundary_excluded_from_edge_detection": True,
            "topology_before_fine_diffvg": True,
            "complexity_penalized": True,
        },
        "outputs": {
            "background_edges": str(edge_path),
            "panel_boundary_overlay": str(overlay_path),
        },
    }
    report_path = output_dir / "panel_detection_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_phase24b(
    job_dir: str | Path,
    *,
    image_path: str | Path | None = None,
    background_known_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_panels: int = 3,
) -> dict[str, Any]:
    job = Path(job_dir)
    image = Path(image_path) if image_path is not None else job / "work" / "normalized_reference.png"
    known = Path(background_known_path) if background_known_path is not None else job / "masks" / "background_known.png"
    out = Path(output_dir) if output_dir is not None else job / "background" / "phase24b"
    return detect_background_panels(image, known, out, max_panels=max_panels)
