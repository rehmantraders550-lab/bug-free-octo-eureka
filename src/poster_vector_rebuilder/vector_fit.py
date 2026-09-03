from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from skimage.color import deltaE_ciede2000, rgb2lab

from .panel_detect import detect_background_panels


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


def _sample_known(rgb: np.ndarray, known: np.ndarray, max_samples: int = 60000):
    ys, xs = np.nonzero(known)
    if len(xs) < 64:
        raise ValueError("background_known has too little authoritative support")
    if len(xs) > max_samples:
        rng = np.random.default_rng(0)
        ids = np.sort(rng.choice(len(xs), max_samples, replace=False))
        xs, ys = xs[ids], ys[ids]
    return xs.astype(np.float64), ys.astype(np.float64), rgb[ys, xs].astype(np.float64)


def _projection(xs: np.ndarray, ys: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg % 180.0)
    return xs * math.cos(theta) + ys * math.sin(theta)


def _line_range(angle_deg: float, w: int, h: int) -> tuple[float, float]:
    corners = np.array([[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]])
    p = _projection(corners[:, 0], corners[:, 1], angle_deg)
    return float(p.min()), float(p.max())


def _gradient_prediction(xs: np.ndarray, ys: np.ndarray, w: int, h: int, params: np.ndarray) -> np.ndarray:
    angle = float(params[0])
    c0 = params[1:4]
    c1 = params[4:7]
    lo, hi = _line_range(angle, w, h)
    t = (_projection(xs, ys, angle) - lo) / max(1e-9, hi - lo)
    t = np.clip(t, 0.0, 1.0)[:, None]
    return c0[None, :] * (1.0 - t) + c1[None, :] * t


def _panel_indicator(xs: np.ndarray, ys: np.ndarray, panel: dict[str, Any]) -> np.ndarray:
    angle = float(panel["angle_deg"])
    theta = math.radians(angle % 180.0)
    nx, ny = -math.sin(theta), math.cos(theta)
    p = xs * nx + ys * ny
    return (p >= float(panel["rho_low"])) & (p <= float(panel["rho_high"]))


def _initial_gradient(xs: np.ndarray, ys: np.ndarray, colors: np.ndarray, w: int, h: int) -> np.ndarray:
    gray = colors @ np.array([0.2126, 0.7152, 0.0722])
    x = xs - xs.mean()
    y = ys - ys.mean()
    design = np.column_stack([np.ones_like(x), x, y])
    coeff, *_ = np.linalg.lstsq(design, gray, rcond=None)
    angle = math.degrees(math.atan2(float(coeff[2]), float(coeff[1]))) % 180.0
    p = _projection(xs, ys, angle)
    q0, q1 = np.percentile(p, [3, 97])
    low = colors[p <= q0 + 1e-9]
    high = colors[p >= q1 - 1e-9]
    c0 = np.median(low, axis=0) if len(low) else np.median(colors, axis=0)
    c1 = np.median(high, axis=0) if len(high) else np.median(colors, axis=0)
    return np.array([angle, *c0.tolist(), *c1.tolist()], dtype=np.float64)


def _fit_candidate(
    xs: np.ndarray,
    ys: np.ndarray,
    colors: np.ndarray,
    w: int,
    h: int,
    panels: list[dict[str, Any]],
    *,
    complexity_penalty: float,
) -> dict[str, Any]:
    base0 = _initial_gradient(xs, ys, colors, w, h)
    indicators = [_panel_indicator(xs, ys, p) for p in panels]
    deltas0: list[float] = []
    base_pred0 = _gradient_prediction(xs, ys, w, h, base0)
    for indicator in indicators:
        if np.count_nonzero(indicator) >= 8:
            delta = np.median(colors[indicator] - base_pred0[indicator], axis=0)
        else:
            delta = np.zeros(3)
        deltas0.extend(delta.tolist())
    p0 = np.array([*base0.tolist(), *deltas0], dtype=np.float64)

    lower = np.array([0.0, 0, 0, 0, 0, 0, 0] + [-128.0] * (3 * len(panels)), dtype=np.float64)
    upper = np.array([180.0, 255, 255, 255, 255, 255, 255] + [128.0] * (3 * len(panels)), dtype=np.float64)

    def predict(params: np.ndarray) -> np.ndarray:
        pred = _gradient_prediction(xs, ys, w, h, params[:7])
        cursor = 7
        for indicator in indicators:
            delta = params[cursor:cursor + 3]
            pred[indicator] += delta
            cursor += 3
        return np.clip(pred, 0.0, 255.0)

    def residual(params: np.ndarray) -> np.ndarray:
        return ((predict(params) - colors) / 255.0).ravel()

    result = least_squares(
        residual,
        np.clip(p0, lower, upper),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.025,
        max_nfev=180,
    )
    pred = predict(result.x)
    true_lab = rgb2lab((colors.reshape(-1, 1, 3) / 255.0).astype(np.float32)).reshape(-1, 3)
    pred_lab = rgb2lab((pred.reshape(-1, 1, 3) / 255.0).astype(np.float32)).reshape(-1, 3)
    de = deltaE_ciede2000(true_lab, pred_lab)
    metrics = {
        "mean_deltaE2000": float(np.mean(de)),
        "median_deltaE2000": float(np.median(de)),
        "p90_deltaE2000": float(np.percentile(de, 90)),
        "rgb_mae_8bit": float(np.mean(np.abs(colors - pred))),
    }
    parameter_count = 7 + 3 * len(panels)
    score = metrics["mean_deltaE2000"] + complexity_penalty * max(0, parameter_count - 7)
    deltas = []
    cursor = 7
    for _ in panels:
        deltas.append([float(v) for v in result.x[cursor:cursor + 3]])
        cursor += 3
    return {
        "panel_count": len(panels),
        "parameter_count": parameter_count,
        "selection_score": float(score),
        "success": bool(result.success),
        "message": str(result.message),
        "angle_deg": float(result.x[0] % 180.0),
        "start_rgb": [float(v) for v in result.x[1:4]],
        "end_rgb": [float(v) for v in result.x[4:7]],
        "panel_deltas_rgb": deltas,
        "metrics": metrics,
    }


def _clip_polygon(poly: list[np.ndarray], normal: np.ndarray, rho: float, keep_greater: bool) -> list[np.ndarray]:
    if not poly:
        return []
    out: list[np.ndarray] = []

    def inside(p: np.ndarray) -> bool:
        value = float(np.dot(normal, p) - rho)
        return value >= -1e-7 if keep_greater else value <= 1e-7

    for a, b in zip(poly, poly[1:] + poly[:1]):
        ia, ib = inside(a), inside(b)
        if ia:
            out.append(a)
        if ia != ib:
            da = float(np.dot(normal, a) - rho)
            db = float(np.dot(normal, b) - rho)
            t = da / max(1e-12, da - db)
            out.append(a + t * (b - a))
    return out


def _panel_polygon(panel: dict[str, Any], w: int, h: int) -> list[tuple[float, float]]:
    theta = math.radians(float(panel["angle_deg"]) % 180.0)
    n = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    poly = [
        np.array([0.0, 0.0]),
        np.array([w - 1.0, 0.0]),
        np.array([w - 1.0, h - 1.0]),
        np.array([0.0, h - 1.0]),
    ]
    poly = _clip_polygon(poly, n, float(panel["rho_low"]), True)
    poly = _clip_polygon(poly, n, float(panel["rho_high"]), False)
    return [(float(p[0]), float(p[1])) for p in poly]


def _rgb(values) -> str:
    v = np.clip(np.rint(values), 0, 255).astype(int)
    return f"rgb({v[0]},{v[1]},{v[2]})"


def _gradient_endpoints(angle_deg: float, w: int, h: int) -> tuple[float, float, float, float]:
    theta = math.radians(angle_deg % 180.0)
    d = np.array([math.cos(theta), math.sin(theta)])
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
    t = (corners - center) @ d
    p0 = center + d * float(t.min())
    p1 = center + d * float(t.max())
    return float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])


def _svg_document(w: int, h: int, model: dict[str, Any], panels: list[dict[str, Any]]) -> str:
    x1, y1, x2, y2 = _gradient_endpoints(model["angle_deg"], w, h)
    start = np.array(model["start_rgb"], dtype=np.float64)
    end = np.array(model["end_rgb"], dtype=np.float64)
    defs = [
        f'<linearGradient id="baseGradient" gradientUnits="userSpaceOnUse" x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}">',
        f'<stop offset="0" stop-color="{_rgb(start)}"/>',
        f'<stop offset="1" stop-color="{_rgb(end)}"/>',
        '</linearGradient>',
    ]
    body = ['<g id="background-base"><rect x="0" y="0" width="100%" height="100%" fill="url(#baseGradient)"/></g>']
    for i, (panel, delta) in enumerate(zip(panels, model["panel_deltas_rgb"])):
        delta = np.array(delta, dtype=np.float64)
        pid = f"panelGradient{i + 1}"
        defs.extend([
            f'<linearGradient id="{pid}" gradientUnits="userSpaceOnUse" x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}">',
            f'<stop offset="0" stop-color="{_rgb(start + delta)}"/>',
            f'<stop offset="1" stop-color="{_rgb(end + delta)}"/>',
            '</linearGradient>',
        ])
        polygon = _panel_polygon(panel, w, h)
        if len(polygon) >= 3:
            points = ' '.join(f'{x:.3f},{y:.3f}' for x, y in polygon)
            body.append(f'<g id="panel-{i + 1}" data-confidence="{float(panel.get("confidence", 0.0)):.4f}"><polygon points="{points}" fill="url(#{pid})"/></g>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        '<metadata>Poster Vector Rebuilder Phase 2.4C deterministic editable background fit</metadata>\n'
        '<defs>' + ''.join(defs) + '</defs>\n' + '\n'.join(body) + '\n</svg>\n'
    )


def fit_background_vectors(
    image_path: str | Path,
    background_known_path: str | Path,
    output_dir: str | Path,
    *,
    phase24b_report_path: str | Path | None = None,
    max_panels: int = 3,
    complexity_penalty: float = 0.06,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = _load_rgb(image_path)
    known = _load_mask(background_known_path, rgb.shape[:2])
    h, w = known.shape
    xs, ys, colors = _sample_known(rgb, known)

    if phase24b_report_path is None:
        phase24b = detect_background_panels(image_path, background_known_path, output_dir / "phase24b", max_panels=max_panels)
    else:
        phase24b = json.loads(Path(phase24b_report_path).read_text(encoding="utf-8"))
    hypotheses = list(phase24b.get("panel_hypotheses", []))[:max_panels]

    candidates = []
    for k in range(0, len(hypotheses) + 1):
        candidates.append(_fit_candidate(xs, ys, colors, w, h, hypotheses[:k], complexity_penalty=complexity_penalty))
    candidates.sort(key=lambda x: x["selection_score"])
    best = candidates[0]
    used_panels = hypotheses[: int(best["panel_count"])]

    svg_path = output_dir / "background_fitted.svg"
    svg_path.write_text(_svg_document(w, h, best, used_panels), encoding="utf-8")
    if "<image" in svg_path.read_text(encoding="utf-8").lower():
        raise RuntimeError("Phase 2.4C output must not embed raster images")

    report = {
        "schema": "poster-vector-rebuilder.phase24c.v1",
        "stage": "Phase 2.4C constrained editable vector fitting",
        "source": str(image_path),
        "background_known": str(background_known_path),
        "dimensions": [w, h],
        "authoritative_pixel_count": int(known.sum()),
        "sample_count": int(len(xs)),
        "best_model": best,
        "candidates": candidates,
        "panels_used": used_panels,
        "complexity_penalty": float(complexity_penalty),
        "optimizer": "scipy.optimize.least_squares",
        "diffvg": {"required": False, "status": "optional_fine_tuning_not_yet_applied"},
        "rules": {
            "visible_pixels_authoritative": True,
            "hidden_pixels_used_for_fitting": False,
            "embedded_raster": False,
            "editable_svg_primitives": True,
        },
        "outputs": {"svg": str(svg_path), "report": str(output_dir / "phase24c_report.json")},
    }
    report_path = output_dir / "phase24c_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
