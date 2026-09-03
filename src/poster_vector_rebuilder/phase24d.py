from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from .vector_fit import _gradient_prediction, _load_mask, _load_rgb, _panel_indicator


def _predict_full(width: int, height: int, model: dict[str, Any], panels: list[dict[str, Any]]) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    xs = xx.ravel().astype(np.float64)
    ys = yy.ravel().astype(np.float64)
    params = np.array([model["angle_deg"], *model["start_rgb"], *model["end_rgb"]], dtype=np.float64)
    pred = _gradient_prediction(xs, ys, width, height, params)
    for panel, delta in zip(panels, model.get("panel_deltas_rgb", [])):
        pred[_panel_indicator(xs, ys, panel)] += np.asarray(delta, dtype=np.float64)
    return np.clip(pred, 0, 255).reshape(height, width, 3).astype(np.uint8)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def recover_hidden_background(
    image_path: str | Path,
    background_known_path: str | Path,
    phase24c_report_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Phase 2.4D: continue the measured vector model through hidden areas.

    This stage deliberately does not overwrite authoritative pixels. Hidden pixels are
    reconstructed from the fitted SVG model first. AI inpainting is left as an optional
    future fallback only for jobs whose acceptance metrics show the mathematical model is
    inadequate.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = _load_rgb(image_path)
    known = _load_mask(background_known_path, rgb.shape[:2])
    phase24c = json.loads(Path(phase24c_report_path).read_text(encoding="utf-8"))
    model = phase24c["best_model"]
    panels = phase24c.get("panels_used", [])
    h, w = known.shape
    predicted = _predict_full(w, h, model, panels)

    recovered = predicted.copy()
    recovered[known] = rgb[known]
    confidence = np.where(known, 255, 128).astype(np.uint8)

    recovered_path = output_dir / "background_recovered.png"
    hypothesis_path = output_dir / "background_model_hypothesis.png"
    confidence_path = output_dir / "background_recovery_confidence.png"
    _save_rgb(recovered_path, recovered)
    _save_rgb(hypothesis_path, predicted)
    cv2.imencode(".png", confidence)[1].tofile(str(confidence_path))

    report = {
        "schema": "poster-vector-rebuilder.phase24d.v1",
        "stage": "Phase 2.4D hidden-area recovery",
        "method": "mathematical_vector_continuation",
        "authoritative_pixels_preserved": True,
        "known_pixel_count": int(known.sum()),
        "hidden_pixel_count": int((~known).sum()),
        "hidden_ratio": float((~known).mean()),
        "confidence": {"known": "A", "hidden": "C"},
        "ai_inpainting_used": False,
        "ai_inpainting_policy": "optional fallback only after acceptance failure",
        "outputs": {
            "recovered": str(recovered_path),
            "model_hypothesis": str(hypothesis_path),
            "confidence": str(confidence_path),
        },
    }
    report_path = output_dir / "phase24d_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _svg_structure(svg_path: str | Path) -> dict[str, Any]:
    text = Path(svg_path).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    tags = [elem.tag.rsplit("}", 1)[-1].lower() for elem in root.iter()]
    group_count = sum(tag == "g" for tag in tags)
    primitive_count = sum(tag in {"path", "rect", "polygon", "ellipse", "circle", "line", "polyline", "text"} for tag in tags)
    raster_count = sum(tag == "image" for tag in tags)
    return {
        "raster_image_count": raster_count,
        "group_count": group_count,
        "editable_primitive_count": primitive_count,
        "has_viewbox": bool(root.attrib.get("viewBox")),
        "raster_free": raster_count == 0,
        "editable": group_count >= 1 and primitive_count >= 1,
    }


def run_phase24_acceptance_gate(
    image_path: str | Path,
    background_known_path: str | Path,
    phase24c_report_path: str | Path,
    svg_path: str | Path,
    output_dir: str | Path,
    *,
    max_mean_delta_e: float = 12.0,
    max_rgb_mae: float = 18.0,
    min_ssim: float = 0.82,
    max_boundary_error: float = 0.035,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = _load_rgb(image_path)
    known = _load_mask(background_known_path, rgb.shape[:2])
    phase24c = json.loads(Path(phase24c_report_path).read_text(encoding="utf-8"))
    model = phase24c["best_model"]
    panels = phase24c.get("panels_used", [])
    h, w = known.shape
    pred = _predict_full(w, h, model, panels)

    truth = rgb[known].astype(np.float32)
    fitted = pred[known].astype(np.float32)
    true_lab = rgb2lab((truth.reshape(-1, 1, 3) / 255.0).astype(np.float32)).reshape(-1, 3)
    pred_lab = rgb2lab((fitted.reshape(-1, 1, 3) / 255.0).astype(np.float32)).reshape(-1, 3)
    de = deltaE_ciede2000(true_lab, pred_lab)
    mae = float(np.mean(np.abs(truth - fitted)))

    masked_truth = rgb.copy()
    masked_pred = pred.copy()
    fill = np.median(rgb[known], axis=0).astype(np.uint8)
    masked_truth[~known] = fill
    masked_pred[~known] = fill
    ssim = float(structural_similarity(masked_truth, masked_pred, channel_axis=2, data_range=255))

    gray_true = cv2.cvtColor(masked_truth, cv2.COLOR_RGB2GRAY)
    gray_pred = cv2.cvtColor(masked_pred, cv2.COLOR_RGB2GRAY)
    edge_true = cv2.Canny(gray_true, 50, 140) > 0
    edge_pred = cv2.Canny(gray_pred, 50, 140) > 0
    valid_edges = known & (edge_true | edge_pred)
    boundary_error = float(np.count_nonzero((edge_true ^ edge_pred) & known) / max(1, np.count_nonzero(valid_edges)))

    structure = _svg_structure(svg_path)
    checks = {
        "mean_deltaE2000": float(np.mean(de)) <= max_mean_delta_e,
        "rgb_mae": mae <= max_rgb_mae,
        "ssim": ssim >= min_ssim,
        "structural_boundary_error": boundary_error <= max_boundary_error,
        "raster_free_svg": structure["raster_free"],
        "editable_svg": structure["editable"] and structure["has_viewbox"],
    }
    passed = all(checks.values())
    report = {
        "schema": "poster-vector-rebuilder.phase24.acceptance.v1",
        "stage": "Phase 2.4 acceptance gate",
        "passed": passed,
        "metrics": {
            "mean_deltaE2000": float(np.mean(de)),
            "median_deltaE2000": float(np.median(de)),
            "p90_deltaE2000": float(np.percentile(de, 90)),
            "rgb_mae_8bit": mae,
            "ssim": ssim,
            "structural_boundary_error": boundary_error,
        },
        "thresholds": {
            "max_mean_delta_e": max_mean_delta_e,
            "max_rgb_mae": max_rgb_mae,
            "min_ssim": min_ssim,
            "max_boundary_error": max_boundary_error,
        },
        "checks": checks,
        "svg_structure": structure,
        "evaluation_scope": "authoritative background pixels only; hidden pixels excluded from truth metrics",
    }
    report_path = output_dir / "phase24_acceptance_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
