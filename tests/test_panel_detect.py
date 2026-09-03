from pathlib import Path

import cv2
import numpy as np
import pytest

from poster_vector_rebuilder.panel_detect import build_background_safe_edges, detect_background_panels


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _make_diagonal_fixture(tmp_path: Path):
    h, w = 260, 360
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 80 + 55 * xx / (w - 1) + 12 * yy / (h - 1)
    base[..., 1] = 130 + 60 * xx / (w - 1) + 20 * yy / (h - 1)
    base[..., 2] = 185 + 50 * xx / (w - 1) + 25 * yy / (h - 1)

    angle = 28.0
    theta = np.deg2rad(angle)
    normal = np.array([-np.sin(theta), np.cos(theta)])
    projection = xx * normal[0] + yy * normal[1]
    rho_low, rho_high = 28.0, 78.0
    band = (projection >= rho_low) & (projection <= rho_high)
    image = base.copy()
    image[band] = np.clip(image[band] + np.array([24, -5, -22], np.float32), 0, 255)

    radial = np.exp(-((xx - 180) ** 2 + (yy - 120) ** 2) / (2 * 150**2))[..., None]
    image = np.clip(image + 8 * radial, 0, 255).astype(np.uint8)

    known = np.full((h, w), 255, np.uint8)
    cv2.rectangle(known, (135, 70), (225, 190), 0, -1)
    cv2.circle(known, (55, 200), 25, 0, -1)

    image_path = tmp_path / "reference.png"
    mask_path = tmp_path / "background_known.png"
    _write_rgb(image_path, image)
    cv2.imwrite(str(mask_path), known)
    return image_path, mask_path, angle, rho_low, rho_high


def test_detects_parallel_panel_and_improves_model(tmp_path: Path):
    image_path, mask_path, angle, rho_low, rho_high = _make_diagonal_fixture(tmp_path)
    report = detect_background_panels(image_path, mask_path, tmp_path / "phase24b")

    assert Path(report["outputs"]["background_edges"]).is_file()
    assert Path(report["outputs"]["panel_boundary_overlay"]).is_file()
    assert Path(report["outputs"]["report"]).is_file()
    assert report["panel_hypothesis_count"] >= 1

    panel = report["panel_hypotheses"][0]
    angle_error = abs(panel["angle_deg"] - angle)
    angle_error = min(angle_error, 180 - angle_error)
    assert angle_error < 3.0
    assert abs(panel["rho_low"] - rho_low) < 6.0
    assert abs(panel["rho_high"] - rho_high) < 6.0
    assert panel["optimization_success"]
    assert panel["confidence"] > 0.70

    selection = report["model_selection"]
    assert selection["best_model"] == "base_plus_1_panel"
    assert selection["best"]["metrics"]["mean_deltaE2000"] < selection["baseline"]["metrics"]["mean_deltaE2000"]
    assert selection["mean_deltaE_improvement_percent_vs_base"] > 50.0
    assert report["rules"]["hidden_pixels_used_for_fitting"] is False


def test_mask_boundary_is_excluded_from_edge_candidates(tmp_path: Path):
    h, w = 180, 240
    image = np.full((h, w, 3), [90, 145, 200], np.uint8)
    cv2.rectangle(image, (75, 50), (165, 130), (245, 40, 40), -1)
    known = np.full((h, w), 255, np.uint8)
    cv2.rectangle(known, (72, 47), (168, 133), 0, -1)

    edge_map, safe, diagnostics = build_background_safe_edges(image, known > 0, exclusion_radius=6)
    boundary = cv2.morphologyEx(known, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    assert diagnostics["mask_boundary_exclusion_radius_px"] == 6
    assert np.count_nonzero((edge_map > 0) & boundary) == 0
    assert np.count_nonzero(safe & (known == 0)) == 0


def test_rejects_empty_authoritative_mask(tmp_path: Path):
    image = np.full((80, 100, 3), 128, np.uint8)
    image_path = tmp_path / "reference.png"
    mask_path = tmp_path / "background_known.png"
    _write_rgb(image_path, image)
    cv2.imwrite(str(mask_path), np.zeros((80, 100), np.uint8))

    with pytest.raises(ValueError, match="authoritative support"):
        detect_background_panels(image_path, mask_path, tmp_path / "phase24b")
