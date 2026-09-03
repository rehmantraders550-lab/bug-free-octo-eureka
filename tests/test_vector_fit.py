from pathlib import Path
import json
import math

import cv2
import numpy as np

from poster_vector_rebuilder.vector_fit import fit_background_vectors


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def test_phase24c_fits_editable_gradient_and_panel(tmp_path: Path):
    h, w = 220, 320
    yy, xx = np.mgrid[0:h, 0:w]
    angle = 24.0
    theta = math.radians(angle)
    d = np.array([math.cos(theta), math.sin(theta)])
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float)
    proj = xx * d[0] + yy * d[1]
    cp = corners[:, 0] * d[0] + corners[:, 1] * d[1]
    t = (proj - cp.min()) / (cp.max() - cp.min())
    c0 = np.array([52, 104, 176], dtype=float)
    c1 = np.array([132, 190, 238], dtype=float)
    image = c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]

    panel_angle = 32.0
    pt = math.radians(panel_angle)
    nx, ny = -math.sin(pt), math.cos(pt)
    rho = xx * nx + yy * ny
    band = (rho >= 20.0) & (rho <= 68.0)
    image[band] = np.clip(image[band] + np.array([20, -8, -18]), 0, 255)
    image = np.rint(image).astype(np.uint8)

    known = np.full((h, w), 255, np.uint8)
    cv2.rectangle(known, (125, 60), (205, 165), 0, -1)

    image_path = tmp_path / "reference.png"
    mask_path = tmp_path / "known.png"
    _write_rgb(image_path, image)
    cv2.imwrite(str(mask_path), known)

    phase24b_report = {
        "panel_hypotheses": [
            {
                "angle_deg": panel_angle,
                "rho_low": 20.0,
                "rho_high": 68.0,
                "confidence": 0.95,
            }
        ]
    }
    report_path = tmp_path / "phase24b.json"
    report_path.write_text(json.dumps(phase24b_report), encoding="utf-8")

    report = fit_background_vectors(
        image_path,
        mask_path,
        tmp_path / "phase24c",
        phase24b_report_path=report_path,
        max_panels=1,
    )

    svg = Path(report["outputs"]["svg"]).read_text(encoding="utf-8")
    assert "<image" not in svg.lower()
    assert "<linearGradient" in svg
    assert "<polygon" in svg
    assert report["best_model"]["panel_count"] == 1
    assert report["best_model"]["metrics"]["mean_deltaE2000"] < 2.5
    assert report["best_model"]["metrics"]["median_deltaE2000"] < 2.0
    assert report["rules"]["hidden_pixels_used_for_fitting"] is False


def test_phase24c_rejects_empty_known_mask(tmp_path: Path):
    image = np.full((60, 80, 3), 128, np.uint8)
    image_path = tmp_path / "reference.png"
    mask_path = tmp_path / "known.png"
    _write_rgb(image_path, image)
    cv2.imwrite(str(mask_path), np.zeros((60, 80), np.uint8))

    try:
        fit_background_vectors(image_path, mask_path, tmp_path / "phase24c")
    except ValueError as exc:
        assert "authoritative support" in str(exc)
    else:
        raise AssertionError("Expected empty authoritative mask to be rejected")
