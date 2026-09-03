import json
from pathlib import Path

import numpy as np
from PIL import Image

from poster_vector_rebuilder.phase24d import recover_hidden_background, run_phase24_acceptance_gate


def _fixture(tmp_path: Path):
    h, w = 100, 160
    x = np.linspace(0.0, 1.0, w)[None, :, None]
    start = np.array([30.0, 90.0, 160.0])[None, None, :]
    end = np.array([90.0, 180.0, 230.0])[None, None, :]
    rgb = np.repeat(start * (1 - x) + end * x, h, axis=0).astype(np.uint8)
    image = tmp_path / "reference.png"
    Image.fromarray(rgb).save(image)

    known = np.full((h, w), 255, dtype=np.uint8)
    known[30:70, 55:105] = 0
    mask = tmp_path / "known.png"
    Image.fromarray(known).save(mask)

    report = {
        "best_model": {
            "angle_deg": 0.0,
            "start_rgb": [30.0, 90.0, 160.0],
            "end_rgb": [90.0, 180.0, 230.0],
            "panel_deltas_rgb": [],
        },
        "panels_used": [],
    }
    report_path = tmp_path / "phase24c_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    svg = tmp_path / "background_fitted.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 100">'
        '<g id="background-base"><rect width="160" height="100" fill="#336699"/></g>'
        '</svg>',
        encoding="utf-8",
    )
    return image, mask, report_path, svg


def test_recovery_preserves_authoritative_pixels(tmp_path: Path):
    image, mask, report, _ = _fixture(tmp_path)
    result = recover_hidden_background(image, mask, report, tmp_path / "recovery")
    recovered = np.asarray(Image.open(result["outputs"]["recovered"]).convert("RGB"))
    source = np.asarray(Image.open(image).convert("RGB"))
    known = np.asarray(Image.open(mask).convert("L")) >= 128
    assert np.array_equal(recovered[known], source[known])
    assert result["ai_inpainting_used"] is False
    assert result["method"] == "mathematical_vector_continuation"


def test_acceptance_gate_passes_exact_editable_gradient(tmp_path: Path):
    image, mask, report, svg = _fixture(tmp_path)
    result = run_phase24_acceptance_gate(
        image,
        mask,
        report,
        svg,
        tmp_path / "gate",
        max_mean_delta_e=1.0,
        max_rgb_mae=1.0,
        min_ssim=0.99,
        max_boundary_error=0.01,
    )
    assert result["passed"] is True
    assert result["checks"]["raster_free_svg"] is True
    assert result["checks"]["editable_svg"] is True


def test_acceptance_gate_rejects_embedded_raster(tmp_path: Path):
    image, mask, report, _ = _fixture(tmp_path)
    svg = tmp_path / "bad.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 100">'
        '<g><image href="data:image/png;base64,AA==" width="1" height="1"/></g>'
        '</svg>',
        encoding="utf-8",
    )
    result = run_phase24_acceptance_gate(image, mask, report, svg, tmp_path / "gate_bad")
    assert result["passed"] is False
    assert result["checks"]["raster_free_svg"] is False
