from pathlib import Path
import numpy as np
from PIL import Image

from poster_vector_rebuilder.background import detect_panel_boundaries, fit_background


def _synthetic_panel() -> tuple[np.ndarray, np.ndarray]:
    h, w = 360, 280
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        rgb[y, :, :] = [80 + y // 9, 145 + y // 12, 205 + y // 20]
    polygon = np.array([[0, 80], [w, 38], [w, 166], [0, 208]], np.int32)
    import cv2
    cv2.fillConvexPoly(rgb, polygon, (65, 118, 180))
    known = np.ones((h, w), dtype=bool)
    known[120:260, 80:200] = False
    return rgb, known


def test_detect_panel_boundaries_ignores_masked_foreground():
    rgb, known = _synthetic_panel()
    lines, strength, edges = detect_panel_boundaries(rgb, known)
    assert strength.shape == known.shape
    assert edges.shape == known.shape
    assert lines
    assert abs(lines[0].angle_deg) < 20
    assert lines[0].color_shift_lab > 1


def test_fit_background_writes_phase_gate_artifacts(tmp_path: Path):
    rgb, known = _synthetic_panel()
    job = tmp_path / "job"
    (job / "work").mkdir(parents=True)
    (job / "masks").mkdir(parents=True)
    Image.fromarray(rgb).save(job / "work" / "normalized_reference.png")
    Image.fromarray((known * 255).astype(np.uint8)).save(job / "masks" / "background_known.png")
    report = fit_background(job)
    assert report["model"]["embedded_raster_images"] == 0
    assert (job / "vector" / "background_master.svg").exists()
    assert (job / "analysis" / "background_fit_report.json").exists()
    assert (job / "delivery" / "background_master_corel.svg").exists()
