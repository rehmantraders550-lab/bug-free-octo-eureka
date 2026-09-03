from pathlib import Path
import json
import numpy as np
from PIL import Image

from poster_vector_rebuilder.normalize import normalize_reference, order_quad, _select_rotation_from_ocr_scores


def test_order_quad():
    pts = np.array([[90, 90], [10, 10], [90, 10], [10, 90]], dtype=np.float32)
    q = order_quad(pts)
    assert q.tolist() == [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]


def test_manual_normalization(tmp_path: Path):
    canvas = np.full((300, 400, 3), 235, dtype=np.uint8)
    import cv2
    poly = np.array([[50, 40], [350, 25], [365, 265], [35, 275]], np.int32)
    cv2.fillConvexPoly(canvas, poly, (40, 120, 210))
    src = tmp_path / "source.png"
    Image.fromarray(canvas).save(src)

    job = tmp_path / "job"
    result = normalize_reference(src, job, rotation="keep", corners=poly.tolist())

    assert (job / "input" / "source_original.png").exists()
    assert (job / "work" / "normalized_reference.png").exists()
    assert (job / "metadata" / "source_manifest.json").exists()
    assert (job / "metadata" / "geometry.json").exists()
    assert result["quad_detection"]["method"] == "manual_override"
    assert result["normalized_width"] > 250
    assert result["normalized_height"] > 200

    manifest = json.loads((job / "metadata" / "source_manifest.json").read_text())
    assert len(manifest["sha256"]) == 64


def test_auto_orientation_requires_clear_ocr_win():
    assert _select_rotation_from_ocr_scores({"keep": 210, "180": 310, "90cw": 20, "90ccw": 15}) == "180"
    assert _select_rotation_from_ocr_scores({"keep": 130, "180": 140, "90cw": 20, "90ccw": 15}) == "keep"
