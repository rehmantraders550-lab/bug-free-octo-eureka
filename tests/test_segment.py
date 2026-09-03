from pathlib import Path
import numpy as np
from PIL import Image

from poster_vector_rebuilder.segment import deterministic_foreground_risk, segment_reference


def test_foreground_risk_detects_nonblue_object():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:] = [90, 170, 225]
    rgb[70:170, 100:220] = [220, 55, 25]
    risk = deterministic_foreground_risk(rgb)
    assert risk.shape == (240, 320)
    assert float(risk[100:140, 130:190].mean()) > float(risk[15:55, 15:70].mean())


def test_segment_reference_writes_confidence_masks(tmp_path: Path):
    job = tmp_path / "job"
    work = job / "work"
    work.mkdir(parents=True)
    rgb = np.zeros((300, 220, 3), dtype=np.uint8)
    rgb[:] = [75, 160, 220]
    rgb[90:210, 60:165] = [240, 240, 240]
    Image.fromarray(rgb).save(work / "normalized_reference.png")

    result = segment_reference(job)
    assert result["ratios"]["background_known"] > 0
    assert result["ratios"]["foreground"] > 0
    for rel in result["outputs"].values():
        assert (job / rel).exists()
