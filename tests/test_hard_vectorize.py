from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from poster_vector_rebuilder.hard_vectorize import vectorize_hard_graphic


def _make_fixture(path: Path) -> None:
    rgb = np.full((180, 240, 4), 255, dtype=np.uint8)
    rgb[..., :3] = [245, 245, 245]
    rgb[25:155, 30:210, :3] = [20, 80, 180]
    rgb[55:125, 70:170, :3] = [245, 245, 245]
    rgb[75:105, 95:145, :3] = [220, 30, 40]
    Image.fromarray(rgb, mode="RGBA").save(path)


def test_hard_vectorize_emits_editable_paths_without_raster(tmp_path: Path):
    source = tmp_path / "graphic.png"
    output = tmp_path / "graphic.svg"
    _make_fixture(source)

    report = vectorize_hard_graphic(
        source,
        output,
        backend="opencv",
        colors=3,
        min_area=4,
        simplify=0.001,
    )

    assert output.exists()
    root = ET.parse(output).getroot()
    tags = [elem.tag.split("}")[-1] for elem in root.iter()]
    assert "path" in tags
    assert "image" not in tags
    assert report["path_count"] >= 2
    assert report["vector_coverage_ratio"] > 0.97
    assert report["rgb_mae_on_vector_coverage"] < 8.0


def test_hard_vectorize_respects_selection_mask(tmp_path: Path):
    source = tmp_path / "graphic.png"
    output = tmp_path / "masked.svg"
    mask_path = tmp_path / "mask.png"
    _make_fixture(source)

    mask = np.zeros((180, 240), dtype=np.uint8)
    mask[20:160, 20:220] = 255
    Image.fromarray(mask, mode="L").save(mask_path)

    report = vectorize_hard_graphic(
        source,
        output,
        mask_path=mask_path,
        backend="opencv",
        colors=3,
        min_area=4,
    )

    assert report["mask_pixels"] == int(np.count_nonzero(mask))
    assert 0 < report["mask_coverage_ratio"] < 1
    assert report["path_count"] > 0


def test_hard_vectorize_rejects_empty_mask(tmp_path: Path):
    source = tmp_path / "graphic.png"
    output = tmp_path / "empty.svg"
    mask_path = tmp_path / "empty-mask.png"
    _make_fixture(source)
    Image.fromarray(np.zeros((180, 240), dtype=np.uint8), mode="L").save(mask_path)

    try:
        vectorize_hard_graphic(source, output, mask_path=mask_path, backend="opencv")
    except ValueError as exc:
        assert "no pixels" in str(exc).lower()
    else:
        raise AssertionError("Expected empty vectorization mask to fail")
