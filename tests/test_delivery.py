from pathlib import Path
import numpy as np
from PIL import Image

from poster_vector_rebuilder.background import fit_background
from poster_vector_rebuilder.typography import recover_text
from poster_vector_rebuilder.assemble import assemble_artwork
from poster_vector_rebuilder.prepress import preflight_pdf


def test_vector_pdf_delivery_is_rendered_and_reported(tmp_path: Path):
    job = tmp_path / "job"
    (job / "work").mkdir(parents=True)
    (job / "masks").mkdir(parents=True)
    image = np.full((180, 140, 3), [90, 160, 220], dtype=np.uint8)
    image[90:, :] = [55, 120, 185]
    Image.fromarray(image).save(job / "work" / "normalized_reference.png")
    Image.fromarray(np.full((180, 140), 255, dtype=np.uint8)).save(job / "masks" / "background_known.png")
    fit_background(job)
    recover_text(job / "work" / "normalized_reference.png", job / "analysis" / "text_layers.json")
    assembly = assemble_artwork(job)
    assert assembly["embedded_raster_images"] == 0
    report = preflight_pdf(job)
    assert report["vector_export"]
    assert report["ghostscript_proof"]
    assert (job / "delivery" / "preflight_report.html").exists()
