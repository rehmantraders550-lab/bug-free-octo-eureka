from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image

from poster_vector_rebuilder import text_reconstruct
from poster_vector_rebuilder.final_assembly import assemble_master_svg, LAYERS
from poster_vector_rebuilder.prepress import svg_preflight


def test_text_reconstruction_high_confidence(monkeypatch, tmp_path):
    img = np.full((100, 240, 3), 245, dtype=np.uint8)
    img[20:50, 30:145] = 20
    src = tmp_path / "in.png"; Image.fromarray(img).save(src)
    tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t30\t20\t50\t30\t96\tHELLO\n5\t1\t1\t1\t1\t2\t85\t20\t60\t30\t94\tWORLD\n"
    monkeypatch.setattr(text_reconstruct, "_run_tesseract", lambda *a, **k: tsv)
    report = text_reconstruct.reconstruct_text(src, tmp_path/"text.svg", exclusion_mask_path=tmp_path/"mask.png")
    assert report["line_count"] == 1
    assert report["font_exact_match"] is False
    root = ET.parse(tmp_path/"text.svg").getroot()
    texts = [e for e in root.iter() if e.tag.endswith("text")]
    assert len(texts) == 1 and texts[0].text == "HELLO WORLD"
    assert np.asarray(Image.open(tmp_path/"mask.png")).sum() > 0


def test_final_assembly_layers_and_preflight(tmp_path):
    bg = tmp_path/"bg.svg"; bg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><rect width="400" height="300" fill="#eee"/></svg>')
    sem = tmp_path/"sem.svg"; sem.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><g><rect id="big" x="20" y="20" width="180" height="100" fill="#c00"/><circle id="tiny" cx="350" cy="250" r="4" fill="#00c"/><polygon id="mid" points="200,160 250,180 215,220" fill="#0c0"/></g></svg>')
    text = tmp_path/"text.svg"; text.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><g><text x="40" y="60">TEST</text></g></svg>')
    out = tmp_path/"master.svg"
    rep = assemble_master_svg(out, width=400, height=300, background_svg=bg, semantic_svg=sem, text_svg=text)
    assert rep["corel_compatibility"] == "pass"
    root = ET.parse(out).getroot()
    ids = {e.get("id") for e in root if e.tag.endswith("g")}
    assert set(LAYERS).issubset(ids)
    pf = svg_preflight(out)
    assert pf["passed"]
    assert pf["text_count"] == 1
    assert not pf["raster_images"]


def test_declared_photo_raster_is_allowed(tmp_path):
    out = tmp_path/"master.svg"
    rep = assemble_master_svg(out, width=200, height=100, photographic_href="assets/photo.png")
    pf = svg_preflight(out)
    assert rep["raster_image_count"] == 1
    assert pf["passed"]
    assert pf["raster_images"][0]["declared_status"] == "raster-photographic-fallback"
