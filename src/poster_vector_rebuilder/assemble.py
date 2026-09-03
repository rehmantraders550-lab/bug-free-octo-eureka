"""CorelDRAW-safe final SVG assembly without hidden raster content."""
from __future__ import annotations

from pathlib import Path
import html
import json
import xml.etree.ElementTree as ET


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def assemble_artwork(job_dir: str | Path) -> dict:
    job = Path(job_dir)
    background_path = job / "vector" / "background_master.svg"
    text_path = job / "analysis" / "text_layers.json"
    if not background_path.exists():
        raise FileNotFoundError("Run the background phase before SVG assembly")
    root = ET.fromstring(background_path.read_text(encoding="utf-8"))
    width, height, viewbox = root.get("width"), root.get("height"), root.get("viewBox")
    ns = "{http://www.w3.org/2000/svg}"
    defs = root.find(f"{ns}defs")
    background = root.find(f"{ns}g")
    text_result = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else {"items": []}
    text_nodes = []
    for index, item in enumerate(text_result.get("items", []), start=1):
        font_size = max(7, item["height"] * 0.92)
        text_nodes.append(
            f"<text id=\"OCR-Text-{index:03d}\" x=\"{item['x']}\" y=\"{item['y'] + item['height']}\" "
            f"font-family=\"sans-serif\" font-size=\"{font_size:.2f}\" fill=\"#202020\" "
            f"aria-label=\"OCR confidence {item['confidence']:.1f}; font unconfirmed\">{_escape(item['text'])}</text>"
        )
    source_defs = ET.tostring(defs, encoding="unicode") if defs is not None else "<defs/>"
    source_background = ET.tostring(background, encoding="unicode") if background is not None else '<g id="00_BACKGROUND"/>'
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{viewbox}">'
        '<title>Poster Vector Rebuilder editable master</title>'
        '<desc>Visible background pixels were vector fitted. OCR text is editable but font identity requires review. Photographic foreground is not falsely represented as vector.</desc>'
        f'{source_defs}<g id="ARTWORK">{source_background}'
        '<g id="10_HERO__UNRECOVERED_PHOTOGRAPHIC_CONTENT" data-confidence="D"/>'
        '<g id="20_BRAND__OCR_TEXT" data-font-status="unconfirmed">' + "".join(text_nodes) + '</g>'
        '<g id="30_DECORATION"/><g id="40_ICONS"/><g id="90_PREPRESS"/></g></svg>'
    )
    delivery = job / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    master = delivery / "artwork_master.svg"
    master.write_text(svg, encoding="utf-8")
    report = {"master_svg": "delivery/artwork_master.svg", "layers": ["00_BACKGROUND", "10_HERO__UNRECOVERED_PHOTOGRAPHIC_CONTENT", "20_BRAND__OCR_TEXT", "30_DECORATION", "40_ICONS", "90_PREPRESS"], "embedded_raster_images": 0, "editable_text_items": len(text_nodes), "foreground_status": "Photographic foreground requires approved raster retention or manual/vector illustration; it was not falsely embedded as vector."}
    (job / "analysis" / "assembly_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
