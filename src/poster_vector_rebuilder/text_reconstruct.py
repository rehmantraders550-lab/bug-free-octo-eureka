from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
import csv
import io
import json
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageOps

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)


def _run_tesseract(image_path: str | Path, language: str, psm: int) -> str:
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("tesseract executable was not found on PATH")
    cmd = [exe, str(image_path), "stdout", "-l", language, "--psm", str(psm), "tsv"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tesseract failed")
    return proc.stdout


def _parse_tsv(tsv: str, min_confidence: float) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf", -1))
            x, y = int(row["left"]), int(row["top"])
            w, h = int(row["width"]), int(row["height"])
        except (TypeError, ValueError, KeyError):
            continue
        if conf < min_confidence or w <= 0 or h <= 0:
            continue
        rows.append({
            "text": text, "confidence": conf, "x": x, "y": y, "width": w, "height": h,
            "block": int(row.get("block_num") or 0), "paragraph": int(row.get("par_num") or 0),
            "line": int(row.get("line_num") or 0), "word": int(row.get("word_num") or 0),
        })
    return rows


def _group_lines(words: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for word in words:
        grouped.setdefault((word["block"], word["paragraph"], word["line"]), []).append(word)
    lines = []
    for key, items in grouped.items():
        items.sort(key=lambda x: (x["x"], x["word"]))
        x0 = min(i["x"] for i in items); y0 = min(i["y"] for i in items)
        x1 = max(i["x"] + i["width"] for i in items); y1 = max(i["y"] + i["height"] for i in items)
        lines.append({
            "text": " ".join(i["text"] for i in items),
            "confidence": float(np.mean([i["confidence"] for i in items])),
            "x": x0, "y": y0, "width": x1-x0, "height": y1-y0,
            "words": items, "key": key,
        })
    lines.sort(key=lambda x: (x["y"], x["x"]))
    return lines


def _estimate_fill(rgb: np.ndarray, line: dict) -> str:
    h, w = rgb.shape[:2]
    x0 = max(0, line["x"]); y0 = max(0, line["y"])
    x1 = min(w, x0 + line["width"]); y1 = min(h, y0 + line["height"])
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return "#000000"
    pixels = roi.reshape(-1, 3).astype(np.float32)
    lum = pixels.mean(axis=1)
    dark = np.percentile(lum, 25); light = np.percentile(lum, 75)
    center = float(np.median(lum))
    target = pixels[lum <= dark] if abs(center-dark) >= abs(light-center) else pixels[lum >= light]
    color = np.median(target if len(target) else pixels, axis=0).astype(np.uint8)
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def reconstruct_text(
    image_path: str | Path,
    output_svg: str | Path,
    *,
    report_path: str | Path | None = None,
    exclusion_mask_path: str | Path | None = None,
    min_confidence: float = 80.0,
    language: str = "eng",
    psm: int = 11,
) -> dict:
    """Recover reliable OCR as editable SVG text without inventing font identity."""
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8); h, w = rgb.shape[:2]
    root = ET.Element(f"{{{SVG_NS}}}svg", {"version":"1.1","width":str(w),"height":str(h),"viewBox":f"0 0 {w} {h}"})
    layer = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id":"OCR_TEXT", f"{{{INKSCAPE_NS}}}label":"OCR_TEXT", f"{{{INKSCAPE_NS}}}groupmode":"layer"})
    status = "complete"; error = None
    try:
        words = _parse_tsv(_run_tesseract(image_path, language, psm), min_confidence)
    except Exception as exc:
        words = []; status = "unavailable"; error = f"{type(exc).__name__}: {exc}"
    lines = _group_lines(words)
    mask = np.zeros((h, w), dtype=np.uint8)
    for i, line in enumerate(lines, 1):
        font_size = max(5.0, line["height"] * 0.88)
        baseline = line["y"] + line["height"] * 0.86
        fill = _estimate_fill(rgb, line)
        elem = ET.SubElement(layer, f"{{{SVG_NS}}}text", {
            "id": f"text_{i:04d}", "x": f"{line['x']:.3f}", "y": f"{baseline:.3f}",
            "font-family": "Arial,Helvetica,sans-serif", "font-size": f"{font_size:.3f}",
            "fill": fill, "data-ocr-confidence": f"{line['confidence']:.2f}",
        })
        elem.text = line["text"]
        pad = max(1, int(round(line["height"] * 0.12)))
        x0=max(0,line["x"]-pad); y0=max(0,line["y"]-pad)
        x1=min(w,line["x"]+line["width"]+pad); y1=min(h,line["y"]+line["height"]+pad)
        mask[y0:y1,x0:x1]=255
    output_svg = Path(output_svg); output_svg.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_svg, encoding="utf-8", xml_declaration=True)
    exclusion_mask_path = Path(exclusion_mask_path) if exclusion_mask_path else output_svg.with_suffix(".text-mask.png")
    Image.fromarray(mask, mode="L").save(exclusion_mask_path)
    report_path = Path(report_path) if report_path else output_svg.with_suffix(".text.json")
    report = {
        "schema":"poster-vector-text-reconstruction-v1", "status":status, "error":error,
        "engine":"tesseract" if status=="complete" else None, "language":language, "psm":psm,
        "min_confidence":min_confidence, "word_count":len(words), "line_count":len(lines),
        "editable_svg_text": True, "font_exact_match": False,
        "font_policy":"OCR recovers text content and geometry; generic font fallback is used unless a later font matcher proves an exact family.",
        "lines":[{k:v for k,v in line.items() if k not in {"words","key"}} for line in lines],
        "outputs":{"svg":str(output_svg),"exclusion_mask":str(exclusion_mask_path),"report":str(report_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
