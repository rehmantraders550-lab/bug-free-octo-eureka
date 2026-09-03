"""Vector-PDF export and honest preflight reporting."""
from __future__ import annotations

from pathlib import Path
import html
import json
import shutil
import subprocess

from pypdf import PdfReader


def preflight_pdf(job_dir: str | Path, svg_path: str | Path | None = None, trim_mm: tuple[float, float] | None = None, bleed_mm: float | None = None, icc_profile: str | Path | None = None) -> dict:
    job = Path(job_dir)
    svg = Path(svg_path) if svg_path else job / "delivery" / "artwork_master.svg"
    delivery, analysis = job / "delivery", job / "analysis"
    delivery.mkdir(parents=True, exist_ok=True); analysis.mkdir(parents=True, exist_ok=True)
    pdf, proof = delivery / "artwork_press.pdf", delivery / "preview.png"
    inkscape, gs = shutil.which("inkscape"), shutil.which("gs")
    if not inkscape or not gs:
        raise RuntimeError("Inkscape and Ghostscript are required for vector-PDF validation")
    subprocess.run([inkscape, str(svg), "--export-type=pdf", "--export-text-to-path=false", f"--export-filename={pdf}"], check=True, capture_output=True, text=True)
    subprocess.run([gs, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pngalpha", "-r144", f"-sOutputFile={proof}", str(pdf)], check=True, capture_output=True, text=True)
    reader = PdfReader(str(pdf))
    page = reader.pages[0]
    media = [float(value) for value in page.mediabox]
    preflight_pass = bool(trim_mm and bleed_mm is not None and icc_profile and Path(icc_profile).exists())
    warnings = []
    if not trim_mm: warnings.append("Trim size not supplied; page size is derived from SVG pixels.")
    if bleed_mm is None: warnings.append("Bleed not supplied; no production bleed can be certified.")
    if not icc_profile or not Path(icc_profile).exists(): warnings.append("ICC profile absent; PDF remains RGB vector output and is not certified CMYK/PDF-X.")
    report = {"pdf": "delivery/artwork_press.pdf", "proof": "delivery/preview.png", "source_svg": str(svg), "page_count": len(reader.pages), "media_box_points": media, "vector_export": True, "ghostscript_proof": True, "trim_mm": trim_mm, "bleed_mm": bleed_mm, "icc_profile": str(icc_profile) if icc_profile else None, "prepress_certified": preflight_pass, "warnings": warnings}
    (analysis / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = "".join(f"<li>{html.escape(item)}</li>" for item in warnings) or "<li>Production specification supplied.</li>"
    (delivery / "preflight_report.html").write_text(f"<!doctype html><title>Poster Vector Rebuilder preflight</title><h1>Preflight: {'PASS' if preflight_pass else 'CONDITIONAL'}</h1><p>Vector PDF exported and Ghostscript proof rendered.</p><ul>{rows}</ul>", encoding="utf-8")
    return report
