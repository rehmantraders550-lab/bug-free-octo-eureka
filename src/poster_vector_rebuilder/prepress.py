from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
import json
import shutil
import subprocess

REQUIRED_LAYERS=("00_BACKGROUND","10_HERO","20_BRAND","30_DECORATION","40_ICONS","90_PREPRESS")


def _run(cmd):
    proc=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,check=False)
    return {"command":[str(x) for x in cmd],"returncode":proc.returncode,"stdout":proc.stdout[-4000:],"stderr":proc.stderr[-4000:]}


def _local(tag): return tag.rsplit("}",1)[-1]


def svg_preflight(svg_path: str | Path) -> dict:
    root=ET.parse(svg_path).getroot(); tags=[_local(e.tag) for e in root.iter()]
    layer_ids={e.get("id") for e in root if _local(e.tag)=="g"}
    images=[e for e in root.iter() if _local(e.tag)=="image"]
    raster_status=[]
    for image in images:
        href=image.get("href") or image.get("{http://www.w3.org/1999/xlink}href") or ""
        raster_status.append({"id":image.get("id"),"href":href,"declared_status":image.get("data-vector-status")})
    checks={
        "viewbox":bool(root.get("viewBox")),
        "required_layers":all(x in layer_ids for x in REQUIRED_LAYERS),
        "no_unsupported_dom":not any(t in {"foreignObject","flowRoot","script"} for t in tags),
        "node_count_practical":len(tags)<=12000,
        "raster_images_declared":all(x["declared_status"]=="raster-photographic-fallback" for x in raster_status),
    }
    return {"checks":checks,"passed":all(checks.values()),"node_count":len(tags),"text_count":tags.count("text"),"raster_images":raster_status,"layer_ids":sorted(x for x in layer_ids if x)}


def export_prepress_package(master_svg: str | Path, output_dir: str | Path, *, proof_dpi: int=150) -> dict:
    """Export editable/press PDFs, proof PNG, and multi-validator preflight."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); master_svg=Path(master_svg)
    inkscape=shutil.which("inkscape"); gs=shutil.which("gs"); qpdf=shutil.which("qpdf"); pdfcpu=shutil.which("pdfcpu")
    if not inkscape: raise RuntimeError("Inkscape is required for vector PDF/proof export")
    editable=output_dir/"artwork_editable.pdf"; press=output_dir/"artwork_press.pdf"; proof=output_dir/"artwork_proof.png"
    commands=[]
    commands.append(_run([inkscape,master_svg,"--export-type=pdf",f"--export-filename={editable}"]))
    if commands[-1]["returncode"]!=0 or not editable.exists(): raise RuntimeError("Inkscape editable PDF export failed")
    if gs:
        commands.append(_run([gs,"-q","-dNOPAUSE","-dBATCH","-sDEVICE=pdfwrite","-dPDFSETTINGS=/prepress","-dEmbedAllFonts=true","-dSubsetFonts=true","-dCompatibilityLevel=1.6",f"-sOutputFile={press}",editable]))
        if commands[-1]["returncode"]!=0 or not press.exists(): raise RuntimeError("Ghostscript press PDF generation failed")
    else:
        shutil.copy2(editable,press)
    commands.append(_run([inkscape,master_svg,f"--export-dpi={proof_dpi}","--export-type=png",f"--export-filename={proof}"]))
    if commands[-1]["returncode"]!=0 or not proof.exists(): raise RuntimeError("Proof generation failed")
    validations=[]
    if gs: validations.append({"tool":"ghostscript",**_run([gs,"-q","-dNOPAUSE","-dBATCH","-sDEVICE=nullpage",press])})
    if qpdf: validations.append({"tool":"qpdf",**_run([qpdf,"--check",press])})
    if pdfcpu: validations.append({"tool":"pdfcpu",**_run([pdfcpu,"validate","-mode","strict",press])})
    svg=svg_preflight(master_svg)
    tool_checks={v["tool"]:v["returncode"]==0 for v in validations}
    available={"inkscape":bool(inkscape),"ghostscript":bool(gs),"qpdf":bool(qpdf),"pdfcpu":bool(pdfcpu)}
    missing_validators=[name for name in ("ghostscript","qpdf","pdfcpu") if not available[name]]
    report={
        "schema":"poster-vector-preflight-v1","passed":svg["passed"] and all(tool_checks.values()),
        "svg":svg,"tool_checks":tool_checks,"tools":available,"missing_optional_validators":missing_validators,
        "commands":commands,"validations":validations,
        "press_pdf_policy":"print-optimized PDF generated with Ghostscript /prepress settings when available; not claimed as certified PDF/X without an explicit ICC output intent.",
        "outputs":{"editable_pdf":str(editable),"press_pdf":str(press),"proof":str(proof),"report":str(output_dir/"preflight_report.json")},
    }
    Path(report["outputs"]["report"]).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
