from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
import json

SVG_NS="http://www.w3.org/2000/svg"
INKSCAPE_NS="http://www.inkscape.org/namespaces/inkscape"
XLINK_NS="http://www.w3.org/1999/xlink"
ET.register_namespace("",SVG_NS); ET.register_namespace("inkscape",INKSCAPE_NS); ET.register_namespace("xlink",XLINK_NS)
LAYERS=("00_BACKGROUND","10_HERO","20_BRAND","30_DECORATION","40_ICONS","90_PREPRESS")


def _local(tag): return tag.rsplit("}",1)[-1]


def _bbox_ratio(elem,w,h):
    tag=_local(elem.tag)
    try:
        if tag=="rect": bw,bh=float(elem.get("width",0)),float(elem.get("height",0))
        elif tag=="circle": r=float(elem.get("r",0)); bw=bh=2*r
        elif tag=="ellipse": bw,bh=2*float(elem.get("rx",0)),2*float(elem.get("ry",0))
        elif tag=="line": bw=abs(float(elem.get("x2",0))-float(elem.get("x1",0))); bh=abs(float(elem.get("y2",0))-float(elem.get("y1",0)))+float(elem.get("stroke-width",1))
        elif tag=="polygon":
            pts=[tuple(map(float,p.split(','))) for p in elem.get("points","").split() if ',' in p]
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; bw=max(xs)-min(xs); bh=max(ys)-min(ys)
        else: return 0.02
        return max(0.0,(bw*bh)/max(w*h,1.0))
    except Exception:
        return 0.02


def _copy_svg(source, target_layer, defs, *, route=False, layers=None, w=1, h=1):
    if not source or not Path(source).exists(): return
    src=ET.parse(source).getroot()
    for child in list(src):
        tag=_local(child.tag)
        if tag=="defs":
            for d in child: defs.append(deepcopy(d))
        elif tag in {"title","desc","metadata"}: continue
        elif route and layers is not None:
            candidates=list(child) if tag=="g" else [child]
            for obj in candidates:
                ratio=_bbox_ratio(obj,w,h)
                dest=layers["10_HERO"] if ratio>=0.08 else layers["40_ICONS"] if ratio<=0.004 else layers["30_DECORATION"]
                dest.append(deepcopy(obj))
        else: target_layer.append(deepcopy(child))


def assemble_master_svg(
    output_svg: str | Path,
    *,
    width: float,
    height: float,
    background_svg: str | Path | None=None,
    semantic_svg: str | Path | None=None,
    text_svg: str | Path | None=None,
    photographic_href: str | None=None,
    report_path: str | Path | None=None,
) -> dict:
    """Assemble the final Corel-friendly master with stable named editing layers."""
    root=ET.Element(f"{{{SVG_NS}}}svg",{"version":"1.1","width":str(width),"height":str(height),"viewBox":f"0 0 {width} {height}"})
    ET.SubElement(root,f"{{{SVG_NS}}}title").text="Editable artwork reconstruction"
    ET.SubElement(root,f"{{{SVG_NS}}}desc").text="Semantic vector reconstruction with confidence-preserving raster fallback for photographic content"
    defs=ET.SubElement(root,f"{{{SVG_NS}}}defs")
    layers={}
    for name in LAYERS:
        layers[name]=ET.SubElement(root,f"{{{SVG_NS}}}g",{"id":name,f"{{{INKSCAPE_NS}}}label":name,f"{{{INKSCAPE_NS}}}groupmode":"layer"})
    _copy_svg(background_svg,layers["00_BACKGROUND"],defs,w=width,h=height)
    _copy_svg(semantic_svg,layers["30_DECORATION"],defs,route=True,layers=layers,w=width,h=height)
    _copy_svg(text_svg,layers["20_BRAND"],defs,w=width,h=height)
    if photographic_href:
        ET.SubElement(layers["10_HERO"],f"{{{SVG_NS}}}image",{
            "id":"photographic_foreground","x":"0","y":"0","width":str(width),"height":str(height),
            "href":photographic_href,f"{{{XLINK_NS}}}href":photographic_href,"preserveAspectRatio":"none",
            "data-vector-status":"raster-photographic-fallback"
        })
    ET.SubElement(layers["90_PREPRESS"],f"{{{SVG_NS}}}rect",{"id":"trim_box","x":"0","y":"0","width":str(width),"height":str(height),"fill":"none","stroke":"none","data-prepress-role":"trim"})
    normalize_corel_tree(root)
    output_svg=Path(output_svg); output_svg.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(output_svg,encoding="utf-8",xml_declaration=True)
    report=corel_compatibility_report(output_svg)
    report_path=Path(report_path) if report_path else output_svg.with_suffix(".assembly.json")
    report.update({
        "schema":"poster-vector-final-assembly-v1",
        "layer_architecture":list(LAYERS),
        "cleanup_policy":{"unsupported_dom_removed":True,"blend_and_filter_styles_removed":True,"semantic_paths_pre_simplified":True},
        "outputs":{"svg":str(output_svg),"report":str(report_path)}
    })
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def normalize_corel_tree(root):
    banned={"foreignObject","flowRoot","flowPara","script"}
    for parent in root.iter():
        for child in list(parent):
            if _local(child.tag) in banned: parent.remove(child)
        style=parent.get("style")
        if style:
            keep=[p for p in style.split(';') if p and not p.strip().startswith(("mix-blend-mode","filter"))]
            if keep: parent.set("style",';'.join(keep))
            else: parent.attrib.pop("style",None)
        parent.attrib.pop("filter",None)
    return root


def corel_compatibility_report(svg_path):
    root=ET.parse(svg_path).getroot(); tags=[_local(e.tag) for e in root.iter()]
    layer_ids={e.get("id") for e in root if _local(e.tag)=="g"}
    unsupported=sorted({t for t in tags if t in {"foreignObject","flowRoot","flowPara","script","filter"}})
    return {
        "corel_compatibility":"pass" if not unsupported and all(x in layer_ids for x in LAYERS) else "fail",
        "required_layers_present":all(x in layer_ids for x in LAYERS), "layer_ids":sorted(x for x in layer_ids if x),
        "unsupported_elements":unsupported, "text_count":tags.count("text"), "raster_image_count":tags.count("image"),
        "vector_primitive_count":sum(tags.count(t) for t in ("rect","circle","ellipse","line","polygon","polyline","path")),
        "node_count":len(tags), "has_viewbox":bool(root.get("viewBox")),
    }
