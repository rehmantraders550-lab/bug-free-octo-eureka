from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
import json, math

import cv2
import numpy as np
from PIL import Image, ImageOps

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)


@dataclass(frozen=True)
class PrimitiveConfig:
    colors: int = 12
    min_area: float = 10.0
    simplify: float = 0.003
    cleanup_radius: int = 0
    circle_aspect_tolerance: float = 0.12
    circle_radial_error: float = 0.08
    ellipse_radial_error: float = 0.10
    rounded_rect_min_fill_ratio: float = 0.72


@dataclass
class PrimitiveObject:
    object_id: str
    primitive: str
    confidence: float
    fill: str | None
    stroke: str | None
    stroke_width: float | None
    source_area: float
    bbox: list[float]
    geometry: dict[str, Any]
    holes: int = 0
    notes: list[str] | None = None


def _hex(color) -> str:
    r, g, b = [int(v) for v in color]
    return f"#{r:02x}{g:02x}{b:02x}"


def _read_rgba(path: str | Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGBA"), dtype=np.uint8)


def _read_mask(path, size, alpha):
    h, w = size
    valid = alpha > 0
    if path is None:
        return valid
    mask = Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    return valid & (np.asarray(mask, dtype=np.uint8) >= 128)


def _quantize(rgb, valid, colors):
    pixels = rgb[valid]
    if pixels.size == 0:
        raise ValueError("Semantic reconstruction mask contains no pixels")
    unique = np.unique(pixels.reshape(-1, 3), axis=0)
    k = max(1, min(int(colors), len(unique)))
    if len(unique) <= k:
        palette = unique.astype(np.uint8)
        lut = {tuple(c.tolist()): i for i, c in enumerate(palette)}
        labels = np.full(valid.shape, -1, dtype=np.int32)
        labels[valid] = np.asarray([lut[tuple(c.tolist())] for c in pixels], dtype=np.int32)
        return labels, palette
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.25)
    _, lab, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = np.full(valid.shape, -1, dtype=np.int32)
    labels[valid] = lab.reshape(-1)
    return labels, np.clip(np.rint(centers), 0, 255).astype(np.uint8)


def _cleanup(mask, radius):
    if radius <= 0:
        return mask
    r = int(radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def _approx(contour, simplify):
    eps = max(0.25, float(simplify) * max(cv2.arcLength(contour, True), 1.0))
    out = cv2.approxPolyDP(contour, eps, True)
    return contour if len(out) < 3 else out


def _right_angle_score(pts):
    p = pts.reshape(-1, 2).astype(np.float64)
    if len(p) != 4:
        return 0.0
    scores = []
    for i in range(4):
        a, b = p[(i-1)%4]-p[i], p[(i+1)%4]-p[i]
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den < 1e-6:
            return 0.0
        scores.append(max(0.0, 1.0 - abs(float(np.dot(a,b)/den))/0.25))
    return float(np.mean(scores))


def _circle_error(contour, center, radius):
    if radius <= 1e-6:
        return 1.0
    pts = contour.reshape(-1, 2).astype(np.float64)
    d = np.linalg.norm(pts - np.asarray(center), axis=1)
    return float(np.mean(np.abs(d-radius))/radius)


def _ellipse_error(contour, fit):
    (cx,cy),(d1,d2),angle = fit
    a,b = max(d1,d2)/2.0, min(d1,d2)/2.0
    if a <= 1e-6 or b <= 1e-6:
        return 1.0
    theta = math.radians(angle if d1 >= d2 else angle+90.0)
    c,s = math.cos(theta), math.sin(theta)
    pts = contour.reshape(-1,2).astype(np.float64)
    dx,dy = pts[:,0]-cx, pts[:,1]-cy
    xp,yp = c*dx+s*dy, -s*dx+c*dy
    rho = np.sqrt((xp/a)**2 + (yp/b)**2)
    return float(np.mean(np.abs(rho-1.0)))


def _rounded_radius(mask, bbox):
    x,y,w,h = bbox
    roi = mask[y:y+h, x:x+w] > 0
    if w < 6 or h < 6:
        return 0.0, 0.0
    vals=[]
    for arr, total in ((np.flatnonzero(roi[0]),w),(np.flatnonzero(roi[-1]),w),(np.flatnonzero(roi[:,0]),h),(np.flatnonzero(roi[:,-1]),h)):
        if arr.size:
            vals.extend([float(arr[0]), float((total-1)-arr[-1])])
    if not vals:
        return 0.0,0.0
    r=min(float(np.median(vals)), min(w,h)/2.0)
    rr=max(1,int(round(r)))
    ideal=np.zeros((h,w),np.uint8)
    cv2.rectangle(ideal,(rr,0),(w-rr-1,h-1),255,-1)
    cv2.rectangle(ideal,(0,rr),(w-1,h-rr-1),255,-1)
    for cx,cy in ((rr,rr),(w-rr-1,rr),(rr,h-rr-1),(w-rr-1,h-rr-1)):
        cv2.circle(ideal,(cx,cy),rr,255,-1)
    inter=np.count_nonzero((ideal>0)&roi); union=np.count_nonzero((ideal>0)|roi)
    return r, float(inter/union) if union else 0.0


def _line(contour):
    (cx,cy),(rw,rh),angle=cv2.minAreaRect(contour)
    length,thick=max(float(rw),float(rh)),min(float(rw),float(rh))
    if length <= 1e-6 or thick <= 0 or thick/length > 0.12:
        return None
    theta=math.radians(angle + (90.0 if rh>rw else 0.0))
    dx,dy=math.cos(theta)*length/2.0, math.sin(theta)*length/2.0
    return {"x1":cx-dx,"y1":cy-dy,"x2":cx+dx,"y2":cy+dy,"stroke_width":max(1.0,thick)}, max(0.0,min(1.0,1.0-(thick/length)/0.12))


def _linear_path(contours):
    parts=[]
    for contour in contours:
        pts=contour.reshape(-1,2)
        if len(pts)<3: continue
        parts.append(f"M {pts[0,0]:.3f} {pts[0,1]:.3f}")
        parts.extend(f"L {x:.3f} {y:.3f}" for x,y in pts[1:])
        parts.append("Z")
    return " ".join(parts)


def _bezier_path(contour):
    pts=contour.reshape(-1,2).astype(np.float64)
    if len(pts)<4:
        return _linear_path([contour])
    parts=[f"M {pts[0,0]:.3f} {pts[0,1]:.3f}"]
    n=len(pts)
    for i in range(n):
        p0,p1,p2,p3=pts[(i-1)%n],pts[i],pts[(i+1)%n],pts[(i+2)%n]
        c1=p1+(p2-p0)/6.0; c2=p2-(p3-p1)/6.0
        parts.append(f"C {c1[0]:.3f} {c1[1]:.3f} {c2[0]:.3f} {c2[1]:.3f} {p2[0]:.3f} {p2[1]:.3f}")
    parts.append("Z")
    return " ".join(parts)


def classify_primitive(contour, *, component_mask, child_contours=None, config=None):
    cfg=config or PrimitiveConfig(); children=child_contours or []
    area=abs(float(cv2.contourArea(contour))); per=max(float(cv2.arcLength(contour,True)),1.0)
    x,y,w,h=cv2.boundingRect(contour); notes=[]
    if children:
        outer=_approx(contour,cfg.simplify); holes=[_approx(c,cfg.simplify) for c in children]
        return "compound",0.98,{"d":_linear_path([outer,*holes]),"fill_rule":"evenodd"},[f"compound shape with {len(children)} hole(s)"]
    line=_line(contour)
    if line is not None and line[1] >= 0.35:
        return "line",line[1],line[0],notes
    approx=_approx(contour,cfg.simplify); rect=cv2.minAreaRect(contour)
    rect_area=max(float(rect[1][0]*rect[1][1]),1.0); rect_fill=min(1.0,area/rect_area)
    if len(approx)==4 and cv2.isContourConvex(approx):
        score=_right_angle_score(approx)
        if score>=0.55 and rect_fill>=0.82:
            (cx,cy),(rw,rh),angle=rect
            if rw<rh: rw,rh,angle=rh,rw,angle+90.0
            return "rectangle",min(0.995,0.58+0.25*score+0.17*rect_fill),{"x":cx-rw/2,"y":cy-rh/2,"width":rw,"height":rh,"rx":0.0,"ry":0.0,"rotation":angle,"cx":cx,"cy":cy},notes
    circularity=float(4.0*math.pi*area/(per*per)); (ccx,ccy),radius=cv2.minEnclosingCircle(contour)
    aspect=float(w/h) if h else 999.0; cerr=_circle_error(contour,(ccx,ccy),radius)
    if 1.0-cfg.circle_aspect_tolerance<=aspect<=1.0+cfg.circle_aspect_tolerance and circularity>=0.80 and cerr<=cfg.circle_radial_error:
        conf=min(0.995,0.50+0.30*min(1.0,circularity)+0.20*max(0.0,1.0-cerr/cfg.circle_radial_error))
        return "circle",conf,{"cx":float(ccx),"cy":float(ccy),"r":float(radius)},notes
    bbox_fill=area/max(float(w*h),1.0)
    if bbox_fill>=cfg.rounded_rect_min_fill_ratio and len(approx)>=6:
        r,iou=_rounded_radius(component_mask,(x,y,w,h))
        if r>=1.0 and iou>=0.88:
            return "rounded_rectangle",min(0.99,0.45+0.45*iou+0.10*min(1.0,bbox_fill)),{"x":float(x),"y":float(y),"width":float(w),"height":float(h),"rx":r,"ry":r,"rotation":0.0,"cx":x+w/2.0,"cy":y+h/2.0},notes
    if len(contour)>=5:
        fit=cv2.fitEllipse(contour); (ecx,ecy),(d1,d2),angle=fit
        a,b=max(d1,d2)/2.0,min(d1,d2)/2.0
        if a>1.0 and b>1.0:
            err=_ellipse_error(contour,fit); ideal=math.pi*a*b; ratio=min(area,ideal)/max(area,ideal)
            if err<=cfg.ellipse_radial_error and ratio>=0.82:
                rot=float(angle if d1>=d2 else angle+90.0)
                return "ellipse",min(0.99,0.45+0.30*ratio+0.25*max(0.0,1.0-err/cfg.ellipse_radial_error)),{"cx":float(ecx),"cy":float(ecy),"rx":float(a),"ry":float(b),"rotation":rot},notes
    if 3<=len(approx)<=12:
        pts=[[float(x),float(y)] for x,y in approx.reshape(-1,2)]
        aa=abs(float(cv2.contourArea(approx))); ratio=min(area,aa)/max(area,aa,1.0)
        return "polygon",min(0.96,0.58+0.38*ratio),{"points":pts},notes
    return "path",0.72,{"d":_bezier_path(approx),"fill_rule":"nonzero"},["irregular geometry retained as simplified cubic Bezier path"]


def _emit(parent,obj):
    common={"id":obj.object_id,"data-primitive":obj.primitive,"data-confidence":f"{obj.confidence:.3f}"}; g=obj.geometry
    if obj.primitive in {"rectangle","rounded_rectangle"}:
        attrs={**common,"x":f"{g['x']:.3f}","y":f"{g['y']:.3f}","width":f"{g['width']:.3f}","height":f"{g['height']:.3f}","fill":obj.fill or "none","stroke":obj.stroke or "none"}
        if g.get("rx",0)>0: attrs.update({"rx":f"{g['rx']:.3f}","ry":f"{g.get('ry',g['rx']):.3f}"})
        if abs(float(g.get("rotation",0)))>1e-3: attrs["transform"]=f"rotate({g['rotation']:.6f} {g['cx']:.6f} {g['cy']:.6f})"
        return ET.SubElement(parent,f"{{{SVG_NS}}}rect",attrs)
    if obj.primitive=="circle":
        return ET.SubElement(parent,f"{{{SVG_NS}}}circle",{**common,"cx":f"{g['cx']:.3f}","cy":f"{g['cy']:.3f}","r":f"{g['r']:.3f}","fill":obj.fill or "none","stroke":obj.stroke or "none"})
    if obj.primitive=="ellipse":
        attrs={**common,"cx":f"{g['cx']:.3f}","cy":f"{g['cy']:.3f}","rx":f"{g['rx']:.3f}","ry":f"{g['ry']:.3f}","fill":obj.fill or "none","stroke":obj.stroke or "none"}
        if abs(float(g.get("rotation",0)))>1e-3: attrs["transform"]=f"rotate({g['rotation']:.6f} {g['cx']:.6f} {g['cy']:.6f})"
        return ET.SubElement(parent,f"{{{SVG_NS}}}ellipse",attrs)
    if obj.primitive=="line":
        return ET.SubElement(parent,f"{{{SVG_NS}}}line",{**common,"x1":f"{g['x1']:.3f}","y1":f"{g['y1']:.3f}","x2":f"{g['x2']:.3f}","y2":f"{g['y2']:.3f}","fill":"none","stroke":obj.stroke or obj.fill or "#000000","stroke-width":f"{(obj.stroke_width or g.get('stroke_width',1.0)):.3f}"})
    if obj.primitive=="polygon":
        pts=" ".join(f"{x:.3f},{y:.3f}" for x,y in g["points"])
        return ET.SubElement(parent,f"{{{SVG_NS}}}polygon",{**common,"points":pts,"fill":obj.fill or "none","stroke":obj.stroke or "none"})
    return ET.SubElement(parent,f"{{{SVG_NS}}}path",{**common,"d":g["d"],"fill":obj.fill or "none","stroke":obj.stroke or "none","fill-rule":g.get("fill_rule","nonzero")})


def reconstruct_semantic_primitives(image_path, output_svg, *, mask_path=None, report_path=None, colors=12, min_area=10.0, simplify=0.003, cleanup_radius=0):
    if colors<1 or min_area<0 or simplify<0:
        raise ValueError("Invalid semantic reconstruction thresholds")
    rgba=_read_rgba(image_path); h,w=rgba.shape[:2]; valid=_read_mask(mask_path,(h,w),rgba[...,3])
    labels,palette=_quantize(rgba[...,:3],valid,colors); cfg=PrimitiveConfig(colors=colors,min_area=min_area,simplify=simplify,cleanup_radius=cleanup_radius)
    root=ET.Element(f"{{{SVG_NS}}}svg",{"version":"1.1","width":str(w),"height":str(h),"viewBox":f"0 0 {w} {h}"})
    layer=ET.SubElement(root,f"{{{SVG_NS}}}g",{"id":"SEMANTIC_OBJECTS",f"{{{INKSCAPE_NS}}}label":"SEMANTIC_OBJECTS",f"{{{INKSCAPE_NS}}}groupmode":"layer"})
    objects=[]; counts={}; no=0
    order=sorted(range(len(palette)),key=lambda i:int(np.count_nonzero(labels==i)),reverse=True)
    for color_idx in order:
        mask=_cleanup(np.where(labels==color_idx,255,0).astype(np.uint8),cleanup_radius)
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if hierarchy is None: continue
        hierarchy=hierarchy[0]
        for i,contour in enumerate(contours):
            if hierarchy[i][3]>=0: continue
            area=abs(float(cv2.contourArea(contour)))
            if area<min_area: continue
            children=[]; child=int(hierarchy[i][2])
            while child>=0:
                if abs(float(cv2.contourArea(contours[child])))>=min_area: children.append(contours[child])
                child=int(hierarchy[child][0])
            component=np.zeros((h,w),np.uint8); cv2.drawContours(component,[contour],-1,255,-1)
            for c in children: cv2.drawContours(component,[c],-1,0,-1)
            primitive,confidence,geometry,notes=classify_primitive(contour,component_mask=component,child_contours=children,config=cfg)
            no+=1; counts[primitive]=counts.get(primitive,0)+1; x,y,bw,bh=cv2.boundingRect(contour)
            fill=_hex(palette[color_idx]); stroke=None; sw=None
            if primitive=="line": stroke,fill,sw=fill,None,float(geometry.get("stroke_width",1.0))
            obj=PrimitiveObject(f"obj_{no:04d}_{primitive}",primitive,float(confidence),fill,stroke,sw,area,[float(x),float(y),float(bw),float(bh)],geometry,len(children),notes or [])
            _emit(layer,obj); objects.append(obj)
    if not objects: raise ValueError("No semantic objects survived the current thresholds")
    output_svg=Path(output_svg); output_svg.parent.mkdir(parents=True,exist_ok=True); ET.ElementTree(root).write(output_svg,encoding="utf-8",xml_declaration=True)
    report_path=Path(report_path) if report_path else output_svg.with_suffix(".semantic.json")
    report={"schema":"poster-vector-semantic-primitives-v1","source_size":{"width":w,"height":h},"mask_pixels":int(np.count_nonzero(valid)),"object_count":len(objects),"primitive_counts":counts,"semantic_object_ratio":float(sum(o.primitive not in {"path","compound"} for o in objects)/len(objects)),"generic_path_ratio":float(sum(o.primitive=="path" for o in objects)/len(objects)),"compound_ratio":float(sum(o.primitive=="compound" for o in objects)/len(objects)),"objects":[asdict(o) for o in objects],"rules":{"semantic_svg_preferred":True,"irregular_geometry_fallback":"path","compound_geometry_fallback":"path-evenodd","photographic_content_claimed_as_vector":False},"outputs":{"svg":str(output_svg),"report":str(report_path)}}
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
