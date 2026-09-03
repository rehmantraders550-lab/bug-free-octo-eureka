from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from .svg_builder import save_svg
from .analyze import save_analysis
from .normalize import normalize_reference
from .segment import segment_reference
from .hard_vectorize import vectorize_hard_graphic
from .semantic_primitives import reconstruct_semantic_primitives
from .panel_detect import run_phase24b
from .vector_fit import fit_background_vectors
from .phase24d import recover_hidden_background, run_phase24_acceptance_gate
from .generalized_preflight import run_blocks_1_to_4
from .orchestrator import run_delivery_pipeline


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_corners(value: str | None):
    if not value: return None
    points=[]
    for pair in value.split(";"):
        xy=pair.split(",")
        if len(xy)!=2: raise argparse.ArgumentTypeError("Corners must be x,y;x,y;x,y;x,y")
        points.append([float(xy[0]),float(xy[1])])
    if len(points)!=4: raise argparse.ArgumentTypeError("Exactly four corner points are required")
    return points


def main() -> None:
    parser=argparse.ArgumentParser(prog="poster-vector")
    sub=parser.add_subparsers(dest="command",required=True)

    p=sub.add_parser("prepare",help="Run generalized raster intake, artwork classification, foreground/background separation and panel detection")
    p.add_argument("image"); p.add_argument("-o","--output",required=True,help="Job directory"); p.add_argument("--max-panels",type=int,default=4)

    p=sub.add_parser("build",help="Build an editable SVG from a reconstruction YAML")
    p.add_argument("config"); p.add_argument("-o","--output",required=True)

    p=sub.add_parser("analyze",help="Analyze a raster reference's low-frequency colour field")
    p.add_argument("image"); p.add_argument("-o","--output",required=True)

    p=sub.add_parser("normalize",help="Preserve and geometrically normalize a photographed reference")
    p.add_argument("image"); p.add_argument("-o","--output",required=True,help="Job directory"); p.add_argument("--rotation",choices=["keep","90cw","90ccw","180"],default="keep"); p.add_argument("--corners",default=None,help="Optional deterministic corner override: x,y;x,y;x,y;x,y")

    p=sub.add_parser("segment",help="Create foreground exclusion and authoritative background masks")
    p.add_argument("job_dir"); p.add_argument("--image",default=None); p.add_argument("--mode",choices=["precision","detail"],default="precision"); p.add_argument("--birefnet-model",default=None); p.add_argument("--sam2-model",default=None); p.add_argument("--sam2-config",default=None); p.add_argument("--sam2-checkpoint",default=None); p.add_argument("--device",default=None); p.add_argument("--manual-foreground-mask",default=None)

    p=sub.add_parser("detect-panels",help="Phase 2.4B: detect and optimize background panel geometry")
    p.add_argument("job_dir"); p.add_argument("--image",default=None); p.add_argument("--background-known",default=None); p.add_argument("--output-dir",default=None); p.add_argument("--max-panels",type=int,default=3)

    p=sub.add_parser("fit-background",help="Phase 2.4C: fit editable vector gradients/panels to authoritative pixels")
    p.add_argument("image"); p.add_argument("--background-known",required=True); p.add_argument("-o","--output-dir",required=True); p.add_argument("--phase24b-report",default=None); p.add_argument("--max-panels",type=int,default=3); p.add_argument("--complexity-penalty",type=float,default=0.06)

    p=sub.add_parser("recover-background",help="Phase 2.4D: continue fitted vector background through hidden regions")
    p.add_argument("image"); p.add_argument("--background-known",required=True); p.add_argument("--phase24c-report",required=True); p.add_argument("-o","--output-dir",required=True)

    p=sub.add_parser("accept-background",help="Run Phase 2.4 quantitative acceptance gate")
    p.add_argument("image"); p.add_argument("--background-known",required=True); p.add_argument("--phase24c-report",required=True); p.add_argument("--svg",required=True); p.add_argument("-o","--output-dir",required=True); p.add_argument("--max-mean-delta-e",type=float,default=12.0); p.add_argument("--max-rgb-mae",type=float,default=18.0); p.add_argument("--min-ssim",type=float,default=0.82); p.add_argument("--max-boundary-error",type=float,default=0.035)

    p=sub.add_parser("hard-vectorize",help="Vectorize a hard-edged logo, icon, badge or flat graphic into editable SVG paths")
    p.add_argument("image"); p.add_argument("-o","--output",required=True); p.add_argument("--mask",default=None); p.add_argument("--report",default=None); p.add_argument("--colors",type=int,default=8); p.add_argument("--min-area",type=float,default=6.0); p.add_argument("--simplify",type=float,default=0.0025); p.add_argument("--cleanup-radius",type=int,default=0); p.add_argument("--backend",choices=["auto","opencv","vtracer"],default="auto")

    p=sub.add_parser("semantic-vectorize",help="Recover editable semantic SVG primitives before generic tracing")
    p.add_argument("image"); p.add_argument("-o","--output",required=True); p.add_argument("--mask",default=None); p.add_argument("--report",default=None); p.add_argument("--colors",type=int,default=12); p.add_argument("--min-area",type=float,default=10.0); p.add_argument("--simplify",type=float,default=0.003); p.add_argument("--cleanup-radius",type=int,default=0)

    p=sub.add_parser("deliver",help="One arbitrary reference image to complete editable SVG/PDF/preflight delivery package")
    p.add_argument("image"); p.add_argument("-o","--output",required=True,help="Job directory"); p.add_argument("--max-panels",type=int,default=4); p.add_argument("--ocr-confidence",type=float,default=80.0)

    args=parser.parse_args()
    if args.command=="prepare": print(run_blocks_1_to_4(args.image,args.output,max_panels=args.max_panels)["outputs"]["manifest"])
    elif args.command=="build": print(save_svg(_load_yaml(args.config),args.output))
    elif args.command=="analyze": print(save_analysis(args.image,args.output))
    elif args.command=="normalize": print(Path(args.output)/normalize_reference(args.image,args.output,rotation=args.rotation,corners=_parse_corners(args.corners))["normalized_path"])
    elif args.command=="segment": print(Path(args.job_dir)/segment_reference(args.job_dir,image_path=args.image,mode=args.mode,birefnet_model=args.birefnet_model,sam2_model=args.sam2_model,sam2_config=args.sam2_config,sam2_checkpoint=args.sam2_checkpoint,device=args.device,manual_foreground_mask=args.manual_foreground_mask)["outputs"]["background_known"])
    elif args.command=="detect-panels": print(run_phase24b(args.job_dir,image_path=args.image,background_known_path=args.background_known,output_dir=args.output_dir,max_panels=args.max_panels)["outputs"]["report"])
    elif args.command=="fit-background": print(fit_background_vectors(args.image,args.background_known,args.output_dir,phase24b_report_path=args.phase24b_report,max_panels=args.max_panels,complexity_penalty=args.complexity_penalty)["outputs"]["report"])
    elif args.command=="recover-background": print(recover_hidden_background(args.image,args.background_known,args.phase24c_report,args.output_dir)["outputs"]["report"])
    elif args.command=="accept-background": print(run_phase24_acceptance_gate(args.image,args.background_known,args.phase24c_report,args.svg,args.output_dir,max_mean_delta_e=args.max_mean_delta_e,max_rgb_mae=args.max_rgb_mae,min_ssim=args.min_ssim,max_boundary_error=args.max_boundary_error)["report"])
    elif args.command=="hard-vectorize": print(vectorize_hard_graphic(args.image,args.output,mask_path=args.mask,report_path=args.report,colors=args.colors,min_area=args.min_area,simplify=args.simplify,cleanup_radius=args.cleanup_radius,backend=args.backend)["outputs"]["svg"])
    elif args.command=="semantic-vectorize": print(reconstruct_semantic_primitives(args.image,args.output,mask_path=args.mask,report_path=args.report,colors=args.colors,min_area=args.min_area,simplify=args.simplify,cleanup_radius=args.cleanup_radius)["outputs"]["svg"])
    elif args.command=="deliver": print(run_delivery_pipeline(args.image,args.output,max_panels=args.max_panels,ocr_confidence=args.ocr_confidence)["outputs"]["master_svg"])


if __name__=="__main__": main()
