from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from .svg_builder import save_svg
from .analyze import save_analysis
from .normalize import normalize_reference
from .segment import segment_reference
from .background import fit_background
from .typography import recover_text
from .assemble import assemble_artwork
from .prepress import preflight_pdf


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_corners(value: str | None):
    if not value:
        return None
    points = []
    for pair in value.split(";"):
        xy = pair.split(",")
        if len(xy) != 2:
            raise argparse.ArgumentTypeError("Corners must be x,y;x,y;x,y;x,y")
        points.append([float(xy[0]), float(xy[1])])
    if len(points) != 4:
        raise argparse.ArgumentTypeError("Exactly four corner points are required")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(prog="poster-vector")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build an editable SVG from a reconstruction YAML")
    build.add_argument("config")
    build.add_argument("-o", "--output", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a raster reference's low-frequency colour field")
    analyze.add_argument("image")
    analyze.add_argument("-o", "--output", required=True)

    normalize = sub.add_parser("normalize", help="Preserve and geometrically normalize a photographed reference")
    normalize.add_argument("image")
    normalize.add_argument("-o", "--output", required=True, help="Job directory")
    normalize.add_argument(
        "--rotation",
        choices=["auto", "keep", "90cw", "90ccw", "180"],
        default="auto",
        help="Rotation applied after perspective rectification",
    )
    normalize.add_argument(
        "--corners",
        default=None,
        help="Optional deterministic corner override: x,y;x,y;x,y;x,y. Any order is accepted.",
    )

    segment = sub.add_parser(
        "segment",
        help="Create foreground exclusion masks and authoritative background-confidence masks",
    )
    segment.add_argument("job_dir", help="Existing normalized job directory")
    segment.add_argument(
        "--image",
        default=None,
        help="Optional normalized image override; defaults to JOB/work/normalized_reference.png",
    )

    background = sub.add_parser("background", help="Detect long panel boundaries and build a constrained editable background SVG")
    background.add_argument("job_dir", help="Job containing work/normalized_reference.png and masks/background_known.png")
    background.add_argument("--image", default=None, help="Optional normalized image override")
    background.add_argument("--known-mask", default=None, help="Optional authoritative background mask override")

    text = sub.add_parser("text", help="Recover OCR text as explicitly confidence-labelled editable text metadata")
    text.add_argument("job_dir")
    text.add_argument("--image", default=None)

    assemble = sub.add_parser("assemble", help="Assemble restricted-primitive editable SVG layers")
    assemble.add_argument("job_dir")

    prepress = sub.add_parser("prepress", help="Export a vector PDF, Ghostscript proof and preflight report")
    prepress.add_argument("job_dir")
    prepress.add_argument("--trim-mm", default=None, help="Trim width,height in mm; required to certify production sizing")
    prepress.add_argument("--bleed-mm", type=float, default=None)
    prepress.add_argument("--icc-profile", default=None, help="ICC profile required for CMYK/PDF-X certification")

    rebuild = sub.add_parser("rebuild", help="Run normalize, segment, constrained background, OCR, assembly and PDF preflight")
    rebuild.add_argument("image")
    rebuild.add_argument("-o", "--output", required=True, help="Job directory")
    rebuild.add_argument("--rotation", choices=["auto", "keep", "90cw", "90ccw", "180"], default="auto")
    rebuild.add_argument("--corners", default=None)
    rebuild.add_argument("--trim-mm", default=None)
    rebuild.add_argument("--bleed-mm", type=float, default=None)
    rebuild.add_argument("--icc-profile", default=None)
    segment.add_argument(
        "--mode",
        choices=["precision", "detail"],
        default="precision",
        help="Precision is conservative and recommended for background fitting",
    )
    segment.add_argument(
        "--birefnet-model",
        default=None,
        help="Optional local/HuggingFace BiRefNet model source",
    )
    segment.add_argument(
        "--sam2-model",
        default=None,
        help="Optional SAM2 HuggingFace model identifier",
    )
    segment.add_argument("--sam2-config", default=None, help="Optional local SAM2 model config")
    segment.add_argument("--sam2-checkpoint", default=None, help="Optional local SAM2 checkpoint")
    segment.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu")
    segment.add_argument(
        "--manual-foreground-mask",
        default=None,
        help="Optional binary mask to force additional pixels into foreground exclusion",
    )

    args = parser.parse_args()

    if args.command == "build":
        out = save_svg(_load_yaml(args.config), args.output)
        print(out)
    elif args.command == "analyze":
        out = save_analysis(args.image, args.output)
        print(out)
    elif args.command == "normalize":
        result = normalize_reference(
            args.image,
            args.output,
            rotation=args.rotation,
            corners=_parse_corners(args.corners),
        )
        print(Path(args.output) / result["normalized_path"])
    elif args.command == "segment":
        result = segment_reference(
            args.job_dir,
            image_path=args.image,
            mode=args.mode,
            birefnet_model=args.birefnet_model,
            sam2_model=args.sam2_model,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            device=args.device,
            manual_foreground_mask=args.manual_foreground_mask,
        )
        print(Path(args.job_dir) / result["outputs"]["background_known"])
    elif args.command == "background":
        result = fit_background(args.job_dir, image_path=args.image, known_mask_path=args.known_mask)
        print(Path(args.job_dir) / result["outputs"]["svg"])
    elif args.command == "text":
        job = Path(args.job_dir)
        result = recover_text(args.image or job / "work" / "normalized_reference.png", job / "analysis" / "text_layers.json")
        print(Path(args.job_dir) / "analysis" / "text_layers.json")
    elif args.command == "assemble":
        result = assemble_artwork(args.job_dir)
        print(Path(args.job_dir) / result["master_svg"])
    elif args.command == "prepress":
        trim = tuple(map(float, args.trim_mm.split(","))) if args.trim_mm else None
        if trim and len(trim) != 2:
            raise argparse.ArgumentTypeError("--trim-mm must be width,height")
        result = preflight_pdf(args.job_dir, trim_mm=trim, bleed_mm=args.bleed_mm, icc_profile=args.icc_profile)
        print(Path(args.job_dir) / result["pdf"])
    elif args.command == "rebuild":
        job = Path(args.output)
        normalize_reference(args.image, job, rotation=args.rotation, corners=_parse_corners(args.corners))
        segment_reference(job)
        fit_background(job)
        recover_text(job / "work" / "normalized_reference.png", job / "analysis" / "text_layers.json")
        assemble_artwork(job)
        trim = tuple(map(float, args.trim_mm.split(","))) if args.trim_mm else None
        result = preflight_pdf(job, trim_mm=trim, bleed_mm=args.bleed_mm, icc_profile=args.icc_profile)
        print(job / result["pdf"])


if __name__ == "__main__":
    main()
