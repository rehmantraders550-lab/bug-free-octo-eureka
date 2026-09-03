from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from .svg_builder import save_svg
from .analyze import save_analysis
from .normalize import normalize_reference
from .segment import segment_reference


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
        choices=["keep", "90cw", "90ccw", "180"],
        default="keep",
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


if __name__ == "__main__":
    main()
