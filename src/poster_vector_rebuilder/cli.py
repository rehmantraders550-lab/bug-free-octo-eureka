from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from .svg_builder import save_svg
from .analyze import save_analysis


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(prog="poster-vector")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build an editable SVG from a reconstruction YAML")
    build.add_argument("config")
    build.add_argument("-o", "--output", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a raster reference's low-frequency colour field")
    analyze.add_argument("image")
    analyze.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    if args.command == "build":
        out = save_svg(_load_yaml(args.config), args.output)
        print(out)
    elif args.command == "analyze":
        out = save_analysis(args.image, args.output)
        print(out)


if __name__ == "__main__":
    main()
