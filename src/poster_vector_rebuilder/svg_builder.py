from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import html
import math
import xml.etree.ElementTree as ET


def _esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def _attrs(**kwargs) -> str:
    out = []
    for key, value in kwargs.items():
        if value is None:
            continue
        key = key.replace("_", "-")
        out.append(f'{key}="{_esc(value)}"')
    return " ".join(out)


def _gradient_stops(stops: Iterable[dict]) -> str:
    rows = []
    for stop in stops:
        rows.append(
            f'<stop offset="{stop["offset"]}" stop-color="{_esc(stop["color"])}" '
            f'stop-opacity="{stop.get("opacity", 1)}"/>'
        )
    return "".join(rows)


def build_svg(config: dict) -> str:
    canvas = config["canvas"]
    width = canvas["width"]
    height = canvas["height"]
    radius = canvas.get("radius", 0)

    defs = []
    body = []

    for g in config.get("linear_gradients", []):
        defs.append(
            f'<linearGradient id="{_esc(g["id"])}" gradientUnits="userSpaceOnUse" '
            f'x1="{g["x1"]}" y1="{g["y1"]}" x2="{g["x2"]}" y2="{g["y2"]}">'
            f'{_gradient_stops(g["stops"])}</linearGradient>'
        )

    for g in config.get("radial_gradients", []):
        defs.append(
            f'<radialGradient id="{_esc(g["id"])}" gradientUnits="userSpaceOnUse" '
            f'cx="{g["cx"]}" cy="{g["cy"]}" r="{g["r"]}">'
            f'{_gradient_stops(g["stops"])}</radialGradient>'
        )

    clip_id = "posterClip"
    defs.append(
        f'<clipPath id="{clip_id}"><rect x="0" y="0" width="{width}" height="{height}" rx="{radius}"/></clipPath>'
    )

    texture = config.get("texture")
    if texture:
        spacing = texture.get("spacing", 18)
        angle = texture.get("angle", 18)
        opacity = texture.get("opacity", 0.08)
        stroke = texture.get("color", "#FFFFFF")
        stroke_width = texture.get("stroke_width", 2)
        pattern_id = "brushTexture"
        defs.append(
            f'<pattern id="{pattern_id}" patternUnits="userSpaceOnUse" width="{spacing}" height="{spacing}" '
            f'patternTransform="rotate({angle})">'
            f'<line x1="0" y1="0" x2="0" y2="{spacing}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"/>'
            f'</pattern>'
        )

    base = config["base"]
    body.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="{radius}" fill="url(#{_esc(base["gradient"])})"/>'
    )

    for layer in config.get("layers", []):
        kind = layer["type"]
        opacity = layer.get("opacity", 1)
        blend = layer.get("blend")
        style = f'mix-blend-mode:{blend};' if blend else None
        if kind == "polygon":
            points = " ".join(f'{x},{y}' for x, y in layer["points"])
            fill = layer.get("fill", "none")
            body.append(
                f'<polygon points="{points}" fill="{_esc(fill)}" opacity="{opacity}"'
                + (f' style="{style}"' if style else "")
                + '/>'
            )
        elif kind == "rect":
            body.append(
                f'<rect x="{layer["x"]}" y="{layer["y"]}" width="{layer["width"]}" height="{layer["height"]}" '
                f'fill="{_esc(layer.get("fill", "none"))}" opacity="{opacity}"'
                + (f' style="{style}"' if style else "")
                + '/>'
            )
        elif kind == "ellipse":
            body.append(
                f'<ellipse cx="{layer["cx"]}" cy="{layer["cy"]}" rx="{layer["rx"]}" ry="{layer["ry"]}" '
                f'fill="{_esc(layer.get("fill", "none"))}" opacity="{opacity}"'
                + (f' style="{style}"' if style else "")
                + '/>'
            )
        else:
            raise ValueError(f"Unsupported layer type: {kind}")

    if texture:
        body.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#{pattern_id})" opacity="1"/>'
        )

    metadata = config.get("metadata", {})
    title = _esc(metadata.get("title", "Poster Vector Reconstruction"))
    desc = _esc(metadata.get("description", "Editable vector reconstruction"))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<title>{title}</title><desc>{desc}</desc>'
        f'<defs>{"".join(defs)}</defs>'
        f'<g clip-path="url(#{clip_id})">{"".join(body)}</g>'
        f'</svg>'
    )


def save_svg(config: dict, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_svg(config), encoding="utf-8")
    return output
