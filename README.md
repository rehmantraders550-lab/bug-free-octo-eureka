# Poster Vector Rebuilder

A focused raster-to-editable-vector toolkit for reconstructing poster backgrounds and print artwork from photographed references.

## Purpose

This repository is being built for a controlled workflow:

**reference photo → perspective/colour analysis → background decomposition → editable SVG → optional differentiable refinement → prepress handoff**

The goal is not to blindly trace every pixel. It is to infer a compact, human-editable vector construction using gradients, translucent panels, glows and geometric primitives.

## Current capabilities

- Layered SVG background construction
- Linear and radial gradient primitives
- Translucent polygon / panel overlays
- Directional vector texture
- Configuration-driven generation
- Raster colour-field sampling helper
- GFC TurboWash background reconstruction starter
- Confidence-aware background-safe edge maps and long panel-boundary detection
- Measured background SVG, inference mask, error heatmap and ΔE report
- Conservative Tesseract text metadata with confidence and font-status labels
- Restricted-primitive SVG assembly plus Inkscape/Ghostscript vector-PDF proofing

## Upstream tools

The pipeline is designed to interoperate with:

- **diffvg** — `BachiLi/diffvg` — differentiable vector graphics rasterization; Apache-2.0.
- **VTracer** — `visioncortex/vtracer` — raster-to-vector tracing; MIT.
- **JPG-to-SVG** — `Furinaaa-Cancan/JPG-to-SVG` — semantic/vectorization reference pipeline. Its repository-level LICENSE currently contains Meta's SAM license, so it is treated here as an **optional external integration/reference and is not vendored**.

See `THIRD_PARTY.md` before redistribution.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

poster-vector build configs/gfc_background.yaml -o output/gfc_background.svg
poster-vector analyze reference.jpg -o output/analysis.json
poster-vector normalize reference.jpg -o jobs/my-poster
poster-vector segment jobs/my-poster
poster-vector background jobs/my-poster
poster-vector text jobs/my-poster
poster-vector assemble jobs/my-poster
poster-vector prepress jobs/my-poster
```

Open the resulting SVG in CorelDRAW, Adobe Illustrator or Inkscape and edit the component gradients/panels individually.

`prepress` always produces a vector PDF and Ghostscript proof. It only marks
the PDF as production-certified after a trim size, bleed and ICC profile have
been supplied; this prevents an RGB proof from being mislabeled as PDF/X CMYK.

## Repository structure

```text
configs/                         reconstruction recipes
examples/                        ready-to-open SVG examples
src/poster_vector_rebuilder/     Python package
scripts/                         optional upstream installers
output/                          generated files (gitignored)
```

## GFC reconstruction model

The supplied GFC background is deliberately represented as a compact design system rather than a bitmap trace:

1. Base blue gradient field
2. Top-left indigo enrichment
3. Central cyan illumination
4. Bottom-right saturated blue enrichment
5. Large diagonal translucent planes
6. Soft radial highlights
7. Fine diagonal brushed texture

This makes the output scalable and practical for offset/digital print artwork.

## Roadmap

- Perspective rectification from photographed sheets
- Foreground masking
- Automatic gradient anchor fitting
- ΔE / perceptual comparison reports
- diffvg optimization stage
- optional VTracer hard-edge asset extraction
- CMYK/ICC and PDF/X prepress export helpers

## Status

Initial production scaffold. The first reconstruction recipe is `configs/gfc_background.yaml` and the matching editable SVG is in `examples/gfc_background.svg`.
