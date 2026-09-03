# Poster Vector Rebuilder / Image-to-Vector File Engine

A general-purpose raster-to-editable-artwork pipeline for reconstructing print graphics from arbitrary reference images.

The engine does **not** blindly trace every pixel. It prefers compact semantic geometry, editable text, measured vector backgrounds, and explicit raster fallback only where the source is genuinely photographic or cannot be represented honestly as vector geometry.

## One-command workflow

```bash
poster-vector deliver reference.png -o output/job
```

The command runs the full pipeline:

1. raster intake and geometric normalization
2. generic artwork classification/routing
3. foreground/background separation
4. automatic region/panel detection
5. constrained editable background fitting and hidden-area continuation
6. semantic primitive/object reconstruction
7. object-level fill/stroke recovery and path simplification
8. confidence-gated OCR to editable SVG text
9. named layer assembly and CorelDRAW normalization
10. editable vector PDF and print-optimized press PDF export
11. proof generation and preflight validation

## Delivery package

The final `delivery/` directory contains:

```text
artwork_master.svg
artwork_editable.pdf
artwork_press.pdf
artwork_proof.png
preflight_report.json
reconstruction_report.json
assets/                         optional raster photographic fallback assets
```

The master SVG uses stable named editing groups:

- `00_BACKGROUND`
- `10_HERO`
- `20_BRAND`
- `30_DECORATION`
- `40_ICONS`
- `90_PREPRESS`

## Semantic SVG policy

Recognizable geometry is emitted as native SVG objects wherever reliable:

- rectangle / rounded rectangle → `<rect>`
- circle → `<circle>`
- ellipse → `<ellipse>`
- line → `<line>`
- polygon → `<polygon>`
- irregular smooth geometry → compact cubic `<path>`
- compound shapes / holes → even-odd `<path>`

This keeps node counts practical and individual elements editable in CorelDRAW, Illustrator, and Inkscape.

## Text policy

Tesseract OCR is used when available. Text above the configured confidence threshold is reconstructed as editable SVG `<text>` and excluded from subsequent foreground vectorization to prevent duplicate traced lettering.

The engine does **not** guess an exact typeface. If the font family cannot be proven, the report records `font_exact_match: false` and uses a generic editable fallback.

## Photographic content policy

Photographic foreground content is never falsely labeled as vector geometry. When classification or semantic reconstruction requires a raster fallback, it is retained as an explicitly marked linked PNG asset inside the editable layer architecture. Reliable OCR text is removed from that raster alpha and retained as editable text.

## Prepress and PDF tools

The delivery workflow uses:

- **Inkscape** — SVG rendering and editable vector PDF export
- **Ghostscript** — press-oriented PDF generation and PDF validation
- **qpdf** — PDF structural validation when installed
- **pdfcpu** — strict PDF validation when installed
- **Tesseract OCR** — editable text recovery when installed

`artwork_press.pdf` is print-optimized, but the engine does **not** claim certified PDF/X compliance unless an explicit ICC output intent is supplied and validated in a future color-management stage.

## Existing specialist commands

The staged commands remain available for engineering and diagnostics:

```bash
poster-vector prepare reference.png -o output/job
poster-vector semantic-vectorize reference.png -o output/semantic.svg --mask foreground.png
poster-vector hard-vectorize reference.png -o output/hard.svg
poster-vector detect-panels output/job
poster-vector fit-background reference.png --background-known background_known.png -o output/background
poster-vector recover-background reference.png --background-known background_known.png --phase24c-report phase24c_report.json -o output/recovery
poster-vector accept-background reference.png --background-known background_known.png --phase24c-report phase24c_report.json --svg background_fitted.svg -o output/gate
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

For the complete delivery pipeline, install Inkscape and the desired preflight/OCR tools listed above on `PATH`.

## Engineering guarantees

- visible source pixels remain authoritative
- hidden background continuation is explicitly lower confidence
- semantic SVG objects are preferred to generic tracing
- photographic raster content is never claimed as true vector geometry
- exact fonts are never guessed
- SVG node counts are preflighted for practical editability
- Python 3.10 and 3.12 are tested remotely
- Inkscape/Ghostscript end-to-end SVG/PDF export is tested remotely

Legacy example configs remain in the repository only as fixtures and regression references; the production pipeline is not optimized around any single brand, poster, or artwork family.
