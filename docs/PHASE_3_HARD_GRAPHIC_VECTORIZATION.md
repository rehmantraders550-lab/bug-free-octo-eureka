# Phase 3 — Hard-Graphic Vectorization

## Scope

This stage reconstructs **hard-edged foreground artwork** such as:

- logos
- icons
- badges
- flat illustrations
- solid-colour symbols
- simple product marks
- geometric decorative assets

It is deliberately **not** used for smooth gradient backgrounds, photographs, or typography.

## Routing rule

```text
segmented foreground object
        ↓
classify as hard graphic
        ↓
apply object mask
        ↓
VTracer when installed
        ↓
OpenCV deterministic fallback
        ↓
editable SVG paths + report
```

The object mask is authoritative: pixels outside the selected mask are never allowed into the trace.

## Backends

### VTracer

Preferred external tracer for difficult multicolour hard graphics. It is invoked as an optional subprocess adapter and is not vendored into this repository.

### OpenCV fallback

Always-available deterministic path:

1. restrict pixels to selected object mask
2. deterministic colour quantization
3. binary mask per palette colour
4. optional conservative morphology cleanup
5. contour tree extraction
6. polygon simplification
7. compound SVG path creation with `fill-rule="evenodd"`
8. named colour groups
9. reconstruction metrics

This keeps the Phase 3 pipeline operational when VTracer is not installed.

## CLI

```bash
poster-vector hard-vectorize object.png \
  --mask object_mask.png \
  -o output/object.svg \
  --report output/object.report.json
```

Force deterministic OpenCV backend:

```bash
poster-vector hard-vectorize object.png \
  --mask object_mask.png \
  -o output/object.svg \
  --backend opencv \
  --colors 8
```

Parameters:

- `--colors`: maximum deterministic palette size
- `--min-area`: reject tiny contour islands
- `--simplify`: contour simplification relative to perimeter
- `--cleanup-radius`: optional morphology cleanup; default `0` preserves measured edges
- `--backend`: `auto`, `opencv`, or `vtracer`

## Outputs

```text
object.svg
object.report.json
```

OpenCV SVG output contains only editable SVG paths/groups; it does not embed the raster source.

The JSON report records:

- backend
- input dimensions
- mask coverage
- palette
- path count
- contour count
- vector coverage
- boundary disagreement ratio
- RGB reconstruction MAE on covered pixels
- confidence class

## Accuracy doctrine

Hard-edge vectorization is allowed to simplify geometry, but it must not invent semantic detail.

Recommended default behavior:

- preserve edges first
- keep morphology disabled unless speckle noise requires it
- use the smallest practical colour palette that preserves intentional colours
- reject tiny islands instead of producing thousands of meaningless nodes
- do not trace smooth gradients into colour bands

## Geometry routing

When an object is recognizably a primitive, later object classification should prefer semantic reconstruction over tracing:

```text
rectangle → <rect>
circle / ellipse → <circle> / <ellipse>
straight polygon → <polygon>
otherwise → path trace
```

The current Phase 3 block supplies the hard-path vectorizer. Primitive recognition is the next foreground-reconstruction sub-stage.

## Acceptance criteria

A hard-graphic vectorization passes this stage when:

- output SVG parses successfully
- at least one editable vector path exists
- no raster `<image>` is embedded by the OpenCV backend
- object selection mask is respected
- report is generated
- path count remains bounded enough for practical editing
- visual edge fidelity is acceptable at intended print size

CorelDRAW interoperability is validated again during final SVG assembly rather than assumed here.
