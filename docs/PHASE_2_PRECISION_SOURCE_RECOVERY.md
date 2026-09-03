# Phase 2 — Precision Source Recovery

## Purpose

Phase 2 converts a photographed/raster reference into a trustworthy, geometrically corrected, element-aware source for precise vector reconstruction.

The phase ends only when the system can generate a clean editable background SVG that is measured against visible source pixels and accompanied by confidence/error reporting.

---

## Governing Rule

```text
Visible source pixels  → measured authority
Hidden source pixels   → inference only
Final artwork          → deterministic vector reconstruction
```

AI must never replace valid visible reference pixels simply because the generated result looks cleaner.

---

# Phase 2 Workflow

## 2.1 Input Preservation

Input:
- JPG
- PNG
- TIFF
- photographed print/poster

Actions:
- preserve the untouched original
- record dimensions, orientation and metadata
- create a working copy

Outputs:
- `source/source_original.*`
- `source/source_manifest.json`

---

## 2.2 Perspective + Geometry Normalization

Primary tool: **OpenCV**

Actions:
- orientation correction
- page/poster boundary detection
- perspective rectification / homography
- crop to artwork boundary
- optional lens correction
- preserve coordinate transform metadata

Rules:
- do not AI-upscale before geometric or colour analysis
- preserve a mapping between normalized and original coordinates

Outputs:
- `normalized/reference_rectified.png`
- `normalized/geometry.json`

---

## 2.3 Foreground Segmentation

Primary: **SAM 2**
Secondary: **BiRefNet**

Create masks for:
- background
- logos
- text
- hero/product objects
- icons
- badges
- splashes / decoration
- shadows / highlights where separable

Outputs:
- `masks/background.png`
- `masks/logo.png`
- `masks/text.png`
- `masks/hero.png`
- `masks/icons.png`
- `masks/decoration.png`
- `masks/combined_foreground.png`
- `masks/segmentation.json`

---

## 2.4 Known-Background Map

Primary: **OpenCV + NumPy + scikit-image**

Purpose: determine which pixels are genuinely usable as background measurements.

Exclude:
- foreground pixels
- mask-edge contamination
- glare/specular highlights
- deep shadows caused by photography
- compression contamination where severe

Generate confidence values for usable samples.

Outputs:
- `background/background_known_mask.png`
- `background/background_samples.json`
- `background/background_confidence.png`

---

## 2.5 Hidden Background Recovery

Workflow engine: **ComfyUI**

Preferred order:
1. **LaMa** — first attempt for smooth gradients and repetitive fields
2. **BrushNet** — difficult or large masks
3. **PowerPaint** — semantic/object-aware difficult cases

Rules:
- inpaint only masked unknown areas
- preserve known pixels unchanged
- mark inferred regions separately
- never treat inferred pixels as equal-confidence truth

Outputs:
- `background/background_inferred.png`
- `background/background_inferred_mask.png`
- `background/background_recovery.json`

---

## 2.6 Background Primitive Discovery

Primary: **OpenCV + SciPy + scikit-image**

Detect/estimate:
- main gradient direction
- gradient stop positions
- gradient colours
- large diagonal panel boundaries
- panel opacity
- radial shading / vignettes
- texture direction and spacing
- large low-frequency light/dark fields

Output:
- `analysis/background_primitives.json`

---

## 2.7 Deterministic Vector Construction

Primary: project SVG builder
Optimization: **diffvg**

Allowed background primitives:
- linear gradients
- radial gradients
- polygons
- rectangles
- ellipses
- clipping paths
- opacity
- restrained vector texture

Output:
- `vector/background_initial.svg`

---

## 2.8 Reference-vs-Vector Optimization

Primary: **diffvg + OpenCV + SciPy**

Loop:

```text
SVG parameters
      ↓
render SVG
      ↓
compare with authoritative visible pixels
      ↓
calculate error
      ↓
adjust parameters
      ↓
repeat until convergence
```

Optimize:
- gradient vectors
- stop positions
- stop colours
- panel vertices
- opacities
- radial centres/radii
- texture parameters

Do not optimize hidden/inpainted areas as if they were measured truth.

Outputs:
- `vector/background_optimized.svg`
- `analysis/error_heatmap.png`
- `analysis/fit_metrics.json`

---

## 2.9 CorelDRAW Compatibility Pass

Primary: **Inkscape CLI + project SVG compatibility profile**

Requirements:
- SVG `viewBox`
- named groups
- simple native paths
- linear/radial gradients
- opacity
- clipping paths only where necessary
- no JavaScript
- no external CSS
- no embedded HTML
- avoid filter-heavy effects
- flatten problematic transforms where useful

Output:
- `delivery/background_master_corel.svg`

---

# Required Phase 2 Directory Structure

```text
jobs/<job_id>/
├── source/
│   ├── source_original.*
│   └── source_manifest.json
├── normalized/
│   ├── reference_rectified.png
│   └── geometry.json
├── masks/
│   ├── background.png
│   ├── logo.png
│   ├── text.png
│   ├── hero.png
│   ├── icons.png
│   ├── decoration.png
│   ├── combined_foreground.png
│   └── segmentation.json
├── background/
│   ├── background_known_mask.png
│   ├── background_samples.json
│   ├── background_confidence.png
│   ├── background_inferred.png
│   ├── background_inferred_mask.png
│   └── background_recovery.json
├── analysis/
│   ├── background_primitives.json
│   ├── error_heatmap.png
│   └── fit_metrics.json
├── vector/
│   ├── background_initial.svg
│   └── background_optimized.svg
└── delivery/
    └── background_master_corel.svg
```

---

# Phase 2 Acceptance Gate

Phase 2 is considered complete only when one test reference can automatically produce all of the following:

- [ ] perspective-corrected artwork
- [ ] separate foreground masks
- [ ] known-background mask
- [ ] confidence map
- [ ] inferred hidden-background preview
- [ ] primitive analysis JSON
- [ ] initial editable vector background
- [ ] optimized editable vector background
- [ ] CorelDRAW-compatible SVG
- [ ] vector/reference comparison render
- [ ] error heatmap
- [ ] numerical fit report
- [ ] inferred-area confidence clearly distinguished from measured areas

---

# Phase 2 Success Criterion

The final result must not merely look similar.

It must be:
- editable
- structurally compact
- traceable to measured source pixels
- confidence-labelled where source information is missing
- reproducible from the same input and configuration
- openable in CorelDRAW as editable vector artwork
