# Phase 2 — Precision Tool Matrix

This file defines exactly which tools are required for Phase 2 and what each tool is allowed to do.

## Tier 1 — Mandatory Core

| Tool | Repository | Role | Why Required |
|---|---|---|---|
| OpenCV | `opencv/opencv` | Perspective correction, geometry, masking, sampling, edge/panel analysis | Deterministic source measurement foundation |
| NumPy | Python package | Pixel arrays, transforms, numerical operations | Core numerical dependency |
| SciPy | Python package | Interpolation and parameter optimization | Needed for stable mathematical fitting |
| scikit-image | `scikit-image/scikit-image` | Lab colour, structural comparisons, segmentation helpers | Improves measurement and error analysis |
| SAM 2 | `facebookresearch/sam2` | Foreground/object segmentation | Primary automatic masking engine |
| diffvg | `BachiLi/diffvg` | Differentiable vector fitting | Core optimizer for matching vector parameters to raster reference |
| Inkscape | `inkscape/inkscape` | SVG rendering, normalization, interoperability checks | CorelDRAW-safe SVG validation |

## Tier 2 — Strongly Recommended Accuracy Modules

| Tool | Repository | Role | Use |
|---|---|---|---|
| BiRefNet | `ZhengPeng7/BiRefNet` | Fine foreground/background separation | Refines boundaries where SAM 2 masks are insufficient |
| LaMa | `advimman/lama` | Background inpainting | First choice for smooth gradients and broad hidden areas |
| ComfyUI | `Comfy-Org/ComfyUI` | AI workflow orchestration | Runs repeatable segmentation/inpainting workflows |

## Tier 3 — Optional Difficult-Case Modules

| Tool | Repository | Role | Trigger |
|---|---|---|---|
| BrushNet | `TencentARC/BrushNet` | Diffusion inpainting | Use only if LaMa produces poor recovery |
| PowerPaint | `open-mmlab/PowerPaint` | Semantic inpainting | Use when semantic reconstruction is needed |
| VTracer | `visioncortex/vtracer` | Hard-edge vectorization | Mainly Phase 3+, but may help geometric decorative elements |
| Potrace | `potrace/potrace` | Monochrome tracing | Optional for binary masks/silhouettes |

## Explicitly Not Required in Phase 2

| Tool | Reason |
|---|---|
| AUTOMATIC1111 | ComfyUI already supplies the workflow engine role |
| `sd-webui-controlnet` | A1111-specific extension; unnecessary for this stack |
| IOPaint | Archived application; useful reference but not a core dependency |
| Real-ESRGAN | Upscaling can distort authoritative source measurements |
| Ghostscript | Needed later for PDF/prepress, not source recovery |
| Scribus | Needed later for print PDF generation |
| qpdf / pdfcpu | Needed later for PDF validation |
| PaddleOCR / Tesseract | Typography reconstruction is a later phase |

---

# Tool Routing Rules

## Geometry
Use **OpenCV only** for:
- perspective
- rotation
- coordinate mapping
- page boundaries
- panel edges

AI must not be responsible for geometric normalization.

## Segmentation
Use:
1. SAM 2
2. BiRefNet refinement where needed
3. manual/heuristic cleanup fallback

## Hidden-pixel recovery
Use:
1. LaMa
2. BrushNet if required
3. PowerPaint only for difficult semantic gaps

Generated pixels remain lower-confidence than visible samples.

## Vector fitting
Use:
1. custom SVG builder for semantic structure
2. SciPy for deterministic parameter fitting
3. diffvg for differentiable fine optimization

## Validation
Use:
- Inkscape render for raster comparison
- OpenCV/scikit-image for fit metrics
- XML/SVG structural checks for editability

---

# Recommended Installation Order

```text
1. OpenCV
2. NumPy
3. SciPy
4. scikit-image
5. Inkscape
6. SAM 2
7. BiRefNet
8. LaMa
9. ComfyUI
10. diffvg
11. BrushNet / PowerPaint only if needed
```

This keeps the deterministic foundation operational even when GPU/AI modules are unavailable.

---

# Hardware Profiles

## CPU-only minimum
Can run:
- OpenCV
- NumPy
- SciPy
- scikit-image
- Inkscape
- SVG generation
- deterministic optimization

Limitations:
- SAM 2 / BiRefNet / inpainting may be slow
- diffvg optimization may be significantly slower

## Recommended local workstation
- NVIDIA GPU with CUDA support
- 12 GB VRAM practical target
- 16–32 GB system RAM
- SSD storage

## Higher-complexity jobs
24 GB+ VRAM gives more headroom for larger segmentation/inpainting models, but must not become a hard product requirement.

---

# Phase 2 Dependency Philosophy

The pipeline must remain modular.

```text
Deterministic core
      ↓
always available

AI modules
      ↓
accuracy/recovery enhancement

Optional difficult-case modules
      ↓
invoked only when required
```

No single AI model should become a point of failure for the entire workflow.
