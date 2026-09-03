# Poster Vector Rebuilder — Resource Manifest v1

This manifest separates mandatory pipeline components from optional accuracy modules and prepress utilities. The project should integrate tools by pinned version or external adapter rather than blindly vendoring whole repositories.

## A. Core reconstruction stack

### ComfyUI — `Comfy-Org/ComfyUI`
Role: orchestration for segmentation/inpainting/reference-conditioned image recovery.
Status: public, active.
Use: API-driven node workflows; do not require AUTOMATIC1111 in parallel.

### OpenCV — `opencv/opencv`
Role: deterministic image geometry and colour analysis.
Use: perspective correction, masks, sampling, panel/edge detection, comparison metrics.

### SAM 2 — `facebookresearch/sam2`
Role: foreground/object segmentation and mask generation.
Use: isolate text/logo/hero objects/background regions before recovery.

### diffvg — `BachiLi/diffvg`
Role: differentiable vector rendering/optimization.
Use: fit vector parameters against reference pixels.

### VTracer — `visioncortex/vtracer`
Role: colour raster-to-vector tracing for hard-edged graphics.
Use: logos/icons/organic graphic elements after segmentation; not for smooth gradient backgrounds.

### Inkscape — `inkscape/inkscape`
Role: SVG normalization and interoperability testing.
Use: simplify/normalize final SVG and validate editable import behaviour.

## B. Background recovery modules

### LaMa — `advimman/lama`
Role: image inpainting.
Preferred use: smooth gradients, repetitive backgrounds, broad object removal.

### BrushNet — `TencentARC/BrushNet`
Role: diffusion-based inpainting.
Preferred use: difficult masks or where LaMa reconstruction is insufficient.

### PowerPaint — `open-mmlab/PowerPaint`
Role: semantic/object-aware inpainting.
Preferred use: optional difficult recovery stage, not authority for visible pixels.

### BiRefNet — `ZhengPeng7/BiRefNet`
Role: high-quality foreground/background separation.
Preferred use: optional complement to SAM 2 for clean masks and soft boundaries.

## C. Typography stack

### PaddleOCR — `PaddlePaddle/PaddleOCR`
Role: primary OCR/text-region analysis.

### Tesseract — `tesseract-ocr/tesseract`
Role: fallback OCR and lightweight deployments.

### fontTools — `fonttools/fonttools`
Role: font metadata/glyph inspection/subsetting and text tooling.

### FontForge — `fontforge/fontforge`
Role: optional font inspection/conversion and outline operations.

## D. Raster restoration — optional only

### Real-ESRGAN — `xinntao/Real-ESRGAN`
Role: super-resolution/restoration.
Rule: never upscale the only authoritative reference before colour/geometry analysis. Use only for optional proof/detail enhancement or secondary inspection.

## E. Prepress stack

### Scribus — `scribusproject/scribus`
Role: print-oriented page assembly, colour/spot workflow and PDF export.

### LittleCMS — `mm2/Little-CMS`
Role: ICC colour-management engine.

### Ghostscript/GhostPDL — `ArtifexSoftware/ghostpdl`
Role: PDF/PostScript processing, proof rendering and separation inspection.
Licensing note: AGPL/commercial licensing implications must be reviewed before distributing a closed-source hosted product that incorporates Ghostscript.

### qpdf — `qpdf/qpdf`
Role: PDF structural validation/manipulation.

### pdfcpu — `pdfcpu/pdfcpu`
Role: PDF page-box and structural utilities; useful supplementary validator.

## F. Explicit non-core / excluded runtime dependencies

### IOPaint — `Sanster/IOPaint`
Useful reference application and model aggregator for inpainting, but the repository is archived. Do not make it a hard production dependency. Integrate required models/workflows directly.

### AUTOMATIC1111 — `AUTOMATIC1111/stable-diffusion-webui`
Not required in the proposed production architecture. ComfyUI covers the orchestration role with better workflow reproducibility for this project.

### sd-webui-controlnet — `Mikubill/sd-webui-controlnet`
Do not depend on the A1111 extension itself. Use ControlNet/reference-conditioning capabilities through ComfyUI-compatible models/nodes.

### JPG-to-SVG — `Furinaaa-Cancan/JPG-to-SVG`
Reference only pending clean licensing review. Do not vendor automatically.

## G. CorelDRAW interoperability

CorelDRAW itself is proprietary and is not a GitHub dependency.

The delivery profile should instead target CorelDRAW-safe SVG constructs:
- paths and Béziers
- rectangles/ellipses/polygons
- linear/radial gradients
- fills and strokes
- opacity
- clipping paths
- named groups
- editable text where fonts are available

Optional CDR-related libraries may be evaluated later, but native CDR generation is not a current requirement. SVG + vector PDF are the canonical outputs.

## Integration policy

1. Prefer subprocess/API adapters over vendoring large upstream codebases.
2. Pin known-good versions in deployment manifests.
3. Record model licenses separately from code licenses.
4. Preserve an offline deterministic path for geometry/vector/PDF stages.
5. Never let AI-generated content silently overwrite measured source data.
6. Every optional component must have a fallback path.
