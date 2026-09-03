# Third-party projects

This repository does not copy/vendor the upstream projects below. They are optional integrations or implementation references and should be installed separately.

## diffvg
- Repository: https://github.com/BachiLi/diffvg
- Purpose: differentiable rasterization and vector parameter refinement.
- Repository license reported by GitHub: Apache License 2.0.

## VTracer
- Repository: https://github.com/visioncortex/vtracer
- Purpose: hard-edge raster-to-vector tracing for logos/icons and secondary assets.
- Repository license reported by GitHub: MIT License.

## JPG-to-SVG
- Repository: https://github.com/Furinaaa-Cancan/JPG-to-SVG
- Purpose: architectural reference for semantic segmentation + hybrid vectorization.
- Important: GitHub currently reports a non-standard/no-assertion repository license and the root LICENSE file contains Meta's SAM license text, despite the README displaying an Apache-2.0 badge. Therefore this project is **not copied or bundled here**. Review its complete licensing/dependency situation before any redistribution or commercial integration.

## Policy for this repository

The core poster-vector-rebuilder code is intentionally self-contained. Optional upstream integrations must remain replaceable. Do not copy upstream source code into this repository without retaining all applicable notices and first resolving the relevant license terms.
