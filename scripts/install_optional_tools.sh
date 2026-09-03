#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

if [ ! -d third_party/diffvg ]; then
  git clone --recursive https://github.com/BachiLi/diffvg.git third_party/diffvg
fi

if [ ! -d third_party/vtracer ]; then
  git clone https://github.com/visioncortex/vtracer.git third_party/vtracer
fi

cat <<'EOF'
Optional upstream sources downloaded into third_party/.

JPG-to-SVG is intentionally NOT cloned automatically because its repository-level licensing metadata and root LICENSE are inconsistent with the README badge. Review THIRD_PARTY.md before integrating it.

Next:
- install/build diffvg according to its upstream README
- install VTracer via its documented Rust/Python workflow
EOF
