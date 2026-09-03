from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np


def analyze_image(path: str | Path) -> dict:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Coarse 5x7 sampling grid. This intentionally captures the low-frequency
    # colour field instead of tracing local foreground detail.
    rows, cols = 7, 5
    samples = []
    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)
        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)
            tile = image[y0:y1, x0:x1]
            med = np.median(tile.reshape(-1, 3), axis=0).astype(int)
            samples.append({
                "row": r,
                "col": c,
                "x": round((x0 + x1) / 2 / w, 4),
                "y": round((y0 + y1) / 2 / h, 4),
                "rgb": med.tolist(),
                "hex": "#%02X%02X%02X" % tuple(med),
            })

    # Low-frequency approximation useful for gradient fitting.
    small = cv2.resize(image, (max(8, w // 32), max(8, h // 32)), interpolation=cv2.INTER_AREA)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    mae = float(np.mean(np.abs(image.astype(np.float32) - smooth.astype(np.float32))))

    return {
        "file": path.name,
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 6),
        "grid": {"rows": rows, "cols": cols, "samples": samples},
        "detail_residual_mae": round(mae, 3),
        "note": "Grid medians include foreground where present; use masks or manually select background-safe samples for precise fitting.",
    }


def save_analysis(path: str | Path, output: str | Path) -> Path:
    result = analyze_image(path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output
