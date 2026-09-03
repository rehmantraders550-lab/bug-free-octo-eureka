"""Conservative editable-text recovery using the installed Tesseract binary."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
import subprocess


def recover_text(image: str | Path, output: str | Path) -> dict:
    image, output = Path(image), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("tesseract")
    result = {"engine": "tesseract", "available": bool(executable), "items": [], "limitations": ["Font identity is not inferred. All recovered text uses a substitute until a human confirms the intended font."]}
    if not executable:
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    process = subprocess.run([executable, str(image), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, check=False)
    if process.returncode != 0:
        result["error"] = process.stderr.strip()
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    rows = csv.DictReader(process.stdout.splitlines(), delimiter="\t")
    for row in rows:
        word = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        # Avoid introducing noisy single-character guesses as editable artwork.
        # Omitted OCR is safer than a confident-looking but incorrect word.
        alpha_num_count = sum(char.isalnum() for char in word)
        if word and confidence >= 85 and alpha_num_count >= 2:
            result["items"].append({"text": word, "x": int(row["left"]), "y": int(row["top"]), "width": int(row["width"]), "height": int(row["height"]), "confidence": confidence, "confidence_class": "B" if confidence >= 80 else "C", "font_status": "unconfirmed substitute"})
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
