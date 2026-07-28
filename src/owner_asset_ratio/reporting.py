from __future__ import annotations

import csv
import json
from pathlib import Path


def write_evaluation_artifacts(summary: dict, directory: str | Path,
                               prefix: str) -> dict[str, str]:
    """Write a coverage table and dependency-free SVG calibration plot."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validation = summary["validation"]
    rows = []
    for model_name in ("deposit", "property_value"):
        metrics = validation.get(model_name) or {}
        rows.extend([
            {
                "model": model_name,
                "nominal_coverage": .5,
                "empirical_coverage": metrics.get("coverage_50"),
                "data_kind": summary["data_kind"],
            },
            {
                "model": model_name,
                "nominal_coverage": .8,
                "empirical_coverage": metrics.get("coverage_80"),
                "data_kind": summary["data_kind"],
            },
        ])
    coverage_path = directory / f"{prefix}_coverage.csv"
    with coverage_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"deposit": "#FFBC00", "property_value": "#2F7ED8"}
    points = []
    for row in rows:
        if row["empirical_coverage"] is None:
            continue
        x = 60 + float(row["nominal_coverage"]) * 420
        y = 500 - float(row["empirical_coverage"]) * 420
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" '
            f'fill="{colors[row["model"]]}"><title>'
            f'{row["model"]}: nominal={row["nominal_coverage"]}, '
            f'empirical={row["empirical_coverage"]}</title></circle>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="560" height="560"
viewBox="0 0 560 560" role="img" aria-label="Prediction interval calibration">
<rect width="560" height="560" fill="white"/>
<text x="280" y="28" text-anchor="middle" font-size="18">
Prediction interval coverage ({summary['data_kind']})</text>
<line x1="60" y1="500" x2="500" y2="500" stroke="#333"/>
<line x1="60" y1="500" x2="60" y2="60" stroke="#333"/>
<line x1="60" y1="500" x2="480" y2="80" stroke="#999" stroke-dasharray="6 5"/>
<text x="280" y="540" text-anchor="middle">Nominal coverage</text>
<text x="16" y="280" transform="rotate(-90 16 280)" text-anchor="middle">
Empirical coverage</text>
{''.join(points)}
<rect x="350" y="36" width="12" height="12" fill="#FFBC00"/>
<text x="368" y="47">deposit</text>
<rect x="430" y="36" width="12" height="12" fill="#2F7ED8"/>
<text x="448" y="47">value</text>
</svg>"""
    svg_path = directory / f"{prefix}_calibration.svg"
    svg_path.write_text(svg, encoding="utf-8")

    json_path = directory / f"{prefix}_evaluation.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "coverage_csv": str(coverage_path),
        "calibration_svg": str(svg_path),
        "evaluation_json": str(json_path),
    }
