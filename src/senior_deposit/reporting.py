from __future__ import annotations

import csv
import json
from pathlib import Path


def write_senior_evaluation_artifacts(
    summary: dict, directory: str | Path, prefix: str = "actual",
) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validation = (summary.get("validation") or {}).get("deposit") or {}
    rows = []
    for label, nominal in (("50", .5), ("80", .8), ("90", .9)):
        empirical = validation.get(f"coverage_{label}")
        if empirical is not None:
            rows.append({
                "model": "unit_deposit",
                "nominal_coverage": nominal,
                "empirical_coverage": empirical,
                "data_kind": summary.get("data_kind"),
            })
    coverage_path = directory / f"{prefix}_coverage.csv"
    with coverage_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model", "nominal_coverage", "empirical_coverage", "data_kind"),
        )
        writer.writeheader()
        writer.writerows(rows)

    points = []
    for row in rows:
        x = 60 + float(row["nominal_coverage"]) * 420
        y = 500 - float(row["empirical_coverage"]) * 420
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#FFBC00">'
            f'<title>nominal={row["nominal_coverage"]}, '
            f'empirical={row["empirical_coverage"]}</title></circle>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="560" height="560"
viewBox="0 0 560 560" role="img" aria-label="Senior deposit interval calibration">
<rect width="560" height="560" fill="white"/>
<text x="280" y="28" text-anchor="middle" font-size="18">
Unit-deposit interval coverage ({summary.get('data_kind')})</text>
<line x1="60" y1="500" x2="500" y2="500" stroke="#333"/>
<line x1="60" y1="500" x2="60" y2="60" stroke="#333"/>
<line x1="60" y1="500" x2="480" y2="80" stroke="#999" stroke-dasharray="6 5"/>
<text x="280" y="540" text-anchor="middle">Nominal coverage</text>
<text x="16" y="280" transform="rotate(-90 16 280)" text-anchor="middle">
Empirical coverage</text>
{''.join(points)}
</svg>"""
    svg_path = directory / f"{prefix}_calibration.svg"
    svg_path.write_text(svg, encoding="utf-8")

    json_path = directory / f"{prefix}_evaluation.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "coverage_csv": str(coverage_path),
        "calibration_svg": str(svg_path),
        "evaluation_json": str(json_path),
    }
