"""다가구 전세가율 통합 모델 운영 CLI."""
from __future__ import annotations

import argparse
import json
import sqlite3

from src import config
from src.owner_asset_ratio import OwnerAssetRatioIntegrationService
from src.senior_deposit import SeniorDepositIntegrationService

from .service import JeonseRatioIntegrationService


def _property(property_id: str) -> dict:
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM properties WHERE property_id=?", (property_id,)
        ).fetchone()
    if not row:
        raise SystemExit(f"property_id를 찾지 못했습니다: {property_id}")
    return dict(row)


def _calculate(args) -> dict:
    prop = _property(args.property_id)
    senior = SeniorDepositIntegrationService().analyze_property(
        prop,
        reference_date=args.reference_date,
        samples=args.input_samples,
        seed=args.seed,
        mode="scenario",
    )
    value = OwnerAssetRatioIntegrationService().analyze_property(
        prop,
        reference_date=args.reference_date,
        samples=args.input_samples,
        seed=args.seed,
    )
    return JeonseRatioIntegrationService().calculate(
        prop,
        senior,
        value,
        samples=args.samples,
        seed=args.seed,
        dependence=args.dependence,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="다가구 전세가율 확률분포 통합 모델"
    )
    sub = root.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-input-models")
    inspect.add_argument("--property-id")

    for name in ("calculate", "sensitivity", "stress-test"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--property-id", required=True)
        cmd.add_argument("--reference-date", default="2026-07-28")
        cmd.add_argument("--samples", type=int, default=20_000)
        cmd.add_argument("--input-samples", type=int, default=5_000)
        cmd.add_argument("--seed", type=int, default=20260728)
        cmd.add_argument(
            "--dependence",
            choices=[
                "independence", "weak_positive",
                "moderate_positive", "strong_positive",
            ],
            default="independence",
        )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inspect-input-models":
        payload = {
            "senior_deposit": {
                "available": SeniorDepositIntegrationService().available,
                "output_unit": "KRW",
                "required_outputs": [
                    "estimated_total_deposit",
                    "estimated_senior_deposit",
                    "conservative_upper_deposit",
                ],
            },
            "property_value": {
                "available": OwnerAssetRatioIntegrationService().available,
                "output_unit": "manwon",
                "required_output": "estimated_property_value",
                "price_basis": "market_value",
            },
            "alignment": {
                "same_building_id": True,
                "maximum_date_difference_days": 31,
                "normalized_unit": "manwon",
            },
        }
        if args.property_id:
            payload["property"] = _property(args.property_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    result = _calculate(args)
    if args.command == "sensitivity":
        result = {
            "building_id": result.get("building_id"),
            "dependence": result.get("dependence"),
            "uncertainty_contribution": result.get(
                "uncertainty_contribution"
            ),
        }
    elif args.command == "stress-test":
        result = {
            "building_id": result.get("building_id"),
            "base": (result.get("ratios") or {}).get("post_contract_ratio"),
            "stress": result.get("stress"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
