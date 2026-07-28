from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.owner_asset_ratio.cli import (
    ARTIFACTS as OWNER_ARTIFACTS,
    PROCESSED as OWNER_PROCESSED,
    RAW as OWNER_RAW,
    _download_rtms,
    collect_buildings,
    preprocess,
)
from src.owner_asset_ratio.pipeline import OwnerAssetRatioPipeline
from .pipeline import SeniorDepositPipeline
from .reporting import write_senior_evaluation_artifacts
from .schemas import validate_senior_labels


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "models" / "senior_deposit"
REPORTS = ROOT / "reports" / "senior_deposit"
LABELS = ROOT / "data" / "labels" / "senior_deposit"
DEFAULT_ARTIFACT = ARTIFACTS / "senior_deposit_actual.joblib"
DEFAULT_OWNER_ARTIFACT = (
    OWNER_ARTIFACTS / "owner_asset_ratio_actual.joblib")


def _mkdirs() -> None:
    for path in (ARTIFACTS, REPORTS, LABELS):
        path.mkdir(parents=True, exist_ok=True)


def audit_data_sources(args) -> None:
    def rows(path: Path) -> int | None:
        if not path.exists():
            return None
        return len(pd.read_csv(path, low_memory=False))

    label_path = LABELS / "labels.csv"
    def schema_audit(raw_path: Path, mapping_path: Path) -> dict:
        mapping = yaml.safe_load(
            mapping_path.read_text(encoding="utf-8")) or {}
        columns = set(pd.read_csv(raw_path, nrows=0).columns)
        missing = {
            canonical: source for canonical, source in mapping.items()
            if source not in columns
        }
        return {
            "mapping": str(mapping_path),
            "mapped_fields": len(mapping),
            "missing_in_actual_response": missing,
            "passed": not missing,
        }

    payload = {
        "building_hub": {
            "path": str(OWNER_RAW / "suwon_building_hub_title.csv"),
            "rows": rows(OWNER_RAW / "suwon_building_hub_title.csv"),
            "kind": "official",
            "schema": schema_audit(
                OWNER_RAW / "suwon_building_hub_title.csv",
                ROOT / "configs" / "senior_deposit" / "building_schema.yaml",
            ),
        },
        "rtms_rent": {
            "path": str(OWNER_RAW / "suwon_sh_rent.csv"),
            "rows": rows(OWNER_RAW / "suwon_sh_rent.csv"),
            "kind": "official",
            "schema": schema_audit(
                OWNER_RAW / "suwon_sh_rent.csv",
                ROOT / "configs" / "senior_deposit" / "rtms_rent_schema.yaml",
            ),
        },
        "processed_buildings": rows(OWNER_PROCESSED / "buildings.csv"),
        "processed_leases": rows(OWNER_PROCESSED / "leases.csv"),
        "verified_labels": rows(label_path),
        "occupancy_observed": bool(label_path.exists()),
        "legal_seniority_observed": bool(label_path.exists()),
        "automatic_court_scraping": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def collect_rent_transactions(args) -> None:
    _download_rtms("lease", args)


def import_labels(args) -> None:
    source = Path(args.path)
    files = (
        sorted(source.glob("*.csv")) if source.is_dir() else [source])
    if not files or not all(path.exists() for path in files):
        raise SystemExit(f"label CSV not found: {source}")
    raw = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in files],
        ignore_index=True,
    )
    clean = validate_senior_labels(raw)
    output = LABELS / "labels.csv"
    clean.to_csv(output, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "out": str(output),
        "rows": len(clean),
        "seniority_labels": int(
            clean["senior_to_target"].notna().sum()),
        "direct_personal_identifiers_stored": False,
    }, ensure_ascii=False, indent=2))


def train_all(args) -> None:
    owner = OwnerAssetRatioPipeline.load(
        args.owner_artifact, allow_synthetic=False)
    leases = pd.read_csv(OWNER_PROCESSED / "leases.csv", low_memory=False)
    model = SeniorDepositPipeline.fit_from_actual_data(
        owner_pipeline=owner, leases=leases, seed=args.seed)
    model.save(args.artifact)
    artifacts = write_senior_evaluation_artifacts(
        model.evaluation_summary(), REPORTS, "actual")
    print(json.dumps({
        "artifact": str(args.artifact),
        "model_mode": model.metadata["model_mode"],
        "evaluation": model.evaluation_summary(),
        "reports": artifacts,
    }, ensure_ascii=False, indent=2))


def train_unit_count(args) -> None:
    owner = OwnerAssetRatioPipeline.load(
        args.owner_artifact, allow_synthetic=False)
    print(json.dumps({
        "status": "reused_actual_model",
        "selected_model": owner.unit_model.selected_name,
        "validation": owner.unit_model.validation_metrics,
        "observed_registry_values_take_priority": True,
    }, ensure_ascii=False, indent=2))


def train_occupancy(args) -> None:
    print(json.dumps({
        "trained": False,
        "model_mode": "scenario_only",
        "reason": "verified current-occupancy labels are unavailable",
        "priors": {
            "low": {"alpha": 7, "beta": 3},
            "baseline": {"alpha": 18, "beta": 2},
            "high": {"alpha": 38, "beta": 2},
        },
    }, ensure_ascii=False, indent=2))


def train_seniority(args) -> None:
    label_path = LABELS / "labels.csv"
    verified = 0
    if label_path.exists():
        labels = validate_senior_labels(
            pd.read_csv(label_path, low_memory=False))
        verified = int(labels["senior_to_target"].notna().sum())
    print(json.dumps({
        "trained": False,
        "model_mode": "scenario_only",
        "verified_labels": verified,
        "reason": (
            "no verified labels" if verified == 0
            else "classifier training requires a separately approved minimum "
                 "sample and grouped evaluation"
        ),
        "fabricated_labels_used": False,
    }, ensure_ascii=False, indent=2))


def train_calibrator(args) -> None:
    print(json.dumps({
        "trained": False,
        "reason": "building-level verified senior-deposit totals are unavailable",
        "simulator_replaced": False,
    }, ensure_ascii=False, indent=2))


def evaluate(args) -> None:
    model = SeniorDepositPipeline.load(args.artifact)
    summary = model.evaluation_summary()
    reports = write_senior_evaluation_artifacts(
        summary, REPORTS, "actual")
    print(json.dumps({
        "evaluation": summary,
        "reports": reports,
    }, ensure_ascii=False, indent=2))


def infer(args) -> None:
    model = SeniorDepositPipeline.load(args.artifact)
    buildings = pd.read_csv(
        args.building_csv or OWNER_PROCESSED / "buildings.csv",
        low_memory=False,
    )
    matches = buildings[
        buildings["building_id"].astype(str).eq(str(args.building_id))]
    if matches.empty:
        raise SystemExit(f"building_id not found: {args.building_id}")
    result = model.infer(
        matches.iloc[0].to_dict(),
        reference_date=args.reference_date,
        samples=args.samples,
        seed=args.seed,
        mode=args.scenario,
        occupancy_scenario=args.occupancy,
        senior_probability=args.senior_probability,
        random_effect_sigma=args.random_effect_sigma,
        target_rooms_excluded=args.target_rooms_excluded,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _collection_arguments(target) -> None:
    target.add_argument("--region", default="suwon", choices=["suwon"])
    target.add_argument("--start-month", default="202407")
    target.add_argument("--end-month", default="202607")
    target.add_argument("--num-rows", type=int, default=1000)
    target.add_argument("--sleep-sec", type=float, default=.05)
    target.add_argument("--timeout", type=int, default=30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suwon probabilistic senior-deposit MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-data-sources")
    audit.set_defaults(func=audit_data_sources)

    buildings = sub.add_parser("collect-buildings")
    buildings.add_argument("--region", default="suwon", choices=["suwon"])
    buildings.add_argument(
        "--legal-dong-codes",
        default=str(
            ROOT / "data" / "downloaded" / "real_estate"
            / "national_legal_dong_codes_20260630.csv"),
    )
    buildings.set_defaults(func=collect_buildings)

    rent = sub.add_parser("collect-rent-transactions")
    _collection_arguments(rent)
    rent.set_defaults(func=collect_rent_transactions)

    labels = sub.add_parser("import-labels")
    labels.add_argument("--path", required=True)
    labels.set_defaults(func=import_labels)

    pre = sub.add_parser("preprocess")
    pre.set_defaults(func=preprocess)

    def owner_artifact_arg(target):
        target.add_argument(
            "--owner-artifact", type=Path, default=DEFAULT_OWNER_ARTIFACT)

    unit = sub.add_parser("train-unit-count")
    owner_artifact_arg(unit)
    unit.set_defaults(func=train_unit_count)

    occupancy = sub.add_parser("train-occupancy")
    occupancy.set_defaults(func=train_occupancy)

    for command in ("train-deposit", "train-all"):
        target = sub.add_parser(command)
        owner_artifact_arg(target)
        target.add_argument(
            "--artifact", type=Path, default=DEFAULT_ARTIFACT)
        target.add_argument("--seed", type=int, default=20260728)
        target.set_defaults(func=train_all)

    seniority = sub.add_parser("train-seniority")
    seniority.set_defaults(func=train_seniority)
    calibrator = sub.add_parser("train-calibrator")
    calibrator.set_defaults(func=train_calibrator)

    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument(
        "--artifact", type=Path, default=DEFAULT_ARTIFACT)
    evaluation.set_defaults(func=evaluate)

    inference = sub.add_parser("infer")
    inference.add_argument("--building-id", required=True)
    inference.add_argument("--reference-date", required=True)
    inference.add_argument(
        "--artifact", type=Path, default=DEFAULT_ARTIFACT)
    inference.add_argument("--building-csv")
    inference.add_argument("--samples", type=int, default=20_000)
    inference.add_argument("--seed", type=int, default=20260728)
    inference.add_argument(
        "--scenario",
        choices=["conservative", "probabilistic", "scenario"],
        default="conservative",
    )
    inference.add_argument(
        "--occupancy", choices=["low", "baseline", "high"],
        default="baseline")
    inference.add_argument("--senior-probability", type=float)
    inference.add_argument("--random-effect-sigma", type=float)
    inference.add_argument("--target-rooms-excluded", type=int, default=1)
    inference.set_defaults(func=infer)
    return parser


def main(argv: list[str] | None = None) -> int:
    _mkdirs()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
