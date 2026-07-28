from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from .data import (
    BuildingRegistryCollector,
    add_past_only_lease_features,
    load_survey_mapping,
    match_sales_to_buildings,
    normalize_building_hub,
    normalize_household_survey,
    normalize_rtms_leases,
    normalize_rtms_sales,
    validate_buildings,
    write_provenance,
)
from .models import OwnerAssetPrior, QuantileModel, UnitCountModel
from .pipeline import (
    DEPOSIT_CATEGORICAL,
    DEPOSIT_NUMERIC,
    UNIT_CATEGORICAL,
    UNIT_NUMERIC,
    VALUE_CATEGORICAL,
    VALUE_NUMERIC,
    OwnerAssetRatioPipeline,
    _temporal_split,
)
from .reporting import write_evaluation_artifacts
from .schemas import BuildingEstimateInput, SUWON_SIGUNGU
from .synthetic import make_synthetic_frames


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "owner_asset_ratio"
INTERIM = ROOT / "data" / "interim" / "owner_asset_ratio"
PROCESSED = ROOT / "data" / "processed" / "owner_asset_ratio"
ARTIFACTS = ROOT / "models" / "owner_asset_ratio"
REPORTS = ROOT / "reports" / "owner_asset_ratio"


def _mkdirs() -> None:
    for path in (RAW, INTERIM, PROCESSED, ARTIFACTS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def _months(start: str, end: str) -> list[str]:
    year, month = int(start[:4]), int(start[4:])
    end_value = int(end)
    result = []
    while year * 100 + month <= end_value:
        result.append(f"{year:04d}{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def _download_rtms(kind: str, args) -> None:
    key = (
        os.environ.get("MOLIT_SERVICE_KEY", "")
        or os.environ.get("MOLIT_RTMS_SERVICE_KEY", "")
    ).strip()
    if not key:
        raise SystemExit(
            "MOLIT_SERVICE_KEY or MOLIT_RTMS_SERVICE_KEY is required. "
            "It is never hardcoded.")
    months = _months(args.start_month, args.end_month)
    codes = list(SUWON_SIGUNGU)
    if kind == "lease":
        from scripts.download_rtms_sh_rent import download
        frame, events = download(
            key, codes, months, args.num_rows, args.sleep_sec, args.timeout)
        out = RAW / "suwon_sh_rent.csv"
    else:
        from scripts.download_rtms_sh_trade import download
        frame, events = download(
            key, codes, months, args.num_rows, args.sleep_sec, args.timeout)
        out = RAW / "suwon_sh_trade.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    write_provenance(
        out.with_suffix(".provenance.json"),
        source_kind="official_rtms", rows=len(frame),
        extra={"months": months, "sigungu_codes": codes,
               "requests": len(events)})
    print(json.dumps({"out": str(out), "rows": len(frame)}, ensure_ascii=False))


def collect_buildings(args) -> None:
    _mkdirs()
    codes_file = Path(args.legal_dong_codes)
    if not codes_file.exists():
        raise SystemExit(
            "A legal-dong code file is required. Supply one code per line with "
            "--legal-dong-codes; no district-average pseudo rows are created.")
    code_frame = pd.read_csv(codes_file, dtype=str)
    code_column = next(
        (name for name in ("법정동코드", "legal_dong_code", "code")
         if name in code_frame),
        None,
    )
    if code_column is None:
        raise SystemExit(
            "legal-dong code file has no 법정동코드/legal_dong_code/code column")
    codes = [
        value for value in code_frame[code_column].dropna().astype(str)
        if value[:5] in SUWON_SIGUNGU and value[5:10] != "00000"
    ]
    frame = BuildingRegistryCollector().collect(codes)
    out = RAW / "suwon_building_hub_title.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    write_provenance(out.with_suffix(".provenance.json"),
                     source_kind="official_building_hub", rows=len(frame))
    print(json.dumps({"out": str(out), "rows": len(frame)}, ensure_ascii=False))


def preprocess(args) -> None:
    _mkdirs()
    building_raw = pd.read_csv(
        RAW / "suwon_building_hub_title.csv", low_memory=False)
    lease_raw = pd.read_csv(RAW / "suwon_sh_rent.csv", low_memory=False)
    sale_raw = pd.read_csv(RAW / "suwon_sh_trade.csv", low_memory=False)
    buildings, quality = validate_buildings(
        normalize_building_hub(building_raw))
    leases = add_past_only_lease_features(normalize_rtms_leases(lease_raw))
    sales = match_sales_to_buildings(
        normalize_rtms_sales(sale_raw), buildings)
    catalog_path = (
        ROOT / "data" / "downloaded" / "real_estate"
        / "national_legal_dong_codes_20260630.csv")
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path, dtype=str)
        name_by_code = dict(zip(
            catalog["법정동코드"], catalog["읍면동명"].fillna("")))
        buildings["legal_dong"] = (
            buildings["legal_dong_code"].astype(str).map(name_by_code))
    else:
        buildings["legal_dong"] = ""
    lease_count = leases.groupby("legal_dong").size()
    sale_count = sales.groupby("legal_dong").size()
    buildings["local_market_count"] = (
        buildings["legal_dong"].map(lease_count).fillna(0)
        + buildings["legal_dong"].map(sale_count).fillna(0)
    ).astype(int)
    buildings.to_csv(
        PROCESSED / "buildings.csv", index=False, encoding="utf-8-sig")
    leases.to_csv(
        PROCESSED / "leases.csv", index=False, encoding="utf-8-sig")
    sales.to_csv(
        PROCESSED / "sales.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "data_quality.json").write_text(
        json.dumps({
            "building": quality, "leases": len(leases), "sales": len(sales),
            "suwon_only": True, "building_level_lease_join_used": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"building": len(buildings), "leases": len(leases),
                      "sales": len(sales)}, ensure_ascii=False))


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(pd.read_csv(PROCESSED / name, low_memory=False) for name in (
        "buildings.csv", "leases.csv", "sales.csv"))  # type: ignore


def _read_survey_csv(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise SystemExit(
        f"Unable to decode household survey CSV: {path}. "
        + " | ".join(errors))


def _load_survey(args) -> pd.DataFrame:
    source = Path(args.survey)
    if source.is_dir():
        files = sorted(source.glob("*.csv"))
        if not files:
            raise SystemExit(f"No survey CSV files found in: {source}")
        raw = pd.concat(
            [_read_survey_csv(path) for path in files],
            ignore_index=True,
            sort=False,
        )
    else:
        raw = _read_survey_csv(source)
    mapping = load_survey_mapping(args.survey_mapping)
    return normalize_household_survey(raw, mapping)


def train_unit(args) -> None:
    buildings, _, _ = _load_processed()
    known = buildings[buildings["registered_units_observed"].notna()]
    ordered = known.sort_values("approval_date")
    split = max(1, int(len(ordered) * .8))
    model = UnitCountModel.fit(
        ordered.iloc[:split], ordered.iloc[split:],
        UNIT_NUMERIC, UNIT_CATEGORICAL)
    joblib.dump(model, ARTIFACTS / "unit_count.joblib")
    print(json.dumps(model.validation_metrics, ensure_ascii=False, indent=2))


def train_deposit(args) -> None:
    _, leases, _ = _load_processed()
    train, validation, _ = _temporal_split(leases, "contract_year_month")
    model = QuantileModel(
        DEPOSIT_NUMERIC, DEPOSIT_CATEGORICAL).fit(
            train, "deposit", validation)
    joblib.dump(model, ARTIFACTS / "deposit_quantile.joblib")
    print(json.dumps(model.validation_, ensure_ascii=False, indent=2))


def train_value(args) -> None:
    _, _, sales = _load_processed()
    direct = sales[sales["match_confidence"].isin(["exact", "high"])]
    if len(direct) < 100:
        raise SystemExit(
            "Not enough exact/high building matches. Partial-lot rows are "
            "kept only for comparables and are not promoted to labels.")
    train, validation, _ = _temporal_split(direct, "contract_year_month")
    model = QuantileModel(
        VALUE_NUMERIC, VALUE_CATEGORICAL).fit(
            train, "sale_price", validation)
    joblib.dump(model, ARTIFACTS / "property_value.joblib")
    print(json.dumps(model.validation_, ensure_ascii=False, indent=2))


def build_owner_prior(args) -> None:
    survey = _load_survey(args)
    model = OwnerAssetPrior().fit(survey)
    joblib.dump(model, ARTIFACTS / "owner_asset_prior.joblib")
    print(json.dumps(model.weighted_quantiles(), ensure_ascii=False, indent=2))


def train_all(args) -> None:
    buildings, leases, sales = _load_processed()
    survey = _load_survey(args)
    model = OwnerAssetRatioPipeline.fit(
        buildings, leases, sales, survey, data_kind="actual",
        seed=args.seed)
    model.save(args.artifact)
    write_evaluation_artifacts(
        model.evaluation_summary(), REPORTS, "actual")
    print(json.dumps(model.evaluation_summary(), ensure_ascii=False, indent=2))


def smoke_train(args) -> None:
    _mkdirs()
    frames = make_synthetic_frames(args.seed)
    model = OwnerAssetRatioPipeline.fit(
        **frames, data_kind="synthetic_smoke_only", seed=args.seed)
    model.save(args.artifact)
    example = model.infer(
        BuildingEstimateInput.from_mapping(
            frames["buildings"].iloc[0].to_dict()),
        samples=args.samples, seed=args.seed)
    (REPORTS / "synthetic_smoke_inference.json").write_text(
        json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "synthetic_smoke_evaluation.json").write_text(
        json.dumps(model.evaluation_summary(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    write_evaluation_artifacts(
        model.evaluation_summary(), REPORTS, "synthetic_smoke")
    print(json.dumps({
        "artifact": str(args.artifact),
        "data_kind": "synthetic_smoke_only",
        "example": example,
    }, ensure_ascii=False, indent=2))


def evaluate_all(args) -> None:
    model = OwnerAssetRatioPipeline.load(
        args.artifact, allow_synthetic=args.allow_synthetic)
    summary = model.evaluation_summary()
    prefix = (
        "actual" if summary["data_kind"] == "actual"
        else "synthetic_smoke")
    write_evaluation_artifacts(summary, REPORTS, prefix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def infer(args) -> None:
    model = OwnerAssetRatioPipeline.load(
        args.artifact, allow_synthetic=args.allow_synthetic)
    if args.building_csv:
        frame = pd.read_csv(args.building_csv, low_memory=False)
    else:
        frame = pd.read_csv(PROCESSED / "buildings.csv", low_memory=False)
    matches = frame[
        frame.get("building_id", pd.Series("", index=frame.index)).astype(str)
        == str(args.building_id)]
    if matches.empty:
        raise SystemExit(f"building_id not found: {args.building_id}")
    result = model.infer(
        matches.iloc[0].to_dict(), samples=args.samples, seed=args.seed,
        occupancy_scenario=args.occupancy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suwon deposit/owner-assets probabilistic pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    buildings = sub.add_parser("collect-buildings")
    buildings.add_argument("--region", default="suwon", choices=["suwon"])
    buildings.add_argument(
        "--legal-dong-codes",
        default=str(
            ROOT / "data" / "downloaded" / "real_estate"
            / "national_legal_dong_codes_20260630.csv"))
    buildings.set_defaults(func=collect_buildings)

    for command, kind in (
        ("collect-leases", "lease"), ("collect-sales", "sale")):
        target = sub.add_parser(command)
        target.add_argument("--region", default="suwon", choices=["suwon"])
        target.add_argument("--start-month", default="202407")
        target.add_argument("--end-month", default="202607")
        target.add_argument("--num-rows", type=int, default=1000)
        target.add_argument("--sleep-sec", type=float, default=.05)
        target.add_argument("--timeout", type=int, default=30)
        target.set_defaults(func=lambda a, k=kind: _download_rtms(k, a))

    pre = sub.add_parser("preprocess")
    pre.set_defaults(func=preprocess)
    unit = sub.add_parser("train-unit-model")
    unit.set_defaults(func=train_unit)
    deposit = sub.add_parser("train-deposit-model")
    deposit.set_defaults(func=train_deposit)
    value = sub.add_parser("train-value-model")
    value.set_defaults(func=train_value)

    def survey_args(target):
        target.add_argument("--survey", required=True)
        target.add_argument("--survey-mapping", required=True)

    prior = sub.add_parser("build-owner-prior")
    survey_args(prior)
    prior.set_defaults(func=build_owner_prior)
    all_parser = sub.add_parser("train-all")
    survey_args(all_parser)
    all_parser.add_argument(
        "--artifact", type=Path,
        default=ARTIFACTS / "owner_asset_ratio_actual.joblib")
    all_parser.add_argument("--seed", type=int, default=20260728)
    all_parser.set_defaults(func=train_all)

    smoke = sub.add_parser("smoke-train")
    smoke.add_argument(
        "--artifact", type=Path,
        default=ARTIFACTS / "owner_asset_ratio_synthetic_smoke.joblib")
    smoke.add_argument("--samples", type=int, default=20_000)
    smoke.add_argument("--seed", type=int, default=20260728)
    smoke.set_defaults(func=smoke_train)

    evaluation = sub.add_parser("evaluate-all")
    evaluation.add_argument(
        "--artifact", type=Path,
        default=ARTIFACTS / "owner_asset_ratio_actual.joblib")
    evaluation.add_argument("--allow-synthetic", action="store_true")
    evaluation.set_defaults(func=evaluate_all)

    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--building-id", required=True)
    infer_parser.add_argument("--building-csv")
    infer_parser.add_argument(
        "--artifact", type=Path,
        default=ARTIFACTS / "owner_asset_ratio_actual.joblib")
    infer_parser.add_argument("--samples", type=int, default=20_000)
    infer_parser.add_argument("--seed", type=int, default=20260728)
    infer_parser.add_argument(
        "--occupancy", choices=["low", "baseline", "high"],
        default="baseline")
    infer_parser.add_argument("--allow-synthetic", action="store_true")
    infer_parser.set_defaults(func=infer)
    return parser


def main(argv: list[str] | None = None) -> int:
    _mkdirs()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0
