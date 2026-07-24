"""Canonical synthetic broker-listing schema.

The schema combines the project's existing risk/recommendation fields with the
items required for a Korean residential ``중개대상물 확인·설명서`` and the
mandatory housing advertisement fields.  It intentionally stores synthetic
broker identities and masked/synthetic addresses; generated rows must never be
presented as real listings.
"""
from __future__ import annotations


SCHEMA_GROUPS: dict[str, list[str]] = {
    "identity_and_provenance": [
        "property_id", "listing_id", "is_synthetic", "synthetic_notice",
        "source_type", "generator_version", "generated_at",
        "source_dataset", "source_record_hash", "source_transaction_type",
        "generation_method", "price_estimation_method", "privacy_distance_score",
        "region_assignment_method", "regional_price_factor",
        "price_diversity_factor", "region_coordinate_source",
        "coordinate_distribution_method",
        "price_model_name", "price_model_holdout_mdape_pct",
        "price_model_holdout_r2",
        "base_rent_record_hash", "base_trade_record_hash",
        "reference_rent_deal_ym", "reference_trade_deal_ym",
        "listing_status", "listing_created_at", "listing_updated_at",
    ],
    "location_and_land_ledger": [
        "sido", "gugun", "dong", "legal_dong_code", "road_address",
        "jibun_address", "address_detail_public", "lat", "lng",
        "lot_main_no", "lot_sub_no", "road_name", "building_main_no",
        "building_sub_no",
        "land_category", "land_area_m2", "land_share_m2",
        "zoning", "land_use_status", "road_access", "land_transaction_permit_zone",
    ],
    "transaction_and_price": [
        "transaction_type", "lease_type", "asking_price_manwon",
        "sale_price_manwon", "sale_subtype", "dealing_type",
        "deposit_manwon", "monthly_rent_manwon", "maintenance_fee_manwon",
        "maintenance_fee_items", "maintenance_fee_other", "price_negotiable",
        "rent_conversion_rate_pct", "move_in_negotiable",
        "contract_type", "contract_term", "use_renewal_right",
        "available_from_date", "occupancy_status", "onetime_fee_manwon",
        "broker_fee_rate_pct", "broker_fee_manwon", "broker_fee_vat_manwon",
        "actual_expense_manwon",
    ],
    "building_ledger_and_unit": [
        "property_type", "house_type", "building_name", "building_use",
        "building_dong", "unit_number_public", "unit_type",
        "building_structure", "approval_date", "build_year",
        "building_age_years", "building_total_area_m2", "building_coverage_ratio_pct",
        "floor_area_ratio_pct", "current_floor", "total_floors", "basement_floors",
        "area_m2", "exclusive_area_m2", "supply_area_m2", "contract_area_m2",
        "room_count", "bathroom_count", "direction", "direction_basis",
        "entrance_type", "building_total_units", "building_total_households",
        "total_complex_buildings", "balcony_expansion", "duplex", "terrace",
        "yard", "rooftop_access", "ceiling_height_m",
    ],
    "facilities_and_condition": [
        "parking_total", "parking_per_household", "parking_method", "elevator_count",
        "heating_method", "heating_fuel", "cooling_facility", "aircon_count",
        "built_in_appliances", "furnished", "pet_allowed", "loan_available",
        "water_supply", "electricity_supply", "gas_supply", "drainage",
        "fire_safety_facility", "security_facility", "accessibility_facility",
        "wall_crack", "water_leak", "wallpaper_condition", "noise_level",
        "floor_condition", "vibration_level", "sunlight_level", "renovation_status", "illegal_building",
        "ledger_discrepancy", "violation_details",
    ],
    "rights_and_safety": [
        "ownership_type", "owner_relation", "trust_registration",
        "seizure_or_provisional_seizure", "easement", "leasehold_registration",
        "tenant_right_registration", "tax_arrears_checked", "tax_arrears_present",
        "landlord_information_presented", "resident_household_certificate_checked",
        "small_deposit_priority_protection_explained", "private_rental_housing",
        "rental_deposit_guarantee_joined", "rental_deposit_guarantee_details",
        "market_price_manwon", "official_land_price_manwon_m2",
        "official_building_price_manwon", "building_total_units",
        "registered_owner_type", "mortgage_max_claim_manwon",
        "senior_rights_total_manwon", "registry_checked_at",
        "building_ledger_checked_at", "deposit_return_guarantee_provider",
        "my_priority_rank", "senior_tenant_count", "senior_deposit_sum_manwon",
        "senior_mortgage_manwon", "mortgage_ltv_pct", "jeonse_ratio_pct",
        "guarantee_eligible", "guarantee_ineligible_reason",
        "acquisition_tax_type", "estimated_acquisition_tax_rate_pct",
        "fraud_label", "fraud_score",
    ],
    "environment_and_access": [
        "subway_walk_minutes", "bus_stop_walk_minutes", "school_walk_minutes",
        "mart_walk_minutes", "hospital_walk_minutes", "park_walk_minutes",
        "noise_source", "odor_source", "flood_risk_level", "nonpreferred_facility",
    ],
    "broker_and_advertisement": [
        "advertisement_medium", "advertisement_title", "advertisement_description",
        "broker_office_name", "broker_registration_no", "broker_representative_name",
        "broker_agent_name", "broker_office_address", "broker_phone",
        "broker_guarantee_type", "broker_guarantee_amount_manwon",
        "broker_guarantee_period", "joint_brokerage", "advertisement_confirmed_at",
        "photo_count", "video_present", "virtual_tour_present",
        "viewing_available", "viewing_method", "exclusive_listing",
    ],
    "explanation_evidence": [
        "evidence_title_deed", "evidence_registry", "evidence_land_ledger",
        "evidence_building_ledger", "evidence_cadastral_map",
        "evidence_land_use_plan", "evidence_owner_request",
        "explanation_completed", "explanation_notes",
    ],
}


def _unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


BROKER_LISTING_COLUMNS = _unique_in_order(
    [column for group in SCHEMA_GROUPS.values() for column in group]
)


# Existing models and tools depend on these names.  Keeping this contract makes
# the richer listing table backwards compatible with the original project.
LEGACY_REQUIRED_COLUMNS = [
    "property_id", "sido", "gugun", "lat", "lng", "lease_type",
    "deposit_manwon", "monthly_rent_manwon", "maintenance_fee_manwon",
    "onetime_fee_manwon", "market_price_manwon", "building_total_units",
    "my_priority_rank", "senior_deposit_sum_manwon", "senior_mortgage_manwon",
    "building_age_years", "area_m2", "fraud_label", "fraud_score",
]


def missing_schema_columns(columns) -> list[str]:
    """Return canonical columns absent from a generated dataframe."""
    existing = set(columns)
    return [c for c in BROKER_LISTING_COLUMNS if c not in existing]
