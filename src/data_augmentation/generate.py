"""Backward-compatible entry point for the multi-source listing generator.

Run from the project root::

    py -3 -m src.data_augmentation.generate --count 20000 --n-users 2000
"""
from src.data_augmentation.universal_generate import (
    DEFAULT_HOUSE_WEIGHTS,
    DEFAULT_TRANSACTION_WEIGHTS,
    GENERATOR_VERSION,
    HOUSE_TYPES,
    TRANSACTION_TYPES,
    assign_fraud_labels,
    build_quality_report,
    generate_properties,
    generate_users,
    load_source_pools,
    lognormal_from_mean_std,
    main,
    sigmoid,
    validate_generated_properties,
)

__all__ = [
    "DEFAULT_HOUSE_WEIGHTS", "DEFAULT_TRANSACTION_WEIGHTS", "GENERATOR_VERSION",
    "HOUSE_TYPES", "TRANSACTION_TYPES", "assign_fraud_labels",
    "build_quality_report", "generate_properties", "generate_users",
    "load_source_pools", "lognormal_from_mean_std", "validate_generated_properties",
]


if __name__ == "__main__":
    main()
