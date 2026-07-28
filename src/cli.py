"""Project-level probabilistic real-estate model command line interface.

Owner-asset-ratio commands remain backward compatible. Senior-deposit commands
are routed to a separate package so legal-seniority scenarios cannot be
confused with the owner-asset model.
"""
from __future__ import annotations

import sys


SENIOR_COMMANDS = {
    "audit-data-sources",
    "collect-rent-transactions",
    "import-labels",
    "train-unit-count",
    "train-occupancy",
    "train-deposit",
    "train-seniority",
    "train-calibrator",
    "train-all-senior",
    "evaluate",
}

JEONSE_RATIO_COMMANDS = {
    "inspect-input-models", "calculate", "sensitivity", "stress-test"
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else ""
    if command in JEONSE_RATIO_COMMANDS:
        from src.jeonse_ratio.cli import main as ratio_main
        return ratio_main(arguments)
    if command == "train-all-senior":
        arguments[0] = "train-all"
        from src.senior_deposit.cli import main as senior_main
        return senior_main(arguments)
    if (command in SENIOR_COMMANDS
            or (command == "infer" and "--reference-date" in arguments)):
        from src.senior_deposit.cli import main as senior_main
        return senior_main(arguments)
    from src.owner_asset_ratio.cli import main as owner_main
    return owner_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
