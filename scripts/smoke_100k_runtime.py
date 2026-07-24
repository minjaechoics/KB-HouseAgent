"""Read-only smoke test for the deployed 100k-property map database."""
from __future__ import annotations

import json
import time

from src.server.app import (
    PropertySearchIn,
    SessionCreate,
    create_session,
    search_properties,
)


def main() -> None:
    started = time.perf_counter()
    empty = create_session(SessionCreate())
    assert empty["atoms"] == [], "blank setup must not create catch-all filters"
    nationwide = search_properties(PropertySearchIn(
        session_id=empty["session_id"], enabled_atom_ids=[], limit=120,
    ))
    nationwide_seconds = time.perf_counter() - started
    assert nationwide["total"] == 100_000, nationwide["total"]
    assert len(nationwide["properties"]) == 120

    started = time.perf_counter()
    regional = create_session(SessionCreate(
        preferred_sido="대전", preferred_gugun="유성구",
        transaction_types=["전세"], house_types=["아파트"],
    ))
    regional_result = search_properties(PropertySearchIn(
        session_id=regional["session_id"],
        enabled_atom_ids=[atom["id"] for atom in regional["atoms"]],
        limit=120,
    ))
    regional_seconds = time.perf_counter() - started
    assert regional_result["total"] > 0
    assert all(row["sido"] == "대전" and row["gugun"] == "유성구"
               for row in regional_result["properties"])
    assert all(row["transaction_type"] == "전세" and row["house_type"] == "아파트"
               for row in regional_result["properties"])

    print(json.dumps({
        "nationwide_total": nationwide["total"],
        "nationwide_returned": len(nationwide["properties"]),
        "nationwide_seconds": round(nationwide_seconds, 3),
        "regional_total": regional_result["total"],
        "regional_seconds": round(regional_seconds, 3),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
