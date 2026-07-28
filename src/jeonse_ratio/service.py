"""Application-facing integration service."""
from __future__ import annotations

from .adapters import DepositModelAdapter, PropertyValueModelAdapter
from .engine import JeonseRatioEngine


class JeonseRatioIntegrationService:
    def __init__(self):
        self.deposit_adapter = DepositModelAdapter()
        self.value_adapter = PropertyValueModelAdapter()
        self.engine = JeonseRatioEngine()

    def calculate(
        self,
        prop: dict,
        senior_result: dict,
        owner_asset_result: dict,
        *,
        samples: int = 20_000,
        seed: int = 42,
        dependence: str = "independence",
    ) -> dict:
        if (
            str(prop.get("transaction_type") or "") != "전세"
            or "다가구" not in str(prop.get("house_type") or "")
        ):
            return {
                "available": False,
                "status": "not_applicable",
                "applicability": "다가구주택 전세 매물에만 적용합니다.",
            }
        try:
            deposit = self.deposit_adapter.adapt(senior_result)
            value = self.value_adapter.adapt(owner_asset_result)
            result = self.engine.calculate(
                deposit,
                value,
                my_deposit_manwon=float(prop.get("deposit_manwon") or 0),
                samples=samples,
                seed=seed,
                dependence=dependence,
            )
            return {
                "available": True,
                "status": "estimated",
                **result,
            }
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "available": False,
                "status": "input_contract_failed",
                "applicability": str(exc),
            }
