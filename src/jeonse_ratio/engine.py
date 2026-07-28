"""Monte Carlo integration of deposit and market-value uncertainty."""
from __future__ import annotations

import numpy as np

from .adapters import DistributionContract
from .distributions import copula_uniforms, inverse_cdf_sample
from .validation import AlignmentPolicy, validate_alignment


RATIO_LEVELS = (.05, .10, .50, .90, .95)
DEPENDENCE_SCENARIOS = {
    "independence": 0.0,
    "weak_positive": 0.2,
    "moderate_positive": 0.4,
    "strong_positive": 0.6,
}


def _summary(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=float), RATIO_LEVELS)
    return {
        f"p{int(level * 100):02d}": round(float(value), 6)
        for level, value in zip(RATIO_LEVELS, q)
    }


def _grade(ratio: float) -> str:
    if ratio >= .8:
        return "위험"
    if ratio >= .7:
        return "주의"
    if ratio <= .6:
        return "안전"
    return "관찰"


class JeonseRatioEngine:
    def __init__(
        self,
        *,
        alignment_policy: AlignmentPolicy = AlignmentPolicy(),
        minimum_property_value_manwon: float = 1_000.0,
        maximum_outlier_fraction: float = .01,
    ):
        self.alignment_policy = alignment_policy
        self.minimum_property_value_manwon = float(
            minimum_property_value_manwon)
        self.maximum_outlier_fraction = float(maximum_outlier_fraction)

    @staticmethod
    def _my_deposit(
        samples: int,
        rng: np.random.Generator,
        fixed: float | None,
        distribution: tuple[float, float, float] | None,
    ) -> np.ndarray | None:
        if fixed is not None:
            if float(fixed) < 0:
                raise ValueError("내 보증금은 음수일 수 없습니다.")
            return np.full(samples, float(fixed))
        if distribution is None:
            return None
        low, mode, high = map(float, distribution)
        if not 0 <= low <= mode <= high:
            raise ValueError("보증금 범위는 min <= mode <= max여야 합니다.")
        return rng.triangular(low, mode, high, samples)

    def _draw(
        self,
        deposit: DistributionContract,
        value: DistributionContract,
        *,
        samples: int,
        seed: int,
        rho: float,
        my_deposit_manwon: float | None,
        my_deposit_distribution: tuple[float, float, float] | None,
    ) -> tuple[dict[str, np.ndarray | None], list[str], dict]:
        rng = np.random.default_rng(int(seed))
        deposit_u, value_u = copula_uniforms(samples, rho, rng)
        warnings: list[str] = []
        crossing_fields = []

        def sample(name: str) -> np.ndarray:
            values, crossing = inverse_cdf_sample(
                deposit.quantiles[name], deposit_u)
            if crossing:
                crossing_fields.append(name)
            return values

        total = sample("total_deposit")
        senior = sample("senior_deposit")
        upper = sample("conservative_upper_deposit")
        property_value, value_crossing = inverse_cdf_sample(
            value.quantiles["property_value"], value_u)
        if value_crossing:
            crossing_fields.append("property_value")
        if np.any(property_value <= 0):
            raise ValueError("건물가치가 0 이하인 샘플이 있습니다.")

        # Preserve the legal/model definitions on every reconstructed draw.
        relation_repairs = int(np.sum(
            (senior > upper) | (upper > total)))
        upper = np.maximum(upper, senior)
        total = np.maximum(total, upper)
        mine = self._my_deposit(
            samples, rng, my_deposit_manwon, my_deposit_distribution)
        small = property_value < self.minimum_property_value_manwon
        if crossing_fields:
            warnings.append(
                "quantile crossing을 단조 누적값으로 보정: "
                + ", ".join(crossing_fields))
        if relation_repairs:
            warnings.append(
                f"복원 sample 중 {relation_repairs}개의 S≤Upper≤All 관계를 보정했습니다.")
        if np.any(small):
            warnings.append(
                f"최소 현실가치 미만 건물가치 sample {int(small.sum())}개를 "
                "제거하지 않고 이상값으로 표시했습니다.")
        return {
            "all": total / property_value,
            "senior": senior / property_value,
            "post": (
                (senior + mine) / property_value
                if mine is not None else None
            ),
            "post_upper": (
                (upper + mine) / property_value
                if mine is not None else None
            ),
            "senior_amount": senior,
            "upper_amount": upper,
            "total_amount": total,
            "property_value": property_value,
            "my_deposit": mine,
        }, warnings, {
            "small_property_value_samples": int(small.sum()),
            "small_property_value_fraction": round(float(np.mean(small)), 6),
            "relation_repairs": relation_repairs,
            "quantile_crossing_fields": crossing_fields,
        }

    def calculate(
        self,
        deposit: DistributionContract,
        value: DistributionContract,
        *,
        my_deposit_manwon: float | None = None,
        my_deposit_distribution: tuple[float, float, float] | None = None,
        samples: int = 20_000,
        seed: int = 42,
        dependence: str = "independence",
        stress_haircuts: tuple[float, ...] = (.05, .10, .20, .30),
    ) -> dict:
        alignment = validate_alignment(
            deposit, value, self.alignment_policy)
        samples = max(1_000, min(100_000, int(samples)))
        if dependence not in DEPENDENCE_SCENARIOS:
            raise ValueError("지원하지 않는 의존성 시나리오입니다.")
        rho = DEPENDENCE_SCENARIOS[dependence]
        draws, warnings, diagnostics = self._draw(
            deposit, value, samples=samples, seed=seed, rho=rho,
            my_deposit_manwon=my_deposit_manwon,
            my_deposit_distribution=my_deposit_distribution,
        )

        ratios = {
            "all_deposit_ratio": _summary(draws["all"]),
            "senior_deposit_ratio": _summary(draws["senior"]),
            "post_contract_ratio": (
                _summary(draws["post"]) if draws["post"] is not None else None),
            "conservative_post_contract_ratio": (
                _summary(draws["post_upper"])
                if draws["post_upper"] is not None else None),
        }
        thresholds = {}
        if draws["post"] is not None:
            for threshold in (.6, .7, .8, .9, 1.0):
                thresholds[
                    f"post_contract_over_{str(threshold).replace('.', '_')}"
                ] = round(float(np.mean(draws["post"] > threshold)), 6)
            thresholds["conservative_over_0_8"] = round(
                float(np.mean(draws["post_upper"] > .8)), 6)
            thresholds["conservative_over_1_0"] = round(
                float(np.mean(draws["post_upper"] > 1.0)), 6)

        sensitivity = {}
        for label, scenario_rho in DEPENDENCE_SCENARIOS.items():
            scenario, _, _ = self._draw(
                deposit, value, samples=samples, seed=seed,
                rho=scenario_rho,
                my_deposit_manwon=my_deposit_manwon,
                my_deposit_distribution=my_deposit_distribution,
            )
            sensitivity[label] = {
                "rho": scenario_rho,
                "post_contract_p50": (
                    _summary(scenario["post"])["p50"]
                    if scenario["post"] is not None else None),
                "post_contract_over_0_8": (
                    round(float(np.mean(scenario["post"] > .8)), 6)
                    if scenario["post"] is not None else None),
            }

        stress = {}
        if draws["post"] is not None:
            for haircut in stress_haircuts:
                if not 0 <= float(haircut) < 1:
                    raise ValueError("가격하락률은 0 이상 1 미만이어야 합니다.")
                stressed = draws["post"] / (1.0 - float(haircut))
                stress[str(float(haircut))] = {
                    "label": f"시장가치 {float(haircut) * 100:.0f}% 하락",
                    "p50": _summary(stressed)["p50"],
                    "p90": _summary(stressed)["p90"],
                    "over_0_8_probability": round(
                        float(np.mean(stressed > .8)), 6),
                }

        # Stable permutation-style variance attribution.
        if draws["post"] is not None:
            base_var = float(np.var(draws["post"]))
            value_fixed = (
                draws["senior_amount"] + draws["my_deposit"]
            ) / np.median(draws["property_value"])
            numerator_fixed = (
                np.median(draws["senior_amount"])
                + np.median(draws["my_deposit"])
            ) / draws["property_value"]
            mine_fixed = (
                draws["senior_amount"] + np.median(draws["my_deposit"])
            ) / draws["property_value"]
            raw = {
                "deposit_model": max(0.0, base_var - float(np.var(numerator_fixed))),
                "property_value_model": max(0.0, base_var - float(np.var(value_fixed))),
                "my_deposit": max(0.0, base_var - float(np.var(mine_fixed))),
            }
            total_raw = sum(raw.values()) or 1.0
            contribution = {
                key: round(value_ / total_raw, 4)
                for key, value_ in raw.items()
            }
            contribution["dependence_assumption"] = round(
                max(item["post_contract_p50"] or 0 for item in sensitivity.values())
                - min(item["post_contract_p50"] or 0 for item in sensitivity.values()),
                4,
            )
        else:
            contribution = {}

        quality_rank = {"low": 0, "medium": 1, "high": 2}
        quality = min(
            (deposit.quality, value.quality),
            key=lambda item: quality_rank.get(item, 0),
        )
        downgrade_reasons = [
            "원시 sample 없이 분위수에서 분포를 복원함",
            "공통 simulation context·paired 잔차가 없어 의존성을 가정함",
        ]
        if deposit.metadata.get("model_mode") == "scenario_only":
            downgrade_reasons.append("선순위 보증금 모델이 scenario-only임")
        if alignment["date_difference_days"]:
            downgrade_reasons.append("두 모델의 기준일이 정확히 같지 않음")
        if diagnostics["small_property_value_fraction"] > \
                self.maximum_outlier_fraction:
            quality = "low"
            downgrade_reasons.append("최소 현실가치 미만 sample 비율이 허용치를 초과")
        elif quality == "high":
            quality = "medium"
        warnings = list(dict.fromkeys([
            *deposit.warnings, *value.warnings, *warnings,
            *downgrade_reasons,
        ]))
        p50 = (
            ratios["post_contract_ratio"]["p50"]
            if ratios["post_contract_ratio"] else None)
        return {
            **alignment,
            "sample_count": samples,
            "seed": int(seed),
            "ratios": ratios,
            "threshold_probabilities": thresholds,
            "risk": {
                "primary_metric": "post_contract_ratio",
                "score": p50,
                "grade": _grade(float(p50)) if p50 is not None else None,
                "safe_threshold": .6,
                "caution_threshold": .7,
                "danger_threshold": .8,
            },
            "inputs": {
                "my_deposit_manwon": my_deposit_manwon,
                "my_deposit_distribution_manwon": my_deposit_distribution,
                "deposit_model_quality": deposit.quality,
                "property_value_model_quality": value.quality,
                "property_value_price_basis": value.metadata.get("price_basis"),
            },
            "dependence": {
                "method": "independent" if rho == 0 else "gaussian_copula",
                "parameter": rho,
                "sensitivity": sensitivity,
                "learned_from_paired_data": False,
            },
            "stress": stress,
            "uncertainty_contribution": contribution,
            "diagnostics": diagnostics,
            "data_quality": quality,
            "warnings": warnings,
            "assumptions": [
                "비율은 1.0에서 자르지 않으며 100%를 초과할 수 있음",
                "근저당은 전세가율에 합산하지 않고 별도 권리부담으로 확인",
                "통합 레이어는 별도 지도학습 모델이 아니며 training loss가 없음",
            ],
            "disclaimer": (
                "이 전세가율은 특정 건물의 모든 실제 임대차계약과 실거래가격을 "
                "직접 확인한 확정값이 아닙니다. 임차보증금 추정분포와 건물 "
                "시장가치 추정분포를 결합한 통계적 추정치입니다. 선순위 "
                "임차보증금 비율은 일반적인 전체 전세가율과 다른 지표입니다."
            ),
        }
