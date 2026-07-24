"""
EasyCodefPy 기반 CODEF 연동 모듈.

기본 모드는 ``ServiceType.SANDBOX``다. EasyCodefPy는 샌드박스 자격증명을
라이브러리 내부에 포함하고 있으며, 샌드박스는 필수 파라미터 검사 후 상품별 고정
테스트 응답을 돌려준다. 따라서 실제 사용자/금융 데이터를 요청하지 않고도 CODEF
요청·토큰·응답 파싱 흐름을 검증할 수 있다.

``CODEF_SERVICE_TYPE=demo|product``로 바꾸고 환경변수 자격증명을 제공하면 같은 코드가
데모/운영으로 전환된다. 패키지 미설치·샌드박스 미지원 상품·네트워크 오류 시에는
랜덤 mock이 아니라 스키마가 고정된 ``codef_sandbox_fixture``를 반환한다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


REGISTER_PATH = "/v1/kr/public/ck/real-estate-register/status"
ACTUAL_PRICE_PATH = "/v1/kr/public/lt/actual-transaction-price-{building_type}/list"
ACCOUNT_PATH = "/v1/kr/bank/p/account/account-list"


@dataclass
class CodefConfig:
    service_type: str = "sandbox"
    client_id: str = ""
    client_secret: str = ""
    public_key: str = "sandbox-public-key"

    @classmethod
    def from_env(cls):
        return cls(
            service_type=os.environ.get("CODEF_SERVICE_TYPE", "sandbox").lower(),
            client_id=os.environ.get("CODEF_CLIENT_ID", ""),
            client_secret=os.environ.get("CODEF_CLIENT_SECRET", ""),
            public_key=os.environ.get("CODEF_PUBLIC_KEY", "sandbox-public-key"),
        )


def _walk_dicts(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _number(value) -> float:
    text = str(value or "0").replace(",", "").replace("원", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


class CodefClient:
    def __init__(self, cfg: CodefConfig | None = None, codef=None):
        self.cfg = cfg or CodefConfig.from_env()
        self._codef = codef
        self._service_type = None
        self._init_error: Exception | None = None
        try:
            from easycodefpy import Codef, ServiceType
            if self._codef is None:
                self._codef = Codef()
            self._codef.public_key = self.cfg.public_key or "sandbox-public-key"
            if self.cfg.service_type == "product":
                self._service_type = ServiceType.PRODUCT
                self._codef.set_client_info(self.cfg.client_id, self.cfg.client_secret)
            elif self.cfg.service_type == "demo":
                self._service_type = ServiceType.DEMO
                self._codef.set_demo_client_info(self.cfg.client_id, self.cfg.client_secret)
            else:
                self._service_type = ServiceType.SANDBOX
        except Exception as exc:
            self._init_error = exc
        self.online = self._codef is not None and self._service_type is not None

    @property
    def mode(self) -> str:
        return self.cfg.service_type if self.online else "sandbox_fixture"

    def _request(self, path: str, body: dict) -> dict:
        if not self.online:
            raise RuntimeError(f"easycodefpy unavailable: {self._init_error}")
        response = self._codef.request_product(path, self._service_type, body)
        payload = json.loads(response) if isinstance(response, str) else response
        if not isinstance(payload, dict):
            raise RuntimeError("CODEF 응답이 JSON 객체가 아닙니다.")
        result = payload.get("result", {})
        if result.get("code") != "CF-00000":
            raise RuntimeError(
                f"CODEF {result.get('code', 'unknown')}: "
                f"{result.get('message') or result.get('extraMessage', '')}"
            )
        return payload

    @staticmethod
    def _fixture_register(address: str, note: str = "") -> dict:
        # 샌드박스가 상품별 고정 응답을 주는 성격에 맞춰 값은 고정한다.
        return {
            "owner": "샌드박스 소유자",
            "senior_mortgage_manwon": 12000.0,
            "has_trust": False,
            "has_seizure": False,
            "mortgage_set_date": "20230115",
            "source": "codef_sandbox_fixture",
            "note": note or "fixed test response",
        }

    @staticmethod
    def _fixture_accounts(note: str = "") -> dict:
        # EasyCodefPy README의 개인 보유계좌 샌드박스 예제 잔액 구조와 동일한 테스트 값.
        balances = [874890, 0, 13110000]
        return {
            "total_asset_manwon": round(sum(balances) / 10000.0, 1),
            "accounts": len(balances),
            "source": "codef_sandbox_fixture",
            "note": note or "fixed test response",
        }

    def real_estate_register(self, address: str, **kwargs) -> dict:
        body = {"organization": "0002", "addr": address, **kwargs}
        try:
            raw = self._request(REGISTER_PATH, body)
            parsed = self._parse_register(raw)
            parsed["source"] = f"codef_{self.cfg.service_type}"
            return parsed
        except Exception as exc:
            return self._fixture_register(address, f"fallback:{type(exc).__name__}")

    def _parse_register(self, raw: dict) -> dict:
        data = raw.get("data", {})
        dicts = list(_walk_dicts(data))
        owner = None
        mortgage_total = 0.0
        has_trust = has_seizure = False
        mortgage_date = None
        for item in dicts:
            owner = owner or item.get("resUserNm") or item.get("resOwnerNm")
            description = " ".join(str(item.get(key, "")) for key in
                                   ("resType", "resContents", "resRegistrationType"))
            if "근저당" in description:
                amount = item.get("resDebtMaxAmt") or item.get("resClaimAmount")
                mortgage_total += _number(amount) / 10000.0
                mortgage_date = mortgage_date or item.get("resRegistrationDate")
            has_trust = has_trust or "신탁" in description
            has_seizure = has_seizure or "압류" in description or "가압류" in description
        return {
            "owner": owner,
            "senior_mortgage_manwon": round(mortgage_total, 1),
            "has_trust": has_trust,
            "has_seizure": has_seizure,
            "mortgage_set_date": mortgage_date,
        }

    def actual_transaction_price(self, sigungu_code: str, deal_ym: str,
                                 building_type: str = "apartment") -> dict:
        path = ACTUAL_PRICE_PATH.format(building_type=building_type)
        try:
            raw = self._request(path, {"lawdCd": sigungu_code, "dealYmd": deal_ym})
            data = raw.get("data", [])
            deals = data if isinstance(data, list) else data.get("resList", [])
            return {"deals": deals, "source": f"codef_{self.cfg.service_type}"}
        except Exception as exc:
            return {
                "deals": [{
                    "resDealYmd": "20250115", "resArea": "59.9",
                    "resDealAmount": "500000000", "resAddress": "샌드박스 주소",
                }],
                "source": "codef_sandbox_fixture",
                "note": f"fallback:{type(exc).__name__}",
            }

    def personal_accounts(self, connected_id: str = "sandbox-connected-id",
                          organization: str = "0004") -> dict:
        try:
            raw = self._request(ACCOUNT_PATH, {
                "connectedId": connected_id,
                "organization": organization,
            })
            data = raw.get("data", {})
            accounts = data.get("resDepositTrust", [])
            total = sum(_number(account.get("resAccountBalance")) for account in accounts)
            return {
                "total_asset_manwon": round(total / 10000.0, 1),
                "accounts": len(accounts),
                "source": f"codef_{self.cfg.service_type}",
            }
        except Exception as exc:
            return self._fixture_accounts(f"fallback:{type(exc).__name__}")


def collect_registry_batch(addresses: list[str], out_csv: str) -> str:
    """주소별 샌드박스/데모/운영 등기 결과를 모델 입력 CSV로 저장한다."""
    import csv
    client = CodefClient()
    rows = [{"address": address, **client.real_estate_register(address)}
            for address in addresses]
    fieldnames = list(rows[0]) if rows else [
        "address", "owner", "senior_mortgage_manwon", "has_trust",
        "has_seizure", "mortgage_set_date", "source", "note",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


if __name__ == "__main__":
    client = CodefClient()
    print(f"[CODEF] mode={client.mode}")
    print(client.real_estate_register("서울 관악구 신림동 테스트 주소"))
    print(client.personal_accounts())
