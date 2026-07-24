"""
실제 KHUG(주택도시보증공사) 데이터를 파싱해 '지역별 위험 앵커'를 만든다.

두 개의 실제 파일을 사용:
  1) 지역별 전세금반환보증 사고현황 (.xlsx)  -> 시군구별 사고건수/사고율
  2) 전세보증금반환보증 발급현황 (.csv)      -> 주택유형(다가구 등)별 발급 추이

이 앵커는 (0) 데이터 증강에서 "지역/주택유형별 사고 확률"의 기준선으로 쓰인다.
합성 데이터의 라벨(전세사기 여부)을 현실 사고율에 정합시키기 위함.
"""
from __future__ import annotations
import re
import pandas as pd
import openpyxl

from src import config


def load_region_accident_stats() -> pd.DataFrame:
    """
    시군구별 사고 통계를 DataFrame으로 반환.
    columns: sido, gugun, accident_count, accident_amount, accident_rate_pct
    accident_rate_pct 는 KHUG 원자료 '사고율(%)'(발급 대비 사고 비율).
    """
    wb = openpyxl.load_workbook(config.RAW_ACCIDENT_XLSX, data_only=True)
    ws = wb["시군구"]
    records = []
    for r in ws.iter_rows(values_only=True):
        code, sido, gugun, cnt, amount, rate = r[0], r[1], r[2], r[3], r[4], r[5]
        # 유효한 '기초지자체' 행만 (소계/합계/헤더 제외)
        if not sido or not gugun:
            continue
        if gugun in ("소계", "기초\n지자체") or sido in ("전국", "수도권", "지방"):
            continue
        if not isinstance(cnt, (int, float)):
            continue
        records.append(
            dict(
                sido=str(sido).strip(),
                gugun=str(gugun).strip(),
                accident_count=int(cnt),
                accident_amount=float(amount) if isinstance(amount, (int, float)) else 0.0,
                accident_rate_pct=float(rate) if isinstance(rate, (int, float)) else 0.0,
            )
        )
    df = pd.DataFrame.from_records(records)
    # 수도권 플래그(문헌상 수도권 주거비/사고 부담이 큼)
    metro = {"서울", "경기", "인천"}
    df["is_metro"] = df["sido"].isin(metro).astype(int)
    return df


def load_issue_trend() -> pd.DataFrame:
    """
    발급현황 CSV -> 주택유형별(특히 '다가구주택') 시계열.
    columns: 기준연월, 구분, 주택, 보증유형, 건수, 금액(백만원), 비율
    """
    # KHUG 원본 CSV는 CP949(EUC-KR) 인코딩인 경우가 많다. UTF-8 우선, 실패 시 CP949.
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(config.RAW_ISSUE_CSV, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("csv", b"", 0, 1, "unknown encoding")
    df.columns = [c.strip() for c in df.columns]
    return df


def multi_family_share() -> float:
    """
    전세보증금반환보증 발급 중 '다가구주택'이 차지하는 비중(건수 기준, 전체 기간 평균).
    합성 데이터에서 다가구 표본 비율의 현실 앵커로 사용.
    """
    df = load_issue_trend()
    jeonse = df[df["보증유형"] == "전세보증금반환보증"]
    total = jeonse["건수"].sum()
    mf = jeonse[jeonse["주택"] == "다가구주택"]["건수"].sum()
    return float(mf / total) if total else 0.0


def region_risk_lookup() -> dict[tuple[str, str], float]:
    """
    (sido, gugun) -> 사고율(0~1 스케일) 딕셔너리.
    합성 라벨 생성 시 지역 baseline hazard로 사용.
    """
    df = load_region_accident_stats()
    return {
        (row.sido, row.gugun): row.accident_rate_pct / 100.0
        for row in df.itertuples()
    }


if __name__ == "__main__":
    df = load_region_accident_stats()
    print(f"[region_stats] 시군구 유효 레코드: {len(df)}")
    print(df.sort_values("accident_rate_pct", ascending=False).head(10).to_string(index=False))
    print(f"\n[region_stats] 다가구주택 발급 비중(건수): {multi_family_share():.3%}")
