"""
콘솔 대화형 CLI.

실행:
    python -m src.agent.cli
    # 실제 OpenAI API LLM 사용(config.py 고정 설정):
    JEONSE_LLM=api python -m src.agent.cli

사용자 프로필을 먼저 입력받고, 이후 자유 대화로 추천/질문을 진행한다.
2단계 확인(조건 → '응' → 추천)과 조건 누락 그룹 표시를 콘솔에서 그대로 보여준다.
"""
from __future__ import annotations
import json
import os
import sys

from src.agent.harness import JeonseAgent


def _prompt(msg, default=None, cast=str):
    raw = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if not raw and default is not None:
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return default


def collect_user() -> dict:
    print("=" * 64)
    print(" 청년 다가구주택 금융 도우미 — 프로필 입력")
    print("=" * 64)
    print("(엔터로 기본값 사용)")
    return dict(
        user_id="cli_user",
        age=_prompt("나이", 29, int),
        monthly_income_manwon=_prompt("월 소득(만원)", 300, float),
        total_asset_manwon=_prompt("총 자산(만원)", 6000, float),
        monthly_living_cost_manwon=_prompt("월 생활비(만원)", 120, float),
        income_decile=_prompt("소득분위(1~10, 모르면 5)", 5, int),
        preferred_sido=_prompt("선호 지역(시도 또는 시군구, 없으면 엔터)", "", str) or None,
    )


def render(resp: dict):
    st = resp["status"]
    if st == "clarify":
        print(f"\n🤖 {resp['message']}")
    elif st == "cancelled":
        print(f"\n🤖 {resp['message']}")
    elif st == "confirm":
        print(f"\n🤖 {resp['message']}")
        print("   ── 확인된 조건 ──")
        for c in resp["confirmed_conditions"]:
            print(f"     • {c}")
        aff = resp["affordability"]
        print(f"   (참고) 적정 전세보증금 ~{aff['recommended_jeonse_deposit_manwon']:.0f}만 / "
              f"월주거비 상한 ~{aff['max_monthly_housing_manwon']:.0f}만")
    elif st == "recommendation":
        _render_primary_answer(
            resp, resp.get("message", "조건에 맞춰 추천 매물을 조회했습니다."))
        _render_reco(resp)
    elif st == "no_result":
        _render_primary_answer(resp, resp.get("message", "조건에 맞는 결과가 없습니다."))
    elif st == "qa":
        _render_primary_answer(resp, _qa_fallback_answer(resp))
        _render_qa(resp)
    else:
        _render_primary_answer(resp, f"[{st}] {resp.get('message', '')}")
    if os.environ.get("JEONSE_SHOW_RAG_TRACE", "1") != "0":
        _render_rag_trace(resp.get("agent_trace"))


def _render_primary_answer(resp: dict, fallback: str) -> None:
    """LLM 합성문을 주 답변으로 표시하고, 없을 때만 안전한 기본 문장을 쓴다."""
    answer = resp.get("answer") or fallback
    print("\n🤖 AI 상담 답변")
    print(f"   {answer}")


def _qa_fallback_answer(resp: dict) -> str:
    """Mock/LLM 장애 시 구조화 결과 앞에 붙일 최소 답변."""
    labels = {
        "finance": resp.get("message", "금융서비스를 조회했습니다."),
        "affordability": "소득과 자산을 기준으로 감당 가능한 주거 예산을 계산했습니다.",
        "contract": "계약 전에 확인할 항목과 권장 특약을 정리했습니다.",
        "lease_compare": "전세와 월세의 월 실부담을 비교했습니다.",
        "cost": "보증금과 부대비용을 포함한 월 실부담을 계산했습니다.",
        "poi": "요청한 주변 시설을 조회했습니다.",
        "market": resp.get("message", "시세 정보를 확인했습니다."),
        "registry": "등기부 확인 방법과 위험 신호를 정리했습니다.",
        "safety": "주변 안전 데이터를 조회했습니다.",
        "convenience": "주변 생활편의 데이터를 조회했습니다.",
    }
    return labels.get(resp.get("qa_type"), resp.get("message", "조회 결과입니다."))


def _render_rag_trace(trace: dict | None):
    """Planner부터 SQL·검증·조회·폴백·합성까지 원본 trace를 표시한다."""
    if not trace:
        return
    print("\n" + "=" * 72)
    print("RAG DEBUG TRACE (API 키·시스템 프롬프트 제외)")
    print("=" * 72)
    print(json.dumps(trace, ensure_ascii=False, indent=2, default=str))
    print("=" * 72)


def _render_reco(resp):
    labels = {0: "✅ 모든 조건 만족", 1: "🟡 조건 1개 양보", 2: "🟠 조건 2개 양보"}
    financing = resp.get("financing_plan")
    if financing:
        print("\n   ── 목표 달성 계산 · 전세 조달 계획 ──")
        print(f"   자기자금 적정예산 {financing['base_jeonse_budget_manwon']:,.0f}만원"
              f" + 직접 전세자금 {financing['direct_loan_limit_manwon']:,.0f}만원"
              f" = 추정 최대 보증금 {financing['estimated_max_deposit_manwon']:,.0f}만원")
        if financing.get("selected_program_name"):
            print(f"   선택 금융상품: {financing['selected_program_name']}")
        if financing.get("limitation"):
            print(f"   제한: {financing['limitation']}")
        programs = resp.get("finance_programs", [])
        print(f"\n   ── 조회 근거 · 목표 관련 금융서비스 DB {len(programs)}건 ──")
        _render_finance_rows(programs)

    count = sum(len(items) for items in resp.get("groups", {}).values())
    print(f"\n   ── 조회 근거 · 부동산 DB {count}건 ──")
    if resp.get("commute_regions"):
        regs = ", ".join(f"{r['gugun']}({r['minutes']:.0f}분)"
                         for r in resp["commute_regions"][:5])
        print(f"   통근 후보 지역: {regs}")
    for k in sorted(resp["groups"]):
        recs = resp["groups"][k]
        if not recs:
            continue
        print(f"\n   {labels.get(k, f'{k}개 양보')}")
        for i, r in enumerate(recs, 1):
            fs = r["fraud_score"]
            fs_s = f"전세사기 추정 위험도 {fs:.2f}" if fs is not None else "위험도 N/A"
            if r["lease_type"] == "매매":
                price = f"매매가 {r.get('sale_price_manwon', 0):,.0f}만"
            else:
                price = f"보증금 {r['deposit_manwon']:,.0f}만"
                if r["lease_type"] == "월세":
                    price += f" 월세 {r['monthly_rent_manwon']:,.0f}만"
            miss = f"  ← 양보: {', '.join(r['missing_conditions'])}" if r["missing_conditions"] else ""
            print(f"     {i}. {r['sido']} {r['gugun']} {r['lease_type']} "
                  f"{price} | {fs_s} | 합성 매물{miss}")


def _render_qa(resp):
    qt = resp.get("qa_type")
    if qt == "finance":
        print(f"\n   ── 조회 근거 · 금융서비스 DB {len(resp['programs'])}건 ──")
        _render_finance_rows(resp["programs"])
    elif qt == "affordability":
        aff = resp["affordability"]
        print("\n   ── 계산 근거 · 주거 예산 ──")
        print(f"   적정 전세보증금: ~{aff['recommended_jeonse_deposit_manwon']:.0f}만원")
        print(f"   월주거비 상한: ~{aff['max_monthly_housing_manwon']:.0f}만원 "
              f"(RIR {aff['rir_at_recommended']:.0%})")
        for n in aff.get("notes", []):
            print(f"   - {n}")
    elif qt == "contract":
        cc = resp["checklist"]
        print("\n   ── 조회 근거 · 계약 체크리스트 ──")
        print("   [계약 전]")
        for x in cc["before_contract"]:
            print(f"     □ {x}")
        print("   [권장 특약]")
        for x in cc["recommended_special_terms"]:
            print(f"     ✎ {x}")
    elif qt == "lease_compare":
        c = resp["comparison"]
        print("\n   ── 계산 근거 · 전세와 월세 비교 ──")
        print(f"   전세(대출 시): {c['jeonse_monthly_if_loan']}만/월")
        print(f"   전세(자기자본): {c['jeonse_monthly_if_own_capital']}만/월")
        print(f"   월세 총부담: {c['wolse_monthly_total']}만/월")
        print(f"   → 대출 기준 더 유리: {c['cheaper_option_if_loan']}")
    elif qt == "cost":
        b = resp["breakdown"]
        print("\n   ── 계산 근거 · 월 실부담금 ──")
        print(f"   월세 {b['monthly_rent']} + 관리비 {b['maintenance']} + "
              f"보증금기회비용 {b['deposit_opportunity_cost_monthly']} + "
              f"일회성분할 {b['onetime_amortized_monthly']}")
        print(f"   = 총 {b['total_monthly_real_cost']}만원/월")
        print(f"   {resp.get('note','')}")
    elif qt == "poi":
        r = resp["result"]
        print(f"\n   ── 조회 근거 · 주변 {r['category']} {r['count']}개 ──")
        print(f"   최근접 거리: {r['nearest_m']}m")
        for p in r["places"]:
            print(f"   • {p['name']} ({p['distance_m']}m)")
        print(f"   {resp.get('note','')}")
    elif qt == "market":
        print("\n   ── 조회 근거 · 시세 데이터 ──")
        ex = resp["example"]
        print(f"   예시: {ex.get('message','')}")
    elif qt == "registry":
        g = resp["guide"]
        print("\n   ── 조회 근거 · 등기부 확인 항목 ──")
        print(f"   {g['message']}")
        for x in g["how_to"]:
            print(f"   □ {x}")
        print("   ⚠ 위험신호:")
        for x in g["danger_signals"]:
            print(f"     • {x}")
    elif qt == "safety":
        r = resp["result"]
        print("\n   ── 조회 근거 · 안전 데이터 ──")
        print(f"   주변 {r['radius_m']}m 치안 안전도: {r['safety_score']}점 ({r['grade']})")
        for k, v in r["counts"].items():
            print(f"   • {r['detail_ko'][k]}: {v}개")
        print(f"   {resp.get('note','')}")
    elif qt == "convenience":
        r = resp["result"]
        print("\n   ── 조회 근거 · 생활편의 데이터 ──")
        print(f"   주변 {r['radius_m']}m 생활편의도: {r['convenience_score']}점 ({r['grade']})")
        for k, v in r["counts"].items():
            mark = "✓" if v > 0 else "✗"
            print(f"   {mark} {r['detail_ko'][k]}: {v}개")
        print(f"   {resp.get('note','')}")
    else:
        print(f"\n   ── 조회 근거 ──\n   {resp.get('message', '')}")


def _render_finance_rows(programs: list[dict]) -> None:
    if not programs:
        print("   조회된 관련 금융상품이 없습니다.")
        return
    for p in programs:
        rate = (f"금리 {p['rate_pct']:g}%" if p.get("rate_pct") is not None
                else "금리 해당 없음/별도 확인")
        amount = (f"한도·지원액 {p['max_amount_manwon']:,.0f}만원"
                  if p.get("max_amount_manwon") else "한도·지원액 별도 확인")
        region = p.get("region_scope") or "지역 상세 확인"
        status = p.get("application_status") or "일정 상세 확인"
        role = f" | {p['goal_role']}" if p.get("goal_role") else ""
        print(f"   • {p['name']} | {p.get('category', '')} | {region}{role}")
        print(f"     {rate} | {amount} | 신청 {status}")
        if p.get("source_url"):
            print(f"     → {p['source_url']}")


def setup_llm_key():
    """고정 OpenAI 설정을 기본으로 사용하고, 명시적 환경설정만 우선한다."""
    import os
    from src import config
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return  # 이미 환경변수로 설정됨
    if config.OPENAI_API_KEY:
        os.environ.setdefault("LLM_PROVIDER", "openai")
        os.environ.setdefault("JEONSE_LLM", "api")
        return
    print("\n[LLM 설정] API 키를 입력하면 실제 LLM으로, 비우면 규칙기반(Mock)으로 동작합니다.")
    provider = input("  Provider (anthropic/openai, 기본 anthropic, 건너뛰려면 엔터): ").strip().lower()
    if provider in ("anthropic", "openai"):
        os.environ["LLM_PROVIDER"] = provider
        key = input(f"  {provider.upper()} API Key: ").strip()
        if key:
            os.environ["ANTHROPIC_API_KEY" if provider == "anthropic"
                       else "OPENAI_API_KEY"] = key
            os.environ["JEONSE_LLM"] = "api"
        else:
            os.environ["JEONSE_LLM"] = "mock"
    else:
        os.environ["JEONSE_LLM"] = "mock"


def main():
    setup_llm_key()
    user = collect_user()
    try:
        agent = JeonseAgent(recommender_name="rule")
    except Exception as exc:
        print("\n[FATAL] 실제 LLM 초기화에 실패했습니다. 규칙 기반으로 숨겨서 실행하지 않습니다.")
        print(f"  {type(exc).__name__}: {exc}")
        print("  OPENAI 설정과 openai 패키지를 확인한 뒤 다시 실행하세요.")
        raise SystemExit(2) from exc
    print("\n[Agent Runtime]")
    print(f"  LLM class : {type(agent.llm).__name__}")
    print(f"  Provider  : {getattr(agent.llm, 'provider', 'offline/rule')}")
    print(f"  Model     : {getattr(agent.llm, 'model', 'none')}")
    print(f"  Text2SQL  : {'enabled' if agent.llm.supports_agentic_calls else 'fallback only'}")
    print(f"  RAG trace : {'visible' if os.environ.get('JEONSE_SHOW_RAG_TRACE', '1') != '0' else 'hidden'}")
    session = agent.new_session(user)
    print("\n무엇이든 물어보세요. (예: '서울 안전한 전세 5천 이내', "
          "'대출 받을 수 있어?', '전세 계약 특약 뭐 넣어?')")
    print("종료: 'quit' 또는 Ctrl-C\n")
    while True:
        try:
            text = input("🙂 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n이용해 주셔서 감사합니다.")
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "종료", "그만"):
            print("이용해 주셔서 감사합니다.")
            break
        try:
            resp = agent.handle(session, text)
            render(resp)
        except Exception as e:
            print(f"\n[오류] {type(e).__name__}: {e}")
            print("다시 시도해 주세요.")
        print()


if __name__ == "__main__":
    main()
