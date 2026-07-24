"""
백그라운드 스케줄러 — 금융지원 제도 12시간 주기 갱신.

실행:
    pip install apscheduler
    python -m src.server.scheduler

운영에서는 서버 프로세스와 분리해 별도 워커/크론으로 돌리는 것을 권장.
(서버가 여러 워커면 스케줄러 중복 실행 방지 위해 단일 워커/락 필요.)
"""
from __future__ import annotations
import time

from src.tools.finance_tool import FinanceTool


def refresh_job():
    tool = FinanceTool()
    result = tool.refresh()   # 실서비스: 정부/기관 크롤링 → upsert
    print(f"[scheduler] finance refresh: {result}")


def main():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        raise SystemExit("pip install apscheduler 필요")
    sched = BackgroundScheduler()
    sched.add_job(refresh_job, "interval", hours=12, next_run_time=None)
    sched.start()
    print("[scheduler] 시작. 12시간마다 금융제도 갱신. Ctrl-C로 종료.")
    refresh_job()  # 시작 시 1회
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
        print("[scheduler] 종료.")


if __name__ == "__main__":
    main()
