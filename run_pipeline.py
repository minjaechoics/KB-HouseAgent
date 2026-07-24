#!/usr/bin/env python3
"""
전체 파이프라인 원커맨드 실행기.

    python run_pipeline.py --n_properties 5000 --n_users 2000

순서:
  (0) 합성 데이터 생성
  (1) 전세사기 위험 모델 실험 + 최종모델 저장
  (4) DB 구축(전세 fraud_score 반영)
  (4) 클릭라벨 생성 + 추천기 실험 + LTR 저장
  통합 스모크 테스트
"""
import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run([sys.executable, "-m"] + cmd)
    if r.returncode != 0:
        print(f"[FAIL] {cmd}")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_properties", type=int, default=5000)
    ap.add_argument("--n_users", type=int, default=2000)
    args = ap.parse_args()

    # (0)
    run(["src.data_augmentation.generate",
         "--n_properties", str(args.n_properties),
         "--n_users", str(args.n_users)])
    # (1)
    run(["src.fraud_risk.train", "--experiment"])
    run(["src.fraud_risk.train", "--model", "logistic",
         "--feature_set", "core_plus", "--save"])
    run(["src.fraud_risk.validate"])   # 검증(교차검증+보정+임계값)
    # (4) DB
    run(["src.db.build_db"])
    # (4) 추천
    run(["src.recommender.click_labels"])
    run(["src.recommender.train", "--experiment"])
    run(["src.recommender.train", "--model", "ltr_lgbm", "--save"])
    # 테스트
    run(["tests.test_data_augmentation"])
    run(["tests.test_integration"])

    print("\n========================================")
    print(" 전체 파이프라인 완료. Agent 데모:")
    print("   python -m src.agent.harness")
    print("========================================")


if __name__ == "__main__":
    main()
