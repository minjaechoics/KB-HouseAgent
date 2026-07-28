"""사용법: python -m src.evaluation.cli [출력 JSON 경로]"""
from __future__ import annotations

import json
import sys

from .runner import AgentEvaluator


def main() -> int:
    evaluator = AgentEvaluator()
    report = evaluator.run()
    gate = evaluator.gate(report)
    report["quality_gate"] = gate
    if len(sys.argv) > 1:
        evaluator.save(report, sys.argv[1])
    print(json.dumps({"metrics": report["metrics"], "quality_gate": gate},
                     ensure_ascii=False, indent=2))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
