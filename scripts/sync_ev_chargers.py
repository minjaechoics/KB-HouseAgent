from __future__ import annotations

import json

from src.tools.ev_charger_tool import sync_ev_chargers


if __name__ == "__main__":
    print(json.dumps(sync_ev_chargers(), ensure_ascii=False, indent=2))
