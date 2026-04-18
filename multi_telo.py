#!/usr/bin/env python3
"""
Multi-Sport TELO Orchestrator — runs all individual sport ELO models and
combines their output into data/predictions_sports.json for the Tipper app.

Each sport has its own dedicated model with sport-specific logic:
  telo_nba.py      — NBA basketball (binary, high-scoring)
  telo_nrl.py      — NRL rugby league (binary, high-scoring, strong HGA)
  telo_nfl.py      — NFL American football (binary, few games, strong regression)
  telo_epl.py      — EPL soccer (3-outcome: win/draw/loss)
  telo_mlb.py      — MLB baseball (binary, 162-game season, very low K)
  telo_nhl.py      — NHL ice hockey (OT-adjusted outcomes)
  telo_aleague.py  — A-League soccer (3-outcome like EPL)

Usage:
    python multi_telo.py              # runs all sports, writes predictions_sports.json
    python multi_telo.py --dry-run    # prints JSON, doesn't write files
"""

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import telo_nba
import telo_nrl
import telo_nfl
import telo_epl
import telo_mlb
import telo_nhl
import telo_aleague

SPORT_MODULES = [
    ("nba",      telo_nba),
    ("nrl",      telo_nrl),
    ("nfl",      telo_nfl),
    ("epl",      telo_epl),
    ("mlb",      telo_mlb),
    ("nhl",      telo_nhl),
    ("aleague",  telo_aleague),
]


def main(dry_run: bool = False) -> None:
    AEST = ZoneInfo("Australia/Melbourne")

    print("[MULTI-TELO] Building individualised sport ELO predictions...")
    output: dict = {
        "meta": {
            "updated": datetime.now(AEST).isoformat(),
            "model":   "Multi-Sport TELO v2.0 (individualised)",
            "sports":  [s for s, _ in SPORT_MODULES],
        },
        "sports": {},
    }

    for sport_id, module in SPORT_MODULES:
        try:
            output["sports"][sport_id] = module.build_predictions()
        except Exception as e:
            print(f"  [{sport_id.upper()}] ERROR: {e}", file=sys.stderr)
            output["sports"][sport_id] = {"upcoming": [], "rankings": []}

    total = sum(len(v["upcoming"]) for v in output["sports"].values())
    print(f"[MULTI-TELO] Done — {total} upcoming predictions across {len(SPORT_MODULES)} sports")

    if dry_run:
        print(json.dumps(output, indent=2))
    else:
        os.makedirs("data", exist_ok=True)
        with open("data/predictions_sports.json", "w") as f:
            json.dump(output, f, indent=2)
        print("[MULTI-TELO] ✓ Written data/predictions_sports.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
