#!/usr/bin/env python3
"""
NBA TELO — ELO model for NBA basketball.

Binary outcomes (overtime always produces a winner).
High-scoring sport: margin multiplier uses a higher log scale cap.
Moderate season regression — rosters are relatively stable within a franchise.

Usage:
    python telo_nba.py          # writes data/predictions_nba.json
    python telo_nba.py --dry-run
"""

import argparse
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from telo_core import (
    INITIAL_ELO, fetch_all_events, parse_event,
    win_prob, margin_multiplier, regress,
    game_locked, build_rankings, write_sport_json,
)

# ── Model parameters ──────────────────────────────────────────────────────────
SPORT_ID  = "nba"
ESPN_PATH = "basketball/nba"
LABEL     = "NBA TELO v1.0"

K         = 20.0   # moderate — 82-game season, each game matters but not hugely
HGA       = 30.0   # real but smaller than contact sports
REGRESS   = 0.20   # mild regression — NBA rosters are stable franchise-to-franchise
HISTORY   = 20     # months of ESPN history to train on
# NBA games can be decided by 30-40 pts; use a slightly higher margin scale
MARGIN_SCALE = 0.020


def update_elo(
    home_elo: float, away_elo: float,
    winner: str, home_score: int, away_score: int,
) -> tuple[float, float]:
    expected = win_prob(home_elo, away_elo, HGA)
    actual   = 1.0 if winner == "home" else 0.0 if winner == "away" else 0.5
    mult     = margin_multiplier(abs(home_score - away_score), scale=MARGIN_SCALE)
    delta    = K * mult * (actual - expected)
    return home_elo + delta, away_elo - delta


def build_predictions() -> dict:
    print(f"  [NBA] Fetching {HISTORY} months of ESPN data...")
    events = fetch_all_events(ESPN_PATH, HISTORY)
    parsed = [p for p in (parse_event(e) for e in events) if p is not None]

    completed = sorted([g for g in parsed if g["complete"]], key=lambda g: g["date"])
    upcoming  = [g for g in parsed if not g["complete"]]
    print(f"  [NBA] {len(completed)} completed, {len(upcoming)} upcoming")

    ratings:          dict[str, float] = {}
    last_season_year: int | None       = None

    for game in completed:
        home, away = game["home"], game["away"]
        if not home or not away:
            continue
        sy = game.get("season_year", 0)
        if last_season_year and sy and sy != last_season_year:
            ratings = regress(ratings, INITIAL_ELO, REGRESS)
        if sy:
            last_season_year = sy
        ratings.setdefault(home, INITIAL_ELO)
        ratings.setdefault(away, INITIAL_ELO)
        ratings[home], ratings[away] = update_elo(
            ratings[home], ratings[away],
            game["winner"], game["home_score"], game["away_score"],
        )

    now_utc = datetime.now(timezone.utc)
    upcoming_out: list[dict] = []

    for game in sorted(upcoming, key=lambda g: g.get("date", "")):
        home, away = game["home"], game["away"]
        if not home or not away:
            continue
        home_elo = ratings.get(home, INITIAL_ELO)
        away_elo = ratings.get(away, INITIAL_ELO)
        h_prob   = round(win_prob(home_elo, away_elo, HGA) * 100, 1)
        margin   = round(abs(home_elo - away_elo + HGA) * 0.25, 1)
        upcoming_out.append({
            "home":       home,
            "away":       away,
            "date":       game.get("date", ""),
            "home_prob":  h_prob,
            "margin":     margin,
            "home_fav":   h_prob >= 50.0,
            "locked":     game_locked(game.get("date", ""), now_utc),
            "home_logo":  game.get("home_logo", ""),
            "away_logo":  game.get("away_logo", ""),
            "home_color": game.get("home_color", ""),
            "away_color": game.get("away_color", ""),
        })

    extra = {g["home"] for g in upcoming} | {g["away"] for g in upcoming}
    return {"upcoming": upcoming_out, "rankings": build_rankings(ratings, extra)}


def main(dry_run: bool = False) -> None:
    AEST = ZoneInfo("Australia/Melbourne")
    data = {
        "meta": {
            "updated": datetime.now(AEST).isoformat(),
            "model":   LABEL,
            "sport":   SPORT_ID,
        },
        **build_predictions(),
    }
    total = len(data["upcoming"])
    print(f"  [NBA] {total} upcoming predictions")
    write_sport_json(SPORT_ID, data, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
