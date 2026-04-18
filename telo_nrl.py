#!/usr/bin/env python3
"""
NRL TELO — ELO model for NRL rugby league.

Binary outcomes (draws are extremely rare in NRL).
High-scoring sport with strong home ground advantage.
Moderate-to-high K factor — results can swing quickly over a season.

Usage:
    python telo_nrl.py          # writes data/predictions_nrl.json
    python telo_nrl.py --dry-run
"""

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from telo_core import (
    INITIAL_ELO, fetch_all_events, parse_event,
    win_prob, margin_multiplier, regress,
    game_locked, build_rankings, write_sport_json,
)

# ── Model parameters ──────────────────────────────────────────────────────────
SPORT_ID  = "nrl"
ESPN_PATH = "rugby-league/nrl"
LABEL     = "NRL TELO v1.0"

K         = 32.0   # high — 27-game season, each result matters
HGA       = 50.0   # strong — crowd noise and travel in Australia is significant
REGRESS   = 0.25   # moderate — squad changes each off-season
HISTORY   = 20     # months of history
MARGIN_SCALE = 0.022  # high-scoring: moderate scale


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
    print(f"  [NRL] Fetching {HISTORY} months of ESPN data...")
    events = fetch_all_events(ESPN_PATH, HISTORY)
    parsed = [p for p in (parse_event(e) for e in events) if p is not None]

    completed = sorted([g for g in parsed if g["complete"]], key=lambda g: g["date"])
    upcoming  = [g for g in parsed if not g["complete"]]
    print(f"  [NRL] {len(completed)} completed, {len(upcoming)} upcoming")

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
    print(f"  [NRL] {len(data['upcoming'])} upcoming predictions")
    write_sport_json(SPORT_ID, data, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
