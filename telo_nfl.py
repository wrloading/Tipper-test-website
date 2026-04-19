#!/usr/bin/env python3
"""
NFL TELO — ELO model for NFL American football.

Only 17 regular-season games, so each result carries enormous weight (high K).
Strongest home ground advantage of any major sport.
Strongest season regression — rosters turn over heavily via draft and free agency.
Ties are technically possible but treated as draws (0.5 outcome).

Usage:
    python telo_nfl.py          # writes data/predictions_nfl.json
    python telo_nfl.py --dry-run
"""

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from telo_core import (
    INITIAL_ELO, RECENT_DAYS, fetch_all_events, parse_event,
    win_prob, margin_multiplier, regress,
    game_locked, build_rankings, write_sport_json,
)

# ── Model parameters ──────────────────────────────────────────────────────────
SPORT_ID  = "nfl"
ESPN_PATH = "football/nfl"
LABEL     = "NFL TELO v1.0"

K         = 32.0   # high — only 17 games in regular season; each game is high-stakes
HGA       = 55.0   # highest of any sport modelled — travel + crowd impact is large
REGRESS   = 0.30   # strongest regression — draft, trades, FA create large roster turnover
HISTORY   = 20     # months
# NFL scores are relatively low; don't over-weight blowouts — use a smaller scale
MARGIN_SCALE = 0.015


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
    print(f"  [NFL] Fetching {HISTORY} months of ESPN data...")
    events = fetch_all_events(ESPN_PATH, HISTORY)
    parsed = [p for p in (parse_event(e) for e in events) if p is not None]

    completed = sorted([g for g in parsed if g["complete"]], key=lambda g: g["date"])
    upcoming  = [g for g in parsed if not g["complete"]]
    print(f"  [NFL] {len(completed)} completed, {len(upcoming)} upcoming")

    ratings:          dict[str, float] = {}
    last_season_year: int | None       = None
    recent_out:       list[dict]       = []

    from datetime import datetime, timedelta, timezone
    recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")

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

        if game.get("date", "")[:10] >= recent_cutoff:
            pre_prob = round(win_prob(ratings[home], ratings[away], HGA) * 100, 1)
            recent_out.append({
                "home": home, "away": away, "date": game["date"],
                "home_prob": pre_prob, "home_fav": pre_prob >= 50.0,
                "winner": game["winner"],
                "home_score": game["home_score"], "away_score": game["away_score"],
                "home_logo": game.get("home_logo", ""), "away_logo": game.get("away_logo", ""),
                "home_color": game.get("home_color", ""), "away_color": game.get("away_color", ""),
            })

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
        # Scale: equal teams → ~3 pt HGA; 100-ELO mismatch → ~8.5 pts total
        margin   = round(abs(home_elo - away_elo + HGA) * 0.055, 1)
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
    return {"upcoming": upcoming_out, "rankings": build_rankings(ratings, extra), "recent": recent_out}


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
    print(f"  [NFL] {len(data['upcoming'])} upcoming predictions")
    write_sport_json(SPORT_ID, data, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
