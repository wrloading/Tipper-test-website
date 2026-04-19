#!/usr/bin/env python3
"""
MLB TELO — ELO model for Major League Baseball.

162-game season means each individual game has low predictive weight → very low K.
Extra innings always produce a winner (binary outcomes).
High roster turnover via trades and free agency → stronger season regression.
Run differential matters but blowouts are common — use a capped margin scale.

Usage:
    python telo_mlb.py          # writes data/predictions_mlb.json
    python telo_mlb.py --dry-run
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
SPORT_ID  = "mlb"
ESPN_PATH = "baseball/mlb"
LABEL     = "MLB TELO v1.0"

K         = 4.0    # very low — over 162 games each result carries small weight
HGA       = 30.0   # MLB home win rate ~54% historically → 30 ELO pts calibrated
REGRESS   = 0.30   # strong — trades, FA, and pitching rotations shift teams significantly
HISTORY   = 20     # months

# Baseball run differentials can be large (15+ runs); cap the multiplier effect
MARGIN_SCALE = 0.012  # lowest scale — run blowouts shouldn't dominate ELO


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
    print(f"  [MLB] Fetching {HISTORY} months of ESPN data...")
    events = fetch_all_events(ESPN_PATH, HISTORY)
    parsed = [p for p in (parse_event(e) for e in events) if p is not None]

    completed = sorted([g for g in parsed if g["complete"]], key=lambda g: g["date"])
    upcoming  = [g for g in parsed if not g["complete"]]
    print(f"  [MLB] {len(completed)} completed, {len(upcoming)} upcoming")

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
        # Scale: equal teams → ~0.35 run HGA; 100-ELO mismatch → ~1.3 runs total
        margin   = round(abs(home_elo - away_elo + HGA) * 0.012, 1)
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
    print(f"  [MLB] {len(data['upcoming'])} upcoming predictions")
    write_sport_json(SPORT_ID, data, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
