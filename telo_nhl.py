#!/usr/bin/env python3
"""
NHL TELO — ELO model for NHL ice hockey.

Low-scoring sport (average ~6 goals per game combined) so margin multiplier
uses a conservative scale — a 4-goal blowout shouldn't dominate ELO the way
40 points would in basketball.

Overtime/shootout handling: NHL games that go to OT always produce a winner,
but the winning team's dominance is lower than a regulation win. OT games
are detected via ESPN's period count (period > 3) and treated as partial
outcomes: OT winner scores 0.75 (not 1.0), OT loser scores 0.25 (not 0.0).
This reflects that teams were essentially equal through 60 minutes.

Usage:
    python telo_nhl.py          # writes data/predictions_nhl.json
    python telo_nhl.py --dry-run
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
SPORT_ID  = "nhl"
ESPN_PATH = "hockey/nhl"
LABEL     = "NHL TELO v1.0 (OT-adjusted)"

K         = 6.0    # low — 82-game season, low-scoring sport, high variance per game
HGA       = 20.0   # modest — NHL arenas are uniform; travel matters somewhat
REGRESS   = 0.25   # moderate
HISTORY   = 20     # months

# Goals per game typically 2-4; a 4-goal margin is huge — keep scale conservative
MARGIN_SCALE   = 0.030  # per-goal sensitivity
OT_WIN_ACTUAL  = 0.75   # OT/SO winner treated as partial win
OT_LOSS_ACTUAL = 0.25   # OT/SO loser treated as partial loss


def update_elo(
    home_elo: float, away_elo: float,
    winner: str, home_score: int, away_score: int,
    ot_game: bool = False,
) -> tuple[float, float]:
    expected = win_prob(home_elo, away_elo, HGA)

    if ot_game and winner != "draw":
        # OT/SO win is worth less than regulation win
        actual = OT_WIN_ACTUAL if winner == "home" else OT_LOSS_ACTUAL
        mult   = 1.0  # no margin multiplier for OT — game ended 1 goal apart by definition
    else:
        actual = 1.0 if winner == "home" else 0.0 if winner == "away" else 0.5
        mult   = margin_multiplier(abs(home_score - away_score), scale=MARGIN_SCALE)

    delta = K * mult * (actual - expected)
    return home_elo + delta, away_elo - delta


def build_predictions() -> dict:
    print(f"  [NHL] Fetching {HISTORY} months of ESPN data...")
    events = fetch_all_events(ESPN_PATH, HISTORY)
    # track_period=True adds final_period to detect OT
    parsed = [p for p in (parse_event(e, track_period=True) for e in events) if p is not None]

    completed = sorted([g for g in parsed if g["complete"]], key=lambda g: g["date"])
    upcoming  = [g for g in parsed if not g["complete"]]
    print(f"  [NHL] {len(completed)} completed, {len(upcoming)} upcoming")

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
        ot = game.get("final_period", 3) > 3
        ratings[home], ratings[away] = update_elo(
            ratings[home], ratings[away],
            game["winner"], game["home_score"], game["away_score"], ot_game=ot,
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
        # Margin in goals — scale down significantly from points-based sports
        margin   = round(abs(home_elo - away_elo + HGA) * 0.10, 1)
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
    print(f"  [NHL] {len(data['upcoming'])} upcoming predictions")
    write_sport_json(SPORT_ID, data, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
