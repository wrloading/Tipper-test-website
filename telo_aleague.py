#!/usr/bin/env python3
"""
A-League TELO — 3-outcome ELO model for Australian A-League Men's soccer.

Like EPL, soccer has three outcomes (win/draw/loss) and this model explicitly
computes draw probability using a Gaussian draw band.

A-League differences from EPL:
- Higher K (32 vs 20) — smaller, more volatile league with fewer teams
- Higher base draw rate — Australian soccer sees more draws than EPL (~28%)
- Stronger season regression — squad quality fluctuates more between seasons
- 27-round season (shorter than EPL), so each game matters more

Usage:
    python telo_aleague.py          # writes data/predictions_aleague.json
    python telo_aleague.py --dry-run
"""

import argparse
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from telo_core import (
    INITIAL_ELO, RECENT_DAYS, fetch_all_events, parse_event,
    margin_multiplier, regress,
    game_locked, build_rankings, write_sport_json,
)

# ── Model parameters ──────────────────────────────────────────────────────────
SPORT_ID  = "aleague"
ESPN_PATH = "soccer/aus.1"
LABEL     = "A-League TELO v1.0 (3-outcome)"

K           = 32.0   # high — 27-round season, smaller league, more volatile
HGA         = 75.0   # A-League home win rate ~44% → 75 ELO pts calibrated
REGRESS     = 0.25   # moderate — squad quality shifts between seasons
HISTORY     = 22     # months

MARGIN_SCALE   = 0.018  # soccer: low-scoring, moderate goal sensitivity

# 3-outcome draw model (tuned for A-League)
BASE_DRAW_RATE = 0.28   # A-League empirical draw frequency slightly higher than EPL
DRAW_SCALE     = 260.0  # draw band width (wider to match higher HGA)


# ── 3-outcome soccer probability model ────────────────────────────────────────

def soccer_probabilities(
    home_elo: float, away_elo: float, hga: float
) -> tuple[float, float, float]:
    """Returns (p_home_win, p_draw, p_away_win)."""
    elo_diff = home_elo - away_elo + hga

    p_draw = BASE_DRAW_RATE * math.exp(-(elo_diff ** 2) / (2 * DRAW_SCALE ** 2))
    p_draw = max(0.0, min(p_draw, 0.45))

    p_binary_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_remaining   = 1.0 - p_draw

    p_home = p_binary_home * p_remaining
    p_away = (1.0 - p_binary_home) * p_remaining
    return p_home, p_draw, p_away


def update_elo(
    home_elo: float, away_elo: float,
    winner: str, home_score: int, away_score: int,
) -> tuple[float, float]:
    p_home, p_draw, _ = soccer_probabilities(home_elo, away_elo, HGA)
    expected = p_home + 0.5 * p_draw
    actual   = 1.0 if winner == "home" else 0.0 if winner == "away" else 0.5
    mult     = margin_multiplier(abs(home_score - away_score), scale=MARGIN_SCALE)
    delta    = K * mult * (actual - expected)
    return home_elo + delta, away_elo - delta


def build_predictions() -> dict:
    print(f"  [A-LEAGUE] Fetching {HISTORY} months of ESPN data...")
    events = fetch_all_events(ESPN_PATH, HISTORY)
    parsed = [p for p in (parse_event(e) for e in events) if p is not None]

    completed = sorted([g for g in parsed if g["complete"]], key=lambda g: g["date"])
    upcoming  = [g for g in parsed if not g["complete"]]
    print(f"  [A-LEAGUE] {len(completed)} completed, {len(upcoming)} upcoming")

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
            p_home, p_draw, p_away = soccer_probabilities(ratings[home], ratings[away], HGA)
            recent_out.append({
                "home": home, "away": away, "date": game["date"],
                "home_prob": round(p_home * 100, 1), "draw_prob": round(p_draw * 100, 1),
                "home_fav": p_home >= p_away,
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
        home_elo            = ratings.get(home, INITIAL_ELO)
        away_elo            = ratings.get(away, INITIAL_ELO)
        p_home, p_draw, p_away = soccer_probabilities(home_elo, away_elo, HGA)
        h_prob              = round(p_home * 100, 1)
        d_prob              = round(p_draw * 100, 1)
        a_prob              = round(p_away * 100, 1)
        # Scale: equal teams → ~0.4 goal HGA; 100-ELO mismatch → ~0.9 goal total
        margin              = round(abs(home_elo - away_elo + HGA) * 0.005, 1)
        upcoming_out.append({
            "home":       home,
            "away":       away,
            "date":       game.get("date", ""),
            "home_prob":  h_prob,
            "draw_prob":  d_prob,
            "away_prob":  a_prob,
            "margin":     margin,
            "home_fav":   h_prob >= a_prob,
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
    print(f"  [A-LEAGUE] {len(data['upcoming'])} upcoming predictions")
    write_sport_json(SPORT_ID, data, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
