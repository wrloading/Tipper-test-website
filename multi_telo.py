#!/usr/bin/env python3
"""
Multi-Sport ELO Model — NBA, NRL, NFL, EPL, MLB, NHL, A-League.

Fetches historical results from ESPN public API, builds per-sport ELO ratings,
and writes data/predictions_sports.json for the Tipper app.

All inputs are pre-kickoff only: once a game's kickoff time passes, its
prediction is marked locked=True and will not change.

Usage:
    python multi_telo.py              # Current year
    python multi_telo.py --dry-run    # Print output, don't write files
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
UA       = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"

INITIAL_ELO  = 1500.0
K_DEFAULT    = 20.0
HGA_DEFAULT  = 40.0
REGRESS_DEFAULT = 0.25

# Sport-specific parameters
SPORT_CONFIG: dict[str, dict] = {
    "nba": {
        "path":    "basketball/nba",
        "label":   "NBA",
        "k":       20.0,
        "hga":     30.0,
        "regress": 0.20,
        "history": 20,  # months of history
    },
    "nrl": {
        "path":    "rugby-league/nrl",
        "label":   "NRL",
        "k":       32.0,
        "hga":     50.0,
        "regress": 0.25,
        "history": 20,
    },
    "nfl": {
        "path":    "football/nfl",
        "label":   "NFL",
        "k":       32.0,
        "hga":     55.0,
        "regress": 0.30,
        "history": 20,
    },
    "epl": {
        "path":    "soccer/eng.1",
        "label":   "EPL",
        "k":       20.0,
        "hga":     40.0,
        "regress": 0.15,
        "history": 22,
    },
    "mlb": {
        "path":    "baseball/mlb",
        "label":   "MLB",
        "k":       4.0,
        "hga":     25.0,
        "regress": 0.30,
        "history": 20,
    },
    "nhl": {
        "path":    "hockey/nhl",
        "label":   "NHL",
        "k":       6.0,
        "hga":     20.0,
        "regress": 0.25,
        "history": 20,
    },
    "aleague": {
        "path":    "soccer/aus.1",
        "label":   "A-League",
        "k":       32.0,
        "hga":     40.0,
        "regress": 0.25,
        "history": 22,
    },
}

# ── ESPN API helpers ───────────────────────────────────────────────────────────

def fetch_scoreboard_range(path: str, start: date, end: date) -> list[dict]:
    dates = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    url   = f"{BASE_URL}/{path}/scoreboard?dates={dates}&limit=500"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        if r.status_code == 200:
            return r.json().get("events", [])
        return []
    except Exception:
        return []


def fetch_all_events(path: str, history_months: int) -> list[dict]:
    """Fetch the last N months of events plus the next 14 days."""
    all_events: list[dict] = []
    seen_ids:   set[str]   = set()

    def add(events: list[dict]) -> None:
        for e in events:
            eid = e.get("id", "")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                all_events.append(e)

    today = date.today()

    # Historical: month by month, oldest first
    for i in range(history_months, 0, -1):
        year  = today.year
        month = today.month - i
        while month <= 0:
            month += 12
            year  -= 1

        m_start = date(year, month, 1)
        if month == 12:
            m_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            m_end = date(year, month + 1, 1) - timedelta(days=1)

        add(fetch_scoreboard_range(path, m_start, m_end))
        time.sleep(0.25)

    # Current month up to +14 days (includes upcoming games)
    cur_start = date(today.year, today.month, 1)
    cur_end   = today + timedelta(days=14)
    add(fetch_scoreboard_range(path, cur_start, cur_end))

    return all_events


# ── Event parsing ─────────────────────────────────────────────────────────────

def parse_event(event: dict) -> Optional[dict]:
    """Convert an ESPN event to a normalised dict. Returns None if unusable."""
    try:
        comp        = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home        = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away        = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            return None

        status   = comp.get("status", {}).get("type", {})
        complete = bool(status.get("completed", False))

        home_name  = home.get("team", {}).get("displayName", "")
        away_name  = away.get("team", {}).get("displayName", "")
        date_str   = event.get("date", "")
        season_yr  = event.get("season", {}).get("year", 0)

        team_info = {
            "home_logo":  ((home.get("team", {}).get("logos") or [{}])[0].get("href", "")
                           or home.get("team", {}).get("logo", "")),
            "away_logo":  ((away.get("team", {}).get("logos") or [{}])[0].get("href", "")
                           or away.get("team", {}).get("logo", "")),
            "home_color": home.get("team", {}).get("color", ""),
            "away_color": away.get("team", {}).get("color", ""),
        }

        if not complete:
            return {
                "complete":   False,
                "home":       home_name,
                "away":       away_name,
                "date":       date_str,
                "season_year": season_yr,
                **team_info,
            }

        try:
            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)
        except (ValueError, TypeError):
            return None

        if home.get("winner"):
            winner = "home"
        elif away.get("winner"):
            winner = "away"
        elif home_score > away_score:
            winner = "home"
        elif away_score > home_score:
            winner = "away"
        else:
            winner = "draw"

        return {
            "complete":    True,
            "home":        home_name,
            "away":        away_name,
            "home_score":  home_score,
            "away_score":  away_score,
            "winner":      winner,
            "date":        date_str,
            "season_year": season_yr,
            **team_info,
        }
    except Exception:
        return None


# ── ELO model ─────────────────────────────────────────────────────────────────

def win_probability(home_elo: float, away_elo: float, hga: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - hga) / 400.0))


def margin_multiplier(margin: int) -> float:
    return 1.0 + 0.025 * math.log(1.0 + abs(margin)) if margin > 0 else 1.0


def update_elo(
    home_elo: float,
    away_elo: float,
    winner: str,
    home_score: int,
    away_score: int,
    k: float,
    hga: float,
) -> tuple[float, float]:
    expected = win_probability(home_elo, away_elo, hga)
    actual   = 1.0 if winner == "home" else 0.0 if winner == "away" else 0.5
    mult     = margin_multiplier(abs(home_score - away_score))
    delta    = k * mult * (actual - expected)
    return home_elo + delta, away_elo - delta


def regress(ratings: dict[str, float], initial: float, factor: float) -> dict[str, float]:
    return {team: elo * (1.0 - factor) + initial * factor for team, elo in ratings.items()}


# ── Per-sport pipeline ────────────────────────────────────────────────────────

def build_sport_predictions(sport_id: str, config: dict) -> dict:
    path    = config["path"]
    k       = config.get("k", K_DEFAULT)
    hga     = config.get("hga", HGA_DEFAULT)
    rv      = config.get("regress", REGRESS_DEFAULT)
    history = config.get("history", 18)
    initial = INITIAL_ELO

    print(f"  [{sport_id.upper()}] Fetching {history} months of ESPN data...")
    events = fetch_all_events(path, history)
    parsed = [p for p in (parse_event(e) for e in events) if p is not None]

    completed = sorted(
        [g for g in parsed if g["complete"]],
        key=lambda g: g.get("date", ""),
    )
    upcoming = [g for g in parsed if not g["complete"]]
    print(f"  [{sport_id.upper()}] {len(completed)} completed, {len(upcoming)} upcoming")

    # Build ELO ratings from completed games
    ratings:          dict[str, float] = {}
    last_season_year: Optional[int]    = None

    for game in completed:
        home, away = game["home"], game["away"]
        if not home or not away:
            continue

        season_yr = game.get("season_year", 0)
        if last_season_year is not None and season_yr and season_yr != last_season_year:
            ratings = regress(ratings, initial, rv)
        if season_yr:
            last_season_year = season_yr

        if home not in ratings:
            ratings[home] = initial
        if away not in ratings:
            ratings[away] = initial

        new_h, new_a = update_elo(
            ratings[home], ratings[away],
            game["winner"], game["home_score"], game["away_score"],
            k, hga,
        )
        ratings[home] = new_h
        ratings[away] = new_a

    # Generate predictions for upcoming games
    now_utc = datetime.now(timezone.utc)
    upcoming_out: list[dict] = []

    for game in sorted(upcoming, key=lambda g: g.get("date", "")):
        home, away = game["home"], game["away"]
        if not home or not away:
            continue

        home_elo = ratings.get(home, initial)
        away_elo = ratings.get(away, initial)
        h_prob   = round(win_probability(home_elo, away_elo, hga) * 100, 1)
        margin   = round(abs(home_elo - away_elo + hga) * 0.25, 1)
        home_fav = h_prob >= 50.0

        # Lock once kickoff has passed
        date_str  = game.get("date", "")
        is_locked = False
        if date_str:
            try:
                game_dt   = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                is_locked = now_utc >= game_dt
            except Exception:
                pass

        upcoming_out.append({
            "home":       home,
            "away":       away,
            "date":       date_str,
            "home_prob":  h_prob,
            "margin":     margin,
            "home_fav":   home_fav,
            "locked":     is_locked,
            "home_logo":  game.get("home_logo", ""),
            "away_logo":  game.get("away_logo", ""),
            "home_color": game.get("home_color", ""),
            "away_color": game.get("away_color", ""),
        })

    # Team rankings by ELO
    all_teams = set(ratings.keys())
    # Also add teams seen only in upcoming
    for g in upcoming:
        all_teams.add(g["home"])
        all_teams.add(g["away"])
    all_teams.discard("")

    rankings = sorted(
        [{"team": t, "elo": round(ratings.get(t, initial))} for t in all_teams],
        key=lambda x: x["elo"],
        reverse=True,
    )

    return {
        "upcoming": upcoming_out,
        "rankings": rankings,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    AEST = ZoneInfo("Australia/Melbourne")

    print("[MULTI-TELO] Building multi-sport ELO predictions...")
    output: dict = {
        "meta": {
            "updated": datetime.now(AEST).isoformat(),
            "model":   "Multi-Sport ELO v1.0",
            "sports":  list(SPORT_CONFIG.keys()),
        },
        "sports": {},
    }

    for sport_id, config in SPORT_CONFIG.items():
        try:
            output["sports"][sport_id] = build_sport_predictions(sport_id, config)
        except Exception as e:
            print(f"  [{sport_id.upper()}] ERROR: {e}", file=sys.stderr)
            output["sports"][sport_id] = {"upcoming": [], "rankings": []}

    total_upcoming = sum(len(v["upcoming"]) for v in output["sports"].values())
    print(f"[MULTI-TELO] Done — {total_upcoming} upcoming game predictions across {len(SPORT_CONFIG)} sports")

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
    args = parser.parse_args()
    main(dry_run=args.dry_run)
