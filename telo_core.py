#!/usr/bin/env python3
"""Shared ESPN fetch, parse, and ELO utilities for TELO sport models."""

import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

# Games to exclude — preseason type, all-star / exhibition team names
PRESEASON_TYPE      = 1
EXHIBITION_RE       = re.compile(
    r'\ball[- ]?star|rising[- ]?star|team lebron|team durant|team curry|'
    r'team giannis|team stephen|nba cup final|pro bowl|skills challenge|'
    r'celebrity game',
    re.IGNORECASE,
)

RECENT_DAYS = 14  # how many days back to include pre-game probs in output

BASE_URL    = "https://site.api.espn.com/apis/site/v2/sports"
UA          = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"
INITIAL_ELO = 1500.0


# ── ESPN fetch ────────────────────────────────────────────────────────────────

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
    """Fetch N months of history plus next 14 days, deduplicated oldest-first."""
    all_events: list[dict] = []
    seen_ids:   set[str]   = set()

    def add(events: list[dict]) -> None:
        for e in events:
            eid = e.get("id", "")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                all_events.append(e)

    today = date.today()
    for i in range(history_months, 0, -1):
        year  = today.year
        month = today.month - i
        while month <= 0:
            month += 12
            year  -= 1
        m_start = date(year, month, 1)
        m_end   = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)) - timedelta(days=1)
        add(fetch_scoreboard_range(path, m_start, m_end))
        time.sleep(0.25)

    cur_start = date(today.year, today.month, 1)
    add(fetch_scoreboard_range(path, cur_start, today + timedelta(days=14)))
    return all_events


# ── Event parsing ─────────────────────────────────────────────────────────────

def parse_event(event: dict, track_period: bool = False) -> Optional[dict]:
    """Convert ESPN event → normalised game dict. track_period=True adds final_period."""
    try:
        # Skip preseason events (type 1)
        if event.get("season", {}).get("type", 2) == PRESEASON_TYPE:
            return None

        comp        = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home        = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away        = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            return None

        status_obj = comp.get("status", {})
        status     = status_obj.get("type", {})
        complete   = bool(status.get("completed", False))
        period     = int(status_obj.get("period", 0) or 0)

        home_name = home.get("team", {}).get("displayName", "")
        away_name = away.get("team", {}).get("displayName", "")

        # Skip all-star / exhibition team names
        if EXHIBITION_RE.search(home_name) or EXHIBITION_RE.search(away_name):
            return None
        date_str  = event.get("date", "")
        season_yr = event.get("season", {}).get("year", 0)

        team_info = {
            "home_logo":  ((home.get("team", {}).get("logos") or [{}])[0].get("href", "")
                           or home.get("team", {}).get("logo", "")),
            "away_logo":  ((away.get("team", {}).get("logos") or [{}])[0].get("href", "")
                           or away.get("team", {}).get("logo", "")),
            "home_color": home.get("team", {}).get("color", ""),
            "away_color": away.get("team", {}).get("color", ""),
        }

        if not complete:
            return {"complete": False, "home": home_name, "away": away_name,
                    "date": date_str, "season_year": season_yr, **team_info}

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

        result = {
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
        if track_period:
            result["final_period"] = period
        return result
    except Exception:
        return None


# ── ELO maths ─────────────────────────────────────────────────────────────────

def win_prob(home_elo: float, away_elo: float, hga: float) -> float:
    """Standard binary ELO home-win probability."""
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - hga) / 400.0))


def margin_multiplier(margin: int, scale: float = 0.025) -> float:
    """Scale ELO update by log of score margin. scale tunes sensitivity."""
    return 1.0 + scale * math.log(1.0 + abs(margin)) if margin > 0 else 1.0


def regress(ratings: dict[str, float], initial: float, factor: float) -> dict[str, float]:
    return {t: elo * (1.0 - factor) + initial * factor for t, elo in ratings.items()}


def game_locked(date_str: str, now_utc: datetime) -> bool:
    if not date_str:
        return False
    try:
        game_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return now_utc >= game_dt
    except Exception:
        return False


def build_rankings(ratings: dict[str, float], extra_teams: set[str]) -> list[dict]:
    all_teams = set(ratings.keys()) | extra_teams
    all_teams.discard("")
    return sorted(
        [{"team": t, "elo": round(ratings.get(t, INITIAL_ELO))} for t in all_teams],
        key=lambda x: x["elo"],
        reverse=True,
    )


# ── Output ────────────────────────────────────────────────────────────────────

def write_sport_json(sport_id: str, data: dict, dry_run: bool = False) -> None:
    if dry_run:
        print(json.dumps(data, indent=2))
    else:
        os.makedirs("data", exist_ok=True)
        path = f"data/predictions_{sport_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Written {path}")
