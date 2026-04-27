from __future__ import annotations

"""
Champion Data ingest — Super Netball.

Champion Data provides an open JSON API at mc.championdata.com that serves
complete fixtures (upcoming + results) for every Super Netball season.

All game records returned match the standard format expected by generate.py:
  {
    'id':         str,   # unique game identifier
    'home_team':  str,   # team nickname e.g. "Mavericks"
    'away_team':  str,
    'home_score': int,   # only on completed games
    'away_score': int,
    'date':       str,   # ISO date string "YYYY-MM-DD"
    'venue':      str,
    'neutral':    bool,
  }

Upcoming game records omit home_score / away_score and include a 'round' key.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL     = 'https://mc.championdata.com/data'
REQUEST_DELAY = 0.35
MAX_RETRIES   = 3

# Competition IDs per sport per season.
# Each season may have a regular-season comp and a finals comp.
COMP_IDS: dict[str, dict[int, list[int]]] = {
    'supernetball': {
        2020: [11108, 11109],
        2021: [11391, 11392],
        2022: [11665, 11666],
        2023: [12045, 12046],
        2024: [12438, 12439],
        2025: [12715, 12716],
        2026: [12949, 12950],
    },
}


def _fetch_fixture(comp_id: int) -> list[dict]:
    """Fetch all matches for a competition ID. Returns raw match dicts."""
    url = f'{BASE_URL}/{comp_id}/fixture.json'
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15,
                                headers={'User-Agent': 'TipperPredictions/1.0'})
            if resp.status_code == 200:
                return resp.json().get('fixture', {}).get('match', [])
            logger.warning(f'Champion Data returned {resp.status_code} for comp {comp_id}')
            return []
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                logger.error(f'Champion Data fetch failed for comp {comp_id}: {e}')
                return []
    return []


def _to_record(match: dict, include_scores: bool = True) -> Optional[dict]:
    """Convert a Champion Data match dict to the standard generate.py format."""
    home = match.get('homeSquadNickname') or match.get('homeSquadName', '')
    away = match.get('awaySquadNickname') or match.get('awaySquadName', '')
    if not home or not away:
        return None

    utc_str = match.get('utcStartTime', '')
    try:
        dt = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
        date_str = dt.date().isoformat()
    except (ValueError, AttributeError):
        return None

    comp_id  = str(match.get('matchId', ''))
    record: dict = {
        'id':       f'supernetball_{comp_id}',
        'home_team': home,
        'away_team': away,
        'date':     date_str,
        'venue':    match.get('venueName', ''),
        'neutral':  False,
        'round':    f'Round {match.get("roundNumber", "?")}',
    }

    if include_scores:
        hs = match.get('homeSquadScore')
        as_ = match.get('awaySquadScore')
        if hs is None or as_ is None:
            return None
        record['home_score'] = int(hs)
        record['away_score'] = int(as_)

    return record


def fetch_season(sport: str, season: int) -> list[dict]:
    """Fetch all completed games for a given season (used by seed / history)."""
    comp_list = COMP_IDS.get(sport, {}).get(season, [])
    games: list[dict] = []
    for comp_id in comp_list:
        matches = _fetch_fixture(comp_id)
        for m in matches:
            if m.get('matchStatus') != 'complete':
                continue
            rec = _to_record(m, include_scores=True)
            if rec:
                games.append(rec)
        time.sleep(REQUEST_DELAY)
    return games


def fetch_recent(sport: str, days: int = 14) -> list[dict]:
    """Fetch completed games from the last `days` days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    current_year = datetime.now(timezone.utc).year

    games: list[dict] = []
    seen: set[str] = set()

    # Check current and previous season in case of year-boundary overlap
    for season in [current_year - 1, current_year]:
        for comp_id in COMP_IDS.get(sport, {}).get(season, []):
            for m in _fetch_fixture(comp_id):
                if m.get('matchStatus') != 'complete':
                    continue
                utc_str = m.get('utcStartTime', '')
                try:
                    game_date = datetime.fromisoformat(
                        utc_str.replace('Z', '+00:00')
                    ).date()
                except (ValueError, AttributeError):
                    continue
                if game_date < cutoff:
                    continue
                rec = _to_record(m, include_scores=True)
                if rec and rec['id'] not in seen:
                    seen.add(rec['id'])
                    games.append(rec)
            time.sleep(REQUEST_DELAY)

    return sorted(games, key=lambda g: g['date'])


def fetch_upcoming(sport: str, days: int = 14) -> list[dict]:
    """Fetch scheduled games in the next `days` days."""
    now      = datetime.now(timezone.utc)
    cutoff   = (now + timedelta(days=days)).date()
    today    = now.date()
    current_year = now.year

    games: list[dict] = []
    seen: set[str] = set()

    for season in [current_year, current_year + 1]:
        for comp_id in COMP_IDS.get(sport, {}).get(season, []):
            for m in _fetch_fixture(comp_id):
                if m.get('matchStatus') == 'complete':
                    continue
                utc_str = m.get('utcStartTime', '')
                try:
                    game_date = datetime.fromisoformat(
                        utc_str.replace('Z', '+00:00')
                    ).date()
                except (ValueError, AttributeError):
                    continue
                if game_date < today or game_date > cutoff:
                    continue
                rec = _to_record(m, include_scores=False)
                if rec and rec['id'] not in seen:
                    seen.add(rec['id'])
                    # Store full ISO datetime for the output
                    rec['date'] = m.get('utcStartTime', rec['date'])
                    games.append(rec)
            time.sleep(REQUEST_DELAY)

    return sorted(games, key=lambda g: g['date'])
