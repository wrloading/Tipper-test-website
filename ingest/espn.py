from __future__ import annotations

"""
ESPN historical results ingester.

Uses the ESPN public scoreboard API to fetch completed game results for any
sport it covers. Iterates through seasons year-by-year to build a full
5-season history.

The ESPN API is the same endpoint the app already uses for live data, which
means team names will always match between historical ratings and current
predictions — no name-normalisation drift.

Rate limiting: ESPN is undocumented and rate-limits aggressively. We use
a 0.3s delay between requests and retry on transient failures.
"""

import time
import logging
from datetime import date, timedelta
from typing import Optional

import requests

from ingest.aliases import normalise

logger = logging.getLogger(__name__)

ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports'

# Paths for each sport — mirrors the app's SPORTS config
ESPN_PATHS: dict[str, str] = {
    'afl':        'australian-football/afl',
    'nba':        'basketball/nba',
    'wnba':       'basketball/wnba',
    'nfl':        'football/nfl',
    'nrl':        'rugby-league/nrl',
    'mlb':        'baseball/mlb',
    'nhl':        'hockey/nhl',
    # Soccer — used by football_data.py for MLS/A-League/UCL/UEL and all upcoming fixtures
    'epl':        'soccer/eng.1',
    'laliga':     'soccer/esp.1',
    'bundesliga': 'soccer/ger.1',
    'seriea':     'soccer/ita.1',
    'ligue1':     'soccer/fra.1',
    'mls':        'soccer/usa.1',
    'aleague':    'soccer/aus.1',
    'ucl':        'soccer/uefa.champions',
    'uel':        'soccer/uefa.europa',
}

# How many months to fetch per API call (ESPN date range param)
CHUNK_MONTHS = 2

REQUEST_DELAY = 0.35   # seconds between requests
MAX_RETRIES   = 3


def _espn_date(d: date) -> str:
    return d.strftime('%Y%m%d')


def _fetch_scoreboard(path: str, date_range: str, limit: int = 200) -> list[dict]:
    """Fetch a single scoreboard page from ESPN. Returns list of events."""
    url = f'{ESPN_BASE}/{path}/scoreboard'
    params = {'dates': date_range, 'limit': limit}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15,
                                headers={'User-Agent': 'TipperPredictions/1.0'})
            if resp.status_code == 200:
                return resp.json().get('events', [])
            if resp.status_code == 429:
                logger.warning('ESPN rate-limited, waiting 5s...')
                time.sleep(5)
            else:
                logger.warning(f'ESPN returned {resp.status_code} for {url}')
                return []
        except requests.RequestException as e:
            logger.warning(f'ESPN request failed (attempt {attempt + 1}): {e}')
            time.sleep(2)

    return []


def _parse_event(event: dict, sport: str) -> Optional[dict]:
    """
    Parse a single ESPN event into a normalised game record.
    Returns None if the game is not complete or data is missing.
    """
    # Skip preseason (type 1) — includes international exhibition games that
    # contaminate ratings with non-league teams (e.g. NBA tours vs NBL/EuroLeague)
    if event.get('season', {}).get('type') == 1:
        return None

    comp = event.get('competitions', [{}])[0]
    status = comp.get('status', {}).get('type', {})
    if not status.get('completed'):
        return None

    competitors = comp.get('competitors', [])
    home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
    away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
    if not home or not away:
        return None

    try:
        home_score = int(home.get('score', 0))
        away_score = int(away.get('score', 0))
    except (ValueError, TypeError):
        return None

    home_name = normalise(home.get('team', {}).get('displayName', ''), sport)
    away_name = normalise(away.get('team', {}).get('displayName', ''), sport)

    if not home_name or not away_name:
        return None

    # Extract date (YYYY-MM-DD)
    raw_date = event.get('date', '')[:10]

    # Determine season from date (ESPN doesn't always expose season year cleanly)
    season_year = int(raw_date[:4]) if raw_date else 0

    venue = (
        comp.get('venue', {}).get('fullName')
        or comp.get('venue', {}).get('address', {}).get('city')
        or ''
    )

    return {
        'id':         event.get('id', ''),
        'date':       raw_date,
        'season':     season_year,
        'home_team':  home_name,
        'away_team':  away_name,
        'home_score': home_score,
        'away_score': away_score,
        'neutral':    comp.get('neutralSite', False),
        'venue':      venue,
    }


def fetch_season(sport: str, season: int) -> list[dict]:
    """
    Fetch all completed games for a sport in a given season year.
    Iterates in 2-month chunks to avoid ESPN's result limits.
    Returns a list of normalised game records sorted by date.
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        logger.error(f'No ESPN path configured for sport: {sport}')
        return []

    # Determine season date range
    # NBA/NHL seasons span two calendar years: season 2023 = Oct 2022–Jun 2023
    # AFL/NRL: season year is the calendar year (Mar–Sep)
    # NFL: season year = year the season starts (Sep–Feb following year)
    if sport in ('nba', 'wnba', 'nhl'):
        start = date(season - 1, 9, 1)
        end   = date(season, 8, 31)
    elif sport == 'nfl':
        start = date(season, 8, 1)
        end   = date(season + 1, 2, 28)
    elif sport in ('afl', 'nrl'):
        start = date(season, 2, 1)
        end   = date(season, 10, 31)
    elif sport == 'mlb':
        start = date(season, 3, 1)
        end   = date(season, 11, 15)
    elif sport == 'mls':
        start = date(season, 2, 1)
        end   = date(season, 12, 15)
    elif sport == 'aleague':
        start = date(season - 1, 10, 1)
        end   = date(season, 6, 30)
    elif sport in ('epl', 'laliga', 'bundesliga', 'seriea', 'ligue1', 'ucl', 'uel'):
        # European leagues: season year = end year (2024 = 2023/24)
        start = date(season - 1, 7, 1)
        end   = date(season, 6, 30)
    else:
        start = date(season, 1, 1)
        end   = date(season, 12, 31)

    games: list[dict] = []
    seen_ids: set[str] = set()
    cursor = start

    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=60), end)
        date_range = f'{_espn_date(cursor)}-{_espn_date(chunk_end)}'

        logger.info(f'  Fetching {sport} {season}: {date_range}')
        events = _fetch_scoreboard(path, date_range)

        for event in events:
            game = _parse_event(event, sport)
            if game and game['id'] not in seen_ids:
                game['season'] = season  # Override with requested season
                games.append(game)
                seen_ids.add(game['id'])

        cursor = chunk_end + timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    games.sort(key=lambda g: g['date'])
    logger.info(f'  {sport} {season}: {len(games)} games fetched')
    return games


def fetch_recent(sport: str, days: int = 14) -> list[dict]:
    """
    Fetch the most recent completed games (last N days).
    Used for daily updates rather than full historical seeding.
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        return []

    today     = date.today()
    from_date = today - timedelta(days=days)
    date_range = f'{_espn_date(from_date)}-{_espn_date(today)}'

    events = _fetch_scoreboard(path, date_range)
    games = []
    seen_ids: set[str] = set()

    for event in events:
        game = _parse_event(event, sport)
        if game and game['id'] not in seen_ids:
            games.append(game)
            seen_ids.add(game['id'])

    return sorted(games, key=lambda g: g['date'])


def fetch_upcoming(sport: str, days: int = 14) -> list[dict]:
    """
    Fetch upcoming (not yet completed) games for prediction generation.
    Returns games in the next N days with team names and dates.
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        return []

    today    = date.today()
    end_date = today + timedelta(days=days)
    date_range = f'{_espn_date(today)}-{_espn_date(end_date)}'

    url    = f'{ESPN_BASE}/{path}/scoreboard'
    params = {'dates': date_range, 'limit': 200}

    try:
        resp = requests.get(url, params=params, timeout=15,
                            headers={'User-Agent': 'TipperPredictions/1.0'})
        if not resp.ok:
            return []
        events = resp.json().get('events', [])
    except requests.RequestException:
        return []

    upcoming = []
    for event in events:
        comp = event.get('competitions', [{}])[0]
        status = comp.get('status', {}).get('type', {})
        if status.get('completed'):
            continue  # Already played

        competitors = comp.get('competitors', [])
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        if not home or not away:
            continue

        home_name = normalise(home.get('team', {}).get('displayName', ''), sport)
        away_name = normalise(away.get('team', {}).get('displayName', ''), sport)

        if not home_name or not away_name:
            continue

        upcoming.append({
            'id':         f'{sport}_{event.get("id", "")}',
            'date':       event.get('date', '')[:10],
            'datetime':   event.get('date', ''),
            'home_team':  home_name,
            'away_team':  away_name,
            'neutral':    comp.get('neutralSite', False),
            'venue':      comp.get('venue', {}).get('fullName', ''),
        })

    return sorted(upcoming, key=lambda g: g['datetime'])
