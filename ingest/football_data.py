from __future__ import annotations

"""
football-data.co.uk ingester.

The gold standard for free historical soccer data. Provides CSV files for
every major European league going back to the 1990s, updated same-day.

URL format: https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv
  season:       e.g. "2324" for 2023/24, "2223" for 2022/23
  league_code:  E0 = EPL, SP1 = La Liga, D1 = Bundesliga, etc.

For MLS and A-League (not on football-data.co.uk), falls back to the
football-data.org API which has broader coverage including Americas/Oceania.
"""

import io
import time
import logging
from typing import Optional

import requests
import csv

from ingest.aliases import normalise

logger = logging.getLogger(__name__)

FD_BASE    = 'https://www.football-data.co.uk/mmz4281'
FD_ALT     = 'https://www.football-data.co.uk'  # For other leagues
FDO_BASE   = 'https://api.football-data.org/v4'  # football-data.org (requires free API key)

REQUEST_DELAY = 0.25


# ── League code mapping ───────────────────────────────────────────────────────
# football-data.co.uk codes

FD_LEAGUE_CODES: dict[str, str] = {
    'epl':        'E0',
    'laliga':     'SP1',
    'bundesliga': 'D1',
    'seriea':     'I1',
    'ligue1':     'F1',
}

# football-data.co.uk season format: 2324 = 2023/24, 2223 = 2022/23
def _fd_season_code(season: int) -> str:
    """Convert a season year (end year) to football-data.co.uk format."""
    return f'{str(season - 1)[2:]}{str(season)[2:]}'


# ── Main European leagues (football-data.co.uk) ───────────────────────────────

def _fetch_fd_csv(league: str, season: int) -> list[dict]:
    """
    Fetch and parse a football-data.co.uk CSV for a league/season.
    Returns a list of normalised game records.
    """
    code       = FD_LEAGUE_CODES.get(league)
    season_str = _fd_season_code(season)
    url        = f'{FD_BASE}/{season_str}/{code}.csv'

    try:
        resp = requests.get(url, timeout=20, headers={'User-Agent': 'Tipper/1.0'})
        if resp.status_code == 404:
            logger.warning(f'football-data: no data for {league} {season} ({url})')
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f'football-data fetch failed for {league} {season}: {e}')
        return []

    games = []
    reader = csv.DictReader(io.StringIO(resp.text))

    for row in reader:
        try:
            # Required fields
            date_str  = row.get('Date', '').strip()
            home_raw  = row.get('HomeTeam', '').strip()
            away_raw  = row.get('AwayTeam', '').strip()
            fthg      = row.get('FTHG', '').strip()  # Full time home goals
            ftag      = row.get('FTAG', '').strip()  # Full time away goals

            if not all([date_str, home_raw, away_raw, fthg, ftag]):
                continue

            home_score = int(fthg)
            away_score = int(ftag)

            # Parse date — format is DD/MM/YY or DD/MM/YYYY
            parts = date_str.split('/')
            if len(parts) == 3:
                day, month, year = parts
                if len(year) == 2:
                    year = f'20{year}' if int(year) < 50 else f'19{year}'
                iso_date = f'{year}-{month.zfill(2)}-{day.zfill(2)}'
            else:
                continue

            home_name = normalise(home_raw, league)
            away_name = normalise(away_raw, league)

            games.append({
                'id':         f'{league}_{iso_date}_{home_name}_{away_name}'.replace(' ', '_'),
                'date':       iso_date,
                'season':     season,
                'home_team':  home_name,
                'away_team':  away_name,
                'home_score': home_score,
                'away_score': away_score,
                'neutral':    False,
            })
        except (ValueError, KeyError):
            continue

    games.sort(key=lambda g: g['date'])
    logger.info(f'  {league} {season}: {len(games)} games from football-data.co.uk')
    return games


# ── MLS / A-League via ESPN fallback ─────────────────────────────────────────
# These leagues aren't on football-data.co.uk, so we use ESPN.

def _fetch_espn_soccer(league: str, season: int) -> list[dict]:
    """Fetch soccer results from ESPN for leagues not on football-data.co.uk."""
    from ingest.espn import fetch_season as espn_fetch
    return espn_fetch(league, season)


# ── UCL / UEL — football-data.co.uk has limited coverage ─────────────────────
# Use the dedicated competition CSVs when available.

UEFA_CODES: dict[str, str] = {
    'ucl': 'EC',   # European Championship — note: UCL is not standardly on fd.co.uk
    'uel': None,
}

def _fetch_uefa(league: str, season: int) -> list[dict]:
    """
    Try football-data.co.uk for UEFA competitions, fall back to ESPN.
    UCL data is sparse on fd.co.uk — ESPN is more reliable for these.
    """
    from ingest.espn import fetch_season as espn_fetch
    return espn_fetch(league, season)


# ── Recent results (for daily updates) ───────────────────────────────────────

def fetch_recent_fd(league: str, days: int = 14) -> list[dict]:
    """
    Fetch the current season's CSV and return only games from the last N days.
    Faster than fetching from ESPN for European leagues.
    """
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    current_season = _current_season_for(league)
    all_games = _fetch_fd_csv(league, current_season)
    return [g for g in all_games if g['date'] >= cutoff]


def _current_season_for(league: str) -> int:
    """
    Determine the current football season's end year.
    European leagues run Aug–May: in Aug 2025, the season is 2025/26 (end year 2026).
    """
    from datetime import date
    today = date.today()
    # Season flips in July
    return today.year + 1 if today.month >= 7 else today.year


# ── Upcoming fixtures ─────────────────────────────────────────────────────────

def fetch_upcoming_fd(league: str, days: int = 14) -> list[dict]:
    """
    Fetch upcoming fixtures for a soccer league.
    football-data.co.uk doesn't provide fixtures — use ESPN for this.
    """
    from ingest.espn import fetch_upcoming as espn_upcoming

    # Map our sport IDs to ESPN soccer paths
    SOCCER_ESPN_PATHS: dict[str, str] = {
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

    if league not in SOCCER_ESPN_PATHS:
        return []

    # Temporarily patch ESPN path for this call
    from ingest import espn as espn_module
    original = espn_module.ESPN_PATHS.get(league)
    espn_module.ESPN_PATHS[league] = SOCCER_ESPN_PATHS[league]

    try:
        result = espn_upcoming(league, days)
    finally:
        if original is not None:
            espn_module.ESPN_PATHS[league] = original
        elif league in espn_module.ESPN_PATHS:
            del espn_module.ESPN_PATHS[league]

    return result


# ── Public entry points ───────────────────────────────────────────────────────

def fetch_season(league: str, season: int) -> list[dict]:
    """
    Fetch all completed games for a soccer league in a given season.
    Routes to the best available data source for that league.
    """
    if league in FD_LEAGUE_CODES:
        return _fetch_fd_csv(league, season)
    elif league in ('mls', 'aleague'):
        return _fetch_espn_soccer(league, season)
    elif league in ('ucl', 'uel'):
        return _fetch_uefa(league, season)
    else:
        logger.warning(f'No data source configured for league: {league}')
        return []


def fetch_upcoming(league: str, days: int = 14) -> list[dict]:
    """Fetch upcoming fixtures for any soccer league."""
    return fetch_upcoming_fd(league, days)
