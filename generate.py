from __future__ import annotations

"""
Daily prediction generator.

Loads current ELO ratings, fetches recent results + upcoming fixtures,
updates ratings with any new completed games, then writes predictions_sports.json.

Run this daily (or on demand). Designed to be idempotent — safe to run multiple
times; duplicate games are deduplicated by game ID.

Usage:
    python generate.py                   # Generate for all sports
    python generate.py --sport afl nba   # Specific sports only
    python generate.py --output path/to/predictions.json
"""

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path

from engine.elo import EloEngine
from engine.config import SPORT_CONFIGS, INGEST_SOURCE
from engine.output import build_sport_output, build_full_output, write_output
from engine.spread import SpreadEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RATINGS_DIR   = Path('data/ratings')
CACHE_DIR     = Path('data/cache')
OUTPUT_PATH   = Path('output/predictions_sports.json')
RECENT_DAYS   = 14
SPREAD_DAYS   = 90   # Longer lookback to warm up SpreadEngine rolling averages
UPCOMING_DAYS = 14


def load_engine(sport: str) -> EloEngine:
    """Load an EloEngine from saved ratings, or create a fresh one."""
    config  = SPORT_CONFIGS[sport]
    engine  = EloEngine(sport=sport, config=config)
    ratings_file = str(RATINGS_DIR / f'{sport}.json')

    if os.path.exists(ratings_file):
        engine.load(ratings_file)
        logger.info(f'[{sport}] Loaded {len(engine.ratings)} team ratings')
    else:
        logger.warning(f'[{sport}] No saved ratings — using fresh engine (run seed.py first)')

    return engine


def get_processed_ids(sport: str) -> set[str]:
    """
    Load the set of game IDs already processed, to avoid double-updating ratings.
    Stored alongside ratings.
    """
    p = RATINGS_DIR / f'{sport}_processed.json'
    if p.exists():
        with open(p) as f:
            return set(json.load(f))
    return set()


def save_processed_ids(sport: str, ids: set[str]) -> None:
    p = RATINGS_DIR / f'{sport}_processed.json'
    with open(p, 'w') as f:
        json.dump(sorted(ids), f)


def fetch_recent(sport: str) -> list[dict]:
    """Fetch recently completed games for ratings update."""
    source = INGEST_SOURCE.get(sport, 'espn')
    if source == 'espn':
        from ingest.espn import fetch_recent as espn_recent
        return espn_recent(sport, days=RECENT_DAYS)
    elif source == 'football_data':
        from ingest.football_data import fetch_recent_fd
        return fetch_recent_fd(sport, days=RECENT_DAYS)
    return []


def fetch_spread_history(sport: str) -> list[dict]:
    """
    Load enough game history to warm up the SpreadEngine rolling averages.
    Strategy:
      1. Load cached season files (seeded data) for the last 2 seasons.
      2. Supplement with a 90-day recent fetch for current-season form.
    Games are returned sorted oldest-first.
    """
    games: dict[str, dict] = {}  # keyed by game id to deduplicate

    # Load from seeded cache files
    current_year = date.today().year
    for season in [current_year - 1, current_year]:
        cache_path = CACHE_DIR / f'{sport}_{season}.json'
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    for g in json.load(f):
                        gid = g.get('id') or f'{g["date"]}_{g["home_team"]}_{g["away_team"]}'
                        games[gid] = g
            except Exception:
                pass

    # Supplement with extended recent fetch
    source = INGEST_SOURCE.get(sport, 'espn')
    try:
        if source == 'espn':
            from ingest.espn import fetch_recent as espn_recent
            for g in espn_recent(sport, days=SPREAD_DAYS):
                gid = g.get('id') or f'{g["date"]}_{g["home_team"]}_{g["away_team"]}'
                games[gid] = g
        elif source == 'football_data':
            from ingest.football_data import fetch_recent_fd
            for g in fetch_recent_fd(sport, days=SPREAD_DAYS):
                gid = g.get('id') or f'{g["date"]}_{g["home_team"]}_{g["away_team"]}'
                games[gid] = g
    except Exception:
        pass

    return sorted(games.values(), key=lambda g: g.get('date', ''))


def build_spread_engine(sport: str, recent_games: list[dict]) -> SpreadEngine:
    """
    Initialise a SpreadEngine with historical + recent game data.
    recent_games (from the ELO update step) are appended last so they
    represent the freshest form.
    """
    engine = SpreadEngine(sport)
    history = fetch_spread_history(sport)

    # Feed history first (oldest → newest), then append any recent games
    # not already in history
    history_ids = {g.get('id') or '' for g in history}
    extra = [
        g for g in recent_games
        if (g.get('id') or '') not in history_ids
    ]

    for g in history + extra:
        engine.record_game(
            g['home_team'], g['away_team'],
            g['home_score'], g['away_score'],
            g.get('neutral', False),
        )

    n_teams = len(engine.teams)
    logger.info(f'[{sport}] SpreadEngine: {n_teams} teams, league avg ≈ {engine._league_avg:.1f}')
    return engine


def fetch_upcoming(sport: str) -> list[dict]:
    """Fetch upcoming fixtures for prediction generation."""
    source = INGEST_SOURCE.get(sport, 'espn')
    if source == 'espn':
        from ingest.espn import fetch_upcoming as espn_upcoming
        return espn_upcoming(sport, days=UPCOMING_DAYS)
    elif source == 'football_data':
        from ingest.football_data import fetch_upcoming as fd_upcoming
        return fd_upcoming(sport, days=UPCOMING_DAYS)
    return []


def _winner_from_scores(home_score: int, away_score: int,
                         home_team: str, away_team: str) -> str:
    if home_score > away_score:
        return home_team
    elif away_score > home_score:
        return away_team
    return None  # Draw


def generate_sport(sport: str) -> dict:
    """
    Full generate cycle for one sport.
    Returns the sport's output section dict, or None on failure.
    """
    logger.info(f'[{sport.upper()}] Starting generate...')

    engine       = load_engine(sport)
    processed    = get_processed_ids(sport)
    new_processed = set()

    # ── Step 1: Update ratings with any new completed games ───────────────────
    recent_games = fetch_recent(sport)
    new_game_count = 0

    for game in recent_games:
        gid = game.get('id', f'{game["date"]}_{game["home_team"]}_{game["away_team"]}')
        if gid in processed:
            continue

        engine.process_game(
            home_team=game['home_team'],
            away_team=game['away_team'],
            home_score=game['home_score'],
            away_score=game['away_score'],
            date=game['date'],
            season=int(game['date'][:4]),
            neutral=game.get('neutral', False),
        )
        new_processed.add(gid)
        new_game_count += 1

    if new_game_count > 0:
        logger.info(f'[{sport}] Updated ratings with {new_game_count} new games')
        all_processed = processed | new_processed
        save_processed_ids(sport, all_processed)
        engine.save(str(RATINGS_DIR / f'{sport}.json'))
    else:
        logger.info(f'[{sport}] No new games to process')

    # ── Step 2: Annotate recent games with winner info for output ─────────────
    for game in recent_games:
        game['winner'] = _winner_from_scores(
            game['home_score'], game['away_score'],
            game['home_team'], game['away_team']
        )

    # ── Step 3: Fetch upcoming fixtures ───────────────────────────────────────
    upcoming_games = fetch_upcoming(sport)
    logger.info(f'[{sport}] {len(upcoming_games)} upcoming games, {len(recent_games)} recent games')

    # ── Step 4: Build SpreadEngine from historical + recent data ─────────────
    spread_engine = build_spread_engine(sport, recent_games)

    # ── Step 5: Build output section ──────────────────────────────────────────
    section = build_sport_output(engine, upcoming_games, recent_games, spread_engine)

    upcoming_count = len(section['upcoming'])
    recent_count   = len(section['recent'])
    logger.info(f'[{sport.upper()}] Done — {upcoming_count} predictions, {recent_count} recent results')

    return section


def main():
    parser = argparse.ArgumentParser(description='Generate TELO predictions JSON')
    parser.add_argument(
        '--sport', nargs='+',
        choices=list(SPORT_CONFIGS.keys()),
        help='Sports to generate (default: all)',
    )
    parser.add_argument(
        '--output', type=str,
        default=str(OUTPUT_PATH),
        help=f'Output path (default: {OUTPUT_PATH})',
    )
    args = parser.parse_args()

    sports = args.sport or list(SPORT_CONFIGS.keys())
    logger.info(f'Generating predictions for {len(sports)} sport(s): {", ".join(sports)}')

    sport_sections: dict[str, dict] = {}

    for sport in sports:
        try:
            section = generate_sport(sport)
            if section:
                sport_sections[sport] = section
        except Exception as e:
            logger.error(f'[{sport}] Generation failed: {e}', exc_info=True)

    if not sport_sections:
        logger.error('No sports generated successfully. Aborting.')
        return

    payload = build_full_output(sport_sections)
    write_output(payload, args.output)

    total_upcoming = sum(len(s['upcoming']) for s in sport_sections.values())
    total_recent   = sum(len(s['recent'])   for s in sport_sections.values())
    logger.info(f'Complete — {total_upcoming} upcoming predictions, {total_recent} recent results across {len(sport_sections)} sports')


if __name__ == '__main__':
    main()
