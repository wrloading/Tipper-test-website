#!/usr/bin/env python3
"""
Seed historical ELO ratings for sports that have no saved ratings file.

Fetches multiple seasons of completed games and runs them through the EloEngine
to produce calibrated starting ratings. Safe to re-run — overwrites existing ratings.

Usage:
    python3 seed.py --sport nbl
    python3 seed.py --sport nbl supernetball afl
    python3 seed.py           # seeds all sports that lack a ratings file
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from engine.elo import EloEngine
from engine.config import SPORT_CONFIGS, SEED_SEASONS, INGEST_SOURCE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RATINGS_DIR = Path('data/ratings')


def seed_sport(sport: str) -> None:
    config  = SPORT_CONFIGS[sport]
    engine  = EloEngine(sport=sport, config=config)
    seasons = SEED_SEASONS.get(sport, [])
    source  = INGEST_SOURCE.get(sport, 'espn')

    if not seasons:
        logger.warning(f'[{sport}] No SEED_SEASONS configured — skipping')
        return

    logger.info(f'[{sport}] Seeding {len(seasons)} seasons via {source}: {seasons}')

    for season in sorted(seasons):
        try:
            if source == 'espn':
                from ingest.espn import fetch_season
                games = fetch_season(sport, season)
            elif source == 'football_data':
                from ingest.football_data import fetch_season_fd
                games = fetch_season_fd(sport, season)
            elif source == 'champion_data':
                from ingest.champion_data import fetch_season
                games = fetch_season(sport, season)
            else:
                logger.warning(f'[{sport}] Unknown source: {source}')
                continue
        except Exception as e:
            logger.error(f'[{sport}] Season {season} fetch failed: {e}')
            continue

        for game in games:
            engine.process_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                home_score=game['home_score'],
                away_score=game['away_score'],
                date=game['date'],
                season=int(game['date'][:4]),
                neutral=game.get('neutral', False),
            )

        logger.info(f'[{sport}] Season {season}: {len(games)} games processed, '
                    f'{len(engine.ratings)} teams rated')

    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    ratings_file = str(RATINGS_DIR / f'{sport}.json')
    engine.save(ratings_file)
    logger.info(f'[{sport}] Saved ratings → {ratings_file}')

    top5 = engine.ratings_table()[:5]
    for r in top5:
        logger.info(f'  #{r["rank"]:2d}  {r["team"]:30s}  {r["rating"]:.0f}')


def main():
    parser = argparse.ArgumentParser(description='Seed ELO ratings from historical seasons')
    parser.add_argument(
        '--sport', nargs='+',
        choices=list(SPORT_CONFIGS.keys()),
        help='Sport(s) to seed (default: all without saved ratings)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-seed even if ratings file already exists',
    )
    args = parser.parse_args()

    if args.sport:
        sports = args.sport
    else:
        sports = [
            s for s in SPORT_CONFIGS
            if args.force or not (RATINGS_DIR / f'{s}.json').exists()
        ]

    if not sports:
        logger.info('All sports already have ratings. Use --force to re-seed.')
        return

    logger.info(f'Seeding {len(sports)} sport(s): {", ".join(sports)}')
    for sport in sports:
        try:
            seed_sport(sport)
        except Exception as e:
            logger.error(f'[{sport}] Seed failed: {e}', exc_info=True)


if __name__ == '__main__':
    main()
