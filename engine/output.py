from __future__ import annotations

"""
Output generator — produces predictions_sports.json.

Takes the current ELO engine state + upcoming/recent game lists and
generates the JSON payload consumed by the Tipper app.

Output shape (matches what the app already expects):
{
  "meta": { "updated": "ISO timestamp", "version": "2.0.0" },
  "sports": {
    "nba": {
      "ratings": { "team": 1542, ... },       // For transparency/display
      "upcoming": [
        {
          "home": "Lakers", "away": "Warriors",
          "date": "2025-04-22",
          "home_prob": 58.3,                  // percentage
          "away_prob": 41.7,
          "draw_prob": null,                  // only for soccer
          "margin": 4.2,                      // predicted home margin
          "home_fav": true,
          "locked": false
        }
      ],
      "recent": [
        {
          "home": "Lakers", "away": "Celtics",
          "date": "2025-04-18",
          "home_prob": 52.1,                  // pre-game probability (for settlement)
          "draw_prob": null,
          "winner": "Celtics",                // actual winner
          "home_score": 108,
          "away_score": 115
        }
      ]
    }
  }
}

The "recent" section is critical for tip settlement — it lets the app
backfill telo_expected values for tips that were submitted with 50/50 defaults.
"""

import json
import os
from datetime import datetime, timezone, date, timedelta
from typing import Optional

from engine.elo import EloEngine

VERSION = '2.0.0'


def _is_locked(game_datetime: str) -> bool:
    """A game is locked once its start time has passed."""
    try:
        dt = datetime.fromisoformat(game_datetime.replace('Z', '+00:00'))
        return dt <= datetime.now(timezone.utc)
    except (ValueError, AttributeError):
        return False


def build_sport_output(
    engine: EloEngine,
    upcoming_games: list[dict],
    recent_games: list[dict],
) -> dict:
    """
    Build the output section for a single sport.

    upcoming_games: list from fetch_upcoming() — not yet started
    recent_games:   list from fetch_recent()   — completed in last 14 days
    """

    # ── Upcoming predictions ──────────────────────────────────────────────────
    upcoming_out = []
    for game in upcoming_games:
        pred = engine.predict(
            game['home_team'],
            game['away_team'],
            neutral=game.get('neutral', False),
        )
        entry: dict = {
            'home':      game['home_team'],
            'away':      game['away_team'],
            'date':      game.get('date', ''),
            'datetime':  game.get('datetime', game.get('date', '')),
            'home_prob': pred['home_prob'],
            'away_prob': pred['away_prob'],
            'margin':    pred['margin'],
            'home_fav':  pred['home_fav'],
            'locked':    _is_locked(game.get('datetime', '')),
            'venue':     game.get('venue', ''),
            'game_id':   game.get('id', ''),
        }
        if 'draw_prob' in pred:
            entry['draw_prob'] = pred['draw_prob']
        else:
            entry['draw_prob'] = None

        upcoming_out.append(entry)

    # ── Recent results (with pre-game probability) ────────────────────────────
    # We need the pre-game probability for these games so the app can use them
    # for tip settlement backfill. We reconstruct this from current ratings
    # (slightly less accurate than pre-game, but close enough for settlement).
    recent_out = []
    for game in recent_games:
        pred = engine.predict(
            game['home_team'],
            game['away_team'],
            neutral=game.get('neutral', False),
        )
        entry: dict = {
            'home':       game['home_team'],
            'away':       game['away_team'],
            'date':       game.get('date', ''),
            'home_prob':  pred['home_prob'],
            'winner':     game.get('winner'),
            'home_score': game.get('home_score'),
            'away_score': game.get('away_score'),
        }
        if 'draw_prob' in pred:
            entry['draw_prob'] = pred['draw_prob']
        else:
            entry['draw_prob'] = None

        recent_out.append(entry)

    return {
        'ratings':  engine.ratings_table(),
        'upcoming': upcoming_out,
        'recent':   recent_out,
    }


def build_full_output(sport_sections: dict[str, dict]) -> dict:
    """
    Wrap all sport sections in the top-level output envelope.
    """
    return {
        'meta': {
            'updated':  datetime.now(timezone.utc).isoformat(),
            'version':  VERSION,
            'engine':   'TELO v2 — Margin-adjusted ELO with home advantage and season regression',
        },
        'sports': sport_sections,
    }


def write_output(payload: dict, path: str) -> None:
    """Write the predictions JSON to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    size_kb = os.path.getsize(path) / 1024
    print(f'[output] Written to {path} ({size_kb:.1f} KB)')


def load_output(path: str) -> Optional[dict]:
    """Load an existing predictions JSON file."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
