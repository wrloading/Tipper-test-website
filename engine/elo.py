from __future__ import annotations

"""
Core ELO engine — margin-adjusted, home-aware, season-regressing.

Design principles:
- Margin-adjusted updates: larger wins carry more information than narrow wins
- Home advantage is an ELO offset, not a separate system — keeps it calibrated
- Season regression: teams revert toward the mean between seasons, accounting
  for roster turnover, coaching changes, and general uncertainty
- Full audit trail: every game update is logged for calibration and review
- Stateless between runs: ratings are serialised to/from JSON so the engine
  can be stopped and restarted without losing state
"""

import math
import json
import os
from typing import Optional


# ── Core math ─────────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
    """
    Logistic win probability for team A vs team B.
    home_advantage is added to A's effective rating (pass 0 for neutral venues).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - (rating_a + home_advantage)) / 400.0))


def margin_multiplier(margin: float) -> float:
    """
    Scales K by the magnitude of the winning margin.
    Adapted from FiveThirtyEight's NFL/NBA ELO models.
    Formula: ln(|margin| + 1), capped at 3 to prevent extreme swings on blowouts.
    A 1-point win → 0.69x, a 20-point win → 3.04x (capped at 3).
    """
    return min(math.log(abs(margin) + 1.0), 3.0)


def elo_update(
    rating_a: float,
    rating_b: float,
    result: float,
    k: float,
    margin: Optional[float] = None,
    home_advantage: float = 0.0,
) -> tuple[float, float]:
    """
    Compute updated ELO ratings after a single game.

    Args:
        rating_a:       Pre-game ELO of team A (home team)
        rating_b:       Pre-game ELO of team B (away team)
        result:         Actual outcome from A's perspective: 1=win, 0.5=draw, 0=loss
        k:              K-factor for this sport
        margin:         Absolute point/goal margin (optional, enables margin scaling)
        home_advantage: ELO offset added to A's rating for home games

    Returns:
        (new_rating_a, new_rating_b)
    """
    exp = expected_score(rating_a, rating_b, home_advantage)
    effective_k = k * (margin_multiplier(margin) if margin is not None else 1.0)
    delta = effective_k * (result - exp)
    return rating_a + delta, rating_b - delta


def season_regress(rating: float, mean: float, factor: float) -> float:
    """
    Regress a team's rating toward the league mean at season start.
    factor=0.33 moves the rating 1/3 of the way back to the mean.
    """
    return rating + (mean - rating) * factor


# ── Engine class ──────────────────────────────────────────────────────────────

class EloEngine:
    """
    Manages ELO ratings for a single sport/league.

    Usage:
        engine = EloEngine(config=SPORT_CONFIGS['nba'])
        engine.process_game('Lakers', 'Warriors', 112, 105, '2024-01-15')
        pred = engine.predict('Lakers', 'Warriors')
        engine.save('data/ratings/nba.json')
    """

    def __init__(self, sport: str, config: dict):
        self.sport         = sport
        self.k             = config['k']
        self.home_adv      = config['home_advantage']
        self.margin_scale  = config.get('margin_scale')          # Only used for logging
        self.regression    = config['season_regression']
        self.mean          = config.get('mean_rating', 1500.0)
        self.draw_base     = config.get('draw_base_rate', None)  # Soccer only
        self.draw_decay    = config.get('draw_decay', 300.0)     # Soccer only
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []
        self._current_season: Optional[int] = None

    # ── Rating access ─────────────────────────────────────────────────────────

    def rating(self, team: str) -> float:
        """Return a team's current rating, defaulting to league mean."""
        return self.ratings.get(team, self.mean)

    def set_rating(self, team: str, value: float) -> None:
        self.ratings[team] = value

    # ── Game processing ───────────────────────────────────────────────────────

    def process_game(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        date: str,
        season: int,
        neutral: bool = False,
    ) -> dict:
        """
        Process a completed game, update ratings, and log the event.
        Returns a dict with pre/post ratings and pre-game probabilities.
        """
        # Apply season regression when a new season starts
        if season != self._current_season:
            if self._current_season is not None:
                self._apply_season_regression()
            self._current_season = season

        r_home = self.rating(home_team)
        r_away = self.rating(away_team)
        home_adv = 0.0 if neutral else self.home_adv

        margin = home_score - away_score
        if margin > 0:
            result = 1.0
        elif margin < 0:
            result = 0.0
        else:
            result = 0.5

        pre_prob = expected_score(r_home, r_away, home_adv)

        new_home, new_away = elo_update(
            r_home, r_away,
            result=result,
            k=self.k,
            margin=margin,
            home_advantage=home_adv,
        )

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

        record = {
            'date':             date,
            'season':           season,
            'home':             home_team,
            'away':             away_team,
            'home_score':       home_score,
            'away_score':       away_score,
            'home_prob_pre':    round(pre_prob, 4),
            'home_rating_pre':  round(r_home, 2),
            'away_rating_pre':  round(r_away, 2),
            'home_rating_post': round(new_home, 2),
            'away_rating_post': round(new_away, 2),
            'neutral':          neutral,
        }
        self.history.append(record)
        return record

    def _apply_season_regression(self) -> None:
        for team in list(self.ratings.keys()):
            self.ratings[team] = season_regress(self.ratings[team], self.mean, self.regression)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
    ) -> dict:
        """
        Generate a win-probability prediction for an upcoming game.
        Returns home_prob, away_prob, draw_prob (soccer only), and expected margin.
        """
        r_home = self.rating(home_team)
        r_away = self.rating(away_team)
        home_adv = 0.0 if neutral else self.home_adv

        home_prob = expected_score(r_home, r_away, home_adv)

        # Draw probability for soccer: declines as ELO gap widens
        draw_prob = None
        if self.draw_base is not None:
            elo_diff = abs(r_home + home_adv - r_away)
            draw_prob = self.draw_base * math.exp(-elo_diff / self.draw_decay)
            # Normalise so home + draw + away = 1
            away_raw = 1.0 - home_prob
            total = home_prob + draw_prob + away_raw
            home_prob  = home_prob  / total
            draw_prob  = draw_prob  / total
            away_prob  = away_raw   / total
        else:
            away_prob = 1.0 - home_prob

        # Expected margin — rough conversion from ELO difference
        elo_edge = (r_home + home_adv - r_away)
        expected_margin = elo_edge / 28.0  # Empirically ~28 ELO pts per point/goal

        result = {
            'home':       home_team,
            'away':       away_team,
            'home_prob':  round(home_prob * 100, 1),
            'away_prob':  round(away_prob * 100, 1),
            'home_rating': round(r_home, 1),
            'away_rating': round(r_away, 1),
            'home_fav':   home_prob >= 0.5,
            'margin':     round(expected_margin, 1),
        }
        if draw_prob is not None:
            result['draw_prob'] = round(draw_prob * 100, 1)

        return result

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist current ratings and metadata to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'sport':          self.sport,
            'current_season': self._current_season,
            'ratings':        {t: round(r, 2) for t, r in sorted(
                                  self.ratings.items(), key=lambda x: -x[1])},
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str) -> None:
        """Load ratings from a previously saved JSON file."""
        if not os.path.exists(path):
            return
        with open(path) as f:
            payload = json.load(f)
        self.ratings = payload.get('ratings', {})
        self._current_season = payload.get('current_season')

    def ratings_table(self) -> list[dict]:
        """Return ratings sorted by strength for display/logging."""
        return [
            {'team': t, 'rating': round(r, 1), 'rank': i + 1}
            for i, (t, r) in enumerate(
                sorted(self.ratings.items(), key=lambda x: -x[1])
            )
        ]
