from __future__ import annotations

"""
Points-based spread prediction engine.

Unlike ELO (which predicts win probability from rating differences),
this model predicts the actual point spread from team offensive and
defensive output, using sport-specific inputs and home/away venue splits.

Model:
  home_expected = (home_team_off[home_venue] + away_team_def[away_venue]) / 2
  away_expected = (away_team_off[away_venue] + home_team_def[home_venue]) / 2
  spread        = home_expected − away_expected

Ratings are tracked as exponentially weighted moving averages (EWMA) so
recent form is weighted more heavily than old results. The decay rate
is calibrated per sport based on season length and game-to-game variance.

Separate home/away splits are used when enough data exists. When splits
are thin (early season or rare venue), the engine blends toward overall
averages and applies a sport-specific home advantage adjustment.

For sports/teams with insufficient recent data, the engine falls back to
an ELO-based spread using a sport-calibrated ELO-to-points divisor.
"""

from typing import Optional


# ── Per-sport configuration ───────────────────────────────────────────────────

SPREAD_CONFIGS: dict[str, dict] = {
    # window:         EWMA equivalent lookback in games; alpha = 2 / (window + 1)
    # home_adv_pts:   Points/goals added when home splits unavailable
    # min_games:      Minimum total games before using a team's rating
    # min_split:      Minimum home (or away) games before trusting splits
    # elo_divisor:    ELO points per game point/goal — used as fallback only

    # ── Basketball ────────────────────────────────────────────────────────────
    # NBA: 82-game season, ~115 pts/game per team, moderate home advantage.
    # FiveThirtyEight calibration: ~3 pts home advantage, ~28 ELO pts per point.
    'nba': {
        'window':       20,
        'home_adv_pts': 3.2,
        'min_games':    4,
        'min_split':    3,
        'elo_divisor':  28.0,
    },
    # WNBA: 40-game season, ~84 pts/game per team.
    'wnba': {
        'window':       15,
        'home_adv_pts': 2.8,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  22.0,
    },

    # ── American Football ─────────────────────────────────────────────────────
    # NFL: 17-game season, ~24 pts/game per team, declining home advantage.
    'nfl': {
        'window':       8,
        'home_adv_pts': 2.7,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  25.0,
    },

    # ── Ice Hockey ────────────────────────────────────────────────────────────
    # NHL: 82-game season, ~3 goals/game per team.
    'nhl': {
        'window':       20,
        'home_adv_pts': 0.25,
        'min_games':    4,
        'min_split':    3,
        'elo_divisor':  7.5,
    },

    # ── Baseball ──────────────────────────────────────────────────────────────
    # MLB: 162-game season, ~4.5 runs/game per team. High game-to-game variance
    # — larger window stabilises the estimates.
    'mlb': {
        'window':       30,
        'home_adv_pts': 0.18,
        'min_games':    8,
        'min_split':    4,
        'elo_divisor':  7.0,
    },

    # ── Australian Football ───────────────────────────────────────────────────
    # AFL: 22-game season, ~95 pts/game per team. Very strong home advantage,
    # especially for interstate travel (Perth, Brisbane, Adelaide clubs).
    'afl': {
        'window':       10,
        'home_adv_pts': 17.0,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  32.0,
    },

    # ── Rugby League ─────────────────────────────────────────────────────────
    # NRL: 24-game season, ~22 pts/game per team.
    'nrl': {
        'window':       10,
        'home_adv_pts': 5.5,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  14.0,
    },

    # ── Soccer ───────────────────────────────────────────────────────────────
    # All soccer leagues: ~1.3–1.5 goals/game per team. Spread in goals.
    # Home advantage varies by league culture.
    'epl': {
        'window':       10,
        'home_adv_pts': 0.38,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    'laliga': {
        'window':       10,
        'home_adv_pts': 0.42,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    'bundesliga': {
        'window':       10,
        'home_adv_pts': 0.37,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    'seriea': {
        'window':       10,
        'home_adv_pts': 0.40,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    'ligue1': {
        'window':       10,
        'home_adv_pts': 0.38,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    'mls': {
        'window':       12,
        'home_adv_pts': 0.45,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    'aleague': {
        'window':       10,
        'home_adv_pts': 0.37,
        'min_games':    3,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    # UCL/UEL: fewer games per team per season, cross-league competition.
    # Lower window — team quality is more stable and form matters less.
    'ucl': {
        'window':       8,
        'home_adv_pts': 0.32,
        'min_games':    2,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
    'uel': {
        'window':       8,
        'home_adv_pts': 0.28,
        'min_games':    2,
        'min_split':    2,
        'elo_divisor':  7.0,
    },
}


# ── Internal trackers ────────────────────────────────────────────────────────

class _VenueStats:
    """EWMA offensive/defensive stats for one team at one specific venue."""

    __slots__ = ('alpha', 'n', 'off', 'def_')

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.n     = 0
        self.off:  Optional[float] = None
        self.def_: Optional[float] = None

    def _update(self, current: Optional[float], value: float) -> float:
        return value if current is None else self.alpha * value + (1.0 - self.alpha) * current

    def record(self, scored: float, allowed: float) -> None:
        self.off  = self._update(self.off,  scored)
        self.def_ = self._update(self.def_, allowed)
        self.n   += 1


class _TeamStats:
    """
    EWMA offensive/defensive statistics for one team.
    Tracks overall averages, home/away splits, and per-venue stats.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha    = alpha
        self.n_home   = 0
        self.n_away   = 0
        self.off:      Optional[float] = None
        self.def_:     Optional[float] = None
        self.home_off: Optional[float] = None
        self.home_def: Optional[float] = None
        self.away_off: Optional[float] = None
        self.away_def: Optional[float] = None
        # venue name → _VenueStats (only populated when venue is known)
        self.venues:   dict[str, _VenueStats] = {}

    def _update(self, current: Optional[float], value: float) -> float:
        return value if current is None else self.alpha * value + (1.0 - self.alpha) * current

    def record(self, scored: float, allowed: float, is_home: bool, venue: str = '') -> None:
        self.off  = self._update(self.off,  scored)
        self.def_ = self._update(self.def_, allowed)
        if is_home:
            self.home_off = self._update(self.home_off, scored)
            self.home_def = self._update(self.home_def, allowed)
            self.n_home  += 1
        else:
            self.away_off = self._update(self.away_off, scored)
            self.away_def = self._update(self.away_def, allowed)
            self.n_away  += 1

        if venue:
            if venue not in self.venues:
                self.venues[venue] = _VenueStats(self.alpha)
            self.venues[venue].record(scored, allowed)

    @property
    def n_games(self) -> int:
        return self.n_home + self.n_away


# ── Public engine ─────────────────────────────────────────────────────────────

class SpreadEngine:
    """
    Predicts point spreads using per-team offensive/defensive ratings.

    Usage:
        engine = SpreadEngine('nba')
        for game in games_sorted_oldest_first:
            engine.record_game(game['home_team'], game['away_team'],
                               game['home_score'], game['away_score'])
        spread = engine.predict_spread('Lakers', 'Warriors')
        # Returns float (positive = home favored) or None if insufficient data.
    """

    def __init__(self, sport: str) -> None:
        cfg = SPREAD_CONFIGS.get(sport, SPREAD_CONFIGS['epl'])
        self.sport         = sport
        self.alpha         = 2.0 / (cfg['window'] + 1)
        self.min_games     = cfg['min_games']
        self.min_split     = cfg['min_split']
        self.home_adv_pts  = cfg['home_adv_pts']
        self.elo_divisor   = cfg['elo_divisor']
        self.teams: dict[str, _TeamStats] = {}
        # Slow EWMA for league-wide scoring average (used for regression)
        self._league_off: Optional[float] = None
        self._league_alpha = 0.005

    # ── Data ingestion ────────────────────────────────────────────────────────

    def record_game(
        self,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        neutral: bool = False,
        venue: str = '',
    ) -> None:
        """Feed a completed game into the engine (call in chronological order)."""
        for name in (home_team, away_team):
            if name not in self.teams:
                self.teams[name] = _TeamStats(self.alpha)

        v = venue.strip() if venue else ''
        self.teams[home_team].record(home_score, away_score, is_home=not neutral, venue=v)
        self.teams[away_team].record(away_score, home_score, is_home=False, venue=v)

        # Update league scoring average
        for score in (home_score, away_score):
            self._league_off = (
                score if self._league_off is None
                else self._league_alpha * score + (1.0 - self._league_alpha) * self._league_off
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _league_avg(self) -> float:
        return self._league_off if self._league_off is not None else 0.0

    MIN_VENUE_GAMES = 5  # Minimum home games at a venue before trusting venue stats

    def _off(self, team: str, prefer_home: bool, venue: str = '') -> float:
        """
        Offensive rating for a team. Priority order:
          1. Venue-specific EWMA (if venue known and enough games there)
          2. Home/away split EWMA
          3. Overall EWMA
          4. League average
        """
        s = self.teams.get(team)
        lg = self._league_avg
        if s is None:
            return lg

        # 1. Venue-specific
        if venue and venue in s.venues:
            vs = s.venues[venue]
            if vs.n >= self.MIN_VENUE_GAMES and vs.off is not None:
                return vs.off
            # Partial blend: weight by how much venue data we have
            if vs.n > 0 and vs.off is not None:
                blend = vs.n / self.MIN_VENUE_GAMES

                # Blend against split or overall as the fallback
                fallback = self._off_split(s, prefer_home, lg)
                return blend * vs.off + (1.0 - blend) * fallback

        # 2. Home/away split
        return self._off_split(s, prefer_home, lg)

    def _off_split(self, s: _TeamStats, prefer_home: bool, lg: float) -> float:
        if prefer_home:
            if s.home_off is not None and s.n_home >= self.min_split:
                return s.home_off
            if s.home_off is not None and s.n_home > 0 and s.off is not None:
                blend = s.n_home / self.min_split
                return blend * s.home_off + (1.0 - blend) * s.off
        else:
            if s.away_off is not None and s.n_away >= self.min_split:
                return s.away_off
            if s.away_off is not None and s.n_away > 0 and s.off is not None:
                blend = s.n_away / self.min_split
                return blend * s.away_off + (1.0 - blend) * s.off
        return s.off if s.off is not None else lg

    def _def(self, team: str, prefer_home: bool, venue: str = '') -> float:
        """
        Defensive rating (pts allowed) for a team. Same priority order as _off.
        """
        s = self.teams.get(team)
        lg = self._league_avg
        if s is None:
            return lg

        # 1. Venue-specific
        if venue and venue in s.venues:
            vs = s.venues[venue]
            if vs.n >= self.MIN_VENUE_GAMES and vs.def_ is not None:
                return vs.def_
            if vs.n > 0 and vs.def_ is not None:
                blend = vs.n / self.MIN_VENUE_GAMES
                fallback = self._def_split(s, prefer_home, lg)
                return blend * vs.def_ + (1.0 - blend) * fallback

        # 2. Home/away split
        return self._def_split(s, prefer_home, lg)

    def _def_split(self, s: _TeamStats, prefer_home: bool, lg: float) -> float:
        if prefer_home:
            if s.home_def is not None and s.n_home >= self.min_split:
                return s.home_def
            if s.home_def is not None and s.n_home > 0 and s.def_ is not None:
                blend = s.n_home / self.min_split
                return blend * s.home_def + (1.0 - blend) * s.def_
        else:
            if s.away_def is not None and s.n_away >= self.min_split:
                return s.away_def
            if s.away_def is not None and s.n_away > 0 and s.def_ is not None:
                blend = s.n_away / self.min_split
                return blend * s.away_def + (1.0 - blend) * s.def_
        return s.def_ if s.def_ is not None else lg

    def _has_split(self, team: str, home: bool) -> bool:
        s = self.teams.get(team)
        if s is None:
            return False
        return (s.n_home if home else s.n_away) >= self.min_split

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_spread(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        venue: str = '',
    ) -> Optional[float]:
        """
        Predict the point spread (positive = home team favored).
        Returns None if neither team has enough data to make a prediction.

        Resolution order for each team's stats:
          1. Per-venue EWMA at this specific ground (if venue provided and enough data)
          2. Home/away split EWMA
          3. Overall EWMA
          4. League average (last resort)

        Home advantage is embedded in venue/split stats when available. A
        proportional flat adjustment is added only when falling back to overall
        averages that don't encode venue context.
        """
        h = self.teams.get(home_team)
        a = self.teams.get(away_team)

        h_games = h.n_games if h else 0
        a_games = a.n_games if a else 0

        if h_games < self.min_games and a_games < self.min_games:
            return None

        v = venue.strip() if venue else ''

        if neutral:
            h_off = self._off(home_team, prefer_home=False)
            h_def = self._def(home_team, prefer_home=False)
            a_off = self._off(away_team, prefer_home=False)
            a_def = self._def(away_team, prefer_home=False)
            home_adv = 0.0
        else:
            # Venue stats only used for the HOME team — they play at this ground
            # every home game so the sample is meaningful. Away teams visit each
            # specific venue too rarely for reliable per-venue stats; use their
            # general away split instead.
            h_off = self._off(home_team, prefer_home=True,  venue=v)
            h_def = self._def(home_team, prefer_home=True,  venue=v)
            a_off = self._off(away_team, prefer_home=False)
            a_def = self._def(away_team, prefer_home=False)

            h_has_venue = bool(v and h and v in h.venues and h.venues[v].n >= self.MIN_VENUE_GAMES)
            h_split = h_has_venue or self._has_split(home_team, home=True)
            a_split = self._has_split(away_team, home=False)
            missing  = (0.0 if h_split else 0.5) + (0.0 if a_split else 0.5)
            home_adv = self.home_adv_pts * missing

        home_expected = (h_off + a_def) / 2.0
        away_expected = (a_off + h_def) / 2.0

        return round(home_expected - away_expected + home_adv, 1)

    def elo_fallback_spread(
        self,
        home_elo: float,
        away_elo: float,
        elo_home_adv: float = 0.0,
        neutral: bool = False,
    ) -> float:
        """
        Fallback spread from ELO when no performance data is available.
        Uses a sport-calibrated ELO-to-points divisor rather than the
        old universal 28.0 constant.
        """
        adv = 0.0 if neutral else elo_home_adv
        return round((home_elo + adv - away_elo) / self.elo_divisor, 1)
