from __future__ import annotations

"""
Per-sport ELO configuration.

Parameters explained:
  k               — K-factor: how much a single game moves the rating.
                    High K = reactive (good for short seasons like NFL).
                    Low K  = stable  (good for long seasons like MLB).

  home_advantage  — ELO points added to home team's effective rating.
                    100 pts ≈ 64% home win probability vs equal opponent.
                    Derived from historical home win rates per sport.

  margin_scale    — Informational only. Typical winning margin — logged
                    alongside updates so calibration is interpretable.

  season_regression — Fraction to regress toward the mean each off-season.
                    0.33 = move 1/3 of the way back to 1500.
                    Higher for sports with more roster churn (NFL, AFL).

  draw_base_rate  — Soccer only. Baseline draw probability when teams are
                    perfectly matched. Varies by league culture.
                    EPL ≈ 0.26, Serie A ≈ 0.30, La Liga ≈ 0.28.

  draw_decay      — Soccer only. Controls how quickly draw probability
                    falls as ELO gap widens. Higher = slower decay.

Sources for parameter choices:
  - FiveThirtyEight ELO methodology (NFL, NBA, MLB)
  - Dixon-Coles Poisson model (soccer draw rates)
  - Historical home advantage studies (AFL: ~59%, EPL: ~46% home wins)
"""

SPORT_CONFIGS: dict[str, dict] = {

    # ── Australian Football League ────────────────────────────────────────────
    # 22 home-and-away games + finals. Very high home advantage (travel costs
    # for interstate clubs). No draws. High variance per game → K=32.
    'afl': {
        'k':                32,
        'home_advantage':   65,    # ~59% home win rate historically
        'margin_scale':     30,    # Typical AFL winning margin ~30–35 pts
        'season_regression': 0.33,
        'mean_rating':      1500,
    },

    # ── National Basketball Association ───────────────────────────────────────
    # 82 games. Moderate home advantage. Best-player variance is high.
    # FiveThirtyEight uses K=20, home_adv=100 (≈3 pts on court).
    'nba': {
        'k':                20,
        'home_advantage':   100,
        'margin_scale':     12,
        'season_regression': 0.25,
        'mean_rating':      1500,
    },

    # ── Women's National Basketball Association ───────────────────────────────
    'wnba': {
        'k':                24,
        'home_advantage':   80,
        'margin_scale':     10,
        'season_regression': 0.30,
        'mean_rating':      1500,
    },

    # ── National Football League ──────────────────────────────────────────────
    # 17 games. High single-game variance. QB quality is enormous.
    # FiveThirtyEight K=20 with home_adv=65 (≈2.5 pts).
    'nfl': {
        'k':                20,
        'home_advantage':   65,
        'margin_scale':     7,
        'season_regression': 0.33,
        'mean_rating':      1500,
    },

    # ── National Rugby League ─────────────────────────────────────────────────
    # 24 games + finals. Similar structure to AFL. Good home advantage.
    'nrl': {
        'k':                32,
        'home_advantage':   60,
        'margin_scale':     12,
        'season_regression': 0.33,
        'mean_rating':      1500,
    },

    # ── English Premier League ────────────────────────────────────────────────
    # 38 games. Draws are significant (~25% of matches).
    # Home advantage lower than most sports — EPL is very even.
    'epl': {
        'k':                20,
        'home_advantage':   80,
        'margin_scale':     1.5,   # Goals
        'season_regression': 0.20,
        'mean_rating':      1500,
        'draw_base_rate':   0.26,  # 26% draw rate when evenly matched
        'draw_decay':       280.0,
    },

    # ── La Liga ───────────────────────────────────────────────────────────────
    'laliga': {
        'k':                20,
        'home_advantage':   90,
        'margin_scale':     1.5,
        'season_regression': 0.20,
        'mean_rating':      1500,
        'draw_base_rate':   0.28,
        'draw_decay':       280.0,
    },

    # ── Bundesliga ────────────────────────────────────────────────────────────
    'bundesliga': {
        'k':                20,
        'home_advantage':   85,
        'margin_scale':     1.5,
        'season_regression': 0.20,
        'mean_rating':      1500,
        'draw_base_rate':   0.25,
        'draw_decay':       280.0,
    },

    # ── Serie A ───────────────────────────────────────────────────────────────
    # Historically highest draw rate in top European leagues.
    'seriea': {
        'k':                20,
        'home_advantage':   90,
        'margin_scale':     1.5,
        'season_regression': 0.20,
        'mean_rating':      1500,
        'draw_base_rate':   0.30,
        'draw_decay':       260.0,
    },

    # ── Ligue 1 ───────────────────────────────────────────────────────────────
    'ligue1': {
        'k':                20,
        'home_advantage':   85,
        'margin_scale':     1.5,
        'season_regression': 0.20,
        'mean_rating':      1500,
        'draw_base_rate':   0.27,
        'draw_decay':       280.0,
    },

    # ── Major League Soccer ───────────────────────────────────────────────────
    # Lower quality, higher variance. Playoff format changes late season.
    'mls': {
        'k':                24,
        'home_advantage':   100,   # Strong home advantage in MLS
        'margin_scale':     1.5,
        'season_regression': 0.33,
        'mean_rating':      1500,
        'draw_base_rate':   0.25,
        'draw_decay':       260.0,
    },

    # ── A-League (Australia) ──────────────────────────────────────────────────
    'aleague': {
        'k':                24,
        'home_advantage':   80,
        'margin_scale':     1.5,
        'season_regression': 0.33,
        'mean_rating':      1500,
        'draw_base_rate':   0.27,
        'draw_decay':       260.0,
    },

    # ── UEFA Champions League ─────────────────────────────────────────────────
    # Cross-league competition. Home advantage significant in knockout legs.
    'ucl': {
        'k':                16,
        'home_advantage':   80,
        'margin_scale':     1.5,
        'season_regression': 0.10,  # Lower — ratings persist across club seasons
        'mean_rating':      1600,   # Higher baseline (only elite clubs participate)
        'draw_base_rate':   0.26,
        'draw_decay':       280.0,
    },

    # ── UEFA Europa League ────────────────────────────────────────────────────
    'uel': {
        'k':                16,
        'home_advantage':   80,
        'margin_scale':     1.5,
        'season_regression': 0.10,
        'mean_rating':      1540,
        'draw_base_rate':   0.26,
        'draw_decay':       280.0,
    },

    # ── Major League Baseball ─────────────────────────────────────────────────
    # 162 games. Enormous sample size → very low K. Starting pitcher dominates.
    # FiveThirtyEight uses K=4 for MLB.
    'mlb': {
        'k':                4,
        'home_advantage':   24,    # Smallest home advantage in major US sports
        'margin_scale':     1.5,   # Runs
        'season_regression': 0.33,
        'mean_rating':      1500,
    },

    # ── National Hockey League ────────────────────────────────────────────────
    # 82 games. Overtime/shootout common — treat OT win as 0.75, SO win as 0.6.
    # Goalie matchup is the biggest single-game variable.
    'nhl': {
        'k':                8,
        'home_advantage':   60,
        'margin_scale':     1.0,   # Goals
        'season_regression': 0.33,
        'mean_rating':      1500,
    },
}


# ── Season year ranges for historical seeding ─────────────────────────────────
# (start_season, end_season) — inclusive. 5 full seasons back from 2025.

SEED_SEASONS: dict[str, list[int]] = {
    'afl':        [2020, 2021, 2022, 2023, 2024],
    'nba':        [2020, 2021, 2022, 2023, 2024],
    'wnba':       [2020, 2021, 2022, 2023, 2024],
    'nfl':        [2020, 2021, 2022, 2023, 2024],
    'nrl':        [2020, 2021, 2022, 2023, 2024],
    'epl':        [2020, 2021, 2022, 2023, 2024],
    'laliga':     [2020, 2021, 2022, 2023, 2024],
    'bundesliga': [2020, 2021, 2022, 2023, 2024],
    'seriea':     [2020, 2021, 2022, 2023, 2024],
    'ligue1':     [2020, 2021, 2022, 2023, 2024],
    'mls':        [2020, 2021, 2022, 2023, 2024],
    'aleague':    [2021, 2022, 2023, 2024, 2025],  # A-League seasons span calendar years
    'ucl':        [2020, 2021, 2022, 2023, 2024],
    'uel':        [2020, 2021, 2022, 2023, 2024],
    'mlb':        [2020, 2021, 2022, 2023, 2024],
    'nhl':        [2021, 2022, 2023, 2024, 2025],
}


# ── Data source routing ───────────────────────────────────────────────────────
# Defines which ingestion module handles each sport.

INGEST_SOURCE: dict[str, str] = {
    'afl':        'espn',         # ESPN Australian football
    'nba':        'espn',
    'wnba':       'espn',
    'nfl':        'espn',
    'nrl':        'espn',
    'epl':        'football_data',
    'laliga':     'football_data',
    'bundesliga': 'football_data',
    'seriea':     'football_data',
    'ligue1':     'football_data',
    'mls':        'football_data',
    'aleague':    'football_data',
    'ucl':        'football_data',
    'uel':        'football_data',
    'mlb':        'espn',
    'nhl':        'espn',
}
