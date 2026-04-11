#!/usr/bin/env python3
"""
TELO Model — AFL ELO-based prediction engine for Tipper.

Fetches live results from the Squiggle public API, computes TELO ratings
for all AFL teams, generates game predictions and win probabilities for
upcoming fixtures, runs a Monte Carlo finals simulation, and writes
data/predictions.json for the website to consume.

Ratings are built from the last 3 seasons with recency bias:
  2 seasons ago : K × 0.50
  1 season ago  : K × 0.75
  Current season: K × 1.00
Season-start regression toward the mean is applied between each year.

Advanced prediction factors (applied to display predictions only):
  - Form streaks       : recent hot/cold runs boost or penalise effective rating
  - Venue-specific HGA : per-venue home advantage computed from 3-year history
  - Head-to-head       : historical matchup record between specific pairs
  - Travel / fatigue   : interstate travel and short-break penalties
  - Injury overrides   : manual TELO adjustment dict (INJURY_OVERRIDES)

Usage:
    python telo.py              # Current year
    python telo.py --year 2025  # Specific year (backfill)
    python telo.py --dry-run    # Print output, don't write files
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import requests

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

SQUIGGLE_BASE = "https://api.squiggle.com.au/"
SQUIGGLE_UA   = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"

# Core TELO model parameters
INITIAL_TELO   = 1500.0   # Starting rating for all teams
K_FACTOR       = 40.0     # Base K — how fast ratings move (higher = more volatile)
HGA            = 65.0     # Default home ground advantage in TELO points
MARGIN_SCALE   = 0.025    # Log-margin multiplier weight (rewards blowouts slightly)
SEASON_REGRESS = 0.25     # Fraction regressed toward mean at season start

# Recency bias: K-factor scaling per season (oldest → newest)
RECENCY_WEIGHTS = [0.50, 0.75, 1.00]
HISTORY_YEARS   = 3       # Seasons to look back (including current)

# Margin prediction calibration
TELO_TO_MARGIN = 0.25     # Each 100 TELO pts ≈ ~25 pts AFL margin

# Monte Carlo
MC_SIMULATIONS = 12000
FINALS_SPOTS   = 8
AEST           = ZoneInfo("Australia/Melbourne")

# ─── ADVANCED FACTORS ────────────────────────────────────────────────────────

# Form streak: rolling last-N results within current season
FORM_WINDOW   = 5       # number of recent games tracked
FORM_MAX_ADJ  = 12.0    # max ±TELO pts from form streak

# Venue-specific HGA: computed from 3-year historical home win rate per venue
VENUE_MIN_GAMES = 20    # min games before trusting computed venue HGA

# Head-to-head: historical record between specific pairs
H2H_MIN_GAMES  = 8      # min meetings before applying H2H adjustment
H2H_MAX_ADJ    = 12.0   # max ±TELO pts from H2H dominance

# Travel and fatigue
TRAVEL_PENALTY  = 10.0  # TELO pts for interstate travel to venue
FATIGUE_DAYS    = 6     # days or fewer = short break
FATIGUE_PENALTY = 6.0   # TELO pts for short-break fatigue

# Manual injury/availability overrides — update before each round as needed.
# Format: {"Team Name": delta_telo}
# Negative = key players out, positive = squad at full strength after injury return.
INJURY_OVERRIDES: dict[str, float] = {
    # "Melbourne": -15.0,
    # "Collingwood": -20.0,
}

# Team home states (for interstate travel detection)
TEAM_STATE: dict[str, str] = {
    "Adelaide":         "SA",
    "Brisbane":         "QLD",
    "Carlton":          "VIC",
    "Collingwood":      "VIC",
    "Essendon":         "VIC",
    "Fremantle":        "WA",
    "Geelong":          "VIC",
    "Gold Coast":       "QLD",
    "GWS":              "NSW",
    "Hawthorn":         "VIC",
    "Melbourne":        "VIC",
    "North Melbourne":  "VIC",
    "Port Adelaide":    "SA",
    "Richmond":         "VIC",
    "St Kilda":         "VIC",
    "Sydney":           "NSW",
    "West Coast":       "WA",
    "Western Bulldogs": "VIC",
}

# ─── SQUIGGLE API ─────────────────────────────────────────────────────────────

def squiggle_get(query: str) -> dict:
    """Make a request to the Squiggle API."""
    url = f"{SQUIGGLE_BASE}?{query}"
    headers = {"User-Agent": SQUIGGLE_UA}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"ERROR fetching {url}: {e}", file=sys.stderr)
        raise


def fetch_games(year: int) -> list[dict]:
    """All games (completed + upcoming) for a given year, sorted by date."""
    data = squiggle_get(f"q=games;year={year}")
    games = data.get("games", [])
    return sorted(games, key=lambda g: g.get("date") or "9999")


def fetch_standings(year: int) -> list[dict]:
    """Current ladder standings from Squiggle."""
    data = squiggle_get(f"q=standings;year={year}")
    return data.get("standings", [])


def fetch_teams() -> list[dict]:
    """All AFL teams."""
    data = squiggle_get("q=teams")
    return data.get("teams", [])

# ─── TELO ENGINE ─────────────────────────────────────────────────────────────

def expected_win_prob(home_telo: float, away_telo: float,
                      neutral: bool = False,
                      hga: Optional[float] = None) -> float:
    """
    Expected probability that home team wins.
    Uses the standard ELO logistic formula.
    hga: override home ground advantage (defaults to HGA constant).
    """
    if neutral:
        effective_hga = 0.0
    elif hga is not None:
        effective_hga = hga
    else:
        effective_hga = HGA
    return 1.0 / (1.0 + 10.0 ** ((away_telo - home_telo - effective_hga) / 400.0))


def margin_k_multiplier(margin: float) -> float:
    """
    Scale K by log of actual margin so dominant wins count more,
    but not linearly (to avoid over-weighting blowouts).
    """
    return 1.0 + MARGIN_SCALE * math.log(1.0 + abs(margin))


def process_game(ratings: dict, home: str, away: str,
                 home_score: int, away_score: int,
                 neutral: bool = False,
                 k_scale: float = 1.0) -> tuple[float, float]:
    """
    Update TELO ratings after a completed game.
    k_scale: multiplier on K_FACTOR for recency bias (0–1).
    Returns (home_delta, away_delta).
    Note: uses flat HGA for rating updates (advanced factors are for display predictions only).
    """
    h = ratings.get(home, INITIAL_TELO)
    a = ratings.get(away, INITIAL_TELO)

    if home_score > away_score:
        actual = 1.0
    elif away_score > home_score:
        actual = 0.0
    else:
        actual = 0.5   # draw (rare in AFL)

    exp    = expected_win_prob(h, a, neutral)
    margin = abs(home_score - away_score)
    mult   = margin_k_multiplier(margin)

    delta         = K_FACTOR * k_scale * mult * (actual - exp)
    ratings[home] = h + delta
    ratings[away] = a - delta
    return delta, -delta


def regress_toward_mean(ratings: dict) -> dict:
    """
    At the start of each new season, regress ratings toward the league mean.
    Prevents dynasty teams from running away with ratings indefinitely.
    """
    if not ratings:
        return ratings
    mean = sum(ratings.values()) / len(ratings)
    return {
        team: telo * (1.0 - SEASON_REGRESS) + mean * SEASON_REGRESS
        for team, telo in ratings.items()
    }

# ─── ADVANCED FACTOR HELPERS ─────────────────────────────────────────────────

def venue_to_state(venue: str) -> Optional[str]:
    """Infer Australian state from venue name (substring matching)."""
    v = venue.upper()
    if any(x in v for x in ("MCG", "DOCKLANDS", "MARVEL", "GMHBA", "KARDINIA",
                              "VICTORIA PARK", "ARDEN ST", "JUNCTION OVAL")):
        return "VIC"
    if any(x in v for x in ("SCG", "ENGIE", "SHOWGROUND", "MANUKA", "CANBERRA")):
        return "NSW"
    if any(x in v for x in ("GABBA", "HERITAGE BANK", "PEOPLE FIRST", "CAZALY",
                              "METRICON", "RIVERWAY")):
        return "QLD"
    if any(x in v for x in ("ADELAIDE OVAL", "NORWOOD", "FOOTBALL PARK")):
        return "SA"
    if any(x in v for x in ("OPTUS", "PERTH STADIUM", "SUBIACO", "DOMAIN STADIUM",
                              "PATERSONS")):
        return "WA"
    if any(x in v for x in ("TIO", "DARWIN", "TRAEGER")):
        return "NT"
    if any(x in v for x in ("YORK PARK", "BLUNDSTONE", "UNIVERSITY OF TASMANIA",
                              "AURORA")):
        return "TAS"
    return None


def form_adjustment(form_history: list[int]) -> float:
    """
    Convert recent results into a TELO rating adjustment.
    form_history: list of 1s (wins) and 0s (losses), most recent last.
    Perfect 5-win streak → +FORM_MAX_ADJ; 5-loss streak → -FORM_MAX_ADJ.
    """
    if not form_history:
        return 0.0
    win_rate = sum(form_history) / len(form_history)
    # 0.5 = neutral, 1.0 = max positive, 0.0 = max negative
    return FORM_MAX_ADJ * (win_rate - 0.5) * 2.0


def h2h_adjustment(h2h_record: dict, home: str, away: str) -> float:
    """
    Head-to-head TELO adjustment (applied to the home team's effective rating).
    Returns a positive value if home has historically dominated, negative if away has.
    """
    key = (home, away)
    rec = h2h_record.get(key)
    if rec is None or rec["total"] < H2H_MIN_GAMES:
        return 0.0
    h2h_rate = rec["wins"] / rec["total"]
    return H2H_MAX_ADJ * (h2h_rate - 0.5) * 2.0


def travel_fatigue_penalty(team: str, venue: str,
                            last_date: dict[str, datetime],
                            game_date_str: str) -> float:
    """
    Compute a TELO penalty for a team based on:
      - Interstate travel: venue is in a different state from team home
      - Short turnaround: played within FATIGUE_DAYS days
    Returns a positive penalty value (subtract from effective rating).
    """
    penalty = 0.0

    # Interstate travel check
    team_home  = TEAM_STATE.get(team)
    game_state = venue_to_state(venue)
    if team_home and game_state and team_home != game_state:
        penalty += TRAVEL_PENALTY

    # Fatigue / short break check
    if team in last_date and game_date_str:
        try:
            last    = last_date[team]
            current = datetime.fromisoformat(game_date_str.replace(" ", "T"))
            days    = (current - last).days
            if 0 <= days <= FATIGUE_DAYS:
                penalty += FATIGUE_PENALTY
        except Exception:
            pass

    return penalty


def compute_venue_hga(venue_stats: dict[str, dict], venue: str) -> float:
    """
    Compute venue-specific HGA from historical home win rate.
    Blends computed value with default HGA based on sample size.
    """
    stats = venue_stats.get(venue)
    if stats is None or stats["total"] < VENUE_MIN_GAMES:
        return HGA

    home_rate = stats["wins"] / stats["total"]
    home_rate = max(0.05, min(0.95, home_rate))   # clamp to avoid log(0)/log(inf)

    # Invert the ELO logistic: p = 1/(1+10^(-hga/400)) → hga = 400 * log10(p/(1-p))
    computed_hga = 400.0 * math.log10(home_rate / (1.0 - home_rate))

    # Blend toward default based on sample size (fully trusted at 2× VENUE_MIN_GAMES)
    weight = min(1.0, stats["total"] / (VENUE_MIN_GAMES * 2.0))
    return HGA * (1.0 - weight) + computed_hga * weight

# ─── PREDICTIONS ─────────────────────────────────────────────────────────────

def predict_margin(home_telo: float, away_telo: float,
                   neutral: bool = False,
                   hga: Optional[float] = None) -> float:
    """Predicted margin in favour of home team (negative = away favoured)."""
    if neutral:
        effective_hga = 0.0
    elif hga is not None:
        effective_hga = hga
    else:
        effective_hga = HGA
    return (home_telo - away_telo + effective_hga) * TELO_TO_MARGIN


def win_probability_pct(home_telo: float, away_telo: float,
                         neutral: bool = False,
                         hga: Optional[float] = None) -> float:
    """Home win probability as a percentage (0–100)."""
    return expected_win_prob(home_telo, away_telo, neutral, hga=hga) * 100.0

# ─── MONTE CARLO SIMULATION ──────────────────────────────────────────────────

def simulate_finals(ratings: dict, remaining_games: list[dict],
                    wins: dict, losses: dict,
                    n: int = MC_SIMULATIONS) -> tuple[dict, dict]:
    """
    Simulate the rest of the season n times.
    Returns (finals_pct, premiers_pct) — percentage of simulations
    where each team made finals / won the premiership.
    Uses base TELO ratings (no per-game adjustments) for simulation stability.
    """
    teams = [t for t in ratings if ratings[t] > 0]
    finals_counts   = defaultdict(int)
    premiers_counts = defaultdict(int)

    for _ in range(n):
        sim_wins = dict(wins)

        for g in remaining_games:
            home, away = g.get("hteam", ""), g.get("ateam", "")
            if not home or not away:
                continue
            if home not in ratings or away not in ratings:
                continue
            p_home = expected_win_prob(ratings[home], ratings[away])
            if random.random() < p_home:
                sim_wins[home] = sim_wins.get(home, 0) + 1
            else:
                sim_wins[away] = sim_wins.get(away, 0) + 1

        ranked = sorted(
            teams,
            key=lambda t: (sim_wins.get(t, 0), ratings.get(t, INITIAL_TELO)),
            reverse=True
        )
        top8 = ranked[:FINALS_SPOTS]
        for t in top8:
            finals_counts[t] += 1

        def sim(t1: str, t2: str) -> str:
            p = expected_win_prob(
                ratings.get(t1, INITIAL_TELO),
                ratings.get(t2, INITIAL_TELO),
                neutral=True
            )
            return t1 if random.random() < p else t2

        if len(top8) < 8:
            premiers_counts[top8[0]] += 1
            continue

        qf1_w = sim(top8[0], top8[3])
        qf1_l = top8[3] if qf1_w == top8[0] else top8[0]
        qf2_w = sim(top8[1], top8[2])
        qf2_l = top8[2] if qf2_w == top8[1] else top8[1]

        ef1_w = sim(top8[4], top8[7])
        ef2_w = sim(top8[5], top8[6])

        sf1_w = sim(qf1_l, ef2_w)
        sf2_w = sim(qf2_l, ef1_w)

        pf1_w = sim(qf1_w, sf2_w)
        pf2_w = sim(qf2_w, sf1_w)

        premiers_counts[sim(pf1_w, pf2_w)] += 1

    return (
        {t: round(finals_counts[t]   / n * 100, 1) for t in teams},
        {t: round(premiers_counts[t] / n * 100, 1) for t in teams},
    )

# ─── DATE FORMATTING ─────────────────────────────────────────────────────────

def format_game_datetime(iso_str: str) -> str:
    """
    Format a Squiggle datetime string for display in Melbourne local time.
    Squiggle returns naive strings already in Melbourne time (AEST/AEDT).
    e.g. "2026-04-09 19:40:00" → "Thu 9 Apr · 7:40pm AEST"
    """
    if not iso_str:
        return ""
    try:
        dt_naive = datetime.fromisoformat(iso_str.replace(" ", "T"))
        local    = dt_naive.replace(tzinfo=AEST)
        tz_name  = local.strftime("%Z")
        day      = local.strftime("%a %-d %b")
        hour     = local.hour % 12 or 12
        minute   = local.strftime("%M")
        ampm     = "am" if local.hour < 12 else "pm"
        time_str = f"{hour}:{minute}{ampm}" if minute != "00" else f"{hour}{ampm}"
        return f"{day} · {time_str} {tz_name}"
    except Exception:
        return iso_str


def round_date_range(games: list[dict]) -> str:
    """Build a human-readable date range for a round."""
    dates = [g.get("date", "") for g in games if g.get("date")]
    if not dates:
        return ""
    try:
        parsed = [
            datetime.fromisoformat(d.replace(" ", "T")).replace(tzinfo=AEST)
            for d in dates
        ]
        parsed.sort()
        first = parsed[0]
        last  = parsed[-1]
        if first.month == last.month:
            return f"{first.strftime('%-d')}–{last.strftime('%-d %b %Y')}"
        return f"{first.strftime('%-d %b')}–{last.strftime('%-d %b %Y')}"
    except Exception:
        return ""

# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def build_predictions(year: int, dry_run: bool = False) -> dict:
    seasons = list(range(year - HISTORY_YEARS + 1, year + 1))  # e.g. [2024, 2025, 2026]
    weights = RECENCY_WEIGHTS

    ratings: dict[str, float] = {}
    all_teams: set[str] = set()

    # ── Advanced factor accumulators ──────────────────────────────────────────
    # venue_stats: historical home wins/totals per venue (all 3 seasons)
    venue_stats: dict[str, dict] = {}
    # h2h_record: home wins/totals for each (home_team, away_team) pair
    h2h_record: dict[tuple, dict] = {}
    # form_tracking: per-team rolling results (1=win, 0=loss), reset each season
    form_tracking: dict[str, list] = defaultdict(list)
    # last_game_date: most recent completed game date per team (reset each season)
    last_game_date: dict[str, datetime] = {}

    for season, k_scale in zip(seasons, weights):
        print(f"[TELO] Fetching {season} AFL data (K×{k_scale:.2f})...")
        try:
            season_games = fetch_games(season)
        except Exception:
            print(f"[TELO] ⚠ Could not fetch {season} data, skipping.", file=sys.stderr)
            continue

        if not season_games:
            print(f"[TELO] ⚠ No games for {season}, skipping.", file=sys.stderr)
            continue

        for g in season_games:
            if g.get("hteam"): all_teams.add(g["hteam"])
            if g.get("ateam"): all_teams.add(g["ateam"])

        for t in all_teams:
            if t not in ratings:
                ratings[t] = INITIAL_TELO

        if ratings and season > seasons[0]:
            ratings = regress_toward_mean(ratings)
            # Reset form and fatigue each new season — form is current-season momentum
            form_tracking = defaultdict(list)
            last_game_date = {}

        completed_season = [g for g in season_games if g.get("complete") == 100]
        print(f"[TELO]   {season}: {len(completed_season)} completed games")

        for g in completed_season:
            home = g.get("hteam", "")
            away = g.get("ateam", "")
            hs   = g.get("hscore")
            as_  = g.get("ascore")
            if not (home and away and hs is not None and as_ is not None):
                continue

            hs, as_ = int(hs), int(as_)
            home_won = hs > as_

            # ── Core TELO update ─────────────────────────────────────────────
            process_game(ratings, home, away, hs, as_, k_scale=k_scale)

            # ── Venue stats accumulation ──────────────────────────────────────
            venue = g.get("venue", "")
            if venue:
                if venue not in venue_stats:
                    venue_stats[venue] = {"wins": 0, "total": 0}
                venue_stats[venue]["total"] += 1
                if home_won:
                    venue_stats[venue]["wins"] += 1

            # ── H2H record accumulation ───────────────────────────────────────
            key = (home, away)
            if key not in h2h_record:
                h2h_record[key] = {"wins": 0, "total": 0}
            h2h_record[key]["total"] += 1
            if home_won:
                h2h_record[key]["wins"] += 1

            # ── Form tracking ─────────────────────────────────────────────────
            form_tracking[home].append(1 if home_won else 0)
            form_tracking[away].append(0 if home_won else 1)
            if len(form_tracking[home]) > FORM_WINDOW:
                form_tracking[home] = form_tracking[home][-FORM_WINDOW:]
            if len(form_tracking[away]) > FORM_WINDOW:
                form_tracking[away] = form_tracking[away][-FORM_WINDOW:]

            # ── Last game date ────────────────────────────────────────────────
            date_str = g.get("date", "")
            if date_str:
                try:
                    dt = datetime.fromisoformat(date_str.replace(" ", "T"))
                    last_game_date[home] = dt
                    last_game_date[away] = dt
                except Exception:
                    pass

    # Current year data
    print(f"[TELO] Fetching {year} standings and upcoming fixtures...")
    games     = fetch_games(year)
    standings = fetch_standings(year)

    if not games:
        print(f"ERROR: No games returned for {year}. Squiggle may not have data yet.", file=sys.stderr)
        sys.exit(1)

    wins_map    = {s["name"]: int(s.get("wins",       0))   for s in standings}
    losses_map  = {s["name"]: int(s.get("losses",     0))   for s in standings}
    rank_map    = {s["name"]: int(s.get("rank",        0))   for s in standings}
    pct_map     = {s["name"]: round(float(s.get("percentage", 0.0)), 1) for s in standings}
    for_map     = {s["name"]: int(s.get("for",         0))   for s in standings}
    against_map = {s["name"]: int(s.get("against",     0))   for s in standings}

    completed = [g for g in games if g.get("complete") == 100]
    upcoming  = [g for g in games if g.get("complete") != 100]

    print(f"[TELO] Current year totals: {len(completed)} completed, {len(upcoming)} upcoming")

    upcoming_rounds  = [g.get("round") for g in upcoming  if g.get("round") is not None]
    completed_rounds = [g.get("round") for g in completed if g.get("round") is not None]
    if upcoming_rounds:
        current_round = min(upcoming_rounds)
    elif completed_rounds:
        current_round = max(completed_rounds)
    else:
        current_round = 0

    print(f"[TELO] Current round: {current_round}")

    # ── Build rounds dict ──────────────────────────────────────────────────
    by_round: dict[int, list] = defaultdict(list)
    for g in games:
        rnum = g.get("round")
        if rnum is not None:
            by_round[rnum].append(g)

    rounds_output: dict[str, dict] = {}

    for rnum in sorted(by_round.keys()):
        rnd_games  = sorted(by_round[rnum], key=lambda g: g.get("date") or "")
        roundname  = rnd_games[0].get("roundname") or f"Round {rnum}"
        date_range = round_date_range(rnd_games)

        games_out: list[dict] = []
        for g in rnd_games:
            home = g.get("hteam", "")
            away = g.get("ateam", "")
            if not home or not away:
                continue

            h_telo = ratings.get(home, INITIAL_TELO)
            a_telo = ratings.get(away, INITIAL_TELO)
            venue  = g.get("venue", "")

            # ── Advanced factor adjustments ───────────────────────────────────
            v_hga       = compute_venue_hga(venue_stats, venue)

            home_form   = form_adjustment(form_tracking.get(home, []))
            away_form   = form_adjustment(form_tracking.get(away, []))

            h2h_adj     = h2h_adjustment(h2h_record, home, away)

            date_str    = g.get("date", "")
            home_travel = travel_fatigue_penalty(home, venue, last_game_date, date_str)
            away_travel = travel_fatigue_penalty(away, venue, last_game_date, date_str)

            home_injury = INJURY_OVERRIDES.get(home, 0.0)
            away_injury = INJURY_OVERRIDES.get(away, 0.0)

            # Effective ratings for this prediction (not stored)
            h_eff = h_telo + home_form + h2h_adj + home_injury - home_travel
            a_eff = a_telo + away_form             + away_injury - away_travel

            h_prob      = round(win_probability_pct(h_eff, a_eff, hga=v_hga), 1)
            pred_margin = round(abs(predict_margin(h_eff, a_eff, hga=v_hga)), 1)
            home_fav    = h_prob >= 50.0

            is_complete = g.get("complete") == 100

            entry: dict = {
                "home":      home,
                "away":      away,
                "venue":     venue,
                "datetime":  format_game_datetime(date_str),
                "home_fav":  home_fav,
                "margin":    pred_margin,
                "home_prob": h_prob,
                "complete":  is_complete,
            }

            if is_complete:
                hs  = int(g.get("hscore") or 0)
                as_ = int(g.get("ascore") or 0)
                actual_home_win = hs > as_
                upset = home_fav != actual_home_win
                entry.update({
                    "home_score": hs,
                    "away_score": as_,
                    "upset":      upset,
                })

            games_out.append(entry)

        round_complete = all(g.get("complete") == 100 for g in rnd_games)
        rounds_output[str(rnum)] = {
            "label":    roundname,
            "range":    date_range,
            "complete": round_complete,
            "games":    games_out,
        }

    # ── Monte Carlo ────────────────────────────────────────────────────────
    print(f"[TELO] Running {MC_SIMULATIONS:,} Monte Carlo simulations...")
    finals_pct, premiers_pct = simulate_finals(
        ratings, upcoming, wins_map, losses_map
    )

    # ── Rankings ───────────────────────────────────────────────────────────
    ranked = sorted(all_teams, key=lambda t: ratings.get(t, INITIAL_TELO), reverse=True)
    rankings_out: list[dict] = []
    for team in ranked:
        telo = ratings.get(team, INITIAL_TELO)
        rankings_out.append({
            "team":           team,
            "telo":           round(telo),
            "wins":           wins_map.get(team, 0),
            "losses":         losses_map.get(team, 0),
            "ladder_rank":    rank_map.get(team, 0),
            "percentage":     pct_map.get(team, 0.0),
            "points_for":     for_map.get(team, 0),
            "points_against": against_map.get(team, 0),
            "finals_pct":     finals_pct.get(team, 0.0),
            "premiers_pct":   premiers_pct.get(team, 0.0),
        })

    # ── Assemble output ────────────────────────────────────────────────────
    output = {
        "meta": {
            "updated":       datetime.now(AEST).isoformat(),
            "year":          year,
            "current_round": current_round,
            "model":         "TELO v2.0",
            "simulations":   MC_SIMULATIONS,
        },
        "rounds":   rounds_output,
        "rankings": rankings_out,
    }

    if dry_run:
        print(json.dumps(output, indent=2))
    else:
        os.makedirs("data", exist_ok=True)
        out_path = "data/predictions.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[TELO] ✓ Written {out_path}")
        print(f"[TELO]   Teams: {len(rankings_out)} | Rounds: {len(rounds_output)} | Current: Round {current_round}")

    return output


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tipper TELO model pipeline")
    parser.add_argument("--year",    type=int, default=datetime.now().year, help="AFL season year")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON, don't write files")
    args = parser.parse_args()

    build_predictions(year=args.year, dry_run=args.dry_run)
