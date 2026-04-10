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
from zoneinfo import ZoneInfo

import requests

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

SQUIGGLE_BASE = "https://api.squiggle.com.au/"
SQUIGGLE_UA   = "Tipper-TELO/1.0 (github.com/wrloading/Tipper-test-website)"

# TELO model parameters
INITIAL_TELO   = 1500.0   # Starting rating for all teams
K_FACTOR       = 40.0     # Base K — how fast ratings move (higher = more volatile)
HGA            = 65.0     # Home ground advantage in TELO points (~6pt margin equiv)
MARGIN_SCALE   = 0.025    # Log-margin multiplier weight (rewards blowouts slightly)
SEASON_REGRESS = 0.25     # Fraction regressed toward mean at season start

# Recency bias: K-factor scaling per season (oldest → newest)
# Seasons processed: [current-2, current-1, current]
RECENCY_WEIGHTS = [0.50, 0.75, 1.00]
HISTORY_YEARS   = 3       # How many seasons to look back (including current)

# Margin prediction calibration
# Empirically: each 100 TELO pts ≈ ~25 pts AFL margin
TELO_TO_MARGIN = 0.25

# Monte Carlo
MC_SIMULATIONS = 12000
FINALS_SPOTS   = 8
AEST           = ZoneInfo("Australia/Melbourne")

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
                      neutral: bool = False) -> float:
    """
    Expected probability that home team wins.
    Uses the standard ELO logistic formula with optional HGA.
    """
    hga = 0.0 if neutral else HGA
    return 1.0 / (1.0 + 10.0 ** ((away_telo - home_telo - hga) / 400.0))


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

# ─── PREDICTIONS ─────────────────────────────────────────────────────────────

def predict_margin(home_telo: float, away_telo: float,
                   neutral: bool = False) -> float:
    """Predicted margin in favour of home team (negative = away favoured)."""
    hga = 0.0 if neutral else HGA
    return (home_telo - away_telo + hga) * TELO_TO_MARGIN


def win_probability_pct(home_telo: float, away_telo: float,
                         neutral: bool = False) -> float:
    """Home win probability as a percentage (0–100)."""
    return expected_win_prob(home_telo, away_telo, neutral) * 100.0

# ─── MONTE CARLO SIMULATION ──────────────────────────────────────────────────

def simulate_finals(ratings: dict, remaining_games: list[dict],
                    wins: dict, losses: dict,
                    n: int = MC_SIMULATIONS) -> tuple[dict, dict]:
    """
    Simulate the rest of the season n times.
    Returns (finals_pct, premiers_pct) — percentage of simulations
    where each team made finals / won the premiership.
    """
    teams = [t for t in ratings if ratings[t] > 0]
    finals_counts   = defaultdict(int)
    premiers_counts = defaultdict(int)

    for _ in range(n):
        sim_wins = dict(wins)

        # Simulate remaining regular-season games
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

        # Rank by wins, tiebreak by TELO rating
        ranked = sorted(
            teams,
            key=lambda t: (sim_wins.get(t, 0), ratings.get(t, INITIAL_TELO)),
            reverse=True
        )
        top8 = ranked[:FINALS_SPOTS]
        for t in top8:
            finals_counts[t] += 1

        # Simulate AFL finals (McIntyre final 8 simplified):
        # Week 1: Qualifying finals (1v4, 2v3) + Elimination finals (5v8, 6v7)
        # Week 2: Semi-finals (QF losers vs EF winners)
        # Week 3: Preliminary finals
        # Week 4: Grand Final
        def sim(t1: str, t2: str) -> str:
            p = expected_win_prob(
                ratings.get(t1, INITIAL_TELO),
                ratings.get(t2, INITIAL_TELO),
                neutral=True
            )
            return t1 if random.random() < p else t2

        if len(top8) < 8:
            # Not enough teams — just pick from top8 by TELO
            premiers_counts[top8[0]] += 1
            continue

        # Qualifying finals — winners go straight to prelim
        qf1_w = sim(top8[0], top8[3])
        qf1_l = top8[3] if qf1_w == top8[0] else top8[0]
        qf2_w = sim(top8[1], top8[2])
        qf2_l = top8[2] if qf2_w == top8[1] else top8[1]

        # Elimination finals
        ef1_w = sim(top8[4], top8[7])
        ef2_w = sim(top8[5], top8[6])

        # Semi-finals (QF losers vs EF winners)
        sf1_w = sim(qf1_l, ef2_w)
        sf2_w = sim(qf2_l, ef1_w)

        # Preliminary finals
        pf1_w = sim(qf1_w, sf2_w)
        pf2_w = sim(qf2_w, sf1_w)

        # Grand Final
        premiers_counts[sim(pf1_w, pf2_w)] += 1

    return (
        {t: round(finals_counts[t]   / n * 100, 1) for t in teams},
        {t: round(premiers_counts[t] / n * 100, 1) for t in teams},
    )

# ─── DATE FORMATTING ─────────────────────────────────────────────────────────

def format_game_datetime(iso_str: str) -> str:
    """
    Convert Squiggle UTC datetime string to AEST/AEDT display string.
    e.g. "2026-04-10 09:30:00" → "Thu 10 Apr · 7:30pm AEDT"
    """
    if not iso_str:
        return ""
    try:
        # Squiggle gives UTC times as naive strings
        dt_utc = datetime.fromisoformat(iso_str.replace(" ", "T"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        local = dt_utc.astimezone(AEST)
        day   = local.strftime("%a %-d %b")
        # Determine timezone abbreviation
        tz_offset = local.utcoffset()
        tz_name   = "AEDT" if tz_offset == timedelta(hours=11) else "AEST"
        # Format time without leading zero
        hour   = local.hour % 12 or 12
        minute = local.strftime("%M")
        ampm   = "am" if local.hour < 12 else "pm"
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
            datetime.fromisoformat(d.replace(" ", "T")).replace(tzinfo=timezone.utc).astimezone(AEST)
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
    # ── Build multi-season ratings with recency bias ───────────────────────
    seasons = list(range(year - HISTORY_YEARS + 1, year + 1))  # e.g. [2024, 2025, 2026]
    weights = RECENCY_WEIGHTS  # [0.50, 0.75, 1.00]

    ratings: dict[str, float] = {}
    all_teams: set[str] = set()

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

        # Collect team names
        for g in season_games:
            if g.get("hteam"): all_teams.add(g["hteam"])
            if g.get("ateam"): all_teams.add(g["ateam"])

        # Initialise any new teams at INITIAL_TELO
        for t in all_teams:
            if t not in ratings:
                ratings[t] = INITIAL_TELO

        # Apply season-start regression (skip for the very first season processed)
        if ratings and season > seasons[0]:
            ratings = regress_toward_mean(ratings)

        # Process completed games for this season
        completed_season = [g for g in season_games if g.get("complete") == 100]
        print(f"[TELO]   {season}: {len(completed_season)} completed games")
        for g in completed_season:
            home = g.get("hteam", "")
            away = g.get("ateam", "")
            hs   = g.get("hscore")
            as_  = g.get("ascore")
            if not (home and away and hs is not None and as_ is not None):
                continue
            process_game(ratings, home, away, int(hs), int(as_), k_scale=k_scale)

    # Current year data is the last season
    print(f"[TELO] Fetching {year} standings and upcoming fixtures...")
    games     = fetch_games(year)
    standings = fetch_standings(year)

    if not games:
        print(f"ERROR: No games returned for {year}. Squiggle may not have data yet.", file=sys.stderr)
        sys.exit(1)

    # Build wins/losses from ladder standings
    wins_map   = {s["name"]: int(s.get("wins",   0)) for s in standings}
    losses_map = {s["name"]: int(s.get("losses", 0)) for s in standings}

    # Separate completed vs upcoming for current year
    completed = [g for g in games if g.get("complete") == 100]
    upcoming  = [g for g in games if g.get("complete") != 100]

    print(f"[TELO] Current year totals: {len(completed)} completed, {len(upcoming)} upcoming")

    # Determine current round (earliest round with any upcoming game)
    upcoming_rounds  = [g.get("round", 0) for g in upcoming  if g.get("round")]
    completed_rounds = [g.get("round", 0) for g in completed if g.get("round")]
    if upcoming_rounds:
        current_round = min(upcoming_rounds)
    elif completed_rounds:
        current_round = max(completed_rounds)
    else:
        current_round = 1

    print(f"[TELO] Current round: {current_round}")

    # ── Build rounds dict ──────────────────────────────────────────────────
    by_round: dict[int, list] = defaultdict(list)
    for g in games:
        rnum = g.get("round", 0)
        if rnum:
            by_round[rnum].append(g)

    rounds_output: dict[str, dict] = {}

    for rnum in sorted(by_round.keys()):
        rnd_games   = sorted(by_round[rnum], key=lambda g: g.get("date") or "")
        is_complete = all(g.get("complete") == 100 for g in rnd_games)
        roundname   = rnd_games[0].get("roundname") or f"Round {rnum}"
        date_range  = round_date_range(rnd_games)

        games_out: list[dict] = []
        for g in rnd_games:
            home = g.get("hteam", "")
            away = g.get("ateam", "")
            if not home or not away:
                continue

            h_telo = ratings.get(home, INITIAL_TELO)
            a_telo = ratings.get(away, INITIAL_TELO)

            h_prob      = round(win_probability_pct(h_telo, a_telo), 1)
            pred_margin = round(abs(predict_margin(h_telo, a_telo)), 1)
            home_fav    = h_prob >= 50.0

            entry: dict = {
                "home":      home,
                "away":      away,
                "venue":     g.get("venue", ""),
                "datetime":  format_game_datetime(g.get("date", "")),
                "home_fav":  home_fav,
                "margin":    pred_margin,
                "home_prob": h_prob,
            }

            if g.get("complete") == 100:
                hs  = int(g.get("hscore") or 0)
                as_ = int(g.get("ascore") or 0)
                actual_home_win = hs > as_
                upset = home_fav != actual_home_win  # fav lost
                entry.update({
                    "home_score": hs,
                    "away_score": as_,
                    "upset":      upset,
                })

            games_out.append(entry)

        rounds_output[str(rnum)] = {
            "label":    roundname,
            "range":    date_range,
            "complete": is_complete,
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
            "team":         team,
            "telo":         round(telo),
            "wins":         wins_map.get(team, 0),
            "losses":       losses_map.get(team, 0),
            "finals_pct":   finals_pct.get(team, 0.0),
            "premiers_pct": premiers_pct.get(team, 0.0),
        })

    # ── Assemble output ────────────────────────────────────────────────────
    output = {
        "meta": {
            "updated":       datetime.now(AEST).isoformat(),
            "year":          year,
            "current_round": current_round,
            "model":         "TELO v1.0",
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
