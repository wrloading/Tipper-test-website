#!/usr/bin/env python3
"""
TELO Model — AFL ELO-based prediction engine for Tipper.

Fetches live results from the Squiggle public API, computes TELO ratings
for all AFL teams, generates game predictions and win probabilities for
upcoming fixtures, runs a Monte Carlo finals simulation, and writes
data/predictions.json and data/player_ratings.json for the website.

Ratings are built from the last 3 seasons with recency bias:
  2 seasons ago : K × 0.50
  1 season ago  : K × 0.75
  Current season: K × 1.00

Advanced prediction factors (applied to per-game display predictions):
  - Form streaks       : rolling last-5 results per team
  - Venue-specific HGA : per-venue home advantage from 3-year history
  - Head-to-head       : historical matchup record between specific pairs
  - Travel / fatigue   : interstate travel and short-break penalties
  - Injury auto-adjust : computed from AFL Fantasy player availability
  - Injury overrides   : INJURY_OVERRIDES dict for manual adjustments

Usage:
    python telo.py              # Current year
    python telo.py --year 2025  # Specific year
    python telo.py --dry-run    # Print output, don't write files
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import requests

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

SQUIGGLE_BASE = "https://api.squiggle.com.au/"
SQUIGGLE_UA   = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"

# Core TELO model parameters
INITIAL_TELO   = 1500.0
K_FACTOR       = 40.0
HGA            = 65.0
MARGIN_SCALE   = 0.025
SEASON_REGRESS = 0.25

RECENCY_WEIGHTS = [0.50, 0.75, 1.00]
HISTORY_YEARS   = 3
TELO_TO_MARGIN  = 0.25
MC_SIMULATIONS  = 12000
FINALS_SPOTS    = 8
AEST            = ZoneInfo("Australia/Melbourne")

# ─── ADVANCED FACTORS ────────────────────────────────────────────────────────

FORM_WINDOW     = 5
FORM_MAX_ADJ    = 12.0
VENUE_MIN_GAMES = 20
H2H_MIN_GAMES   = 8
H2H_MAX_ADJ     = 12.0
TRAVEL_PENALTY  = 10.0
FATIGUE_DAYS    = 6
FATIGUE_PENALTY = 6.0

# Manual injury/availability overrides (take priority over auto-computed values)
# Format: {"Team Name": delta_telo}  — negative = key players out
INJURY_OVERRIDES: dict[str, float] = {
    # "Melbourne": -15.0,
}

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

# ─── AFL FANTASY PLAYER DATA ─────────────────────────────────────────────────

AFL_FANTASY_PLAYERS_URL = "https://fantasy.afl.com.au/data/afl/players.json"
AFL_FANTASY_UA          = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"

PLAYER_INITIAL_TELO = 1500.0
PLAYER_SCORE_SCALE  = 8.0     # TELO pts per avg Fantasy pt above/below league avg
PLAYER_MIN_GAMES    = 2       # games before a rating is used in team impact calc
SQUAD_IMPACT_SCALE  = 0.30    # player-avg delta → team TELO impact multiplier

# AFL Fantasy position IDs (verified: 1=DEF, 2=MID, 3=RUC, 4=FWD)
FANTASY_POSITIONS: dict[int, str] = {1: "DEF", 2: "MID", 3: "RUC", 4: "FWD"}

# AFL Fantasy squad_id → Squiggle team name
FANTASY_SQUAD_MAP: dict[int, str] = {
    10:   "Adelaide",
    20:   "Brisbane",
    30:   "Carlton",
    40:   "Collingwood",
    50:   "Essendon",
    60:   "Fremantle",
    70:   "Geelong",
    1000: "Gold Coast",
    1010: "GWS",
    80:   "Hawthorn",
    90:   "Melbourne",
    100:  "North Melbourne",
    110:  "Port Adelaide",
    120:  "Richmond",
    130:  "St Kilda",
    160:  "Sydney",
    150:  "West Coast",
    140:  "Western Bulldogs",
}

# ─── SQUIGGLE API ─────────────────────────────────────────────────────────────

def squiggle_get(query: str) -> dict:
    url = f"{SQUIGGLE_BASE}?{query}"
    try:
        r = requests.get(url, headers={"User-Agent": SQUIGGLE_UA}, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"ERROR fetching {url}: {e}", file=sys.stderr)
        raise


def fetch_games(year: int) -> list:
    data = squiggle_get(f"q=games;year={year}")
    games = data.get("games", [])
    return sorted(games, key=lambda g: g.get("date") or "9999")


def fetch_standings(year: int) -> list:
    data = squiggle_get(f"q=standings;year={year}")
    return data.get("standings", [])

# ─── AFL FANTASY API ─────────────────────────────────────────────────────────

def fetch_fantasy_players() -> list:
    """Fetch player data from AFL Fantasy API. Returns empty list on failure."""
    try:
        r = requests.get(
            AFL_FANTASY_PLAYERS_URL,
            headers={"User-Agent": AFL_FANTASY_UA, "Accept-Encoding": "gzip, deflate"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("players", [])
    except Exception as e:
        print(f"[TELO] ⚠ Fantasy player fetch failed: {e}", file=sys.stderr)
        return []


def compute_player_ratings(players_raw: list) -> list:
    """
    Convert AFL Fantasy player data to player TELO ratings.
    Returns list sorted by telo descending.
    """
    processed = []
    for p in players_raw:
        stats = p.get("stats") or {}
        avg   = float(stats.get("avg_points") or 0)
        games = int(stats.get("games_played") or 0)
        if avg <= 0:
            continue

        team = FANTASY_SQUAD_MAP.get(p.get("squad_id", 0))
        if not team:
            continue

        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        if not name:
            continue

        positions = p.get("positions") or [2]
        pos = FANTASY_POSITIONS.get(positions[0] if positions else 2, "MID")

        raw_status = p.get("status", "playing")
        if raw_status == "playing":
            status = "available"
        elif raw_status in ("injured", "medical_sub"):
            status = "injured"
        else:
            status = "unavailable"

        last3 = float(stats.get("last_3_avg") or avg)
        processed.append({
            "id":     int(p.get("id", 0)),
            "name":   name,
            "team":   team,
            "pos":    pos,
            "avg":    round(avg, 1),
            "last3":  round(last3, 1),
            "games":  games,
            "cost":   int(p.get("cost", 0)),
            "status": status,
        })

    if not processed:
        return []

    trusted = [p for p in processed if p["games"] >= PLAYER_MIN_GAMES]
    if not trusted:
        trusted = processed
    league_avg = sum(p["avg"] for p in trusted) / len(trusted)

    for p in processed:
        p["telo"] = round(PLAYER_INITIAL_TELO + (p["avg"] - league_avg) * PLAYER_SCORE_SCALE)

    return sorted(processed, key=lambda p: p["telo"], reverse=True)

# ─── TELO ENGINE ─────────────────────────────────────────────────────────────

def expected_win_prob(home_telo: float, away_telo: float,
                      neutral: bool = False,
                      hga: Optional[float] = None) -> float:
    if neutral:
        effective_hga = 0.0
    elif hga is not None:
        effective_hga = hga
    else:
        effective_hga = HGA
    return 1.0 / (1.0 + 10.0 ** ((away_telo - home_telo - effective_hga) / 400.0))


def margin_k_multiplier(margin: float) -> float:
    return 1.0 + MARGIN_SCALE * math.log(1.0 + abs(margin))


def process_game(ratings: dict, home: str, away: str,
                 home_score: int, away_score: int,
                 neutral: bool = False, k_scale: float = 1.0) -> tuple:
    h = ratings.get(home, INITIAL_TELO)
    a = ratings.get(away, INITIAL_TELO)
    actual = 1.0 if home_score > away_score else (0.0 if away_score > home_score else 0.5)
    exp    = expected_win_prob(h, a, neutral)
    delta  = K_FACTOR * k_scale * margin_k_multiplier(abs(home_score - away_score)) * (actual - exp)
    ratings[home] = h + delta
    ratings[away] = a - delta
    return delta, -delta


def regress_toward_mean(ratings: dict) -> dict:
    if not ratings:
        return ratings
    mean = sum(ratings.values()) / len(ratings)
    return {t: v * (1.0 - SEASON_REGRESS) + mean * SEASON_REGRESS for t, v in ratings.items()}

# ─── ADVANCED FACTOR HELPERS ─────────────────────────────────────────────────

def venue_to_state(venue: str) -> Optional[str]:
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
    if any(x in v for x in ("OPTUS", "PERTH STADIUM", "SUBIACO", "DOMAIN STADIUM")):
        return "WA"
    if any(x in v for x in ("TIO", "DARWIN", "TRAEGER")):
        return "NT"
    if any(x in v for x in ("YORK PARK", "BLUNDSTONE", "AURORA")):
        return "TAS"
    return None


def form_adjustment(form_history: list) -> float:
    if not form_history:
        return 0.0
    return FORM_MAX_ADJ * (sum(form_history) / len(form_history) - 0.5) * 2.0


def h2h_adjustment(h2h_record: dict, home: str, away: str) -> float:
    rec = h2h_record.get((home, away))
    if rec is None or rec["total"] < H2H_MIN_GAMES:
        return 0.0
    return H2H_MAX_ADJ * (rec["wins"] / rec["total"] - 0.5) * 2.0


def travel_fatigue_penalty(team: str, venue: str, last_date: dict, date_str: str) -> float:
    penalty = 0.0
    team_home  = TEAM_STATE.get(team)
    game_state = venue_to_state(venue)
    if team_home and game_state and team_home != game_state:
        penalty += TRAVEL_PENALTY
    if team in last_date and date_str:
        try:
            days = (datetime.fromisoformat(date_str.replace(" ", "T")) - last_date[team]).days
            if 0 <= days <= FATIGUE_DAYS:
                penalty += FATIGUE_PENALTY
        except Exception:
            pass
    return penalty


def compute_venue_hga(venue_stats: dict, venue: str) -> float:
    stats = venue_stats.get(venue)
    if stats is None or stats["total"] < VENUE_MIN_GAMES:
        return HGA
    home_rate    = max(0.05, min(0.95, stats["wins"] / stats["total"]))
    computed_hga = 400.0 * math.log10(home_rate / (1.0 - home_rate))
    weight       = min(1.0, stats["total"] / (VENUE_MIN_GAMES * 2.0))
    return HGA * (1.0 - weight) + computed_hga * weight

# ─── PREDICTIONS ─────────────────────────────────────────────────────────────

def predict_margin(home_telo: float, away_telo: float,
                   neutral: bool = False, hga: Optional[float] = None) -> float:
    effective_hga = 0.0 if neutral else (hga if hga is not None else HGA)
    return (home_telo - away_telo + effective_hga) * TELO_TO_MARGIN


def win_probability_pct(home_telo: float, away_telo: float,
                         neutral: bool = False, hga: Optional[float] = None) -> float:
    return expected_win_prob(home_telo, away_telo, neutral, hga=hga) * 100.0

# ─── MONTE CARLO SIMULATION ──────────────────────────────────────────────────

def simulate_finals(ratings: dict, remaining_games: list,
                    wins: dict, losses: dict, n: int = MC_SIMULATIONS) -> tuple:
    teams = [t for t in ratings if ratings[t] > 0]
    finals_counts   = defaultdict(int)
    premiers_counts = defaultdict(int)

    for _ in range(n):
        sim_wins = dict(wins)
        for g in remaining_games:
            home, away = g.get("hteam", ""), g.get("ateam", "")
            if not home or not away or home not in ratings or away not in ratings:
                continue
            if random.random() < expected_win_prob(ratings[home], ratings[away]):
                sim_wins[home] = sim_wins.get(home, 0) + 1
            else:
                sim_wins[away] = sim_wins.get(away, 0) + 1

        ranked = sorted(teams,
                         key=lambda t: (sim_wins.get(t, 0), ratings.get(t, INITIAL_TELO)),
                         reverse=True)
        top8 = ranked[:FINALS_SPOTS]
        for t in top8:
            finals_counts[t] += 1

        def sim(t1: str, t2: str) -> str:
            return t1 if random.random() < expected_win_prob(
                ratings.get(t1, INITIAL_TELO), ratings.get(t2, INITIAL_TELO), neutral=True
            ) else t2

        if len(top8) < 8:
            premiers_counts[top8[0]] += 1
            continue

        qf1_w = sim(top8[0], top8[3]); qf1_l = top8[3] if qf1_w == top8[0] else top8[0]
        qf2_w = sim(top8[1], top8[2]); qf2_l = top8[2] if qf2_w == top8[1] else top8[1]
        ef1_w = sim(top8[4], top8[7])
        ef2_w = sim(top8[5], top8[6])
        pf1_w = sim(qf1_w, sim(qf1_l, ef2_w))
        pf2_w = sim(qf2_w, sim(qf2_l, ef1_w))
        premiers_counts[sim(pf1_w, pf2_w)] += 1

    return (
        {t: round(finals_counts[t]   / n * 100, 1) for t in teams},
        {t: round(premiers_counts[t] / n * 100, 1) for t in teams},
    )

# ─── DATE FORMATTING ─────────────────────────────────────────────────────────

def format_game_datetime(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        local   = datetime.fromisoformat(iso_str.replace(" ", "T")).replace(tzinfo=AEST)
        tz_name = local.strftime("%Z")
        day     = local.strftime("%a %-d %b")
        hour    = local.hour % 12 or 12
        minute  = local.strftime("%M")
        ampm    = "am" if local.hour < 12 else "pm"
        time_str = f"{hour}:{minute}{ampm}" if minute != "00" else f"{hour}{ampm}"
        return f"{day} · {time_str} {tz_name}"
    except Exception:
        return iso_str


def round_date_range(games: list) -> str:
    dates = [g.get("date", "") for g in games if g.get("date")]
    if not dates:
        return ""
    try:
        parsed = sorted(
            datetime.fromisoformat(d.replace(" ", "T")).replace(tzinfo=AEST)
            for d in dates
        )
        first, last = parsed[0], parsed[-1]
        if first.month == last.month:
            return f"{first.strftime('%-d')}–{last.strftime('%-d %b %Y')}"
        return f"{first.strftime('%-d %b')}–{last.strftime('%-d %b %Y')}"
    except Exception:
        return ""

# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def build_predictions(year: int, dry_run: bool = False) -> dict:
    seasons = list(range(year - HISTORY_YEARS + 1, year + 1))
    weights = RECENCY_WEIGHTS

    ratings: dict[str, float] = {}
    all_teams: set = set()

    venue_stats:    dict = {}
    h2h_record:     dict = {}
    form_tracking:  dict = defaultdict(list)
    last_game_date: dict = {}

    for season, k_scale in zip(seasons, weights):
        print(f"[TELO] Fetching {season} AFL data (K×{k_scale:.2f})...")
        try:
            season_games = fetch_games(season)
        except Exception:
            print(f"[TELO] ⚠ Could not fetch {season}, skipping.", file=sys.stderr)
            continue
        if not season_games:
            continue

        for g in season_games:
            if g.get("hteam"): all_teams.add(g["hteam"])
            if g.get("ateam"): all_teams.add(g["ateam"])

        for t in all_teams:
            if t not in ratings:
                ratings[t] = INITIAL_TELO

        if ratings and season > seasons[0]:
            ratings        = regress_toward_mean(ratings)
            form_tracking  = defaultdict(list)
            last_game_date = {}

        completed_season = [g for g in season_games if g.get("complete") == 100]
        print(f"[TELO]   {season}: {len(completed_season)} completed games")

        for g in completed_season:
            home, away = g.get("hteam", ""), g.get("ateam", "")
            hs,   as_  = g.get("hscore"),    g.get("ascore")
            if not (home and away and hs is not None and as_ is not None):
                continue
            hs, as_ = int(hs), int(as_)
            home_won = hs > as_

            process_game(ratings, home, away, hs, as_, k_scale=k_scale)

            venue = g.get("venue", "")
            if venue:
                vs = venue_stats.setdefault(venue, {"wins": 0, "total": 0})
                vs["total"] += 1
                if home_won:
                    vs["wins"] += 1

            rec = h2h_record.setdefault((home, away), {"wins": 0, "total": 0})
            rec["total"] += 1
            if home_won:
                rec["wins"] += 1

            form_tracking[home].append(1 if home_won else 0)
            form_tracking[away].append(0 if home_won else 1)
            form_tracking[home] = form_tracking[home][-FORM_WINDOW:]
            form_tracking[away] = form_tracking[away][-FORM_WINDOW:]

            if g.get("date"):
                try:
                    dt = datetime.fromisoformat(g["date"].replace(" ", "T"))
                    last_game_date[home] = dt
                    last_game_date[away] = dt
                except Exception:
                    pass

    # ── Player data ────────────────────────────────────────────────────────────
    print("[TELO] Fetching AFL Fantasy player data...")
    player_ratings = compute_player_ratings(fetch_fantasy_players())
    print(f"[TELO]   {len(player_ratings)} players rated")

    team_squads: dict = defaultdict(list)
    for p in player_ratings:
        team_squads[p["team"]].append(p)

    full_squad_telo_avg: dict = {}
    for team, squad in team_squads.items():
        trusted = [p for p in squad if p["games"] >= PLAYER_MIN_GAMES]
        if trusted:
            full_squad_telo_avg[team] = sum(p["telo"] for p in trusted) / len(trusted)

    auto_injury_deltas: dict = {}
    for team, squad in team_squads.items():
        avg = full_squad_telo_avg.get(team, PLAYER_INITIAL_TELO)
        available = [p for p in squad
                     if p["status"] == "available" and p["games"] >= PLAYER_MIN_GAMES]
        if available:
            avail_avg = sum(p["telo"] for p in available) / len(available)
            delta = (avail_avg - avg) * SQUAD_IMPACT_SCALE
            if abs(delta) > 0.5:
                auto_injury_deltas[team] = round(delta, 1)

    # ── Current year ───────────────────────────────────────────────────────────
    print(f"[TELO] Fetching {year} standings and upcoming fixtures...")
    games     = fetch_games(year)
    standings = fetch_standings(year)

    if not games:
        print(f"ERROR: No games for {year}.", file=sys.stderr)
        sys.exit(1)

    wins_map    = {s["name"]: int(s.get("wins",       0))   for s in standings}
    losses_map  = {s["name"]: int(s.get("losses",     0))   for s in standings}
    rank_map    = {s["name"]: int(s.get("rank",        0))   for s in standings}
    pct_map     = {s["name"]: round(float(s.get("percentage", 0.0)), 1) for s in standings}
    for_map     = {s["name"]: int(s.get("for",         0))   for s in standings}
    against_map = {s["name"]: int(s.get("against",     0))   for s in standings}

    completed = [g for g in games if g.get("complete") == 100]
    upcoming  = [g for g in games if g.get("complete") != 100]
    print(f"[TELO] {len(completed)} completed, {len(upcoming)} upcoming")

    upcoming_rounds  = [g.get("round") for g in upcoming  if g.get("round") is not None]
    completed_rounds = [g.get("round") for g in completed if g.get("round") is not None]
    if upcoming_rounds:
        current_round = min(upcoming_rounds)
    elif completed_rounds:
        current_round = max(completed_rounds)
    else:
        current_round = 0

    print(f"[TELO] Current round: {current_round}")

    # ── Build rounds ────────────────────────────────────────────────────────
    by_round: dict = defaultdict(list)
    for g in games:
        rnum = g.get("round")
        if rnum is not None:
            by_round[rnum].append(g)

    def squad_for_display(team: str) -> list:
        return [
            {"name": p["name"], "telo": p["telo"], "pos": p["pos"], "status": p["status"]}
            for p in sorted(team_squads.get(team, []),
                            key=lambda x: x["telo"], reverse=True)
            if p["games"] >= PLAYER_MIN_GAMES
        ][:30]

    rounds_output: dict = {}

    for rnum in sorted(by_round.keys()):
        rnd_games  = sorted(by_round[rnum], key=lambda g: g.get("date") or "")
        roundname  = rnd_games[0].get("roundname") or f"Round {rnum}"
        date_range = round_date_range(rnd_games)
        games_out: list = []

        for g in rnd_games:
            home, away = g.get("hteam", ""), g.get("ateam", "")
            if not home or not away:
                continue

            h_telo   = ratings.get(home, INITIAL_TELO)
            a_telo   = ratings.get(away, INITIAL_TELO)
            venue    = g.get("venue", "")
            date_str = g.get("date", "")

            v_hga       = compute_venue_hga(venue_stats, venue)
            home_form   = form_adjustment(form_tracking.get(home, []))
            away_form   = form_adjustment(form_tracking.get(away, []))
            h2h_adj     = h2h_adjustment(h2h_record, home, away)
            home_travel = travel_fatigue_penalty(home, venue, last_game_date, date_str)
            away_travel = travel_fatigue_penalty(away, venue, last_game_date, date_str)

            home_injury = INJURY_OVERRIDES.get(home, auto_injury_deltas.get(home, 0.0))
            away_injury = INJURY_OVERRIDES.get(away, auto_injury_deltas.get(away, 0.0))

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
                entry.update({
                    "home_score": hs,
                    "away_score": as_,
                    "upset":      home_fav != (hs > as_),
                })

            if team_squads:
                entry["home_squad"]        = squad_for_display(home)
                entry["away_squad"]        = squad_for_display(away)
                entry["home_squad_impact"] = round(home_injury, 1)
                entry["away_squad_impact"] = round(away_injury, 1)

            games_out.append(entry)

        rounds_output[str(rnum)] = {
            "label":    roundname,
            "range":    date_range,
            "complete": all(g.get("complete") == 100 for g in rnd_games),
            "games":    games_out,
        }

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    print(f"[TELO] Running {MC_SIMULATIONS:,} Monte Carlo simulations...")
    finals_pct, premiers_pct = simulate_finals(ratings, upcoming, wins_map, losses_map)

    # ── Rankings ───────────────────────────────────────────────────────────────
    ranked = sorted(all_teams, key=lambda t: ratings.get(t, INITIAL_TELO), reverse=True)
    rankings_out: list = [{
        "team":           team,
        "telo":           round(ratings.get(team, INITIAL_TELO)),
        "wins":           wins_map.get(team, 0),
        "losses":         losses_map.get(team, 0),
        "ladder_rank":    rank_map.get(team, 0),
        "percentage":     pct_map.get(team, 0.0),
        "points_for":     for_map.get(team, 0),
        "points_against": against_map.get(team, 0),
        "finals_pct":     finals_pct.get(team, 0.0),
        "premiers_pct":   premiers_pct.get(team, 0.0),
    } for team in ranked]

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
        with open("data/predictions.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"[TELO] ✓ Written data/predictions.json ({len(rankings_out)} teams, {len(rounds_output)} rounds)")

        if player_ratings:
            trusted    = [p for p in player_ratings if p["games"] >= PLAYER_MIN_GAMES]
            league_avg = sum(p["avg"] for p in trusted) / len(trusted) if trusted else 0
            with open("data/player_ratings.json", "w") as f:
                json.dump({
                    "updated":    datetime.now(AEST).isoformat(),
                    "league_avg": round(league_avg, 1),
                    "players":    player_ratings,
                }, f, indent=2)
            print(f"[TELO] ✓ Written data/player_ratings.json ({len(player_ratings)} players)")

    return output


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int, default=datetime.now().year)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build_predictions(year=args.year, dry_run=args.dry_run)
