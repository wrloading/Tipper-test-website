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
import re
import sys
import time
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
    "Adelaide":                "SA",
    "Brisbane Lions":          "QLD",
    "Carlton":                 "VIC",
    "Collingwood":             "VIC",
    "Essendon":                "VIC",
    "Fremantle":               "WA",
    "Geelong":                 "VIC",
    "Gold Coast":              "QLD",
    "Greater Western Sydney":  "NSW",
    "Hawthorn":                "VIC",
    "Melbourne":               "VIC",
    "North Melbourne":         "VIC",
    "Port Adelaide":           "SA",
    "Richmond":                "VIC",
    "St Kilda":                "VIC",
    "Sydney":                  "NSW",
    "West Coast":              "WA",
    "Western Bulldogs":        "VIC",
}

# ─── AFL FANTASY PLAYER DATA ─────────────────────────────────────────────────

AFL_FANTASY_PLAYERS_URL = "https://fantasy.afl.com.au/data/afl/players.json"
AFL_FANTASY_UA          = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"

PLAYER_INITIAL_TELO = 1500.0
PLAYER_SCORE_SCALE  = 4.0     # TELO pts per avg PI pt above/below league avg
PLAYER_MIN_GAMES    = 2       # games before a rating is used in team impact calc
SQUAD_IMPACT_SCALE  = 0.30    # player-avg delta → team TELO impact multiplier
PI_RECENCY_DECAY    = 0.85    # per-game exponential decay (half-life ≈ 4.3 games)
RATING_MAX          = 99      # FIFA/2K-style rating ceiling
RATING_MIN          = 40      # FIFA/2K-style rating floor

# AFL Fantasy position IDs (verified: 1=DEF, 2=MID, 3=RUC, 4=FWD)
FANTASY_POSITIONS: dict[int, str] = {1: "DEF", 2: "MID", 3: "RUC", 4: "FWD"}

# AFL Fantasy squad_id → Squiggle team name
FANTASY_SQUAD_MAP: dict[int, str] = {
    10:   "Adelaide",
    20:   "Brisbane Lions",
    30:   "Carlton",
    40:   "Collingwood",
    50:   "Essendon",
    60:   "Fremantle",
    70:   "Geelong",
    1000: "Gold Coast",
    1010: "Greater Western Sydney",
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

# ─── AFL TABLES SCRAPING ─────────────────────────────────────────────────────

AFLTABLES_BASE  = "https://afltables.com"
AFLTABLES_UA    = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"
AFLTABLES_DELAY = 0.25  # seconds between requests (be a good citizen)

# Non-player rows that appear in the Match Statistics table footer
AFLTABLES_SKIP_ROWS = {"totals", "total", "opposition", "rushed", "behinds"}

# Normalise AFL Tables team name (from page heading) → Squiggle team name
AFLTABLES_TEAM_MAP: dict[str, str] = {
    "adelaide":                "Adelaide",
    "bris. lions":             "Brisbane Lions",
    "brisbane lions":          "Brisbane Lions",
    "brisbane":                "Brisbane Lions",
    "carlton":                 "Carlton",
    "collingwood":             "Collingwood",
    "essendon":                "Essendon",
    "fremantle":               "Fremantle",
    "geelong":                 "Geelong",
    "gold coast":              "Gold Coast",
    "gw sydney":               "Greater Western Sydney",
    "greater western sydney":  "Greater Western Sydney",
    "hawthorn":                "Hawthorn",
    "melbourne":               "Melbourne",
    "nth melbourne":           "North Melbourne",
    "north melbourne":         "North Melbourne",
    "port adelaide":           "Port Adelaide",
    "richmond":                "Richmond",
    "st kilda":                "St Kilda",
    "sydney":                  "Sydney",
    "west coast":              "West Coast",
    "w. bulldogs":             "Western Bulldogs",
    "western bulldogs":        "Western Bulldogs",
}

# ─── FOOTYWIRE TEAM SELECTIONS ────────────────────────────────────────────────

FOOTYWIRE_UA = "Tipper-TELO/2.0 (github.com/wrloading/Tipper-test-website)"

# FootyWire player link team slug → Squiggle team name
FOOTYWIRE_TEAM_SLUG_MAP: dict[str, str] = {
    "adelaide-crows":                  "Adelaide",
    "brisbane-lions":                  "Brisbane Lions",
    "carlton-blues":                   "Carlton",
    "collingwood-magpies":             "Collingwood",
    "essendon-bombers":                "Essendon",
    "fremantle-dockers":               "Fremantle",
    "geelong-cats":                    "Geelong",
    "gold-coast-suns":                 "Gold Coast",
    "greater-western-sydney-giants":   "Greater Western Sydney",
    "hawthorn-hawks":                  "Hawthorn",
    "kangaroos":                       "North Melbourne",
    "melbourne-demons":                "Melbourne",
    "port-adelaide-power":             "Port Adelaide",
    "richmond-tigers":                 "Richmond",
    "st-kilda-saints":                 "St Kilda",
    "sydney-swans":                    "Sydney",
    "west-coast-eagles":               "West Coast",
    "western-bulldogs":                "Western Bulldogs",
}

# FootyWire game-heading team names → Squiggle team name
FOOTYWIRE_NAME_MAP: dict[str, str] = {
    "adelaide":                "Adelaide",
    "brisbane":                "Brisbane Lions",
    "brisbane lions":          "Brisbane Lions",
    "carlton":                 "Carlton",
    "collingwood":             "Collingwood",
    "essendon":                "Essendon",
    "fremantle":               "Fremantle",
    "geelong":                 "Geelong",
    "gold coast":              "Gold Coast",
    "gws":                     "Greater Western Sydney",
    "greater western sydney":  "Greater Western Sydney",
    "hawthorn":                "Hawthorn",
    "melbourne":               "Melbourne",
    "north melbourne":         "North Melbourne",
    "port adelaide":           "Port Adelaide",
    "richmond":                "Richmond",
    "st kilda":                "St Kilda",
    "sydney":                  "Sydney",
    "west coast":              "West Coast",
    "western bulldogs":        "Western Bulldogs",
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

# ─── AFL TABLES FUNCTIONS ────────────────────────────────────────────────────

def fetch_afltables_season_links(year: int) -> list:
    """Return all game stat page URLs for the given year from AFL Tables."""
    try:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
    except ImportError:
        print("[TELO] ⚠ beautifulsoup4 not installed — skipping AFL Tables", file=sys.stderr)
        return []
    page_url = f"{AFLTABLES_BASE}/afl/seas/{year}.html"
    try:
        r = requests.get(page_url, headers={"User-Agent": AFLTABLES_UA}, timeout=20)
        r.raise_for_status()
        soup  = BeautifulSoup(r.text, "html.parser")
        seen  = set()
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"stats/games/{year}/" not in href or not href.endswith(".html"):
                continue
            full = urljoin(page_url, href)
            if full not in seen:
                seen.add(full)
                links.append(full)
        return links
    except Exception as e:
        print(f"[TELO] ⚠ AFL Tables season page failed: {e}", file=sys.stderr)
        return []


def fetch_afltables_game_lineup(url: str) -> tuple:
    """
    Fetch per-player game stats from an AFL Tables game stat page.
    Returns (home_lineup, away_lineup, home_team, away_team) where each lineup is a
    list of {name, di, gl, tk, cl, ho, pct, pi} dicts. Returns (None,None,None,None) on error.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None, None, None, None
    try:
        r = requests.get(url, headers={"User-Agent": AFLTABLES_UA}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        stats_tables = [t for t in soup.find_all("table")
                        if "Match Statistics" in t.get_text()]
        if len(stats_tables) < 2:
            return None, None, None, None

        def parse_stats_table(table) -> tuple:
            rows = table.find_all("tr")
            if len(rows) < 3:
                return None, None

            heading   = rows[0].get_text(separator=" ", strip=True)
            team_raw  = heading.split("Match Statistics")[0].strip()
            team_name = AFLTABLES_TEAM_MAP.get(team_raw.lower())

            headers = [th.get_text(strip=True) for th in rows[1].find_all(["th", "td"])]
            try:
                name_idx = headers.index("Player")
            except ValueError:
                return team_name, None

            def col(name: str) -> int:
                return headers.index(name) if name in headers else -1

            ki_idx  = col("KI");  mk_idx  = col("MK");  hb_idx  = col("HB")
            di_idx  = col("DI");  gl_idx  = col("GL");  bh_idx  = col("BH")
            ho_idx  = col("HO");  tk_idx  = col("TK");  rb_idx  = col("RB")
            if_idx  = col("IF");  cl_idx  = col("CL");  cg_idx  = col("CG")
            fa_idx  = col("FA");  cp_idx  = col("CP");  mi_idx  = col("MI")
            ga_idx  = col("GA");  pct_idx = len(headers) - 1  # %P always last

            def safe_int(texts: list, idx: int) -> int:
                if idx < 0 or idx >= len(texts):
                    return 0
                try:
                    return int(texts[idx]) if texts[idx] else 0
                except (ValueError, TypeError):
                    return 0

            players = []
            for row in rows[2:]:
                cells = row.find_all(["th", "td"])
                if len(cells) <= name_idx:
                    continue
                texts = [c.get_text(strip=True).replace("\xa0", "").strip() for c in cells]
                row_label = texts[0].lower() if texts else ""
                if row_label in AFLTABLES_SKIP_ROWS:
                    continue
                name_raw = texts[name_idx] if name_idx < len(texts) else ""
                if not name_raw or name_raw.isdigit():
                    continue
                if "," in name_raw:
                    last, first = name_raw.split(",", 1)
                    name = f"{first.strip()} {last.strip()}"
                else:
                    name = name_raw

                s = lambda idx: safe_int(texts, idx)
                ki = s(ki_idx); mk = s(mk_idx); hb = s(hb_idx)
                di = s(di_idx); gl = s(gl_idx); bh = s(bh_idx)
                ho = s(ho_idx); tk = s(tk_idx); rb = s(rb_idx)
                if_ = s(if_idx); cl = s(cl_idx); cg = s(cg_idx)
                fa = s(fa_idx); cp = s(cp_idx); mi = s(mi_idx)
                ga = s(ga_idx); pct = s(pct_idx)

                # Enhanced Player Impact — AFL Fantasy-calibrated weights
                # Kicks > handballs, marks inside 50 are premium, negatives for clangers/FA
                pi = round(
                    ki  * 3.2 + hb  * 2.2 + mk  * 3.2
                  + gl  * 8.0 + bh  * 0.5
                  + tk  * 4.0 + ho  * 1.0
                  + cl  * 4.0 + cp  * 1.5
                  + mi  * 5.0 + if_ * 1.5
                  + rb  * 1.0 + ga  * 2.0
                  - cg  * 1.0 - fa  * 3.0
                )
                players.append({
                    "name": name,
                    "di": di, "gl": gl, "tk": tk,
                    "cl": cl, "ho": ho, "pct": pct, "pi": max(pi, 0),
                })
            return team_name, players if players else None

        home_team, home_lineup = parse_stats_table(stats_tables[0])
        away_team, away_lineup = parse_stats_table(stats_tables[1])
        return home_lineup, away_lineup, home_team, away_team

    except Exception as e:
        print(f"[TELO] ⚠ AFL Tables lineup failed ({url}): {e}", file=sys.stderr)
        return None, None, None, None

# ─── FOOTYWIRE SELECTIONS ────────────────────────────────────────────────────

def fetch_footywire_selections() -> dict:
    """
    Fetch current round team selections from FootyWire.
    Returns dict keyed by (home_squiggle, away_squiggle) → {
        round, home_named, away_named, home_outs, away_outs
    } where each named list is [{name, pos}].
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {}
    url = "https://www.footywire.com/afl/footy/afl_team_selections"
    try:
        r = requests.get(url, headers={"User-Agent": FOOTYWIRE_UA}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"[TELO] ⚠ FootyWire selections failed: {e}", file=sys.stderr)
        return {}

    # Parse round number from <h1> heading
    round_num = 0
    h1 = soup.find("h1")
    if h1:
        m = re.search(r"Round\s+(\d+)", h1.get_text())
        if m:
            round_num = int(m.group(1))

    selections: dict = {}

    for title_td in soup.find_all("td", class_="tbtitle"):
        game_text = title_td.get_text(strip=True)
        # "Adelaide v Carlton (Adelaide Oval)"
        m = re.match(r"^(.+?)\s+v\s+(.+?)(?:\s*\(|$)", game_text)
        if not m:
            continue
        home_sq = FOOTYWIRE_NAME_MAP.get(m.group(1).strip().lower())
        away_sq = FOOTYWIRE_NAME_MAP.get(m.group(2).strip().lower())
        if not home_sq or not away_sq:
            print(f"[TELO]   FootyWire: unknown teams in '{game_text}'", file=sys.stderr)
            continue

        # Data row immediately follows the title row
        title_row = title_td.find_parent("tr")
        if not title_row:
            continue
        data_row = title_row.find_next_sibling("tr")
        if not data_row:
            continue
        tds = data_row.find_all("td", recursive=False)
        if len(tds) < 3:
            continue
        left_td, mid_td, right_td = tds[0], tds[1], tds[2]

        # ── Extract named players from the middle lineup grid ──────────────────
        home_players: list = []
        away_players: list = []
        lineup_table = mid_td.find("table")
        if lineup_table:
            for row in lineup_table.find_all("tr"):
                cells = row.find_all("td")
                if not cells:
                    continue
                pos_raw = cells[0].get_text(strip=True).replace("\xa0", "").strip()
                pos = pos_raw if pos_raw in ("FB", "HB", "C", "HF", "FF", "Fol") else None
                if not pos:
                    continue
                for cell in cells[1:]:
                    link = cell.find("a", href=True)
                    if not link:
                        continue
                    m2 = re.match(r"pp-(.+?)--", link["href"])
                    if not m2:
                        continue
                    slug = m2.group(1)
                    name = link.get_text(strip=True)
                    mapped = FOOTYWIRE_TEAM_SLUG_MAP.get(slug)
                    if mapped == home_sq:
                        home_players.append({"name": name, "pos": pos})
                    elif mapped == away_sq:
                        away_players.append({"name": name, "pos": pos})

        # ── Extract interchange + outs from side columns ───────────────────────
        def parse_side_column(td) -> tuple:
            interchange: list = []
            outs: list = []
            section = None
            for row in td.find_all("tr"):
                for cell in row.find_all("td"):
                    bold = cell.find("b")
                    if bold:
                        label = bold.get_text(strip=True).lower()
                        section = ("interchange" if "interchange" in label
                                   else "outs" if label == "outs"
                                   else None)
                        continue
                    link = cell.find("a", href=True)
                    if link and section:
                        name = link.get_text(strip=True)
                        if section == "interchange":
                            interchange.append({"name": name, "pos": "INT"})
                        elif section == "outs":
                            outs.append(name)
            return interchange, outs

        home_bench, home_outs = parse_side_column(left_td)
        away_bench, away_outs = parse_side_column(right_td)

        # Merge and deduplicate (lineup grid + interchange)
        def dedup(lst: list) -> list:
            seen: set = set()
            out: list = []
            for p in lst:
                if p["name"] not in seen:
                    seen.add(p["name"])
                    out.append(p)
            return out

        selections[(home_sq, away_sq)] = {
            "round":      round_num,
            "home_named": dedup(home_players + home_bench),
            "away_named": dedup(away_players + away_bench),
            "home_outs":  home_outs,
            "away_outs":  away_outs,
        }

    print(f"[TELO]   FootyWire: {len(selections)} games with selections (Round {round_num})")
    return selections

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

    # ── AFL Tables historical lineups ──────────────────────────────────────────
    print("[TELO] Fetching AFL Tables historical game lineups...")
    # Key: (YYYYMMDD, squiggle_home_name, squiggle_away_name)  →  {home_lineup, away_lineup}
    afltables_lineups: dict = {}
    # Most recent lineup per team: team → [{name, pos}]  (for fallback on unannounced games)
    team_recent_lineup: dict = {}
    # Season stats accumulator: full_name → {di, gl, tk, cl, pi, games}
    player_season_stats: dict = {}
    try:
        game_links = fetch_afltables_season_links(year)
        print(f"[TELO]   {len(game_links)} game stat pages found on AFL Tables")
        for i, link_url in enumerate(game_links):
            if i > 0:
                time.sleep(AFLTABLES_DELAY)
            h_lineup, a_lineup, h_team, a_team = fetch_afltables_game_lineup(link_url)
            if h_lineup and a_lineup and h_team and a_team:
                m = re.search(r"(\d{8})\.html$", link_url)
                if m:
                    date_key = m.group(1)  # YYYYMMDD
                    afltables_lineups[(date_key, h_team, a_team)] = {
                        "home_lineup": h_lineup,
                        "away_lineup": a_lineup,
                    }
                    # Update per-team latest lineup (processed in date order → last write wins)
                    team_recent_lineup[h_team] = [{"name": p["name"], "pos": ""} for p in h_lineup]
                    team_recent_lineup[a_team] = [{"name": p["name"], "pos": ""} for p in a_lineup]
                    # Accumulate season stats per player (full names from AFL Tables)
                    for p in h_lineup + a_lineup:
                        acc = player_season_stats.setdefault(p["name"], {
                            "di": 0, "gl": 0, "tk": 0, "cl": 0, "games": 0,
                            "pi_list": [],   # per-game PI in chronological order
                        })
                        acc["di"]    += p["di"]
                        acc["gl"]    += p["gl"]
                        acc["tk"]    += p["tk"]
                        acc["cl"]    += p["cl"]
                        acc["games"] += 1
                        acc["pi_list"].append(p["pi"])
        print(f"[TELO]   {len(afltables_lineups)} completed game lineups cached, "
              f"{len(player_season_stats)} players in stats index")
    except Exception as e:
        print(f"[TELO] ⚠ AFL Tables scraping error: {e}", file=sys.stderr)
        afltables_lineups = {}
        team_recent_lineup = {}

    def weighted_pi_avg(pi_list: list) -> float:
        """
        Exponentially weighted average of per-game PI values.
        Most recent game has weight 1.0; each prior game is multiplied by PI_RECENCY_DECAY.
        This captures form shifts and role changes without discarding history entirely.
        """
        if not pi_list:
            return 0.0
        n = len(pi_list)
        weights = [PI_RECENCY_DECAY ** (n - 1 - i) for i in range(n)]
        return sum(pi * w for pi, w in zip(pi_list, weights)) / sum(weights)

    # Build name-lookup index for matching abbreviated FootyWire names to full AFL Tables names
    # Index key: normalised last name → [(full_name, avg_stats_dict), ...]
    _name_idx: dict = {}
    for full_name, acc in player_season_stats.items():
        g = max(acc["games"], 1)
        avg = {
            "avg_di": round(acc["di"] / g, 1),
            "avg_gl": round(acc["gl"] / g, 1),
            "avg_tk": round(acc["tk"] / g, 1),
            "avg_cl": round(acc["cl"] / g, 1),
            "avg_pi": round(weighted_pi_avg(acc["pi_list"]), 1),  # recency-weighted
            "games":  acc["games"],
        }
        # Index by last word of name (handles "De Koning" → "Koning")
        last = full_name.split()[-1].lower()
        _name_idx.setdefault(last, []).append((full_name, avg))

    def lookup_avg_stats(display_name: str) -> Optional[dict]:
        """Match 'J De Koning' or 'Sam De Koning' to stats by last name + first initial."""
        parts = display_name.strip().split()
        if len(parts) < 2:
            return None
        initial = parts[0][0].upper()
        last    = parts[-1].lower()
        candidates = _name_idx.get(last, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][1]
        # Multiple players share last name — disambiguate by first initial
        for full_name, avg in candidates:
            if full_name.split()[0][0].upper() == initial:
                return avg
        return None

    # ── PI-based P-TELO: override Fantasy ratings with AFL Tables data ────────────
    # Compute each player's recency-weighted PI average, normalise against the league,
    # and use that as their P-TELO.  Players without AFL Tables data keep their
    # Fantasy-derived rating as a fallback.
    pi_rated = [
        (full_name, weighted_pi_avg(acc["pi_list"]))
        for full_name, acc in player_season_stats.items()
        if acc["games"] >= PLAYER_MIN_GAMES
    ]
    if pi_rated:
        _league_pi_avg = sum(w for _, w in pi_rated) / len(pi_rated)
        # map: name → (telo, weighted_avg_pi)
        _pi_data_map: dict[str, tuple] = {
            name: (
                round(PLAYER_INITIAL_TELO + (w_pi - _league_pi_avg) * PLAYER_SCORE_SCALE),
                round(w_pi, 1),
            )
            for name, w_pi in pi_rated
        }
        # Also store per-player last-3-games simple average PI
        _pi_last3_map: dict[str, float] = {
            full_name: round(sum(acc["pi_list"][-3:]) / len(acc["pi_list"][-3:]), 1)
            for full_name, acc in player_season_stats.items()
            if acc["games"] >= PLAYER_MIN_GAMES and acc["pi_list"]
        }
    else:
        _league_pi_avg = 0.0
        _pi_data_map   = {}
        _pi_last3_map  = {}

    # Keep a simple telo-only map for downstream usage
    _pi_telo_map = {n: v[0] for n, v in _pi_data_map.items()}

    def _resolve_pi_data(full_name: str) -> Optional[tuple]:
        """Return (telo, avg_pi) for a player by exact or last+initial match."""
        if full_name in _pi_data_map:
            return _pi_data_map[full_name]
        parts = full_name.strip().split()
        if len(parts) < 2:
            return None
        initial = parts[0][0].upper()
        last    = parts[-1].lower()
        matches = [(n, v) for n, v in _pi_data_map.items()
                   if n.split()[-1].lower() == last and n.split()[0][0].upper() == initial]
        return matches[0][1] if len(matches) == 1 else None

    def _resolve_pi_telo(full_name: str) -> Optional[int]:
        d = _resolve_pi_data(full_name)
        return d[0] if d else None

    def _resolve_last3_pi(full_name: str) -> Optional[float]:
        if full_name in _pi_last3_map:
            return _pi_last3_map[full_name]
        parts = full_name.strip().split()
        if len(parts) < 2:
            return None
        initial = parts[0][0].upper()
        last    = parts[-1].lower()
        matches = [(n, v) for n, v in _pi_last3_map.items()
                   if n.split()[-1].lower() == last and n.split()[0][0].upper() == initial]
        return matches[0][1] if len(matches) == 1 else None

    # Patch player_ratings: telo → PI-based, add avg_pi and last3_pi fields
    for _pr in player_ratings:
        _pd = _resolve_pi_data(_pr["name"])
        if _pd is not None:
            _pr["telo"]    = _pd[0]
            _pr["avg_pi"]  = _pd[1]
        _l3 = _resolve_last3_pi(_pr["name"])
        if _l3 is not None:
            _pr["last3_pi"] = _l3

    # Compute per-player scaled rating (40-99) and league rank across ALL rated players
    # Use PI-rated players to set the scale anchors, then apply to everyone
    _all_eligible = [p for p in player_ratings if p["games"] >= PLAYER_MIN_GAMES]
    if _all_eligible:
        _sorted_by_telo = sorted(_all_eligible, key=lambda p: p["telo"], reverse=True)
        _hi = _sorted_by_telo[0]["telo"]
        _lo = _sorted_by_telo[-1]["telo"]
        _span = (_hi - _lo) if _hi != _lo else 1
        for _rank, _pr in enumerate(_sorted_by_telo, 1):
            _pr["player_rating"] = round(RATING_MIN + (_pr["telo"] - _lo) / _span * (RATING_MAX - RATING_MIN))
            _pr["player_rank"]   = _rank

    full_squad_telo_avg.clear()
    auto_injury_deltas.clear()
    for _team, _squad in team_squads.items():
        _trusted = [p for p in _squad if p["games"] >= PLAYER_MIN_GAMES]
        if _trusted:
            full_squad_telo_avg[_team] = sum(p["telo"] for p in _trusted) / len(_trusted)
    for _team, _squad in team_squads.items():
        _base = full_squad_telo_avg.get(_team, PLAYER_INITIAL_TELO)
        _avail = [p for p in _squad if p["status"] == "available" and p["games"] >= PLAYER_MIN_GAMES]
        if _avail:
            _avail_avg = sum(p["telo"] for p in _avail) / len(_avail)
            _delta = (_avail_avg - _base) * SQUAD_IMPACT_SCALE
            if abs(_delta) > 0.5:
                auto_injury_deltas[_team] = round(_delta, 1)
    print(f"[TELO]   PI-based P-TELO: {len(_pi_telo_map)} players rated, "
          f"{sum(1 for p in player_ratings if _resolve_pi_telo(p['name']) is not None)} Fantasy entries updated")

    # P-TELO lookup index for named squad win-prob adjustment
    _ptelo_idx: dict = {}
    for _pr in player_ratings:
        _name_parts = _pr["name"].split()
        # Index by last word AND full surname (handles "De Goey" → both "goey" and "de goey")
        _ptelo_idx.setdefault(_name_parts[-1].lower(), []).append(_pr)
        if len(_name_parts) > 2:
            _full_last = " ".join(_name_parts[1:]).lower()
            _ptelo_idx.setdefault(_full_last, []).append(_pr)

    def _match_player(display_name: str, team: str) -> Optional[dict]:
        """Return the player_ratings entry for an abbreviated name + team."""
        parts = display_name.strip().split()
        if len(parts) < 2:
            return None
        initial = parts[0][0].upper()
        # Try full surname first (e.g. "J De Goey" → "de goey"), then last word
        last_full = " ".join(parts[1:]).lower()
        last_word = parts[-1].lower()
        candidates = _ptelo_idx.get(last_full) or _ptelo_idx.get(last_word, [])
        if not candidates:
            return None
        team_cands = [c for c in candidates if c["team"] == team]
        search = team_cands if team_cands else candidates
        if len(search) == 1:
            return search[0]
        for c in search:
            if c["name"].split()[0][0].upper() == initial:
                return c
        return None

    def lookup_player_telo(display_name: str, team: str) -> Optional[float]:
        c = _match_player(display_name, team)
        return float(c["telo"]) if c else None

    def lookup_player_telo_data(display_name: str, team: str) -> Optional[dict]:
        """Return {telo, pos, player_rating, player_rank} for a named player."""
        c = _match_player(display_name, team)
        if not c:
            return None
        return {
            "telo":          int(c["telo"]),
            "pos":           c["pos"],
            "player_rating": c.get("player_rating"),
            "player_rank":   c.get("player_rank"),
        }

    def compute_named_squad_delta(named: list, team: str) -> Optional[float]:
        """
        Return the TELO adjustment implied by a team's actual named squad
        vs their full-squad baseline.  Requires ≥10 matched players.
        """
        base = full_squad_telo_avg.get(team)
        if not base:
            return None
        telos = [t for t in (lookup_player_telo(p.get("name", ""), team) for p in named)
                 if t is not None]
        if len(telos) < 10:
            return None
        delta = (sum(telos) / len(telos) - base) * SQUAD_IMPACT_SCALE
        return round(delta, 1) if abs(delta) > 0.3 else 0.0

    # ── FootyWire team selections ──────────────────────────────────────────────
    print("[TELO] Fetching FootyWire team selections...")
    try:
        footywire_sels = fetch_footywire_selections()
    except Exception as e:
        print(f"[TELO] ⚠ FootyWire selections error: {e}", file=sys.stderr)
        footywire_sels = {}

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

            is_complete = g.get("complete") == 100

            home_injury = INJURY_OVERRIDES.get(home, auto_injury_deltas.get(home, 0.0))
            away_injury = INJURY_OVERRIDES.get(away, auto_injury_deltas.get(away, 0.0))

            # For upcoming games with announced named squads, refine squad delta
            # using actual named-player P-TELO vs full-squad baseline
            if not is_complete:
                sel = footywire_sels.get((home, away))
                if sel:
                    h_delta = compute_named_squad_delta(sel["home_named"], home)
                    a_delta = compute_named_squad_delta(sel["away_named"], away)
                    if h_delta is not None and home not in INJURY_OVERRIDES:
                        home_injury = h_delta
                    if a_delta is not None and away not in INJURY_OVERRIDES:
                        away_injury = a_delta

            h_eff = h_telo + home_form + h2h_adj + home_injury - home_travel
            a_eff = a_telo + away_form             + away_injury - away_travel

            h_prob      = round(win_probability_pct(h_eff, a_eff, hga=v_hga), 1)
            pred_margin = round(abs(predict_margin(h_eff, a_eff, hga=v_hga)), 1)
            home_fav    = h_prob >= 50.0

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
                # Attach AFL Tables lineup if available
                if date_str and afltables_lineups:
                    date_key = date_str[:10].replace("-", "")
                    lineup_data = afltables_lineups.get((date_key, home, away))
                    if lineup_data:
                        entry["home_lineup"] = lineup_data["home_lineup"]
                        entry["away_lineup"] = lineup_data["away_lineup"]

            # Upcoming game squad: prefer FootyWire named squad, fallback to last week's lineup
            if not is_complete:
                sel = footywire_sels.get((home, away))
                if sel:
                    entry["home_named"]           = sel["home_named"]
                    entry["away_named"]           = sel["away_named"]
                    entry["home_outs"]            = sel["home_outs"]
                    entry["away_outs"]            = sel["away_outs"]
                    entry["selections_announced"] = True
                    entry["selections_round"]     = sel["round"]
                else:
                    h_recent = team_recent_lineup.get(home)
                    a_recent = team_recent_lineup.get(away)
                    if h_recent or a_recent:
                        entry["home_named"]           = h_recent or []
                        entry["away_named"]           = a_recent or []
                        entry["home_outs"]            = []
                        entry["away_outs"]            = []
                        entry["selections_announced"] = False
                # Attach season avg stats + P-TELO to each named player
                for side, team_name in (("home_named", home), ("away_named", away)):
                    enriched = []
                    for p in entry.get(side, []):
                        merged = {**p}
                        avg = lookup_avg_stats(p["name"])
                        if avg:
                            merged.update(avg)
                        pd = lookup_player_telo_data(p["name"], team_name)
                        if pd:
                            merged["ptelo"]         = pd["telo"]
                            merged["ppos"]          = pd["pos"]
                            merged["player_rating"] = pd.get("player_rating")
                            merged["player_rank"]   = pd.get("player_rank")
                        enriched.append(merged)
                    if enriched:
                        entry[side] = enriched
                # Enrich outs: convert name strings → objects with avg stats + P-TELO
                for side_outs, team_name in (("home_outs", home), ("away_outs", away)):
                    raw_outs = entry.get(side_outs, [])
                    if raw_outs and isinstance(raw_outs[0], str):
                        enriched_outs = []
                        for name in raw_outs:
                            out_obj = {"name": name}
                            avg = lookup_avg_stats(name)
                            if avg:
                                out_obj.update(avg)
                            pd = lookup_player_telo_data(name, team_name)
                            if pd:
                                out_obj["ptelo"]         = pd["telo"]
                                out_obj["player_rating"] = pd.get("player_rating")
                                out_obj["player_rank"]   = pd.get("player_rank")
                            enriched_outs.append(out_obj)
                        entry[side_outs] = enriched_outs
                # Squad impact delta from injury auto-adjust
                entry["home_squad_impact"] = round(home_injury, 1)
                entry["away_squad_impact"] = round(away_injury, 1)

            games_out.append(entry)

        rounds_output[str(rnum)] = {
            "label":    roundname,
            "range":    date_range,
            "complete": all(g.get("complete") == 100 for g in rnd_games),
            "games":    games_out,
        }

    # ── Positional league ratings ───────────────────────────────────────────────
    # For each team, compute avg P-TELO by position category using their most
    # recent named squad, then rank all 18 teams per category.
    FW_POS_CAT   = {"FB": "DEF", "HB": "DEF", "C": "MID", "HF": "MID", "FF": "FWD", "Fol": "MID"}
    FAN_POS_CAT  = {"DEF": "DEF", "MID": "MID", "FWD": "FWD", "RUC": "MID"}
    POS_CATS     = ("DEF", "MID", "FWD")

    # Walk rounds in order; later rounds overwrite earlier so we keep the freshest squad
    team_latest_named: dict = {}
    for rnum_str in sorted(rounds_output, key=lambda x: int(x)):
        for ge in rounds_output[rnum_str]["games"]:
            if ge.get("complete"):
                continue
            for side, tk in (("home_named", "home"), ("away_named", "away")):
                players = ge.get(side, [])
                team    = ge.get(tk)
                if players and team:
                    team_latest_named[team] = players

    # Per-team, per-category avg P-TELO
    team_pos_avgs: dict = {}
    for team, players in team_latest_named.items():
        cat_telos: dict = {c: [] for c in POS_CATS}
        for p in players:
            ptelo = p.get("ptelo")
            if not ptelo:
                continue
            cat = FW_POS_CAT.get(p.get("pos", "")) or FAN_POS_CAT.get(p.get("ppos", ""))
            if cat:
                cat_telos[cat].append(ptelo)
        avgs = {c: round(sum(v) / len(v)) for c, v in cat_telos.items() if v}
        if avgs:
            team_pos_avgs[team] = avgs

    # Store raw avg P-TELO per category (team ratings shown as-is)
    positional_ratings: dict = {t: {} for t in team_pos_avgs}
    for cat in POS_CATS:
        for team, avgs in team_pos_avgs.items():
            if cat in avgs:
                positional_ratings[team][cat] = {"avg": avgs[cat]}

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
        "rounds":              rounds_output,
        "rankings":            rankings_out,
        "positional_ratings":  positional_ratings,
    }

    if dry_run:
        print(json.dumps(output, indent=2))
    else:
        os.makedirs("data", exist_ok=True)
        with open("data/predictions.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"[TELO] ✓ Written data/predictions.json ({len(rankings_out)} teams, {len(rounds_output)} rounds)")

        if player_ratings:
            with open("data/player_ratings.json", "w") as f:
                json.dump({
                    "updated":    datetime.now(AEST).isoformat(),
                    "league_avg": round(_league_pi_avg, 1),
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
