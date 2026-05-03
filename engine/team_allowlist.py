"""
Per-sport allowlist of canonical team names.

Used by `engine/output.build_sport_output` to filter ratings_table() output
before it gets serialised into data/predictions_sports.json.

Why this exists: the engine's ratings dict accumulates stale keys over time —
international exhibition opponents in NBA (Cairns Taipans, Real Madrid, etc.),
all-star squads in WNBA, conference labels in NFL (AFC/NFC), and renamed
franchises in MLB/NHL (Cleveland Indians, Utah Hockey Club, Oakland Athletics).
The ingest-time filter in `ingest/espn.NBA_TEAMS` only blocks new ingestion;
historical residue stays in `data/ratings/{sport}.json` and gets re-serialised
on every run unless filtered at output time.

Rules:
  - Allowlist holds the *canonical current name only*. Renamed/old keys are
    dropped, not merged. Rating-history continuity for renamed teams is a
    separate piece of work if it ends up mattering.
  - Sports without an allowlist entry (soccer leagues, NRL, supernetball) pass
    through unfiltered.
  - NRL is intentionally absent — its ratings file is empty (see
    LAUNCH_BLOCKERS.md). Adding it here would do nothing useful.

If a sport's rules change (new franchise, expansion team, relocation) update
the relevant frozenset and re-run generate.py.
"""

NBA_TEAMS = frozenset({
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
    'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
    'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
    'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
    'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans',
    'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic',
    'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers',
    'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz',
    'Washington Wizards',
})

WNBA_TEAMS = frozenset({
    'Atlanta Dream', 'Chicago Sky', 'Connecticut Sun', 'Dallas Wings',
    'Golden State Valkyries', 'Indiana Fever', 'Las Vegas Aces',
    'Los Angeles Sparks', 'Minnesota Lynx', 'New York Liberty',
    'Phoenix Mercury', 'Seattle Storm', 'Washington Mystics',
})

NFL_TEAMS = frozenset({
    'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
    'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
    'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
    'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars',
    'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers',
    'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings',
    'New England Patriots', 'New Orleans Saints', 'New York Giants',
    'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers',
    'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers',
    'Tennessee Titans', 'Washington Commanders',
})

# MLB — current canonical names. 'Athletics' (without city) is ESPN's 2025+
# designation; 'Oakland Athletics' is the old key. 'Cleveland Guardians' since
# 2022; 'Cleveland Indians' is the old key.
MLB_TEAMS = frozenset({
    'Arizona Diamondbacks', 'Athletics', 'Atlanta Braves', 'Baltimore Orioles',
    'Boston Red Sox', 'Chicago Cubs', 'Chicago White Sox', 'Cincinnati Reds',
    'Cleveland Guardians', 'Colorado Rockies', 'Detroit Tigers', 'Houston Astros',
    'Kansas City Royals', 'Los Angeles Angels', 'Los Angeles Dodgers',
    'Miami Marlins', 'Milwaukee Brewers', 'Minnesota Twins', 'New York Mets',
    'New York Yankees', 'Philadelphia Phillies', 'Pittsburgh Pirates',
    'San Diego Padres', 'San Francisco Giants', 'Seattle Mariners',
    'St. Louis Cardinals', 'Tampa Bay Rays', 'Texas Rangers', 'Toronto Blue Jays',
    'Washington Nationals',
})

# NHL — 'Utah Mammoth' is the 2025-26 official designation. 'Utah Hockey Club'
# was the interim 2024-25 name and 'Arizona Coyotes' the franchise's pre-relocation
# name; both are stale keys.
NHL_TEAMS = frozenset({
    'Anaheim Ducks', 'Boston Bruins', 'Buffalo Sabres', 'Calgary Flames',
    'Carolina Hurricanes', 'Chicago Blackhawks', 'Colorado Avalanche',
    'Columbus Blue Jackets', 'Dallas Stars', 'Detroit Red Wings',
    'Edmonton Oilers', 'Florida Panthers', 'Los Angeles Kings', 'Minnesota Wild',
    'Montreal Canadiens', 'Nashville Predators', 'New Jersey Devils',
    'New York Islanders', 'New York Rangers', 'Ottawa Senators',
    'Philadelphia Flyers', 'Pittsburgh Penguins', 'San Jose Sharks',
    'Seattle Kraken', 'St. Louis Blues', 'Tampa Bay Lightning',
    'Toronto Maple Leafs', 'Utah Mammoth', 'Vancouver Canucks',
    'Vegas Golden Knights', 'Washington Capitals', 'Winnipeg Jets',
})

# AFL — matches the names the ESPN-ingest engine actually stores. Note: the
# AFL-specific telo.py + Squiggle pipeline (data/predictions.json) uses
# different canonical names — 'Greater Western Sydney' vs 'GWS GIANTS',
# 'Gold Coast' vs 'Gold Coast SUNS'. This alias mismatch between the two
# pipelines is a known issue (see LAUNCH_BLOCKERS.md) but doesn't affect
# the predictions_sports.json output filtered here.
AFL_TEAMS = frozenset({
    'Adelaide', 'Brisbane Lions', 'Carlton', 'Collingwood', 'Essendon',
    'Fremantle', 'Geelong', 'Gold Coast SUNS', 'GWS GIANTS', 'Hawthorn',
    'Melbourne', 'North Melbourne', 'Port Adelaide', 'Richmond', 'St Kilda',
    'Sydney', 'West Coast', 'Western Bulldogs',
})

NBL_TEAMS = frozenset({
    'Adelaide 36ers', 'Brisbane Bullets', 'Cairns Taipans', 'Illawarra Hawks',
    'Melbourne United', 'New Zealand Breakers', 'Perth Wildcats',
    'South East Melbourne Phoenix', 'Sydney Kings', 'Tasmania JackJumpers',
})

ALLOWLIST: dict[str, frozenset[str]] = {
    'nba':  NBA_TEAMS,
    'wnba': WNBA_TEAMS,
    'nfl':  NFL_TEAMS,
    'mlb':  MLB_TEAMS,
    'nhl':  NHL_TEAMS,
    'afl':  AFL_TEAMS,
    'nbl':  NBL_TEAMS,
}


def filter_ratings(sport: str, ratings: list[dict]) -> list[dict]:
    """
    Filter a ratings_table() list by the per-sport allowlist and re-rank.

    The input is the output of EloEngine.ratings_table() — a list of
    {'team', 'rating', 'rank'} sorted by rating desc. After filtering out
    non-allowlisted entries, ranks are re-numbered 1..N so the output is
    self-consistent.

    Sports without an allowlist entry (soccer leagues, supernetball, NRL)
    pass through unchanged.
    """
    allow = ALLOWLIST.get(sport)
    if allow is None:
        return ratings
    filtered = [r for r in ratings if r['team'] in allow]
    return [{**r, 'rank': i + 1} for i, r in enumerate(filtered)]
