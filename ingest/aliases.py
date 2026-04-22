from __future__ import annotations

"""
Team name normalisation.

The problem: data sources use different names for the same team.
  ESPN:              "Golden State Warriors"
  football-data.co.uk: "Man City", "Man United", "Spurs"
  AFL Tables:        "GWS", "Brisbane Lions"

The solution: a canonical name per team + alias maps that route any known
variant to the canonical form. The canonical names match ESPN display names
since that's what the app already uses for game matching.

To add a new alias: add it to the relevant sport's dict below.
The normalise() function handles everything else.
"""

from typing import Optional

# ── Soccer — maps source names → ESPN canonical names ────────────────────────

EPL_ALIASES: dict[str, str] = {
    # football-data.co.uk uses short names
    'Man City':           'Manchester City',
    'Man United':         'Manchester United',
    'Spurs':              'Tottenham Hotspur',
    'Wolves':             'Wolverhampton Wanderers',
    'Brighton':           'Brighton & Hove Albion',
    'Nott\'m Forest':     'Nottingham Forest',
    'Sheffield Utd':      'Sheffield United',
    'West Brom':          'West Bromwich Albion',
    'QPR':                'Queens Park Rangers',
    'Huddersfield':       'Huddersfield Town',
    'Stoke':              'Stoke City',
    "Norwich":            'Norwich City',
    'Swansea':            'Swansea City',
    'Middlesbrough':      'Middlesbrough',
    'Blackburn':          'Blackburn Rovers',
    'Cardiff':            'Cardiff City',
    'Hull':               'Hull City',
    'Ipswich':            'Ipswich Town',
    'Leicester':          'Leicester City',
    'Leeds':              'Leeds United',
    'Coventry':           'Coventry City',
    'Derby':              'Derby County',
    'Millwall':           'Millwall',
    'Preston':            'Preston North End',
    'Watford':            'Watford',
    'Wigan':              'Wigan Athletic',
    'Bolton':             'Bolton Wanderers',
    'Burnley':            'Burnley',
    'Luton':              'Luton Town',
    'Sunderland':         'Sunderland',
    'Brentford':          'Brentford',
}

LA_LIGA_ALIASES: dict[str, str] = {
    'Ath Bilbao':         'Athletic Club',
    'Ath Madrid':         'Atletico Madrid',
    'Atletico Madrid':    'Atletico Madrid',
    'Betis':              'Real Betis',
    'Celta':              'Celta Vigo',
    'Espanol':            'Espanyol',
    'La Coruna':          'Deportivo La Coruna',
    'Leganes':            'Leganes',
    'Malaga':             'Malaga',
    'R Sociedad':         'Real Sociedad',
    'Sevilla':            'Sevilla',
    'Sp Gijon':           'Sporting Gijon',
    'Valencia':           'Valencia',
    'Valladolid':         'Valladolid',
    'Villarreal':         'Villarreal',
    'Cadiz':              'Cadiz',
    'Girona':             'Girona',
    'Las Palmas':         'Las Palmas',
    'Almeria':            'Almeria',
}

BUNDESLIGA_ALIASES: dict[str, str] = {
    'Augsburg':           'Augsburg',
    'Bayern Munich':      'Bayern Munich',
    'Bochum':             'Bochum',
    'Dortmund':           'Borussia Dortmund',
    'Leverkusen':         'Bayer Leverkusen',
    'Ein Frankfurt':      'Eintracht Frankfurt',
    'Fortuna Dusseldorf': 'Fortuna Dusseldorf',
    'Freiburg':           'Freiburg',
    'Greuther Furth':     'Greuther Furth',
    'Hamburger SV':       'Hamburger SV',
    'Hannover':           'Hannover 96',
    'Hertha':             'Hertha Berlin',
    'Hoffenheim':         'Hoffenheim',
    'Koln':               'Cologne',
    'Mainz':              'Mainz',
    'Mgladbach':          'Borussia Monchengladbach',
    'Nurnberg':           'Nurnberg',
    'Paderborn':          'Paderborn',
    'RB Leipzig':         'RB Leipzig',
    'Schalke 04':         'Schalke 04',
    'Stuttgart':          'Stuttgart',
    'Union Berlin':       'Union Berlin',
    'Werder Bremen':      'Werder Bremen',
    'Wolfsburg':          'Wolfsburg',
}

SERIE_A_ALIASES: dict[str, str] = {
    'AC Milan':           'AC Milan',
    'Atalanta':           'Atalanta',
    'Benevento':          'Benevento',
    'Bologna':            'Bologna',
    'Cagliari':           'Cagliari',
    'Chievo':             'Chievo',
    'Como':               'Como',
    'Cremonese':          'Cremonese',
    'Empoli':             'Empoli',
    'Fiorentina':         'Fiorentina',
    'Frosinone':          'Frosinone',
    'Genoa':              'Genoa',
    'Hellas Verona':      'Hellas Verona',
    'Inter':              'Inter Milan',
    'Inter Milan':        'Inter Milan',
    'Juventus':           'Juventus',
    'Lazio':              'Lazio',
    'Lecce':              'Lecce',
    'Monza':              'Monza',
    'Napoli':             'Napoli',
    'Parma':              'Parma',
    'Roma':               'AS Roma',
    'AS Roma':            'AS Roma',
    'Salernitana':        'Salernitana',
    'Sampdoria':          'Sampdoria',
    'Sassuolo':           'Sassuolo',
    'SPAL':               'SPAL',
    'Spezia':             'Spezia',
    'Torino':             'Torino',
    'Udinese':            'Udinese',
    'Venice':             'Venezia',
    'Venezia':            'Venezia',
}

LIGUE1_ALIASES: dict[str, str] = {
    'Ajaccio':            'Ajaccio',
    'Angers':             'Angers',
    'Auxerre':            'Auxerre',
    'Bordeaux':           'Bordeaux',
    'Brest':              'Brest',
    'Caen':               'Caen',
    'Clermont':           'Clermont Foot',
    'Dijon':              'Dijon',
    'Guingamp':           'Guingamp',
    'Le Havre':           'Le Havre',
    'Lens':               'Lens',
    'Lille':              'Lille',
    'Lorient':            'Lorient',
    'Lyon':               'Olympique Lyonnais',
    'Marseille':          'Olympique de Marseille',
    'Metz':               'Metz',
    'Monaco':             'AS Monaco',
    'Montpellier':        'Montpellier',
    'Nantes':             'Nantes',
    'Nice':               'Nice',
    'Nimes':              'Nimes',
    'Paris SG':           'Paris Saint-Germain',
    'PSG':                'Paris Saint-Germain',
    'Reims':              'Reims',
    'Rennes':             'Rennes',
    'Rodez':              'Rodez',
    'Moulins':            'Moulins',
    'Bordeaux':           'Bordeaux',
    'Strasbourg':         'Strasbourg',
    'Toulouse':           'Toulouse',
    'Troyes':             'Troyes',
}

MLS_ALIASES: dict[str, str] = {
    'Atlanta Utd':               'Atlanta United FC',
    'Atlanta United':            'Atlanta United FC',
    'Austin':                    'Austin FC',
    'Charlotte':                 'Charlotte FC',
    'Chicago':                   'Chicago Fire',
    'Cincinnati':                'FC Cincinnati',
    'Colorado':                  'Colorado Rapids',
    'Columbus':                  'Columbus Crew',
    'DC United':                 'D.C. United',
    'Dallas':                    'FC Dallas',
    'Houston':                   'Houston Dynamo',
    'Inter Miami':                'Inter Miami CF',
    'Kansas City':               'Sporting Kansas City',
    'LAFC':                      'Los Angeles FC',
    'LA Galaxy':                 'LA Galaxy',
    'Miami':                     'Inter Miami CF',
    'Minnesota':                 'Minnesota United',
    'Nashville':                 'Nashville SC',
    'New England':               'New England Revolution',
    'NY City':                   'New York City FC',
    'NY Red Bulls':              'New York Red Bulls',
    'Orlando City':              'Orlando City SC',
    'Philadelphia':              'Philadelphia Union',
    'Portland':                  'Portland Timbers',
    'Real Salt Lake':            'Real Salt Lake',
    'San Jose':                  'San Jose Earthquakes',
    'Seattle':                   'Seattle Sounders',
    'St. Louis City':            'St. Louis City SC',
    'Toronto':                   'Toronto FC',
    'Vancouver':                 'Vancouver Whitecaps',
}

ALEAGUE_ALIASES: dict[str, str] = {
    'Adelaide':           'Adelaide United',
    'Brisbane':           'Brisbane Roar',
    'Central Coast':      'Central Coast Mariners',
    'Melbourne City':     'Melbourne City',
    'Melbourne Victory':  'Melbourne Victory',
    'Newcastle Jets':     'Newcastle Jets',
    'Perth':              'Perth Glory',
    'Sydney FC':          'Sydney FC',
    'Wellington':         'Wellington Phoenix',
    'Western Sydney':     'Western Sydney Wanderers',
    'Western United':     'Western United',
    'Macarthur':          'Macarthur FC',
}

AFL_ALIASES: dict[str, str] = {
    'Adelaide Crows':        'Adelaide',
    'Adelaide':              'Adelaide',
    'Brisbane Lions':        'Brisbane Lions',
    'Brisbane':              'Brisbane Lions',
    'Carlton Blues':         'Carlton',
    'Carlton':               'Carlton',
    'Collingwood Magpies':   'Collingwood',
    'Collingwood':           'Collingwood',
    'Essendon Bombers':      'Essendon',
    'Essendon':              'Essendon',
    'Fremantle Dockers':     'Fremantle',
    'Fremantle':             'Fremantle',
    'Geelong Cats':          'Geelong',
    'Geelong':               'Geelong',
    'Gold Coast Suns':       'Gold Coast',
    'Gold Coast':            'Gold Coast',
    'GWS Giants':            'Greater Western Sydney',
    'Greater Western Sydney': 'Greater Western Sydney',
    'GWS':                   'Greater Western Sydney',
    'Hawthorn Hawks':        'Hawthorn',
    'Hawthorn':              'Hawthorn',
    'Melbourne Demons':      'Melbourne',
    'Melbourne':             'Melbourne',
    'North Melbourne Kangaroos': 'North Melbourne',
    'North Melbourne':       'North Melbourne',
    'Kangaroos':             'North Melbourne',
    'Port Adelaide Power':   'Port Adelaide',
    'Port Adelaide':         'Port Adelaide',
    'Richmond Tigers':       'Richmond',
    'Richmond':              'Richmond',
    'St Kilda Saints':       'St Kilda',
    'St Kilda':              'St Kilda',
    'Sydney Swans':          'Sydney',
    'Sydney':                'Sydney',
    'West Coast Eagles':     'West Coast',
    'West Coast':            'West Coast',
    'Western Bulldogs':      'Western Bulldogs',
    'Footscray':             'Western Bulldogs',
}

# ── Master alias map ──────────────────────────────────────────────────────────

_ALIAS_MAP: dict[str, dict[str, str]] = {
    'afl':        AFL_ALIASES,
    'epl':        EPL_ALIASES,
    'laliga':     LA_LIGA_ALIASES,
    'bundesliga': BUNDESLIGA_ALIASES,
    'seriea':     SERIE_A_ALIASES,
    'ligue1':     LIGUE1_ALIASES,
    'mls':        MLS_ALIASES,
    'aleague':    ALEAGUE_ALIASES,
    'ucl':        EPL_ALIASES,   # UCL uses same team names as European leagues
    'uel':        EPL_ALIASES,
}


def normalise(team: str, sport: str) -> str:
    """
    Return the canonical team name for a given sport.
    If no alias is found, returns the name as-is (stripped).
    """
    aliases = _ALIAS_MAP.get(sport, {})
    stripped = team.strip()
    return aliases.get(stripped, stripped)
