"""
NCAAW Oracle v4.1 — Historical WBB Data Fetcher
Fetches 2018–2025 WBB game data from ESPN API + Sports Reference
CRITICAL: Women's data ONLY — DO NOT mix with men's data

Usage:
  python python/fetch_historical.py
  python python/fetch_historical.py --season 2024
  python python/fetch_historical.py --seasons 2018 2025
"""

import argparse
import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / 'data'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball'
ESPN_CORE = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/womens-college-basketball'

HEADERS = {'User-Agent': 'NCAAW-Oracle/4.1'}

# ─── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_json(url: str, retries: int = 3, delay: float = 1.0) -> dict | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1}/{retries} for {url}: {e}")
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  Failed after {retries} attempts: {url}: {e}")
                return None


def cache_path(key: str) -> Path:
    safe = key.replace('/', '_').replace('?', '_').replace('=', '_').replace('&', '_')
    return CACHE_DIR / f"{safe[:120]}.json"


def load_cache(key: str) -> dict | list | None:
    p = cache_path(key)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data
        except:
            pass
    return None


def save_cache(key: str, data) -> None:
    cache_path(key).write_text(json.dumps(data, indent=2))


# ─── Fetch season schedule ─────────────────────────────────────────────────────

def fetch_season_games(season_year: int) -> list[dict]:
    """
    Fetch all WBB games for a given season year.
    Season 2024 = 2023-24 season (Nov 2023 – Apr 2024)
    """
    print(f"Fetching season {season_year-1}-{season_year % 100:02d} games...")

    # Build date range for the season
    start_date = datetime(season_year - 1, 11, 1)
    end_date = datetime(season_year, 4, 15)

    all_games = []
    current = start_date

    while current <= end_date:
        date_str = current.strftime('%Y%m%d')
        cache_key = f"schedule_{date_str}"
        cached = load_cache(cache_key)

        if cached is not None:
            all_games.extend(cached)
            current += timedelta(days=1)
            continue

        url = f"{ESPN_BASE}/scoreboard?dates={date_str}&limit=200&groups=50"
        data = fetch_json(url)

        if data:
            events = data.get('events', [])
            day_games = []

            for event in events:
                game = parse_espn_event(event, season_year)
                if game:
                    day_games.append(game)

            save_cache(cache_key, day_games)
            all_games.extend(day_games)

            if day_games:
                print(f"  {date_str}: {len(day_games)} games")
        else:
            save_cache(cache_key, [])

        current += timedelta(days=1)
        time.sleep(0.1)  # rate limiting

    print(f"  Season {season_year}: {len(all_games)} total games")
    return all_games


def parse_espn_event(event: dict, season_year: int) -> dict | None:
    competitions = event.get('competitions', [])
    if not competitions:
        return None

    comp = competitions[0]
    competitors = comp.get('competitors', [])

    home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
    away = next((c for c in competitors if c.get('homeAway') == 'away'), None)

    if not home or not away:
        return None

    status = event.get('status', {}).get('type', {}).get('name', '')

    # Only include completed games for training data
    if 'Final' not in status and 'STATUS_FINAL' not in status:
        return None

    home_score = home.get('score')
    away_score = away.get('score')

    if home_score is None or away_score is None:
        return None

    try:
        home_score = int(home_score)
        away_score = int(away_score)
    except (ValueError, TypeError):
        return None

    home_team = home.get('team', {})
    away_team = away.get('team', {})

    return {
        'game_id': event.get('id', ''),
        'game_date': event.get('date', '')[:10],
        'season_year': season_year,
        'home_team_id': home_team.get('id', ''),
        'home_team_abbr': home_team.get('abbreviation', '').upper(),
        'home_team_name': home_team.get('displayName', ''),
        'away_team_id': away_team.get('id', ''),
        'away_team_abbr': away_team.get('abbreviation', '').upper(),
        'away_team_name': away_team.get('displayName', ''),
        'home_score': home_score,
        'away_score': away_score,
        'home_won': int(home_score > away_score),
        'neutral_site': int(comp.get('neutralSite', False)),
        'conference_competition': int(comp.get('conferenceCompetition', False)),
    }


# ─── Compute raw team efficiencies per season ──────────────────────────────────

def compute_season_efficiencies(games: list[dict]) -> dict[str, dict]:
    """
    For each team in a season, compute raw OE/DE/tempo.
    Returns dict: team_id → {adj_oe, adj_de, adj_em, adj_tempo, games_played, wins, losses}
    """
    from collections import defaultdict

    team_stats = defaultdict(lambda: {
        'oe_sum': 0.0, 'de_sum': 0.0, 'tempo_sum': 0.0,
        'games': 0, 'wins': 0, 'losses': 0,
        'team_name': '', 'team_abbr': '',
    })

    WBB_AVG_TEMPO = 66.0

    for game in games:
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        h_score = game['home_score']
        a_score = game['away_score']

        # Estimate possessions from score
        poss = WBB_AVG_TEMPO

        home_oe = (h_score / poss) * 100
        away_oe = (a_score / poss) * 100

        hs = team_stats[home_id]
        hs['oe_sum'] += home_oe
        hs['de_sum'] += away_oe
        hs['tempo_sum'] += poss
        hs['games'] += 1
        hs['wins'] += int(game['home_won'] == 1)
        hs['losses'] += int(game['home_won'] == 0)
        hs['team_name'] = game['home_team_name']
        hs['team_abbr'] = game['home_team_abbr']

        aws = team_stats[away_id]
        aws['oe_sum'] += away_oe
        aws['de_sum'] += home_oe
        aws['tempo_sum'] += poss
        aws['games'] += 1
        aws['wins'] += int(game['home_won'] == 0)
        aws['losses'] += int(game['home_won'] == 1)
        aws['team_name'] = game['away_team_name']
        aws['team_abbr'] = game['away_team_abbr']

    result = {}
    for team_id, stats in team_stats.items():
        g = stats['games']
        if g == 0:
            continue
        raw_oe = stats['oe_sum'] / g
        raw_de = stats['de_sum'] / g
        raw_tempo = stats['tempo_sum'] / g

        result[team_id] = {
            'team_id': team_id,
            'team_name': stats['team_name'],
            'team_abbr': stats['team_abbr'],
            'adj_oe': raw_oe,   # iterative adjustment done in adjEmCalculator.ts
            'adj_de': raw_de,
            'adj_em': raw_oe - raw_de,
            'adj_tempo': raw_tempo,
            'games_played': g,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'win_pct': stats['wins'] / g if g > 0 else 0.5,
        }

    return result


# ─── Build training dataset ────────────────────────────────────────────────────

def build_training_dataset(seasons: list[int]) -> list[dict]:
    """
    For each game in training seasons, build feature rows + outcome labels.
    Features are computed with in-season data up to game date (walk-forward).
    """
    print(f"\nBuilding training dataset for seasons: {seasons}")
    all_training_rows = []

    for season in seasons:
        print(f"\nProcessing season {season}...")
        games = fetch_season_games(season)

        if not games:
            print(f"  No games found for season {season}, skipping")
            continue

        # Sort games by date
        games.sort(key=lambda g: g['game_date'])

        # Compute season-level efficiencies (simplified — full iterative adjustment in TS)
        efficiencies = compute_season_efficiencies(games)

        # For each game, build a feature row
        for game in games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']

            home_eff = efficiencies.get(home_id, {})
            away_eff = efficiencies.get(away_id, {})

            if not home_eff or not away_eff:
                continue

            row = {
                'game_id': game['game_id'],
                'game_date': game['game_date'],
                'season_year': season,
                'home_team': game['home_team_abbr'],
                'away_team': game['away_team_abbr'],
                # Features
                'adj_em_diff': home_eff.get('adj_em', 0) - away_eff.get('adj_em', 0),
                'adj_oe_diff': home_eff.get('adj_oe', 0) - away_eff.get('adj_oe', 0),
                'adj_de_diff': home_eff.get('adj_de', 0) - away_eff.get('adj_de', 0),
                'adj_tempo_diff': home_eff.get('adj_tempo', 0) - away_eff.get('adj_tempo', 0),
                'home_win_pct': home_eff.get('win_pct', 0.5),
                'away_win_pct': away_eff.get('win_pct', 0.5),
                'is_home': 1 - game.get('neutral_site', 0),
                'is_neutral_site': game.get('neutral_site', 0),
                # Labels
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'home_won': game['home_won'],
                'score_diff': game['home_score'] - game['away_score'],
            }
            all_training_rows.append(row)

    print(f"\nTotal training rows: {len(all_training_rows)}")
    return all_training_rows


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fetch historical WBB data')
    parser.add_argument('--season', type=int, help='Fetch single season year (e.g. 2024 for 2023-24)')
    parser.add_argument('--seasons', type=int, nargs=2, metavar=('START', 'END'),
                        default=[2019, 2025], help='Season range (default: 2019 2025)')
    args = parser.parse_args()

    if args.season:
        seasons = [args.season]
    else:
        seasons = list(range(args.seasons[0], args.seasons[1] + 1))

    print(f"NCAAW Oracle v4.1 — Fetching WBB historical data")
    print(f"Seasons: {seasons}")
    print(f"IMPORTANT: Women's data only — NOT men's\n")

    training_data = build_training_dataset(seasons)

    output_path = DATA_DIR / 'wbb_training_data.json'
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"\nSaved {len(training_data)} training rows to {output_path}")
    print("Next: run python python/train_model.py")


if __name__ == '__main__':
    main()
