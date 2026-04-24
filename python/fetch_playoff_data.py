#!/usr/bin/env python3
"""
NCAAW Women's Tournament Data Fetcher — last 5 seasons, ESPN API seasontype=3.
Output: data/playoff_data.csv
"""
import sys, json, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

OUT_CSV   = DATA_DIR / "playoff_data.csv"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball"
HEADERS   = {"User-Agent": "NCAAW-Oracle/4.1"}
TOURNAMENT_YEARS = [2021, 2022, 2023, 2024, 2025]
K_FACTOR   = 20.0
LEAGUE_ELO = 1500.0


def espn_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"  Failed: {e}"); return {}


def fetch_tournament_games(year: int) -> list:
    cache = CACHE_DIR / f"ncaaw_tournament_{year}.json"
    if cache.exists(): return json.loads(cache.read_text())

    games = []
    start = datetime(year, 3, 16); end = datetime(year, 4, 10)
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        data = espn_get(f"{ESPN_BASE}/scoreboard?dates={date_str}&seasontype=3&limit=50")
        for ev in data.get("events", []):
            comps = ev.get("competitions", [{}])[0]
            if not ev.get("status", {}).get("type", {}).get("completed", False): continue
            home = next((c for c in comps.get("competitors",[]) if c.get("homeAway")=="home"), None)
            away = next((c for c in comps.get("competitors",[]) if c.get("homeAway")=="away"), None)
            if not home or not away: continue
            h_s = int(home.get("score",0) or 0); a_s = int(away.get("score",0) or 0)
            h_id = home.get("team",{}).get("abbreviation", home.get("team",{}).get("id",""))
            a_id = away.get("team",{}).get("abbreviation", away.get("team",{}).get("id",""))
            if not h_id or not a_id or (h_s==0 and a_s==0): continue
            games.append({"game_id": ev.get("id",""), "game_date": current.strftime("%Y-%m-%d"),
                          "home_team": str(h_id), "away_team": str(a_id),
                          "home_score": h_s, "away_score": a_s, "season": year})
        current += timedelta(days=1); time.sleep(0.3)
    cache.write_text(json.dumps(games, indent=2))
    return games


def main():
    print("NCAAW Women's Tournament Data Fetcher"); print("=" * 40)
    all_rows = []
    elo = defaultdict(lambda: LEAGUE_ELO)

    for year in TOURNAMENT_YEARS:
        print(f"\nYear {year}")
        games = fetch_tournament_games(year)
        print(f"  Fetched {len(games)} tournament games")

        for g in games:
            h, a = g["home_team"], g["away_team"]
            h_elo = elo[h]; a_elo = elo[a]
            label = 1 if g["home_score"] > g["away_score"] else 0
            all_rows.append({
                "season": year, "game_id": g["game_id"], "game_date": g["game_date"],
                "home_team": h, "away_team": a,
                "home_score": g["home_score"], "away_score": g["away_score"],
                "label": label, "home_win": label, "is_playoff": 1, "is_neutral": 1,
                "elo_diff": h_elo - a_elo,
            })
            exp = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
            elo[h] = h_elo + K_FACTOR * (label - exp)
            elo[a] = a_elo + K_FACTOR * ((1-label) - (1-exp))

    if not all_rows:
        print("\nNo data fetched."); return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} tournament games to {OUT_CSV}")


if __name__ == "__main__":
    main()
