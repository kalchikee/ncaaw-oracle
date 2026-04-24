#!/usr/bin/env python3
"""
NCAAW Oracle v4.1 — Live Predictions
Fetches today's/upcoming women's college basketball games from ESPN,
computes features, runs the trained logistic regression model, and prints predictions.

Usage:
  python python/predict.py            # today's games
  python python/predict.py --date 20260412
"""

import argparse
import json
import math
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR  = ROOT / "data"
HIST_DATA = DATA_DIR / "wbb_training_data.json"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball"
HEADERS   = {"User-Agent": "NCAAW-Oracle/4.1"}

WBB_AVG_OE = 93.5   # league avg offensive efficiency (points per 100 possessions)


# ── ESPN helpers ───────────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return None


def fetch_games(date_str: str) -> list[dict]:
    url = f"{ESPN_BASE}/scoreboard?dates={date_str}&limit=200"
    data = fetch_json(url)
    if not data:
        return []
    games = []
    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        comp   = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        games.append({
            "status":    status,
            "neutral":   int(comp.get("neutralSite", False)),
            "home_id":   home.get("team", {}).get("id", ""),
            "home_abbr": home.get("team", {}).get("abbreviation", "").upper(),
            "home_name": home.get("team", {}).get("displayName", ""),
            "away_id":   away.get("team", {}).get("id", ""),
            "away_abbr": away.get("team", {}).get("abbreviation", "").upper(),
            "away_name": away.get("team", {}).get("displayName", ""),
        })
    return games


def fetch_team_stats(team_id: str) -> dict:
    url = f"{ESPN_BASE}/teams/{team_id}?enable=record,stats"
    data = fetch_json(url)
    if not data:
        return {}
    items = data.get("team", {}).get("record", {}).get("items", [])
    total = next((i for i in items if i.get("type") == "total"), {})
    return {s["name"]: s["value"] for s in total.get("stats", [])}


# ── Model loading ──────────────────────────────────────────────────────────────

def is_tournament_season(date_str: str) -> bool:
    """Women's Tournament runs mid-March through early April."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return False
    return (d.month == 3 and d.day >= 16) or (d.month == 4 and d.day <= 10)


def load_playoff_model() -> dict | None:
    """Load playoff-specific model (different format from regular model)."""
    try:
        po = MODEL_DIR / "playoff_coefficients.json"
        ps = MODEL_DIR / "playoff_scaler.json"
        pm = MODEL_DIR / "playoff_metadata.json"
        if not (po.exists() and ps.exists() and pm.exists()):
            return None
        return {
            "coeff":  json.loads(po.read_text()),
            "scaler": json.loads(ps.read_text()),
            "meta":   json.loads(pm.read_text()),
        }
    except Exception as e:
        print(f"  Playoff model load failed: {e}")
        return None


def predict_proba_playoff(playoff_model: dict, fv: dict) -> float:
    """Sigmoid prediction using playoff model's list-format coefficients."""
    features  = playoff_model["meta"]["feature_names"]
    coeff_arr = playoff_model["coeff"]["coefficients"]
    intercept = playoff_model["coeff"]["intercept"]
    mean      = playoff_model["scaler"]["mean"]
    scale     = playoff_model["scaler"]["scale"]

    x = [(fv.get(f, 0.0) - mean[i]) / (scale[i] if scale[i] != 0 else 1.0)
         for i, f in enumerate(features)]
    logit = sum(c * xi for c, xi in zip(coeff_arr, x)) + intercept
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, logit))))


def load_model() -> dict | None:
    try:
        weights = json.loads((MODEL_DIR / "model_weights.json").read_text())
        calib   = json.loads((MODEL_DIR / "calibration_map.json").read_text())
        return {"weights": weights, "calib": calib}
    except Exception as e:
        print(f"  Model load failed: {e}")
        return None


def predict_proba(model: dict, fv: list[float]) -> float:
    """Dot product + sigmoid + isotonic calibration (un-scaled weights)."""
    w    = model["weights"]["weights"]
    bias = model["weights"]["bias"]
    logit = sum(w[i] * fv[i] for i in range(len(fv))) + bias
    raw   = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, logit))))

    bins = model["calib"]["bins"]
    cals = model["calib"]["calibrated"]
    if raw <= bins[0]:
        return cals[0]
    if raw >= bins[-1]:
        return cals[-1]
    for i in range(len(bins) - 1):
        if bins[i] <= raw <= bins[i + 1]:
            t = (raw - bins[i]) / (bins[i + 1] - bins[i])
            return cals[i] + t * (cals[i + 1] - cals[i])
    return raw


# ── Feature builder ────────────────────────────────────────────────────────────

def build_feature_vector(game: dict, h_stats: dict, a_stats: dict) -> list[float]:
    """
    Build 21-element feature vector matching NCAAW model's FEATURE_NAMES.
    Manual normalization matches train_model.py export_weights logic.
    """
    is_home    = 1.0 - float(game["neutral"])
    is_neutral = float(game["neutral"])

    def eff(stats: dict) -> dict:
        gp   = max(1, stats.get("gamesPlayed", 1))
        wins = stats.get("wins", gp / 2)
        wp   = wins / gp
        pts  = stats.get("avgPointsFor",  70.0)
        opp  = stats.get("avgPointsAgainst", 70.0)
        # scale to per-100-possessions using WBB tempo ~67
        OE = (pts / 67.0) * 100
        DE = (opp / 67.0) * 100
        EM = OE - DE
        return {"oe": OE, "de": DE, "em": EM, "wp": wp}

    h = eff(h_stats)
    a = eff(a_stats)

    adj_em_diff  = h["em"]  - a["em"]
    adj_oe_diff  = h["oe"]  - a["oe"]
    adj_de_diff  = h["de"]  - a["de"]

    # Pythagorean from efficiencies (exponent 10 for CBB)
    def pyth(oe: float, de: float) -> float:
        return oe ** 10 / (oe ** 10 + de ** 10 + 1e-9)
    pyth_diff = pyth(h["oe"], h["de"]) - pyth(a["oe"], a["de"])

    # Approximate Elo diff from win percentages
    elo_diff_approx = (h["wp"] - a["wp"]) * 400

    # Build 21-element vector (same order as FEATURE_NAMES)
    return [
        elo_diff_approx / 200,  # elo_diff_norm
        adj_em_diff / 20,       # adj_em_diff_norm
        adj_oe_diff / 15,       # adj_oe_diff_norm
        adj_de_diff / 15,       # adj_de_diff_norm
        pyth_diff,              # pythagorean_diff
        0.0,                    # efg_pct_diff_norm
        0.0,                    # tov_pct_diff_norm
        0.0,                    # oreb_pct_diff_norm
        0.0,                    # star_impact_diff_norm
        0.0,                    # rest_days_diff_norm
        is_home,                # is_home
        is_neutral,             # is_neutral_site
        0.0,                    # conf_quality_diff_norm
        0.0,                    # recruiting_diff
        0.0,                    # returning_minutes_diff
        0.0,                    # blend_weight_home
        0.0,                    # blend_weight_away
        1.0,                    # star_available_home
        1.0,                    # star_available_away
        0.0,                    # star_penalty_home_norm
        0.0,                    # star_penalty_away_norm
    ]


# ── Printing ───────────────────────────────────────────────────────────────────

def pad(s: str, w: int) -> str:
    return s[:w].ljust(w)


def print_predictions(results: list, date_str: str) -> None:
    width = 90
    print("\n" + "=" * width)
    print(f"  NCAAW ORACLE v4.1  |  {date_str}  |  {len(results)} games")
    print("=" * width)
    print("  " + pad("MATCHUP", 30) + pad("HOME WIN%", 11) + pad("AWAY WIN%", 11) + "PICK")
    print("-" * width)
    for r in sorted(results, key=lambda x: -max(x["home_prob"], x["away_prob"])):
        matchup  = f"{r['home_abbr']} vs {r['away_abbr']}"
        home_pct = f"{r['home_prob']*100:.1f}%"
        away_pct = f"{r['away_prob']*100:.1f}%"
        pick     = r["home_abbr"] if r["home_prob"] >= r["away_prob"] else r["away_abbr"]
        star     = " *" if max(r["home_prob"], r["away_prob"]) >= 0.70 else ""
        neutral  = " [N]" if r["neutral"] else ""
        print(f"  {pad(matchup + neutral, 30)}{pad(home_pct, 11)}{pad(away_pct, 11)}{pick}{star}")
    print("-" * width)
    print("* = high confidence (>= 70%)  |  [N] = neutral site\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    date_str = args.date

    print(f"=== NCAAW Oracle v4.1 — Predictions for {date_str} ===\n")

    model = load_model()
    if not model:
        print("ERROR: Could not load model. Run: python python/train_model.py")
        return

    # Load playoff model during tournament season
    playoff_model = None
    if is_tournament_season(date_str):
        playoff_model = load_playoff_model()
        if playoff_model:
            print("  [TOURNAMENT MODE] Using Women's Tournament playoff model")

    print(f"Fetching games for {date_str}...")
    games = fetch_games(date_str)

    if not games:
        for offset in list(range(1, 8)) + list(range(-1, -8, -1)):
            d = (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=offset)).strftime("%Y%m%d")
            games = fetch_games(d)
            if games:
                date_str = d
                label = "next" if offset > 0 else "most recent"
                print(f"  No games today — showing {label} games ({d})")
                break

    if not games:
        # Fall back to ESPN's current window (no date filter)
        data = fetch_json(f"{ESPN_BASE}/scoreboard?limit=200")
        if data:
            games = []
            for event in data.get("events", []):
                comp = (event.get("competitions") or [{}])[0]
                competitors = comp.get("competitors", [])
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                games.append({
                    "status":    event.get("status", {}).get("type", {}).get("name", ""),
                    "neutral":   int(comp.get("neutralSite", False)),
                    "home_id":   home.get("team", {}).get("id", ""),
                    "home_abbr": home.get("team", {}).get("abbreviation", "").upper(),
                    "home_name": home.get("team", {}).get("displayName", ""),
                    "away_id":   away.get("team", {}).get("id", ""),
                    "away_abbr": away.get("team", {}).get("abbreviation", "").upper(),
                    "away_name": away.get("team", {}).get("displayName", ""),
                })
            if games:
                date_str = "most recent window"
                print(f"  Using ESPN's current window")

    if not games:
        print("No games found. Season may be over.")
        return

    scheduled = [g for g in games if "SCHEDULED" in g["status"]] or games
    print(f"  Found {len(scheduled)} game(s)\n")

    results = []
    for game in scheduled:
        h_stats = fetch_team_stats(game["home_id"])
        a_stats = fetch_team_stats(game["away_id"])
        time.sleep(0.1)

        if playoff_model:
            # Playoff model uses elo_diff (approx from win pcts) + is_neutral
            h_gp   = max(1, h_stats.get("gamesPlayed", 1))
            a_gp   = max(1, a_stats.get("gamesPlayed", 1))
            h_wp   = h_stats.get("wins", h_gp / 2) / h_gp
            a_wp   = a_stats.get("wins", a_gp / 2) / a_gp
            po_fv  = {"elo_diff": (h_wp - a_wp) * 400, "is_neutral": float(game["neutral"])}
            home_p = predict_proba_playoff(playoff_model, po_fv)
        else:
            fv     = build_feature_vector(game, h_stats, a_stats)
            home_p = predict_proba(model, fv)
        away_p  = 1.0 - home_p

        results.append({
            "home_abbr": game["home_abbr"],
            "away_abbr": game["away_abbr"],
            "home_name": game["home_name"],
            "away_name": game["away_name"],
            "home_prob": home_p,
            "away_prob": away_p,
            "neutral":   game["neutral"],
        })

    print_predictions(results, date_str)


if __name__ == "__main__":
    main()
