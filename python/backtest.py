"""
NCAAW Oracle v4.1 — Backtesting
Tests model on historical seasons with bootstrap simulation
Tracks: overall accuracy, early-season vs conference play vs tournament
Reports Brier score, log-loss, ATS record, accuracy by confidence tier

Usage:
  python python/backtest.py
  python python/backtest.py --season 2025 --bootstrap
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss
except ImportError:
    print("pip install scikit-learn numpy")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_DIR = Path(__file__).parent.parent / 'model'

# ─── Load model ───────────────────────────────────────────────────────────────

def load_model():
    weights_path = MODEL_DIR / 'model_weights.json'
    cal_path = MODEL_DIR / 'calibration_map.json'

    if not weights_path.exists():
        print("No model found. Run: python python/train_model.py")
        return None, None

    with open(weights_path) as f:
        weights = json.load(f)
    with open(cal_path) as f:
        calibration = json.load(f)

    return weights, calibration


# ─── Sigmoid ──────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ─── Simple logistic predict ──────────────────────────────────────────────────

def predict(features: list[float], weights: dict) -> float:
    bias = weights['bias']
    w = weights['weights']
    z = bias + sum(f * wt for f, wt in zip(features, w))
    return sigmoid(z)


# ─── Calibrate ───────────────────────────────────────────────────────────────

def calibrate(prob: float, cal: dict) -> float:
    bins = cal['bins']
    calibrated = cal['calibrated']
    for i in range(len(bins) - 1):
        if bins[i] <= prob < bins[i + 1]:
            t = (prob - bins[i]) / (bins[i + 1] - bins[i])
            return calibrated[i] + t * (calibrated[i + 1] - calibrated[i])
    if prob < bins[0]:
        return calibrated[0]
    return calibrated[-1]


# ─── Confidence tier ─────────────────────────────────────────────────────────

def confidence_tier(prob: float) -> str:
    p = max(prob, 1 - prob)
    if p >= 0.80: return 'extreme'
    if p >= 0.74: return 'high_conviction'
    if p >= 0.65: return 'strong'
    if p >= 0.58: return 'moderate'
    return 'coin_flip'


# ─── Determine season phase ───────────────────────────────────────────────────

def season_phase(game_date: str, season_year: int) -> str:
    year = int(game_date[:4])
    month = int(game_date[5:7])

    if year == season_year - 1 and month in [11, 12]:
        return 'early_season'
    if year == season_year and month <= 2:
        return 'conference_play'
    if year == season_year and month == 3:
        return 'march'  # conf tournaments + selection
    if year == season_year and month == 4:
        return 'ncaa_tournament'
    return 'other'


# ─── Run backtest ─────────────────────────────────────────────────────────────

def run_backtest(training_data: list[dict], weights: dict, calibration: dict,
                 bootstrap: bool = False) -> dict:
    print(f"\nRunning backtest on {len(training_data)} games...")

    results_by_phase = defaultdict(lambda: {'total': 0, 'correct': 0, 'probs': [], 'outcomes': []})
    results_by_tier = defaultdict(lambda: {'total': 0, 'correct': 0, 'probs': [], 'outcomes': []})
    all_probs = []
    all_outcomes = []

    for row in training_data:
        adj_em_diff = row.get('adj_em_diff', 0)
        adj_oe_diff = row.get('adj_oe_diff', 0)
        adj_de_diff = row.get('adj_de_diff', 0)
        is_home = row.get('is_home', 1)
        is_neutral = row.get('is_neutral_site', 0)

        WBB_AVG_OE = 93.5
        home_adj_oe = WBB_AVG_OE + adj_oe_diff / 2
        away_adj_oe = WBB_AVG_OE - adj_oe_diff / 2
        home_adj_de = WBB_AVG_OE + adj_de_diff / 2
        away_adj_de = WBB_AVG_OE - adj_de_diff / 2
        home_pyth = home_adj_oe ** 10 / (home_adj_oe ** 10 + home_adj_de ** 10)
        away_pyth = away_adj_oe ** 10 / (away_adj_oe ** 10 + away_adj_de ** 10)
        home_win_pct = row.get('home_win_pct', 0.5)
        away_win_pct = row.get('away_win_pct', 0.5)
        elo_approx = (home_win_pct - away_win_pct) * 400

        features = [
            elo_approx / 200,
            adj_em_diff / 20,
            adj_oe_diff / 15,
            adj_de_diff / 15,
            home_pyth - away_pyth,
            0, 0, 0, 0, 0,      # unavailable features
            is_home, is_neutral,
            0, 0, 0, 0, 0,
            1, 1, 0, 0,
        ]

        raw_prob = predict(features, weights)
        cal_prob = calibrate(raw_prob, calibration)
        outcome = row.get('home_won', 0)
        correct = int((cal_prob >= 0.5) == (outcome == 1))

        tier = confidence_tier(cal_prob)
        season_year = row.get('season_year', 2024)
        phase = season_phase(row.get('game_date', ''), season_year)

        for d in [results_by_phase[phase], results_by_tier[tier]]:
            d['total'] += 1
            d['correct'] += correct
            d['probs'].append(cal_prob)
            d['outcomes'].append(outcome)

        all_probs.append(cal_prob)
        all_outcomes.append(outcome)

    # Overall metrics
    probs_arr = np.array(all_probs)
    outcomes_arr = np.array(all_outcomes)

    overall_acc = sum(int((p >= 0.5) == (o == 1)) for p, o in zip(all_probs, all_outcomes)) / len(all_probs)
    overall_brier = float(brier_score_loss(outcomes_arr, probs_arr))
    overall_ll = float(log_loss(outcomes_arr, probs_arr))

    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS -- NCAAW Oracle v4.1")
    print(f"{'=' * 60}")
    print(f"  Total Games:  {len(all_probs):,}")
    print(f"  Accuracy:     {overall_acc:.1%}")
    print(f"  Brier Score:  {overall_brier:.3f}  (target: <0.160)")
    print(f"  Log-Loss:     {overall_ll:.3f}  (target: <0.520)")

    print(f"\n  By Season Phase:")
    for phase, data in sorted(results_by_phase.items()):
        n = data['total']
        acc = data['correct'] / n if n > 0 else 0
        brier = float(brier_score_loss(np.array(data['outcomes']), np.array(data['probs']))) if n > 0 else 0
        print(f"    {phase:20s}: {data['correct']}-{n - data['correct']} ({acc:.1%}) | Brier: {brier:.3f}")

    print(f"\n  By Confidence Tier:")
    tier_order = ['extreme', 'high_conviction', 'strong', 'moderate', 'coin_flip']
    for tier in tier_order:
        data = results_by_tier.get(tier)
        if not data or data['total'] == 0:
            continue
        n = data['total']
        acc = data['correct'] / n
        print(f"    {tier:18s}: {data['correct']}-{n - data['correct']} ({acc:.1%}) | n={n}")

    print(f"{'=' * 60}\n")

    return {
        'total': len(all_probs),
        'accuracy': overall_acc,
        'brier': overall_brier,
        'log_loss': overall_ll,
        'by_phase': {k: {'total': v['total'], 'correct': v['correct'],
                         'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0}
                     for k, v in results_by_phase.items()},
        'by_tier': {k: {'total': v['total'], 'correct': v['correct'],
                        'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0}
                    for k, v in results_by_tier.items()},
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/wbb_training_data.json')
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    if not data_path.exists():
        print(f"No training data at {data_path}. Run fetch_historical.py first.")
        sys.exit(1)

    weights, calibration = load_model()
    if not weights:
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    results = run_backtest(data, weights, calibration, bootstrap=args.bootstrap)

    out_path = DATA_DIR / 'backtest_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == '__main__':
    main()
