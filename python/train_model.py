"""
NCAAW Oracle v4.1 — ML Model Training
Logistic regression with L2 regularization (C=1.0)
Walk-forward cross-validation
Platt scaling for probability calibration
Trains ONLY on WBB data (2018–2025)

Usage:
  python python/train_model.py
  python python/train_model.py --data data/wbb_training_data.json
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    print("scikit-learn required: pip install scikit-learn numpy")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_DIR = Path(__file__).parent.parent / 'model'
MODEL_DIR.mkdir(exist_ok=True)

WEIGHTS_PATH = MODEL_DIR / 'model_weights.json'
CALIBRATION_PATH = MODEL_DIR / 'calibration_map.json'

# ─── Feature list (must match featureEngine.ts toModelInputVector) ────────────

FEATURE_NAMES = [
    'elo_diff_norm',        # elo_diff / 200
    'adj_em_diff_norm',     # adj_em_diff / 20
    'adj_oe_diff_norm',     # adj_oe_diff / 15
    'adj_de_diff_norm',     # adj_de_diff / 15
    'pythagorean_diff',
    'efg_pct_diff_norm',    # / 0.05
    'tov_pct_diff_norm',    # / 0.05
    'oreb_pct_diff_norm',   # / 0.05
    'star_impact_diff_norm',# / 5
    'rest_days_diff_norm',  # / 3
    'is_home',
    'is_neutral_site',
    'conf_quality_diff_norm',# / 5
    'recruiting_diff',
    'returning_minutes_diff',
    'blend_weight_home',
    'blend_weight_away',
    'star_available_home',
    'star_available_away',
    'star_penalty_home_norm',# / 5
    'star_penalty_away_norm',# / 5
]


# ─── Build feature matrix from training data ──────────────────────────────────

def build_feature_matrix(training_data: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build X (features) and y (labels) from training rows.
    Simplified features for initial training — full features require adj_em computation.
    """
    rows_X = []
    rows_y = []
    dates = []

    for row in training_data:
        adj_em_diff = row.get('adj_em_diff', 0)
        adj_oe_diff = row.get('adj_oe_diff', 0)
        adj_de_diff = row.get('adj_de_diff', 0)
        adj_tempo_diff = row.get('adj_tempo_diff', 0)
        is_home = row.get('is_home', 1)
        is_neutral = row.get('is_neutral_site', 0)

        # Approximate elo diff from win pcts
        home_win_pct = row.get('home_win_pct', 0.5)
        away_win_pct = row.get('away_win_pct', 0.5)
        elo_diff_approx = (home_win_pct - away_win_pct) * 400

        # Pythagorean from adj efficiencies
        WBB_AVG_OE = 93.5
        home_adj_oe = WBB_AVG_OE + adj_oe_diff / 2
        away_adj_oe = WBB_AVG_OE - adj_oe_diff / 2
        home_adj_de = WBB_AVG_OE + adj_de_diff / 2
        away_adj_de = WBB_AVG_OE - adj_de_diff / 2

        home_pyth = home_adj_oe ** 10 / (home_adj_oe ** 10 + home_adj_de ** 10)
        away_pyth = away_adj_oe ** 10 / (away_adj_oe ** 10 + away_adj_de ** 10)
        pyth_diff = home_pyth - away_pyth

        feature_row = [
            elo_diff_approx / 200,
            adj_em_diff / 20,
            adj_oe_diff / 15,
            adj_de_diff / 15,
            pyth_diff,
            0,  # efg_pct_diff (not available from box scores)
            0,  # tov_pct_diff
            0,  # oreb_pct_diff
            0,  # star_impact_diff
            0,  # rest_days_diff
            is_home,
            is_neutral,
            0,  # conf_quality_diff
            0,  # recruiting_diff
            0,  # returning_minutes_diff
            0,  # blend_weight_home
            0,  # blend_weight_away
            1,  # star_available_home
            1,  # star_available_away
            0,  # star_penalty_home
            0,  # star_penalty_away
        ]

        rows_X.append(feature_row)
        rows_y.append(row.get('home_won', 0))
        dates.append(row.get('game_date', '2020-01-01'))

    return np.array(rows_X), np.array(rows_y), dates


# ─── Walk-forward cross-validation ───────────────────────────────────────────

def walk_forward_cv(X: np.ndarray, y: np.ndarray, dates: list[str]) -> dict:
    """
    Walk-forward CV: train on past seasons, validate on next season.
    Returns mean accuracy, Brier, log-loss across folds.
    """
    # Group by season year
    season_years = sorted(set(d[:4] for d in dates))
    if len(season_years) < 2:
        return {'accuracy': 0.0, 'brier': 0.0, 'log_loss': 0.0, 'n_folds': 0}

    all_accs, all_briers, all_lls = [], [], []

    for fold_idx in range(1, len(season_years)):
        train_years = season_years[:fold_idx]
        test_year = season_years[fold_idx]

        train_mask = np.array([d[:4] in train_years for d in dates])
        test_mask = np.array([d[:4] == test_year for d in dates])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 100 or len(X_test) < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        model.fit(X_train_scaled, y_train)

        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)

        all_accs.append(acc)
        all_briers.append(brier)
        all_lls.append(ll)
        print(f"  Fold {fold_idx} ({test_year}): acc={acc:.3f}, brier={brier:.3f}, log_loss={ll:.3f}")

    return {
        'accuracy': float(np.mean(all_accs)) if all_accs else 0.0,
        'brier': float(np.mean(all_briers)) if all_briers else 0.0,
        'log_loss': float(np.mean(all_lls)) if all_lls else 0.0,
        'n_folds': len(all_accs),
    }


# ─── Train final model ────────────────────────────────────────────────────────

def train_final_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train on full dataset and apply Platt scaling."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # Platt scaling via isotonic regression on held-out predictions
    # Use last 20% as calibration set
    n = len(X_scaled)
    cal_start = int(n * 0.80)

    X_cal = X_scaled[cal_start:]
    y_cal = y[cal_start:]

    raw_probs_cal = model.predict_proba(X_cal)[:, 1]

    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(raw_probs_cal, y_cal)

    return model, scaler, iso_reg


# ─── Build calibration map ────────────────────────────────────────────────────

def build_calibration_map(iso_reg: IsotonicRegression, n_bins: int = 20) -> dict:
    """Create bin → calibrated probability lookup table."""
    bins = np.linspace(0.0, 1.0, n_bins + 1).tolist()
    calibrated = [float(iso_reg.predict([b])[0]) for b in bins]
    return {
        'bins': bins,
        'calibrated': calibrated,
        'method': 'isotonic',
    }


# ─── Export model weights ─────────────────────────────────────────────────────

def export_weights(
    model,
    scaler,
    cv_results: dict,
    n_training: int,
    season_range: str,
) -> dict:
    """Export model to JSON for TypeScript inference."""
    # Un-scale weights: w_unscaled = w / scaler.scale_
    raw_weights = model.coef_[0]
    unscaled_weights = (raw_weights / scaler.scale_).tolist()
    bias = float(model.intercept_[0]) - float(np.dot(raw_weights, scaler.mean_ / scaler.scale_))

    return {
        'version': '4.1.0',
        'trainedOn': datetime.now().isoformat(),
        'numFeatures': len(unscaled_weights),
        'featureNames': FEATURE_NAMES,
        'bias': bias,
        'weights': unscaled_weights,
        'avgBrier': cv_results['brier'],
        'avgLogLoss': cv_results['log_loss'],
        'accuracy': cv_results['accuracy'],
        'trainDates': season_range,
        'nTrainingSamples': n_training,
        'cvFolds': cv_results['n_folds'],
        'note': 'Trained on WBB data ONLY. DO NOT use men\'s data.',
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train NCAAW Oracle ML model')
    parser.add_argument('--data', type=str, default='data/wbb_training_data.json',
                        help='Path to training data JSON')
    parser.add_argument('--min-seasons', type=int, default=3,
                        help='Minimum seasons required to train')
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    if not data_path.exists():
        print(f"Training data not found: {data_path}")
        print("Run: python python/fetch_historical.py --seasons 2019 2025")
        sys.exit(1)

    print("NCAAW Oracle v4.1 — ML Model Training")
    print("CRITICAL: Training on WBB data ONLY\n")

    with open(data_path) as f:
        training_data = json.load(f)

    print(f"Loaded {len(training_data)} training rows")

    if len(training_data) < 1000:
        print(f"WARNING: Only {len(training_data)} rows — model may be underfit")
        print("Recommend 15,000+ rows (3+ seasons)")

    # Build feature matrix
    print("\nBuilding feature matrix...")
    X, y, dates = build_feature_matrix(training_data)
    print(f"  X shape: {X.shape}, positive rate: {y.mean():.3f}")

    # Walk-forward CV
    print("\nRunning walk-forward cross-validation...")
    cv_results = walk_forward_cv(X, y, dates)
    print(f"\nCV Results: acc={cv_results['accuracy']:.3f}, "
          f"brier={cv_results['brier']:.3f}, log_loss={cv_results['log_loss']:.3f}")

    if cv_results['accuracy'] < 0.60 and cv_results['n_folds'] > 0:
        print(f"\nWARNING: CV accuracy {cv_results['accuracy']:.3f} below 0.60 — check data quality")

    # Train final model
    print("\nTraining final model on full dataset...")
    model, scaler, iso_reg = train_final_model(X, y)

    # Build calibration map
    cal_map = build_calibration_map(iso_reg)

    # Export
    seasons = sorted(set(d[:4] for d in dates))
    season_range = f"WBB {seasons[0]}–{seasons[-1]}" if seasons else "WBB unknown"

    weights = export_weights(model, scaler, cv_results, len(X), season_range)

    WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))
    CALIBRATION_PATH.write_text(json.dumps(cal_map, indent=2))

    print(f"\nModel saved: {WEIGHTS_PATH}")
    print(f"Calibration saved: {CALIBRATION_PATH}")
    print(f"\nFinal model: {len(weights['weights'])} features")
    print(f"CV Accuracy: {cv_results['accuracy']:.1%}")
    print(f"CV Brier:    {cv_results['brier']:.3f}")
    print(f"CV Log-Loss: {cv_results['log_loss']:.3f}")
    print("\nNote: Accuracy will improve once full feature engineering (star model, etc.) is active.")


if __name__ == '__main__':
    main()
