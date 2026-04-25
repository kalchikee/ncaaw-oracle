// NCAAW Oracle v4.1 — Logistic Regression Meta-Model + Platt Scaling
// Trained on 2018–2025 WBB data ONLY (~33,000 games)
// L2 regularization (C=1.0), walk-forward cross-validation
// Platt scaling especially important for WBB due to more lopsided games
// DO NOT use men's data — will bias predictions

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type { FeatureVector } from '../types.js';
import { toModelInputVector } from '../features/featureEngine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const WEIGHTS_PATH = resolve(__dirname, '../../model/model_weights.json');
const CALIBRATION_PATH = resolve(__dirname, '../../model/calibration_map.json');

interface ModelWeights {
  version: string;
  trainedOn: string;         // date trained
  numFeatures: number;
  bias: number;
  weights: number[];
  avgBrier: number;
  avgLogLoss: number;
  accuracy: number;
  trainDates: string;        // e.g. "2018–2025 WBB"
}

interface CalibrationMap {
  bins: number[];            // input probability bins
  calibrated: number[];      // calibrated probabilities per bin
  method: string;            // 'platt' | 'isotonic'
}

let _model: ModelWeights | null = null;
let _calibration: CalibrationMap | null = null;

// ─── Load model ───────────────────────────────────────────────────────────────

export function loadModel(): boolean {
  try {
    if (existsSync(WEIGHTS_PATH)) {
      _model = JSON.parse(readFileSync(WEIGHTS_PATH, 'utf8')) as ModelWeights;

      // ─── Sanity checks ─────────────────────────────────────────────────
      // weights[] is positional and `numFeatures` is the source of truth.
      // If the array length disagrees, the dot product would mis-pair
      // coefficients with features — refuse to use the model.
      const weightsLen = Array.isArray(_model.weights) ? _model.weights.length : -1;
      if (weightsLen !== _model.numFeatures) {
        logger.error(
          { numFeatures: _model.numFeatures, weights: weightsLen },
          'ML model load aborted: weights[] length disagrees with numFeatures — refusing to use the model.',
        );
        _model = null;
        return false;
      }

      // All-zero / NaN check — strong signal of a JSON shape mismatch.
      let nonZero = 0;
      let hasNaN = false;
      for (const v of _model.weights) {
        if (typeof v !== 'number' || Number.isNaN(v)) { hasNaN = true; break; }
        if (v !== 0) nonZero++;
      }
      if (hasNaN) {
        logger.error({ features: weightsLen }, 'ML model has NaN/non-numeric weights — refusing to use the model.');
        _model = null;
        return false;
      }
      if (nonZero === 0) {
        logger.error(
          { features: weightsLen },
          'ML model loaded but ALL weights are zero — JSON shape likely mismatched. Refusing to use the model.',
        );
        _model = null;
        return false;
      }

      logger.info(
        { version: _model.version, accuracy: _model.accuracy, features: weightsLen, nonZeroWeights: nonZero },
        'ML model loaded',
      );
    }
    if (existsSync(CALIBRATION_PATH)) {
      _calibration = JSON.parse(readFileSync(CALIBRATION_PATH, 'utf8')) as CalibrationMap;
      logger.info({ method: _calibration.method, bins: _calibration.bins.length }, 'Calibration map loaded');
    }
    return _model !== null;
  } catch (err) {
    logger.warn({ err }, 'Failed to load ML model — falling back to Monte Carlo');
    return false;
  }
}

export function isModelLoaded(): boolean { return _model !== null; }

export function getModelInfo(): ModelWeights | null { return _model; }

// ─── Sigmoid ──────────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── Logistic regression predict ─────────────────────────────────────────────

function logisticPredict(features: number[]): number {
  if (!_model) throw new Error('Model not loaded');
  const { bias, weights } = _model;
  let z = bias;
  for (let i = 0; i < Math.min(features.length, weights.length); i++) {
    z += features[i] * weights[i];
  }
  return sigmoid(z);
}

// ─── Platt scaling calibration ────────────────────────────────────────────────

function calibrate(rawProb: number): number {
  if (!_calibration) return rawProb;

  const { bins, calibrated } = _calibration;
  if (bins.length === 0) return rawProb;

  // Find the bin
  for (let i = 0; i < bins.length - 1; i++) {
    if (rawProb >= bins[i] && rawProb < bins[i + 1]) {
      // Linear interpolation within bin
      const t = (rawProb - bins[i]) / (bins[i + 1] - bins[i]);
      return calibrated[i] + t * (calibrated[i + 1] - calibrated[i]);
    }
  }

  // Edge cases
  if (rawProb < bins[0]) return calibrated[0];
  return calibrated[calibrated.length - 1];
}

// ─── Main predict ─────────────────────────────────────────────────────────────

export function predict(fv: FeatureVector, mcWinPct: number): number {
  if (!_model) {
    // No model — return MC probability as calibrated
    return mcWinPct;
  }

  try {
    const features = toModelInputVector(fv);
    const rawProb = logisticPredict(features);
    const calibratedProb = calibrate(rawProb);

    // Clip to reasonable range
    return Math.max(0.05, Math.min(0.95, calibratedProb));
  } catch (err) {
    logger.warn({ err }, 'ML prediction failed — falling back to MC');
    return mcWinPct;
  }
}

// ─── Simple fallback when model is not trained ────────────────────────────────
// Uses Elo win probability + AdjEM logistic blend

export function fallbackPredict(
  eloDiff: number,
  adjEmDiff: number,
  isHome: boolean
): number {
  const homeBonus = isHome ? 100 : 0; // Elo equivalent of HCA
  const eloProb = 1 / (1 + Math.pow(10, -(eloDiff + homeBonus) / 400));
  const adjEmProb = sigmoid(adjEmDiff / 12.0);
  const blended = 0.5 * eloProb + 0.5 * adjEmProb;
  return Math.max(0.05, Math.min(0.95, blended));
}
