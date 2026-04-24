// NCAAW Oracle v4.1 — Market Edge Detection
// WBB betting lines are the LEAST efficient of any major sport.
// Larger and more frequent edges than men's CBB.
// Not all games have lines — only show edge when a line exists.

import type { Prediction, ConfidenceTier, EdgeTier } from '../types.js';
import { impliedHomeProbFromSpread, impliedProbFromML } from '../api/oddsClient.js';

// ─── Confidence tier classification ──────────────────────────────────────────

export function getConfidenceTier(calibratedProb: number): ConfidenceTier {
  const p = Math.max(calibratedProb, 1 - calibratedProb);
  if (p >= 0.80) return 'extreme';
  if (p >= 0.74) return 'high_conviction';
  if (p >= 0.65) return 'strong';
  if (p >= 0.58) return 'moderate';
  return 'coin_flip';
}

// Expected accuracy by tier
export const TIER_ACCURACY_TARGETS: Record<ConfidenceTier, string> = {
  coin_flip:        '~54%',
  moderate:         '~62%',
  strong:           '~70–74%',
  high_conviction:  '~78–82%',
  extreme:          '~82–85%',
};

// ─── Edge computation ─────────────────────────────────────────────────────────

export interface EdgeResult {
  edge: number;               // model_prob - vegas_implied_prob (home team)
  edgeTier: EdgeTier;
  vegasHomePct: number;
  modelHomePct: number;
  spreadUsed?: number;
}

export function computeEdge(
  modelHomePct: number,
  vegasSpread?: number,
  homeML?: number,
): EdgeResult | null {
  if (vegasSpread === undefined && homeML === undefined) return null;

  let vegasHomePct: number;

  if (homeML !== undefined) {
    vegasHomePct = impliedProbFromML(homeML);
  } else if (vegasSpread !== undefined) {
    vegasHomePct = impliedHomeProbFromSpread(vegasSpread);
  } else {
    return null;
  }

  const edge = modelHomePct - vegasHomePct;
  const absEdge = Math.abs(edge);

  let edgeTier: EdgeTier;
  if (absEdge >= 0.12) edgeTier = 'large';
  else if (absEdge >= 0.08) edgeTier = 'meaningful';
  else if (absEdge >= 0.04) edgeTier = 'small';
  else edgeTier = 'agreement';

  return {
    edge,
    edgeTier,
    vegasHomePct,
    modelHomePct,
    spreadUsed: vegasSpread,
  };
}

// ─── Format edge for Discord ──────────────────────────────────────────────────

export function formatEdge(edge: number): string {
  const pct = (Math.abs(edge) * 100).toFixed(1);
  const dir = edge > 0 ? '▲' : '▼';
  return `${dir}${pct}%`;
}

export function getEdgeEmoji(tier: EdgeTier): string {
  switch (tier) {
    case 'large':      return '🚀';
    case 'meaningful': return '💡';
    case 'small':      return '📌';
    default:           return '';
  }
}

// ─── Should show in embed ─────────────────────────────────────────────────────

export function shouldShowPrediction(tier: ConfidenceTier): boolean {
  // Show all tiers except coin_flip in embeds
  return tier !== 'coin_flip';
}

export function isHighConviction(calibratedProb: number): boolean {
  const p = Math.max(calibratedProb, 1 - calibratedProb);
  return p >= 0.74;
}

export function isExtremeConviction(calibratedProb: number): boolean {
  const p = Math.max(calibratedProb, 1 - calibratedProb);
  return p >= 0.80;
}

// ─── Bet recommendation ───────────────────────────────────────────────────────

export function getBetRecommendation(pred: Prediction): string | null {
  const tier = pred.confidence_tier;
  const edgeTier = pred.edge_tier;

  if (tier === 'extreme') return 'STRONG BET';
  if (tier === 'high_conviction' && (edgeTier === 'meaningful' || edgeTier === 'large')) return 'BET';
  if (tier === 'high_conviction') return 'LEAN';
  if (tier === 'strong' && edgeTier === 'large') return 'LEAN';
  return null;
}
