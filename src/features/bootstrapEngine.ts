// NCAAW Oracle v4.1 — Prior-Year Bootstrap Engine
// WBB has higher roster continuity than men's (fewer early WNBA departures)
// so returning_minutes is weighted HIGHER (0.30 vs 0.25 for men's)
// Formula: Preseason_Rating = 0.35×prior_AdjEM + 0.30×returning_minutes_adj +
//           0.20×recruiting_composite + 0.15×portal_impact

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import { getBootstrapBlendWeights } from '../season-manager/seasonManager.js';
import type { PriorYearData, WBBTeam } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PRIOR_YEAR_PATH = resolve(__dirname, '../../config/prior_year.json');
const PORTAL_PATH = resolve(__dirname, '../../config/portal_impact.json');

let _priorYearData: Map<string, PriorYearData> = new Map();
let _portalData: Map<string, number> = new Map();  // teamId → portal WAR

// ─── Load configs ─────────────────────────────────────────────────────────────

export function loadBootstrapData(): void {
  // Load prior year data
  if (existsSync(PRIOR_YEAR_PATH)) {
    try {
      const data = JSON.parse(readFileSync(PRIOR_YEAR_PATH, 'utf8')) as PriorYearData[];
      _priorYearData = new Map(data.map(d => [d.teamId, d]));
      logger.info({ teams: _priorYearData.size }, 'Prior-year bootstrap data loaded');
    } catch (err) {
      logger.warn({ err }, 'Failed to load prior_year.json — bootstrap unavailable');
    }
  } else {
    logger.warn('prior_year.json not found — early-season predictions will use defaults');
  }

  // Load portal impact
  if (existsSync(PORTAL_PATH)) {
    try {
      const data = JSON.parse(readFileSync(PORTAL_PATH, 'utf8')) as Record<string, number>;
      _portalData = new Map(Object.entries(data));
    } catch {
      // silently ignore
    }
  }
}

export function getPriorYearData(teamId: string): PriorYearData | null {
  return _priorYearData.get(teamId) ?? null;
}

// ─── Compute preseason rating ─────────────────────────────────────────────────
// Preseason_Rating = 0.35 × prior_AdjEM + 0.30 × returning_minutes_adj +
//                   0.20 × recruiting_composite + 0.15 × portal_impact
// Higher weight on returning minutes (0.30 vs 0.25 men's) — WBB roster continuity
// Lower weight on recruiting (0.20 vs 0.25 men's) — women's rankings less predictive

export function computePreseasonRating(prior: PriorYearData): number {
  const portalWAR = _portalData.get(prior.teamId) ?? prior.portalImpact ?? 0;

  // Normalize each component to a -15 to +15 range (AdjEM-like scale)
  const priorComponent = prior.adjEM;                              // already on right scale
  const returningAdj = (prior.returningMinutesPct - 0.65) * 30;  // 65% returning = neutral
  const recruitingAdj = (prior.recruitingComposite - 0.5) * 20;  // 0.5 = average
  const portalAdj = portalWAR;                                     // already in AdjEM units

  const preseasonRating =
    0.35 * priorComponent +
    0.30 * returningAdj +
    0.20 * recruitingAdj +
    0.15 * portalAdj;

  return preseasonRating;
}

// ─── Blend current-season with prior-year ─────────────────────────────────────

export interface BlendedEfficiency {
  adjEM: number;
  adjOE: number;
  adjDE: number;
  adjTempo: number;
  isBootstrap: boolean;
  priorWeight: number;
}

export function blendEfficiency(
  team: WBBTeam,
  prior: PriorYearData | null
): BlendedEfficiency {
  const gamesPlayed = team.gamesPlayed;
  const [priorWeight, inSeasonWeight] = getBootstrapBlendWeights(gamesPlayed);

  if (!prior || priorWeight === 0) {
    return {
      adjEM: team.adjEM,
      adjOE: team.adjOE,
      adjDE: team.adjDE,
      adjTempo: team.adjTempo,
      isBootstrap: false,
      priorWeight: 0,
    };
  }

  const blendedEM = priorWeight * prior.adjEM + inSeasonWeight * team.adjEM;
  const blendedOE = priorWeight * prior.adjOE + inSeasonWeight * team.adjOE;
  const blendedDE = priorWeight * prior.adjDE + inSeasonWeight * team.adjDE;
  const blendedTempo = priorWeight * prior.adjTempo + inSeasonWeight * team.adjTempo;

  return {
    adjEM: blendedEM,
    adjOE: blendedOE,
    adjDE: blendedDE,
    adjTempo: blendedTempo,
    isBootstrap: priorWeight > 0.10,
    priorWeight,
  };
}

// ─── Star player dominance model ──────────────────────────────────────────────
// WBB: star players have LARGER impact than men's due to:
// - Greater talent gaps
// - Higher individual usage rates
// - Fewer possessions per game amplifies individual impact
// A dominant player can shift win probability by 10–15%

export interface StarPlayerEffect {
  starBPM: number;
  starUsage: number;
  starAvailable: boolean;
  injuryPenalty: number;   // AdjEM penalty when star is out
  impact: number;          // combined star impact score
}

export function computeStarPlayerEffect(
  teamId: string,
  players: Array<{ bpm: number; usageRate: number; injured?: boolean; injuryStatus?: string }>,
  priorStarBPM?: number
): StarPlayerEffect {
  if (players.length === 0) {
    return {
      starBPM: priorStarBPM ?? 0,
      starUsage: 0.20,
      starAvailable: true,
      injuryPenalty: 0,
      impact: priorStarBPM ?? 0,
    };
  }

  // Find top player by BPM
  const sorted = [...players].sort((a, b) => b.bpm - a.bpm);
  const star = sorted[0];

  const starAvailable = !star.injured && star.injuryStatus !== 'Out';
  const injuryPenalty = starAvailable ? 0 :
    // Usage-weighted penalty: star BPM × usage / 0.20 (league avg usage)
    Math.abs(star.bpm) * (star.usageRate / 0.20) * 0.8;

  // Star impact = BPM × usage_premium (players with >25% usage in WBB are truly dominant)
  const usagePremium = Math.max(1.0, star.usageRate / 0.22);
  const impact = star.bpm * usagePremium * (starAvailable ? 1 : 0);

  return {
    starBPM: star.bpm,
    starUsage: star.usageRate,
    starAvailable,
    injuryPenalty,
    impact,
  };
}

// Convert star impact to AdjEM delta
// In WBB: 1 BPM point ≈ 1.5 AdjEM points for a star (due to higher usage concentration)
export function starImpactToAdjEM(impact: number): number {
  return impact * 1.5;
}

// ─── WNBA draft impact ────────────────────────────────────────────────────────
// Players leaving early for WNBA draft → subtract their WAR pre-season

export function estimateDraftLossImpact(
  players: Array<{ bpm: number; usageRate: number; wnnaDraftProspect?: boolean }>
): number {
  let impact = 0;
  for (const p of players) {
    if (p.wnnaDraftProspect) {
      // Expected loss = starImpact × probability of leaving (assume 0.7 for projected picks)
      impact += Math.abs(p.bpm) * p.usageRate * 0.7;
    }
  }
  return impact;
}

// ─── Home court advantage (WBB-specific) ─────────────────────────────────────
// Power conference programs with high attendance have larger HCA

const POWER_CONF_HCA_MAP: Record<string, number> = {
  // Programs with known large HCA
  'SC': 4.5,    // Colonial Life Arena sells out
  'IOWA': 4.5,  // Carver-Hawkeye
  'LSU': 4.2,
  'TENN': 4.0,
  'UCONN': 3.8,
  'UCLA': 3.5,
  'STAN': 3.5,
};

const DEFAULT_POWER_HCA = 3.5;    // power conference home
const DEFAULT_MID_MAJOR_HCA = 3.0; // mid-major home

export function getHomeCourtAdvantage(teamAbbr: string, conference: string): number {
  if (POWER_CONF_HCA_MAP[teamAbbr]) return POWER_CONF_HCA_MAP[teamAbbr];

  const powerConfs = ['SEC', 'ACC', 'Big 12', 'Big Ten', 'Pac-12', 'Big East'];
  if (powerConfs.some(c => conference.includes(c))) return DEFAULT_POWER_HCA;
  return DEFAULT_MID_MAJOR_HCA;
}
