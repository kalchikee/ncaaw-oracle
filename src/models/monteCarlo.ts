// NCAAW Oracle v4.1 — Monte Carlo Simulation Engine
// 10,000 Normal distribution simulations
// WBB-specific calibration: σ ~11 pts, avg score ~62 PPG, tempo 58–75 poss/game
// Lower 3PT% → less shooting variance than men's
// OT: ~7 additional points per team per OT period

import type { FeatureVector, MonteCarloResult } from '../types.js';

const N_SIMULATIONS = 10_000;

// WBB constants
const WBB_LEAGUE_AVG_OE = 93.5;     // pts per 100 possessions
const WBB_LEAGUE_AVG_DE = 93.5;
const WBB_LEAGUE_AVG_TEMPO = 66.0;  // possessions per 40 min (range 58–75)
const WBB_SCORE_STD = 11.0;         // σ per team (vs ~12.5 in NBA)
const WBB_OT_EXTRA_SCORE = 7.0;     // pts per team in OT

// Home court advantage: +3.0 to +4.5 pts for power programs
const DEFAULT_HCA_PTS = 3.5;

// ─── Normal random (Box-Muller) ───────────────────────────────────────────────

function normalRandom(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

// ─── Expected score estimator ─────────────────────────────────────────────────

interface ExpectedScores {
  homeExpPts: number;
  awayExpPts: number;
  homeStd: number;
  awayStd: number;
  expectedPoss: number;
}

export function estimateExpectedScores(fv: FeatureVector): ExpectedScores {
  // Decode team efficiencies from diff features
  // adj_oe_diff = home_adjOE - away_adjOE
  const homeAdjOE = WBB_LEAGUE_AVG_OE + fv.adj_oe_diff / 2;
  const awayAdjOE = WBB_LEAGUE_AVG_OE - fv.adj_oe_diff / 2;

  // adj_de_diff = home_adjDE - away_adjDE (lower DE is better defense)
  const homeAdjDE = WBB_LEAGUE_AVG_DE + fv.adj_de_diff / 2;
  const awayAdjDE = WBB_LEAGUE_AVG_DE - fv.adj_de_diff / 2;

  // Tempo: expected possessions = average of both teams' adjusted tempos
  const homeTempo = WBB_LEAGUE_AVG_TEMPO + fv.adj_tempo_diff / 2;
  const awayTempo = WBB_LEAGUE_AVG_TEMPO - fv.adj_tempo_diff / 2;
  const expectedPoss = Math.max(55, Math.min(78, (homeTempo + awayTempo) / 2));

  // Expected raw pts: off_eff × (opp_def / league_avg) × poss/100
  const homeRawPts = homeAdjOE * (awayAdjDE / WBB_LEAGUE_AVG_DE) * (expectedPoss / 100);
  const awayRawPts = awayAdjOE * (homeAdjDE / WBB_LEAGUE_AVG_DE) * (expectedPoss / 100);

  // Home court advantage
  const homeAdv = fv.is_neutral_site === 1 ? 0 : DEFAULT_HCA_PTS;

  // Rest adjustment: each day of extra rest ≈ +0.25 pts (smaller than NBA)
  const restBonus = Math.max(-2.5, Math.min(2.5, fv.rest_days_diff * 0.25));

  // Star player adjustment
  // When star is out, apply injury penalty to that team's expected score
  const homeStarPenalty = fv.star_available_home === 0 ? fv.star_injury_penalty_home * 0.6 : 0;
  const awayStarPenalty = fv.star_available_away === 0 ? fv.star_injury_penalty_away * 0.6 : 0;

  // Star impact differential (positive = home star > away star)
  const starImpactHome = fv.star_player_impact_diff * 0.12;

  const homeExpPts = Math.max(45, homeRawPts + homeAdv + restBonus + starImpactHome / 2 - homeStarPenalty);
  const awayExpPts = Math.max(45, awayRawPts - restBonus - starImpactHome / 2 - awayStarPenalty);

  // Variance: WBB games have slightly less variance than NBA
  // Lower 3PT% reduces "bomb" variance. Calibrated σ ~11.
  const tempoScale = expectedPoss / WBB_LEAGUE_AVG_TEMPO;
  const homeStd = WBB_SCORE_STD * Math.sqrt(tempoScale);
  const awayStd = WBB_SCORE_STD * Math.sqrt(tempoScale);

  return { homeExpPts, awayExpPts, homeStd, awayStd, expectedPoss };
}

// ─── Main Monte Carlo ─────────────────────────────────────────────────────────

export function runMonteCarlo(fv: FeatureVector): MonteCarloResult {
  const { homeExpPts, awayExpPts, homeStd, awayStd } = estimateExpectedScores(fv);

  let homeWins = 0;
  let totalHomeScore = 0;
  let totalAwayScore = 0;
  let otGames = 0;

  for (let i = 0; i < N_SIMULATIONS; i++) {
    const homeScore = normalRandom(homeExpPts, homeStd);
    const awayScore = normalRandom(awayExpPts, awayStd);

    if (Math.abs(homeScore - awayScore) < 0.5) {
      // OT (approximate ~3.5% of WBB games go to OT)
      otGames++;
      const otHome = normalRandom(WBB_OT_EXTRA_SCORE, 3.0);
      const otAway = normalRandom(WBB_OT_EXTRA_SCORE, 3.0);
      if (homeScore + otHome >= awayScore + otAway) homeWins++;
    } else {
      if (homeScore > awayScore) homeWins++;
    }

    totalHomeScore += homeScore;
    totalAwayScore += awayScore;
  }

  const homeWinPct = homeWins / N_SIMULATIONS;
  const expectedHomeScore = totalHomeScore / N_SIMULATIONS;
  const expectedAwayScore = totalAwayScore / N_SIMULATIONS;

  return {
    homeWinPct,
    awayWinPct: 1 - homeWinPct,
    expectedHomeScore,
    expectedAwayScore,
    expectedSpread: expectedHomeScore - expectedAwayScore,
    otProb: otGames / N_SIMULATIONS,
    simulations: N_SIMULATIONS,
  };
}
