// NCAAW Oracle v4.1 — Feature Engineering
// 40+ WBB-specific features with women's-calibrated averages
// CRITICAL: Use WBB D-I averages — eFG% ~45%, TOV% ~21%, ORB% ~31%
// DO NOT use men's averages — will systematically bias feature values

import { logger } from '../logger.js';
import {
  fetchTeamDetails, fetchTeamRoster, fetchTeamLastGameDate,
  defaultWBBTeam, getCurrentWBBSeason,
} from '../api/espnClient.js';
import { getEloDiff, eloWinProb, log5Prob } from './eloEngine.js';
import { getAdjEM, loadAdjEfficiency } from '../custom-adj-em/adjEmCalculator.js';
import {
  loadBootstrapData, getPriorYearData, blendEfficiency,
  computeStarPlayerEffect, starImpactToAdjEM, getHomeCourtAdvantage,
} from './bootstrapEngine.js';
import type { WBBGame, WBBTeam, FeatureVector } from '../types.js';

// WBB D-I League averages (women's, NOT men's)
const WBB_AVG_EFG = 0.450;
const WBB_AVG_TOV = 0.210;
const WBB_AVG_ORB = 0.310;
const WBB_AVG_FT_RATE = 0.300;
const WBB_AVG_3PT_PCT = 0.310;
const WBB_AVG_2PT_PCT = 0.460;
const WBB_AVG_BLK = 0.100;
const WBB_AVG_STL = 0.120;

// WBB conference quality index (higher = stronger conference)
const CONFERENCE_QUALITY: Record<string, number> = {
  'SEC': 9.5,
  'ACC': 9.0,
  'Big 12': 8.8,
  'Big Ten': 8.5,
  'Pac-12': 8.0,
  'Big East': 7.5,
  'American': 6.5,
  'Mountain West': 6.0,
  'WAC': 5.5,
  'Horizon': 5.0,
  'Summit': 5.0,
  'Unknown': 5.0,
};

// ─── Main feature computation ─────────────────────────────────────────────────

export async function computeFeatures(
  game: WBBGame,
  gameDate: string
): Promise<FeatureVector> {
  const homeId = game.homeTeam.teamId;
  const awayId = game.awayTeam.teamId;
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;

  logger.debug({ home: homeAbbr, away: awayAbbr }, 'Computing WBB features');

  // Load support data
  loadBootstrapData();
  loadAdjEfficiency();

  const season = getCurrentWBBSeason();

  // Fetch in parallel
  const [homeTeamRaw, awayTeamRaw, homePlayers, awayPlayers, homeLastGame, awayLastGame] =
    await Promise.all([
      fetchTeamDetails(homeId, season).then(t => t ?? defaultWBBTeam(homeAbbr)),
      fetchTeamDetails(awayId, season).then(t => t ?? defaultWBBTeam(awayAbbr)),
      fetchTeamRoster(homeId, season),
      fetchTeamRoster(awayId, season),
      fetchTeamLastGameDate(homeAbbr, gameDate),
      fetchTeamLastGameDate(awayAbbr, gameDate),
    ]);

  // Apply custom AdjEM if available
  const homeAdj = getAdjEM(homeId);
  const awayAdj = getAdjEM(awayId);

  const homeTeam: WBBTeam = homeAdj
    ? { ...homeTeamRaw, adjOE: homeAdj.adjOE, adjDE: homeAdj.adjDE, adjEM: homeAdj.adjEM, adjTempo: homeAdj.adjTempo }
    : homeTeamRaw;
  const awayTeam: WBBTeam = awayAdj
    ? { ...awayTeamRaw, adjOE: awayAdj.adjOE, adjDE: awayAdj.adjDE, adjEM: awayAdj.adjEM, adjTempo: awayAdj.adjTempo }
    : awayTeamRaw;

  // Prior-year data for bootstrap
  const homePrior = getPriorYearData(homeId);
  const awayPrior = getPriorYearData(awayId);

  // Blended efficiency (bootstrap for early-season teams)
  const homeBlended = blendEfficiency(homeTeam, homePrior);
  const awayBlended = blendEfficiency(awayTeam, awayPrior);

  // ── Elo ───────────────────────────────────────────────────────────────────
  const eloDiff = getEloDiff(homeAbbr, awayAbbr);

  // ── Adjusted efficiency diffs ─────────────────────────────────────────────
  const adjEmDiff = homeBlended.adjEM - awayBlended.adjEM;
  const adjOeDiff = homeBlended.adjOE - awayBlended.adjOE;
  const adjDeDiff = homeBlended.adjDE - awayBlended.adjDE;
  const adjTempoDiff = homeBlended.adjTempo - awayBlended.adjTempo;

  // ── Pythagorean ───────────────────────────────────────────────────────────
  const pythagoreanDiff = homeTeam.pythagoreanWinPct - awayTeam.pythagoreanWinPct;

  // ── Four Factors diffs (normalized to WBB averages) ───────────────────────
  const efgDiff = (homeTeam.efgPct - WBB_AVG_EFG) - (awayTeam.efgPct - WBB_AVG_EFG);
  const tovDiff = (homeTeam.tovPct - WBB_AVG_TOV) - (awayTeam.tovPct - WBB_AVG_TOV);
  const orebDiff = (homeTeam.orbPct - WBB_AVG_ORB) - (awayTeam.orbPct - WBB_AVG_ORB);
  const ftRateDiff = (homeTeam.ftRate - WBB_AVG_FT_RATE) - (awayTeam.ftRate - WBB_AVG_FT_RATE);
  const threePtDiff = (homeTeam.threePtPct - WBB_AVG_3PT_PCT) - (awayTeam.threePtPct - WBB_AVG_3PT_PCT);
  const twoPtDiff = (homeTeam.twoPtPct - WBB_AVG_2PT_PCT) - (awayTeam.twoPtPct - WBB_AVG_2PT_PCT);
  const blkDiff = (homeTeam.blockPct - WBB_AVG_BLK) - (awayTeam.blockPct - WBB_AVG_BLK);
  const stlDiff = (homeTeam.stealPct - WBB_AVG_STL) - (awayTeam.stealPct - WBB_AVG_STL);

  // ── Star player model ─────────────────────────────────────────────────────
  const homeStarEffect = computeStarPlayerEffect(
    homeId, homePlayers,
    homePrior?.starPlayerBPM
  );
  const awayStarEffect = computeStarPlayerEffect(
    awayId, awayPlayers,
    awayPrior?.starPlayerBPM
  );

  const starImpactDiff = starImpactToAdjEM(homeStarEffect.impact) - starImpactToAdjEM(awayStarEffect.impact);

  // ── Recruiting / continuity (bootstrapped) ────────────────────────────────
  const recruitingDiff = (homePrior?.recruitingComposite ?? 0.5) - (awayPrior?.recruitingComposite ?? 0.5);
  const returningMinutesDiff = (homePrior?.returningMinutesPct ?? 0.65) - (awayPrior?.returningMinutesPct ?? 0.65);
  const portalDiff = (homePrior?.portalImpact ?? 0) - (awayPrior?.portalImpact ?? 0);

  // Experience proxy: use games played as experience indicator
  const experienceDiff = homeTeam.gamesPlayed - awayTeam.gamesPlayed;

  // ── Strength of schedule ──────────────────────────────────────────────────
  const sosDiff = (homeTeam.strengthOfSchedule ?? 0) - (awayTeam.strengthOfSchedule ?? 0);

  // ── Conference quality ────────────────────────────────────────────────────
  const homeConfQuality = CONFERENCE_QUALITY[homeTeam.conference] ?? 5.0;
  const awayConfQuality = CONFERENCE_QUALITY[awayTeam.conference] ?? 5.0;
  const confQualityDiff = homeConfQuality - awayConfQuality;

  // ── Bench depth (proxy: games played for back-rotation players) ───────────
  const benchDepthDiff = 0; // populated when detailed roster data available

  // ── Rest / fatigue ────────────────────────────────────────────────────────
  const homeRestDays = homeLastGame
    ? Math.min(14, Math.floor((new Date(gameDate).getTime() - new Date(homeLastGame).getTime()) / 86400000))
    : 3;
  const awayRestDays = awayLastGame
    ? Math.min(14, Math.floor((new Date(gameDate).getTime() - new Date(awayLastGame).getTime()) / 86400000))
    : 3;
  const restDaysDiff = homeRestDays - awayRestDays;

  // ── Venue ─────────────────────────────────────────────────────────────────
  const isHome = game.neutralSite ? 0 : 1;
  const isNeutralSite = game.neutralSite ? 1 : 0;

  return {
    game_id: game.gameId,
    home_team: homeAbbr,
    away_team: awayAbbr,
    game_date: gameDate,

    early_season_home: homeBlended.isBootstrap,
    early_season_away: awayBlended.isBootstrap,
    blend_weight_home: homeBlended.priorWeight,
    blend_weight_away: awayBlended.priorWeight,

    elo_diff: eloDiff,
    adj_em_diff: adjEmDiff,
    adj_oe_diff: adjOeDiff,
    adj_de_diff: adjDeDiff,
    adj_tempo_diff: adjTempoDiff,
    pythagorean_diff: pythagoreanDiff,

    efg_pct_diff: efgDiff,
    tov_pct_diff: tovDiff,
    oreb_pct_diff: orebDiff,
    ft_rate_diff: ftRateDiff,
    three_pt_pct_diff: threePtDiff,
    two_pt_pct_diff: twoPtDiff,
    block_pct_diff: blkDiff,
    steal_pct_diff: stlDiff,

    recruiting_composite_diff: recruitingDiff,
    returning_minutes_diff: returningMinutesDiff,
    portal_impact_diff: portalDiff,
    star_player_impact_diff: starImpactDiff,
    experience_diff: experienceDiff,

    sos_diff: sosDiff,
    conference_quality_diff: confQualityDiff,
    bench_depth_diff: benchDepthDiff,

    is_home: isHome,
    is_neutral_site: isNeutralSite,

    rest_days_diff: restDaysDiff,

    star_available_home: homeStarEffect.starAvailable ? 1 : 0,
    star_available_away: awayStarEffect.starAvailable ? 1 : 0,
    star_injury_penalty_home: homeStarEffect.injuryPenalty,
    star_injury_penalty_away: awayStarEffect.injuryPenalty,

    has_line: false,
  };
}

// ─── Logistic regression input vector ─────────────────────────────────────────
// The 15 core features used by the ML meta-model

export function toModelInputVector(fv: FeatureVector): number[] {
  return [
    fv.elo_diff / 200,                  // normalize
    fv.adj_em_diff / 20,
    fv.adj_oe_diff / 15,
    fv.adj_de_diff / 15,
    fv.pythagorean_diff,
    fv.efg_pct_diff / 0.05,
    fv.tov_pct_diff / 0.05,
    fv.oreb_pct_diff / 0.05,
    fv.star_player_impact_diff / 5,
    fv.rest_days_diff / 3,
    fv.is_home,
    fv.is_neutral_site,
    fv.conference_quality_diff / 5,
    fv.recruiting_composite_diff,
    fv.returning_minutes_diff,
    fv.blend_weight_home,               // early season flag
    fv.blend_weight_away,
    fv.star_available_home,
    fv.star_available_away,
    fv.star_injury_penalty_home / 5,
    fv.star_injury_penalty_away / 5,
  ];
}
