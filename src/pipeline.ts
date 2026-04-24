// NCAAW Oracle v4.1 — Daily Pipeline
// Orchestrates: Season Gate → Fetch → Features → Monte Carlo → ML Model → Edge → Store → Discord

import 'dotenv/config';
import { logger } from './logger.js';
import { fetchSchedule, fetchCompletedGames, getCurrentWBBSeason, getTop25TeamIds } from './api/espnClient.js';
import { initializeOdds, getOddsForGame, hasAnyOdds } from './api/oddsClient.js';
import { computeFeatures } from './features/featureEngine.js';
import { runMonteCarlo } from './models/monteCarlo.js';
import { loadModel, predict as mlPredict, isModelLoaded, getModelInfo, fallbackPredict } from './models/metaModel.js';
import { computeEdge } from './features/marketEdge.js';
import { getConfidenceTier } from './features/marketEdge.js';
import { initDb, closeDb, upsertPrediction, updatePredictionResult, upsertGameResult, recordDailyAccuracy, getSeasonRecord, getPredictionsByDate } from './db/database.js';
import { seedElos, updateElo } from './features/eloEngine.js';
import { getSeasonStatus } from './season-manager/seasonManager.js';
import {
  sendDailyPredictionsEmbed, sendWeeklyRecapEmbed,
  sendSeasonSummaryEmbed, sendPreseasonEmbed,
} from './discord/embedBuilder.js';
import type { WBBGame, Prediction, PipelineOptions, FeatureVector } from './types.js';

const MODEL_VERSION = '4.1.0';

// ─── Main pipeline ────────────────────────────────────────────────────────────

export async function runPipeline(options: PipelineOptions = {}): Promise<Prediction[]> {
  const today = new Date().toISOString().split('T')[0];
  const gameDate = options.date ?? today;
  const season = getCurrentWBBSeason();

  logger.info({ gameDate, version: MODEL_VERSION }, '=== NCAAW Oracle v4.1 Pipeline Start ===');

  // ── Season gate ─────────────────────────────────────────────────────────────
  const seasonStatus = getSeasonStatus(gameDate);
  logger.info({ phase: seasonStatus.phase, label: seasonStatus.label }, 'Season phase');

  if (!seasonStatus.isActive && !options.forceRefresh) {
    logger.info('Season is dormant — exiting. No Discord message sent.');
    return [];
  }

  // ── Init DB + Elo ───────────────────────────────────────────────────────────
  await initDb();
  seedElos();

  // ── Load ML model ───────────────────────────────────────────────────────────
  const modelLoaded = loadModel();
  if (modelLoaded) {
    const info = getModelInfo();
    logger.info({ version: info?.version, accuracy: info?.accuracy }, 'ML model loaded');
  } else {
    logger.info('ML model not found — using fallback (Elo + AdjEM logistic). Run: npm run train');
  }

  // ── Score yesterday's games ─────────────────────────────────────────────────
  const yesterday = new Date(gameDate);
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toISOString().split('T')[0];
  const yesterdayResults = await scoreYesterdayGames(yesterdayStr);

  // ── Load Vegas odds ─────────────────────────────────────────────────────────
  await initializeOdds(gameDate);
  if (hasAnyOdds()) {
    logger.info('Vegas lines loaded');
  }

  // ── Fetch Top 25 rankings ───────────────────────────────────────────────────
  const top25Ids = await getTop25TeamIds();
  if (top25Ids.size > 0) {
    logger.info({ count: top25Ids.size }, 'Top 25 filter active — only predicting ranked-team games');
  } else {
    logger.warn('Top 25 unavailable — predicting all games');
  }

  // ── Fetch today's schedule ──────────────────────────────────────────────────
  const allGames = await fetchSchedule(gameDate);
  if (allGames.length === 0) {
    logger.warn({ gameDate }, 'No games found for date');
    if (options.alertMode === 'morning') {
      await sendDailyPredictionsEmbed([], gameDate, yesterdayResults);
    }
    closeDb();
    return [];
  }

  // Filter to games where at least one team is in the Top 25
  const games = top25Ids.size > 0
    ? allGames.filter(g =>
        top25Ids.has(g.homeTeam.teamId) || top25Ids.has(g.awayTeam.teamId)
      )
    : allGames;

  logger.info({ total: allGames.length, top25Games: games.length }, 'Schedule fetched');

  // ── Process each game ───────────────────────────────────────────────────────
  const predictions: Prediction[] = [];

  for (const game of games) {
    try {
      const pred = await processGame(game, gameDate, modelLoaded, options.forceRefresh);
      if (pred) {
        predictions.push(pred);
        upsertPrediction(pred);
      }
    } catch (err) {
      logger.error(
        { err, gameId: game.gameId, home: game.homeTeam.teamAbbr, away: game.awayTeam.teamAbbr },
        'Failed to process game'
      );
    }
  }

  logger.info({ processed: predictions.length, total: games.length }, 'Pipeline complete');

  // ── Print to console ────────────────────────────────────────────────────────
  if (options.verbose !== false) {
    printPredictions(predictions, gameDate, modelLoaded, seasonStatus.label);
  }

  // ── Send Discord embed ──────────────────────────────────────────────────────
  if (options.alertMode === 'morning') {
    await sendDailyPredictionsEmbed(predictions, gameDate, yesterdayResults);
  }

  closeDb();
  return predictions;
}

// ─── Score yesterday's results + update Elo ──────────────────────────────────

async function scoreYesterdayGames(dateStr: string): Promise<Prediction[]> {
  try {
    const results = await fetchCompletedGames(dateStr);
    const yesterdayPreds = getPredictionsByDate(dateStr);

    for (const result of results) {
      // Update prediction with actual scores
      updatePredictionResult(result.gameId, result.homeScore, result.awayScore);

      // Update Elo
      const mov = result.homeScore - result.awayScore;
      updateElo(result.homeTeam, result.awayTeam, result.homeScore > result.awayScore, 'regular', mov);

      // Store game result
      upsertGameResult({
        game_id: result.gameId,
        game_date: dateStr,
        home_team: result.homeTeam,
        away_team: result.awayTeam,
        home_score: result.homeScore,
        away_score: result.awayScore,
        home_won: result.homeScore > result.awayScore,
      });
    }

    // Record daily accuracy
    if (results.length > 0) recordDailyAccuracy(dateStr);

    return yesterdayPreds;
  } catch (err) {
    logger.warn({ err, dateStr }, 'Failed to score yesterday\'s games');
    return [];
  }
}

// ─── Single game processing ───────────────────────────────────────────────────

async function processGame(
  game: WBBGame,
  gameDate: string,
  modelLoaded: boolean,
  allowFinal = false
): Promise<Prediction | null> {
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;

  // Skip already-finished games (unless --force-refresh is set, e.g. for examples)
  if (!allowFinal && game.status.toLowerCase().includes('final')) {
    logger.debug({ status: game.status }, 'Skipping completed game');
    return null;
  }

  // ── Feature vector ──────────────────────────────────────────────────────────
  const fv: FeatureVector = await computeFeatures(game, gameDate);

  // ── Vegas odds ──────────────────────────────────────────────────────────────
  const line = getOddsForGame(homeAbbr, awayAbbr);
  if (line) {
    fv.has_line = true;
    fv.vegas_home_prob = line.spread !== undefined
      ? (await import('./api/oddsClient.js')).impliedHomeProbFromSpread(line.spread)
      : undefined;
  }

  // ── Monte Carlo ─────────────────────────────────────────────────────────────
  const mc = runMonteCarlo(fv);
  fv.mc_home_win_pct = mc.homeWinPct;
  fv.mc_expected_home_pts = mc.expectedHomeScore;
  fv.mc_expected_away_pts = mc.expectedAwayScore;
  fv.mc_expected_spread = mc.expectedSpread;

  // ── ML calibration ──────────────────────────────────────────────────────────
  let calibratedProb: number;
  if (modelLoaded) {
    calibratedProb = mlPredict(fv, mc.homeWinPct);
  } else {
    calibratedProb = fallbackPredict(fv.elo_diff, fv.adj_em_diff, fv.is_home === 1);
  }

  // ── Edge detection ──────────────────────────────────────────────────────────
  let modelEdge: number | undefined;
  let edgeTier: Prediction['edge_tier'] | undefined;

  if (line) {
    const edgeResult = computeEdge(calibratedProb, line.spread, line.homeML);
    if (edgeResult) {
      modelEdge = edgeResult.edge;
      edgeTier = edgeResult.edgeTier;
      fv.model_edge = modelEdge;
    }
  }

  const tier = getConfidenceTier(calibratedProb);
  const isEarlySeason = fv.early_season_home || fv.early_season_away;

  const pred: Prediction = {
    game_id: game.gameId,
    game_date: gameDate,
    home_team: homeAbbr,
    away_team: awayAbbr,
    mc_prob: mc.homeWinPct,
    calibrated_prob: calibratedProb,
    confidence_tier: tier,
    expected_home_score: mc.expectedHomeScore,
    expected_away_score: mc.expectedAwayScore,
    expected_spread: mc.expectedSpread,
    has_line: !!line,
    vegas_spread: line?.spread,
    vegas_ml_home: line?.homeML,
    model_edge: modelEdge,
    edge_tier: edgeTier,
    early_season: isEarlySeason,
    feature_vector: fv,
    model_version: MODEL_VERSION,
    created_at: new Date().toISOString(),
  };

  return pred;
}

// ─── Console output ───────────────────────────────────────────────────────────

function printPredictions(
  predictions: Prediction[],
  gameDate: string,
  modelLoaded: boolean,
  seasonLabel: string
): void {
  console.log('\n' + '═'.repeat(80));
  console.log(`  NCAAW ORACLE v4.1 — ${gameDate} — ${seasonLabel} — TOP 25 ONLY`);
  console.log('═'.repeat(80));

  const sorted = [...predictions].sort((a, b) => {
    const pa = Math.max(a.calibrated_prob, 1 - a.calibrated_prob);
    const pb = Math.max(b.calibrated_prob, 1 - b.calibrated_prob);
    return pb - pa;
  });

  for (const pred of sorted) {
    const p = Math.max(pred.calibrated_prob, 1 - pred.calibrated_prob);
    const pick = pred.calibrated_prob >= 0.5 ? pred.home_team : pred.away_team;
    const prob = (p * 100).toFixed(1);
    const tier = pred.confidence_tier.toUpperCase().replace('_', ' ');
    const early = pred.early_season ? ' 🟡' : '';
    const edge = pred.model_edge !== undefined
      ? ` | Edge: ${(pred.model_edge * 100).toFixed(1)}%`
      : '';
    const spread = pred.vegas_spread !== undefined
      ? ` | Spread: ${pred.vegas_spread > 0 ? '+' : ''}${pred.vegas_spread}`
      : '';

    console.log(`  ${pred.away_team.padEnd(8)} @ ${pred.home_team.padEnd(8)} | Pick: ${pick.padEnd(8)} ${prob}% | ${tier.padEnd(16)}${spread}${edge}${early}`);
  }

  console.log('═'.repeat(80));
  console.log(`  Total: ${predictions.length} games | Model: ${modelLoaded ? 'ML + Calibration' : 'Fallback (Elo + AdjEM)'}`);
  console.log('═'.repeat(80) + '\n');
}
