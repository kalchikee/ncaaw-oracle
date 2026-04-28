// NCAAW Oracle v4.1 — CLI Entry Point
// Usage:
//   npm start                              → predictions for today
//   npm start -- --date 2026-01-15        → predictions for specific date
//   npm start -- --alert morning          → daily predictions + Discord embed
//   npm start -- --alert recap            → score yesterday + send recap
//   npm start -- --alert weekly           → weekly recap embed
//   npm start -- --alert bracket          → tournament bracket embed
//   npm start -- --alert preseason        → preseason "online" embed
//   npm start -- --help                   → show help

import 'dotenv/config';
import { logger } from './logger.js';
import { runPipeline } from './pipeline.js';
import { initDb, closeDb, getSeasonRecord } from './db/database.js';
import { getSeasonStatus, isSelectionMonday, isSeasonEndDay, hasSentSeasonSummary, markSeasonSummarySent } from './season-manager/seasonManager.js';
import { sendWeeklyRecapEmbed, sendSeasonSummaryEmbed, sendPreseasonEmbed, sendDailyRecapEmbed } from './discord/embedBuilder.js';
import { runBracketSim } from './bracket-sim/bracketSimulator.js';
import { sendBracketEmbed } from './discord/embedBuilder.js';
import { getCurrentWBBSeason } from './api/espnClient.js';
import type { PipelineOptions } from './types.js';

type AlertMode = 'morning' | 'recap' | 'weekly' | 'bracket' | 'preseason' | null;

// ─── CLI parsing ──────────────────────────────────────────────────────────────

function parseArgs(): PipelineOptions & { help: boolean; alertMode: AlertMode } {
  const args = process.argv.slice(2);
  const opts: PipelineOptions & { help: boolean; alertMode: AlertMode } = {
    help: false,
    verbose: true,
    forceRefresh: false,
    alertMode: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--help': case '-h': opts.help = true; break;
      case '--date': case '-d': opts.date = args[++i]; break;
      case '--force-refresh': case '-f': opts.forceRefresh = true; break;
      case '--quiet': case '-q': opts.verbose = false; break;
      case '--alert': case '-a': {
        const mode = args[++i];
        const valid: AlertMode[] = ['morning', 'recap', 'weekly', 'bracket', 'preseason'];
        if (valid.includes(mode as AlertMode)) {
          opts.alertMode = mode as AlertMode;
        } else {
          console.error(`Unknown alert mode: "${mode}". Valid: ${valid.join(', ')}`);
          process.exit(1);
        }
        break;
      }
      default:
        if (/^\d{4}-\d{2}-\d{2}$/.test(arg)) opts.date = arg;
    }
  }
  return opts;
}

// ─── Help ─────────────────────────────────────────────────────────────────────

function printHelp(): void {
  console.log(`
NCAAW Oracle v4.1 — Women's Basketball ML Prediction Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE:
  npm start                              Predict today's games
  npm start -- --date 2026-01-15         Predict specific date
  npm start -- --alert morning           Daily predictions + Discord embed
  npm start -- --alert weekly            Weekly recap embed
  npm start -- --alert bracket           Tournament bracket embed
  npm start -- --alert preseason         Preseason "online" embed
  npm start -- --force-refresh           Bypass cache

ENVIRONMENT VARIABLES:
  DISCORD_WEBHOOK_URL   Discord webhook URL for embeds (required for alerts)
  ODDS_API_KEY          The Odds API key (optional — skip for no Vegas lines)
  DB_PATH               SQLite DB path (default: ./data/oracle.sqlite)
  LOG_LEVEL             Log level: debug | info | warn | error (default: info)

SEASON SCHEDULE:
  Oct: Preseason setup     Nov: Season begins (early-season bootstrap active)
  Jan: Conference play     Mar: Conf tournaments + NCAA Selection Monday
  Apr: Championship → dormant
`);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  const today = opts.date ?? new Date().toISOString().split('T')[0];
  const season = getCurrentWBBSeason();
  const seasonStatus = getSeasonStatus(today);

  logger.info({ today, season, phase: seasonStatus.phase }, 'NCAAW Oracle v4.1 starting');

  // ── Preseason setup ──────────────────────────────────────────────────────────
  if (opts.alertMode === 'preseason') {
    await initDb();
    await sendPreseasonEmbed(season);
    closeDb();
    return;
  }

  // ── Weekly recap ─────────────────────────────────────────────────────────────
  if (opts.alertMode === 'weekly') {
    await initDb();
    const weekEnd = today;
    const weekStart = new Date(today);
    weekStart.setDate(weekStart.getDate() - 6);
    const weekStartStr = weekStart.toISOString().split('T')[0];

    // Compute week number (approximate)
    const seasonStartDate = new Date('2025-11-01');
    const daysSinceStart = (new Date(today).getTime() - seasonStartDate.getTime()) / 86400000;
    const weekNum = Math.max(1, Math.ceil(daysSinceStart / 7));

    await sendWeeklyRecapEmbed(weekNum, weekStartStr, weekEnd);
    closeDb();
    return;
  }

  // ── End-of-day recap ─────────────────────────────────────────────────────────
  if (opts.alertMode === 'recap') {
    await initDb();
    await sendDailyRecapEmbed(today);
    closeDb();
    return;
  }

  // ── Bracket simulation (Selection Monday or manual trigger) ──────────────────
  if (opts.alertMode === 'bracket' || isSelectionMonday(today)) {
    await initDb();
    const sim = await runBracketSim(10000);
    await sendBracketEmbed(sim);
    closeDb();
    return;
  }

  // ── Season summary (day after championship) ───────────────────────────────────
  // Fires only on the narrow SEASON_END date AND only once per season (idempotent).
  // Previously this triggered every day after April 7 because isSeasonEndDay
  // used >= and there was no idempotency tracker.
  if (isSeasonEndDay(today) && seasonStatus.phase === 'season_end') {
    if (hasSentSeasonSummary(season)) {
      logger.info({ season, today }, 'season summary already sent — skipping Discord');
      return;
    }
    await initDb();
    await sendSeasonSummaryEmbed(season);
    markSeasonSummarySent(season);
    closeDb();
    return;
  }

  // After the season-summary window has closed (post-SEASON_END), the daily
  // workflow has nothing to do. Bail silently rather than falling through to
  // the daily-predictions pipeline (which would query an empty schedule).
  if (seasonStatus.phase === 'season_end') {
    logger.info({ today, season }, 'season is over — skipping Discord');
    return;
  }

  // ── Daily predictions ──────────────────────────────────────────────────────────
  await runPipeline({
    date: today,
    verbose: opts.verbose,
    forceRefresh: opts.forceRefresh,
    alertMode: opts.alertMode === 'morning' ? 'morning' : opts.alertMode,
  });
}

main().catch(err => {
  logger.error({ err }, 'Fatal error');
  process.exit(1);
});
