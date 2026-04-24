// NCAAW Oracle v4.1 — Discord Embed Builder + Webhook Sender
// Daily embed: teal (#00838F)
// Weekly recap: green/red based on week result
// Tournament bracket: gold (#FFD700)
// Running season record + high-conviction + ATS in every embed

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getPredictionsByDate, getSeasonRecord, updatePredictionResult } from '../db/database.js';
import {
  getConfidenceTier, isHighConviction, isExtremeConviction,
  formatEdge, getEdgeEmoji,
} from '../features/marketEdge.js';
import { getEarlySeasonLabel } from '../season-manager/seasonManager.js';
import type { Prediction, SeasonRecord, BracketSimResult } from '../types.js';

// ─── Discord types ────────────────────────────────────────────────────────────

interface DiscordField {
  name: string;
  value: string;
  inline?: boolean;
}

interface DiscordEmbed {
  title?: string;
  description?: string;
  color?: number;
  fields?: DiscordField[];
  footer?: { text: string };
  timestamp?: string;
  thumbnail?: { url: string };
}

interface DiscordPayload {
  content?: string;
  embeds: DiscordEmbed[];
}

// ─── Colors ───────────────────────────────────────────────────────────────────

const COLORS = {
  daily: 0x00838F,          // Teal — daily predictions
  recap_good: 0x27ae60,     // Green — good week
  recap_bad: 0xe74c3c,      // Red — bad week
  recap_neutral: 0x95a5a6,  // Gray — neutral
  bracket: 0xFFD700,        // Gold — tournament
  preseason: 0x8e44ad,      // Purple — season start
} as const;

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }

  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(12000),
    });

    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }

    logger.info('Discord embed sent');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
    return false;
  }
}

// ─── Formatters ───────────────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function acc(correct: number, total: number): string {
  if (total === 0) return 'N/A';
  return `${correct}-${total - correct} (${pct(correct / total)})`;
}

function brierStr(sum: number, count: number): string {
  if (count === 0) return 'N/A';
  return (sum / count).toFixed(3);
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
}

function getPickTeam(pred: Prediction): { team: string; prob: number } {
  if (pred.calibrated_prob >= 0.5) return { team: pred.home_team, prob: pred.calibrated_prob };
  return { team: pred.away_team, prob: 1 - pred.calibrated_prob };
}

function getConfidenceEmoji(prob: number): string {
  const p = Math.max(prob, 1 - prob);
  if (p >= 0.80) return '🚀';
  if (p >= 0.74) return '🟢';
  if (p >= 0.65) return '🔵';
  return '⚪';
}

// ─── Daily Predictions Embed ──────────────────────────────────────────────────
// Color: Teal (#00838F)
// Shows: all High Conviction (74%+) games individually
// Running season record + yesterday's results

export async function sendDailyPredictionsEmbed(
  predictions: Prediction[],
  dateStr: string,
  yesterdayResults?: Prediction[]
): Promise<boolean> {
  const seasonRecord = getSeasonRecord();
  const highConv = predictions.filter(p => isHighConviction(p.calibrated_prob));
  const extreme = predictions.filter(p => isExtremeConviction(p.calibrated_prob));

  const hasEarlySeason = predictions.some(p => p.early_season);
  const earlyLabel = hasEarlySeason ? ' 🟡 EARLY SEASON' : '';

  const totalGames = predictions.length;
  const hcCount = highConv.length;
  const exCount = extreme.length;

  // Season stats fields
  const fields: DiscordField[] = [
    {
      name: '🏆 Season Record',
      value: acc(seasonRecord.correct, seasonRecord.total),
      inline: true,
    },
    {
      name: '🎯 High Conviction (74%+)',
      value: acc(seasonRecord.hc_correct, seasonRecord.hc_total),
      inline: true,
    },
    {
      name: '🎰 vs Vegas ATS',
      value: acc(seasonRecord.ats_correct, seasonRecord.ats_total),
      inline: true,
    },
  ];

  // Yesterday's results (if provided)
  if (yesterdayResults && yesterdayResults.length > 0) {
    const yesterdayHC = yesterdayResults.filter(p => isHighConviction(p.calibrated_prob) && p.correct !== undefined);
    if (yesterdayHC.length > 0) {
      const resultLines = yesterdayHC.map(p => {
        const icon = p.correct ? '✅' : '❌';
        const pick = getPickTeam(p);
        const score = p.actual_home_score !== undefined
          ? ` (${p.home_team} ${p.actual_home_score}–${p.actual_away_score} ${p.away_team})`
          : '';
        return `${icon} **${pick.team}** ${pct(pick.prob)}${score}`;
      });
      fields.push({
        name: "📋 Yesterday's HC Results",
        value: resultLines.slice(0, 5).join('\n') || 'No results',
        inline: false,
      });
    }
  }

  // High Conviction games (74%+) — each gets its own field
  if (highConv.length === 0) {
    fields.push({
      name: '📊 Today\'s Games',
      value: `${totalGames} games scheduled. No High Conviction picks today.\n\nModel is seeing close matchups — skip or wait for stronger edges.`,
      inline: false,
    });
  } else {
    // Sort: extreme first, then by probability
    const sorted = [...highConv].sort((a, b) => {
      const pa = Math.max(a.calibrated_prob, 1 - a.calibrated_prob);
      const pb = Math.max(b.calibrated_prob, 1 - b.calibrated_prob);
      return pb - pa;
    });

    for (const pred of sorted.slice(0, 10)) {
      const pick = getPickTeam(pred);
      const emoji = getConfidenceEmoji(pred.calibrated_prob);
      const tier = isExtremeConviction(pred.calibrated_prob) ? ' 🚀 EXTREME' : '';
      const earlyTag = pred.early_season ? ' 🟡' : '';

      let spreadStr = '';
      if (pred.vegas_spread !== undefined) {
        const spreadSign = pred.vegas_spread > 0 ? '+' : '';
        spreadStr = ` | Spread: **${pred.home_team} ${spreadSign}${pred.vegas_spread}**`;
      }

      let edgeStr = '';
      if (pred.model_edge !== undefined && pred.edge_tier && pred.edge_tier !== 'agreement') {
        const edgeEmoji = getEdgeEmoji(pred.edge_tier);
        edgeStr = ` | Edge: ${edgeEmoji}${formatEdge(pred.model_edge)}`;
      }

      fields.push({
        name: `${emoji} ${pred.away_team} @ ${pred.home_team}${tier}${earlyTag}`,
        value: `Pick: **${pick.team}** (${pct(pick.prob)})${spreadStr}${edgeStr}`,
        inline: false,
      });
    }

    // Strong picks (65–74%) — condensed
    const strong = predictions.filter(p => {
      const tier = getConfidenceTier(p.calibrated_prob);
      return tier === 'strong';
    });
    if (strong.length > 0) {
      const strongLines = strong.map(p => {
        const pick = getPickTeam(p);
        return `• ${p.away_team} @ ${p.home_team}: **${pick.team}** ${pct(pick.prob)}`;
      });
      fields.push({
        name: `🔵 Strong Picks (65–74%) — ${strong.length}`,
        value: strongLines.slice(0, 8).join('\n'),
        inline: false,
      });
    }
  }

  const embed: DiscordEmbed = {
    title: `🏀 NCAAW Oracle — ${formatDate(dateStr)} Predictions`,
    description: `**${totalGames}** Top 25 games today. **${hcCount}** High Conviction, **${exCount}** Extreme.${earlyLabel}`,
    color: COLORS.daily,
    fields,
    footer: { text: `NCAAW Oracle v4.1${hasEarlySeason ? ' | 🟡 EARLY SEASON — prior-year bootstrap active' : ''}` },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── Weekly Recap Embed ───────────────────────────────────────────────────────
// Color: Green if week acc > 55%, Red if < 45%, Gray otherwise

export async function sendWeeklyRecapEmbed(
  weekNum: number,
  weekStart: string,
  weekEnd: string
): Promise<boolean> {
  const seasonRecord = getSeasonRecord();
  const recentPreds = getPredictionsByDate(weekEnd);

  // Compute this week's record
  let weekTotal = 0, weekCorrect = 0, weekHCTotal = 0, weekHCCorrect = 0;
  let weekExTotal = 0, weekExCorrect = 0, weekATSTotal = 0, weekATSCorrect = 0;
  let weekBrierSum = 0;

  for (const wr of seasonRecord.week_records) {
    if (wr.start_date === weekStart) {
      weekTotal = wr.total;
      weekCorrect = wr.correct;
      break;
    }
  }

  const weekAcc = weekTotal > 0 ? weekCorrect / weekTotal : 0;
  const color = weekAcc >= 0.60 ? COLORS.recap_good
    : weekAcc <= 0.44 ? COLORS.recap_bad
    : COLORS.recap_neutral;

  // Week-by-week trend
  const trendLines = seasonRecord.week_records.slice(-8).map(wr =>
    `W${wr.week}: ${wr.correct}-${wr.total - wr.correct} (${pct(wr.accuracy)})`
  );

  const fields: DiscordField[] = [
    {
      name: '📅 This Week',
      value: acc(weekCorrect, weekTotal),
      inline: true,
    },
    {
      name: '🏆 Season Record',
      value: acc(seasonRecord.correct, seasonRecord.total),
      inline: true,
    },
    {
      name: '🎯 High Conviction (74%+)',
      value: acc(seasonRecord.hc_correct, seasonRecord.hc_total),
      inline: true,
    },
    {
      name: '🚀 Extreme (80%+)',
      value: acc(seasonRecord.extreme_correct, seasonRecord.extreme_total),
      inline: true,
    },
    {
      name: '🎰 vs Vegas ATS',
      value: acc(seasonRecord.ats_correct, seasonRecord.ats_total),
      inline: true,
    },
    {
      name: '📉 Season Brier Score',
      value: brierStr(seasonRecord.brier_sum, seasonRecord.games_with_brier),
      inline: true,
    },
  ];

  if (trendLines.length > 0) {
    fields.push({
      name: '📈 Week-by-Week Trend',
      value: '```\n' + trendLines.join('\n') + '\n```',
      inline: false,
    });
  }

  const embed: DiscordEmbed = {
    title: `📊 NCAAW Oracle — Week ${weekNum} Recap`,
    description: `Covering **${weekStart}** through **${weekEnd}**`,
    color,
    fields,
    footer: { text: 'NCAAW Oracle v4.1' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── NCAA Tournament Bracket Embed ───────────────────────────────────────────
// Color: Gold (#FFD700)
// Women's tournament: 68 teams (since 2022), Selection Monday (not Sunday)

export async function sendBracketEmbed(sim: BracketSimResult): Promise<boolean> {
  const fields: DiscordField[] = [];

  // Per-region breakdown
  for (const [region, teams] of Object.entries(sim.regions)) {
    const topTeams = teams.slice(0, 4);
    const regionLines = topTeams.map(t => {
      const r32 = (t.advancementProbs['R32'] ?? 0) * 100;
      const s16 = (t.advancementProbs['S16'] ?? 0) * 100;
      const e8 = (t.advancementProbs['E8'] ?? 0) * 100;
      return `**(${t.seed}) ${t.teamName}** — R32: ${r32.toFixed(0)}% | S16: ${s16.toFixed(0)}% | E8: ${e8.toFixed(0)}%`;
    });
    fields.push({
      name: `📍 ${region} Region`,
      value: regionLines.join('\n'),
      inline: false,
    });
  }

  // Final Four odds
  const f4Lines = sim.finalFour.slice(0, 8).map(t =>
    `**${t.team}**: ${(t.prob * 100).toFixed(1)}%`
  );
  fields.push({
    name: '🏟️ Final Four Odds',
    value: f4Lines.join(' | '),
    inline: false,
  });

  // Championship odds
  const champLines = sim.championship.slice(0, 5).map(t =>
    `**${t.team}**: ${(t.prob * 100).toFixed(1)}%`
  );
  fields.push({
    name: '🏆 Championship Odds',
    value: champLines.join(' | '),
    inline: false,
  });

  // Cinderella watch
  if (sim.cinderellaWatch.length > 0) {
    const cinderellaLines = sim.cinderellaWatch.slice(0, 5).map(t =>
      `**(${t.seed}) ${t.team}**: ${(t.sweetSixteenProb * 100).toFixed(1)}% S16`
    );
    fields.push({
      name: '🎭 Cinderella Watch',
      value: cinderellaLines.join('\n'),
      inline: false,
    });
  }

  const embed: DiscordEmbed = {
    title: `🏆 NCAAW Oracle — ${sim.year} NCAA Tournament Bracket`,
    description: `Based on **${sim.simulations.toLocaleString()}** tournament simulations. 68-team bracket.`,
    color: COLORS.bracket,
    fields,
    footer: { text: 'NCAAW Oracle v4.1 | Selection Monday bracket projection' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── Preseason "Online" Embed ─────────────────────────────────────────────────

export async function sendPreseasonEmbed(season: string): Promise<boolean> {
  const embed: DiscordEmbed = {
    title: `🏀 NCAAW Oracle v4.1 — ${season} Season Online`,
    description: `The NCAAW Oracle prediction engine is now active for the **${season}** women's basketball season.\n\nDaily predictions begin with the first games of the season. Model is bootstrapped with prior-year AdjEM + recruiting composites for early-season teams.`,
    color: COLORS.preseason,
    fields: [
      {
        name: '🎯 Accuracy Targets',
        value: '72–76% all games | 78–82% at 74%+ | 82–85% at 80%+',
        inline: false,
      },
      {
        name: '⚙️ Architecture',
        value: 'Custom AdjEM → Logistic Regression → Monte Carlo (10K) → Platt Scaling → Edge Detection',
        inline: false,
      },
      {
        name: '📅 Season Schedule',
        value: 'Daily predictions @ 9AM ET | Monday recaps | NCAA Tournament bracket sim on Selection Monday',
        inline: false,
      },
      {
        name: '🌟 Key Features',
        value: '• Custom opponent-adjusted efficiency (no KenPom for WBB)\n• Star player dominance model\n• WNBA draft impact tracking\n• Prior-year bootstrap for early season\n• Less efficient betting market = larger edges',
        inline: false,
      },
    ],
    footer: { text: 'NCAAW Oracle v4.1 | GitHub Actions hosted | Zero-cost' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── Season Summary Embed ─────────────────────────────────────────────────────

export async function sendSeasonSummaryEmbed(season: string): Promise<boolean> {
  const record = getSeasonRecord();
  const overallAcc = record.total > 0 ? record.correct / record.total : 0;
  const color = overallAcc >= 0.72 ? COLORS.recap_good : overallAcc >= 0.65 ? COLORS.recap_neutral : COLORS.recap_bad;

  const fields: DiscordField[] = [
    { name: '🏆 Final Season Record', value: acc(record.correct, record.total), inline: true },
    { name: '🎯 High Conviction (74%+)', value: acc(record.hc_correct, record.hc_total), inline: true },
    { name: '🚀 Extreme (80%+)', value: acc(record.extreme_correct, record.extreme_total), inline: true },
    { name: '🎰 vs Vegas ATS', value: acc(record.ats_correct, record.ats_total), inline: true },
    { name: '📉 Season Brier', value: brierStr(record.brier_sum, record.games_with_brier), inline: true },
  ];

  // Best weeks
  const sortedWeeks = [...record.week_records].sort((a, b) => b.accuracy - a.accuracy);
  if (sortedWeeks.length > 0) {
    const bestWeeks = sortedWeeks.slice(0, 3).map(w =>
      `W${w.week}: ${w.correct}-${w.total - w.correct} (${pct(w.accuracy)})`
    );
    fields.push({ name: '🌟 Best Weeks', value: bestWeeks.join('\n'), inline: false });
  }

  const embed: DiscordEmbed = {
    title: `🏀 NCAAW Oracle — ${season} Season Complete`,
    description: `The ${season} women's basketball season has concluded. Here's the final performance summary.`,
    color,
    fields,
    footer: { text: 'NCAAW Oracle v4.1 | See you in November!' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── Daily End-of-Day Recap Embed ─────────────────────────────────────────────
// Sent after games complete — shows how the day's picks performed
// Color: Green if day acc > 65%, Red if < 40%, Orange otherwise

export async function sendDailyRecapEmbed(dateStr: string): Promise<boolean> {
  const preds = getPredictionsByDate(dateStr).filter(p => p.correct !== undefined);
  const seasonRecord = getSeasonRecord();

  if (preds.length === 0) {
    const embed: DiscordEmbed = {
      title: `🏀 NCAAW Oracle — ${formatDate(dateStr)} Recap`,
      description: `No results recorded yet for ${formatDate(dateStr)}. Results update after games complete.`,
      color: COLORS.recap_neutral,
      footer: { text: 'NCAAW Oracle v4.1' },
      timestamp: new Date().toISOString(),
    };
    return sendWebhook({ embeds: [embed] });
  }

  const hcPreds = preds.filter(p => isHighConviction(p.calibrated_prob));
  const extremePreds = preds.filter(p => isExtremeConviction(p.calibrated_prob));
  const allCorrect = preds.filter(p => p.correct).length;
  const hcCorrect = hcPreds.filter(p => p.correct).length;
  const exCorrect = extremePreds.filter(p => p.correct).length;

  const dayAcc = preds.length > 0 ? allCorrect / preds.length : 0;
  const color = dayAcc >= 0.65 ? COLORS.recap_good
    : dayAcc <= 0.40 ? COLORS.recap_bad
    : 0xE67E22;

  const fields: DiscordField[] = [
    {
      name: "📅 Today's Record",
      value: acc(allCorrect, preds.length),
      inline: true,
    },
    {
      name: '🎯 High Conviction (74%+)',
      value: hcPreds.length > 0 ? acc(hcCorrect, hcPreds.length) : 'No picks',
      inline: true,
    },
    {
      name: '🚀 Extreme (80%+)',
      value: extremePreds.length > 0 ? acc(exCorrect, extremePreds.length) : 'No picks',
      inline: true,
    },
    {
      name: '🏆 Season Record',
      value: acc(seasonRecord.correct, seasonRecord.total),
      inline: true,
    },
    {
      name: '🎯 Season HC (74%+)',
      value: acc(seasonRecord.hc_correct, seasonRecord.hc_total),
      inline: true,
    },
    {
      name: '🎰 Season ATS',
      value: acc(seasonRecord.ats_correct, seasonRecord.ats_total),
      inline: true,
    },
  ];

  // High Conviction pick results
  if (hcPreds.length > 0) {
    const sorted = [...hcPreds].sort((a, b) => {
      const pa = Math.max(a.calibrated_prob, 1 - a.calibrated_prob);
      const pb = Math.max(b.calibrated_prob, 1 - b.calibrated_prob);
      return pb - pa;
    });

    const resultLines = sorted.map(p => {
      const icon = p.correct ? '✅' : '❌';
      const pick = getPickTeam(p);
      const scoreStr = p.actual_home_score !== undefined
        ? ` — ${p.home_team} **${p.actual_home_score}**, ${p.away_team} **${p.actual_away_score}**`
        : '';
      const edgeStr = p.model_edge !== undefined && p.edge_tier && p.edge_tier !== 'agreement'
        ? ` ${getEdgeEmoji(p.edge_tier)}${formatEdge(p.model_edge)}`
        : '';
      const exTag = isExtremeConviction(p.calibrated_prob) ? ' 🚀' : '';
      return `${icon}${exTag} **${pick.team}** ${pct(pick.prob)} vs ${p.calibrated_prob >= 0.5 ? p.away_team : p.home_team}${edgeStr}${scoreStr}`;
    });

    fields.push({
      name: `🎯 High Conviction Results (${hcPreds.length} picks)`,
      value: resultLines.slice(0, 10).join('\n'),
      inline: false,
    });
  }

  // Other picks condensed
  const otherPreds = preds.filter(p => !isHighConviction(p.calibrated_prob));
  if (otherPreds.length > 0) {
    const otherCorrect = otherPreds.filter(p => p.correct).length;
    const otherLines = otherPreds.map(p => {
      const icon = p.correct ? '✅' : '❌';
      const pick = getPickTeam(p);
      return `${icon} ${p.away_team} @ ${p.home_team}: **${pick.team}** ${pct(pick.prob)}`;
    });
    fields.push({
      name: `📋 Other Picks — ${otherCorrect}/${otherPreds.length} correct`,
      value: otherLines.slice(0, 8).join('\n'),
      inline: false,
    });
  }

  // Best / worst pick highlights
  const bestPick = preds
    .filter(p => p.correct && p.actual_home_score !== undefined)
    .sort((a, b) => Math.max(b.calibrated_prob, 1 - b.calibrated_prob) - Math.max(a.calibrated_prob, 1 - a.calibrated_prob))[0];
  const worstPick = preds
    .filter(p => !p.correct && p.actual_home_score !== undefined)
    .sort((a, b) => Math.max(b.calibrated_prob, 1 - b.calibrated_prob) - Math.max(a.calibrated_prob, 1 - a.calibrated_prob))[0];

  if (bestPick || worstPick) {
    const highlights: string[] = [];
    if (bestPick) {
      const pick = getPickTeam(bestPick);
      highlights.push(`🌟 **Best:** ${pick.team} ${pct(pick.prob)} ✅ (${bestPick.home_team} ${bestPick.actual_home_score}–${bestPick.actual_away_score} ${bestPick.away_team})`);
    }
    if (worstPick) {
      const pick = getPickTeam(worstPick);
      highlights.push(`💔 **Miss:** ${pick.team} ${pct(pick.prob)} ❌ (${worstPick.home_team} ${worstPick.actual_home_score}–${worstPick.actual_away_score} ${worstPick.away_team})`);
    }
    fields.push({ name: '📌 Highlights', value: highlights.join('\n'), inline: false });
  }

  const grade = dayAcc >= 0.80 ? 'A+' : dayAcc >= 0.72 ? 'A' : dayAcc >= 0.65 ? 'B+'
    : dayAcc >= 0.58 ? 'B' : dayAcc >= 0.50 ? 'C' : 'D';

  const embed: DiscordEmbed = {
    title: `🏀 NCAAW Oracle — ${formatDate(dateStr)} Results`,
    description: `Day Grade: **${grade}** | ${allCorrect}/${preds.length} correct (${pct(dayAcc)}) across Top 25 games`,
    color,
    fields,
    footer: { text: 'NCAAW Oracle v4.1' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}
