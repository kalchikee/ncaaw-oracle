// NCAAW Oracle v4.1 — Custom Adjusted Efficiency Calculator
// No KenPom equivalent exists for WBB — we build it in-house.
// Uses iterative opponent-adjustment methodology (KenPom-style) on women's box score data.
// WBB league averages: ~62 PPG, ~66 poss/40 min, AdjEM range typically -25 to +35

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type { AdjEfficiency, WBBTeam } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const DATA_DIR = resolve(__dirname, '../../data');
const CACHE_DIR = resolve(__dirname, '../../cache');
mkdirSync(DATA_DIR, { recursive: true });
mkdirSync(CACHE_DIR, { recursive: true });

// WBB D-I League averages
const WBB_LEAGUE_AVG_PPG = 62.0;
const WBB_LEAGUE_AVG_POSS = 66.0;
const WBB_LEAGUE_AVG_OE = (WBB_LEAGUE_AVG_PPG / WBB_LEAGUE_AVG_POSS) * 100; // ~93.9

export interface GameBoxScore {
  homeTeamId: string;
  awayTeamId: string;
  homeScore: number;
  awayScore: number;
  homePoss: number;    // estimated possessions
  awayPoss: number;
  neutralSite: boolean;
  gameDate: string;
}

let _adjEfficiency: Map<string, AdjEfficiency> = new Map();

const ADJ_EM_CACHE = resolve(CACHE_DIR, 'adj_em.json');

// ─── Load/save cache ──────────────────────────────────────────────────────────

export function loadAdjEfficiency(): Map<string, AdjEfficiency> {
  if (existsSync(ADJ_EM_CACHE)) {
    try {
      const data = JSON.parse(readFileSync(ADJ_EM_CACHE, 'utf8')) as AdjEfficiency[];
      _adjEfficiency = new Map(data.map(e => [e.teamId, e]));
      logger.debug({ teams: _adjEfficiency.size }, 'Custom AdjEM loaded from cache');
    } catch {
      _adjEfficiency = new Map();
    }
  }
  return _adjEfficiency;
}

export function saveAdjEfficiency(): void {
  writeFileSync(ADJ_EM_CACHE, JSON.stringify([..._adjEfficiency.values()], null, 2));
}

export function getAdjEM(teamId: string): AdjEfficiency | null {
  return _adjEfficiency.get(teamId) ?? null;
}

// ─── Possession estimator ─────────────────────────────────────────────────────
// WBB possession estimate: (FGA - OREB + TOV + 0.44 × FTA)
// Without play-by-play, estimate from score: score / (OE/100)
export function estimatePossessions(score: number, tempo = WBB_LEAGUE_AVG_POSS): number {
  return Math.round(tempo);
}

// ─── Iterative opponent adjustment ───────────────────────────────────────────
// Classic KenPom methodology adapted for WBB:
//   AdjOE_i = RawOE_i × (League_AvgDE / AvgOppDE_i)
//   AdjDE_i = RawDE_i × (League_AvgOE / AvgOppOE_i)
// Iterate until convergence (~10 iterations)

export function computeAdjEfficiency(
  games: GameBoxScore[],
  teamIds: string[],
  maxIterations = 12
): Map<string, AdjEfficiency> {
  if (games.length === 0) {
    logger.warn('No games provided for AdjEM computation');
    return _adjEfficiency;
  }

  // Step 1: Compute raw efficiencies
  const rawOE = new Map<string, { sum: number; poss: number; games: number }>();
  const rawDE = new Map<string, { sum: number; poss: number; games: number }>();
  const rawTempo = new Map<string, { sum: number; games: number }>();

  for (const id of teamIds) {
    rawOE.set(id, { sum: 0, poss: 0, games: 0 });
    rawDE.set(id, { sum: 0, poss: 0, games: 0 });
    rawTempo.set(id, { sum: 0, games: 0 });
  }

  for (const g of games) {
    const homePoss = g.homePoss || estimatePossessions(g.homeScore);
    const awayPoss = g.awayPoss || estimatePossessions(g.awayScore);
    const avgPoss = (homePoss + awayPoss) / 2;

    // Home team offensive efficiency
    const homeRawOE = (g.homeScore / avgPoss) * 100;
    // Away team offensive efficiency
    const awayRawOE = (g.awayScore / avgPoss) * 100;

    const homeOE = rawOE.get(g.homeTeamId);
    if (homeOE) { homeOE.sum += homeRawOE; homeOE.poss += avgPoss; homeOE.games++; }

    const awayOE = rawOE.get(g.awayTeamId);
    if (awayOE) { awayOE.sum += awayRawOE; awayOE.poss += avgPoss; awayOE.games++; }

    const homeDE = rawDE.get(g.homeTeamId);
    if (homeDE) { homeDE.sum += awayRawOE; homeDE.poss += avgPoss; homeDE.games++; }

    const awayDE = rawDE.get(g.awayTeamId);
    if (awayDE) { awayDE.sum += homeRawOE; awayDE.poss += avgPoss; awayDE.games++; }

    const homeTempo = rawTempo.get(g.homeTeamId);
    if (homeTempo) { homeTempo.sum += avgPoss; homeTempo.games++; }

    const awayTempo = rawTempo.get(g.awayTeamId);
    if (awayTempo) { awayTempo.sum += avgPoss; awayTempo.games++; }
  }

  // Initialize adj efficiency with raw values
  const adjOE = new Map<string, number>();
  const adjDE = new Map<string, number>();
  const adjTempoMap = new Map<string, number>();

  for (const id of teamIds) {
    const oe = rawOE.get(id);
    const de = rawDE.get(id);
    const tempo = rawTempo.get(id);

    adjOE.set(id, oe && oe.games > 0 ? oe.sum / oe.games : WBB_LEAGUE_AVG_OE);
    adjDE.set(id, de && de.games > 0 ? de.sum / de.games : WBB_LEAGUE_AVG_OE);
    adjTempoMap.set(id, tempo && tempo.games > 0 ? tempo.sum / tempo.games : WBB_LEAGUE_AVG_POSS);
  }

  // Step 2: Build opponent schedule lists
  const oppOE = new Map<string, string[]>();   // for each team: list of opponents' IDs
  const oppDE = new Map<string, string[]>();

  for (const id of teamIds) {
    oppOE.set(id, []);
    oppDE.set(id, []);
  }

  for (const g of games) {
    oppOE.get(g.homeTeamId)?.push(g.awayTeamId);   // home offense faced away defense
    oppOE.get(g.awayTeamId)?.push(g.homeTeamId);
    oppDE.get(g.homeTeamId)?.push(g.awayTeamId);
    oppDE.get(g.awayTeamId)?.push(g.homeTeamId);
  }

  // Step 3: Iterative adjustment
  for (let iter = 0; iter < maxIterations; iter++) {
    const newAdjOE = new Map<string, number>();
    const newAdjDE = new Map<string, number>();

    for (const id of teamIds) {
      const oe = rawOE.get(id);
      if (!oe || oe.games === 0) { newAdjOE.set(id, WBB_LEAGUE_AVG_OE); newAdjDE.set(id, WBB_LEAGUE_AVG_OE); continue; }

      const rawOEVal = oe.sum / oe.games;
      const rawDEVal = (rawDE.get(id)?.sum ?? 0) / Math.max(1, rawDE.get(id)?.games ?? 1);

      // Average opponent defensive efficiency (what defense the offense faced)
      const oppsForOE = oppOE.get(id) ?? [];
      const avgOppDE = oppsForOE.length > 0
        ? oppsForOE.reduce((s, oppId) => s + (adjDE.get(oppId) ?? WBB_LEAGUE_AVG_OE), 0) / oppsForOE.length
        : WBB_LEAGUE_AVG_OE;

      // Average opponent offensive efficiency (what offense the defense faced)
      const oppsForDE = oppDE.get(id) ?? [];
      const avgOppOE = oppsForDE.length > 0
        ? oppsForDE.reduce((s, oppId) => s + (adjOE.get(oppId) ?? WBB_LEAGUE_AVG_OE), 0) / oppsForDE.length
        : WBB_LEAGUE_AVG_OE;

      // Adjust: if you played weak opponents, scale down; strong opponents, scale up
      newAdjOE.set(id, rawOEVal * (WBB_LEAGUE_AVG_OE / Math.max(avgOppDE, 1)));
      newAdjDE.set(id, rawDEVal * (WBB_LEAGUE_AVG_OE / Math.max(avgOppOE, 1)));
    }

    // Re-normalize to league average
    const oeMean = [...newAdjOE.values()].reduce((s, v) => s + v, 0) / newAdjOE.size;
    const deMean = [...newAdjDE.values()].reduce((s, v) => s + v, 0) / newAdjDE.size;
    const oeScale = WBB_LEAGUE_AVG_OE / oeMean;
    const deScale = WBB_LEAGUE_AVG_OE / deMean;

    newAdjOE.forEach((v, k) => newAdjOE.set(k, v * oeScale));
    newAdjDE.forEach((v, k) => newAdjDE.set(k, v * deScale));

    // Copy to working maps
    newAdjOE.forEach((v, k) => adjOE.set(k, v));
    newAdjDE.forEach((v, k) => adjDE.set(k, v));
  }

  // Step 4: Store results
  const now = new Date().toISOString();
  _adjEfficiency.clear();

  for (const id of teamIds) {
    const oe = adjOE.get(id) ?? WBB_LEAGUE_AVG_OE;
    const de = adjDE.get(id) ?? WBB_LEAGUE_AVG_OE;
    const rawOeData = rawOE.get(id);

    _adjEfficiency.set(id, {
      teamId: id,
      teamName: id,
      adjOE: oe,
      adjDE: de,
      adjEM: oe - de,
      adjTempo: adjTempoMap.get(id) ?? WBB_LEAGUE_AVG_POSS,
      rawOE: rawOeData ? rawOeData.sum / Math.max(1, rawOeData.games) : WBB_LEAGUE_AVG_OE,
      rawDE: rawDE.get(id) ? (rawDE.get(id)!.sum / Math.max(1, rawDE.get(id)!.games)) : WBB_LEAGUE_AVG_OE,
      gamesPlayed: rawOeData?.games ?? 0,
      lastUpdated: now,
    });
  }

  saveAdjEfficiency();
  logger.info({ teams: _adjEfficiency.size, iterations: maxIterations }, 'Custom AdjEM computed');
  return _adjEfficiency;
}

// ─── Apply custom AdjEM to WBBTeam ────────────────────────────────────────────

export function applyAdjEfficiencyToTeam(team: WBBTeam): WBBTeam {
  const adj = _adjEfficiency.get(team.teamId);
  if (!adj) return team;
  return {
    ...team,
    adjOE: adj.adjOE,
    adjDE: adj.adjDE,
    adjEM: adj.adjEM,
    adjTempo: adj.adjTempo,
  };
}

// ─── Blend with prior year (bootstrap) ───────────────────────────────────────

export function blendWithPriorYear(
  current: AdjEfficiency,
  priorAdjEM: number,
  priorAdjOE: number,
  priorAdjDE: number,
  priorWeight: number
): AdjEfficiency {
  const w = Math.max(0, Math.min(1, priorWeight));
  const iw = 1 - w;
  return {
    ...current,
    adjOE: w * priorAdjOE + iw * current.adjOE,
    adjDE: w * priorAdjDE + iw * current.adjDE,
    adjEM: w * priorAdjEM + iw * current.adjEM,
  };
}
