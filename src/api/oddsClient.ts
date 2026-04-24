// NCAAW Oracle v4.1 — Odds API Client
// WBB betting lines are the least efficient of any major sport.
// Not all games will have lines — focus on those that do.

import fetch from 'node-fetch';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CACHE_DIR = resolve(__dirname, '../../cache');
mkdirSync(CACHE_DIR, { recursive: true });

const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';
const CACHE_TTL_MS = 2 * 60 * 60 * 1000; // 2 hours

export interface VegasLine {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  spread: number;          // home team spread (negative = home favored)
  homeML?: number;         // home moneyline (American odds)
  awayML?: number;
  total?: number;          // over/under
  source: string;
}

let _oddsMap = new Map<string, VegasLine>();

function cachePath(key: string): string {
  return resolve(CACHE_DIR, `odds_${key}.json`);
}

function readCache<T>(key: string): T | null {
  const p = cachePath(key);
  if (!existsSync(p)) return null;
  try {
    const { ts, data } = JSON.parse(readFileSync(p, 'utf8'));
    if (Date.now() - ts > CACHE_TTL_MS) return null;
    return data as T;
  } catch { return null; }
}

function writeCache(key: string, data: unknown): void {
  writeFileSync(cachePath(key), JSON.stringify({ ts: Date.now(), data }));
}

// ─── Main: load odds for a date ───────────────────────────────────────────────

export async function initializeOdds(dateStr: string): Promise<void> {
  _oddsMap.clear();

  const apiKey = process.env.ODDS_API_KEY;
  if (!apiKey) {
    logger.info('ODDS_API_KEY not set — running without Vegas lines');
    return;
  }

  const key = `ncaaw_${dateStr}`;
  const cached = readCache<VegasLine[]>(key);
  if (cached) {
    cached.forEach(l => _oddsMap.set(normalizeTeam(l.homeTeam) + '_' + normalizeTeam(l.awayTeam), l));
    logger.info({ count: _oddsMap.size }, 'Vegas lines loaded from cache');
    return;
  }

  const url = `${ODDS_API_BASE}/sports/basketball_ncaaw/odds/?apiKey=${apiKey}&regions=us&markets=spreads,h2h&oddsFormat=american`;
  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!resp.ok) { logger.warn({ status: resp.status }, 'Odds API error'); return; }

    const data = await resp.json() as Record<string, unknown>[];
    const lines: VegasLine[] = [];

    for (const game of data) {
      const homeTeam = String(game.home_team ?? '');
      const awayTeam = String(game.away_team ?? '');
      const bookmakers = (game.bookmakers as Record<string, unknown>[]) ?? [];

      // Prefer DraftKings, fallback to first available
      const book = bookmakers.find((b: Record<string, unknown>) => b.key === 'draftkings')
        ?? bookmakers[0];
      if (!book) continue;

      const markets = (book.markets as Record<string, unknown>[]) ?? [];
      const spreadsMarket = markets.find((m: Record<string, unknown>) => m.key === 'spreads');
      const h2hMarket = markets.find((m: Record<string, unknown>) => m.key === 'h2h');

      let spread = 0;
      let homeML: number | undefined;
      let awayML: number | undefined;

      if (spreadsMarket) {
        const outcomes = spreadsMarket.outcomes as Record<string, unknown>[];
        const homeOutcome = outcomes?.find((o: Record<string, unknown>) => o.name === homeTeam);
        spread = Number(homeOutcome?.point ?? 0);
      }

      if (h2hMarket) {
        const outcomes = h2hMarket.outcomes as Record<string, unknown>[];
        homeML = Number(outcomes?.find((o: Record<string, unknown>) => o.name === homeTeam)?.price ?? undefined);
        awayML = Number(outcomes?.find((o: Record<string, unknown>) => o.name === awayTeam)?.price ?? undefined);
      }

      const line: VegasLine = {
        gameId: String(game.id ?? ''),
        homeTeam,
        awayTeam,
        spread,
        homeML,
        awayML,
        source: String(book.key ?? 'unknown'),
      };

      lines.push(line);
      _oddsMap.set(normalizeTeam(homeTeam) + '_' + normalizeTeam(awayTeam), line);
    }

    writeCache(key, lines);
    logger.info({ count: _oddsMap.size }, 'Vegas lines fetched from Odds API');
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch odds — continuing without lines');
  }
}

// ─── Lookup ───────────────────────────────────────────────────────────────────

export function getOddsForGame(homeAbbr: string, awayAbbr: string): VegasLine | null {
  const key = normalizeTeam(homeAbbr) + '_' + normalizeTeam(awayAbbr);
  return _oddsMap.get(key) ?? null;
}

export function hasAnyOdds(): boolean {
  return _oddsMap.size > 0;
}

// ─── Convert spread/ML → implied home win probability ────────────────────────

export function impliedHomeProbFromSpread(spread: number): number {
  // In WBB, 1 point on the spread ≈ 3.1% win probability shift
  // This is slightly higher than NBA (2.8%) due to lower scoring
  const pct = 0.5 + (-spread) * 0.031;
  return Math.max(0.05, Math.min(0.95, pct));
}

export function impliedProbFromML(ml: number): number {
  if (ml >= 0) return 100 / (ml + 100);
  return (-ml) / (-ml + 100);
}

function normalizeTeam(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]/g, '');
}
